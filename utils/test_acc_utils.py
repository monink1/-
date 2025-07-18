import torch
import torch.nn as nn
from tqdm import tqdm
from utils.dataloaders import create_dataloader_fromdir
from utils.general import non_max_suppression
from PIL import Image
from torchvision import transforms
from classify.utils_cls.tools import resize
import numpy as np

def image_matting(bbox, img, img_size):
    '''
    bbox: tensor（x, y, w, h, conf, cls) shape(num_target, 6)
    img:输入图片tensor(B C H W), 0~1
    '''
    #   存放提取出来的图片
    images = torch.zeros((bbox.shape[0], 3, img_size, img_size))

    bbox = bbox.clone().data.cpu().numpy()
    img = img.clone()
    img = img.data.cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(np.uint8(img * 255))

    for i in range(bbox.shape[0]):
        box = bbox[i]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        new_img = img.crop((x1, y1, x2, y2))
        #   保存剪裁出尺寸大于50*50的图片
        # if new_img.width >= 50 and new_img.height >= 50:
        #     new_img.save("./data/hat_cls/1_{}.jpg".format(num_matting))
        #     num_matting += 1

        # new_img = resize(new_img, img_size)
        new_img = new_img.resize((img_size, img_size))
        new_img = torch.from_numpy(np.array(new_img).transpose(2, 0, 1) / 255.0)
        new_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(new_img)
        images[i] = new_img.clone()

    return images

def test_acc_for_project(path1, path2, model, sec_classify_model, batch_size, img_size, stride, half, cuda, device, color, conf_thres, iou_thres, class_id, sec_imgsz, class_det=None):
    '''
    函数作用：计算检测告警的准确率
    path1：需要告警的图片文件夹路径
    path2；不需要告警的图片文件夹路径
    model: 检测的模型
    sec_classify_model：二次分类的模型
    img_size：检测的输入图片大小
    class_id：需要检测到的类别
    sec_imgsz：二次分类的图片输入大小
    class_det：检测模型类别过滤，如yolov权重会检测80个类别，只需检测人的话，此值为0
    '''

    test_loader1 = create_dataloader_fromdir(path1,
                                     img_size,
                                     batch_size,
                                     stride,
                                     pad=0.0,  # pad,
                                     rect=False,  # rect,
                                     workers=8,
                                     prefix=color)[0]

    test_loader2 = create_dataloader_fromdir(path2,
                                             img_size,
                                             batch_size,
                                             stride,
                                             pad=0.0,  # pad,
                                             rect=False,  # rect,
                                             workers=8,
                                             prefix=color)[0]
    pbar1 = tqdm(test_loader1)
    pbar2 = tqdm(test_loader2)

    TP = 0.0
    FN = 0.0
    TN = 0.0
    FP = 0.0

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar1):
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        out, train_out = model(im, augment=False, val=True)  # inference, loss outputs

        # NMS
        out = non_max_suppression(out, conf_thres, iou_thres, multi_label=False, classes=class_det)

        # -------- 计算准确率，有无灭火器、有无火苗 -----------------#
        for i, pred in enumerate(out):
            # 检测+分类
            if sec_classify_model is not None:
                matting_imgs = image_matting(pred, im[i], img_size=sec_imgsz)
                matting_imgs = matting_imgs.to(device)
                sec_outputs = sec_classify_model(matting_imgs)
                _, sec_preds = torch.softmax(sec_outputs, dim=1).max(1)
                pred[:, -1] = sec_preds.float()
            if pred.shape[0] != 0 and (sum(pred[:, -1] == int(class_id[0])) != 0 or sum(pred[:, -1] == int(class_id[1])) != 0):
                TP += 1.0 # 检测正确
            else:
                FN += 1.0 # 漏检

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar2):
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        out, train_out = model(im, augment=False, val=True)  # inference, loss outputs

        # NMS
        out = non_max_suppression(out, conf_thres, iou_thres, multi_label=False, classes=class_det)

        # -------- 计算准确率，有无灭火器、有无火苗 -----------------#
        for i, pred in enumerate(out):
            # 检测+分类
            if sec_classify_model is not None:
                matting_imgs = image_matting(pred, im[i], img_size=sec_imgsz)
                matting_imgs = matting_imgs.to(device)
                sec_outputs = sec_classify_model(matting_imgs)
                _, sec_preds = torch.softmax(sec_outputs, dim=1).max(1)
                pred[:, -1] = sec_preds.float()
            if pred.shape[0] == 0 or (sum(pred[:, -1] == int(class_id[0])) == 0 or sum(pred[:, -1] == int(class_id[1])) == 0):
                TN += 1.0 # 没有目标且检测结果也没有目标
            else:
                FP += 1.0 # 误检
    acc = (TP + TN) / (TP + FP + TN + FN)
    acc *= 100
    return acc, TP, TN, FP, FN

