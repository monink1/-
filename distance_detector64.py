# -*- coding: utf-8 -*-

# YOLOv5 ? by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shutil
from matplotlib import font_manager

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from matplotlib import font_manager
from utils.torch_utils import select_device, time_sync
from classify.models_cls.models import Resnet18, BCNN, MobilenetV2Ori
from classify.utils_cls.tools import resize


def read_excel_data(file_path):
    """
    读取Excel文件并处理数据
    """
    try:
        # 读取Excel文件，不使用表头，不转换数据类型
        df = pd.read_excel(file_path, header=None, dtype=str)  # 使用str类型避免自动转换
        
        # 将所有单元格内容转换为float类型，如果转换失败则保持原样
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                try:
                    df.iloc[i, j] = float(df.iloc[i, j])
                except (ValueError, TypeError):
                    # 如果转换失败，保持原样
                    pass
        
        # 将NaN值替换为0
        df = df.fillna(0)
        
        # 确保所有数值都是float类型
        df = df.astype(float)
        
        return df
    except Exception as e:
        print(f"读取Excel文件时出错: {str(e)}")
        return pd.DataFrame()


def image_matting(bbox, img, num_matting, img_size):
    '''
    bbox: tensor（x, y, w, h, conf, cls) shape(num_target, 6)
    img:输入图片tensor(B C H W), 0~1
    '''
    #   存放提取出来的图片
    images = torch.zeros((bbox.shape[0], 3, img_size, img_size))
    bbox = bbox.clone().data.cpu().numpy()
    img = img.clone()
    img = img.data.cpu().numpy()[0]
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

        new_img = resize(new_img, img_size)
        # new_img = new_img.resize((img_size, img_size))
        new_img = torch.from_numpy(np.array(new_img).transpose(2, 0, 1) / 255.0)
        new_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(new_img)
        images[i] = new_img.clone()

    return images, num_matting


# 生成热力图
def generate_heatmap(df, title, save_path):
    # 获取网格大小
    xpart, ypart = 64, 36  # 固定网格大小
    
    # 创建热力图
    plt.figure(figsize=(xpart * 2.5, ypart * 2.5))  # 保持画布大小不变
    
    # 自定义颜色映射：0为白，逐渐变黄、红，最大为黑
    colors_list = [
        (1, 1, 1),    # white (0)
        (1, 1, 0),    # yellow (中等)
        (1, 0, 0),    # red (较高)
        (0, 0, 0)     # black (最大)
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_white_yellow_red_black", colors_list)
    
    # 设置字体
    try:
        if platform.system() == 'Windows':
            font_path = "C:/Windows/Fonts/arial.ttf"
        else:  # Linux/Mac
            font_path = "./Arial.ttf"
        prop = font_manager.FontProperties(fname=font_path)
    except Exception as e:
        print(f"警告：无法加载字体文件，使用默认字体：{str(e)}")
        prop = font_manager.FontProperties()

    # 绘制热力图
    sns.heatmap(df, cmap=cmap, annot=True, fmt='.0f',
               linewidths=1, linecolor='black',
               cbar=False,  # 移除颜色条
               annot_kws={'size': 20, 'fontproperties': prop},  # 将字体大小设置为20
               square=True,  # 保持单元格为正方形
               xticklabels=True,  # 显示x轴标签
               yticklabels=True)  # 显示y轴标签

    # 分别设置标签的字体
    plt.xticks(fontproperties=prop, fontsize=16)
    plt.yticks(fontproperties=prop, fontsize=16)

    # 添加标题和标签
    plt.title(title, fontsize=18)
    plt.xlabel('Columns', fontsize=16)
    plt.ylabel('Rows', fontsize=16)

    # 调整布局
    plt.tight_layout()
    
    # 保存热力图
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5l.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.6,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        sec_classify=False,
        sec_classify_weight="",
        sec_img_size=112,
        sec_num_classes=2,
        show_classes=None,
        target_classes=None,
        cls=None,  # 要识别的类别名称，如 suitcase 或 person
        xlsx=None,  # Excel文件路径
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    #   判断路径是否是图片，视频的格式
    is_file = Path(source).suffix[1:].lower() in (IMG_FORMATS + VID_FORMATS)
    #   判断路径是否是链接
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)  # 判断路径是否只有数字、是否为文本或是否为链接/文件
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    #   保存模型的路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / "target-imgs").mkdir(exist_ok=True) if target_classes else None

    # 初始化区域划分变量
    ypart = 36  # 行数
    xpart = 64  # 列数
    
    # 计算每个网格的宽度和高度
    xpx, ypx = 0, 0
    xzone = xpx / xpart
    yzone = ypx / ypart
    
    # 初始化网格计数矩阵
    numbernet = np.zeros((ypart, xpart))  # 初始化为0，只记录有效框
    act_numbernet = np.zeros((ypart, xpart))  # 初始化为0，只记录有效框
    tensornet = np.zeros((ypart, xpart))
    actual_heightnet = np.zeros((ypart, xpart))
    
    # 初始化DataFrame用于存储处理后的数据
    processed_pixel_df = pd.DataFrame(np.zeros((ypart, xpart)))
    processed_actual_df = pd.DataFrame(np.zeros((ypart, xpart)))
    
    # 初始化高度数据列表
    height_data = []
    height_ratios = []
    
    # 初始化统计变量
    total_boxes = 0
    filtered_boxes = 0
    aspect_ratio_stats = {}

    # 如果提供了Excel文件，读取并处理数据
    if xlsx:
        try:
            # 复制原始Excel文件到输出目录
            original_division_copy = save_dir / f'Original_Regional_division_{Path(source).stem}.xlsx'
            shutil.copy2(xlsx, original_division_copy)
            
            # 读取复制后的文件，指定header=None以防止第一行被当作列名
            excel_data = pd.read_excel(original_division_copy, header=None)
            
            # 确保Excel数据的形状与区域划分匹配
            xpart, ypart = 64, 36  # 固定网格大小
            
            # 创建新的DataFrame，确保大小为36×64
            new_data = pd.DataFrame(np.zeros((ypart, xpart)))
            print(f"创建的新DataFrame形状: {new_data.shape}")
            
            # 将原始数据复制到新DataFrame中
            for i in range(min(ypart, excel_data.shape[0])):
                for j in range(min(xpart, excel_data.shape[1])):
                    pixel_height = excel_data.iloc[i, j]
                    
                    if pd.isna(pixel_height) or pixel_height == 0:
                        new_data.iloc[i, j] = 0
                    else:
                        try:
                            new_data.iloc[i, j] = 67 / pixel_height
                        except ZeroDivisionError:
                            new_data.iloc[i, j] = 0
                        except Exception as e:
                            print(f"处理单元格({i},{j})时出错: {str(e)}")
                            new_data.iloc[i, j] = 0

            # 验证数据完整性
            if new_data.isna().any().any():
                print("警告: 处理后的数据中包含NaN值，将被替换为0")
                new_data = new_data.fillna(0)

            excel_data = new_data
            print(f"处理后的excel_data形状: {excel_data.shape}")
            print(f"处理后的excel_data数据:\n{excel_data}")
            
            # 使用ExcelWriter保存数据，确保所有行都被保存
            with pd.ExcelWriter(original_division_copy, engine='openpyxl', mode='w') as writer:
                try:
                    # 不设置index=False，这样可以保留所有行
                    excel_data.to_excel(writer, sheet_name='Sheet1', header=False, 
                                      startrow=0, startcol=0, na_rep='0', index=False)
                    writer.save()
                    print(f"已成功保存处理后的Excel文件到: {original_division_copy}")
                except Exception as e:
                    print(f"保存Excel文件时出错: {str(e)}")
                    # 如果保存失败，尝试使用备份
                    if os.path.exists(original_division_copy):
                        os.remove(original_division_copy)
                    shutil.copy2(xlsx, original_division_copy)
                    print(f"已恢复原始Excel文件: {original_division_copy}")

            # 重新读取保存的文件，验证数据是否完整
            try:
                saved_data = pd.read_excel(original_division_copy, header=None)
                print(f"重新读取的数据形状: {saved_data.shape}")
                print(f"重新读取的数据:\n{saved_data}")
                
                # 验证数据是否正确保存
                # if not np.array_equal(saved_data.values, excel_data.values):
                #     print("警告: 保存的数据与处理后的数据不完全匹配")
                #     # 如果数据不匹配，恢复原始文件
                #     if os.path.exists(original_division_copy):
                #         os.remove(original_division_copy)
                #     shutil.copy2(xlsx, original_division_copy)
                #     print(f"已恢复原始Excel文件: {original_division_copy}")
                #     xlsx = None
            except Exception as e:
                print(f"重新读取Excel文件时出错: {str(e)}")
                xlsx = None

        except Exception as e:
            print(f"处理Excel数据时出错: {str(e)}")
            # 如果出错，恢复原始文件
            if os.path.exists(original_division_copy):
                os.remove(original_division_copy)
            shutil.copy2(xlsx, original_division_copy)
            print(f"已恢复原始Excel文件: {original_division_copy}")
            xlsx = None
    last_boxh = 0
    new_boxh = 0



    # Load model
    device = select_device(device)
    #   检测编译框架
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt  # names 为目标检测任务的目标类别
    #   确保输入图片的尺寸能被32整除，如果不能调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # 类别名称映射 - 常用的别名映射到COCO标准名称
    class_name_mapping = {
        "human": "person",
        "people": "person",
        "luggage": "suitcase",
        "bag": "suitcase",
        "handbag": "handbag",
        "backpack": "backpack",
        "head": "head"
    }
    
    # 处理类别过滤
    target_cls_idx = None
    if cls is not None:
        # 检查是否需要通过映射转换类名
        if cls in class_name_mapping:
            print(f"注意: 将类别名 '{cls}' 映射为标准COCO类别名 '{class_name_mapping[cls]}'")
            cls = class_name_mapping[cls]
        
        # 检查类别是否存在
        if cls in names:
            target_cls_idx = names.index(cls)
            print(f"将只检测类别: {cls} (索引: {target_cls_idx})")
        else:
            # 显示所有可用的类别
            print(f"错误: 找不到类别 '{cls}'")
            print(f"可用的类别有: {', '.join(names)}")
            print("常用类别: person (人), suitcase (行李箱), backpack (背包), handbag (手提包)")
            return

    # Dataloader
    #   使用视频流或者页面
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        #   直接从source文件夹下读取图片或者视频
        #   dataset为一个可迭代的对象，可以用for循环访问
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)
        # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)  # 核查!!! 上面为l
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # --------------------------------#
    #   使用二次分类
    # --------------------------------#
    if sec_classify != "":
        if sec_classify == "resnet18":
            classify_model = Resnet18(sec_num_classes)
        elif sec_classify == "mobilenetv2":
            classify_model = MobilenetV2Ori(sec_num_classes)
        elif sec_classify == "BCNN":
            classify_model = BCNN(sec_num_classes)
        classify_model.load_state_dict(torch.load(sec_classify_weight, map_location='cpu'))
        classify_model.eval()
        classify_model.to(device)
        # names = ['work_clothes', 'pedestrian', 'winter_work_clothes', 'cleaning']
        names = ['helmet', 'head']
    #   剪裁出的图片数量
    num_matting = 0

    max_dis = 0.0
    last_xcenter = None
    det_count = []  # 初始化为列表
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt, det_count = 0, [], [0.0, 0.0, 0.0], []  # 确保det_count是列表
    aspect_ratio_stats = {}  # 初始化高宽比统计字典
    detected_objects = []  # 初始化检测对象列表
    frame_idx = 0
    if xlsx:
        # 创建比例矩阵
        ratio_matrix = read_excel_data(original_division_copy)
        print(f"ratio_matrix形状: {ratio_matrix.shape}")
        print(f"ratio_matrix前几行:\n{ratio_matrix.head()}")
        print(f"ratio_matrix索引范围: 行({ratio_matrix.index[0]}-{ratio_matrix.index[-1]}), 列({ratio_matrix.columns[0]}-{ratio_matrix.columns[-1]})")
    for path, im, im0s, vid_cap, s in dataset:
        #   path：视频文件路径
        #   im：转化后的图片或者帧
        #   im0s：读出的原始图片

        # ------- added by wcf
        if s.startswith("video"):
            count_frame_wcf = int(s[s.index("(") + 1:s.index(")")].split("/")[0])
            if not count_frame_wcf % 1 == 0:  # 跳帧检测
                cv2.imshow(str(Path(path)), im0s)
                cv2.waitKey(1)  # 1 millisecond
                continue

        # ------- added by wcf

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        #   是否启用半精度
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            #  增加一个维度，将图片从CHW转为BCHW
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        visualize_save_target = False  # 出现检测目标出现时!!!

        '''
        pred形状为（1，num_boxes, 5+num_class)
        pred[..., 0:4]为预测框坐标 = 预测框坐标为xywh(中心点 + 宽长)格式
        pred[..., 4]为objectness置信度
        pred[..., 5:-1]为分类结果
        '''
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        #   pred是一个列表list[torch.tensor], 长度为batch_size
        #   每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        # Second-stage classifier (optional)
        # pred = utils_cls.general.apply_classifier(pred, classifier_model, im, im0s)

        if sec_classify != "":
            images_for_cls, num_matting = image_matting(pred[0], im, 0, sec_img_size)
            images_for_cls = images_for_cls.to(device)
            outputs = classify_model(images_for_cls)
            _, preds = outputs.max(dim=1)
            preds = preds.view((-1,))
            pred[0][:, -1] = preds

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # 处理检测结果
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # 获取图片尺寸
                xpx, ypx = im0.shape[1], im0.shape[0]
                
                # 计算每个网格的宽度和高度
                xzone = xpx / xpart
                yzone = ypx / ypart
                
                # 处理检测结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    
                    # 只处理指定类别
                    if target_cls_idx is not None and c != target_cls_idx:
                        continue

                    # 计算目标框中心点所在的网格位置
                    center_x = (xyxy[0] + xyxy[2]) / 2  # 中心点x坐标
                    center_y =  xyxy[3] # 底部中心点y坐标
                    
                    # 计算网格索引
                    xi = int(center_x / xzone)  # 列索引 (0-31)
                    yi = int(center_y / yzone)  # 行索引 (0-17)
                    
                    # 确保索引在有效范围内
                    if xi < 0 or xi >= xpart or yi < 0 or yi >= ypart:
                        continue

                    # 计算框的尺寸
                    boxw = xyxy[2] - xyxy[0]
                    boxh = xyxy[3] - xyxy[1]
                    height_ratio = float(boxh / boxw) if boxw > 0 else 0
                    total_boxes += 1

                    # 记录所有类别的高宽比
                    aspect_ratio_stats[height_ratio] = aspect_ratio_stats.get(height_ratio, 0) + 1

                    # 根据类别应用不同的高宽比过滤条件
                    # if names[c] == 'suitcase':
                    #     # 过滤掉高宽比不合适的框
                    #     if height_ratio < 1.42 or height_ratio > 1.58:
                    #         filtered_boxes += 1
                    #         continue
                    # elif names[c] == 'person':
                        # 过滤掉高宽比不合适的框
                        # if height_ratio < 2.2 or height_ratio > 2.9:
                        #     filtered_boxes += 1
                        #     continue

                    # 绘制框和基本标签（类别和置信度）
                    label = f'{names[c]} {conf:.2f}' if not hide_conf else f'{names[c]}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # 初始化excel_val
                    excel_val = 0

                    # 收集数据用于后续处理
                    if names[c] == 'person':
                        height_data.append({
                            'frame': frame_idx,
                            'pixel_height': boxh,
                            'row': yi,
                            'col': xi
                        })
                    
                    # 更新网格计数和累计值
                    if names[c] == 'person':
                        numbernet[yi][xi] += 1
                        tensornet[yi][xi] += boxh

                    # # 如果提供了xlsx文件，则收集实际高度数据
                    # if xlsx is not None and ratio_matrix is not None:
                    #     try:
                    #         # 确保索引在有效范围内
                    #         if 0 <= yi < ypart and 0 <= xi < xpart:
                    #             # 获取DataFrame的行数和列数
                    #             num_rows = len(ratio_matrix)
                    #             num_cols = len(ratio_matrix.columns)
                                
                    #             # 检查索引是否在DataFrame范围内
                    #             if 0 <= yi < num_rows and 0 <= xi < num_cols:
                    #                 excel_val = ratio_matrix.iloc[yi, xi]
                    #                 if excel_val > 0:  # 确保比例大于0
                    #                     actual_height = boxh * excel_val
                    #                     height_data[-1]['actual_height'] = actual_height  # 更新最后一条记录的实际高度
                    #                     actual_heightnet[yi][xi] += actual_height
                    #     except Exception as e:
                    #         print(f"处理高度数据时出错: {str(e)}")
                    #         pass  # 忽略错误，继续处理下一个目标

                    # 计算高宽比并保存
                    height_ratios.append(height_ratio)

                    # 在框的右侧添加高度信息并处理高度数据
                    if xlsx is not None and ratio_matrix is not None:
                        try:
                            # 确保索引在有效范围内
                            if 0 <= yi < ypart and 0 <= xi < xpart:
                                excel_val = ratio_matrix.iloc[yi, xi]
                                actual_height = boxh * excel_val
                                    
                                if excel_val > 0:  # 确保比例大于0
                                    # 对person处理实际高度数据
                                    if names[c] == 'person':
                                        # 更新实际高度数据
                                        actual_heightnet[yi][xi] += actual_height
                                        act_numbernet[yi][xi] += 1
                                        
                                        # 确保height_data中包含实际高度列
                                        if len(height_data) == 0 or height_data[-1].get('actual_height') is None:
                                            height_data.append({
                                                'row': yi,
                                                'col': xi,
                                                'pixel_height': tensornet[yi][xi] / numbernet[yi][xi],
                                                'actual_height': actual_heightnet[yi][xi] / act_numbernet[yi][xi]
                                            })
                                        else:
                                            height_data[-1]['actual_height'] = actual_heightnet[yi][xi] / act_numbernet[yi][xi]
                                    
                                    # 对person和suitcase显示实际高度
                                    if names[c] == 'person':
                                        height_label = f'height:{actual_height:.1f}'
                                    elif names[c] == 'suitcase':
                                        height_label = f'height:{actual_height:.1f}'
                                    else:
                                        height_label = f'pixel height:{boxh:.1f}'
                                    
                                    font_scale = 0.5
                                    font_thickness = 1
                                    # 获取文本大小
                                    (text_width, text_height), _ = cv2.getTextSize(height_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                                    
                                    # 计算标签位置（在框的右侧）
                                    x = int(xyxy[2])  # 框的右边界
                                    y = int((xyxy[1]+xyxy[3])/2)  # 框的中间
                                    
                                    # 绘制高度标签背景
                                    cv2.rectangle(im0, 
                                                (x + 5, y - text_height - 5),
                                                (x + text_width + 10, y + 5),
                                                colors(c, True), -1)
                                    
                                    # 绘制高度标签文本
                                    cv2.putText(im0, height_label,
                                              (x + 7, y),
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, [225, 255, 255],
                                              thickness=font_thickness,
                                              lineType=cv2.LINE_AA)
                        except Exception as e:
                            pass  # 忽略错误，继续处理下一个目标
                    else:
                        # 如果没有提供xlsx文件，只显示像素高度
                        height_label = f'pixel height:{boxh:.1f}'
                        
                        # 只对person保存像素高度数据
                        if names[c] == 'person':
                            if len(height_data) == 0 or 'pixel_height' not in height_data[-1]:
                                height_data.append({
                                    'row': yi,
                                    'col': xi,
                                    'pixel_height': tensornet[yi][xi] / numbernet[yi][xi]
                                })
                        
                        # 绘制高度标签
                        font_scale = 0.5
                        font_thickness = 1
                        # 获取文本大小
                        (text_width, text_height), _ = cv2.getTextSize(height_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        
                        # 计算标签位置（在框的右侧）
                        x = int(xyxy[2])  # 框的右边界
                        y = int((xyxy[1]+xyxy[3])/2)  # 框的中间
                        
                        # 绘制高度标签背景
                        cv2.rectangle(im0, 
                                    (x + 5, y - text_height - 5),
                                    (x + text_width + 10, y + 5),
                                    colors(c, True), -1)
                        
                        # 绘制高度标签文本
                        cv2.putText(im0, height_label,
                                  (x + 7, y),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, [225, 255, 255],
                                  thickness=font_thickness,
                                  lineType=cv2.LINE_AA)
                    
                    # # 保存高度数据
                    # if xlsx and excel_val != 0 and numbernet[yi][xi] > 0:
                    #     if len(height_data) == 0 or height_data[-1].get('actual_height') is None:
                    #         height_data.append({
                    #             'row': yi,
                    #             'col': xi,
                    #             'pixel_height': tensornet[yi][xi] / numbernet[yi][xi],
                    #             'actual_height': actual_heightnet[yi][xi] / numbernet[yi][xi]
                    #         })
                    # else:
                    #     if len(height_data) == 0 or 'actual_height' not in height_data[-1]:
                    #         height_data.append({
                    #             'row': yi,
                    #             'col': xi,
                    #             'pixel_height': tensornet[yi][xi] / numbernet[yi][xi]
                    #         })
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label=label, color=colors(c, True))
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow(str(p), cv2.resize(im0, (np.array(im0.shape[:2])[::-1] * 0.5).astype(int), cv2.INTER_LINEAR))
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t2 - t1:.3f}s)')

            # 打印检测结果
            if cls is not None:
                if detected_objects:
                    LOGGER.info(f'{s}{len(detected_objects)}{names[int(cls)]} Done. ({t2 - t1:.3f}s)')
                else:
                    LOGGER.info(f'{s}0{cls} Done. ({t2 - t1:.3f}s)')

        frame_idx += 1

    # 处理并保存数据
    if len(height_data) > 0:
        # 创建DataFrame
        df = pd.DataFrame(height_data)
        video_name = Path(source).stem
        print("正在保存像素高度")
        
        # 创建64x36的网格数据
        heatmap_data = pd.DataFrame(np.zeros((36, 64)))
        
        # 计算每个网格的像素高度平均值
        for yi in range(36):
            for xi in range(64):
                if numbernet[yi][xi] > 0:
                    heatmap_data.iloc[yi, xi] = tensornet[yi][xi] / numbernet[yi][xi]
        
        # 保存像素高度数据（按网格格式保存）
        pixel_height_excel_path = str(save_dir / f'Pixel_Height_Average_{video_name}.xlsx')
        heatmap_data.to_excel(pixel_height_excel_path, index=False, header=False)
        print(f"已保存像素高度Excel文件到: {pixel_height_excel_path}")
        
        # 生成像素高度热力图
        pixel_heatmap_path = str(save_dir / f'Pixel_Height_Heatmap_{video_name}.png')
        
        generate_heatmap(heatmap_data, f'Pixel Height Heatmap - {video_name}', pixel_heatmap_path)
        print(f"已保存像素高度热力图到: {pixel_heatmap_path}")
        
        # 如果提供了xlsx文件，则计算并保存实际高度数据
        if xlsx:
            # 获取输出视频的路径
            output_video_path = Path(source).resolve()
            output_video_name = output_video_path.stem
            
            # 读取复制后的文件
            original_df = pd.read_excel(original_division_copy, header=None)
            
            # 计算实际高度
            actual_height_data = pd.DataFrame(np.zeros((36, 64)))
            for yi in range(36):
                for xi in range(64):
                    if act_numbernet[yi][xi] > 0:
                        actual_height_data.iloc[yi, xi] = actual_heightnet[yi][xi] / act_numbernet[yi][xi]
            
            # 保存实际高度平均值
            actual_height_excel_path = str(save_dir / f'Actual_Height_Average_{output_video_name}.xlsx')
            actual_height_data.to_excel(actual_height_excel_path, index=False, header=False)
            print(f"已保存实际高度平均值Excel文件到: {actual_height_excel_path}")
            
            # 生成实际高度热力图
            actual_heatmap_path = str(save_dir / f'Actual_Height_Heatmap_{output_video_name}.png')
            generate_heatmap(actual_height_data, f'Actual Height Heatmap - {output_video_name}', actual_heatmap_path)
            print(f"已保存实际高度热力图到: {actual_heatmap_path}")

    else:
        print(f"height_data值为{height_data}")
    # 生成高宽比分布图
    def generate_ratio_histogram(height_ratios, video_name, save_dir):
        plt.figure(figsize=(12, 6))
        
        # 设置区间，以0.1为单位
        bins = np.arange(0, max(height_ratios) + 0.1, 0.1)
        
        # 绘制柱状图
        n, bins, patches = plt.hist(height_ratios, bins=bins, edgecolor='black', alpha=0.7)
        
        # 在每个柱子上添加数量标签
        for i in range(len(n)):
            if n[i] > 0:  # 只在有数据的柱子上显示标签
                plt.text(bins[i], n[i], int(n[i]), 
                        ha='center', va='bottom')
        
        # 设置标题和标签
        plt.title(f'Height-Width Ratio Distribution - {video_name}', fontsize=14)
        plt.xlabel('Height-Width Ratio', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # 设置x轴刻度，每0.1显示一个刻度
        plt.xticks(bins[::1], [f'{x:.1f}' for x in bins[::1]], rotation=45)
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置y轴为整数刻度
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        ratio_plot_path = str(save_dir / f'height_width_ratio_{video_name}.png')
        plt.savefig(ratio_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"高宽比分布图已保存到: {ratio_plot_path}")

    if height_ratios:  # 只在有检测结果时生成图表
        generate_ratio_histogram(height_ratios, Path(source).stem, save_dir)

    # 如果指定了--xlsx参数，将xlsx路径下的所有文件复制到输出文件夹
    if xlsx:
        xlsx_path = Path(xlsx)
        if xlsx_path.exists():
            # 获取Excel文件所在目录的上一级目录
            parent_dir = xlsx_path.parent
            output_path = save_dir
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            
            # 复制上一级目录下的所有文件
            for file in parent_dir.glob('*'):
                if file.is_file():
                    shutil.copy2(file, output_path)
            print(f"已将所有文件从 {parent_dir} 复制到 {output_path}")



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'yolov5l.pt', help='model path(s)')
    parser.add_argument('--source', type=str,
                        default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / "data/1.falldown.yaml", help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--xlsx', type=str, help='Path to Excel file for height calculation')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')  # 不保存结构化结果
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # 结果中显示置信度
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--sec_classify', type=str, default="", help="是否启用二次分类器")
    parser.add_argument("--sec_classify_weight", type=str, default="./classify/runs/exp/weights/best.pt", help="二次分类的模型权重")
    parser.add_argument("--sec_img_size", type=int, default=112, help="二次分类网络的输入大小")
    parser.add_argument("--sec_num_classes", type=int, default=2, help="二次分类的网络")
    parser.add_argument('--show-classes', nargs='+', type=str, help='classes to display on the det-img.')  # 检测结果图像上展示的类别信息
    parser.add_argument('--target_classes', nargs='+', type=str,  help='classes needed to select.')  # 筛选出指定类的检测结果
    parser.add_argument('--cls', type=str, help='指定要识别的目标类别，例如 suitcase 或 person')
 
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)