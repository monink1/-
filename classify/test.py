import os
import PIL
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import yaml
from torchvision import transforms
import copy
from pathlib import Path
import sys

import argparse
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("./models_cls")
sys.path.append("../")
from utils_cls.dataloader import load_image_dataset
from utils_cls.tools import resize, init_seed, reduce_tensor
from models_cls.models import Resnet18, MobilenetV2Ori, Resnet50, Resnet11

from utils.general import increment_path


def test(opt):
    used_model, weights, test_path, classes_num, batch_size, save_dir, img_size = \
        opt.used_model, opt.weights, opt.test_path, opt.classes_num, opt.batch_size, \
            opt.save_dir, opt.img_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if used_model == "resnet18":
        model = Resnet18(classes_num)
        model.load_state_dict(torch.load(weights, map_location=device))
    elif used_model == "mobilenetv2":
        model = MobilenetV2Ori(classes_num)
        model.load_state_dict(torch.load(weights, map_location=device))
    elif used_model == "resnet50":
        model = Resnet50(classes_num)
        model.load_state_dict(torch.load(weights, map_location=device))
    elif used_model == "resnet11":
        model = Resnet11(classes_num)
        model.load_state_dict(torch.load(weights, map_location=device))

    model.to(device)
    model.eval()

    w = str(increment_path(Path(save_dir).absolute() / opt.name, exist_ok=False))
    w = Path(w)
    w.mkdir(parents=True, exist_ok=True)

    if opt.export_name.endswith(".onnx"):
        import onnx
        im = torch.ones((3, 3, img_size, img_size)).float()
        f = os.path.join(str(w), used_model + "-" + opt.onnx_name)
        torch.onnx.export(
            model.cpu(),  # --dynamic only compatible with cpu
            im.cpu(),
            f,
            verbose=False,
            opset_version=7,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=False,
            keep_initializers_as_inputs=True,
            input_names=['data'],
            output_names=['output'],
            dynamic_axes={
                'data': {
                    0: 'batch'},
                'output': {
                    0: 'batch'}
            })
        return
    if opt.export_name.endswith(".caffemodel"):
        from pytorch2caffe import pytorch2caffe
        input_ = torch.ones(1, 3, img_size, img_size)
        save_path = str(w) + '/' + opt.export_name.split('.')[0]
        pytorch2caffe.trans_net(model.cpu(), input_, save_path)
        pytorch2caffe.save_prototxt('{}.prototxt'.format(save_path))
        pytorch2caffe.save_caffemodel('{}.caffemodel'.format(save_path))
        return

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_set = load_image_dataset(test_path, test_transform, img_size=img_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    num_batch = len(test_loader)
    pbar = tqdm(enumerate(test_loader), total=num_batch, leave=True)

    all_preds = torch.empty((0,))
    all_targets = torch.empty((0,))

    for _, (datas, targets) in pbar:
        datas = datas.to(device)
        outputs = model(datas)
        _, preds = outputs.max(1)
        all_preds = torch.cat((all_preds, preds.data.cpu()), dim=0)
        all_targets = torch.cat((all_targets, targets), dim=0)

    test_acc = (all_preds == all_targets).sum() / all_targets.shape[0]
    print("test accuracy is ", test_acc)

    sns.set()
    confusion_matix = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    f, ax = plt.subplots()
    sns.heatmap(confusion_matix, annot=True, ax=ax, fmt='.4f')
    ax.set_title("confusion matrix")
    ax.set_xlabel("predict")
    ax.set_ylabel("true")
    plt.savefig(os.path.join(str(w), "confusion_matrix.png"), format="png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--used_model", type=str, default="resnet18", help="使用的模型")
    parser.add_argument("--weights", type=str, default="./runs/exp/weights/best.py", help="模型权重")
    parser.add_argument("--test_path", type=str, default="", help="测试的图片文件夹路径")
    parser.add_argument("--classes_num", type=int, default=2, help="类别数")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="./runs/test")
    parser.add_argument("--name", type=str, default='exp', help="保存的模型的文件夹名")
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--export_name", type=str, default="helmet.onnx or helmet.caffemodel", help="输出模型类别")

    opt = parser.parse_args()

    test(opt)


if __name__ == "__main__":
    main()
