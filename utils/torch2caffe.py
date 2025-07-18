from torch.autograd import Variable
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Detect_(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, anchors):  # detection layer
        super().__init__()
        self.nc = 0  # number of classes
        self.no = 0  # number of outputs per anchor
        self.nl = 0  # number of detection layers
        self.na = 0  # number of anchors
        self.grid = []  # init grid
        self.anchor_grid = []  # init anchor grid
        self.m = nn.ModuleList()  # output conv
        self.inplace = None  # use inplace ops (e.g. slice assignment)
        self.f = []
        self.i = 0

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # test = F.sigmoid(x[i].reshape((-1, )))
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = F.sigmoid(x[i])

        return (x, _)

def torch2caffe(model, img_size, export_name):
    model.eval()
    device = model.device

    detect_old = model.model.model[24]
    detect_new = Detect_(detect_old.anchors)

    detect_new.nc = detect_old.nc
    detect_new.no = detect_old.no
    detect_new.na = detect_old.na
    detect_new.nl = detect_old.nl
    detect_new.f = detect_old.f
    detect_new.m = detect_old.m
    detect_new.i = detect_old.i
    detect_new.inplace = detect_old.inplace

    detect_new.to(model.device)


    model.model.model[24] = detect_new.eval()

    input = torch.ones((1, 3, img_size, img_size)).to(device)
    f = './onnx+pt/' + export_name
    out_put = model(input)
    print(input.dtype)


    torch.onnx.export(
        model.cpu(),  # --dynamic only compatible with cpu
        input.cpu(),
        f,
        verbose=False,
        opset_version=11,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=False,
        input_names=['data'],
        output_names=['out_8', "out_16", 'out_32'],
    )

from models.yolo import Detect
def select_classes(model, splice_classes):
    model.eval()
    device = model.device
    detect_old = model.model.model[24]
    detect_new = Detect(anchors=detect_old.anchors, ch=(128, 128, 128))
    detect_new.nc = len(splice_classes)
    detect_new.no = detect_new.nc + 5
    detect_new.na = detect_old.na
    detect_new.nl = detect_old.nl
    detect_new.f = detect_old.f
    detect_new.i = detect_old.i
    detect_new.stride = detect_old.stride
    detect_new.inplace = detect_old.inplace
    detect_new.m = nn.ModuleList([nn.Conv2d(m.in_channels, detect_new.no * detect_new.na, 1) for m in detect_old.m])
    detect_new.to(device)

    for i in range(detect_new.nl):
        old_weight = detect_old.m[i].weight.data
        old_bias = detect_old.m[i].bias.data
        # print("old_weight_shape: ", old_weight.shape)
        # print("old_bias_shape: ", old_bias.shape)
        new_weight = detect_new.m[i].weight.data
        new_bias = detect_new.m[i].bias.data
        # print("new_weight_shape: ", new_weight.shape)
        # print("new_bias_shape: ", new_bias.shape)
        w1 = torch.cat([old_weight[:5, ...]] + [old_weight[5 + cls, ...][None] for cls in splice_classes], dim=0)
        w2 = torch.cat([old_weight[85:90, ...]] + [old_weight[90 + cls, ...][None] for cls in splice_classes], dim=0)
        w3 = torch.cat([old_weight[170:175, ...]] + [old_weight[175 + cls, ...][None] for cls in splice_classes], dim=0)
        w_new = torch.cat((w1, w2, w3), dim=0)
        detect_new.m[i].weight.data = w_new

        b1 = torch.cat([old_bias[:5]] + [old_bias[5 + cls][None] for cls in splice_classes], dim=0)
        b2 = torch.cat([old_bias[85:90]] + [old_bias[90 + cls][None] for cls in splice_classes], dim=0)
        b3 = torch.cat([old_bias[170:175]] + [old_bias[175 + cls][None] for cls in splice_classes], dim=0)
        b_new = torch.cat((b1, b2, b3), dim=0)
        detect_new.m[i].bias.data = b_new

    model.model.model[24] = detect_new.eval()
    return model

from pytorch2caffe import pytorch2caffe
def convert_to_caffe(model, imgsz, model_name):
    model.eval()
    device = model.device

    detect_old = model.model.model[24]
    detect_new = Detect_(detect_old.anchors)

    detect_new.nc = detect_old.nc
    detect_new.no = detect_old.no
    detect_new.na = detect_old.na
    detect_new.nl = detect_old.nl
    detect_new.f = detect_old.f
    detect_new.m = detect_old.m
    detect_new.i = detect_old.i
    detect_new.inplace = detect_old.inplace

    detect_new.to(model.device)


    model.model.model[24] = detect_new.eval()
    save_path = './onnx+pt/' + model_name.split('.')[0]
    input = torch.ones([1, 3, imgsz, imgsz]).to(model.device)
    pytorch2caffe.trans_net(model, input, save_path)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(save_path))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(save_path))
