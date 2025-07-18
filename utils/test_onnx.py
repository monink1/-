import onnxruntime
import torch
import numpy as np
from pathlib import Path
import cv2
import os
import sys

ROOT = Path(__file__).resolve()
sys.path.append(str(ROOT.parents[0]))
sys.path.append(str(ROOT.parents[1]))

from dataloaders import LoadImages
from general import check_version, non_max_suppression, scale_coords, xyxy2xywh
from plots import Annotator, colors
import argparse


class Decode:
    def __init__(self, nc=1):
        self.nc = nc
        self.no = nc + 5
        self.nl = 3
        self.na = 3
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.anchors = torch.tensor([[10, 13, 16, 30, 33, 23],
                                     [30, 61, 62, 45, 59, 119],
                                     [116, 90, 156, 198, 373, 326]]).float().view(self.nl, -1, 2)
        self.stride = torch.tensor([8, 16, 32])
        self.anchors = self.anchors / self.stride.view(-1, 1, 1)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i]
            y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))

        return torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


def test_onnx(onnx_path, img_path, save_dir, imgsz, classes_num, iou_thres, conf_thres, classes=None,
              names=["item"], hide_labels=False, hide_conf=False):
    '''
    onnx_path: 需要测试的onnx模型文件
    img_path：需要测试的图片路径或者文件夹路径
    '''

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    dataset = LoadImages(img_path, img_size=imgsz, stride=32, auto=False)
    decode = Decode(classes_num)

    session = onnxruntime.InferenceSession(onnx_path)
    in_name = [input.name for input in session.get_inputs()]
    out_name = [output.name for output in session.get_outputs()]
    for path, im, im0s, vid_cap, s in dataset:
        im = im.astype(np.float32)
        im /= 255.0
        if len(im) == 3:
            im = im[None]
        data_output = session.run(out_name, {in_name[0]: im})
        data_output = [torch.from_numpy(data) for data in data_output]
        output = decode.forward(data_output)
        pred = non_max_suppression(output, conf_thres, iou_thres, classes=classes, max_det=100)

        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()  # for save_crop
            annotator = Annotator(im0, line_width=4, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh_ = xyxy2xywh(torch.tensor(xyxy).view(1, 4))

                    if True:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if True:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str,
                        default=None,
                        help="要测试的onnx文件路径")
    parser.add_argument("--img_path", type=str, default=None,
                        help="测试图片路径或者文件夹路径")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="检测结果保存路径")
    parser.add_argument("--imgsz", type=int, default=640, help="图片大小")
    parser.add_argument("--classes_num", type=int, default=1, help="检测类别数")
    parser.add_argument("--iou_thres", type=float, default=0.6)
    parser.add_argument("--conf_thres", type=float, default=0.2)
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="图片大小")
    parser.add_argument("--names", type=str, nargs="+", default=["item"], help="图片大小")
    parser.add_argument("--hide_labels", action="store_true", help="是否隐藏label")
    parser.add_argument("--hide_conf", action="store_true", help="是否隐藏分数")

    return parser.parse_args()


def main():
    opt = parser_opt()
    test_onnx(**vars(opt))


if __name__ == "__main__":
    main()
