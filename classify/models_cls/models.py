import os
import math
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


def model_pretrained(name):
    model_root = "/mnt/datacenter/wangcaifeng/workspace/z.model_zoos/2.model_pretrained/cls"
    return os.path.join(model_root, name)


class Resnet18(nn.Module):
    def __init__(self, classes_num=2, pretrained=None):
        super(Resnet18, self).__init__()
        #   目标类别数
        self.classes_num = classes_num
        self.net = models.resnet18(pretrained=False)
        if pretrained is not None:
            self.net.load_state_dict(torch.load(pretrained))
        in_channels = self.net.fc.in_features
        self.net.fc = nn.Linear(in_channels, self.classes_num)

    def forward(self, x):
        x = self.net(x)

        return x


class Resnet18_Bcnn(nn.Module):
    def __init__(self, classes_num=2, pretrained=None):
        super(Resnet18_Bcnn, self).__init__()
        #   目标类别数
        self.classes_num = classes_num
        self.net = models.resnet18(pretrained=False)
        if pretrained is not None:
            self.net.load_state_dict(torch.load(pretrained))
        in_channels = self.net.fc.in_features
        self.net.fc = nn.Sequential()
        self.fc = nn.Linear(2 * in_channels, self.classes_num)
        nn.init.kaiming_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0)

    def forward(self, x):
        x = self.net(x)
        x = torch.cat((x, x), dim=1)
        x = self.fc(x)

        return x


class BCNN(nn.Module):
    def __init__(self, classes_num):
        super(BCNN, self).__init__()
        self.classes_num = classes_num
        self.model1 = models.resnet18(pretrained=False)
        self.model1.load_state_dict(torch.load(model_pretrained("resnet18-f37072fd.pth")))
        in_channels1 = self.model1.fc.in_features
        self.model1.fc = nn.Sequential()

        self.model2 = models.mobilenet_v2(pretrained=False)
        self.model2.load_state_dict(torch.load(model_pretrained("mobilenet_v2-b0353104.pth")))
        in_channels2 = self.model2.classifier[1].in_features
        self.model2.classifier = nn.Sequential()

        self.fc = nn.Linear(in_channels1 + in_channels2, self.classes_num)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)

        return x


class Resnet18_SPP(nn.Module):
    def __init__(self, classes_num, predtrained=False):
        super(Resnet18_SPP, self).__init__()
        self.classes_num = classes_num


# torchvision中mobilenetv2使用了adaptive_avg_pool2d，目前无法转为caffe模型
class MobilenetV2Ori(nn.Module):
    def __init__(self, classes_num=2, pretrained=None):
        super(MobilenetV2Ori, self).__init__()
        self.classes_num = classes_num
        self.net = models.mobilenet_v2(pretrained=False)
        if pretrained is not None:
            self.net.load_state_dict(torch.load(pretrained))
        in_channels = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_channels, self.classes_num, bias=True)
        print(self.net)

    def forward(self, x):
        return self.net(x)


# MobilenetV2，舍弃了adaptive_avg_pool，使用mean代替
class MobilenetV2(nn.Module):
    def __init__(self, classes_num=2, pretrained=None):
        super(MobilenetV2, self).__init__()
        self.classes_num = classes_num
        model = models.mobilenet_v2(pretrained=False)
        if pretrained is not None:
            model.load_state_dict(torch.load(pretrained))
        self.features = model.features
        in_channels = model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_channels, self.classes_num)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


class Resnet50(nn.Module):
    def __init__(self, classes_num, pretrained=None):
        super(Resnet50, self).__init__()
        self.classes_num = classes_num
        self.net = models.resnet50(pretrained=False)
        if pretrained is not None:
            self.net.load_state_dict(torch.load(pretrained))
        in_channels = self.net.fc.in_features
        self.net.fc = nn.Linear(in_channels, self.classes_num)

    def forward(self, x):
        x = self.net(x)

        return x


class Resnet101(nn.Module):
    def __init__(self, classes_num, pretrained=None):
        super(Resnet101, self).__init__()
        self.classes_num = classes_num
        self.net = models.resnet101(pretrained=False)
        if pretrained is not None:
            self.net.load_state_dict(torch.load(pretrained))
        in_channels = self.net.fc.in_features
        self.net.fc = nn.Linear(in_channels, self.classes_num)

    def forward(self, x):
        x = self.net(x)

        return x


# 自定义网络结构用于联咏芯片部署
class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x1 = x + self.conv1(x)
        x2 = x1 + self.conv2(x1)
        return x2


class Resnet11(nn.Module):
    def __init__(self, classes_num=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 56x56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 28x28

            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            ResidualBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            ResidualBlock(256),
            nn.MaxPool2d(kernel_size=7, stride=1, padding=0),

        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, classes_num)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCNN(classes_num=2).to(device)
    input_ = torch.rand((1, 3, 112, 112), device=device)
    print(model)
