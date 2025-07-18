import cv2
import torch
import random
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.distributed as dist


def resize(img, target_size):
    """
    img: 输入图片，PIL打开的图片
    in_size: 目标大小 224
    """
    #   如果输入是array，转成PIL格式
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    img_width, img_height = img.size
    # --------------------------------------#
    #   对图片进行同比缩放
    # --------------------------------------#
    scale = min(target_size / img_width, target_size / img_height)
    nw = int(img_width * scale)
    nh = int(img_height * scale)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2

    # --------------------------------------#
    #   多余的部分加上灰条
    # --------------------------------------#
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    new_img = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    new_img.paste(img, (dx, dy))

    return new_img


def init_seed(seed=0):
    '''
    初始化随机数种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，为所有GPU设置随机种子


def reduce_tensor(tensor, option='sum'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if option == "mean":
        rt /= dist.get_world_size()

    return rt


if __name__ == '__main__':
    img1 = Image.open("./0_30.jpg")
    img_new = np.array(img1)
    img_new = Image.fromarray(img_new)
    img1 = transforms.PILToTensor()(img1)
    img2 = cv2.imread("./0_30.jpg")
    img2 = torch.from_numpy(img2)
    new_img = resize(img2, 224)
    new_img.show()

    print(new_img.size)
