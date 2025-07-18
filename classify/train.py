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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("./models_cls")
from utils_cls.dataloader import load_image_dataset
from utils_cls.tools import resize, init_seed, reduce_tensor
from models_cls.models import Resnet18, MobilenetV2Ori, Resnet50, Resnet101, Resnet18_Bcnn, BCNN, MobilenetV2, Resnet11
from utils_cls.loss import Focal_loss

sys.path.append("..")
from utils.general import increment_path

import argparse
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as Sampler
from utils.torch_utils import select_device

local_rank = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
rank = int(os.getenv('RANK', -1))
world_size = int(os.getenv('WORLD_SIZE', 1))


def train(opt, device):
    pretrained_weight, save_dir, epochs, learning_rate, batch_size, optimizer, data, workers, used_model, img_size, loss_mode = \
        opt.pretrained_weight, opt.save_dir, opt.epochs, opt.learning_rate, opt.batch_size, opt.optimizer, \
            opt.data, opt.workers, opt.used_model, opt.img_size, opt.loss_mode
    with open(data, 'r') as f:
        data_dict = yaml.safe_load(f)
    #   获取训练/测试文件夹路径
    train_dir = data_dict["train"]
    val_dir = data_dict["val"]
    classes_num = data_dict["nc"]

    best_fi = 0.0

    #   保存模型路径
    if local_rank in [-1, 0]:
        w = str(increment_path(Path(save_dir).absolute() / opt.name, exist_ok=False))
        w = Path(w) / "weights"
        w.mkdir(parents=True, exist_ok=True)
        best_pt = w / 'best.pt'
        last_pt = w / 'last.pt'

    #   创建数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )

    #  创建dataloader
    train_dataset = load_image_dataset(train_dir, train_transform, img_size=img_size)
    val_dataset = load_image_dataset(val_dir, val_transform, img_size=img_size)

    #   如果是GPU且是DDP模式，构建DDP sampler
    if torch.cuda.is_available() and rank != -1:
        train_sampler = Sampler(train_dataset)
        val_sampler = Sampler(val_dataset)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    #   创建dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler,
                              num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=workers)

    #   创建模型
    if used_model == "resnet18":
        model = Resnet18(classes_num, pretrained_weight)
    elif used_model == "mobilenetv2":
        model = MobilenetV2Ori(classes_num, pretrained_weight)
    elif used_model == "resnet50":
        model = Resnet50(classes_num, pretrained_weight)
    elif used_model == "resnet101":
        model = Resnet101(classes_num, pretrained_weight)
    elif used_model == "resnet18_bcnn":
        model = Resnet18_Bcnn(classes_num, pretrained_weight)
    elif used_model == "BCNN":
        model = BCNN(classes_num)
    elif used_model == "mobilenetv2_1":
        model = MobilenetV2(classes_num, pretrained_weight)
    elif used_model == "resnet11":
        model = Resnet11(classes_num)
    model.to(device)

    #   将模型加载到DDP中
    if torch.cuda.is_available() and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 同步BN
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    #   优化器选择
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.00005)

    #   学习率下降策略
    multi_step_schrdule = optim.lr_scheduler.MultiStepLR(optimizer, [20, 60, 120, 140], 0.1)
    # multi_step_schrdule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0000001)
    # multi_step_schrdule = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=learning_rate, step_size_up=20, mode="triangular2", cycle_momentum=False)

    #   损失函数选择
    if loss_mode == "focal_loss":
        weigth_cls = torch.FloatTensor([3.0, 1.0]).to(device)
        gamma = 2.0
        criterion = Focal_loss(weigth_cls, gamma, device).to(device)
    elif loss_mode == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    if local_rank in [-1, 0]:
        print("model save path: " + str(w.parents[0]))
    #   开始训练
    for epoch in range(epochs):
        model.train()

        #   打印进度条
        pbar = enumerate(train_loader)
        num_batch = len(train_loader)
        if local_rank in [-1, 0]:
            pbar = tqdm(pbar, total=num_batch, leave=True)

        #   DDP模式下的shuffle操作
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)

        #   训练
        for i, (datas, labels) in pbar:
            datas, labels = datas.to(device), labels.to(device)
            outputs = model(datas)
            _, preds = outputs.max(dim=1)
            train_acc = torch.sum(preds == labels) / preds.shape[0]
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #   再gpu0中打印信息
            # if local_rank in [-1, 0] and i % 10 == 0:
            #     print("[{}/{}]training loss: {:4f}  training accuracy: {:.4f}".format(epoch, epochs, loss.item(), train_acc.item()))
            if local_rank in [-1, 0]:
                pbar.set_description(f'Epoch [{epoch}/{epochs}]')
                pbar.set_postfix(loss=loss.item(), training_acc=train_acc.item())
        #   执行学习率下降策略
        multi_step_schrdule.step()

        #   验证
        if local_rank in [-1, 0]:
            with torch.no_grad():
                val_pbar = enumerate(val_loader)
                val_batch = len(val_loader)
                if local_rank == 0:
                    val_pbar = tqdm(val_pbar, total=val_batch, leave=True)
                model.eval()
                all_preds = torch.empty((0,))
                all_labels = torch.empty((0,))

                for i, (datas, labels) in val_pbar:
                    datas, labels = datas.to(device), labels.to(device)
                    outputs = model(datas)
                    _, preds = torch.max(outputs, dim=1)
                    all_preds = torch.cat((all_preds, preds.data.cpu()), dim=0)
                    all_labels = torch.cat((all_labels, labels.data.cpu()), dim=0)

                val_acc = (all_preds == all_labels).sum() / all_preds.shape[0]
                print(f"val accuracy: {val_acc.item()}")

                #   混淆矩阵
                sns.set()
                confusion_matix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
                f, ax = plt.subplots()
                sns.heatmap(confusion_matix, annot=True, ax=ax, fmt='.4f')
                ax.set_title("confusion matrix")
                ax.set_xlabel("predict")
                ax.set_ylabel("true")
                plt.close()

                '''
                #  多gpu测试时计算准确率
                if torch.cuda.is_available() and rank != -1:
                    dist.barrier()
                    correct_pred = reduce_tensor(correct_pred, "sum")
                    total = reduce_tensor(total, "sum")
                if local_rank in [-1, 0]:
                    val_acc = correct_pred / total
                    print(f"val accuracy: {val_acc.item()}")
                '''

            #   保存模型
            if local_rank in [-1, 0]:
                #   保存最优模型
                if val_acc.item() >= best_fi:
                    plt.savefig(os.path.join(str(w.parents[0]), "best_confusion_matrix.png"), format="png")
                    best_fi = val_acc.item()
                    if type(model) in [nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]:
                        best_model_state = copy.deepcopy(model.module.state_dict())
                        torch.save(best_model_state, best_pt, _use_new_zipfile_serialization=False)
                    else:
                        best_model_state = copy.deepcopy(model.state_dict())
                        torch.save(best_model_state, best_pt, _use_new_zipfile_serialization=False)

                #   保存最新模型
                plt.savefig(os.path.join(str(w.parents[0]), "last_confusion_matrix.png"), format="png")
                if type(model) in [nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]:
                    torch.save(model.module.state_dict(), last_pt, _use_new_zipfile_serialization=False)
                else:
                    torch.save(model.state_dict(), last_pt, _use_new_zipfile_serialization=False)

    if local_rank in [-1, 0]:
        print("model save path: " + str(w.parents[0]))


def parse_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data", type=str, default="./data/data.yaml",
                       help="file that include the path of training images and test images")
    parse.add_argument("--epochs", type=int, default=100)
    parse.add_argument("--learning_rate", type=float, default=0.01)
    parse.add_argument("--batch_size", type=int, default=64)
    parse.add_argument("--optimizer", type=str, default="adam", help="[SGD, adam]")
    parse.add_argument("--local_rank", type=int, default=-1)
    parse.add_argument("--seed", type=int, default=0, help="")
    parse.add_argument("--workers", type=int, default=8, help="加载数据的线程数")
    parse.add_argument("--save_dir", type=str, default="./runs/train/")
    parse.add_argument("--name", type=str, default='exp', help="保存的模型的文件夹名")
    parse.add_argument("--pretrained_weight", type=str, default='./data/resnet18-f37072fd.pth',
                       help="分类模型的预训练权重")
    parse.add_argument("--used_model", type=str, default="resnet18", help="选择分类网络模型")
    parse.add_argument("--img_size", type=int, default=112)
    parse.add_argument("--loss_mode", type=str, default="cross_entropy", help="focal_loss or cross_entropy")
    parse.add_argument("--device", default='0')

    return parse.parse_args()


def main(opt):
    if local_rank in [-1, 0]:
        print("start training")
    #   初始化随机数种子，不同的GPU使用不同的随机数种子
    init_seed(opt.seed + 1 + rank)
    #   配置单机多卡训练
    device = select_device(opt.device,
                           batch_size=opt.batch_size)  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)

    train(opt, device)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
