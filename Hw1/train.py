# @Author : LiZhongzheng
# 开发时间  ：2025-03-09 16:37
# 训练模型

import os
import csv
import math
import torch
import shutil
import swanlab
import numpy as np
from sympy.physics.units import momentum
from data import *
from model import *
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter  # 将日志数据写入到 runs 目录中
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_val_data_process(data_set, val_ratio, seed=999):
    """ 划分训练集和验证集 """
    # 训练集用于训练模型，验证集用于评估模型的性能。
    val_size = int(val_ratio * len(data_set))  # 验证集所占的大小
    train_size = len(data_set) - val_size  # 训练集所占的大小
    # 数据集的划分
    train_data, val_data = random_split(data_set, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_data), np.array(val_data)


def set_seed(seed):
    """ 设置随机种子，保证可复现 """
    regular_seed = seed
    np.random.seed(regular_seed)
    torch.manual_seed(regular_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(regular_seed)


def predict(test_loader, model, device):  # 使用训练好的神经网络模型 对测试数据集 进行预测
    """ 预测模型输出 """
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):  # 进度条
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def select_feature(train_data, val_data, test_data, select_all=True):
    y_train = train_data[:, -1]  # 最后一列
    y_val = val_data[:, -1]

    raw_x_train = train_data[:, :-1]  # 不要最后一列元素
    raw_x_val = val_data[:, :-1]

    raw_x_test = test_data  # 用于测试集

    if select_all:  # 选择全部特征的话，用于训练
        feat_idx = list(range(raw_x_train.shape[1]))  # 特征的数目
        print(feat_idx)
    else:
        feat_idx = [0, 1, 2, 3, 4]
    return raw_x_train[:, feat_idx], raw_x_val[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_val


def model_train_process(train_loader, val_loader, model, config, device):
    if os.path.exists('runs'):
        print("I will create the new file,and I am deleting the old file!\n")
        shutil.rmtree('runs')  # 删除整个文件夹

    # criterion = nn.MSELoss(reduction='mean')  # 均方误差函数 #返回样本的平均值 #default
    criterion = nn.SmoothL1Loss(reduction='mean')

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.7,
                                weight_decay=1e-5)  # 随机梯度下降法
    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    writer = SummaryWriter()  # 可视化

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    epochs = config['epochs']  # 训练次数
    best_loss = math.inf  # inf：无穷大
    step = 0
    early_stopping = 0  # 模型若没有改变，则加一

    for epoch in range(epochs):  # 开始训练~~~
        # 训练模式
        model.train()
        loss_record = []
        train_process = tqdm(train_loader, position=0, leave=True, disable=True)  # 进度条
        # train_process = train_loader
        for x, y in train_process:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)  # 预测值与真实值之间的差距
            loss.backward()
            optimizer.step()  # 更新模型
            step += 1
            loss_record.append(loss.detach().item())  # detach()相当于这个数据的张量形式

            # 显示训练过程
            train_process.set_description(f'Epoch[{epoch + 1}/{epochs}]')
            train_process.set_postfix({f'loss': loss.detach().item()})  # 当前的损失函数的值

        # 平均损失值
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('loss/train', mean_train_loss, epoch)  # loss函数值的图的绘制
        swanlab.log({"train_loss": mean_train_loss}, step=epoch)  # 记录训练损失

        # 验证模式
        model.eval()
        loss_record = []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                loss_record.append(loss.detach().item())
        mean_val_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch[{epoch + 1}/{epochs}]:Train loss:{mean_train_loss:.4f},val loss:{mean_val_loss:.4f}\n')
        writer.add_scalar('loss/val', mean_train_loss, mean_val_loss, step)
        swanlab.log({"val_loss": mean_val_loss}, step=epoch)  # 记录验证损失

        if mean_val_loss < best_loss:  # 是否需要优化模型
            best_loss = mean_val_loss
            torch.save(model.state_dict(), config["save_model"])
            print('Saving  model with loss{:.3f}'.format(best_loss))
            early_stopping = 0
        else:
            early_stopping += 1

        if early_stopping >= config['early_stopping']:
            print("模型没有提升，中止训练！\n")
            return


if __name__ == '__main__':
    # gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {  # 定义参数的值
        'seed': 5201314,
        'select_all': True,
        'val_ratio': 0.2,
        'epochs': 5000,
        'batch_size': 128,  # 256
        'learning_rate': 0.00001,
        'early_stopping': 800,  # 连续执行800次后若是model没有更新，退出执行
        'save_model': 'models/model.ckpt',  # TensorFlow中用于存储模型参数的一种文件格式 #中断继续

    }
    np.save('config.npy', config)  # 保存x_train
    # 初始化 swanlab
    swanlab.init(
        experiment_name="covid_model_train",
        config=config
    )
    # 填入具体的参数
    set_seed(config['seed'])
    train_data = pd.read_csv('ml2023spring-hw1/covid_train.csv').values
    test_data = pd.read_csv('ml2023spring-hw1/covid_test.csv').values
    # 划分数据集
    train_data, val_data = train_val_data_process(train_data, config['val_ratio'], config['seed'])
    print(
        f"""train_data size: {train_data.shape} valid_data size: {val_data.shape} test_data size: {test_data.shape}""")
    # 选择特征
    x_train, x_val, x_test, y_train, y_val = select_feature(train_data, val_data, test_data, config['select_all'])
    np.save('x_train.npy', x_train)  # 保存x_train
    print(f'number of features: {x_train.shape[1]}')
    # 构造数据集
    train_dataset = MyDataset(x_train, y_train)
    val_dataset = MyDataset(x_val, y_val)
    test_dataset = MyDataset(x_test)
    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    np.save('test_loader.npy', test_loader)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    model = MyModel(input=x_train.shape[1]).to(device)
    model_train_process(train_loader, val_loader, model, config, device)
