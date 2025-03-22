# @Author : LiZhongzheng
# 开发时间  ：2025-03-09 15:29
# 构建模型： 创建一个神经网络模型来预测目标值。
import torch
import torch.nn as nn
from torchsummary import summary


class MyModel(nn.Module):
    def __init__(self, input):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input, 16),  # 全连接层后常加上一个激活函数
            nn.ReLU(),  # 如果没有激活函数，神经网络将只能表示线性函数，即使有再多的层，其整体功能也等同于一个简单的线性模型
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        # self.layers = nn.Sequential(
        #     nn.Linear(input, 64),  # 全连接层后常加上一个激活函数
        #     nn.ReLU(),  # 如果没有激活函数，神经网络将只能表示线性函数，即使有再多的层，其整体功能也等同于一个简单的线性模型
        #     nn.Linear(64, 1),
        # )
        # for m in self.modules():
        #     """权重初始化"""
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = MyModel(2).to(device)
#     print(summary(model, (2,)))
