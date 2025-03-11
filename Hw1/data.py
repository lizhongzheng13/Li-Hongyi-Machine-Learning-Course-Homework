# @Author : LiZhongzheng
# 开发时间  ：2025-03-09 16:16
# 用于处理数据的部分东西
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, x, y=None):  # x是输入的特征值，y是目标值。当y为none的时候表明这是我们需要预测的部分
        super(MyDataset, self).__init__()
        self.x = torch.FloatTensor(x)  # 转为张量的格式
        # self.y = torch.FloatTensor(y)
        if y is not None:
            self.y = torch.FloatTensor(y)
        else:
            self.y = None

    def __getitem__(self, index):
        if self.y is None:  # 做预测
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
