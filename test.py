import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output1, output2, target):
        # 计算output1和target的交叉熵损失
        l1 = self.cross_entropy_loss(output1, target)
        l1 = torch.sigmoid(l1)
        # 计算output2和target的交叉熵损失
        l2 = self.cross_entropy_loss(output2, target)
        
        # 将l1和l2逐元素相乘
        multiplied_loss = l1 * l2
        
        # 返回平均值
        return multiplied_loss.mean()

# 使用示例
output1 = torch.randn(10, 5, requires_grad=True)  # 假设有10个样本，5个类别
output2 = torch.randn(10, 5, requires_grad=True)  # 假设有10个样本，5个类别
target = torch.randint(0, 5, (10,))  # 目标类别 (10个样本)

criterion = CustomCrossEntropyLoss()
loss = criterion(output1, output2, target)
loss.backward()

print(loss)