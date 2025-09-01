import torch.nn as nn

class FindMaxNumModel(nn.Module):     
    """ 自定义模型，识别最大数字的模型 """
    
    def __init__(self, input_size):
        super(FindMaxNumModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is None:
            # 预测模式，返回预测结果
            return x
        else:
            # 训练模式，返回loss
            return self.loss(x, y)