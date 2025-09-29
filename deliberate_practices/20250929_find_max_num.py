#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

"""
练习：找出一个数组中的最大值的下标

步骤：
构建模型类
构建数据集
构建训练函数
构建预测函数
"""

# 当前文件目录
CURRENT_DIR = os.path.dirname(__file__)
# 模型目录
MODEL_DIR = os.path.join(CURRENT_DIR, 'models_output/model_20250929.pth')

class Net(nn.Module):
    
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.l = nn.Linear(input_size, input_size)
        self.loss = nn.functional.cross_entropy
        
    def forward(self, x, y=None):
        x = self.l(x)
        return x if y is None else self.loss(x, y)
    
# 构建数据集
def generate_datas(input_size, total):
    x, y = [], []
    for _ in range(total):
        _x = np.random.random(input_size)
        _y = np.argmax(_x)
        x.append(_x)
        y.append(_y)
    x = np.array(x)
    y = np.array(y)
    return torch.FloatTensor(x), torch.LongTensor(y)

# 构建预测函数
def evaluate(model, input_size, total=500):
    model.eval()
    x, y = generate_datas(input_size, total)
    y_pred = model(x)
    return (y_pred.argmax(dim=1) == y).float().mean().item()
    
# 构建训练函数
def train():
    input_size = 5
    epoch_total = 200
    batch_size = 20
    batch_total = 1000
    learn_rate = 0.001
    
    batch_max = math.ceil(batch_total / batch_size)
    
    # 生成训练数据
    train_x_lst, train_y_lst = generate_datas(input_size, batch_total)
    
    # 创建模型
    model = Net(input_size)
    
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    # 日志收集
    train_logs = []
    
    for epoch in range(epoch_total):
        model.train()
        
        # loss日志收集
        watch_logs = []
        
        # 每次都打乱数据集
        indices = torch.randperm(batch_total)
        train_x_lst = train_x_lst[indices]
        train_y_lst = train_y_lst[indices]
        
        for batch_index in range(batch_max):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, batch_total)
            
            train_x = train_x_lst[start_index:end_index]
            train_y = train_y_lst[start_index:end_index]
            
            optim.zero_grad()
            loss = model.forward(train_x, train_y)
            loss.backward()
            optim.step()
            
            watch_logs.append(loss.item())
            
        # 计算平均loss
        mean_loss = np.mean(watch_logs)
        
        # 预测准确率
        acc = evaluate(model, input_size)
        print(f'epoch: [{epoch+1}/{epoch_total}], loss: {mean_loss}, acc: {acc}')
        
        # 记录训练日志
        train_logs.append([acc, mean_loss])
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_DIR)
        
    # 打印训练日志图
    plt_x_lst = range(len(train_logs))
    plt.plot(plt_x_lst, [l[0] for l in train_logs], label='acc')
    plt.plot(plt_x_lst, [l[1] for l in train_logs], label='loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    input_size = 5
    
    # 测试生成数据集函数
    x, y = generate_datas(input_size, 1)
    for _x, _y in zip(x, y):
        print(_x, _y)
        
    # 训练
    train()
    
    # # 使用训练好的模型
    # model = Net(input_size)
    # model.load_state_dict(torch.load(MODEL_DIR))
    # acc = evaluate(model, input_size)
    # print(f'acc: {acc}')