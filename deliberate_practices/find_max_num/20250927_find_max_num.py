#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
练习：找出一个数组中的最大值的下标

步骤：
构建模型
构建生成数据集函数
构建训练函数
构建预测函数
运行
"""

import math
import torch
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt

# 当前文件路径
import os
cur_path = os.path.abspath(os.path.dirname(__file__))

# 模型保存路径
MODEL_PATH = os.path.join(cur_path, 'models_output/model_20250927.pth')


class Net(nn.Module):
    
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.loss = nn.functional.cross_entropy
        
    def forward(self, x, y=None):
        x = self.linear(x)
        return x if y is None else self.loss(x, y)
    
# 生成数据集
def generate_dataset(input_size, total):
    x, y = [], []
    for _ in range(total):
        _x = np.random.random(input_size)
        _y = np.argmax(_x)
        x.append(_x)
        y.append(_y)
        
    x = np.array(x)
    y = np.array(y)
    return torch.FloatTensor(x), torch.LongTensor(y)

# 预测模型
def evaluate(model, input_size, total=500):
    model.eval()
    x, y = generate_dataset(input_size, total)
    y_pred = model(x)
    y_pred = torch.argmax(y_pred, dim=1)
    acc = (y_pred == y).sum().item() / len(y)
    return acc

# 训练函数
def train():
    epoch_total = 200
    batch_size = 2000
    batch_total = 100000
    learn_rate = 0.001
    input_size = 5
    
    max_batch = math.ceil(batch_total / batch_size)
    
    # 创建模型
    model = Net(input_size)
    
    # 创建优化器
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    # 创建数据集
    x_train_lst, y_train_lst = generate_dataset(input_size, batch_total)
    
    # 训练日志记录
    train_logs = []
    
    for epoch in range(epoch_total):
        model.train()
        
        # 单轮训练日志记录
        watch_logs = []
        
        # 每次都打乱训练数据
        indices = torch.randperm(batch_total)
        x_train_lst = x_train_lst[indices]
        y_train_lst = y_train_lst[indices]
        
        for batch_index in range(max_batch):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, batch_total)
            x_train = x_train_lst[start_index:end_index]
            y_train = y_train_lst[start_index:end_index]
            
            optim.zero_grad()
            loss = model(x_train, y_train)
            loss.backward()
            optim.step()
            
            # 记录日志
            watch_logs.append(loss.item())
            
        # 测试模型准确率 + 平均损失
        acc = evaluate(model, input_size)
        mean_loss = np.mean(watch_logs)
        train_logs.append([acc, mean_loss])
        
        print(f'epoch: [{epoch+1}/{epoch_total}], loss: {mean_loss}, acc: {acc}')
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        
    # 打印训练日志
    x_lst = range(len(train_logs))
    plt.plot(x_lst, [l[0] for l in train_logs], label='acc')
    plt.plot(x_lst, [l[1] for l in train_logs], label='loss')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    input_size = 5
    
    # 测试数据集生成
    x, y = generate_dataset(input_size, 1)
    for _x, _y in zip(x, y):
        print(_x, _y)
        
    # 训练
    train()
    
    # 使用训练后的模型测试
    # model = Net(input_size)
    # model.load_state_dict(torch.load(MODEL_PATH))
    # acc = evaluate(model, input_size)
    # print(f'acc: {acc}')
        