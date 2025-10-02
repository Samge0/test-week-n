#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
练习：找出一个数组中的最大值的下标

步骤：
构建模型类
构建数据集函数
构建训练函数
构建测试函数
"""

import os
import time
import math
import torch
import swanlab
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# 当前目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型目录
MODEL_PATH = os.path.join(CURRENT_DIR, "models_output/model_20251002.pth")

# 训练配置
train_config = {
    "input_size": 5,
    "epochs": 200,
    "batch_size": 20,
    "batch_total": 1000,
    "lr": 0.001,
    "current_date": "20251002",
}

# 构建模型类
class Net(nn.Module):

    def __init__(self, input_size) -> None:
        super().__init__()
        self.l = nn.Linear(input_size, input_size)
        self.loss = nn.functional.cross_entropy
        
    def forward(self, x, y=None):
        x = self.l(x)
        return x if y is None else self.loss(x, y)

# 生成测试集的函数
def generate_dates(input_size, total):
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
    x, y = generate_dates(input_size, total)
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        return (y_pred.argmax(dim=1) == y).float().mean().item()
    
# 构建训练函数
def train():
    input_size = int(train_config.get('input_size', 5))
    epochs = int(train_config.get('epochs', 200))
    batch_size = int(train_config.get('batch_size', 20))
    batch_total = int(train_config.get('batch_total', 1000))
    lr = float(train_config.get('lr', 0.001))
    current_date = train_config.get('current_date', '') 
    batch_max = math.ceil(batch_total/batch_size)
    
    # 生成训练数据
    train_x_lst, train_y_lst = generate_dates(input_size, batch_total)
    
    # 构建模型 + 优化器
    model = Net(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 日志记录
    train_logs = []
    swanlab.init(
        workspace="samge",
        project="test-week-n",
        config=train_config,
        model=model,
        optimizer=optim,
        experiment_name=f"find_max_num-{current_date}-{int(time.time())}",
    )
    
    for epoch in range(epochs):
        model.train()
        
        # 每次都打乱数据
        indices = torch.randperm(batch_total)
        train_x_lst = train_x_lst[indices]
        train_y_lst = train_y_lst[indices]
        
        # 每批次的loss记录
        watch_logs = []
        
        for batch_index in range(batch_max):
            start_index = batch_index * batch_size
            end_index = min(start_index+batch_size, batch_total)
            
            train_x = train_x_lst[start_index:end_index]
            train_y = train_y_lst[start_index:end_index]
            
            optim.zero_grad()
            loss = model(train_x, train_y)
            loss.backward()
            optim.step()
            
            watch_logs.append(loss.item())
            
        # 计算平均损失
        mean_loss = np.mean(watch_logs)
        # 计算预测的准确率
        acc = evaluate(model, input_size)
        
        train_logs.append([acc, mean_loss])
        swanlab.log({"acc": acc, "loss": mean_loss})
        print(f"[{epoch+1}/{epochs}], acc: {acc}, loss: {mean_loss}")
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        
    # 训练完成，打印日志图
    swanlab.finish()
    plt_x_lst = range(len(train_logs))
    plt.plot(plt_x_lst, [l[0] for l in train_logs], label='acc')
    plt.plot(plt_x_lst, [l[1] for l in train_logs], label='loss')
    plt.legend()
    plt.show()
            
    pass

if __name__ == '__main__':
    input_size = 5
    
    x, y = generate_dates(input_size, 1)
    for _x, _y in zip(x, y):
        print(_x, _y)
        
    # 训练
    train()
    
    # 使用训练后的模型测试
    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    acc = evaluate(model, input_size)
    print(f'acc: {acc}')
        
        
        
        
        
        
        
    
