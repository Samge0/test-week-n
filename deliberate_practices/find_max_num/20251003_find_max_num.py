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
import math
import time
import torch
import swanlab
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 当前目录
CURRENT_DIR = os.path.dirname(__file__)
# 模型目录
MODEL_PATH = os.path.join(CURRENT_DIR, "models_output/model_20251003.pth")

# 训练配置
train_config = {
    "input_size": 5,
    "epochs": 200,
    "batch_size": 20,
    "batch_total": 1000,
    "lr": 0.001,
    "current_date": "20251003",
}

# 定义模型类
class Net(nn.Module):
    
    def __init__(self, input_size) -> None:
        super().__init__()
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

# 预测函数
def evaluate(model, input_size, total=500):
    model.eval()
    x, y = generate_datas(input_size, total)
    with torch.no_grad():
        y_pred = model(x)
        return (y_pred.argmax(dim=1) == y).float().mean().item()
    
# 训练函数
def train():
    input_size = int(train_config.get('input_size', 0))
    epochs = int(train_config.get('epochs', 0))
    batch_size = int(train_config.get('batch_size', 0))
    batch_total = int(train_config.get('batch_total', 0))
    lr = float(train_config.get('lr', 0))
    current_date = float(train_config.get('current_date', ''))
    
    batch_max = math.ceil(batch_total/batch_size)
    
    model = Net(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_x_lst, train_y_lst = generate_datas(input_size, batch_total)
    
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
        
        indices = torch.randperm(batch_total)
        train_x_lst = train_x_lst[indices]
        train_y_lst = train_y_lst[indices]
        
        watch_logs = []
        for batch_index in range(batch_max):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, batch_total)
            train_x = train_x_lst[start_index:end_index]
            train_y = train_y_lst[start_index:end_index]
            
            optim.zero_grad()
            loss = model(train_x, train_y)
            loss.backward()
            optim.step()
            
            watch_logs.append(loss.item())
            
        # loss + acc
        mean_loss = np.mean(watch_logs)
        acc = evaluate(model, input_size)
        
        train_logs.append([acc, mean_loss])
        swanlab.log({"acc": acc, "loss": mean_loss})
        print(f"[{epoch+1}/{epochs}], acc: {acc}, loss: {mean_loss}")
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        
    swanlab.finish()
    plt_x_lst = range(len(train_logs))
    plt.plot(plt_x_lst, [l[0] for l in train_logs], label = 'acc')
    plt.plot(plt_x_lst, [l[1] for l in train_logs], label = 'loss')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    input_size = 5
    x, y = generate_datas(input_size, 1)
    for _x, _y in zip(x, y):
        print(_x, _y)
        
    # 训练
    train()
    
    # 使用模型预测
    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    acc = evaluate(model, input_size)
    print(f"acc: {acc}")