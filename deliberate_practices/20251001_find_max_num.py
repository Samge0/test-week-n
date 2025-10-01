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
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# 当前目录
CURRENT_DIR = os.path.dirname(__file__)
# 模型目录
MODEL_PATH = os.path.join(CURRENT_DIR, "models_output/model_20251001.pth")

# 训练参数
train_config = {
    'input_size': 5,
    'epochs': 200,
    'batch_size': 20,
    'batch_total': 1000,
    'lr': 0.001,
    'current_date': '20251001',
}

# 构建模型类
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
    for i in range(total):
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
    with torch.no_grad():
        y_pred = model(x)
        return (y_pred.argmax(dim=1) == y).float().mean().item()

# 构建训练函数
def train():
    input_size = int(train_config.get("input_size", 5))
    epochs = int(train_config.get("epochs", 200))
    batch_size = int(train_config.get("batch_size", 20))
    batch_total = int(train_config.get("batch_total", 1000))
    lr = float(train_config.get("lr", 0.01))
    batch_max = math.ceil(batch_total / batch_size)
    
    # 创建训练数据
    train_x_lst, train_y_lst = generate_datas(input_size, batch_total)
    
    # 构建模型 = 优化器
    model = Net(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    # swanlab日志记录
    swanlab.init(
        workspace="samge",
        project="test-week-n",
        config=train_config,
        model=model,
        optimizer=optim,
        experiment_name=f"find_max_num-{train_config.get('current_date', '')}-{int(time.time())}",
    )
    
    # 训练日志记录
    train_logs = []
    
    for epoch in range(epochs):
        model.train()
        
        # 每次打乱数据
        indices = torch.randperm(batch_total)
        train_x_lst = train_x_lst[indices]
        train_y_lst = train_y_lst[indices]
        
        # 每批次的loss记录
        warch_logs = []
        
        for batch_index in range(batch_max):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, batch_total)
            
            train_x = train_x_lst[start_index:end_index]
            train_y = train_y_lst[start_index:end_index]
            
            optim.zero_grad()
            loss = model(train_x, train_y)
            loss.backward()
            optim.step()
            
            warch_logs.append(loss.item())
            
        # 记录评价损失
        mean_loss = np.mean(warch_logs)
        
        # 预测准确率
        acc = evaluate(model, input_size)
        
        # 记录训练日志
        train_logs.append([acc, mean_loss])
        swanlab.log({"acc": acc, "loss": mean_loss})
        print(f"epoch: [{epoch+1}/{epochs}], acc: {acc}, loss: {mean_loss}")
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        
    # 训练完成，打印日志
    swanlab.finish()
    plt_x_lst = range(len(train_logs))
    plt.plot(plt_x_lst, [log[0] for log in train_logs], label="acc")
    plt.plot(plt_x_lst, [log[1] for log in train_logs], label="loss")
    plt.legend()
    plt.show()
        
if __name__ == '__main__':
    input_size = 5
    
    # 测试生成数据集
    x, y = generate_datas(input_size, 1)
    for _x, _y in zip(x, y):
        print(_x, _y)
        
    # 训练
    train()
    
    # 使用已有模型测试
    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    acc = evaluate(model, input_size)
    print(f"acc: {acc}")