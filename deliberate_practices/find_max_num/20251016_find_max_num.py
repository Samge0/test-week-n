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

import os, math, time, torch, swanlab
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 当前目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型路径
MODEL_PATH = os.path.join(CURRENT_DIR, 'models_output/model_20251017.pth')

# 训练配置
train_config = {
    "input_size": 6,
    "epochs": 300,
    "batch_size": 40,
    "batch_total": 2000,
    "lr": 0.001,
    "current_date": "20251016"
}

# 模型类
class Net(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.l = nn.Linear(input_size, input_size)
        self.loss = F.cross_entropy
        
    def forward(self, x, y=None):
        x = self.l(x)
        return x if y is None else self.loss(x, y)
    
# 生成模拟数据
def generate_datas(input_size, total):
    x = np.random.random((total, input_size))
    y = np.argmax(x, axis=1)
    return torch.FloatTensor(x), torch.LongTensor(y)

# 预测
def evaluate(model, input_size, total=500):
    model.eval()
    x, y = generate_datas(input_size, total)
    with torch.no_grad():
        y_pred = model(x)
        return y_pred.argmax(dim=1).eq(y).sum().item() / total
    
# 训练
def train():
    input_size = train_config["input_size"]
    epochs = train_config["epochs"]
    batch_size = train_config["batch_size"]
    batch_total = train_config["batch_total"]
    lr = train_config["lr"]
    current_date = train_config["current_date"]
    
    batch_max = math.ceil(batch_total / batch_size)
    
    train_x, train_y = generate_datas(input_size, batch_total)
    model = Net(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_logs = []
    swanlab.init(
        workspace="samge",
        project="test-week-n",
        config=train_config,
        tags=["find-max-num", "test", current_date],
        experiment_name=f"find-max-num-{current_date}-{int(time.time())}"
    )
    
    for epoch in range(epochs):
        model.train()
        
        indices = torch.randperm(batch_total)
        train_x = train_x[indices]
        train_y = train_y[indices]
        
        loss_logs = []
        
        for batch_idx in range(batch_max):
            start_i = batch_idx * batch_size
            end_i = min(start_i + batch_size, batch_total)
            x = train_x[start_i:end_i]
            y = train_y[start_i:end_i]
            
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            loss_logs.append(loss.item())
            
        acc = evaluate(model, input_size)
        mean_loss = np.mean(loss_logs)
        train_logs.append([acc, mean_loss])
        swanlab.log({"acc": acc, "loss": mean_loss})
        print(f"epoch: [{epoch+1}/{epochs}], acc: {acc}, loss: {mean_loss}")
        
        torch.save(model.state_dict(), MODEL_PATH)
        
    swanlab.finish()
    plt.plot(np.array(train_logs)[:, 0], label="acc")
    plt.plot(np.array(train_logs)[:, 1], label="loss")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    # 训练
    train()
    
    # 测试
    input_size = train_config["input_size"]
    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"acc: {evaluate(model, input_size):.4f}")