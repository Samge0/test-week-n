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
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 当前目录
CURRENT_DIR = os.path.dirname(__file__)
# 模型目录
MODEL_PATH = os.path.join(CURRENT_DIR, "models_output/model_20251004.pth")

# 训练配置
train_config = {
    "input_size": 5,
    "epochs": 200,
    "batch_size": 20,
    "batch_total": 1000,
    "lr": 0.001,
    "current_date": "20251004",
}


# 创建模型类
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

# 构建预测函数
def evaluate(model, input_size, total=500):
    model.eval()
    x, y = generate_datas(input_size, total)
    with torch.no_grad():
        y_pred = model(x)
        return (y_pred.argmax(dim=1) == y).sum().item() / total
    
# 训练
def train():
    input_size = int(train_config.get('input_size', 5))
    epochs = int(train_config.get('epochs', 200))
    batch_size = int(train_config.get('batch_size', 20))
    batch_total = int(train_config.get('batch_total', 1000))
    lr = float(train_config.get('lr', 0.001))
    current_date = str(train_config.get('current_date',''))
    batch_max = math.ceil(batch_total / batch_size)
    
    # 训练数据集
    train_x_lst, train_y_lst = generate_datas(input_size, batch_total)
    
    # 模型+优化器
    model = Net(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练日志记录
    train_logs = []
    swanlab.init(
        workspace="samge",
        project="test-week-n",
        model=model,
        optimizer=optim,
        experiment_name=f"find_max_num-{current_date}-{int(time.time())}",
        config=train_config,
        tags=["test", "deliberate_practice", "find_max_num", current_date],
    )
    
    for epoch in range(epochs):
        model.train()
        
        # 打乱数据
        indices = torch.randperm(batch_total)
        train_x_lst = train_x_lst[indices]
        train_y_lst = train_y_lst[indices]
        
        # 记录训练loss
        watch_loss = []
        
        for batch_idx in range(batch_max):
            start_index = batch_idx * batch_size
            end_index = min(start_index+batch_size, batch_total)
            x = train_x_lst[start_index:end_index]
            y = train_y_lst[start_index:end_index]
            
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            
            watch_loss.append(loss.item())
        
        # 计算平均loss+acc
        mean_loss = np.mean(watch_loss)
        acc = evaluate(model, input_size)
        train_logs.append([acc, mean_loss])
        swanlab.log({"acc": acc, "loss": mean_loss})
        print(f"epoch: {epoch+1}/{epochs}, loss: {mean_loss:.4f}, acc: {acc:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        
    swanlab.finish()
    plt.plot(np.array(train_logs)[:, 0], label="acc")
    plt.plot(np.array(train_logs)[:, 1], label="loss")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    input_size = int(train_config.get('input_size', 5))
    
    # 训练
    train()
    
    # 测试
    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"acc: {evaluate(model, input_size)}")
