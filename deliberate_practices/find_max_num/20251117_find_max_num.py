#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time, torch, swanlab
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# 当前日期
CURRENT_DATE = time.strftime("%Y%m%d", time.localtime())

# 当前目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型目录
MODEL_PATH = f"{CURRENT_DIR}/models_output/model_{CURRENT_DATE}.pth"

# 模型配置
train_config = {
    "input_size": 17,
    "epochs": 200,
    "batch_size": 20,
    "batch_total": 20000,
    "lr": 0.001,
    "current_date": CURRENT_DATE
}

class Net(nn.Module):
    
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.l = nn.Linear(input_size, input_size)
        self.loss = F.cross_entropy
        
    def forward(self, x, y=None):
        x = self.l(x)
        return x if y is None else self.loss(x, y)
    
def generate_datas(input_size, total):
    x = np.random.random((total, input_size))
    y = np.argmax(x, axis=1)
    return torch.FloatTensor(x), torch.LongTensor(y)

def evaluate(model, input_size, total=500):
    model.eval()
    x, y = generate_datas(input_size, total)
    with torch.no_grad():
        y_pred = model(x)
        return (y_pred.argmax(dim=1) == y).float().mean()
    
def train():
    epochs, input_size, batch_size, batch_total, lr = train_config["epochs"], train_config["input_size"], train_config["batch_size"], train_config["batch_total"], train_config["lr"]
    batch_max = math.ceil(batch_total / batch_size)
    
    train_logs = []
    swanlab.init(
        namespace="samge",
        project="test-week-n",
        config=train_config,
        experiment_name=f"find_max_num_{CURRENT_DATE}_{int(time.time())}",
        tags=["find_max_num", "test", CURRENT_DATE]
    )
    
    model = Net(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_x, train_y = generate_datas(input_size, batch_total)
    
    for epoch in range(epochs):
        model.train()
        
        indices = torch.randperm(batch_total)
        train_x, train_y = train_x[indices], train_y[indices]
        
        loss_logs = []
        
        for i in range(batch_max):
            start_i = i * batch_size
            end_i = min(start_i + batch_size, batch_total)
            x, y = train_x[start_i:end_i], train_y[start_i:end_i]
            
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
    train()

    input_size = train_config["input_size"]
    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"acc: {evaluate(model, input_size)}")
