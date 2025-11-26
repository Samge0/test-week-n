#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, os, time, torch, swanlab
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

CURRENT_DATE = "20251126"

# 当前目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型路径
MODEL_PATH = os.path.join(CURRENT_DIR, f"models_output/model_{CURRENT_DATE}.pth")

train_config = {
    "input_size": 26,
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
    
def generate_data(total, input_size):
    x = np.random.random((total, input_size))
    y = np.argmax(x, axis=1)
    return torch.FloatTensor(x), torch.LongTensor(y)

def evaluate(model, input_size, total=500):
    model.eval()
    x, y = generate_data(total, input_size)
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
        experiment_name=f"find-max-num-{CURRENT_DATE}-{int(time.time())}",
        tags=["find-max-num", "test", CURRENT_DATE]
    )
    
    train_x, train_y = generate_data(batch_total, input_size)
    model = Net(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
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

