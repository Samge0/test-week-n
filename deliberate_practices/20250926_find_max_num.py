#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

"""
练习：找出一个数据中的最大值下标

流程：
构建模型
构建生成数据集的函数
构建训练函数
构建测试函数
运行
"""

# 当前文件夹路径
CURRENT_PATH = os.path.dirname(__file__)

# 模型保存路径
MODEL_PATH = f"{CURRENT_PATH}/models_output/model_20250926.pth"

# 构建模型
class Net(nn.Module):
    
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.loss = nn.functional.cross_entropy
        
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is None:
            return x
        else:
            return self.loss(x, y)
    
# 构建生成数据集的函数
def generate_datas(input_size, total):
    """ 构建数据集 """
    x, y = [], []
    for _ in range(total):
        _x = np.random.random(input_size)
        _y = np.argmax(_x)
        x.append(_x)
        y.append(_y)
        
    x = np.array(x)
    y = np.array(y)
    return torch.FloatTensor(x), torch.LongTensor(y)


# 评估模型的函数
def evaluate(model, input_size, total=500):
    """ 评估模型准确率 """
    model.eval()
    
    # 生成随机的测试集
    x, y = generate_datas(input_size, total)
    
    # 预测
    y_pred = model(x)
    
    # 收集预测结果
    correct, wrong = 0, 0
    for y_p, y_t in zip(y_pred, y):
        if int(torch.argmax(y_p)) == y_t: 
            correct += 1
        else:
            wrong += 1
        
    # 打印准确率
    correct_rate = correct / (correct + wrong)
    print(f"evaluate total: {total}, correct: {correct}, wrong: {wrong}, correct rate: {correct_rate}")
    return correct_rate



def train():
    """ 训练函数 """
    epoch_total = 20 # 训练总轮数
    batch_size = 20 # 每批次训练的样本个数
    batch_total = 1000 # 每轮训练的总样本个数
    input_size = 5 # 输入向量的维度
    learn_rate = 0.01 # 学习率
    # 最大批次
    max_batch = math.ceil(batch_total / batch_size) 
    
    # 创建模型
    model = Net(input_size)
    
    # 创建优化器
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    # 创建训练数据集
    train_x_lst, train_y_lst = generate_datas(input_size, batch_total)
    
    # 创建日志记录对象
    train_logs = []
    
    for epoch in range(epoch_total):
        model.train()
        
        # 每轮都打乱训练数据集
        indices = torch.randperm(batch_total)
        train_x_lst = train_x_lst[indices]
        train_y_lst = train_y_lst[indices]
        
        # 每轮的损失情况
        watch_logs = []
        
        for batch_index in range(max_batch):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, batch_total)
            
            x = train_x_lst[start_index: end_index]
            y = train_y_lst[start_index: end_index]
            
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            
            # 收集损失情况
            watch_logs.append(loss.item())
        
        # 一轮训练结束，打印loss情况
        mean_loss = np.mean(watch_logs)
        print(f"Epoch: {epoch}, Loss: {mean_loss}")
        
        # 测试一轮训练后的准确率
        acc = evaluate(model, input_size)
        print(f"[{epoch+1}/{epoch_total}] Accuracy: {acc}")
        
        # 记录训练日志
        train_logs.append([acc, mean_loss])
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        
    # 所有训练结束，绘制训练日志图
    log_x_lst = range(len(train_logs))
    plt.plot(log_x_lst, [l[0] for l in train_logs], label="Accuracy")
    plt.plot(log_x_lst, [l[1] for l in train_logs], label="Loss")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    
    input_size = 5
    
    # 测试数据集生成
    x, y = generate_datas(input_size, 1)
    for _x, _y in zip(x, y):
        print(_x, _y)
        
    # 训练
    train()
    
    # 使用训练好的模型预测
    # model = Net(input_size)
    # model.load_state_dict(torch.load(MODEL_PATH))
    # evaluate(model, input_size)