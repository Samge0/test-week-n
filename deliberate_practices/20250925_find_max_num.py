#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
练习：找出一个数组中的最大值

流程：
构建模型
用torch创建随机的训练集、测试集，输出指定长度的数组+对应的最大值下标
构建训练函数
构建测试函数
训练模型
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 当前文件目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型保存路径
MODEL_SAVE_PATH = f"{CURRENT_DIR}/models_ouput/model_20250925.pth"

# 构建模型
class Net(nn.Module):
    def __init__(self, size):
        super(Net, self).__init__()
        self.liner = nn.Linear(size, size)
        self.loss = nn.functional.cross_entropy


    def forward(self, x, y=None):
        x = self.liner(x)
        return x if y is None else self.loss(x, y)

def generate_datas(size, total):
    """ 构建数据集 """
    x, y = [], []
    for _ in range(total):
        _x = np.random.random(size)
        x.append(_x)
        y.append(np.argmax(_x))
    return torch.FloatTensor(np.array(x)), torch.LongTensor(np.array(y))

def evaluate(model, size, total):
    """ 评估模型准确率 """
    model.eval()
    
    # 创建测试集
    _x, _y = generate_datas(size, total)
    
    # 模型预测
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(_x)
        for y_p, y_t in zip(y_pred, _y):
            if int(np.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
                
    correct_rate = correct / (correct + wrong)
    print(f"Evaluate total: {len(_y)}, correct: {correct}, wrong: {wrong}, Accuracy: {correct_rate}")
    
    return correct_rate
    
def train():
    """ 训练函数 """
    # 定义训练参数
    epoch_num = 200 # 训练轮数
    batch_size = 20 # 批次大小
    train_total = 1000 # 每轮总共训练的样本总数
    input_size = 5 # 输入向量的维度
    learning_rate = 0.001 # 学习率
    
    # 构建模型
    model = Net(input_size)
    
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 收集训练日志
    train_logs = []
    
    # 创建训练集
    train_x_lst, train_y_lst = generate_datas(input_size, train_total)
    
    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_logs = []
        
        # 每轮训练开始前，打乱数据，增加robust
        indices = torch.randperm(train_total)
        train_x_lst = train_x_lst[indices]
        train_y_lst = train_y_lst[indices]
        
        # 计算每轮的最大训练批次
        max_batch = math.ceil(train_total / batch_size)
        
        for batch_index in range(max_batch):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, train_total)
            
            _x = train_x_lst[start_index: end_index]
            _y = train_y_lst[start_index: end_index]

            optim.zero_grad()
            loss = model(_x, _y)
            loss.backward()
            optim.step()
            
            watch_logs.append(loss.item())
            
        # 打印每轮的训练情况
        print(f"[{epoch}/{epoch_num}] loss: {np.mean(watch_logs)}")
        
        # 测试本轮训练效果
        acc = evaluate(model, input_size, 500)
        
        # 记录训练日志
        train_logs.append([acc, float(np.mean(watch_logs))])
        
        # 每次训练都自动保存模型
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
    # 训练完毕后，绘制训练日志图
    _plt_x = range(len(train_logs))
    plt.plot(_plt_x, [l[0] for l in train_logs], label="acc")
    plt.plot(_plt_x, [l[1] for l in train_logs], label="loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    # 测试生成数据集
    input_size = 5
    x, y = generate_datas(input_size, 1)
    zip_data = zip(x, y)
    for _x, _y in zip_data:
        print(_x, _y)
        
    # 训练
    train()
    
    # # 使用已经训练好的模型进行预测
    # model = Net(input_size)
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    # evaluate(model, 5, 1000)