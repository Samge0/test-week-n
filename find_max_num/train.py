import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

import models
import dataset

# 当前文件目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型输出目录
MODEL_DIR = os.path.join(CURRENT_DIR, "models_output")
os.makedirs(MODEL_DIR, exist_ok=True)

# best模型保存路径
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model_best.pt")

# last模型保存路径
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "model_last.pt")

def evaluate(model):
    """ 评估模型准确率 """
    model.eval()
    test_sample_num = 500
    x, y = dataset.generate_random_datas(test_sample_num, 5)
    
    print("本次预测集中共有%d个样本" % len(x))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
                
    correct_rate = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, correct_rate))
    return correct_rate

def train():
    """ 主训练函数 """
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_total = 1000  # 每轮训练总共训练的样本总数
    
    input_size = 5  # 输入向量维度
    
    learning_rate = 0.001  # 学习率
    
    save_epoch_interval = 50  # 经过多少轮自动保存模型
    
    # 建立模型
    model = models.FindMaxNumModel(input_size)
    
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 收集训练日志
    train_logs = []
    
    # 创建模拟训练集
    train_x, train_y = dataset.generate_random_datas(train_total, input_size)
    
    # 当前最好的精度
    best_acc = 0
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        # 在每个epoch开始时打乱数据，增加robust
        indices = torch.randperm(train_total)
        train_x = train_x[indices]
        train_y = train_y[indices]
        
        # 最大批次
        max_batch = math.ceil(train_total / batch_size)
        
        for batch_index in range(max_batch):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, train_total)  # 防止越界
            
            x = train_x[start_index: end_index]
            y = train_y[start_index: end_index]
            
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            
            watch_loss.append(loss.item())
            
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model)  # 测试本轮模型结果
        train_logs.append([acc, float(np.mean(watch_loss))])
        
        # 保存准确度最高的模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        
        # 自动保存指定轮数迭代的模型
        if save_epoch_interval > 0 and (epoch + 1) % save_epoch_interval == 0:
            save_path = os.path.join(MODEL_DIR, "model_%d.pt" % (epoch + 1))
            torch.save(model.state_dict(), save_path)
        
    # 保存模型    
    torch.save(model.state_dict(), LAST_MODEL_PATH)
    
    # hui绘曲线
    plt.plot(range(len(train_logs)), [l[0] for l in train_logs], label="acc")
    plt.plot(range(len(train_logs)), [l[1] for l in train_logs], label="loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train()
    