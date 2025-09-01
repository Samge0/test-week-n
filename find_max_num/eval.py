import torch
import train
import models
import dataset

def parse_eval():
    """ 评估模型准确率 """
    
    # 创建模型
    input_size = 5  # 输入向量维度
    model = models.FindMaxNumModel(input_size)
    model.load_state_dict(torch.load(train.BEST_MODEL_PATH))
    
    # 创建测试数据
    x, y = dataset.generate_random_datas(10, input_size)
    
    print("本次预测集中共有%d个样本" % len(x))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for current_x, y_p, y_t in zip(x, y_pred, y):  # 与真实标签进行对比
            y_p_index = int(torch.argmax(y_p))
            is_correct = y_p_index == int(y_t)
            if is_correct:
                correct += 1
            else:
                wrong += 1
                
            # 打印预测结果
            print(f"【{is_correct}】{current_x} => 预测下标：{y_p_index}, 正确下标：{y_t}")
                
    correct_rate = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, correct_rate))
    

if __name__ == '__main__':
    parse_eval()
    