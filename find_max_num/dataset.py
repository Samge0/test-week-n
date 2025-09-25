import torch
import numpy as np
    
def generate_random_date(size):
    """ 生成一个随机数据 """
    x = np.random.random(size)
    y = np.argmax(x)
    return x, y 

def generate_random_datas(total, size):
    """ 生成一个随机数据列表 """
    x = []
    y = []
    
    for _ in range(total):
        x_, y_ = generate_random_date(size)
        x.append(x_)
        y.append(y_)
    
    # 将列表转换为numpy数组，然后转换为tensor，避免性能警告
    x_array = np.array(x)
    y_array = np.array(y)
    return torch.FloatTensor(x_array), torch.LongTensor(y_array)  


if __name__ == '__main__':
    x, y = generate_random_datas(5, 5)
    zip_data = zip(x, y)
    for _x, _y in zip_data:
        print(_x, _y)