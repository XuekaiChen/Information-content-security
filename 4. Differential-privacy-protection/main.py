import numpy as np
import math
import csv
import pandas as pd

diff_val = 3  # 隐私保护预算参数的取值
f = 1  # 敏感度


# 计算x
def cal_x(D, p):
    return (-1.0) * D * np.sign(p - 0.5) * np.log(1.0 - 2.0 * math.fabs(p - 0.5))


# 计算噪声
def cal_all_noise(rand_n):
    rand = np.random.random(rand_n)
    e = np.log(diff_val)  # 隐私保护预算参数
    D = f / e  # 噪声规模

    noise = []
    for i in range(rand.size):
        noise.append(cal_x(D, rand[i]))

    return rand, noise


if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv("data.csv", index_col=False)
    province = list(df['Province'])
    count = list(df['Count'])

    # 计算并加入噪声
    rand, noise = cal_all_noise(len(province))
    output = []
    for idx, (p, c) in enumerate(zip(province, count)):
        output.append(c+noise[idx])

    df['rand'] = rand
    df['noise'] = noise
    df['output'] = output

    print(df)
