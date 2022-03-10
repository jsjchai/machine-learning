import numpy as np


def mean_squared_error(y, t):
    '''
    : 均方误差
    :param y: 神经网络的输出
    :param t: 监督数据
    :return: 均方误差值
    '''
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    '''
    ：交叉熵误差
    :param y: 神经网络的输出
    :param t: 监督数据
    :return: 交叉熵误差值
    '''
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def mini_batch_cross_entropy_error(y, t):
    '''
    ：mini-batch版交叉熵误差的实现
    :param y: 神经网络的输出
    :param t: 监督数据
    :return: 误差值
    '''
    ## 维度
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print("mean_squared_error:", mean_squared_error(np.array(y), np.array(t)))
print("cross_entropy_error:", cross_entropy_error(np.array(y), np.array(t)))
print("cross_entropy_error:", mini_batch_cross_entropy_error(np.array(y), np.array(t)))
