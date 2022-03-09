import numpy as np


def AND(x1, x2):
    '''
    :param x1: x1
    :param x2: x2
    :return: 与门
    '''

    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def NAND(x1, x2):
    '''
    :param x1: x1
    :param x2: x2
    :return: 与非门
    '''
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 仅权重和偏置与AND不同！
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    '''
      :param x1: x1
      :param x2: x2
      :return: 或门
    '''
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 仅权重和偏置与AND不同！
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    '''
    :param x1: x1
    :param x2: x2
    :return: 异或门
    '''
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print("AND:", AND(1, 2))
print("NAND:", NAND(1, 2))
print("OR:", OR(1, 2))
print("XOR:", XOR(1, 2))
