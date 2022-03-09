import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    '''
    : 阶跃函数
    :param x: 实数（浮点数） 数组
    :return: x>0 1 x<= 0
    '''
    y = x > 0
    return y.astype(np.int32)


def sigmoid(x):
    '''
    : 逻辑函数 S函数
    :param x: 实数（浮点数） 数组
    :return:  逻辑函数值
    '''
    return 1 / (1 + np.exp(-x))


def relu(x):
    '''
    : 线性整流函数
    :param x: 实数（浮点数） 数组
    :return:  x > 0 x  x<=0 0
    '''
    return np.maximum(0, x)


print("step_function:", step_function(np.array([-1.0, 2.0])))
print("sigmoid:", sigmoid(np.array([-1.0, 2.0])))
print("relu:", relu(np.array([-1.0, 2.0])))

# 阶跃函数的图形
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()

# sigmoid函数的图形
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()


# relu函数的图形
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()
