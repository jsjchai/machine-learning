import numpy as np


# f(x0,x1)=x0*x0+x1*x1
def function_2(x):
    return x[0] ** 2 + x[1] ** 2


# 梯度
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值

    return grad


# 梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


init_x = np.array([-3.0, 4.0])
print("梯度", numerical_gradient(function_2, init_x))
print("梯度下降", gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
print("梯度下降,lr=10.0", gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
print("梯度下降,lr=1e-10", gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
