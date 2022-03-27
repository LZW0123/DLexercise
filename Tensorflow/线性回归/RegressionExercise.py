import numpy as np


# 每一步的梯度下降
def step_gradient(w_current, b_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))  # 样本数
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b=2*(wx+b-y)
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        # grad_w=2*x*(wx+b-y)
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)  # 除个平均值
    # update parameter
    new_grad_w = w_current - learningRate * w_gradient
    new_grad_b = b_current - learningRate * b_gradient
    return [new_grad_w, new_grad_b]


# 完善成最终的梯度下降
def gradient_descent_runner(points, start_w, start_b, learningRate, numIterations):
    w = start_w
    b = start_b
    for i in range(numIterations):
        w, b = step_gradient(w, b, np.array(points), learningRate)
    return [w, b]


def compute_loss(points, w, b): # 计算平均误差
    N = len(points)
    loss = 0
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        loss += (w * x + b - y) ** 2
    return loss/float(N)


def main():
    points = np.genfromtxt("data.csv", delimiter=",")  # CSV文件常以','作为分隔符
    LearnRate = 0.0001
    Initial_w = 0
    Initial_b = 0
    print('首次的loss', compute_loss(points, Initial_w, Initial_b))
    [w, b] = gradient_descent_runner(points, Initial_w, Initial_b, LearnRate, 10000)
    print('最后的loss', compute_loss(points, w, b))


if __name__ == '__main__':
    main()
