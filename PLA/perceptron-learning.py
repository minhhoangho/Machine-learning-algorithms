# source: https://machinelearningcoban.com/2017/01/21/perceptron/

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(2)

def display(X, label):
    X0 = X[0, :]
    X1 = X[label == 1, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()






means = [[2,2], [4,2]]
cov = [[0.3, 0.2] , [0.2, 0.3]]

N = 10

X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T


X = np.concatenate((X0, X1), axis=1)
y = y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)

# Xbar
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

# tính đầu ra khi biết đầu vào x và weights w.
def h(w, x):
    return np.sign(np.dot(w.T, x))

# kiểm tra xem thuật toán đã hội tụ chưa.
# Ta chỉ cần so sánh h(w, X) với ground truth y.
# Nếu giống nhau thì dừng thuật toán.
def has_converged(X, y ,w):
    return np.array_equal(h(w, X), y)

# hàm chính thực hiện PLA
def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        # mix data
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi:  # misclassified point
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)

        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)


d = X.shape[0]

w_init = np.random.rand(d, 1)
w, m = perceptron(X, y, w_init)


from matplotlib.animation import FuncAnimation



def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2], 'k')
    else:
        x10 = -w0/w1
        return plt.plot([x10, x10], [-100, 100], 'k')

def viz_alg_1d_2(w):
    it = len(w)
    fig, ax = plt.subplots(figsize=(5, 5))

    def update(i):
        ani = plt.cla()
        # points
        ani = plt.plot(X0[0, :], X0[1, :], 'b^', markersize=8, alpha=.8)
        ani = plt.plot(X1[0, :], X1[1, :], 'ro', markersize=8, alpha=.8)
        ani = plt.axis([0, 6, -2, 4])
        i2 = i if i < it else it - 1
        ani = draw_line(w[i2])
        if i < it - 1:
            # draw one  misclassified point
            circle = plt.Circle((X[1, m[i]], X[2, m[i]]), 0.15, color='k', fill=False)
            ax.add_artist(circle)
        # hide axis
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        label = 'PLA: iter %d/%d' % (i2, it - 1)
        ax.set_xlabel(label)
        return ani, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, it + 2), interval=1000)
    # save
    # anim.save('pla_vis.gif', dpi=100, writer='imagemagick')
    plt.show()


viz_alg_1d_2(w)