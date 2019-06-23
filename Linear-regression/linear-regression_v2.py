# Source: https://machinelearningcoban.com/2016/12/28/linearregression/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('data/Advertising.csv')







# mẫu số liệu  (x :Chiều cao, y: Cân nặng)
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
Y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# exit(0)
#
X = np.array([data.values[:, 2]]).T
Y = np.array([data.values[:, 4]]).T


one = np.ones((X.shape[0], 1))
X_train = np.concatenate((one, X), axis=1)


# Calculating weights of the fitting line
A = np.dot(X_train.T, X_train)
b = np.dot(X_train.T, Y)
weight = np.dot(np.linalg.inv(A), b)
print("Ket qua: ")
print(weight)


# Preparing the fitting line
w_0 = weight[0][0]
w_1 = weight[1][0]
print("Gia tri du doan cho X = 19 :")
print(w_1*19 + w_0)
x0 = np.linspace(0, 50, 100)
y0 = w_0 + w_1*x0


# Drawing the fitting line
plt.plot(X.T, Y.T, 'ro')     # data
plt.plot(x0, y0)               # the fitting line

plt.show()
