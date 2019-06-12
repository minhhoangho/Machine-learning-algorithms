# Source: https://machinelearningcoban.com/2016/12/28/linearregression/

import numpy as np
import matplotlib.pyplot as plt


# mẫu số liệu  (x :Chiều cao, y: Cân nặng)
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
Y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

one = np.ones((X.shape[0], 1))
X_train = np.concatenate((one, X), axis=1)

# Calculating weights of the fitting line
A = np.dot(X_train.T, X_train)
b = np.dot(X_train.T, Y)
weight = np.dot(np.linalg.inv(A), b)
print(weight)

# Preparing the fitting line
w_0 = weight[0][0]
w_1 = weight[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0


# Drawing the fitting line
plt.plot(X.T, Y.T, 'ro')     # data
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )