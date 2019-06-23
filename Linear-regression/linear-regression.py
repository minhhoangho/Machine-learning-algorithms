import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('data/Advertising.csv')


#mấu số liệu
X = data.values[:, 2]
Y = data.values[:, 4]

#
plt.scatter(X, Y, marker='o')
# plt.show();
# exit(0)

def predict(new_radio, weight, bias):
    return weight * new_radio + bias


def cost_function(X, Y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (Y[i] - X[i] * weight + bias) ** 2
    return sum_error


def update_weight_bias(X, Y, weight, bias, learning_rate):
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2 * X[i] * (Y[i] - (weight * X[i] + bias))
        bias_temp += -2 * (Y[i] - (weight * X[i] + bias))
    weight -= (weight_temp / n) * learning_rate
    bias -= (bias_temp / n) * learning_rate
    return weight, bias


def train(X, Y, weight, bias, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        weight, bias = update_weight_bias(X, Y, weight, bias, learning_rate)
        cost = cost_function(X, Y, weight, bias)
        cost_history.append(cost)
    return weight, bias, cost_history


weight, bias, cost = train(X, Y, 4, 0.4, 0.0001, 100)
x_hypothesis = np.linspace(0, 50, 100)
y_hypothesis = weight * x_hypothesis +bias

plt.plot(x_hypothesis, y_hypothesis, '-r' )
print("Ket qua: ")
print(weight, bias)


print("Gia tri du doan cho X = 19 :")

print(predict(19, weight, bias))
plt.show()