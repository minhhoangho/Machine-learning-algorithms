import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target
print('Number of classes: %d' % len(np.unique(iris_y)))
print('Number of data points: %d' % len(iris_y))

X0 = iris_X[iris_y == 0, :]
print('\nSamples from class 0:\n', X0[:5, :])

X1 = iris_X[iris_y == 1, :]
print('\nSamples from class 1:\n', X1[:5, :])

X2 = iris_X[iris_y == 2, :]
print('\nSamples from class 2:\n', X2[:5, :])

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

print("Training size: %d" % len(y_train))
print("Test size: %d" % len(y_test))


clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print("Result:")
print("Predicted: ", y_predict)
print("Actual value: ", y_test)