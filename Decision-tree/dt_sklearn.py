import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

data_frame = pd.read_csv('data/kyphosis.csv')

X = data_frame.drop('Kyphosis', axis=1)
y = data_frame['Kyphosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)
predictions =  dtree.predict(X_test)

# print(predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))