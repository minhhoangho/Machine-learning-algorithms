import pandas as pd
import numpy as np

data = pd.read_csv('data/iris.csv')

labels = list(set(data.values[:, 4]))
data = data.sample(frac=1)
X_train = data.values[:100, :]
Y_test = data.values[101:, :]
class_1 = X_train[[i for i in range(X_train.shape[0]) if X_train[i][4] == labels[0]], :]
class_2 = X_train[[i for i in range(X_train.shape[0]) if X_train[i][4] == labels[1]], :]
class_3 = X_train[[i for i in range(X_train.shape[0]) if X_train[i][4] == labels[2]], :]
print("Labels:")
print(labels)
print("Train set: ", X_train.shape)
print("Test set: ", Y_test.shape)


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += (instance1[i] - instance2[i]) ** 2
    return np.sqrt(distance)


def get_neighbors(trainning_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for i in range(len(trainning_set)):
        dis = euclidean_distance(trainning_set[i], test_instance, length)
        distances.append((trainning_set[i], dis))

    distances = sorted(distances, key=lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def response(neighbors):
    class_votes = {}

    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)

    return sorted_votes[0][0]


def check_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] is predictions[i]:
            correct += 1
    return (correct / float(len(test_set)))


if __name__ == "__main__":
    # prepare data
    predictions = []
    k = 5
    for i in range(len(Y_test)):
        neighbours = get_neighbors(trainning_set=X_train, test_instance=Y_test[i], k=k)
        result = response(neighbours)
        predictions.append(result)

        print('> Predicted=%7s   -----  Actual=%s' % (predictions[i], Y_test[i][-1]))
    accuracy = check_accuracy(test_set=Y_test, predictions=predictions)
    print("Accuracy= %7f " %accuracy )
