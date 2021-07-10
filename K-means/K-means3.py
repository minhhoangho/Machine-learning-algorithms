import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Euclidian: (x1, y1) , (x2, y2) => (x1-x2) **2  + (y1-y2)**2
#
# Manhattan: (x1, y1) , (x2, y2) => abs(x1-x2) + abs(y1-y2)
#
# trọng số: ...
#
# Minkowski: ...




def load_dataset(path):
    data = pd.read_csv(path, index_col=None)
    data = data.sample(frac=1)
    labels = list(set(data.values[:, 5]))
    X_train = data.values[:100, 1:3]
    X_temp = data.values[:100, -1].reshape((-1,1))
    X_train = np.concatenate((X_train, X_temp), axis=1)

    X_test = data.values[101:, 1:3]
    X_temp = data.values[101:, -1].reshape((-1, 1))
    X_test = np.concatenate((X_test, X_temp), axis=1)
    X_test = X_test[:2, :]
    # print(X_test)
    # exit()


    class_1 = X_train[[i for i in range(X_train.shape[0]) if X_train[i][-1] == labels[0]], :]

    class_2 = X_train[[i for i in range(X_train.shape[0]) if X_train[i][-1] == labels[1]], :]
    class_3 = X_train[[i for i in range(X_train.shape[0]) if X_train[i][-1] == labels[2]], :]

    plt.scatter(class_1[:, 0], class_1[:, 1], c='r', label=class_1[0,-1])
    plt.scatter(class_2[:, 0], class_2[:, 1], c='b', label=class_2[0,-1])
    plt.scatter(class_3[:, 0], class_3[:, 1], c='g', label=class_3[0,-1])
    plt.legend()
    plt.show()
    print(class_1)
    exit(0)
    print("Labels:")
    print(labels)
    print("Train set: ", X_train.shape)
    print("Test set: ", X_test.shape)

    return X_train, X_test


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        # instance1[i] - instance2[i] là các thuộc tính của dữ liệu
        distance += (instance1[i] - instance2[i]) ** 2
    return np.sqrt(distance)


def get_neighbors(trainning_set, test_instance, k):
    distances = []
    print(test_instance)
    length = len(test_instance) - 1 # phải trừ 1 do label
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
    print("OK")
    X_train, X_test = load_dataset('./data/Iris.csv')
    predictions = []
    k = 5
    for i in range(len(X_test)):
        neighbours = get_neighbors(trainning_set=X_train, test_instance=X_test[i], k=k)
        result = response(neighbours)
        predictions.append(result)
        print('> Predicted=%7s   -----  Actual=%s' % (predictions[i], X_test[i][-1]))
    accuracy = check_accuracy(test_set=X_test, predictions=predictions)
    print("Accuracy= %7f " % accuracy)

