import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter
import hashlib


class Knn:
    def __init__(self, k, dist):
        self.X_train = None
        self.y_train = None
        self.k = k
        self.dist = dist

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def __get_neighbors(self, k, test_instance):
        distances = []
        for indexx in range(self.X_train.shape[0]):
            if self.dist == "euclidean":
                dist = self.__euclidean_distance(test_instance, self.X_train.iloc[indexx])
            elif self.dist == "manhattan":
                dist = self.__manhattan_distance(test_instance, self.X_train.iloc[indexx])
            elif self.dist == "hamming":
                dist = self.__hamming_distance(test_instance, self.X_train.iloc[indexx])
            else:
                raise ValueError("Incorrect dist parameter passed")

            distances.append((self.X_train.iloc[indexx], dist, self.y_train.iloc[indexx]))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:k]
        return neighbors

    def __vote(self, neighbors):
        class_counter = Counter()
        for neighbor in neighbors:
            class_counter[neighbor[2]] += 1
        return class_counter.most_common(1)[0][0]

    def predict(self, x_test):
        predicted = []
        for index in range(len(x_test)):
            flower = X_test.iloc[index]
            neighbours = knn.__get_neighbors(self.k, flower)
            predicted.append(knn.__vote(neighbours))
        return predicted

    @staticmethod
    def __euclidean_distance(point_a, point_b):
        return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    @staticmethod
    def __manhattan_distance(point_a, point_b):
        return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])

    @staticmethod
    def __hamming_distance(point_a, point_b):
        chaine1 = hashlib.md5(str(point_a).encode()).hexdigest()
        chaine2 = hashlib.md5(str(point_b).encode()).hexdigest()
        return len(list(filter(lambda x: ord(x[0]) ^ ord(x[1]), zip(chaine1, chaine2))))


iris = datasets.load_iris()
training = iris["data"]
training_labels = iris["target"]

training_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['target'])

labels_df = training_df["target"]
training_df.drop("target", axis=1, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(training_df, labels_df, test_size=0.25, random_state=42)

knn = Knn(5, dist="hamming")
knn.train(X_train, Y_train)
predicted = knn.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, predicted))
