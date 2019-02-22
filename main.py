#!/usr/bin/python3
# O(n)

from mnist import MNIST
from scipy.misc import toimage
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        index = 0

        for row in X_test:
            print('KNN predicting at index ' + str(index))

            label = self.closest(row)
            predictions.append(label)

            index += 1

        return predictions

    def closest(self, row):
        best_dist = self.euc(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            dist = self.euc(row, self.X_train[i])

            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]

    def euc(self, a, b):
        return distance.euclidean(a, b)

print('Getting MNIST data...')
mndata = MNIST('./mnist')

clf = ScrappyKNN()

train_number = 60000
test_number = 100

print('(' + str(train_number) + '):')

print('Retrieving training data...')
X_train, y_train = mndata.load_training()

X_train = X_train[0:train_number]
y_train = y_train.tolist()[0:train_number]

print('Retrieving testing data...')
X_test, y_test = mndata.load_testing()

X_test = X_test[0:test_number]
y_test = y_test.tolist()[0:test_number]

print('Learning...')
clf.fit(X_train, y_train)

print('Predicting...')
predictions = clf.predict(X_test)

print("-\nPredictions:")
print(predictions)

accuracy = accuracy_score(y_test, predictions)

print('Accuracy: ' + str(accuracy))