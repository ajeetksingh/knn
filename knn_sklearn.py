import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from mnist import MNIST

mndata = MNIST('/home/ajeet/data');
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)

acc = accuracy_score(y_test, predicted)
print "Accuracy: " + str(acc)

accr = knn.score(X_test, y_test)
print "Scoring Accuracy: " + str(accr)
