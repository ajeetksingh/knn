
# generic modules
import os
import sys

# module to load Yan LeCun's MNIST data
from mnist import MNIST

# custom functions
from distance import euclidean, squaredEuclidean

# NUMPY
import numpy as np

#loading mnist data
digits_data = MNIST('/home/ajeet/data')

X_train, y_train = digits_data.load_training()
X_test, y_test = digits_data.load_testing()

# converting list to nparray
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# k: value in kNN
k = int(sys.argv[1])

correct_pred = 0;
for i in xrange(0, 10000): 
	query = X_test[i]
	distance = []
	for j in xrange(0, 60000):
		# euclidean_distance
		dist = euclidean(X_train[j], query)
		distance.append(dist)
	
	# finding the nearest neighbor
	# sort the distance matrix
	distance = np.array(distance)
	indices = np.argsort(distance)

	topk = indices[:k]
	pred_labels = y_train[topk];
	found = np.where(pred_labels == y_test)
	if len(found):
		correct_pred = correct_pred + 1
        print "Accuracy after "+ str(i + 1) + " examples: " + str((correct_pred/(i+1)) * 100) + "%"
accuracy = correct_pred/100
print "Final Accuracy: " + str(accuracy)
