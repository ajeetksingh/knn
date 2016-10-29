import numpy as np
import math

def euclidean(a, b):
	diff = a - b;
	dist = np.sqrt(np.sum(diff * np.transpose(diff)))
	return dist

def squaredEuclidean(a, b):
	return math.pow(dist, 2)
