# Example of making predictions
from math import sqrt
import csv 
import numpy as np
from stopwatch import Stopwatch

#starting stopwatch
stopwatch = Stopwatch()
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
#grabs data from the file
new_dataset = np.genfromtxt('data.csv', delimiter = ',')	
 
 
# Test distance function
dataset = [[1,2,"+"],
	[3,4,"+"],
	[6,4,"+"],
	[2,1,"-"],
	[4,1,"-"],
	[5,2,"-"],
	]


prediction1 = predict_classification(dataset, new_dataset[0], 1)
print('e7  Got %s.' % ( prediction1))
prediction2 = predict_classification(dataset, new_dataset[1], 1)
print('e8 Got %s.' % ( prediction2))

stopwatch.stop()

print('Duration of the calculation : %s' % (str(stopwatch)) )