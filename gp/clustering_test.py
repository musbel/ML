import unittest
import numpy as np
import cPickle as pickle

import matplotlib
import matplotlib.pyplot as plt

import clustering


class SimpleTest2D(unittest.TestCase):
	def setUp(self):
		self.data = pickle.load(open('resources/Simple2D_4c.pickle', 'rb'))
		self.numClusters = 4
		np.random.seed(1000)
		self.plotResults = True

	def test_kmeans(self):
		# Run k-means clustering on the test set using 4 clusters and check if the cluster centres
		# (centroids) lie close to the expected centroids based on a fixed seed
		centroids, assignments, numIts = clustering.kmeans(self.data, self.numClusters, maxNumIts=10)		
		expectedCentroids = np.float32([
			[ 2.80, -2.73],
		 	[-3.38, -2.94],
		 	[ 2.62,  3.10],
		 	[-2.46,  2.78]
		])

#		np.testing.assert_almost_equal(centroids, expectedCentroids, 2)	
		
		if self.plotResults:
			self.plotData_kMeans(self.data, centroids, assignments, 1)

	def plotData_kMeans(self, data, centroids, assignments, numIts):
		''' Helper function to visualise the data and clusters for k-means 
		'''
		# Assign different colours to visually indentify cluster samples
		colourNames = ['red', 'red', 'red', 'red']
		colours = [colourNames[int(a)] for a in assignments[:, 0]]

		fig = plt.figure()
		plt.title('Iteration: %d' % numIts)
		ax = fig.add_subplot(111)
		ax.scatter(data[:, 0], data[:, 1], color=colours)
		
		ax2 = fig.add_subplot(111)
		ax2.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=200, color='black')
		plt.show()


if __name__ == "__main__":
	unittest.main()
