import numpy as np


def euclideanDist(a, b):
	return np.linalg.norm(a - b)
#	return np.sqrt(np.sum(np.power(a - b, 2)))

def randomInit(dataSet, numClusters):
	'''
	Initialise cluster centroids randomly within the data channel range.
	
	param dataSet:     The data set used to bound the randomly generated cluster centroids
	param numClusters: Number of clusters to initialise
	return:            Randomly initialised cluster centroids
	'''
	numClusters = max(1, numClusters)
	numSamples, numChannels = dataSet.shape
	centroids = np.mat(np.zeros((numClusters, numChannels)))
	
	channelMin, channelMax = np.min(dataSet, axis=0), np.max(dataSet, axis=0)
	centroids = (channelMax - channelMin) * np.random.rand(numClusters, 1) + channelMin
	return centroids
	
def kmeans(dataSet, numClusters, distanceMeasure=euclideanDist, createCentroids=randomInit, maxNumIts=-1):
	'''
	Cluster a data set into a given number of clusters and return the cluster centres and assignments.
	
	param dataSet:         The data set which will be clustered
	param numClusters:     Number of clusters the data set is separated into
	param distanceMeasure: A callback distance function used to measure the distance between 
	                       data samples
	param createCentroids: A callback initalisation function for the cluster centroids
	param maxNumIts:       The maximum number of iterations used. The iterations continue until 
	                       convergence is achieved if the value is -1

	return: A tuple of cluster centroids, cluster assignments and number of iterations used.
	        The cluster assignments is a 2D matrix of size numSamples x 2, where the first column
	        specifies the cluster index, and the second column contains the distance to the cluster
	        centroid.
	'''
	numClusters = max(1, numClusters)
	numSamples, numChannels = dataSet.shape
	
	# Create a cluster assignment matrix to represent each sample (clusterId, distanceToCentroid)
	clusterAssignment = np.mat(np.zeros((numSamples, 2)))
	
	# Start by initialising the cluster centroids based on a given strategy
	centroids = createCentroids(dataSet, numClusters)
	
	# Run EM algorithm to iteratively find the optimal centroids for the clusters
	numIts = 0
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		
		# Expectation step - Assign each data sample to the closest cluster centroids
		for si in range(numSamples):
			dists = map(distanceMeasure, centroids, np.tile(dataSet[si, :], (numClusters, 1)))
			minIndex = np.argmin(dists)
			minDist = dists[minIndex]

			if clusterAssignment[si, 0] != minIndex: 
				clusterChanged = True
			
			clusterAssignment[si, :] = minIndex, minDist**2

		# Maximisation step - Calculate cluster centroids based on current assignments
		for clusterId in range(numClusters):
			# Extract data assigned to cluster and calculate the mean
			clusterData = dataSet[np.nonzero(clusterAssignment[:, 0].A == clusterId)[0]]
			centroids[clusterId, :] = np.mean(clusterData, axis=0)
			
		numIts += 1
		if maxNumIts != -1 and numIts >= maxNumIts:
			break

	return centroids, clusterAssignment, numIts
