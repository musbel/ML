#!/usr/bin/python

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rv_discrete

class DiscreteRV:
	def __init__(self, values, probabilities):
		self.values = values
		self.probabilities = probabilities
		
		# Compute CDF
		self.sortOrder = np.argsort(self.probabilities)
		self.cdf = np.float32(self.probabilities)[self.sortOrder]
		self.cdf.sort()

		for pi in range(1, len(self.cdf)):
			self.cdf[pi] += self.cdf[pi - 1]

	def rvs(self, sampleSize=10):
		# Generate uniform samples [0, 1] of a given size
		uniformSamples = np.random.rand(sampleSize)

		# Initialise our discrete samples
		sample = []

		# Use the CDF to see which value from the discrete distribution
		# we should pick
		for uniformSample in uniformSamples:
			for ci, cumulativeProb in enumerate(self.cdf):
				if cumulativeProb >= uniformSample:
					valueIndex = self.sortOrder[ci]
					sample.append(self.values[valueIndex])
					break

		return sample


def plotDiscreteHistogram(sample):
	heights, bins = np.histogram(sample, bins=[0.5, 1.5, 2.5, 3.5, 4.5])
	print heights

	counts, bins, patches = plt.hist(sample, bins=[0.5, 1.5, 2.5, 3.5, 4.5])
	plt.xticks(x)
	plt.show()

def testSample(values, probabilities):
	rv = DiscreteRV(values, probabilities)
	sample = rv.rvs()
	
	plotDiscreteHistogram(sample)

def centralLimitTheroemTest(values, probabilities, numSampleMeans=1, sampleSize=10):
	rv = DiscreteRV(values, probabilities)

	sampleMeans = []

	# Take sample means 'numSampleMeans' times
	for i in range(numSampleMeans):
		sample = rv.rvs(sampleSize)
		
		# Calculate mean
		sampleMean = np.mean(sample)
		sampleMeans.append(sampleMean)
		sample, sampleMean

	heights, bins = np.histogram(sampleMeans, bins='auto')
	print heights

	counts, bins, patches = plt.hist(sampleMeans, bins='auto', normed=True)

	mean = np.mean(sampleMeans)
	stdev = np.std(sampleMeans)
	print('Mean: %f' % mean)
	print('Stdev: %f' % stdev)

	x = np.arange(mean - stdev * 3, mean + stdev * 3, 0.01)
	plt.plot(x, normalDist(x, mean=mean, stdev=stdev), color='red')

	plt.show()

def normalDist(x, **kwargs):
	mean = 0 if 'mean' not in kwargs else kwargs['mean']
	stdev = 1 if 'stdev' not in kwargs else kwargs['stdev']
	return np.exp(-((x - mean)**2 / (2 * stdev**2))) / math.sqrt(2 * math.pi * stdev**2)


if __name__ == "__main__":
	# Central Limit Theorem test
	# Create an obviously not normally distributed Probability Mass Function
	x = [1, 2, 3, 4]
	px = [0.4, 0.1, 0.1, 0.4]

	#testSample(x, px)
	centralLimitTheroemTest(x, px, numSampleMeans=1000, sampleSize=500)

	#sample_scipy = rv_discrete(values=(x, px)).rvs(size=10)
	#print sample_scipy

	

