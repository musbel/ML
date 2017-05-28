#!/usr/bin/python

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import numpy as np
import math

def createPlot(title='Untitled', labels=[], funcs=[], intervals=[[0.0, 1.0]], **kwargs):
	if not funcs or len(labels) != len(funcs): return
	
	ts = []
	for interval in intervals:
		ts.append(np.arange(interval[0], interval[1], 0.01))

	fig = plt.figure('Figure: %s' % title)

	for label, func in zip(labels, funcs):
		for t in ts:
			plt.plot(t, func(t, **kwargs), label=label)

	plt.grid(True)	
	plt.legend()

	plt.axhline(0, color='black')
	plt.axvline(0, color='black')

def pseudoErf(x):
	return (2.0 / math.pi**0.5) * (x - pow(x, 3)/3 + pow(x, 5)/10 - pow(x, 7)/42 + pow(x, 9)/216)

def standardNormalCDF(x, **kwargs):
	return 1.0 - 0.5 * np.exp(-1.2 * pow(x, 1.3))

def standardNormalCDF2(x, **kwargs):
	return [(1.0 + math.erf(k / math.sqrt(2.0))) / 2.0 for k in x]

def standardNormalCDF3(x, **kwargs):
	return [(1.0 + pseudoErf(k / math.sqrt(2.0))) / 2.0 for k in x]

def standardNormalCDF4(x, **kwargs):
	return [CDF(k) for k in x]

def CDF(x):
	s = value = float(x)
	
	# Note: Has to start from 1 as x is already used in value
	#       as the first element
	for n in range(1, 100):
		value = value * x * x / (2 * n + 1)
		s += value
	
	return 0.5 + (s / math.sqrt(2.0 * math.pi)) * np.exp(-(x * x) / 2.0)

def standardNormalDist(x, **kwargs):
	return np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)	

def normalDist(x, **kwargs):
	mean = 0 if 'mean' not in kwargs else kwargs['mean']
	stdev = 1 if 'stdev' not in kwargs else kwargs['stdev']
	return np.exp(-((x - mean)**2 / (2 * stdev**2))) / math.sqrt(2 * math.pi * stdev**2)

def calculateProbability(start, end):
	v = standardNormalCDF2([start, end])
	return v[1] - v[0]

def calculateProbabilityDiscrete(x, delta=0.5):
	v = standardNormalCDF2([x - delta, x + delta])
	return v[1] - v[0]

def normalise(x, mean, stdev):
	return (x - mean) / stdev

if __name__ == "__main__":

	print calculateProbability(-1, 1)
	print calculateProbability(-2, 2)
	print calculateProbability(-3, 3)
	print calculateProbabilityDiscrete(1)
	print calculateProbabilityDiscrete(1, delta=0)
	
	print CDF(1), CDF(-1)
	print CDF(1) - CDF(-1)
	
	m, s = -5, 10
	print CDF(normalise(5, m, s)) - CDF(normalise(-15, m, s))

#	createPlot('N1', ['Standard Normal'], [standardNormalDist], intervals=[[-4, 4]])
#	createPlot('N1', ['Standard Normal (2)'], [normalDist], intervals=[[-4, 4]])
#	createPlot('N1', ['Standard Normal (2)'], [normalDist], intervals=[[-25, 25]], mean=-5.0, stdev=10.0)
	createPlot('N3', ['Standard Normal CDF'], [standardNormalCDF2], intervals=[[-4, 4]])
	createPlot('N3', ['Standard Normal CDF (IP)'], [standardNormalCDF4], intervals=[[-4, 4]])

	plt.show()