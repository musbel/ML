#!/usr/bin/python

'''
Testing softmax and cross entropy calculations using a simple linear model placeholder
'''

import numpy as np

class LinearModel:
	def __init__(self):
		self.W = np.float32([])
		self.bias = np.float32([])

	def predict(X):
		'''
		Prediction for input based on learned weights and bias.
		
		:param X: Input data
		
		''' 
		return np.float32([3.0, 1.0, 0.2]) # Test predefined dummy values
		return np.dot(self.W, X) + self.b

def softmax(X):
	'''
	Compute softmax values for each sets of scores in x.
	
	:param X: Predicted values
	'''
	if X is None: return None
	e_x = np.exp(X)
	return e_x / np.sum(e_x, axis=0)

def cross_entropy(S, L):
	'''
	Compute cross entropy for given softmax values and labels.
	
	:param S: Softmax values
	:param L: Labels (hot labels)
	'''
	return -np.sum(np.dot(L, np.log(S)))


if __name__ == "__main__":
	model = LinearModel()
	scores = model.predict()

	print(softmax(scores))

	# Plot softmax curves
	import matplotlib.pyplot as plt
	x = np.arange(-2.0, 6.0, 0.1)
	rangeScores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

	scores_softmax = softmax(rangeScores)
	if scores_softmax is not None:
		plt.plot(x, softmax(rangeScores).T, linewidth=2)
		plt.show()

	print
	print '-- Multinomial Logistic Classification (MLC) --'
	print 'Scores (logits):', scores
	print 'Softmax:', softmax(scores)
	print 'Cross entropy:', cross_entropy(softmax(scores), np.float32([[1, 0, 0]]))

	#print 'Low entropy:', cross_entropy(np.float32([[0.99, 0.001, 0.001]]).T, np.float32([[1, 0, 0]]))
	

