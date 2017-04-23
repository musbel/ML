#!/usr/bin/python

'''
Multinomial Logistic Classification
Testing softmax and cross entropy calculations using a simple linear model placeholder.
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
		:return:  Scores (logits) based on the weights, input data and bias
		''' 
		return np.float32([3.0, 1.0, 0.2]) # Test predefined dummy values
		return np.dot(self.W, X) + self.b

def softmax(X):
	'''
	Compute softmax values for each sets of scores in x.
	
	:param X: Predicted scores (logits)
	:return:  Probability values [0..1] for the input scores
	'''
	if X is None: return None
	e_x = np.exp(X)
	return e_x / np.sum(e_x, axis=0)

def cross_entropy(S, L):
	'''
	Compute cross entropy for given softmax values and labels.
	It essentially computes the distance between the probabilities
	and a given label assignment, whereby a smaller distance
	indicates a small entropy for the label.

	Representation: D(S(wx + b), L)
	
	:param S: Softmax values
	:param L: Labels (one-hot encoding)
	:return:  Entropy for given softmax values and labels (distance between them)
	'''
	return -np.sum(np.dot(L, np.log(S)))

# TODO: Test calculating the Average Cross-Entropy by minimising the following
#       loss function using gradient descent:
#
#       L = (1/N) * SUM(D(S(w * x_i + b), L_i)
#
#       Do this by implementing a rudimentary 'fit' function in the dummy linear model
#


if __name__ == "__main__":
	# Create a dummy linear model and pretend we have predicted some score values (logits)
	model = LinearModel()
	scores = model.predict()

	# Calculate the softmax probabilities for the scores and the cross entropy given one-hot labels
	labels = np.identity(scores.shape[0])
	print '-- Multinomial Logistic Classification (MLC) --'
	print 'Scores (logits):', scores
	print 'Softmax:', softmax(scores)
	entropies = map(cross_entropy, softmax(scores), labels)
	for entropy, label in zip(entropies, labels):
		print('Cross entropy %s: %.3f' % (label, entropy))
	
	print('Lowest entropy for label: %s' % labels[np.argmin(entropies)])

	#print 'Low entropy:', cross_entropy(np.float32([[0.99, 0.001, 0.001]]).T, np.float32([[1, 0, 0]]))
	

