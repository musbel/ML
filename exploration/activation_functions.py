#!/usr/bin/python

'''
Exploring and visualising activation functions and their derivatives
'''

import numpy as np

# Using a simple class structure to make it easier to add/remove visualisations below
class Activation:
	@property 
	def label(self): pass
	def function(self, x): pass
	def derivative(self, x): pass

class Linear(Activation):
	def __init__(self, slope=1.0):
		self.slope = slope

	@property
	def 	label(self):
		return 'linear (%.1f)' % self.slope
	
	def function(self, x):
		return self.slope * x
		
	def derivative(self, x):
		return np.ones_like(x)
		
class Sigmoid(Activation):
	def __init__(self, alpha=1.0):
		self.alpha = alpha

	@property
	def 	label(self): 
		return 'sigmoid (%.1f)' % self.alpha
	
	def function(self, x):
		return 1.0 / (1.0 + np.exp(-x * self.alpha))

	def derivative(self, x):
		s = self.function(x) # chain rule
		return s * (1 - s)

class Tanh(Activation):
	@property
	def 	label(self): 
		return 'tanh'

	def function(self, x):
		return np.tanh(x)
		# Play with different ways of calculating tanh
		#return np.sinh(x) / np.cosh(x)
		#e_x = np.exp(-2 * x)
		#return (1 - e_x) / (1 + e_x)

	def derivative(self, x):
		return 1 - pow(self.function(x), 2)

class Relu(Activation):
	@property
	def 	label(self): 
		return 'relu'

	def function(self, x):
		return np.maximum(0, x)
		#return np.float32([max(0, x_i) for x_i in x])

	def derivative(self, x):
		return np.float32([1 if x_i > 0 else 0 for x_i in x])

class Softplus(Activation):
	@property
	def 	label(self): 
		return 'softplus'

	def function(self, x):
		return np.log(1 + np.exp(x)) # ln (base e)

	def derivative(self, x):
		# The derivative of the softplus function is the sigmoid (logistic) function
		return 1.0 / (1.0 + np.exp(-x))

class Gaussian(Activation):
	@property
	def 	label(self): 
		return 'gaussian'

	def function(self, x):
		return np.exp(-pow(x, 2))

	def derivative(self, x):
		return -2.0 * x * self.function(x) # chain rule


# Visualise the output of the different activation functions from an input range
import matplotlib.pyplot as plt
x = np.arange(-4.0, 4.0, 0.1)

activations = []
#activations.append(Linear())
activations.append(Sigmoid())
activations.append(Sigmoid(2.5))
activations.append(Tanh())
activations.append(Relu())
activations.append(Softplus())
activations.append(Gaussian())

# Visualise activation functions
y = np.vstack([activation.function(x) for activation in activations])
h = plt.plot(x, y.T, linewidth=2)
plt.title('Activation functions')
plt.figlegend(h, (activation.label for activation in activations), 'upper right')

# Visualise derivatives for activation functions
dy = np.vstack([activation.derivative(x) for activation in activations])
plt.figure()
plt.title('Derivatives for activation functions')
plt.plot(x, dy.T, linewidth=2)
plt.figlegend(h, (activation.label for activation in activations), 'upper right')
plt.show()

# Notice how the gradient is stronger when using the tanh activation function when
# compared to the sigmoid function, better for gradient descent

