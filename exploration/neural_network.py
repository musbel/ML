#!/usr/bin/python

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


class NeuralNetwork:
	def __init__(self, useRegulariser=True, Lambda=0.0001):
		# Initialise hyper parameters
		self.numInputNodes = 2  # Number of channels per sample (data points)
		self.numOutputNodes = 1 # Number of predicted channels (estimate, class label)
		self.numHiddenNodes = 3

		self.useRegulariser = useRegulariser
		self.Lambda = Lambda

		# Initialise model parameters		
		# Initialise weight matrices using a normal distribution
		self.W1 = np.random.rand(self.numInputNodes, self.numHiddenNodes)
		self.W2 = np.random.rand(self.numHiddenNodes, self.numOutputNodes)
		
	def forward(self, X):
		'''
		Push the input data forward through the network to compute a prediction estimate y_hat

		X:  Input data (numSamples x numInputNodes)
		W1: Weights from input to hidden layer (n x numHiddenNodes)
		
		1) From inputs to first hidden layer
		z2: Hidden layer propagation (from inputs) -> Activity of second (hidden) layer
		    (numSamples x numHiddenNodes)
		    => np.dot(X, W)
		a2: Hidden layer activity by applying an activation function
		    sigma is the sigmoid activation function (numSamples x numHiddenNodes)
		    => sigma(z2)
	         => sigma(np.dot(X, W1))
		
		2) From hidden layer to outputs
		W2:    Weights from the second (hidden) layer to the output layer (z3)
		       (numHiddenNodes x numOutputNodes)
		z3:    Activity of the third (output) layer
		       (numSamples x numHiddenNodes) * (numHiddenNodes x numOutputNodes)
		       => (numSamples, numOutputNodes)
		       => One or more (depending on output nodes) activity values, one for each input sample
		       => z3 = a2 * W2
		y_hat: Estimate (prediction) for outputs by employing an activation function (e.g. softmax to
		       produce probability values 0..1)
		       => y_hat = sigma(z3)
		       => sigma(np.dot(a2 W2))
		       Putting it all together:
			  y_hat = sigma(sigma(X, W1), W2)
		'''
		
		# Propagate input to second (hidden) layer
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		
		self.z3 = np.dot(self.a2, self.W2)
		self.y_hat = self.sigmoid(self.z3)
		
		return self.y_hat

	def computeCostFunction(self, X, y):
		'''
		Calculate the cost function error
		
		Here we're using a simple sum of squared distances as a cost function (y - y_hat)
		 => J = SUM[1/2 * (y - y_hat)^2]

		Where y_hat is the y estimate calculated by pushing the input values through
		the network
		'''
		
		# Start by getting the y estimate y_hat by pushing the inputs through the network
		self.y_hat = self.forward(X)
		
		# Compute cost function (here we use sum of squares error/cost function)
		#  -> J = 1/2 * SUM[(y - y_hat)^2]
		J = 0.5 * np.sum((y - self.y_hat)**2)
		
		if self.useRegulariser:
			# Add the square of our weights to our cost function to penalise overly
			# complex models. This way models with larger weights cost more
			squareWeightsSum = np.sum(self.W1**2) + np.sum(self.W2**2)
			
			# Use Lambda hyper parameter to tune the relative cost
			regularisationTerm = (self.Lambda / 2) * squareWeightsSum
			
			# We need to normalise the non-regulariser term of our cost function
			# to ensure that our ratio of the two error terms does not change
			# with respect to the number of samples
			numSamples, numChannels = X.shape
			J = J / numSamples + regularisationTerm
		
		return J
	
	def computeCostFunctionDerivatives(self, X, y):
		'''
		Takes the input samples and target values and calculate the cost function partial
		derivatives with respect to the weights
		
		Minimise cost function J
		 - We want to minimise the cost function during back-propagation and gradient descent
		
		Here we're using a simple sum of squared distances as a cost function (y - y_hat)
		 => J = SUM[1/2 * (y - y_hat)^2]
		 => J = SUM[1/2 * (y - sigma(sigma(X, W1), W2))^2]
		
		Derivative (partial): dJ/dW
		Partial derivative because we're only considering one weight at a time
		 - Derive an expression for dJ/dW that will give us the rate of change of
		   J with respect to W for any value in W
		
		If a value for dJ/dW is positive: Cost function is going uphill
		If a value for dJ/dW is negative: Cost function is going downhill
		We obviously want to go downhill during gradient descent
		
		Non-convex issue for gradient descent -> Get stuck in local minimum
		 - Sum of squared errors is a convex quadratic function (x^2) so using it here
		   as a cost function helps (check cross-entropy)
		 - Using Stochastic Gradient Descent (one sample at a time from a shuffled set)
		   as opposed to the standard Gradient Descent (batch) it's often possible to
		   still efficiently find a good minimum even with non-convex loss (cost) functions
		
		Using batch for now
		Compute gradients for W1 and W2 as the weights are split across these two matrices:
		 - dJ/dW1: (input size x hidden layer size)  -> (numInputNodes x numHiddenNodes)
		 - dJ/dW2: (hidden layer size x output size) -> (numHiddenNodes x numOutputNodes)
		
		#### dJ/dW2 ####
		dJ/dW2: Here we can see how the W2 directly affect the output values
		The sum in the cost function adds the error for each sample to create an overall cost
		 => dJ/dW2 = d(SUM[1/2 * (y - y_hat)^2])/dW2
		
		We can take advantage of the sum rule in differentiation:
		The derivative of the sum == The sum of the derivatives
		 => dJ/dW2 = SUM[d(1/2 * (y - y_hat)^2)]/dW2
		
		Let's simplify for now by removing the sum and look at a derivative for a single sample
		 => dJ/dW2 = d(1/2 * (y - y_hat)^2)/dW2
		
		Use the power rule: E.g. x^2 = 2x
		 => dJ/dW2 = (1/2 * 2 * (y - y_hat))
		 => dJ/dW2 = (y - y_hat) -> Now we need to multiply with the inside derivate (see chain rule)
		
		Now use the chain rule: (f(g))' = f(g)' * g'
		 - Product of derivatives: dz/dx = dz/dy * dy/dx
		Looking at (y - y_hat):
		 - Inside: dy/dW2 - dy_hat/dW2
	 	 - y are the target labels which are constant so they disappear -> dy/dW2 = 0
		 - y_hat changes with respect to W2 -> -1 * dy_hat/dW2
		
		So what we've got now using the chain rule (outside * inside)
		 => dJ/dW2 = (y - y_hat) * -dy_hat/dW2
		
		>> dJ/dW2 = -(y - y_hat) * dy_hat/dW2
		
		What is dy_hat/dW2?
		 - The equation for y_hat is the activation function of z3:
		   => y_hat = sigma(z3) (see forward function)
		 - Break dy_hat/dW2 using the chain rule:
		   => dy_hat/dW2 = dy_hat/dz3 * dz3/dW2
		 - dy_hat/dz3 is just the derivative for the activation function (here sigmoid)
		   => f'(z) = sigma'(z) = e^-z / (1 + e^-z)^2
		   => dy_hat/dW2 = sigma'(z3) * dz3/dW2
		
		>> dJ/dW2 = -(y - y_hat) * sigma'(z3) * dz3/dW2
		 
		Now we look at dz3/dW2
		  - dz3 = np.dot(a2, W2) 
		    (activities from hidden layer multipled with the corresponding weights added together)
		  - Looking at just one hidden layer node -> Simple linear relationship [a2(1) * W2(1, 1) = z3]
		  - W2 is the variable we're looking at z3 with respect to
		  - Therefore: a2(1) * W2(1, 1) = z3 -> (simplified) aw = z -> a is the slope!
		  - Equivalent to back-propagating the error to each weight
		    - By multiplying with the activity of each synapse, weights that contribute more to the
		      overall error will have larger activations which yield larger dJ/dW values and are
		      therefore more responsible for the overall error -> Changed more during Gradient Descent
		  - dz3/dW2 -> The activation matrix ...
		 
		 >> dJ/dW2 = -(y - y_hat) * sigma'(z) * a2 [a2 activation matrix -> slopes]

		 Back-propagating error -> delta
		  => delta3 = -(y - y_hat) * sigma'(z)
		  => dJ/dW2 = delta3 * a2
		 
		 1) dJ/dW2 = a2.T * delta3
		 
		 The delta has to be multiplied with each activity -> Transpose A2 and matrix multiply by delta3
		  => np.dot(A2.T, delta3)
		  => Note: Also takes care of summing up the dJ/dW terms we ignored earlier
		
		 How to look at this:
		  - Each sample has a certain cost and gradient
		  - The gradient for each sample pulls our gradient descent algorithm in a certain direction
		    (every sample gets a vote on which way is downhill)
		  - Batch gradient descent adds together everyone's vote and move in that direction (compromise)

		#### dJ/dW1 ####
		It starts the same way as dW1 (looking at a single sample)
		 => dJ/dW1 = d(1/2 * (y - y_hat)^2)/dW1
		
		We use the power rule and chain rule again
		 => dJ/dW1 = (1/2 * 2 * (y - y_hat)) * -dy_hat/dW1
		 => dJ/dW1 = -(y - y_hat) * dy_hat/dW1
		 => dJ/dW1 = -(y - y_hat) * dy_hat/dz3 * dz3/dW1 
		 => dy_hat/dz3 = sigma'(z3) [see above]
		 => dJ/dW1 = -(y - y_hat) * sigma'(z3) * dz3/dW1
		 => delta3 = -(y - y_hat) * sigma'(z3)
		
		>> dJ/dW1 = delta3 * dz3/dW1
		
		Now we look at dz3/dW1. We need to take the derivate across our synapses which is different to
		before where we took the derivate with respect to the weights on our synapses
		Apply the chain rule:
		 => dJ/dW1 = delta3 * dz3/da2 * da2/dW1
		
		(dz3/da3) What's the derivative of z3 with the respect of a2?
		 - The slope is equal to the weight of a respective synapse, e.g. W2_11
		 - We achieve this by simply multiplying the transposed weights of the synapses
		 - Replacing dz3/da2 in the equation gives us:
		 => dJ/dW1 = delta3 * W2.T * da2/dW1
		
		Now we look at da2/dW1
		Applying the chain rule gives us:
		 => da2/dW1 = da2/dz2 * dz2/dW1
		
		(da2/dz2) What's the derivative of a2 with the respect of z2? 
		 - The derivative again is just the derivative of the activation function
		 => da2/dz2 = sigma'(z2)
		 
		>> dJ/dW1 = delta3 * W2.T * sigma'(z2) * dz2/dW1
		
		Now we have to deal with dz2/dW1
		 - Linear relationship between z2 and W1, where the input values X are the slope
		   (similar to the scenario earlier where a2 was the slope in the linear relationship)
		 - Therefore, we can substitute the derivative dz2/dW1 with the X transposed
		   (putting it at the front)
		
		>> dJ/dW1 = X.T * delta3 * W2.T * sigma'(z2)
		
		We can combine terms into delta2
		 => delta2 = delta3 * W2.T * sigma'(z2)
		
		2) dJ/dW1 = X.T * delta2
		'''
		
		# First compute the y estimates, y_hat, using the forward function
		self.y_hat = self.forward(X)
	
		# Compute delta3 using element-wise multiply
		#  -> delta3 = -(y - y_hat) * sigma'(z3)
		delta3 = np.multiply(-(y - self.y_hat), self.sigmoid_prime(self.z3))
		
		# Compute dJ/dW2
		#  -> dJ/dW2 = a2.T * delta3
		dJdW2 = np.dot(self.a2.T, delta3)
		
		# Compute delta2
		#  -> delta2 = delta3 * W2.T * sigma'(z2)
		delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)
		
		# Compute dJ/dW1
		#  -> dJ/dW1 = X.T * delta2
		dJdW1 = np.dot(X.T, delta2)
		
		if self.useRegulariser:
			numSamples, numChannels = X.shape
			dJdW1 = dJdW1 / numSamples + self.Lambda * self.W1
			dJdW2 = dJdW2 / numSamples + self.Lambda * self.W2
		
		return dJdW1, dJdW2

	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.computeCostFunctionDerivatives(X, y)
		
		# Roll the derivatives into a vector
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

	def getParams(self):
		# Roll the parameters W1 and W2 into a vector
		# - Ravel turns the matrix into a vector
		# - Concatenate both weight vectors into a single vector
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		'''
		Update weight parameters from a single vector containing all 
		concatenated weights
		'''
		
		# We have to know where each sub-vector for the weights starts and ends
		
		# W1: (numInputNodes x numHiddenNodes) -> Reshape from a vector into matrix
		W1_start = 0
		W1_end = self.numInputNodes * self.numHiddenNodes
		W1_params = params[W1_start:W1_end]
		self.W1 = W1_params.reshape(self.numInputNodes, self.numHiddenNodes)
		
		# W2: (numHiddenNodes x numOutputNodes) -> Reshape from a vector into matrix
		W2_end = W1_end + self.numHiddenNodes * self.numOutputNodes
		W2_params = params[W1_end:W2_end]
		self.W2 = W2_params.reshape(self.numHiddenNodes, self.numOutputNodes)

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def sigmoid_prime(self, z):
		return np.exp(-z) / (1 + np.exp(-z))**2
		

class Trainer:
	def __init__(self, NN):
		self.NN = NN
		
		# Training parameters
		self.maxiter = 200
		
		# A list of training costs
		self.J = []
		self.X = None
		self.y = None
		
		self.J_test = []
		self.X_test = None
		self.y_test = None
		
		self.result = None
	
	def minimise(self, params, X, y):
		'''
		A callback function for optimisers, where we update the
		network parameters and compute the cost and gradient
		'''
		self.NN.setParams(params)
		
		cost = self.NN.computeCostFunction(X, y)
		grad = self.NN.computeGradients(X, y)
		
		return cost, grad
	
	def optimiserCallback(self, params):
		'''
		A callback to keep track of the optimisation process
		'''
		self.NN.setParams(params)
		
		# Log training costs
		self.J.append(self.NN.computeCostFunction(self.X, self.y))
		
		if self.X_test is not None and self.y_test is not None:
			self.J_test.append(self.NN.computeCostFunction(self.X_test, self.y_test))

	def train(self, X, y, X_test=None, y_test=None):
		# Clear training costs and initialise callback variables
		self.J, self.X, self.y = [], X, y
		self.J_test, self.X_test, self.y_test = [], X_test, y_test
		
		# Get initial parameters
		initialParams = self.NN.getParams()
		
		# Set up optimiser and ask it to minimise our cost function
		options = {
			'maxiter': self.maxiter,
			'disp': True
		}
		
		self.result = optimize.minimize(self.minimise, initialParams, jac=True, method='BFGS', args=(X, y), \
		                               options=options, callback=self.optimiserCallback)
		
		# Update network parameters
		self.NN.setParams(self.result.x)


class Normaliser():
	def __init__(self, X, y):
		self.X_max = np.amax(X, axis=0)
		self.y_max = 100.

	def normalise(self, X, y):
		X_norm = X / self.X_max
		y_norm = y / self.y_max
		return X_norm, y_norm


def testNetwork1():
	# Test the network using some data
	# Our input data is samples of [# hours sleep, # hours study]
	X = np.float32([[3, 5], [5, 1], [10, 2]]) # (3x2) -> 3 samples with 2 input values

	# Our output data is the test score for each sample
	y = np.float32([[75], [82], [93]]) # (3x1) -> 3 samples with 1 target value each
	
	normaliser = Normaliser(X, y)
	X_norm, y_norm = normaliser.normalise(X, y)
	
	# Create and train a network using the normalised inputs and output targets
	nn = NeuralNetwork()
	trainer = Trainer(nn)
	trainer.train(X_norm, y_norm)
		
	# Test by predicting the training data (should be spot on)
	print nn.forward(X_norm)
	
	plotDataRange(nn)
	plt.show()

def testNetworkOverfitting(useRegulariser=False):
	# Test the network using some data that's more prone to overfitting
	# Our input data is samples of [# hours sleep, # hours study]
	X = np.float32([[3, 5], [5, 1], [10, 2], [6, 1.5]])

	# Our output data is the test score for each sample
	y = np.float32([[75], [82], [93], [70]])

	normaliser = Normaliser(X, y)
	X_norm, y_norm = normaliser.normalise(X, y)
	
	X_test = np.float32([[4, 5.5], [4.5, 1], [9, 2.5], [6, 2]])
	y_test = np.float32([[70], [89], [85], [75]])
	X_test, y_test = normaliser.normalise(X_test, y_test)
	
	# Create and train a network using the normalised inputs and output targets
	nn = NeuralNetwork(useRegulariser)
	trainer = Trainer(nn)
	trainer.train(X_norm, y_norm, X_test, y_test)
		
	# Test by predicting the training data (should be spot on)
	print nn.forward(X_norm)
	
	plotTraining(trainer)
	plotDataRange(nn, X_norm, y_norm)
	plt.show()

def plotTraining(trainer):
	fig = plt.figure()
	plt.plot(trainer.J, label='Training Error')
	if trainer.J_test: plt.plot(trainer.J_test, label='Testing Error')
	plt.grid(True)
	plt.ylabel('Cost')
	plt.xlabel('Iterations')
	plt.legend(loc='upper center')
	plt.yticks(np.arange(0.0, 0.05, 0.01))

def plotDataRange(nn, X=None, y=None):
	# Look at different combinations
	numSamples = 100
	hoursSleep = np.linspace(0, 10, numSamples) # n samples between 0-10
	hoursStudy = np.linspace(0, 5, numSamples)  # n samples between 0-5
	
	# Normalise
	hoursSleep = hoursSleep / 10 # Note: Hard coded max value of 10
	hoursStudy = hoursStudy / 5  # Note: Hard coded max value of 5
	
	a, b = np.meshgrid(hoursSleep, hoursStudy)
	
	# Same as: yy = np.tile(hoursStudy, (100, 1)).T
	yy = np.dot(hoursStudy.reshape(numSamples, 1), np.ones((1, numSamples)))
	
	xx = np.dot(hoursSleep.reshape(numSamples, 1), np.ones((1, numSamples))).T
	
	# Create inputs from the range of sample data
	inputs = np.zeros((a.size, 2))
	inputs[:, 0] = a.ravel()
	inputs[:, 1] = b.ravel()
	
	predictions = nn.forward(inputs)
	
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(xx, yy, 100 * predictions.reshape(100, 100), cmap=plt.cm.jet)
	ax.set_xlabel('Hours Sleep')
	ax.set_ylabel('Hours Study')
	ax.set_zlabel('Test Score (predicted)')
	
	# Plot data points to visualise e.g. overfitting
	if X is not None and y is not None:
		ax.scatter(X[:, 0], X[:, 1], 100 * y, c='k', alpha=1, s=40)


if __name__ == "__main__":	
#	nn = testNetwork1()
	nn = testNetworkOverfitting(useRegulariser=True)

