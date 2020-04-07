from math import exp
from random import seed
import random
from PIL import Image

# iterate through training list and create dataset of pixels
def normalize_dataset(file):
	'''Iterate through each .pgm file in list to create list of pixels. 
	input: file - containing list of filenames (i.e. downgesture_train.list.txt
	output: dataset - list of lists
	'''
	with open(file, 'r') as train_data:
		lines = train_data.read().splitlines()
		dataset = []
		for line in lines:
			img = Image.open(line)
			grey_img = img.convert(mode='L')
			# extract pixel greyscale values
			pix_val = list(grey_img.getdata())
			# normalize pixels to values between 0 and 1
			for i in range(len(pix_val)):
				pix_val[i] = pix_val[i] / 255
			# append label of image to pixel dataset
			if 'down' in line:
				pix_val.append(1)
			else:
				pix_val.append(0)
			dataset.append(pix_val)

	return dataset

def initialize_network(n_inputs, n_hidden, n_outputs):
	'''Initialize a set of hidden layer weights and outout layer weights.
	Final output will be a list-of-lists, one list per layer.
	In both hidden and ouput layer lists, one dict per perceptron. Each dict has # weights = # input xis.
	input: n_inputs - # of xis per record/image
	input: n_hidden - user specified # of perceptrons in hidden layer
	input: n_outputs - # of distinct labels'''
	network = []
	#bias is added with + 1
	hidden_layer = [{'weights': [random.uniform(-.01, .01) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	#bias is added with + 1 
	output_layer = [{'weights': [random.uniform(-.01, .01) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def summation(weights, inputs):
	''' sum the weights * inputs
	input: weights - weights of a single perceptron
	input: inputs - inputs fed to a single perceptron
	output: sum_wx - sum(w1x1 + w2x2 + ... + wdxd) + bias'''
	sum_wx = weights[-1]  # last value in weights is bias
	for i in range(len(weights) - 1):
		sum_wx += (inputs[i] * weights[i])
	return sum_wx

def sigmoid(sum_wx):
	'''S(x) = 1/(1 + e^(-x))'''
	return 1.0 / (1.0 + exp(-sum_wx))

def sigmoid_derivative(output):
	# S'(x) = x(1 - x)
	return output * (1.0 - output)

def feed_forward(network, inputs):
	'''Push inputs through network to generate output
	input: network - structure containing layer perceptrons and their weights
	input: inputs - a list of attributes of single training item
	output: inputs - list of sigmoid output of perceptrons in last layer
	'''
	for layer in network:
		new_inputs = []
		# for each perceptron, we calculate output by taking the sigmoid function of the sum of inputs
		for perceptron in layer:
			sum_wx = summation(perceptron['weights'], inputs)
			perceptron['output'] = sigmoid(sum_wx) #Update the output of each perceptron
			new_inputs.append(perceptron['output'])
		inputs = new_inputs
	return inputs #only outputs of the last layer are returned


def backward_propagate_error(network, expected):
	'''Starting with the last layer, working backwards, calculates the delta term
	for each perceptron. Each perceptron's delta is then added to the network structure'''
	for i in reversed(range(len(network))):
		layer = network[i] #Select network layer
		errors = []

		if i != len(network) - 1: #If we aren't in the last layer
			for j in range(len(layer)):
				error = 0.0
				for perceptron in network[i + 1]:
					error += (perceptron['weights'][j] * perceptron['delta'])
				errors.append(error)
		else: #If we are in the last layer
			for j in range(len(layer)):
				perceptron = layer[j]
				errors.append(expected[j] - perceptron['output'])
		for j in range(len(layer)):
			perceptron = layer[j]
			perceptron['delta'] = errors[j] * sigmoid_derivative(perceptron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):

	for i in range(len(network)):
		inputs = row[:-1]
		#If we aren't in the first layer, select inputs to be the outputs of the previous layer
		if i != 0: 
			inputs = [perceptron['output'] for perceptron in network[i - 1]]
		for perceptron in network[i]:
			for j in range(len(inputs)):
				perceptron['weights'][j] += l_rate * perceptron['delta'] * inputs[j]
			perceptron['weights'][-1] += l_rate * perceptron['delta']

# Train a network for a fixed number of epochs
def train_network(network, dataset, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in dataset:
			outputs = feed_forward(network, row)
			expected = [0 for i in range(n_outputs)]
			#If label = 1, then expected = [0, 1]
			#Else if label = 0, then expected = [1, 0]
			expected[row[-1]] = 1 
			sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, error=%.3f' % (epoch, sum_error))


def predict(network, row):
	outputs = feed_forward(network, row)
	return outputs.index(max(outputs))

dataset = normalize_dataset('downgesture_train.list.txt')
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset])) #number of unique classes
#n_outputs = 2
n_hidden = 100
learning_rate = 0.1
epochs = 10
network = initialize_network(n_inputs, n_hidden, n_outputs)
train_network(network, dataset, learning_rate, epochs, n_outputs)
for layer in network:
	print(layer)
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))
