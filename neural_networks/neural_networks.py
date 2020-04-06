from math import exp
from random import seed
import random
from PIL import Image


# iterate through training list and create dataset of pixels
def normalize_dataset(file):
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
	network = []
	hidden_layer = [{'weights': [random.uniform(-.01, .01) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights': [random.uniform(-.01, .01) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network


def summation(weights, inputs):
	# sum_wx = sum(w1x1 + w2x2 + ... + wdxd) + bias
	# w = weight, x = input
	sum_wx = weights[-1]  # last value in weights is bias
	for i in range(len(weights) - 1):
		sum_wx += (inputs[i] * weights[i])
	return sum_wx


def sigmoid(sum_wx):
	# S(x) = 1/(1 + e^(-x))
	return 1.0 / (1.0 + exp(-sum_wx))


def sigmoid_derivative(output):
	# S'(x) = x(1 - x)
	return output * (1.0 - output)


# Push inputs through network to generate output
def feed_forward(network, inputs):
	for layer in network:
		new_inputs = []
		# for each perceptron, we calculate output by taking the sigmoid function of the sum of inputs
		for perceptron in layer:
			sum_wx = summation(perceptron['weights'], inputs)
			perceptron['output'] = sigmoid(sum_wx)
			new_inputs.append(perceptron['output'])
		inputs = new_inputs

	return inputs


# Backpropagate error and store in perceptrons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = []
		if i != len(network) - 1:
			for j in range(len(layer)):
				error = 0.0
				for perceptron in network[i + 1]:
					error += (perceptron['weights'][j] * perceptron['delta'])
				errors.append(error)
		else:
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
n_outputs = len(set([row[-1] for row in dataset]))
n_hidden = 100
learning_rate = 0.1
epochs = 1000
network = initialize_network(n_inputs, n_hidden, n_outputs)
train_network(network, dataset, learning_rate, epochs, n_outputs)
for layer in network:
	print(layer)
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))

