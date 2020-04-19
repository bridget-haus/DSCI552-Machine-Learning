from PIL import Image
import random
from math import exp

import numpy as np  # For visualization
import matplotlib.pyplot as plt  # For visualization


class simple_network():
	'''Builds an object to represent simple, single-layer, feed-forward neural network.'''

	def __init__(self, *args, **kwargs):
		'''Store the user imput parameters only.
		In this step, weights are not yet initialized.'''
		self.hidden_layer_size = kwargs.get('hidden_layer_size')  # Model parameters
		self.l_rate = kwargs.get('l_rate')
		self.max_epochs = kwargs.get('max_epochs')

		self.train_data = []  # Training data and model
		self.network = []
		self.sum_errors = []

		self.test_data = []  # Testing data and stats
		self.accuracy = 0
		self.precision = 0
		self.recall = 0
		self.confusion_table = []
		self.sum_tn = 0
		self.sum_fp = 0
		self.sum_fn = 0
		self.sum_tp = 0
		self.total = 0

	def normalize_dataset(self, file):
		'''Iterate through each .pgm file in list to create list of pixels.
		output: dataset - list of lists'''
		with open(file, 'r') as train_data:
			lines = train_data.read().splitlines()
			# dataset = []
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
				# add to train or test dataset, depending on file argument
				if file == 'downgesture_train.list.txt':
					self.train_data.append(pix_val)
				else:
					self.test_data.append(pix_val)

	def summation(self, weights, inputs):
		''' sum the weights * inputs
		input: weights - weights of a single perceptron
		input: inputs - inputs fed to a single perceptron
		output: sum_wx - sum(w1x1 + w2x2 + ... + wdxd) + bias'''
		sum_wx = weights[-1]  # last value in weights is bias
		for i in range(len(weights) - 1):
			sum_wx += (inputs[i] * weights[i])
		return sum_wx

	def sigmoid(self, sum_wx):
		'''S(x) = 1/(1 + e^(-x))'''
		return 1.0 / (1.0 + exp(-sum_wx))

	def feed_forward(self, inputs):
		'''Push inputs through network to generate output
		input: network - structure containing layer perceptrons and their weights
		input: inputs - a list of attributes of single training item
		output: inputs - list of sigmoid output of perceptrons in last layer'''
		for layer in self.network:
			new_inputs = []
			# for each perceptron, we calculate output by taking the sigmoid function of the sum of inputs
			for perceptron in layer:
				sum_wx = self.summation(perceptron['weights'], inputs)
				perceptron['output'] = self.sigmoid(sum_wx)  # Update the output of each perceptron
				new_inputs.append(perceptron['output'])
			# after all perceptrons finish outputting
			# set inputs to new_inputs to prepare for next layer
			inputs = new_inputs
		outputs = inputs  # only outputs of the last layer are returned
		return outputs

	def predict(self, row):
		'''Get the outputs of the last layer and tell us which class index ranked higher. '''
		outputs = self.feed_forward(row)
		return outputs.index(max(outputs))

	def train_model(self, file):
		'''Driver function for building the neural network model.
		input: file - containing list of filenames (i.e. downgesture_train.list.txt)'''

		def create_layers(num_inputs, num_hidden, num_outputs):
			'''Initialize a set of hidden layer weights and outout layer weights.
			Final output will be a list-of-lists, one list per layer.
			In both hidden and ouput layer lists, one dict per perceptron. Each dict has # weights = # input xis.
			input: n_inputs - # of xis per record/image
			input: n_hidden - user specified # of perceptrons in hidden layer
			input: n_outputs - # of distinct labels'''

			hidden_layer = []
			# initialize random weight for each input to connect to hidden + bias
			for i in range(num_hidden):
				input_weights = []
				for j in range(num_inputs + 1):
					input_weights.append(random.uniform(-.01, .01))
				i_weights = {'weights': input_weights}
				hidden_layer.append(i_weights)
			output_layer = []
			# initialize random weight for each hidden to connect to output + bias
			for i in range(num_outputs):
				output_weights = []
				for j in range(num_hidden + 1):
					output_weights.append(random.uniform(-.01, .01))
				o_weights = {'weights': output_weights}
				output_layer.append(o_weights)

			self.network = [hidden_layer, output_layer]

		def sigmoid_derivative(output):
			'''S'(x) = x(1 - x)'''
			return output * (1.0 - output)

		def backward_propagate_error(expected, row):
			'''Starting with the last layer, working backwards, calculates the delta term
			for each perceptron. Each perceptron's delta is then added to the network structure.
			Then the delta is used with learning rate to update weights and bias terms.'''
			for i in reversed(range(len(self.network))):
				layer = self.network[i]  # Select network layer
				errors = []
				inputs = row[:-1]
				if i != len(self.network) - 1:  # If we aren't in the last layer
					for j in range(len(layer)):
						error = 0.0
						for perceptron in self.network[i + 1]:
							error += (perceptron['weights'][j] * perceptron['delta'])
						errors.append(error)
						perceptron = layer[j]
						perceptron['delta'] = errors[j] * sigmoid_derivative(perceptron['output'])

						for jj in range(len(inputs)):  # update weights
							perceptron['weights'][jj] += self.l_rate * perceptron['delta'] * inputs[jj]
						perceptron['weights'][-1] += self.l_rate * perceptron['delta']

				else:  # If we are in the last layer
					inputs = [perceptron['output'] for perceptron in self.network[i - 1]]
					for j in range(len(layer)):
						perceptron = layer[j]
						errors.append(expected[j] - perceptron['output'])
						perceptron['delta'] = errors[j] * sigmoid_derivative(perceptron['output'])

						for jj in range(len(inputs)):  # update weights
							perceptron['weights'][jj] += self.l_rate * perceptron['delta'] * inputs[jj]
						perceptron['weights'][-1] += self.l_rate * perceptron['delta']

		def train_network(n_outputs):
			'''Given a # of epochs, train a network example-by-example.
			After each image example, calculate error and back propagate error recursively
			through the pre-existing weights.'''
			sum_errors = []
			for epoch in range(self.max_epochs):
				sum_error = 0
				for row in self.train_data:
					outputs = self.feed_forward(row)
					expected = [0 for i in range(n_outputs)]
					# If label = 1, then expected = [0, 1]
					# Else if label = 0, then expected = [1, 0]
					expected[row[-1]] = 1
					sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
					backward_propagate_error(expected, row)
				print('>epoch=%d, error=%.3f' % (epoch, sum_error))
				sum_errors.append(sum_error)
			self.sum_errors = enumerate(sum_errors, 1)

		# TRAIN - DRIVER CODE
		self.normalize_dataset(file)
		n_inputs = len(self.train_data[0]) - 1
		n_outputs = len(set([row[-1] for row in self.train_data]))  # number of unique classes
		create_layers(n_inputs, self.hidden_layer_size, n_outputs)
		train_network(n_outputs)

	def test_model(self, file):
		'''Call this method after model is built, when testing new examples.
		This will classify the test examples, record results, and provide accuracy, recall, precision stats.
		input: file - containing list of filenames (i.e. downgesture_test.list.txt)'''

		def create_test_stats(accuracy_table):
			'''Calculates the test stats and saves to class.'''
			confusion_table = []
			sum_tn = 0
			sum_fp = 0
			sum_fn = 0
			sum_tp = 0
			total = 0
			for outcome in accuracy_table:
				if outcome['expected'] == 0 and outcome['got'] == 0:  # True Negative
					confusion_table.append(
						{'expected': outcome['expected'], 'got': outcome['got'], 'tp': 0, 'tn': 1, 'fp': 0, 'fn': 0})
					sum_tn += 1
				elif outcome['expected'] == 0 and outcome['got'] == 1:  # False Positive
					confusion_table.append(
						{'expected': outcome['expected'], 'got': outcome['got'], 'tp': 0, 'tn': 0, 'fp': 1, 'fn': 0})
					sum_fp += 1
				elif outcome['expected'] == 1 and outcome['got'] == 0:  # False Negative
					confusion_table.append(
						{'expected': outcome['expected'], 'got': outcome['got'], 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 1})
					sum_fn += 1
				elif outcome['expected'] == 1 and outcome['got'] == 1:  # True Positive
					confusion_table.append(
						{'expected': outcome['expected'], 'got': outcome['got'], 'tp': 1, 'tn': 0, 'fp': 0, 'fn': 0})
					sum_tp += 1
				total += 1
			try:
				accuracy = round(((sum_tp + sum_tn) / total), 4)
			except ZeroDivisionError:
				accuracy = 0
			try:
				precision = round((sum_tp / (sum_tp + sum_fp)), 4)
			except ZeroDivisionError:
				precision = 0
			try:
				recall = round((sum_tp / (sum_tp + sum_fn)), 4)
			except ZeroDivisionError:
				recall = 0
			# Save all calculations to the class
			self.accuracy = accuracy
			self.precision = precision
			self.recall = recall
			self.confusion_table = confusion_table
			self.sum_tn = sum_tn
			self.sum_fp = sum_fp
			self.sum_fn = sum_fn
			self.sum_tp = sum_tp
			self.total = total

		# TEST - DRIVER CODE
		self.normalize_dataset(file)
		accuracy_table = []
		for row in self.test_data:
			prediction = self.predict(row)
			accuracy_table.append({'expected': row[-1], 'got': prediction})
		create_test_stats(accuracy_table)
		print('Accuracy Table')
		for row in accuracy_table:
			print(row)
		print('Accuracy: ', self.accuracy)
		print('Precision: ', self.precision)
		print('Recall: ', self.recall)

	def plot_train_errors(self):
		'''Plot a y-axis graph of erros per iterations'''
		iters = []
		errors = []
		for i in self.sum_errors:
			iters.append(i[0])
			errors.append(i[1])
		iters = np.asarray(iters)  # x
		errors = np.asarray(errors)  # y1
		fig, ax1 = plt.subplots()
		color = 'tab:red'
		ax1.set_xlabel('epochs')
		ax1.set_ylabel('sum errors', color=color)
		ax1.plot(iters, errors, color=color)
		ax1.tick_params(axis='y', labelcolor=color)

		fig.tight_layout()
		plt.show()


if __name__ == '__main__':
	# INITIALIZE MODEL
	ff_model = simple_network(hidden_layer_size=10,
							  l_rate=0.1, max_epochs=10)
	# TRAIN MODEL
	file = 'downgesture_train.list.txt'
	ff_model.train_model(file)
	# TEST MODEL
	file = 'downgesture_test.list.txt'
	ff_model.test_model(file)
	# PLOT TRAINING ERRORS
	ff_model.plot_train_errors()
