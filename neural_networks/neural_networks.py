from math import exp
import random
from PIL import Image
import numpy as np #For visualization
import matplotlib.pyplot as plt #For visualization


def main():

	n_hidden = 20
	l_rate = 0.1
	epochs = 10
	test_model(n_hidden, l_rate, epochs)


def normalize_dataset(file):
	'''Iterate through each .pgm file in list to create list of pixels.
	input: file - containing list of filenames (i.e. downgesture_train.list.txt)
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


def create_layers(num_inputs, num_hidden, num_outputs):

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

	layers = [hidden_layer, output_layer]

	return layers


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
	'''S'(x) = x(1 - x)'''
	return output * (1.0 - output)


def feed_forward(layers, inputs):
	'''Push inputs through network to generate output
	input: network - structure containing layer perceptrons and their weights
	input: inputs - a list of attributes of single training item
	output: inputs - list of sigmoid output of perceptrons in last layer
	'''
	for layer in layers:
		outputs = []
		# for each perceptron, we calculate output by taking the sigmoid function of the sum of inputs
		for perceptron in layer:
			sum_wx = summation(perceptron['weights'], inputs)
			perceptron['output'] = sigmoid(sum_wx) #Update the output of each perceptron
			outputs.append(perceptron['output'])

		#after all perceptrons finish outputting
		#set inputs to new_inputs to prepare for next layer
		inputs = outputs

	return inputs #only outputs of the last layer are returned


def backward_propagate_update(layers, expected, row, l_rate):
	'''Starting with the last layer, working backwards, calculates the delta term
	for each perceptron. Each perceptron's delta is then added to the layers structure'''
	for i in reversed(range(len(layers))):
		layer = layers[i] #Select network layer
		errors = []

		if i != len(layers) - 1: #If we aren't in the last layer
			for j in range(len(layer)):
				error = 0.0
				for perceptron in layers[i + 1]:
					error += (perceptron['weights'][j] * perceptron['delta'])
				errors.append(error)
		else: #If we are in the last layer
			for j in range(len(layer)):
				perceptron = layer[j]
				errors.append(expected[j] - perceptron['output'])
		for j in range(len(layer)):
			perceptron = layer[j]
			perceptron['delta'] = errors[j] * sigmoid_derivative(perceptron['output'])

	for i in range(len(layers)):
		inputs = row[:-1]
		#If we aren't in the first layer,
		#select inputs to be the outputs of the previous layer
		if i != 0:
			inputs = [perceptron['output'] for perceptron in layers[i - 1]]
		for perceptron in layers[i]:
			for j in range(len(inputs)):
				perceptron['weights'][j] += l_rate * perceptron['delta'] * inputs[j]
			perceptron['weights'][-1] += l_rate * perceptron['delta']


def train_layers(layers, dataset, l_rate, n_epoch, n_outputs):
	'''Given a # of epochs, train a network example-by-example.
	After each image example, calculate error and back propagate error recursively
	through the pre-existing weights.'''
	sum_errors = []
	for epoch in range(n_epoch):
		sum_error = 0
		for row in dataset:
			outputs = feed_forward(layers, row)
			expected = [0 for i in range(n_outputs)]
			#If label = 1, then expected = [0, 1]
			#Else if label = 0, then expected = [1, 0]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
			backward_propagate_update(layers, expected, row, l_rate)
		print(f'epoch={epoch}, error={sum_error}')
		sum_errors.append(sum_error)
	sum_errors = enumerate(sum_errors, 1)
	return sum_errors


def predict(layers, row):
	'''Get the outputs of the last layer and tell us which class index ranked higher. '''
	outputs = feed_forward(layers, row)
	return outputs.index(max(outputs))


def build_train_model(num_hidden, l_rate, epochs):
	train_data = normalize_dataset('downgesture_train.list.txt')
	num_inputs = len(train_data[0]) - 1
	num_outputs = len(set([row[-1] for row in train_data])) #number of unique classes
	layers = create_layers(num_inputs, num_hidden, num_outputs)
	sum_errors = train_layers(layers, train_data, l_rate, epochs, num_outputs)
	return layers, sum_errors


class test_stats():
	def __init__(self, accuracy_table):
		'''Build up a confusion matrix to enable calculation of
		accuracy, precision, and recall. Then store then in the class object
		for easy retreival later on.'''
		confusion_table = []
		sum_tn = 0
		sum_fp = 0
		sum_fn = 0
		sum_tp = 0
		total = 0
		for outcome in accuracy_table:
			if outcome['expected'] ==0 and outcome['got'] == 0: #True Negative
				confusion_table.append({'expected': outcome['expected'], 'got': outcome['got'], 'tp': 0, 'tn': 1, 'fp': 0, 'fn': 0})
				sum_tn += 1
			elif outcome['expected'] ==0 and outcome['got'] == 1: #False Positive
				confusion_table.append({'expected': outcome['expected'], 'got': outcome['got'], 'tp': 0, 'tn': 0, 'fp': 1, 'fn': 0})
				sum_fp += 1
			elif outcome['expected'] ==1 and outcome['got'] == 0: #False Negative
				confusion_table.append({'expected': outcome['expected'], 'got': outcome['got'], 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 1})
				sum_fn += 1
			elif outcome['expected'] ==1 and outcome['got'] == 1: #True Positive
				confusion_table.append({'expected': outcome['expected'], 'got': outcome['got'], 'tp': 1, 'tn': 0, 'fp': 0, 'fn': 0})
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
		#Save all calculations to the class
		self.accuracy = accuracy
		self.precision = precision
		self.recall = recall
		self.confusion_table = confusion_table
		self.sum_tn = sum_tn
		self.sum_fp = sum_fp
		self.sum_fn = sum_fn
		self.sum_tp = sum_tp
		self.total = total


def plot_errors(iter_errors):
    '''Plot a y-axis graph of erros per iterations'''
    iters = []
    errors = []
    for i in iter_errors:
        iters.append(i[0])
        errors.append(i[1])
    iters = np.asarray(iters) #x
    errors = np.asarray(errors) #y1

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('errors', color=color)
    ax1.plot(iters, errors, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


def test_model(n_hidden, l_rate, epochs):
	#Train a neural network
	network, sum_errors = build_train_model(n_hidden, l_rate, epochs)
	#load the test dataset
	test_data = normalize_dataset('downgesture_test.list.txt')
	accuracy_table = []
	for row in test_data:
		prediction = predict(network, row)
		#print('Expected=%d, Got=%d' % (row[-1], prediction))
		accuracy_table.append({'expected': row[-1], 'got': prediction})
	test_results = test_stats(accuracy_table)

	print('Accuracy Table')
	for row in accuracy_table:
		print(row)

	print('Accuracy: ', test_results.accuracy)
	print('Precision: ', test_results.precision)
	print('Recall: ', test_results.recall)

	plot_errors(sum_errors)


if __name__ == '__main__':
	main()
