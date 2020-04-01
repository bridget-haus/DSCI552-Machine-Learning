from PIL import Image
import random
from math import exp


def main():

	dataset = normalize_dataset('downgesture_train.list.txt')

	hidden_layer, output_layer = initialize_weights(2, 1, 2)
	print(hidden_layer)
	print(output_layer)
	inputs = dataset[0]
	_, output = feed_forward(hidden_layer, output_layer, inputs)
	for o in output:
		print(o['output'])


# iterate through training list and create dataset of pixels
def normalize_dataset(file):

	with open(file, 'r') as train_data:
		lines = train_data.read().splitlines()
		dataset = []
		for line in lines:
			img = Image.open(line)
			grey_img = img.convert(mode='L')
			pix_val = list(grey_img.getdata())
			# normalize pixels to values between 0 and 1
			for i in range(len(pix_val)):
				pix_val[i] = pix_val[i]/255
			# append label to pixel dataset
			if 'down' in line:
				pix_val.append(1)
			else:
				pix_val.append(0)
			dataset.append(pix_val)

	return dataset


# setup NN with random weights for given input, hidden and output layers
def initialize_weights(inputs, hidden, outputs):
	hidden_layer = []
	input_weights = []
	# initialize random weight for each input to connect to hidden + bias
	for i in range(inputs + 1):
		input_weights.append(random.uniform(-.01, .01))
	for i in range(hidden):
		weights = {'w': input_weights}
		hidden_layer.append(weights)
	output_layer = []
	output_weights = []
	# initialize random weight for each hidden to connect to output + bias
	for i in range(hidden + 1):
		output_weights.append(random.uniform(-.01, .01))
	for i in range(outputs):
		weights = {'w': output_weights}
		output_layer.append(weights)

	return hidden_layer, output_layer


def summation(inputs, weights):

	# sum_wx = sum(w1x1 + w2x2 + ... + wdxd) + bias
	# w = weight, x = input
	sum_wx = weights[-1]  # last value in weights is bias
	for i in range(len(weights)-1):
		sum_wx += (inputs[i] * weights[i])
	return sum_wx


def sigmoid(sum_wx):

	# S(x) = 1/(1 + e^(-x))
	return 1.0 / (1.0 + exp(-sum_wx))


# Push inputs through network to generate output
def feed_forward(hidden_layer, output_layer, inputs):

	# for each perceptron, we calculate output by taking the sigmoid function of the sum of inputs
	for h_perceptron in hidden_layer:
		sum_wx = summation(inputs, h_perceptron['w'])
		h_perceptron['output'] = sigmoid(sum_wx)
	for o_perceptron in output_layer:
		sum_wx = summation(inputs, o_perceptron['w'])
		o_perceptron['output'] = sigmoid(sum_wx)

	return hidden_layer, output_layer

# standard squared error = sum [(label - prediction)^2]


if __name__ == '__main__':
	main()
