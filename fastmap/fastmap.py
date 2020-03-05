import random
import math
import matplotlib.pyplot as plt


def main():

	wordlist_dict, data_dict = create_dicts('fastmap-wordlist.txt', 'fastmap-data.txt')
	coordinate_map = fastmap(2, [], wordlist_dict, data_dict)
	print(coordinate_map)
	plot_words(coordinate_map, wordlist_dict)


def create_dicts(word_text, data_text):

	# read fastmap-wordlist into dictionary
	with open(word_text) as f:
		lines = f.readlines()
		counter = 1
		wordlist_dict = {}
		for line in lines:
			line = line.rstrip()
			wordlist_dict[line] = counter
			counter += 1

	# read fastmap-data into dictionary
	with open(data_text) as f:
		lines = f.read().splitlines()
		data_dict = {}
		for line in lines:
			line = line.split('\t')
			line = [int(i) for i in line]
			try:
				inner_dict1 = data_dict[line[0]]
			except KeyError:
				data_dict[line[0]] = {}
				inner_dict1 = data_dict[line[0]]
			try:
				inner_dict2 = data_dict[line[1]]
			except KeyError:
				data_dict[line[1]] = {}
				inner_dict2 = data_dict[line[1]]

			inner_dict1[line[1]] = line[2]
			inner_dict2[line[0]] = line[2]

	return wordlist_dict, data_dict


def distance_function(object1, object2, coordinate_map, data_dict):

	# the distance between an object and itself is 0
	if object1 == object2:
		return 0, object2

	# original distance comes from the data fastmap-data text file
	old_distance = data_dict[object1][object2]
	subtractors = 0

	# (xi - xj)^2
	for xi in coordinate_map:
		subtractors += (xi[object1 - 1] - xi[object2 - 1]) ** 2

	# new distance = sqrt(old_distance^2 - (xi - xj)^2)
	distance = old_distance**2 - subtractors
	distance = math.sqrt(distance)

	return distance, object2


def furthest_function(current_point, prev_point, coordinate_map, data_dict):

	# for a random point, create a list of every distance between it and all other points
	distances = []
	for i in range(1, len(data_dict) + 1):
		distances.append(distance_function(current_point, i, coordinate_map, data_dict))

	# take the max distance from the random point
	max_value = max(distances)
	if max_value[1] == prev_point:
		# if you've found the farthest pair, stop
		return current_point, max_value[1]
	# if you haven't found the farthest pair, find the furthest point from the last max
	return furthest_function(max_value[1], current_point, coordinate_map, data_dict)


def fastmap(k, coordinate_map, wordlist_dict, data_dict):

	# execute for only k dimensions
	if len(coordinate_map) == k:
		return coordinate_map

	# obj_a and obj_b are the furthest points
	obj_a, obj_b = furthest_function(random.randint(1, len(wordlist_dict)), -1, coordinate_map, data_dict)
	current_coordinates = []
	# for every other obj_i, calculate xi formula
	for obj_i in wordlist_dict.values():
		dai, _ = distance_function(obj_a, obj_i, coordinate_map, data_dict)
		dab, _ = distance_function(obj_a, obj_b, coordinate_map, data_dict)
		dib, _ = distance_function(obj_i, obj_b, coordinate_map, data_dict)
		xi = (dai**2 + dab**2 - dib**2) / (2*dab)
		# make xi the first coordinate for each word
		current_coordinates.append(xi)
	coordinate_map.append(current_coordinates)
	# recurse
	coordinate_map = fastmap(k, coordinate_map, wordlist_dict, data_dict)
	return coordinate_map


def plot_words(coordinate_map, wordlist_dict):

	# plot x and y coordinates with word annotations
	for index in range(len(wordlist_dict)):
		x = coordinate_map[0][index]
		y = coordinate_map[1][index]
		for k, v in wordlist_dict.items():
			if v == index + 1:
				plt.annotate(k, (x, y))
				plt.scatter(x, y, color='blue')

	plt.show()


if __name__ == "__main__":
    main()