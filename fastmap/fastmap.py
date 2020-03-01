import random


def create_dicts(word_text, data_text):

	with open(word_text) as f:
		lines = f.readlines()
		counter = 1
		wordlist_dict = {}
		for line in lines:
			line = line.rstrip()
			wordlist_dict[line] = counter
			counter += 1

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


def distance_function(object1, object2):

	if object1 == object2:
		return 0
	else:
		return data_dict[object1][object2]


def furthest_function(current_point, prev_point):

	d = data_dict[current_point]
	max_value = max(d.values())
	for k, v in d.items():
		if v == max_value:
			if k == prev_point:
				return current_point, k
			return furthest_function(k, current_point)


def fastmap(k, current_step, coordinate_map):

	obj_a, obj_b = furthest_function(random.randint(1, len(wordlist_dict)), -1)
	current_coordinates = []
	for obj_i in wordlist_dict.values():
		dai = distance_function(obj_a, obj_i)
		dab = distance_function(obj_a, obj_b)
		dib = distance_function(obj_i, obj_b)
		for dimension in coordinate_map:
			dai -= (dimension[obj_a - 1] - dimension[obj_i - 1])**2
		xi = (dai**2 + dab**2 - dib**2) / (2*dab)
		current_coordinates.append(xi)
	coordinate_map.append(current_coordinates)
	print(coordinate_map)


wordlist_dict, data_dict = create_dicts('fastmap-wordlist.txt', 'fastmap-data.txt')

fastmap(2, 1, [])