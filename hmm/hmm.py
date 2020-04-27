import numpy as np
import math


def get_data(filename):

	with open(filename) as f:

		lines = f.readlines()

		grid = []
		for line in lines[2:12]:
			line = line.rstrip()
			line = line.split(' ')
			line = [int(i) for i in line]
			grid.append(line)

		towers = []
		for line in lines[16:20]:
			line = line.rstrip()
			line = line.split(' ')
			line = line[2:]
			line = [int(i) for i in line]
			towers.append(line)

		noisy_distance = []
		for line in lines[24:]:
			line = line.rstrip()
			line = line.split(' ')
			line = [float(i) for i in line]
			noisy_distance.append(line)

	return grid, towers, noisy_distance


def trajectory_probability(x_prior, y_prior, x, y, grid):

	# given x_prior,y_prior coordinates check grid and assess how many neighbors are available

	free_neighbors = 0

	if x_prior-1 >= 0 and grid[x_prior-1][y_prior]==1: # moving left is available
		free_neighbors += 1
	if x_prior+1 <= 9 and grid[x_prior+1][y_prior]==1: # moving right is available
		free_neighbors += 1
	if y_prior-1 >= 0 and grid[x_prior][y_prior-1]==1: # moving down is available
		free_neighbors += 1
	if y_prior+1 <= 9 and grid[x_prior][y_prior+1]==1: # moving up is available
		free_neighbors += 1

	# given number of available neighbors, calculate probability robot moved to new coordinates

	if x-1 == x_prior and y == y_prior and grid[x][y] == 1: # probability of moving left
		return 1/free_neighbors
	elif x+1 == x_prior and y == y_prior and grid[x][y] == 1: # probability of moving right
		return 1/free_neighbors
	elif x == x_prior and y == y_prior-1 and grid[x][y] == 1: # probability of moving down
		return 1/free_neighbors
	elif x == x_prior and y == y_prior+1 and grid[x][y] == 1: # probability of moving up
		return 1/free_neighbors
	else:
		return 0


def initial_probability(grid):

	valid_cells = {}

	counter = 1
	x = 0
	y = 0
	for row in grid:
		for column in row:
			# if the cell is valid, create a dict where key = 1-87 free spaces and value = (x,y) coordinates
			if column == 1:
				valid_cells[counter] = [x, y]
				counter += 1
				y += 1
			elif column == 0:
				y += 1
		y = 0
		x += 1

	# there are 87 valid cells to start. robot has uniform probability of starting in each of them
	init_prob = 1/len(valid_cells)

	return valid_cells, init_prob


def transition_matrix(valid_cells, grid):

	# initialize 87x87 matrix to represent 87 free cells

	trans_matrix = np.zeros((len(valid_cells), len(valid_cells)))

	for row in range(len(valid_cells)):
		for column in range(len(valid_cells)):
			# position of x_prior and y_prior at time t-1
			x_prior, y_prior = valid_cells[row + 1]
			# position of x, y at time t
			x, y = valid_cells[column + 1]
			# fill matrix with probability of every possible prior moving to every possible new
			trans_matrix[row][column] = trajectory_probability(x_prior,y_prior, x, y, grid)

	return trans_matrix


def dist_to_tower_range(valid_cells, towers):

	# euclidean distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)

	min_rand_noise = 0.7
	max_rand_noise = 1.3

	dist_tower_range = {}
	for cell in valid_cells.values():
		coordinates = tuple(cell)
		dist_list = []
		for tower in towers:
			dist = math.sqrt((cell[0] - tower[0])**2 + (cell[1] - tower[1])**2)
			# include min and max random noise to generate possible range
			random_noise = [round(dist * min_rand_noise,1), round(dist * max_rand_noise,1)]
			dist_list.append(random_noise)
		# dict with 87 values. key = free cells and values = list of ranges to 4 towers
		dist_tower_range[coordinates] = dist_list

	return dist_tower_range


def likely_free_cells(valid_cells, noisy_distance, dist_tower_range):

	coord_list = []
	for key in valid_cells.keys():
		coordinates = tuple(valid_cells[key])
		count = 0
		for i in range(0,4):
			# for each free cell and for each tower, check if noisy distance recorded is in valid range
			tower = noisy_distance[i]
			min = dist_tower_range[coordinates][i][0]
			max = dist_tower_range[coordinates][i][1]
			if min <= tower[i] <= max:
				count += 1
		if count == 4:
			# if coordinate is in valid range of tower distances, append
			coord_list.append(coordinates)

	return coord_list


# read txt file into data structures
grid, towers, noisy_distance = get_data('hmm-data.txt')

# probability that if robot start at (0,0) robot will move to (0,1)
probability = trajectory_probability(0, 0, 0, 1, grid)

# probability of robot's initial position in any 1 cell
valid_cells, init_prob = initial_probability(grid)

# given previous cell, what is conditional probability robot moved to another cell
transition_matrix(valid_cells, grid)

# distance from every free cell to tower with random noise element
dist_tower_range = dist_to_tower_range(valid_cells, towers)

# given noisy distance, which cells are most probable during each 11 time-steps
likely_cells = {}
for i in range(0, 11):
	coordinates = likely_free_cells(valid_cells, noisy_distance, dist_tower_range)
	try:
		likely_cells[i].append(coordinates)
	except KeyError:
		likely_cells[i] = [coordinates]