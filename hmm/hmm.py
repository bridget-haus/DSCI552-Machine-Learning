import numpy as np


def main():

	# read txt file into data structures
	grid, towers, noisy_distance = get_data('hmm-data.txt')

	# probability that if robot start at (0,0) robot will move to (0,1)
	probability = trajectory_probability(0, 0, 0, 1, grid)

	# probability of robot's initial position in any 1 cell
	valid_cells, init_prob = initial_probability(grid)

	# given previous cell, what is conditional probability robot moved to another cell
	trans_matrix = transition_matrix(valid_cells, grid)


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

	# given number of available neighbors, calculate probability you moved to new coordinates

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

	trans_matrix = np.zeros((len(valid_cells), len(valid_cells)))

	for row in range(len(valid_cells)):
		for column in range(len(valid_cells)):
			x_prior, y_prior = valid_cells[row + 1]
			x, y = valid_cells[column + 1]
			trans_matrix[row][column] = trajectory_probability(x_prior,y_prior, x, y, grid)

	return trans_matrix


if __name__ == 'main':
	main()

