
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

	# given number of available neighbors, calculate probability you moved to new coordiantes

	if x-1 == x_prior and y == y_prior and grid[x][y] == 1: # probability of moving left
		print(1/free_neighbors)
	elif x+1 == x_prior and y == y_prior and grid[x][y] == 1: # probability of moving right
		print(1/free_neighbors)
	elif x == x_prior and y == y_prior-1 and grid[x][y] == 1: # probability of moving down
		print(1/free_neighbors)
	elif x == x_prior and y == y_prior+1 and grid[x][y] == 1: # probability of moving up
		print(1/free_neighbors)
	else:
		print(0)


grid, towers, noisy_distance = get_data('hmm-data.txt')

# this is the proabbility that if you start at (0,0) you will move to (0,1)
trajectory_probability(0, 0, 0, 1, grid)

