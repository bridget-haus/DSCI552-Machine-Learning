
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


def free_neighbors(x, y, grid):

	# given x,y coordinate check grid and asses how many neighbors are available

	free_neighbors = 0

	if x-1 >= 0 and grid[x-1][y]==1:
		free_neighbors += 1
	if x+1 <= 10 and grid[x+1][y]==1:
		free_neighbors += 1
	if y-1 >= 0 and grid[x][y-1]==1:
		free_neighbors += 1
	if y+1 <= 10 and grid[x][y+1] ==1:
		free_neighbors += 1

	print(free_neighbors)


grid, towers, noisy_distance = get_data('hmm-data.txt')
free_neighbors(3, 3, grid)
