
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


get_data('hmm-data.txt')
