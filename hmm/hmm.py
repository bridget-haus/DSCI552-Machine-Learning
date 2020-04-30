import numpy as np
import math
import pandas as pd

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

def initial_probability(grid):
    '''Calculate probability of robot's initial position in any 1 cell.'''
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
    #Make a matrix with initial probability for each valid cell
    init_prob = round((1/len(valid_cells)), 5)
    I = {}
    for key in valid_cells.keys():
        I[key] = init_prob 
    return valid_cells, init_prob, I

def get_trans_neighbors(valid_cells, grid):
    '''For each valid cell, get the id and coordinates of its neighbor.
    This structure will be used to populate the trans_matrix.'''
    
    #Use this for looking up neighbor cell id
    validcells_keys = list(valid_cells.keys())
    validcells_vals = list(valid_cells.values())
    
    free_neighbors_outer = {}
    for key in valid_cells.keys():
        
        free_neighbors_inner = {}

        x_prior = valid_cells[key][0]
        y_prior = valid_cells[key][1]

        if x_prior-1 >= 0 and grid[x_prior-1][y_prior]==1: # moving left is available
            nb_id = validcells_keys[validcells_vals.index([x_prior-1, y_prior])]
            free_neighbors_inner[nb_id] = [x_prior-1, y_prior]
        if x_prior+1 <= 9 and grid[x_prior+1][y_prior]==1: # moving right is available
            nb_id = validcells_keys[validcells_vals.index([x_prior+1, y_prior])]
            free_neighbors_inner[nb_id] = [x_prior+1, y_prior]
        if y_prior-1 >= 0 and grid[x_prior][y_prior-1]==1: # moving down is available
            nb_id = validcells_keys[validcells_vals.index([x_prior, y_prior-1])]
            free_neighbors_inner[nb_id] = [x_prior, y_prior-1]
        if y_prior+1 <= 9 and grid[x_prior][y_prior+1]==1: # moving up is available
            nb_id = validcells_keys[validcells_vals.index([x_prior, y_prior+1])]
            free_neighbors_inner[nb_id] = [x_prior, y_prior+1]

        free_neighbors_outer[key] = free_neighbors_inner
    return free_neighbors_outer

def transition_matrix(free_neighbors):
    '''Create a constant transition matrix, which are probabilities of 
    transitioning to new state conditioned on hidden state.
    Rows - prior valid cells, ascending from 1-87. Row values sum to 1.
    Columns - current valid cells, ascending from 1-87.
    Ex: Row one is for cell coordinates 0,1. Row two is for cell coordinates 0,1... etc.
    '''
    # initialize 87x87 matrix to represent 87 free cells
    trans_matrix = np.zeros((len(valid_cells), len(valid_cells)))

    for key_out in free_neighbors.keys():
        vals_outer = free_neighbors[key_out]

        neighbor_count = len(vals_outer.keys())
        prior_prob = round((1 / neighbor_count), 5)
        
        #Fill out the trans matrix, make sure rows sum to 1
        for key_in in vals_outer:
            trans_matrix[key_out -1][key_in-1] = prior_prob
    return trans_matrix

def dist_to_tower_range(valid_cells, towers):

    # euclidean distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)

    min_rand_noise = 0.7
    max_rand_noise = 1.3

    dist_tower_range = pd.DataFrame()

    coordinates_list = valid_cells.values()
    dist_tower_range['coordinates'] = coordinates_list
    
    towers_e = enumerate(towers, 1)
    for i, tower in towers_e:
        tower_dists = []
        for cell in coordinates_list:
            dist = math.sqrt((cell[0] - tower[0])**2 + (cell[1] - tower[1])**2)
            # include min and max random noise to generate possible range
            random_noise = [round(dist * min_rand_noise,1), round(dist * max_rand_noise,1)]
            tower_dists.append(random_noise)
        dist_tower_range[i] = tower_dists

    return dist_tower_range

def find_obs_likely_cells(noisy_distance, dist_tower_range, valid_cells):
    '''For each observation distances, find a set of likely cell indexes from the 87 valid cells.
    Use each observation's likely_cells to compute emissions matrix later on.'''
    obs_likely_cells = {}
    for i, obs in enumerate(noisy_distance, 1):
        obs_i = enumerate(obs, 1)

        #Select the coordinates of all valid cells
        tow_coords = dist_tower_range.loc[:, 1]
        tow_indexes = list(tow_coords.index.values)

        #For each observation's tower_dist
        for j, tower_dist in obs_i:
            #find cells that that contain the observation's tower_dist
            tower_matches = {}
            for index, coord in zip(tow_indexes, tow_coords):
                if tower_dist >= coord[0] and tower_dist <= coord[1]:
                    #Save the index and coordinate of the likely cell
                    tower_matches[index] = valid_cells.get(index)

            #When the last tower distances are checked for matches, don't update the tow_coords
            try: 
                tow_coords_next = dist_tower_range.loc[list(tower_matches.keys()), j+1]
            except KeyError:
                break

            #Narrow down the likely tower coordinates, before checking the next tower
            tow_coords = tow_coords_next
            tow_indexes = list(tow_coords.index.values)
        obs_likely_cells[i] = tower_matches
    return obs_likely_cells

# read txt file into data structures
grid, towers, noisy_distance = get_data('hmm-data.txt')

# probability of robot's initial position in any 1 cell
valid_cells, init_prob, I = initial_probability(grid)

#Get coordinates of free neighbors for each valid cell
free_neighbors = get_trans_neighbors(valid_cells, grid)

# given previous cell, what is conditional probability robot moved to another cell
trans_matrix = transition_matrix(free_neighbors)

# distance from every free cell to tower with random noise element
dist_tower_range = dist_to_tower_range(valid_cells, towers)

# given noisy distance, which cells are most probable during each 11 time-steps
obs_likely_cells = find_obs_likely_cells(noisy_distance, dist_tower_range, valid_cells)