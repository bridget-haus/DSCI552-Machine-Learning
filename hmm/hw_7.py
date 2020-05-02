import math 
import numpy as np
from scipy.spatial import distance
import pandas as pd
import random
def tower_distance(state, towers):
    distance_array = np.zeros(4)
    (point1, point2) = state
    for index, tower in enumerate(towers):
        z = (point1 - tower[0])**2 + (point2 - tower[1])**2
        euclidean_dist = math.sqrt(z)
        distance_array[index] = euclidean_dist
    return distance_array
        

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
        
        grid = np.matrix(grid)
#         towers = np.matrix(towers)
        noisy_distance = np.matrix(noisy_distance)

    return grid, towers, noisy_distance


def get_states_array(grid):
    """ return the position of the free states, eliminate all states that are an obsticle"""
    states = []
    (xlen,ylen) = grid.shape
    for x in range(xlen):
        for y in range(ylen):
            if grid.item(x,y) == 1:
                states.append((x,y))
    return states

def get_emission_probability(states, towers, observations):
    """Return all the possible positions that is determined from the euclidean distances from the towers
    the state has to be within all observable range of each tower.
    we have 10 quasi-known stats. We want to determine all the possible locations for the 10 instances
    iterate through each state for each each oberservable instance and assign a 1 if it is an eligible state, a 0 otherwise
    """
    emissions = np.zeros((len(observations), len(states)))
    i = 0
    k = 0
    for observation in observations: 
        j = 0
        for state in states:
            distances = tower_distance(state,towers)
            if (distances[0] <= observation.item(0)/0.7 and distances[0] >= observation.item(0) /1.3):
                if (distances[1] <= observation.item(1) / 0.7 and distances[1] >= observation.item(1) / 1.3):
                    if (distances[2] <= observation.item(2) / 0.7 and distances[2] >= observation.item(2) / 1.3):
                        if (distances[3] <= observation.item(3) / 0.7 and distances[3] >= observation.item(3) / 1.3):
                            emissions[i][j] = 1
#                             print(i,j)
            j += 1
        i += 1
    
    return emissions

def get_transition_probabilities(states_array, emission_matrix):
    """
    get the probability for each state going to each state 
    check to see is the neighboring states are free or blocked
    count all the free spaces and get the probabilities
    the probability is determined from one state to another
            ex. (0,0) to (0,1) is 50%
                b/c from (0,0) it can for to either (1,0) and (0,1)
    store in a dictionary where each key is str((0,0),(0,1)) (i.e from_state,to_state)
    Completely fill out the dictionary with 0's for all non-neighboring states
    """
    transition_probability = np.zeros((len(states_array), len(states_array)))
    propability_dictionary = {}
    probability_dictionary = {}
    for index,state in enumerate(states_array):
#         print(index,state)
        count = 0
        if (state[0]+1, state[1]) in states_array:
            count += 1
        if (state[0], state[1]+1) in states_array:
            count += 1
        if (state[0]-1, state[1]) in states_array:
            count += 1
        if (state[0], state[1]-1) in states_array:
            count += 1
        probability = 1/count
        propability_dictionary[str(state)] = []
        if state == (0,0):
            probability_dictionary[str(state)+str(state)] = 0
        if (state[0]+1, state[1]) in states_array:
            probability_dictionary[str(state)+str((state[0]+1, state[1]))] = probability
        else:
            probability_dictionary[str(state)+str((state[0]+1, state[1]))] = 0
            
        if (state[0], state[1]+1) in states_array:
            probability_dictionary[str(state)+str((state[0], state[1]+1))] = probability
        else:
            probability_dictionary[str(state)+str((state[0], state[1]+1))] = 0
            
        if (state[0]-1, state[1]) in states_array:
            probability_dictionary[str(state)+str((state[0]-1, state[1]))] = probability
        else:
            probability_dictionary[str(state)+str((state[0]-1, state[1]))] = 0
            
        if (state[0], state[1]-1) in states_array:
            probability_dictionary[str(state)+str((state[0], state[1]-1))] = probability
        else:
            probability_dictionary[str(state)+str((state[0], state[1]-1))] = 0

    for state in states_array:
        for state2 in states_array:
            key = str(state)+str(state2)
            if key not in probability_dictionary:
                probability_dictionary[key] = 0
        
    return probability_dictionary



def viterbi(noisy_distance, states, emission_matrix, transition_probability):
    viterbi_matrix = np.zeros((len(noisy_distance), len(states)))
    viterbi_matrix_previous = np.zeros((len(noisy_distance), len(states)))
    viterbi_matrix_previous = viterbi_matrix_previous.astype(int)
    state_steps = len(states)
    observation_steps = len(noisy_distance)
    counter = 0
    for i in emission_matrix[0]: #get the probability of being at one of the initial positions at state 1
        if i == 1:
            counter += 1
    for i in range(state_steps): # 0 to 87
        viterbi_matrix[0][i] = 1/counter * emission_matrix[0][i] 

    for observation_index in range(observation_steps): # [1, 11]
        if observation_index > 0:
            for current_state_index in range(state_steps): # [0, 87]
                maximum_probability = 0
                for previous_state_index in range(len(states)): # [0, 87]
                    state_to_state_probability = viterbi_matrix[observation_index - 1][previous_state_index] * transition_probability[str(states[current_state_index]) + str(states[previous_state_index])] * emission_matrix[observation_index - 1][previous_state_index] 
                    if state_to_state_probability > maximum_probability:
                        maximum_probability = state_to_state_probability
                        viterbi_matrix[observation_index][current_state_index] = maximum_probability
                        viterbi_matrix_previous[observation_index][current_state_index] = previous_state_index

    cols = []
    for state in states:
        cols.append(str(state))

    print('Viterbi Matrix')
    df2 = pd.DataFrame(np.matrix(viterbi_matrix),columns=cols)
    print(df2)
    # for row in viterbi_matrix:
    #         print(np.array(row))

    temporary_maximum_probability = -1.0
    maximum_state = ()
    final_optimal_path_of_robot = []
    previous_state_index = 0

    for state_index,probability_value in enumerate(viterbi_matrix[-1]):
        if probability_value > temporary_maximum_probability:
            temporary_maximum_probability = probability_value
            maximum_state = states[state_index]
            previous_state_index = state_index

    final_optimal_path_of_robot.append(maximum_state)

    for t in range(len(viterbi_matrix) - 1, 0, -1):
        previous_state_index = viterbi_matrix_previous[t][previous_state_index]
        final_optimal_path_of_robot.insert(0,states[previous_state_index])

    print ("Most likely states at each oberservation:")
    print(final_optimal_path_of_robot)
    
    
def main():
    filename = 'hmm-data.txt'
    grid, towers, noisy_distance = get_data(filename) 
    states = get_states_array(grid)
    emissions_matrix = get_emission_probability(states, towers, noisy_distance)
    # print(emissions_matrix)
    probability_dictionary = get_transition_probabilities(states,emissions_matrix)
    # print(probability_dictionary)

    viterbi(noisy_distance,states,emissions_matrix,probability_dictionary)

if __name__ == "__main__": 
        main()

