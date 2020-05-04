#REFERENCE 
# SLIDES: http://www.cs.jhu.edu/~langmea/resources/lecture_notes/hidden_markov_models.pdf
# SOURCE CODE: https://nbviewer.jupyter.org/gist/BenLangmead/7460513

import numpy as np
import math

class HMM(object):
    ''' Simple Hidden Markov Model implementation.  User provides
        transition, emission and initial probabilities in dictionaries
        mapping 2-character codes onto floating-point probabilities
        for those table entries.  States and emissions are represented
        with single characters.  Emission symbols comes from a finite.  '''
    
    def __init__(self, A, E, I):
        ''' Initialize the HMM given transition, emission and initial
            probability tables. '''
        
        # put state labels to the set self.Q
        self.Q, self.S = set(), set() # states and symbols
        for a, prob in A.items():
            asrc, adst = a.split('|')
            self.Q.add(asrc)
            self.Q.add(adst)
            
        # add all the symbols to the set self.S
        for e, prob in E.items():
            eq, es = e.split('|')
            self.Q.add(eq)
            self.S.add(es)
        
        self.Q = sorted(list(self.Q))
        self.S = sorted(list(self.S))
        
        # create maps from state labels / emission symbols to integers
        # that function as unique IDs
        qmap, smap = {}, {}
        for i in range(len(self.Q)): qmap[self.Q[i]] = i
        for i in range(len(self.S)): smap[self.S[i]] = i
        lenq = len(self.Q)
        
        # create and populate transition probability matrix
        self.A = np.zeros(shape=(lenq, lenq), dtype=float)
        for a, prob in A.items():
            asrc, adst = a.split('|')
            self.A[qmap[asrc], qmap[adst]] = prob
        # make A stochastic (i.e. make rows add to 1)
        self.A /= self.A.sum(axis=1)[:, np.newaxis]
        
        # create and populate emission probability matrix
        self.E = np.zeros(shape=(lenq, len(self.S)), dtype=float)
        for e, prob in E.items():
            eq, es = e.split('|')
            self.E[qmap[eq], smap[es]] = prob
        # make E stochastic (i.e. make rows add to 1)
        self.E /= self.E.sum(axis=1)[:, np.newaxis]
        
        # initial probabilities
        self.I = [ 0.0 ] * len(self.Q)
        for a, prob in I.items():
            self.I[qmap[a]] = prob
        # make I stochastic (i.e. adds to 1)
        self.I = np.divide(self.I, sum(self.I))
        
        self.qmap, self.smap = qmap, smap
        
    def viterbi(self, x):
        ''' Given sequence of emissions, return the most probable path
            along with its probability. '''
        x = list(map(self.smap.get, x)) # turn emission characters into ids
        nrow, ncol = len(self.Q), len(x)
        mat   = np.zeros(shape=(nrow, ncol), dtype=float) # prob
        matTb = np.zeros(shape=(nrow, ncol), dtype=int)   # backtrace
        # Fill in first column
        for i in range(0, nrow):
            mat[i, 0] = self.E[i, x[0]] * self.I[i]
        # Fill in rest of prob and Tb tables
        for j in range(1, ncol):
            for i in range(0, nrow):
                ep = self.E[i, x[j]]
                mx, mxi = mat[0, j-1] * self.A[0, i] * ep, 0
                for i2 in range(1, nrow):
                    pr = mat[i2, j-1] * self.A[i2, i] * ep
                    if pr > mx:
                        mx, mxi = pr, i2
                mat[i, j], matTb[i, j] = mx, mxi
        # Find final state with maximal probability
        omx, omxi = mat[0, ncol-1], 0
        for i in range(1, nrow):
            if mat[i, ncol-1] > omx:
                omx, omxi = mat[i, ncol-1], i
        # Backtrace
        i, p = omxi, [omxi]
        for j in range(ncol-1, 0, -1):
            i = matTb[i, j]
            p.append(i)
        p = ''.join(map(lambda x: self.Q[x], p[::-1]))
        return omx, p # Return probability and path

def main():
	#Feed in transition, emission, and intiail probabilities
	hmm = HMM({"F|F":0.9, "F|L":0.1, "L|F":0.1, "L|L":0.9}, # transition matrix A
	          {"F|H":0.5, "F|T":0.4, "L|H":0.75, "L|T":0.15}, # emission matrix E
	          {"F":0.5, "L":0.5}) # initial probabilities I

	# Now we experiment with viterbi decoding
	obs_heads_tails = "TTTTTHHHHHTTTHHHHHTHTHTHHHHHHHH"
	print("Heads/Tails observations: ", obs_heads_tails, '\n')

	jprobOpt, path = hmm.viterbi(obs_heads_tails)
	print("Best path probability: ", jprobOpt, '\n')
	print("Best path: ", path, '\n')

if __name__ == "__main__": 
	main()


