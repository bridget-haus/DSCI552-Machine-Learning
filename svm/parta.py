import numpy as np
import cvxopt
import quadprog
import matplotlib.pyplot as plt
import sys


class SVM:

    def __init__(self, filename):
        self.dat = np.loadtxt(filename,dtype='float',delimiter=',')
        self.X = self.dat[:,0:2]
        self.Y = self.dat[:,2]
        self.rows, self.cols = self.X.shape
        self.hyperplane()
        self.support_vectors()
        self.weights()
        self.plot()
        # Need to get the intercept
        #self.kernel_function() --> create a kernel function for non linear data
        
    def hyperplane(self):
        '''Identify each datapoint's alpha values'''
        self.xixj = np.zeros((self.rows, self.rows))
        self.yiyj = np.zeros((self.rows, self.rows))
        
        for i in range(self.rows):
            for j in range(self.rows):
                self.xixj[i, j] = np.dot(self.X[i], self.X[j])
                self.yiyj[i, j] = np.dot(self.Y[i], self.Y[j])

        P = self.yiyj * self.xixj
        Q = np.ones(self.rows) * -1
        G = np.diag(np.ones(self.rows) * -1)
        H = np.zeros(self.rows)
        A = self.Y
        B = 0.0
        self.alphas = self.cvxopt_solve_qp(P, Q, G, H, A, B)
        print('\n alphas are:')
        print(self.alphas)
        
    def cvxopt_solve_qp(self, P, Q, G=None, h=None, A=None, b=None):
        '''Find alphas that satisfy the QP optimization problem.'''
        args = [cvxopt.matrix(P), cvxopt.matrix(Q)]
        if G is not None:
            args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
            if A is not None:
                args.extend([cvxopt.matrix(A, (1,100)), cvxopt.matrix(b)])
        sol = cvxopt.solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        return np.array(sol['x']).reshape((P.shape[1],))
    
    def support_vectors(self):
        '''Identify the support vectors that are non-zero points'''
        self.sv_index = np.where(self.alphas>0.00001)[0]
        self.sv_x = self.X[self.sv_index]
        self.sv_y = self.Y[self.sv_index]
        print('\n Support Vectors are:')
        print('X: \n', self.sv_x)
        print('Y: \n',self.sv_y)

    def weights(self):
        '''Derive weights from all alphas
        Dervice bias b, from weights and support vectors'''
        self.weights = np.sum(self.alphas[self.sv_index][:, np.newaxis] * self.sv_y[:, np.newaxis] * self.sv_x, 0)[:, np.newaxis]
        print(f'weights: \n {self.weights}')
        sv_grid = (self.alphas > 0.00001).flatten()
        b = self.Y[sv_grid][np.newaxis].T - np.dot(self.X[sv_grid], self.weights)
        print('b:', b[0])
    
    def plot(self):
        '''Plot the support vectors'''
        plt.scatter(self.X[:,0],self.X[:,1],c=self.Y)         
        plt.scatter(self.sv_x[:,0],self.sv_x[:,1], marker = 'D')
        plt.show()
        
    def _str__(self):
        return 'done'

filename = 'linsep.txt'
p = SVM(filename)
print(p)