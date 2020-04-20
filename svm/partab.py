import numpy as np
import cvxopt
import matplotlib.pyplot as plt
from numpy import linalg as LA


def main():

    print('Do you want to run this for linear or non-linear data?')
    response = input()
    if response == 'linear':
        filename = 'linsep.txt'
        p = SVM(filename, response)
    elif response == 'non-linear':
        filename = 'nonlinsep.txt'
        p = SVM(filename, response)
    else:
        print('please choose linear or non-linear')


class SVM:

    def __init__(self, filename, response):
        self.dat = np.loadtxt(filename,dtype='float',delimiter=',')
        self.X = self.dat[:,0:2]
        self.Y = self.dat[:,2]
        self.rows, self.cols = self.X.shape
        self.omiga = 0.01
        self.gamma = .01
        if response == 'linear':
            self.Q()
        else:
            self.Q_kernel()
        self.hyperplane()
        self.support_vectors()
        if response == 'linear':
            self.weights()
        self.plot(response, self.weights)

        
    def Q(self):
        self.xixj = np.zeros((self.rows, self.rows))
        self.yiyj = np.zeros((self.rows, self.rows))
        
        for i in range(self.rows):
            for j in range(self.rows):
                self.xixj[i, j] = np.dot(self.X[i], self.X[j])
                self.yiyj[i, j] = np.dot(self.Y[i], self.Y[j])
        self.Q = self.yiyj * self.xixj
    
    def Q_kernel(self):
        self.xixj = np.zeros((self.rows, self.rows))
        self.yiyj = np.zeros((self.rows, self.rows))
        for i in range(self.rows):
            for j in range(self.rows):
                self.xixj[i, j] = np.exp(self.gamma*-LA.norm(np.subtract(self.X[i], self.X[j]))** 2)
                self.yiyj[i, j] = np.dot(self.Y[i], self.Y[j])
        self.Q = self.yiyj * self.xixj 
        
    def hyperplane(self):           
        P = self.Q
        q = np.ones(self.rows) * -1
        G = np.diag(np.ones(self.rows) * -1)
        h = np.zeros(self.rows)
        A = self.Y
        b = 0.0
        self.alphas = self.cvxopt_solve_qp(P, q, G, h, A, b)
        print('\n alphas are:')
        print(self.alphas)


    def cvxopt_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
    #     P = .5 * (P + P.T)  # make sure P is symmetric
        args = [cvxopt.matrix(P), cvxopt.matrix(q)]
        if G is not None:
            args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
            if A is not None:
                args.extend([cvxopt.matrix(A, (1,100)), cvxopt.matrix(b)])
        sol = cvxopt.solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        return np.array(sol['x']).reshape((P.shape[1],))


    def support_vectors(self):
        self.sv_index = np.where(self.alphas>0.00001)[0]
        self.sv_x = self.X[self.sv_index]
        self.sv_y = self.Y[self.sv_index]
        print('\n Support Vectors are:')
        print(self.sv_x, self.sv_y)
        return(self.sv_index, self.sv_x, self.sv_y)

    def weights(self):
        self.weights = np.sum(self.alphas[self.sv_index][:, np.newaxis] * self.sv_y[:, np.newaxis] * self.sv_x, 0)[:, np.newaxis]
        print("weights:")
        print(self.weights)
        return self.weights

    
    def plot(self, response, weights):
        plt.scatter(self.X[:,0],self.X[:,1],c=self.Y)
        plt.scatter(self.sv_x[:,0],self.sv_x[:,1], marker = 'D')
        if response == 'linear':
            #Use the first support vector to get the intercpet
            i = 0 #Each support vector should yield the same intercept i = [0,1,2] if three support vectors
            self.intercept = self.Y[self.sv_index][i]-np.dot(self.weights.T, self.X[self.sv_index][i])
            print("intercept:")
            print(self.intercept)
            x_values = np.linspace(0 , 1, 5) # Get a random array of x values in the plot window
            y_values = []
            for x in x_values:
                y = self.weights[0, 0] * x / self.weights[1, 0] + self.intercept
                y_values.append(-y)
            plt.plot(x_values, y_values)
            m = -weights[0] / weights[1]
            b = -self.intercept / weights[1]

            print(f'y = {m[0]}x + {b[0]}')

        plt.show()
        
    def _str__(self):
        return 'done'


if __name__ == '__main__':
    main()
