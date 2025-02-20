import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn import svm
import math


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=100.0):
   # print(-linalg.norm(x-y)**2)
    x=np.asarray(x)
    y=np.asarray(y)
    return np.exp((-linalg.norm(x-y)**2) / (2 * (sigma ** 2)))


from cvxopt import matrix
class HYP_SVM(object):

    def __init__(self, kernel=gaussian_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
    def m_func(self, X_train,X_test, y):
        n_samples, n_features = X_train.shape[0],X_train.shape[1]
        nt_samples, nt_features= X_test.shape[0],X_test.shape[1]
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i,j] = gaussian_kernel(X_train[i], X_train[j])
               # print(K[i,j])
        X_train=np.asarray(X_train)
        X_test=np.asarray(X_test)
        K1 = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K1[i,j] = gaussian_kernel(X_train[i], X_train[j])
               # print(K[i,j])
        print(K1.shape)
        P = cvxopt.matrix(np.outer(y,y) * self.K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        A = matrix(A, (1,n_samples), 'd') #changes done
        b = cvxopt.matrix(0.0)
        #print(P,q,A,b)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
            
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        print(solution['status'])
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        a_org = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        #print(sv.shape)
        ind = np.arange(len(a))[sv]
        self.a_org=a
        self.a = a[sv]
        self.sv = X_train[sv]
        self.sv_y = y[sv]
        self.sv_yorg=y
        self.kernel = gaussian_kernel
        X_train=np.asarray(X_train)
        b = 0
        for n in range(len(self.a)):
            b += self.sv_y[n]
            b -= np.sum(self.a * self.sv_y * self.K[ind[n],sv])
        b /= len(self.a)
       # print(self.a_org[1])
        #print(self.a_org.shape,self.sv_yorg.shape,K.shape)
        w_phi=0
        total=0
        for n in range(len(self.a_org)):
            w_phi = self.a_org[n] * self.sv_yorg[n] * K1[n] 
        self.d_hyp=np.zeros(n_samples)
        for n in range(len(self.a_org)):
            self.d_hyp += self.sv_yorg[n]*(w_phi+b)
        func=np.zeros((n_samples))
        func=np.asarray(func)
        typ=2
        if(typ==1):
            for i in range(n_samples):
                func[i]=1-(self.d_hyp[i]/(np.amax(self.d_hyp[i])+0.000001))
        beta=0.2
        if(typ==2):
            for i in range(n_samples):
                func[i]=2/(1+beta*self.d_hyp[i])
        #r_max= 0.04
        r_min=1
        self.m=func*r_min
        # #self.m = func
        print(self.m.shape)
        #tempfrac = n_samples + nt_samples
        #self.m=np.append(self.m,func[768:tempfrac]*r_max)
        print(self.m.shape)
        
 ##############################################################################


    def fit(self, X_train,X_test, y):
        self.kernel = gaussian_kernel
        n_samples, n_features = X_train.shape[0],X_train.shape[1] 
        nt_samples, nt_features = X_test.shape[0],X_test.shape[1]
        # Gram matrix

        print(self.K.shape)

        P = cvxopt.matrix(np.outer(y,y) * self.K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        A = matrix(A, (1,n_samples), 'd') #changes done
        b = cvxopt.matrix(0.0)
        #print(P,q,A,b)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
            
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        print(solution['status'])
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        a_org = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        for i in range(n_samples):
            sv=np.logical_or(self.a_org <self.m, self.a_org > 1e-5)
        #print(sv.shape)
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X_train[sv]
        self.sv_y = y[sv]
        #print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * self.K[ind[n],sv])
        self.b /= len(self.a)
        print(self.b)

        # Weight vector
        if self.kernel == gaussian_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else :
            self.w = None        
        
    def project(self, X):
        if self.w is None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            X=np.asarray(X)
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * gaussian_kernel(X[i], sv)
                y_predict[i] = s
              #  print(y_predict[i])
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))