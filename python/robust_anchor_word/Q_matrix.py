
import numpy as np
import time
import scipy.sparse
import math
from  helper_functions import *
# rectify imports
import scipy as sp
import scipy.sparse.linalg as spalg

# Given a sparse CSC document matrix M (with floating point entries),
# comptues the word-word correlation matrix Q
def generate_Q_matrix(M, words_per_doc=None):
    
    simulation_start = time.time()
    
    vocabSize = M.shape[0]
    numdocs = M.shape[1]
    
    diag_M = np.zeros(vocabSize)

    for j in range(M.indptr.size - 1):
        
        # start and end indices for column j
        start = M.indptr[j]
        end = M.indptr[j + 1]
        
        wpd = np.sum(M.data[start:end])
        if words_per_doc != None and wpd != words_per_doc:
            print('Error: words per doc incorrect')
        
        row_indices = M.indices[start:end]
        
        diag_M[row_indices] += M.data[start:end]/(wpd*(wpd-1))
        M.data[start:end] = M.data[start:end]/math.sqrt(wpd*(wpd-1))
    
    
    Q = M*M.transpose()/numdocs
    Q = Q.todense()
    Q = np.array(Q, copy=False)

    diag_M = diag_M/numdocs
    Q = Q - np.diag(diag_M)
    
    print(('Sum of entries in Q is ', np.sum(Q)))
    print(('Multiplying Q took ', str(time.time() - simulation_start), 'seconds'))
    
    print("type(Q): {}".format(type(Q)))
    return Q


## rectification code
## Generating Rectified Co-Occurence using Dykstra's Algorithm
def proj_psd(Q,k):
    """
    Generating the Positive Semi Definite Matrix Projection
    """
    eigvals, eigvecs = spalg.eigs(Q)
    #print(eigvals.shape, eigvecs.shape)
    eigvals[eigvals.argsort()[:-k]] = 0
    # eigvecs * eigvals calculation
    MatPSD = np.einsum('ij,j->ij',eigvecs, eigvals)
    # (eigvecs * eigvals) * eigvecs' calculation
    # MatPSD = sp.sparse.csc_matrix(MatPSD) * sp.sparse.csc_matrix(eigvecs.T)
    MatPSD = sp.sparse.csc_matrix(np.real(MatPSD)) * sp.sparse.csc_matrix(np.real(eigvecs.T))
    return MatPSD

def proj_norm(Q):
    """
    Generating the Normalized Matrix Projection
    """
    #MatNorm = Q.todense()
    MatNorm = Q + ((1 - Q.sum())/(Q.shape[0] **2))
    return MatNorm

def proj_nn(Q):
    """
    Generating Non-Negative Matrix
    """
    MatNN = Q
    MatNN[np.where(MatNN < 0)] = 0
    return MatNN

def rectify(C, k=10):
    """
    rectify by projecting on to the three different spaces
    """
    #Number of iterations
    T = 10
    #Total number of clusters
    # k = 10
    # changed to parameter as this is the number of topics we are learning.
    # retain only k largest positive eigen values
    
    P1 = np.zeros(C.shape)
    P2 = np.zeros(C.shape)
    P3 = np.zeros(C.shape)
    X0 = C
    for t in range(T):
        
        print("Iteration",t)
        
        #Projecting to Positive Semi Definite Matrix
        X0 = X0 + P1
        del P1
        X0 = sp.sparse.csc_matrix(X0)
        X1 = proj_psd(X0,k)
        P1 = X0 - X1
        del X0
        
        #Projecting to the Normalized Matrix
        X1 = X1 + P2
        del P2
        X2 = proj_norm(X1)
        P2 = X1 - X2
        del X1
        
        #Projecting to the Non Negative Matrix
        X2 = X2 + P3
        del P3
        X0 = proj_nn(X2)
        P3 = X2 - X0
        del X2
        
    # Rectified Co-Occurence Matrix is C_star    
    C_star = X0
    print("type(C_star): {}".format(type(C_star)))
    # some post processing
    # C_star = C_star.todense()
    C_star = np.array(C_star, copy=False)
    
    print("Test for Non-Negativity: ", np.all(C_star >= 0))
    return C_star
