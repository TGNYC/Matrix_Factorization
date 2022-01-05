# Matrix Factorization Using Gradient Descent
# by Tejas Gupta (github@tgnyc)

import numpy as np

def loss(M, M_approx, O):
    """
    Returns mean squared error loss over matrix of observed values O
    of numpy matrix M and its approximation matrix M_approx
    """
    return 1/np.sum(O) * np.sum(O * (M - M_approx)**2)
    
def normalize_matrix(M, O):
    """
    Returns transformed numpy matrix M whose average over observed 
    entries O is 0 and empirical distribution of its entries would
    resemble normal distribution
    """
    a = 1/np.sum(O) * np.sum(M*O) # zeros bias
    s = (1/np.sum(O) * np.sum(((M-a)*O)**2))**0.5 # normalizes scale
    return (M-a)/s

def random_factors(m, n, d):
    """
    initialize and return matrices A and B using zero-mean unit-variance
    Gaussian per entry where A has shape m x d and B has shape d x n
    """
    A = np.random.randn(m, d)
    B = np.random.randn(d, n) 
    # normalize the row of A and columns of B:
    A = A / np.linalg.norm(A, axis=1, keepdims = 1)
    B = B / np.linalg.norm(B, axis=0, keepdims = 1)
    return A, B

def factor_gradient(M, O, A, B):
    """
    Given 2D numpy array M of size m x n containing observed values,
    2D numpy array O of size m x n containing 1s in observed entries of
    M and 0s in unobserved entries of M, 
    and 2D numpy arrays A of size m x d and B of d x n,
    returns gradients of loss function w.r.t. A and w.r.t. B
    """
    dL = 2 / np.sum(O) * ((A@B - M) * O)
    dA = dL @ B.T
    dB = A.T @ dL
    return dA, dB

def gen_lorma(M, O, d, epochs, eta):
    """
    Uses stochastic gradient descent with given number of epochs and 
    list of floats eta which are the learning rates per epoch
    to generate a low-rank matrix approximation of 
    2D numpy array M of shape m x n with observed entries in 2D numpy 
    array O with embedding dimension d (much smaller than m or n)

    Prints loss every two hundred epochs to standard output

    Returns factorization 2D numpy matrices A of shape m x d and B
    of shape d x n and list of losses per epoch
    """
    m, n = M.shape
    A, B = random_factors(m, n, d)
    epoch_losses = [loss(M, A@B, O)]
    for e in range(epochs):
        dA, dB = factor_gradient(M, O, A, B)
        A = A - eta[e] * dA
        B = B - eta[e] * dB
        if (e + 1) % (epochs // 10) == 0:
            epoch_losses.append(loss(M, A@B, O))
            print((e + 1), ': ', epoch_losses[-1].round(4))
    return A, B, epoch_losses

