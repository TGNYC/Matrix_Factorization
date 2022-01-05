from matrix_factorization import *

k = 2
epochs = 2000
eta = .1 * np.ones(epochs)
M = np.array([[1, 5, 5, 1], 
              [1, 1, 1, 1], 
              [1, 5, 5, 1], 
              [5, 5, 5, 5], 
              [5, 1, 1, 5], 
              [5, 1, 1, 5]])
O = np.ones((6, 4))

A, B, losses = gen_lorma(M, O, 2, epochs, eta)
"""
m, n = 6, 4
A = np.random.randn(m, k)
B = np.random.randn(k, n) # 5 points for A,B initialization
apperr = [loss(M, A@B, O)]
for e in range(2000):
    M_approx = A @ B
    delta = 2 * (M_approx - M) * O / np.sum(O)
    dA = (delta @ B.T)
    dB = (A.T @ delta)
    A = A - eta[e] * dA
    B = B - eta[e] * dB
"""
if np.array_equal(np.rint(A@B), M): print("Matrix factorization found") 
print("A = \n", A)
print("B = \n", B)
print("M = \n", M)
print("AB = \n", np.rint(A@B))