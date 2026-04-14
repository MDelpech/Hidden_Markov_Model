import numpy as np

def forward_algo(A, B, pi, Y):
    T = len(Y)
    N = A.shape[0]
    
    forward = np.zeros((N, T))
    
    # initialisation
    forward[:, 0] = pi * B[:, Y[0]]
    
    # recursion
    for t in range(1, T):
        for s in range(N):
            forward[s, t] = np.dot(forward[:, t-1], A[:, s]) * B[s, Y[t]]
    
    return forward

def likelihood(A, B, pi, Y):
    alpha = forward_algo(A, B, pi, Y)
    return np.sum(alpha[:, -1])

# Test des fonctions

A = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

B = np.array([
    [0.9, 0.1],   
    [0.2, 0.8]   
])

pi = np.array([0.6, 0.4])
Y = [0, 1, 0]

alpha = forward_algo(A, B, pi, Y)
print(alpha)
print("Likelihood =", np.sum(alpha[:, -1]))

