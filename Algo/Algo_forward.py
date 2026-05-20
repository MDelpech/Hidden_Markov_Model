import numpy as np
from Simulation_data import A, B, PI, view

### Problème 1 ###

def forward_algo(PI, A, B, Y):
    T=len(Y)
    N=A.shape[0]
    alpha = np.zeros((N,T))
    alpha[:, 0] = PI * B[:, Y[0]]
    for t in range(1,len(Y)):
        for s in range(len(A)):
            alpha[s,t]=np.dot(alpha[:,t-1],A[:,s])*B[s, Y[t]]
    return np.sum(alpha[:,-1])

vraissemblance = forward_algo(PI, A, B, view)
#print(vraissemblance)