import numpy as np
from Simulation_data import A, B, PI, view,trajectory

### Problème 2 ###
def viterbi(PI,A,B,Y):
    T=len(Y)
    N=A.shape[0]

    delta = np.zeros((N,T))
    phi = np.zeros((N,T))

    delta[:,0] = PI * B[:,Y[0]]

    for t in range(1,T): 
        for s in range(N): 
            scores = delta[:, t-1] * A[:, s]
            phi[s, t] = np.argmax(scores)
            delta[s, t] = np.max(scores) * B[s, Y[t]]

    q = np.zeros(T, dtype=int)
    d = delta[:,-1]

    q[-1] = np.argmax(d)

    for t in range(T-2,-1,-1): 
        q[t] = phi[q[t+1],t+1]

    return q


chemin = viterbi(PI,A,B,view)
### print(chemin)
### print(trajectory)

accuracy = np.mean(np.array(chemin) == np.array(trajectory))
### print(f"\nTaux de bonnes prédictions : {accuracy:.2%}")