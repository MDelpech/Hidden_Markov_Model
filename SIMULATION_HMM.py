import numpy as np



# États possibles
states = ["soleil", "nuageux", "pluie"]
observations = ["Parapluie", "Pas parapluie"]

# Matrice de transition
# Chaque ligne correspond à l'état actuel
# Chaque colonne correspond au prochain état
PI = np.array([0.5, 0.3, 0.2])

A = np.array([
    [0.7, 0.295, 0.005],  # soleil -> soleil, nuageux, pluie
    [0.3, 0.4, 0.3],  # nuageux -> soleil, nuageux, pluie
    [0.005, 0.395, 0.6]   # pluie -> soleil, nuageux, pluie
])

# Matrice de probabilité d'observations
# Chaque ligne correspond à l'état actuel
# Chaque colonne correspond à une observation depuis l'état actuel
B=np.array([
    [0.1, 0.9],    # soleil -> parapluie, pas de parapluie
    [0.4, 0.6],    # nuageux -> parapluie, pas de parapluie
    [0.9, 0.1]     # pluie -> parapluie, pas de parapluie
])

# Vérification : chaque ligne doit faire 1 pour P et B
if not np.allclose(A.sum(axis=1), 1):
    raise ValueError("Chaque ligne de la matrice P doit sommer à 1")
if not np.allclose(B.sum(axis=1),1):
    raise ValueError("Chaque ligne de la matrice B doit sommer à 1")

def simulate_HMM(A, B, PI, n_steps):
    current = np.random.choice(len(PI), p=PI)
    states_sequence = [current]
    observation = np.random.choice(len(B[0]), p=B[current])
    observations_sequence = [observation]

    for _ in range(n_steps):
        current = np.random.choice(len(A), p=A[current])
        states_sequence.append(current)
        observation = np.random.choice(len(B[0]), p=B[current])
        observations_sequence.append(observation)

    return [states_sequence, observations_sequence]

# Simulation
n_steps = 20

[trajectory, view] = simulate_HMM(A, B, PI, n_steps)

# Affichage avec les noms des états
pretty_observations = [observations[i] for i in view]

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
print(chemin)
print(trajectory)

accuracy = np.mean(np.array(chemin) == np.array(trajectory))
print(f"\nTaux de bonnes prédictions : {accuracy:.2%}")




