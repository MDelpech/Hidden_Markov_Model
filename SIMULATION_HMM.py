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
n_steps = 1000

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
#print(vraissemblance)



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


### Problème 3 ###

def encoder_observations(Y, observations_possible):
    # Si Y contient déjà des indices : [0, 1, 0, 0, ...]
    if all(isinstance(y, (int, np.integer)) for y in Y):
        return np.array(Y, dtype=int)

    # Si Y contient des noms : ["Parapluie", "Pas parapluie", ...]
    obs_to_index = {obs: i for i, obs in enumerate(observations_possible)}
    return np.array([obs_to_index[y] for y in Y], dtype=int)

def baum_welch(Y, N, observations_possible, epsilon=1e-6, max_iter=1000):
    Y = encoder_observations(Y, observations_possible)

    T = len(Y)
    M = len(observations_possible)

    # Initialisation aléatoire
    PI = np.random.random(N)
    PI /= PI.sum()
    A = np.random.random((N, N))
    A /= A.sum(axis=1, keepdims=True)
    B = np.random.random((N, N))
    B /= B.sum(axis=1, keepdims=True)

    ancienne_vraisemblance = -np.inf

    for iteration in range(max_iter):

        # Forward avec normalisation
        alpha = np.zeros((N, T))
        c = np.zeros(T)

        alpha[:, 0] = PI * B[:, Y[0]]
        c[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / c[0]

        for t in range(1, T):
            for s in range(N):
                alpha[s, t] = np.dot(alpha[:, t - 1], A[:, s]) * B[s, Y[t]]

            c[t] = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / c[t]

        vraisemblance = np.sum(np.log(c))

        # Backward avec normalisation
        beta = np.zeros((N, T))
        beta[:, -1] = 1

        for t in range(T - 2, -1, -1):
            for s in range(N):
                beta[s, t] = np.sum(A[s, :] * B[:, Y[t + 1]] * beta[:, t + 1])
                beta[s, t] = beta[s, t] / c[t + 1]

        # Gamma
        gamma = np.zeros((N, T))

        for t in range(T):
            denom = np.sum(alpha[:, t] * beta[:, t])
            for s in range(N):
                gamma[s, t] = alpha[s, t] * beta[s, t] / denom

        # Xi
        xi = np.zeros((N, N, T - 1))

        for t in range(T - 1):
            denom = 0

            for i in range(N):
                for j in range(N):
                    denom += (
                        alpha[i, t]
                        * A[i, j]
                        * B[j, Y[t + 1]]
                        * beta[j, t + 1]
                    )

            for i in range(N):
                for j in range(N):
                    xi[i, j, t] = (
                        alpha[i, t]
                        * A[i, j]
                        * B[j, Y[t + 1]]
                        * beta[j, t + 1]
                    ) / denom

        # Mise à jour de PI
        PI = gamma[:, 0]

        # Mise à jour de A
        for i in range(N):
            for j in range(N):
                A[i, j] = np.sum(xi[i, j, :]) / np.sum(gamma[i, :-1])

        # Mise à jour de B
        for i in range(N):
            for k in range(M):
                somme = 0

                for t in range(T):
                    if Y[t] == k:
                        somme += gamma[i, t]

                B[i, k] = somme / np.sum(gamma[i, :])

        # Critère d'arrêt
        if abs(vraisemblance - ancienne_vraisemblance) < epsilon:
            break

        ancienne_vraisemblance = vraisemblance

    return PI, A, B

PI_est, A_est, B_est = baum_welch(view,3,observations,1e-6,1000)
print(PI_est, A_est, B_est)

