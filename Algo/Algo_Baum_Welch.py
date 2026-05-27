import numpy as np
from Simulation_data import A, B, PI, view, observations

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
    B = np.random.random((N, M))
    B /= B.sum(axis=1, keepdims=True)

    ancienne_vraisemblance = -np.inf
    log_vraisemblance = []

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
        log_vraisemblance.append(vraisemblance)

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

    return PI, A, B, vraisemblance, log_vraisemblance

PI_est, A_est, B_est, Vrai, log_vrai = baum_welch(view,3,observations,1e-6,1000)

ordre = [1, 2, 0]

PI_aligne = PI_est[ordre]
A_aligne = A_est[np.ix_(ordre, ordre)]
B_aligne = B_est[ordre, :]

print("PI aligné :")
print(PI_aligne)

print("A aligné :")
print(A_aligne)

print("B aligné :")
print(B_aligne)