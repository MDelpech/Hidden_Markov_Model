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