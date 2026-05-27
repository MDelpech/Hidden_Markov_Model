import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

def distance_l1_proba(p, q):
        return 0.5 * np.sum(np.abs(p - q))
    
def distance_transition(A_ref, A_test):
    """
    Distance moyenne entre deux matrices de transition.
    D_A = (1/N) * somme_i 1/2 * somme_j |a_ij - a'_ij|
    """
    N = A_ref.shape[0]
    return (1 / N) * np.sum([
        0.5 * np.sum(np.abs(A_ref[i, :] - A_test[i, :]))
        for i in range(N)
    ])


def distance_emission(B_ref, B_test):
    """
    Distance moyenne entre deux matrices d'émission discrètes.
    D_B = (1/N) * somme_i 1/2 * somme_y |b_i(y) - b'_i(y)|
    """
    N = B_ref.shape[0]
    return (1 / N) * np.sum([
        0.5 * np.sum(np.abs(B_ref[i, :] - B_test[i, :]))
        for i in range(N)
    ])


def durees_moyennes_regimes(A):
    """
    Durée moyenne du régime i :
    d_i = 1 / (1 - a_ii)
    """
    diag = np.diag(A)

    # Sécurité numérique si a_ii est trop proche de 1
    eps = 1e-12
    return 1 / np.maximum(1 - diag, eps)


def aligner_etats(reference, estimation):
    """
    Aligne les états d'une estimation sur ceux d'une estimation de référence.

    reference = (PI_ref, A_ref, B_ref)
    estimation = (PI, A, B)

    On teste toutes les permutations possibles et on garde celle qui minimise
    une distance globale sur PI, A et B.
    """

    PI_ref, A_ref, B_ref = reference
    PI, A, B = estimation

    N = len(PI)

    meilleure_distance = np.inf
    meilleure_permutation = None
    meilleur_triplet = None

    for perm in permutations(range(N)):
        perm = np.array(perm)

        # Permutation des états
        PI_perm = PI[perm]
        A_perm = A[perm, :][:, perm]
        B_perm = B[perm, :]

        d_pi = distance_l1_proba(PI_ref, PI_perm)
        d_A = distance_transition(A_ref, A_perm)
        d_B = distance_emission(B_ref, B_perm)

        distance_totale = d_pi + d_A + d_B

        if distance_totale < meilleure_distance:
            meilleure_distance = distance_totale
            meilleure_permutation = perm
            meilleur_triplet = (PI_perm, A_perm, B_perm)

    return meilleur_triplet, meilleure_permutation, meilleure_distance