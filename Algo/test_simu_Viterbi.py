import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from Simulation_data import A, B, PI, view


def viterbi_log(PI, A, B, Y):
    Y = np.asarray(Y, dtype=int)
    T = len(Y)
    N = len(PI)

    log_PI = np.log(PI)
    log_A = np.log(A)
    log_B = np.log(B)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    delta[0] = log_PI + log_B[:, Y[0]]

    for t in range(1, T):
        for j in range(N):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = np.max(scores) + log_B[j, Y[t]]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])

    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


def forward_backward(PI, A, B, Y):
    Y = np.asarray(Y, dtype=int)
    T = len(Y)
    N = len(PI)

    alpha = np.zeros((T, N))
    beta = np.zeros((T, N))
    c = np.zeros(T)

    alpha[0] = PI * B[:, Y[0]]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]

    for t in range(1, T):
        alpha[t] = alpha[t - 1] @ A * B[:, Y[t]]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]

    beta[-1] = 1

    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, Y[t + 1]] * beta[t + 1])
        beta[t] /= c[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    return gamma


def analyser_viterbi(PI, A, B, Y, etats_reels=None, horizon=220):
    Y = np.asarray(Y, dtype=int)

    viterbi_path = viterbi_log(PI, A, B, Y)
    gamma = forward_backward(PI, A, B, Y)

    posterior_path = np.argmax(gamma, axis=1)
    local_path = np.argmax(B[:, Y].T, axis=1)

    h = min(horizon, len(Y))

    couleurs = ["#f4cf5d", "#9aa5b1", "#4e8bc8"]
    labels = ["soleil", "nuageux", "pluie"]
    cmap = ListedColormap(couleurs)

    legend = [
        Line2D([0], [0], marker="s", linestyle="", color=c, label=l, markersize=9)
        for c, l in zip(couleurs, labels)
    ]

    print("\n===== Analyse Viterbi =====\n")

    D_VP = np.mean(viterbi_path != posterior_path)
    print(f"Désaccord Viterbi / posterior decoding : {D_VP:.2%}")

    taux_changement_viterbi = np.mean(viterbi_path[1:] != viterbi_path[:-1])
    taux_changement_posterior = np.mean(posterior_path[1:] != posterior_path[:-1])

    print(f"Taux de changements Viterbi : {taux_changement_viterbi:.2%}")
    print(f"Taux de changements posterior : {taux_changement_posterior:.2%}")

    proba_post_viterbi = np.mean(gamma[np.arange(len(Y)), viterbi_path])
    print(f"Probabilité postérieure moyenne de l'état Viterbi : {proba_post_viterbi:.2%}")

    # Graphe 1 : Viterbi / posterior / local
    data = np.vstack([
        viterbi_path[:h],
        posterior_path[:h],
        local_path[:h],
    ])

    plt.figure(figsize=(9, 3.2))
    plt.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
    plt.yticks([0, 1, 2], ["Viterbi", "Posterior", "Local"])
    plt.xlabel("t")
    plt.title(f"Comparaison des décodages sur les {h} premières dates")
    plt.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    plt.tight_layout()

    # Graphe 2 : Viterbi / états réels si disponibles
    if etats_reels is not None:
        etats_reels = np.asarray(etats_reels, dtype=int)

        data_reel = np.vstack([
            etats_reels[:h],
            viterbi_path[:h],
        ])

        plt.figure(figsize=(10.5, 2.6))
        plt.imshow(data_reel, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
        plt.yticks([0, 1], ["Réel", "Viterbi"])
        plt.xlabel("t")
        plt.title(f"Comparaison Viterbi / états réels sur les {h} premières dates")
        plt.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
        plt.tight_layout()

    plt.show()

    return viterbi_path, posterior_path, local_path
