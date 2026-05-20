import numpy as np
import matplotlib.pyplot as plt

from Simulation_data import A, B, PI, view, observations
from Algo_Baum_Welch import baum_welch


### Test de l'algorithme de Baum-Welch ###

### convergence de la log vraisemblance au cours des itérations ###

def analyser_convergence(log_likelihoods, T, burn_in=10):
    import numpy as np
    import matplotlib.pyplot as plt

    log_likelihoods = np.array(log_likelihoods)

    nb_iter = len(log_likelihoods)

    if nb_iter < 2:
        print("Pas assez d'itérations pour analyser la convergence.")
        return

    deltas = np.diff(log_likelihoods)
    deltas_moyens = deltas / T
    gains_relatifs = deltas / (1 + np.abs(log_likelihoods[:-1]))

    print("\n===== Analyse de convergence =====\n")

    print(f"Nombre d'itérations : {nb_iter}")
    print(f"Log-vraisemblance initiale : {log_likelihoods[0]:.6f}")
    print(f"Log-vraisemblance finale   : {log_likelihoods[-1]:.6f}")
    print(f"Gain total                 : {log_likelihoods[-1] - log_likelihoods[0]:.6f}")
    print(f"Gain final par observation : {deltas_moyens[-1]:.6e}")
    print(f"Gain relatif final         : {gains_relatifs[-1]:.6e}")

    print("\n===== Vérification de la monotonie =====\n")

    indices_baisse = np.where(deltas < 0)[0]

    if len(indices_baisse) == 0:
        print("OK : la log-vraisemblance est bien non décroissante.")
    else:
        print("Attention : baisse de log-vraisemblance détectée.")
        for idx in indices_baisse:
            print(
                f"Itération {idx} -> {idx + 1} : "
                f"Delta = {deltas[idx]:.6e}, "
                f"Delta/T = {deltas_moyens[idx]:.6e}"
            )

    print("\n===== Critère d'arrêt =====\n")

    if abs(gains_relatifs[-1]) < 1e-6:
        print("Convergence détectée selon le critère relatif r_k < 10^-6.")
    elif abs(deltas_moyens[-1]) < 1e-6:
        print("Convergence détectée selon le critère moyen Delta_k/T < 10^-6.")
    else:
        print("La convergence n'est pas encore clairement atteinte.")

    if nb_iter >= 1000:
        print("Attention : l'algorithme est allé jusqu'à max_iter.")
        print("Cela signifie que ton critère d'arrêt dans Baum-Welch n'a probablement pas arrêté l'algorithme.")

    iterations = np.arange(nb_iter)
    delta_iterations = np.arange(1, nb_iter)

    eps = 1e-300

    # Graphe 1 : log-vraisemblance complète
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, log_likelihoods, marker="o", markersize=3)
    plt.xlabel("Itération")
    plt.ylabel("Log-vraisemblance")
    plt.title("Log-vraisemblance complète")
    plt.grid(True)

    # Graphe 2 : zoom après burn-in
    if nb_iter > burn_in:
        plt.figure(figsize=(8, 5))
        plt.plot(iterations[burn_in:], log_likelihoods[burn_in:], marker="o", markersize=3)
        plt.xlabel("Itération")
        plt.ylabel("Log-vraisemblance")
        plt.title(f"Zoom de la log-vraisemblance après {burn_in} itérations")
        plt.grid(True)

    # Graphe 3 : gain moyen en échelle log
    plt.figure(figsize=(8, 5))
    plt.semilogy(delta_iterations, np.abs(deltas_moyens) + eps, marker="o", markersize=3)
    plt.axhline(1e-6, linestyle="--", label="seuil 10^-6")
    plt.axhline(1e-5, linestyle="--", label="seuil 10^-5")
    plt.xlabel("Itération")
    plt.ylabel(r"$|\Delta_k| / T$")
    plt.title("Gain moyen par observation en échelle logarithmique")
    plt.legend()
    plt.grid(True)

    # Graphe 4 : gain relatif en échelle log
    plt.figure(figsize=(8, 5))
    plt.semilogy(delta_iterations, np.abs(gains_relatifs) + eps, marker="o", markersize=3)
    plt.axhline(1e-6, linestyle="--", label="seuil 10^-6")
    plt.xlabel("Itération")
    plt.ylabel(r"$|r_k|$")
    plt.title("Gain relatif en échelle logarithmique")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    PI_est, A_est, B_est, vraisemblance, log_likelihoods = baum_welch(
        Y=view,
        N=3,
        observations_possible=observations,
        epsilon=1e-6,
        max_iter=1000
    )

    T = len(view)

    print("\nPI estimé :")
    print(PI_est)

    print("\nA estimé :")
    print(A_est)

    print("\nB estimé :")
    print(B_est)

    analyser_convergence(log_likelihoods, T)