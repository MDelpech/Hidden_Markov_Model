import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

from Simulation_data import A, B, PI, view, observations
from Algo_Baum_Welch import baum_welch
from fonction import distance_l1_proba, distance_transition, distance_emission, durees_moyennes_regimes, aligner_etats


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
    
    
    ### Test de la stabilité des paramètres estimés ###
def analyser_stabilite_parametres ( baum_welch_func,Y,N,observations_possible,
                                        nb_initialisations=10, epsilon=1e-6,
                                        max_iter=1000, plot=True):
        estimations = []

        print("\n===== Lancement des différentes initialisations =====\n")

        for m in range(nb_initialisations):
            result = baum_welch_func(
                Y=Y,
                N=N,
                observations_possible=observations_possible,
                epsilon=epsilon,
                max_iter=max_iter
            )

            # Compatible avec :
            # return PI, A, B, vraisemblance
            # ou return PI, A, B, vraisemblance, log_likelihoods
            if len(result) == 4:
                PI_est, A_est, B_est, loglik = result
                log_likelihoods = None
            else:
                PI_est, A_est, B_est, loglik, log_likelihoods = result

            estimations.append({
                "PI": PI_est,
                "A": A_est,
                "B": B_est,
                "loglik": loglik,
                "log_likelihoods": log_likelihoods
            })

            print(f"Initialisation {m + 1} : log-vraisemblance finale = {loglik:.6f}")

        # Choix de la meilleure estimation comme référence
        logliks = np.array([e["loglik"] for e in estimations])
        meilleur_index = np.argmax(logliks)

        PI_ref = estimations[meilleur_index]["PI"]
        A_ref = estimations[meilleur_index]["A"]
        B_ref = estimations[meilleur_index]["B"]

        reference = (PI_ref, A_ref, B_ref)

        print("\n===== Meilleure initialisation =====\n")
        print(f"Meilleure initialisation : {meilleur_index + 1}")
        print(f"Meilleure log-vraisemblance : {logliks[meilleur_index]:.6f}")
        print("\nA de référence :")
        print(A_ref)

        distances_pi = []
        distances_A = []
        distances_B = []
        distances_durees = []
        ecarts_loglik_par_obs = []

        estimations_alignees = []

        T = len(Y)

        print("\n===== Comparaison après alignement des états =====\n")

        for m, estimation in enumerate(estimations):
            PI = estimation["PI"]
            A = estimation["A"]
            B = estimation["B"]

            estimation_alignee, permutation, _ = aligner_etats(
                reference=reference,
                estimation=(PI, A, B)
            )

            PI_aligne, A_aligne, B_aligne = estimation_alignee

            d_pi = distance_l1_proba(PI_ref, PI_aligne)
            d_A = distance_transition(A_ref, A_aligne)
            d_B = distance_emission(B_ref, B_aligne)

            durees_ref = durees_moyennes_regimes(A_ref)
            durees_alignees = durees_moyennes_regimes(A_aligne)

            d_durees = np.abs(durees_alignees - durees_ref) / np.maximum(durees_ref, 1e-12)
            d_durees_moyen = np.mean(d_durees)

            ecart_loglik = (logliks[meilleur_index] - estimation["loglik"]) / T

            distances_pi.append(d_pi)
            distances_A.append(d_A)
            distances_B.append(d_B)
            distances_durees.append(d_durees_moyen)
            ecarts_loglik_par_obs.append(ecart_loglik)

            estimations_alignees.append({
                "PI": PI_aligne,
                "A": A_aligne,
                "B": B_aligne,
                "permutation": permutation
            })

            print(f"Initialisation {m + 1}")
            print(f"  permutation utilisée       : {permutation}")
            print(f"  écart loglik / observation : {ecart_loglik:.6e}")
            print(f"  D_pi                       : {d_pi:.6e}")
            print(f"  D_A                        : {d_A:.6e}")
            print(f"  D_B                        : {d_B:.6e}")
            print(f"  écart moyen durées régimes : {d_durees_moyen:.6e}")
            print()

        distances_pi = np.array(distances_pi)
        distances_A = np.array(distances_A)
        distances_B = np.array(distances_B)
        distances_durees = np.array(distances_durees)
        ecarts_loglik_par_obs = np.array(ecarts_loglik_par_obs)

        print("\n===== Résumé de stabilité =====\n")

        print(f"D_pi moyen : {np.mean(distances_pi):.6e}")
        print(f"D_A moyen  : {np.mean(distances_A):.6e}")
        print(f"D_B moyen  : {np.mean(distances_B):.6e}")
        print(f"Écart moyen des durées : {np.mean(distances_durees):.6e}")
        print(f"Écart max loglik / observation : {np.max(ecarts_loglik_par_obs):.6e}")

        print("\n===== Diagnostic =====\n")

        if np.max(ecarts_loglik_par_obs) < 1e-3:
            print("Les différentes initialisations donnent des log-vraisemblances finales proches.")
        else:
            print("Certaines initialisations donnent une log-vraisemblance nettement moins bonne.")

        if np.mean(distances_A) < 0.05 and np.mean(distances_B) < 0.05:
            print("Les matrices A et B sont globalement stables après alignement des états.")
        elif np.mean(distances_A) < 0.10 and np.mean(distances_B) < 0.10:
            print("Les matrices A et B sont modérément stables.")
        else:
            print("Les paramètres estimés varient fortement selon l'initialisation.")

        if np.mean(distances_durees) > 0.20:
            print("Attention : les durées moyennes des régimes varient sensiblement.")
            print("Deux solutions peuvent avoir des vraisemblances proches mais des régimes différents.")

        if plot:
            indices = np.arange(1, nb_initialisations + 1)

            plt.figure(figsize=(8, 5))
            plt.bar(indices, ecarts_loglik_par_obs)
            plt.xlabel("Initialisation")
            plt.ylabel("Écart de log-vraisemblance par observation")
            plt.title("Comparaison des log-vraisemblances finales")
            plt.grid(True)

            plt.figure(figsize=(8, 5))
            plt.bar(indices, distances_pi)
            plt.xlabel("Initialisation")
            plt.ylabel(r"$D_\pi$")
            plt.title("Stabilité de la distribution initiale")
            plt.grid(True)

            plt.figure(figsize=(8, 5))
            plt.bar(indices, distances_A)
            plt.xlabel("Initialisation")
            plt.ylabel(r"$D_A$")
            plt.title("Stabilité de la matrice de transition A")
            plt.grid(True)

            plt.figure(figsize=(8, 5))
            plt.bar(indices, distances_B)
            plt.xlabel("Initialisation")
            plt.ylabel(r"$D_B$")
            plt.title("Stabilité de la matrice d'émission B")
            plt.grid(True)

            plt.figure(figsize=(8, 5))
            plt.bar(indices, distances_durees)
            plt.xlabel("Initialisation")
            plt.ylabel("Écart relatif moyen")
            plt.title("Stabilité des durées moyennes des régimes")
            plt.grid(True)

            plt.show()

        return {
            "estimations": estimations,
            "estimations_alignees": estimations_alignees,
            "meilleur_index": meilleur_index,
            "reference": {
                "PI": PI_ref,
                "A": A_ref,
                "B": B_ref,
                "loglik": logliks[meilleur_index],
                "durees": durees_moyennes_regimes(A_ref)
            },
            "distances_pi": distances_pi,
            "distances_A": distances_A,
            "distances_B": distances_B,
            "distances_durees": distances_durees,
            "ecarts_loglik_par_obs": ecarts_loglik_par_obs
        }

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

    ## analyser_convergence(log_likelihoods, T)
    resultats_stabilite = analyser_stabilite_parametres(
    baum_welch_func=baum_welch,
    Y=view,
    N=3,
    observations_possible=observations,
    nb_initialisations=10,
    epsilon=1e-6,
    max_iter=1000,
    plot=True
    )