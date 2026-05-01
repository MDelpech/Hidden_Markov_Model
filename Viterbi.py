import numpy as np

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(
                [(V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states]
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        path = newpath

    (prob, state) = max([(V[-1][y], y) for y in states])
    return (prob, path[state])

states = ('Rainy', 'Sunny')
observations = ('Walk', 'Shop', 'Walk','Shop','Shop','Shop')
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
transition_probability = {
   'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
   }
emission_probability = {
   'Rainy' : {'Walk': 0.1, 'Shop': 0.4, 'Clean': 0.5},
   'Sunny' : {'Walk': 0.6, 'Shop': 0.3, 'Clean': 0.1},
   }

print(viterbi(observations, states, start_probability, transition_probability, emission_probability))