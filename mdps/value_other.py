import mdptoolbox.example
from helpers.open_ai_convert import OpenAI_MDPToolbox
from helpers.plot_graphs import plot_discount, plot_ep

disc = [0.1, 0.3, 0.5, 0.7, 0.9]
ep = [0.00099, 0.001, 0.005, 0.01, 0.03]

ex = OpenAI_MDPToolbox('FrozenLake-v0', False)
P = ex.P
R = ex.R
results = []
for d in disc:
    vi = mdptoolbox.mdp.ValueIteration(
        P,
        R,
        d,
        epsilon=0.01,
        max_iter=1000
    )
    vi.run()
    print('value iteration value function:', vi.V)
    print('value iteration iterations:', vi.iter)
    print('value iteration time:', vi.time)
    print('value iteration best policy:', vi.policy)
    results.append(vi)

plot_discount(disc, results, 'Value Iteration Discount Other', 'value_iteration_discount_other')

results = []
for e in ep:
    vi = mdptoolbox.mdp.ValueIteration(
        P,
        R,
        0.3,
        epsilon=e,
        max_iter=1000
    )
    vi.run()
    print('value iteration value function:', vi.V)
    print('value iteration iterations:', vi.iter)
    print('value iteration time:', vi.time)
    print('value iteration best policy:', vi.policy)
    results.append(vi)

plot_ep(ep, results, 'Value Iteration Epsilon Other', 'value_iteration_epsilon_other')
