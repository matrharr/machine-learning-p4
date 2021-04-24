import mdptoolbox.example
from helpers.open_ai_convert import OpenAI_MDPToolbox
from helpers.plot_graphs import plot_discount

disc = [0.1, 0.3, 0.5, 0.7, 0.9]

ex = OpenAI_MDPToolbox('FrozenLake-v0', False)
P = ex.P
R = ex.R
results = []
for d in disc:
    pi = mdptoolbox.mdp.PolicyIteration(
        P, # transitions
        R, # rewards
        d, # discount
        policy0=None,
        max_iter=1000,
        eval_type=0,
    )
    pi.run()
    print('policy iteration value function:', pi.V)
    print('policy iteration iterations:', pi.iter)
    print('policy iteration time:', pi.time)
    print('policy iteration best policy:', pi.policy)
    results.append(pi)

plot_discount(disc, results, 'Policy Iteration Discount Other', 'policy_iteration_discount_other')