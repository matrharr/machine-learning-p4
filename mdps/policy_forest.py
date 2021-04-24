import mdptoolbox.example
from helpers.plot_graphs import plot_discount

disc = [0.1, 0.3, 0.5, 0.7, 0.9]

P, R = mdptoolbox.example.forest(
    S=500,
    r1=4,
    r2=2,
    p=0.1,
    is_sparse=False
)
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

plot_discount(disc, results, 'Policy Iteration Discount Forest', 'policy_iteration_discount_forest')