import mdptoolbox.example
from helpers.plot_graphs import plot_rewards
from hiive.mdptoolbox.mdp import ValueIteration

disc = [0.1, 0.3, 0.5, 0.7, 0.9]
ep = [0.00099, 0.001, 0.005, 0.01, 0.03]

P, R = mdptoolbox.example.forest(
    S=500,
    r1=100,
    r2=2,
    p=0.1,
    is_sparse=False
)
results = []
for d in disc:
    vi = ValueIteration(
        P,
        R,
        d,
        epsilon=0.001,
        max_iter=1000
    )
    vi.run()
    print('value iteration value function:', vi.V)
    print('value iteration iterations:', vi.iter)
    print('value iteration time:', vi.time)
    print('value iteration best policy:', vi.policy)
    results.append(vi)

plot_rewards(
    disc, results, 'Value Iteration Discount/Rewards Forest',
    'value_iteration_discount_rewards_forest', 'Discount'
)

results = []
for e in ep:
    vi = ValueIteration(
        P,
        R,
        0.9,
        epsilon=e,
        max_iter=1000
    )
    vi.run()
    print('value iteration value function:', vi.V)
    print('value iteration iterations:', vi.iter)
    print('value iteration time:', vi.time)
    print('value iteration best policy:', vi.policy)
    results.append(vi)

plot_rewards(
    ep, results, 'Value Iteration Epsilon/Rewards Forest',
    'value_iteration_epsilon_rewards_forest', 'Epsilon'
)
