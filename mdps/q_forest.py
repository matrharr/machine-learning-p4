import mdptoolbox.example
from helpers.plot_graphs import plot_discount, plot_rewards
from hiive.mdptoolbox.mdp import QLearning

P, R = mdptoolbox.example.forest(
    S=500,
    r1=100,
    r2=2,
    p=0.1,
    is_sparse=False
)
disc = [0.1, 0.3, 0.5, 0.7, 0.9]
# ep = [1.0, 0.9, 0.5, 0.3, 0.1, 0.01]
ep = [0.00099, 0.001, 0.005, 0.01, 0.03]
alpha = [1.0, 0.9, 0.5, 0.3, 0.1, 0.01]

results = []
for d in disc:
    ql = QLearning(
        P, # transitions
        R, # rewards
        d, # discount
        alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
        epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99,
        n_iter=10000, skip_check=False, iter_callback=None,
        run_stat_frequency=None
    )
    ql.run()
    # print('q learning Q matrix:', ql.Q)
    print('q learning value function:', ql.V)
    # print('q learning mean discrepancy:', ql.mean_discrepancy)
    print('q learning best policy:', ql.policy)
    results.append(ql)

plot_rewards(
    disc, results, 'Q-Learning Discount/Rewards Forest',
    'q_learning_discount_rewards_forest', 'Discount'
)


results = []
for e in ep:
    ql = QLearning(
        P, # transitions
        R, # rewards
        0.7, # discount
        alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
        epsilon=e, epsilon_min=0.000001, epsilon_decay=0.99,
        n_iter=10000, skip_check=False, iter_callback=None,
        run_stat_frequency=None
    )
    ql.run()
    # print('q learning Q matrix:', ql.Q)
    print('q learning value function:', ql.V)
    # print('q learning mean discrepancy:', ql.mean_discrepancy)
    print('q learning best policy:', ql.policy)
    results.append(ql)

plot_rewards(
    ep, results, 'Q-Learning Epsilon/Rewards Forest',
    'q_learning_epsilon_rewards_forest', 'Epsilon'
)


results = []
for a in alpha:
    ql = QLearning(
        P, # transitions
        R, # rewards
        0.7, # discount
        alpha=a, alpha_decay=0.99, alpha_min=0.001,
        epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.99,
        n_iter=10000, skip_check=False, iter_callback=None,
        run_stat_frequency=None
    )
    ql.run()
    # print('q learning Q matrix:', ql.Q)
    print('q learning value function:', ql.V)
    # print('q learning mean discrepancy:', ql.mean_discrepancy)
    print('q learning best policy:', ql.policy)
    results.append(ql)

plot_rewards(
    alpha, results, 'Q-Learning Alpha/Rewards Forest',
    'q_learning_alpha_rewards_forest', 'Alpha'
)

print('----------------Best QL Forest---------------')
ql = QLearning(
        P, # transitions
        R, # rewards
        0.7, # discount
        alpha=0.01, alpha_decay=0.99, alpha_min=0.001,
        epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.99,
        n_iter=10000, skip_check=False, iter_callback=None,
        run_stat_frequency=None
    )
ql.run()
# print('q learning Q matrix:', ql.Q)
print('q learning value function:', ql.V)
# print('q learning mean discrepancy:', ql.mean_discrepancy)
print('q learning best policy:', ql.policy)
