import mdptoolbox.example
from helpers.open_ai_convert import OpenAI_MDPToolbox
from hiive.mdptoolbox.mdp import QLearning
from helpers.plot_graphs import plot_rewards

ex = OpenAI_MDPToolbox('FrozenLake-v0', False)
P = ex.P
R = ex.R
disc = [0.1, 0.3, 0.5, 0.7, 0.9, 2.1, 4, 5, 10]

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
    disc, results, 'Q-Learning Discount/Rewards FrozenLake',
    'q_learning_discount_rewards_frozenlake', 'Discount'
)

ep = [1.0, 0.9, 0.5, 0.3, 0.1, 0.01]

results = []
for e in ep:
    ql = QLearning(
        P, # transitions
        R, # rewards
        0.9, # discount
        alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
        epsilon=e, epsilon_min=0.1, epsilon_decay=0.99,
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
    ep, results, 'Q-Learning Epsilon/Rewards FrozenLake',
    'q_learning_epsilon_rewards_frozenlake', 'Epsilon'
)

alpha = [1.0, 0.9, 0.5, 0.3, 0.1, 0.01]

results = []
for a in alpha:
    ql = QLearning(
        P, # transitions
        R, # rewards
        0.9, # discount
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
    alpha, results, 'Q-Learning Alpha/Rewards FrozenLake',
    'q_learning_alpha_rewards_frozenlake', 'Alpha'
)
