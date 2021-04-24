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
    ql = mdptoolbox.mdp.QLearning(
        P, # transitions
        R, # rewards
        d, # discount
        n_iter=10000,
    )
    ql.run()
    # print('q learning Q matrix:', ql.Q)
    print('q learning value function:', ql.V)
    print('q learning mean discrepancy:', ql.mean_discrepancy)
    print('q learning best policy:', ql.policy)
    results.append(ql)


# plot_discount(disc, results, 'Q-Learning Discount Forest', 'q_learning_discount_forest')