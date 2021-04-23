import mdptoolbox.example

P, R = mdptoolbox.example.forest(
    S=500,
    r1=4,
    r2=2,
    p=0.1,
    is_sparse=False
)
ql = mdptoolbox.mdp.QLearning(
    P, # transitions
    R, # rewards
    0.3, # discount
    n_iter=10000,
)
ql.run()
print('q learning Q matrix:', ql.Q)
print('q learning value function:', ql.V)
print('q learning mean discrepancy:', ql.mean_discrepancy)
print('q learning best policy:', ql.policy)
