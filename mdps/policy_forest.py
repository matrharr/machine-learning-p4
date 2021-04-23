import mdptoolbox.example

P, R = mdptoolbox.example.forest(
    S=500,
    r1=4,
    r2=2,
    p=0.1,
    is_sparse=False
)
pi = mdptoolbox.mdp.PolicyIteration(
    P, # transitions
    R, # rewards
    0.3, # discount
    policy0=None,
    max_iter=1000, 
    eval_type=0, 
)
pi.run()
print('policy iteration value function:', pi.V)
print('policy iteration iterations:', pi.iter)
print('policy iteration time:', pi.time)
print('policy iteration best policy:', pi.policy)
