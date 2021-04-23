import mdptoolbox.example

P, R = mdptoolbox.example.forest(
    S=500,
    r1=4,
    r2=2,
    p=0.1,
    is_sparse=False
)
vi = mdptoolbox.mdp.ValueIteration(
    P, 
    R, 
    0.3,
    epsilon=0.01, max_iter=1000, initial_value=0,
)
vi.run()
print('value iteration value function:', vi.V)
print('value iteration iterations:', vi.iter)
print('value iteration time:', vi.time)
print('value iteration best policy:', vi.policy)
