import mdptoolbox.example
from helpers.open_ai_convert import OpenAI_MDPToolbox

ex = OpenAI_MDPToolbox('FrozenLake-v0', False)
P = ex.P
R = ex.R
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
