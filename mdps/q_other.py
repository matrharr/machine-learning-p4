import mdptoolbox.example
from helpers.open_ai_convert import OpenAI_MDPToolbox

ex = OpenAI_MDPToolbox('FrozenLake-v0', False)
P = ex.P
R = ex.R
ql = mdptoolbox.mdp.QLearning(
    P, # transitions
    R, # rewards
    0.3, # discount
    n_iter=10000,
)
ql.run()
# print('q learning Q matrix:', ql.Q)
print('q learning value function:', ql.V)
print('q learning mean discrepancy:', ql.mean_discrepancy)
print('q learning best policy:', ql.policy)