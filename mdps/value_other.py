import mdptoolbox.example
from helpers.open_ai_convert import OpenAI_MDPToolbox

ex = OpenAI_MDPToolbox('FrozenLake-v0', False)
P = ex.P
R = ex.R
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
