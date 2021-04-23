import time
import numpy as np

np.random.seed(0)

# start = time.time()
# import mdps.policy_forest
# end = time.time()
# print('policy forest time taken: ', end - start)

# start = time.time()
# import mdps.policy_other
# end = time.time()
# print('policy other time taken: ', end - start)

# start = time.time()
# import mdps.value_forest
# end = time.time()
# print('value forest time taken: ', end - start)

# start = time.time()
# import mdps.value_other
# end = time.time()
# print('value other time taken: ', end - start)

# start = time.time()
# import mdps.q_forest
# end = time.time()
# print('q forest time taken: ', end - start)

# start = time.time()
# import mdps.q_other
# end = time.time()
# print('q other time taken: ', end - start)

from mdptoolbox.openai import OpenAI_MDPToolbox

ex = OpenAI_MDPToolbox('taxi', True)
print(ex.R)
print(ex.P)
