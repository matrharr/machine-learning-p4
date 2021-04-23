import time
import numpy as np
np.random.seed(0)

print('-------policy iteration------')
print('######### forest ############')
start = time.time()
import mdps.policy_forest
end = time.time()
print('policy forest time taken: ', end - start)

print('######### other ##############')
start = time.time()
import mdps.policy_other
end = time.time()
print('policy other time taken: ', end - start)

print('------value iteration---------')
print('######### forest ############')
start = time.time()
import mdps.value_forest
end = time.time()
print('value forest time taken: ', end - start)

print('######### other ##############')
start = time.time()
import mdps.value_other
end = time.time()
print('value other time taken: ', end - start)

print('------Q learning--------')
print('######### forest ############')
start = time.time()
import mdps.q_forest
end = time.time()
print('q forest time taken: ', end - start)

print('######### other ##############')
start = time.time()
import mdps.q_other
end = time.time()
print('q other time taken: ', end - start)
