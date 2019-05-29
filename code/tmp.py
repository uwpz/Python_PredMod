
import numpy as np
states = [False, False, False, False, True, True, False, True, False, False, False, False, False, False, False, False]
a = np.where(states)[0]
a = np.nonzero(states)
states[list(a)]


a = np.arange(10)
np.random.seed(42)
np.random.shuffle(a)
b = np.array_split(a, 3)
del b[1]
b

a = [1,2]
b=1
b in a

a = True if 1==1 else False
a