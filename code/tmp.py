
from collections import defaultdict


dd = defaultdict(None, {"1st": "erster"})
dd.values()
dd = dd.setdefault(1)

tmp = df.pclass.map(dd)

tmp

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 100000
x = np.random.standard_normal(n)
y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
ax = axs[0]
hb = ax.hexbin(x, y, gridsize=50, cmap='inferno')


import matplotlib.pyplot as plt
x = range(5)
y = [20, 35, 30, 35, 27]
plt.barh(x,y,0.35)
plt.show()