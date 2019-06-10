
from collections import defaultdict


dd = defaultdict(None, {"1st": "erster"})
dd.values()
dd = dd.setdefault(1)

tmp = df.pclass.map(dd)

tmp
