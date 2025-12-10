from itertools import permutations
import numpy as np

nums = ['-1', '-0', '-0', '-0', '-0']

seen = set()
res = []

for p in permutations(nums, len(nums)):
    if p in seen:
        continue
    seen.add(p)
    s = "[" + "".join(str(x) for x in p) + "]"  + ", 1, 1, 0, 1, 1, 1, 20"
    res.append(s)

for s in res:
    print(s)
