from itertools import permutations
import numpy as np

nums = ['+4', '-1', '-0', '-0']

seen = set()
res = []

for p in permutations(nums, len(nums)):
    if p in seen:
        continue
    seen.add(p)
    s = "[" + "".join(str(x) for x in p) + "]"  + ", 170, 2, 3230, 598, 3828, 1778, 13026"
    res.append(s)

for s in res:
    print(s)
