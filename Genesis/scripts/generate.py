from itertools import permutations
import numpy as np

nums = ['-2', '-2', '-1', '-0', '-0']

seen = set()
res = []

for p in permutations(nums, len(nums)):
    if p in seen:
        continue
    seen.add(p)
    s = "[" + "".join(str(x) for x in p) + "]" + ", 170, 3, 3232, 596, 3828, 1538, 11266"
    res.append(s)

for s in res:
    print(s)
