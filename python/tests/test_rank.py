import alpha.algo as algo
import numpy as np
import time


class Context:
  def __init__(self, start=0, groups=1, flags=0):
    self.start = start
    self.groups = groups
    self.flags = flags


_ALGO_CTX_ = Context()


a = np.random.rand(5000_0000)

_ALGO_CTX_.groups = 100
t1 = time.time()
algo.RANK(a)
t2 = time.time()
print(len(a), t2 - t1)
