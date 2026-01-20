# THIS FILE IS AUTO-GENERATED, DO NOT EDIT

import numpy as np
from . import _algo

def BARSLAST(
  input: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
  """
  Bars since last condition true
  
  Ref: https://www.amibroker.com/guide/afl/barslast.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x, dtype=float) for x in input]
    input = [x.astype(bool) for x in input]
    _algo.barslast(r, input)
    return r
  else:
    r = np.empty_like(input, dtype=float)
    input = input.astype(bool)
    _algo.barslast(r, input)
    return r

def BARSSINCE(
  input: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
  """
  Bars since first condition true
  
  Ref: https://www.amibroker.com/guide/afl/barssince.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x, dtype=float) for x in input]
    input = [x.astype(bool) for x in input]
    _algo.barssince(r, input)
    return r
  else:
    r = np.empty_like(input, dtype=float)
    input = input.astype(bool)
    _algo.barssince(r, input)
    return r

def COUNT(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Count periods where condition is true
  
  Ref: https://www.amibroker.com/guide/afl/count.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x, dtype=float) for x in input]
    input = [x.astype(bool) for x in input]
    _algo.count(r, input, periods)
    return r
  else:
    r = np.empty_like(input, dtype=float)
    input = input.astype(bool)
    _algo.count(r, input, periods)
    return r

def DMA(
  input: np.ndarray | list[np.ndarray], alpha: float
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average
  
  https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  
  current = alpha * current + (1 - alpha) * previous
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.dma(r, input, alpha)
    return r
  else:
    r = np.empty_like(input)
    _algo.dma(r, input, alpha)
    return r

def HHV(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Highest High Value
  
  Ref: https://www.amibroker.com/guide/afl/hhv.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.hhv(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.hhv(r, input, periods)
    return r

def HHVBARS(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Bars since Highest High Value
  
  Ref: https://www.amibroker.com/guide/afl/hhvbars.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.hhvbars(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.hhvbars(r, input, periods)
    return r

def LLV(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Lowest Low Value
  
  Ref: https://www.amibroker.com/guide/afl/llv.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.llv(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.llv(r, input, periods)
    return r

def LLVBARS(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Bars since Lowest Low Value
  
  Ref: https://www.amibroker.com/guide/afl/llvbars.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.llvbars(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.llvbars(r, input, periods)
    return r

def MA(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Moving Average
  
  https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
  
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ma(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.ma(r, input, periods)
    return r

def REF(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Reference to value N periods ago
  
  Ref: https://www.amibroker.com/guide/afl/ref.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ref(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.ref(r, input, periods)
    return r

def SMA(
  input: np.ndarray | list[np.ndarray], n: int, m: int
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average (variant of EMA)
  
  alpha = m / n
  
  https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.sma(r, input, n, m)
    return r
  else:
    r = np.empty_like(input)
    _algo.sma(r, input, n, m)
    return r

