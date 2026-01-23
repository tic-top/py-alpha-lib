List of available functions with python type hints:

the `np.ndarray` is `ndarray` type in `numpy` package

- BARSLAST(input: np.ndarray[bool]): Calculate number of bars since last condition true
- BARSSINCE(input: np.ndarray[bool]): Calculate number of bars since first condition true
- COUNT(input: np.ndarray[bool], periods: int): Calculate number of periods where condition is true in passed `periods` window
- CROSS(a: np.ndarray[float], b: np.ndarray[float]): For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i] alias: golden_cross, cross_ge
- DMA(input: np.ndarray[float], weight: float): Exponential Moving Average current = weight * current + (1 - weight) * previous
- EMA(input: np.ndarray[float], periods: int): Exponential Moving Average (variant of well-known EMA) weight = 2 / (n + 1)
- HHV(input: np.ndarray[float], periods: int): Find highest value in a preceding `periods` window
- HHVBARS(input: np.ndarray[float], periods: int): The number of periods that have passed since the array reached its `periods` period high
- LLV(input: np.ndarray[float], periods: int): Find lowest value in a preceding `periods` window
- LLVBARS(input: np.ndarray[float], periods: int): The number of periods that have passed since the array reached its periods period low
- LONGCROSS(a: np.ndarray[float], b: np.ndarray[float], n: int): For 2 arrays A and B, return true if previous N periods A < B, Current A >= B
- MA(input: np.ndarray[float], periods: int): Simple Moving Average, also known as arithmetic moving average
- RANK(input: np.ndarray[float]): Calculate rank cross group dimension, the ctx.groups() is the number of groups
- RCROSS(a: np.ndarray[float], b: np.ndarray[float]): For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i] alias: death_cross, cross_le
- REF(input: np.ndarray[float], periods: int): Right shift input array by `periods`, r[i] = input[i - periods]
- RLONGCROSS(a: np.ndarray[float], b: np.ndarray[float], n: int): For 2 arrays A and B, return true if previous N periods A > B, Current A <= B
- SMA(input: np.ndarray[float], n: int, m: int): Exponential Moving Average (variant of well-known EMA) weight = m / n
- SUM(input: np.ndarray[float], periods: int): Calculate sum of values in preceding `periods` window  If periods is 0, it calculates the cumulative sum from the first valid value.
- SUMBARS(input: np.ndarray[float], amount: float): Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
- TS_RANK(input: np.ndarray[float], periods: int): Calculate rank in a sliding window with size `periods`
