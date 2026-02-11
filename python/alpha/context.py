# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""
Generic ExecContext for alpha factor computation.

Supports both WorldQuant Alpha 101 and GTJA Alpha 191 naming conventions.
All method names from both DSLs are available — e.g., CORRELATION and CORR
both map to alpha.CORR.

Usage:
  import alpha
  from alpha.context import ExecContext

  data = pl.read_csv("data.csv").sort(["securityid", "tradetime"])
  securities = data["securityid"].n_unique()
  trades = data["tradetime"].n_unique()
  alpha.set_ctx(groups=securities)

  ctx = ExecContext(data, securities=securities, trades=trades)
"""

import numpy as np
import alpha


def _returns(a: np.ndarray) -> np.ndarray:
  return a / alpha.REF(a, 1) - 1


class ExecContext:
  """
  Unified execution context for alpha factor expressions.

  Provides:
    - OHLCV data fields (OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP)
    - Derived fields (RETURNS/RET, DTM, DBM, TR, HD, LD, SEQUENCE)
    - All operator methods used by both wq101 and gtja191 transpiler output
    - Dynamic ADV{n} access (average daily volume over n days)
    - Variable aliases (AMOUNT=VOLUME, VOL=VOLUME)

  Args:
    data: DataFrame (polars or pandas) with columns: open, high, low, close, vol, vwap
    securities: Number of securities (stocks). Required for derived fields.
    trades: Number of trading days. Required for derived fields.
  """

  def __init__(self, data, securities: int = 0, trades: int = 0):
    self.OPEN = data["open"].to_numpy()
    self.HIGH = data["high"].to_numpy()
    self.LOW = data["low"].to_numpy()
    self.CLOSE = data["close"].to_numpy()
    self.VOLUME = data["vol"].to_numpy().astype(np.float64)
    self.VWAP = data["vwap"].to_numpy()
    self.RETURNS = _returns(self.CLOSE)
    self.RET = self.RETURNS

    if securities > 0 and trades > 0:
      self.BANCHMARKINDEXCLOSE = np.repeat(self.CLOSE[0:trades], securities)
      self.BANCHMARKINDEXOPEN = np.repeat(self.OPEN[0:trades], securities)
      self.DTM = self._calc_DTM()
      self.DBM = self._calc_DBM()
      self.TR = self._calc_TR()
      self.HD = self.HIGH - alpha.REF(self.HIGH, 1)
      self.LD = self.LOW - alpha.REF(self.LOW, 1)
      self._SEQUENCE = np.tile(
        np.arange(1, trades + 1, dtype=np.float64), securities
      )

  def __call__(self, name: str) -> np.ndarray:
    if name.startswith("ADV"):
      n = name[3:]
      if len(n) == 0:
        return self.VOLUME
      return alpha.MA(self.VOLUME, int(n))
    if name == "AMOUNT" or name == "VOL":
      return self.VOLUME
    if name == "SEQUENCE":
      return self._SEQUENCE
    return getattr(self, name)

  # ── Derived field helpers ──────────────────────────────────────────

  def _calc_DTM(self):
    return np.where(
      self.OPEN <= alpha.REF(self.OPEN, 1),
      0,
      np.maximum(
        self.HIGH - self.OPEN, self.OPEN - alpha.REF(self.OPEN, 1)
      ),
    )

  def _calc_DBM(self):
    return np.where(
      self.OPEN >= alpha.REF(self.OPEN, 1),
      0,
      np.maximum(
        self.OPEN - self.LOW, self.OPEN - alpha.REF(self.OPEN, 1)
      ),
    )

  def _calc_TR(self):
    return np.maximum(
      np.maximum(
        self.HIGH - self.LOW,
        np.abs(self.HIGH - alpha.REF(self.CLOSE, 1)),
      ),
      np.abs(self.LOW - alpha.REF(self.CLOSE, 1)),
    )

  # ── Rolling window operators ───────────────────────────────────────

  def SUM(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.SUM(a, int(w))

  def SUMAC(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.SUM(a, int(w))

  def MA(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.MA(a, int(w))

  def MEAN(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.MA(a, int(w))

  def SMA(self, a, *args) -> np.ndarray:
    """SMA with 2 args = simple MA, with 3 args = EMA variant (weight=m/n)."""
    if len(args) == 1:
      # wq101 style: SMA(data, window) = simple moving average
      return alpha.MA(a, int(args[0]))
    else:
      # gtja191 style: SMA(data, m, n) = EMA variant
      return alpha.SMA(a, int(args[0]), int(args[1]))

  def EMA(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.EMA(a, int(w))

  def STDDEV(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.STDDEV(a, int(w))

  def STD(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.STDDEV(a, int(w))

  def VAR(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.VAR(a, int(w))

  # ── Correlation / Covariance ───────────────────────────────────────

  def CORR(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.CORR(a, b, int(w))

  def CORRELATION(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.CORR(a, b, int(w))

  def COV(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.COV(a, b, int(w))

  def COVARIANCE(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.COV(a, b, int(w))

  # ── Extremes ───────────────────────────────────────────────────────

  def TSMAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.HHV(a, int(w))

  def TS_MAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.HHV(a, int(w))

  def TSMIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LLV(a, int(w))

  def TS_MIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LLV(a, int(w))

  def MAX(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)

  def MIN(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

  def HIGHDAY(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.HHVBARS(a, int(w))

  def LOWDAY(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LLVBARS(a, int(w))

  # ── Rank ───────────────────────────────────────────────────────────

  def RANK(self, a: np.ndarray) -> np.ndarray:
    return alpha.RANK(np.asarray(a, dtype=np.float64))

  def TSRANK(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.TS_RANK(a, int(w))

  def TS_RANK(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.TS_RANK(a, int(w))

  def TS_ARGMAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return int(w) - alpha.HHVBARS(a, int(w))

  def TS_ARGMIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return int(w) - alpha.LLVBARS(a, int(w))

  # ── Shift / Delta ──────────────────────────────────────────────────

  def DELTA(self, a: np.ndarray, p: int) -> np.ndarray:
    return a - alpha.REF(a, int(p))

  def DELAY(self, a: np.ndarray, p: int) -> np.ndarray:
    return alpha.REF(a, int(p))

  # ── Weighted averages ──────────────────────────────────────────────

  def DECAY_LINEAR(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LWMA(np.asarray(a, dtype=np.float64), int(w))

  def DECAYLINEAR(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LWMA(np.asarray(a, dtype=np.float64), int(w))

  def WMA(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LWMA(np.asarray(a, dtype=np.float64), int(w))

  # ── Products ───────────────────────────────────────────────────────

  def PRODUCT(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.PRODUCT(a, int(w))

  def PROD(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.PRODUCT(a, int(w))

  # ── Regression ─────────────────────────────────────────────────────

  def REGBETA(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.REGBETA(a, b, int(w))

  def REGRESI(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.REGRESI(a, b, int(w))

  # ── Conditional / counting ─────────────────────────────────────────

  def COUNT(self, cond: np.ndarray, w: int) -> np.ndarray:
    return alpha.COUNT(cond.astype(np.float64), int(w))

  def SUMIF(self, a: np.ndarray, w: int, cond) -> np.ndarray:
    return alpha.SUMIF(
      np.asarray(a, dtype=np.float64),
      np.asarray(cond, dtype=np.float64),
      int(w),
    )

  def FILTER(self, a: np.ndarray, cond: np.ndarray) -> np.ndarray:
    return np.where(cond, a, np.nan)

  # ── Math ───────────────────────────────────────────────────────────

  def SCALE(self, a: np.ndarray, k: int = 1) -> np.ndarray:
    return a * k / np.abs(a).sum()

  def SIGNEDPOWER(self, a: np.ndarray, p) -> np.ndarray:
    return np.sign(a) * np.power(np.abs(a), p)

  def LOG(self, a: np.ndarray) -> np.ndarray:
    return np.log(a)

  def ABS(self, a: np.ndarray) -> np.ndarray:
    return np.abs(a)

  def SIGN(self, a: np.ndarray) -> np.ndarray:
    return np.sign(a)
