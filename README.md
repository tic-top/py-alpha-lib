# alpha-lib

High-performance quantitative finance algorithm library, implemented in Rust with Python bindings (PyO3).

Provides efficient rolling-window calculations commonly used in factor-based quantitative trading.

## Performance

Benchmarked on Alpha 101, 4000 stocks x 261 trading days (1,044,000 data points per factor):

| Implementation | Factors | Data Load | Compute | Total | Speedup |
|---|---|---|---|---|---|
| pandas | 75 | 31.2s | 2,643s | 2,675s (44min) | 1x |
| polars_ta | 81 | 0.3s | 58s | 58s | 46x |
| **alpha-lib** | **82** | **0.2s** | **2.9s** | **3.2s** | **847x** |

See [COMPARISON.md](COMPARISON.md) for per-factor timing and correctness analysis.

## Installation

```bash
pip install py-alpha-lib
```

## Quick Start

```python
import alpha
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

# 3-period moving average (partial results during warm-up)
result = alpha.MA(data, 3)
# [1.  1.5 2.  3.  4.  5.  6.  7.  8.  9.]

# Strict mode: NaN until window is full
alpha.set_ctx(flags=alpha.FLAG_STRICTLY_CYCLE)
result = alpha.MA(data, 3)
# [nan nan 2.  3.  4.  5.  6.  7.  8.  9.]

# Skip NaN values
alpha.set_ctx(flags=alpha.FLAG_SKIP_NAN)
data_nan = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
result = alpha.MA(data_nan, 3)
# [1.  1.5 nan 3.  4.5 5.  6.  7.  8.  9.]
```

## Context Settings

Control computation behavior via `alpha.set_ctx()`:

- **`groups`** — Number of securities in the data array. Each group is processed independently and in parallel. Required for cross-sectional operations like `RANK`.
- **`start`** — Starting index for calculation (default: 0).
- **`flags`** — Bitwise flags:
  - `FLAG_SKIP_NAN` (1): Skip NaN values in rolling windows.
  - `FLAG_STRICTLY_CYCLE` (2): Return NaN until window is full (matches pandas `rolling()` default).
  - Combine with `|`: `flags=FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`

## Algorithms

| Name | Description |
|---|---|
| BARSLAST | Bars since last condition true |
| BARSSINCE | Bars since first condition true |
| BINS | Discretize input into n bins |
| CORR | Rolling correlation |
| COUNT | Count of true values in window |
| COV | Rolling covariance |
| CROSS | Golden cross detection (A crosses above B) |
| DMA | Exponential moving average (custom weight) |
| EMA | Exponential moving average (weight = 2/(n+1)) |
| FRET | Future return calculation |
| GROUP_RANK | Rank percentage within category group |
| GROUP_ZSCORE | Z-score within category group |
| HHV / LLV | Highest / lowest value in window |
| HHVBARS / LLVBARS | Bars since highest / lowest value |
| INTERCEPT | Linear regression intercept |
| LONGCROSS | A < B for N periods then A >= B |
| LWMA | Linear weighted moving average |
| MA | Simple moving average |
| NEUTRALIZE | Neutralize categorical effect |
| PRODUCT | Rolling product |
| RANK | Cross-sectional rank (percentage) |
| RCROSS | Death cross detection (A crosses below B) |
| REF | Shift array by N periods |
| REGBETA | Regression coefficient (beta) |
| REGRESI | Regression residual |
| RLONGCROSS | A > B for N periods then A <= B |
| SLOPE | Linear regression slope |
| SMA | EMA variant (weight = m/n) |
| STDDEV | Rolling standard deviation |
| SUM | Rolling sum (0 = cumulative) |
| SUMBARS | Bars until sum reaches threshold |
| SUMIF | Conditional rolling sum |
| TS_BACKFILL | Forward-fill NaN with last valid value |
| TS_CORR | Time series correlation |
| TS_COUNT_NANS | Rolling count of NaN values in window |
| TS_ENTROPY | Rolling Shannon entropy over binned window |
| TS_KURTOSIS | Rolling excess kurtosis (Fisher-adjusted) |
| TS_MIN_MAX_DIFF | Rolling range (max - min) over window |
| TS_MOMENT | Rolling k-th central moment |
| TS_RANK | Rank within sliding window |
| TS_SKEWNESS | Rolling skewness (Fisher-Pearson adjusted) |
| TS_WEIGHTED_DELAY | Exponentially weighted lag (LWMA of lagged series) |
| TS_ZSCORE | Rolling z-score over window |
| VAR | Rolling variance |
| ZSCORE | Cross-sectional z-score |

Full function signatures: [python/alpha/algo.md](python/alpha/algo.md)

## Factor Expression Transpiler

Convert factor expressions to Python code:

```bash
python -m alpha.lang examples/wq101/alpha101.txt
```

Reads expressions from [`examples/wq101/alpha101.txt`](examples/wq101/alpha101.txt) and generates Python code using alpha-lib functions.

## WorldQuant Alpha 101

Full implementation of [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) in [`examples/wq101/`](examples/wq101/):

- `al/` — alpha-lib implementation (Rust backend)
- `pd_/` — pandas reference (DolphinDB port)
- `pl_/` — polars_ta reference

```bash
# Run specific factors
examples/wq101/main.py --with-al 1 2 3 4

# Run all factors
examples/wq101/main.py --with-al -s 1 -e 102

# Compare with pandas
examples/wq101/main.py --with-pd --with-al -s 1 -e 15
```

Benchmark scripts in [`benchmarks/`](benchmarks/).

## Using in Your Project

### Install from PyPI

```bash
pip install py-alpha-lib
```

Or add to `requirements.txt`:

```
py-alpha-lib>=0.1.2
```

### Install from Source (GitHub)

```bash
pip install git+https://github.com/msd-rs/py-alpha-lib.git
```

Or in `requirements.txt`:

```
py-alpha-lib @ git+https://github.com/msd-rs/py-alpha-lib.git
```

Pin to a specific commit:

```
py-alpha-lib @ git+https://github.com/msd-rs/py-alpha-lib.git@main
```

> Requires Rust toolchain installed — `pip` will invoke `maturin` to compile the Rust extension automatically.

### Plug and Play

```python
import alpha
import numpy as np

# 1. Load your data as numpy arrays (one per field)
close = df["close"].to_numpy()
volume = df["vol"].to_numpy().astype(np.float64)

# 2. Configure context
alpha.set_ctx(
    groups=num_securities,     # number of stocks
    flags=alpha.FLAG_SKIP_NAN  # optional: skip NaN in rolling windows
)

# 3. Call operators directly
ma20 = alpha.MA(close, 20)
std20 = alpha.STDDEV(close, 20)
rank = alpha.RANK(close)  # cross-sectional rank (requires groups)
corr = alpha.CORR(close, volume, 10)

# 4. Or use the transpiler for factor expressions
#    python -m alpha.lang your_factors.txt > factors.py
```

Data layout: flat 1D array `[stock1_day1, stock1_day2, ..., stockN_dayM]`, sorted by security then time. The `groups` parameter tells the library where each stock's data begins.

## Development

Requirements:
- Rust (latest stable)
- Python 3.11+
- [maturin](https://github.com/PyO3/maturin)

```bash
# Build and install in development mode
maturin develop --release

# Run tests
cargo test
```

### Vibe Coding

When adding new algorithms with LLM assistance, provide [the function list](python/alpha/algo.md) as context. Use the skill [add_algo.md](.agent/skills/add_algo.md) for guided implementation.
