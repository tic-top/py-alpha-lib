# alpha-lib

High-performance quantitative finance algorithm library, implemented in Rust with Python bindings (PyO3).

Provides efficient rolling-window calculations commonly used in factor-based quantitative trading.

## Performance

Benchmarked on Alpha 101, 4000 stocks x 261 trading days (1,044,000 data points per factor):

| Implementation | Factors | Data Load | Compute | Total | Speedup |
|---|---|---|---|---|---|
| pandas | 75 | 31.2s | 2,643s | 2,675s (44min) | 1x |
| polars_ta | 81 | 0.3s | 58s | 58s | 46x |
| **alpha-lib** | **101** | **0.3s** | **3.6s** | **3.9s** | **729x** |

See [COMPARISON.md](COMPARISON.md) for per-factor timing and correctness analysis.

## Installation

```bash
pip install git+https://github.com/msd-rs/py-alpha-lib.git
```

> Requires Rust toolchain installed — `pip` will invoke `maturin` to compile the Rust extension automatically.

## Usage

### Context Settings

Control computation behavior via `alpha.set_ctx()`:

- **`groups`** — Number of securities in the data array. Each group is processed independently and in parallel. Required for cross-sectional operations like `RANK`.
- **`start`** — Starting index for calculation (default: 0).
- **`flags`** — Bitwise flags:
  - `FLAG_SKIP_NAN` (1): Skip NaN values in rolling windows.
  - `FLAG_STRICTLY_CYCLE` (2): Return NaN until window is full (matches pandas `rolling()` default).
  - Combine with `|`: `flags=FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`

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


### Example 1: Plug and Play

```python
import alpha
from alpha.context import ExecContext

# ExecContext auto-infers groups from securityid/tradetime columns
# and calls alpha.set_ctx(groups=...) automatically
data = pl.read_csv("data.csv").sort(["securityid", "tradetime"])
ctx = ExecContext(data)

# Call operators directly on numpy arrays
close = data["close"].to_numpy()
ma20 = alpha.MA(close, 20)
rank = alpha.RANK(close)       # cross-sectional rank (groups auto-configured)
corr = alpha.CORR(close, data["vol"].to_numpy().astype(float), 10)
```

Data layout: flat 1D array `[stock1_day1, stock1_day2, ..., stockN_dayM]`, sorted by security then time. The `groups` parameter tells the library where each stock's data begins.


### Example 2: Factor Expression Transpiler

Convert factor expressions to Python code, then run:

```bash
python -m alpha.lang examples/wq101/alpha101.txt
```

```python
# 3. Use generated code
from alpha.context import ExecContext
from factors import alpha_001

data = pl.read_csv("data.csv").sort(["securityid", "tradetime"])
ctx = ExecContext(data)  # auto-infers groups
result = alpha_001(ctx)
```

## Benchmarking

### GTJA Alpha 191

Implementation of 190/191 factors from the GTJA (国泰君安) Alpha 191 factor set in [`examples/gtja191/`](examples/gtja191/):

| Metric | Value |
|---|---|
| Computable | 190 / 191 |
| Compute time | ~4.5s (4000 stocks × 261 days) |
| Avg per factor | 24ms |

```bash
python -m examples.gtja191.al 143     # run specific factor
python -m examples.gtja191.al          # run all factors
```

### WorldQuant Alpha 101

Full implementation of [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) in [`examples/wq101/`](examples/wq101/):

- `al/` — alpha-lib implementation (Rust backend)
- `pd_/` — pandas reference (DolphinDB port)
- `pl_/` — polars_ta reference

```bash
examples/wq101/main.py --with-al 1 2 3 4 # Run specific factors
examples/wq101/main.py --with-al -s 1 -e 102 # Run all factors
examples/wq101/main.py --with-pd --with-al -s 1 -e 15 # Compare with pandas
```

Benchmark scripts in [`benchmarks/`](benchmarks/).

### Supported Algorithms

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
| SCAN_ADD | Conditional cumulative sum (SELF recursion) |
| SCAN_MUL | Conditional cumulative product (SELF recursion) |
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
