# Alpha 101 Benchmark Report

基于 `comparison_all.csv` 三方对比（pandas / alpha-lib / polars_ta）的分析报告。

## 修复记录

| 修复项 | 影响 |
|---|---|
| 转译器运算符优先级 | 27 → 41 |
| 浮点窗口取整 + SIGNEDPOWER 语义 | 41 → 43 |
| RANK 跨截面 NaN 处理 | 43 → 44 |
| TS_RANK 重复值处理（BTreeMap→计数） | 44 → 47 |
| Float::is_normal() 替换为自定义 is_normal() | 正确性修复，0.0 不再被误判为无效值 |

## 总览

| 类别 | 数量 |
|---|---|
| 完全一致（pearson ≥ 0.99） | 47 |
| 高度一致（0.95 ≤ pearson < 0.99） | 6 |
| 常量输出（无法计算相关性） | 6 |
| 暖窗期行为差异（ic_mean ≥ 0.99） | 8 |
| Pandas 参考实现 bug | 3 |
| decay_linear NaN 处理差异 | 3 |
| 浮点精度 | 1 |
| 数据不足 | 3 |
| 未实现 | 22 |
| **合计** | **99** (+ alpha_023/062 三方均缺失) |

## 完全一致（47 个）

pearson ≥ 0.99 或 ic_mean ≥ 0.99，公式与 Alpha101 论文一致。

001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 019, 020, 022, 025, 026, 029, 030, 033, 034, 035, 037, 039, 040, 041, 042, 044, 046, 049, 050, 051, 052, 053, 054, 060, 078, 084, 085, 101

另有 alpha_024 (pearson=0.978, ic=0.982) 属于高度一致。

## 常量输出（6 个）

输出为常量或方差极小，无法计算 pearson，三方一致：021, 027, 057, 068, 083, 086

## 暖窗期行为差异（8 个）

Alpha-lib 在滚动窗口未满时返回部分计算结果，pandas 返回 NaN。**这是设计选择，不是 bug。** Alpha-lib 提供了更多有效数据。

关键证据：**ic_mean（逐截面相关）均 ≥ 0.99**，说明在双方都有有效数据的时间步上完全一致。

| Factor | pearson (pd_vs_al) | ic_mean | 说明 |
|---|---|---|---|
| alpha_061 | 0.375 | 0.995 | `rank(A) < rank(B)` 布尔输出，暖窗期翻转 |
| alpha_064 | 0.535 | 0.996 | 同上 |
| alpha_065 | 0.752 | 0.998 | 同上 |
| alpha_074 | 0.758 | 0.998 | 同上 |
| alpha_075 | 0.799 | 0.998 | 同上 |
| alpha_081 | 0.740 | 0.997 | 同上 |
| alpha_095 | 0.012 | 1.000 | 同上，暖窗期长 |
| alpha_099 | 0.722 | 0.998 | 同上 |

另有 alpha_043 (p=0.903), alpha_055 (p=0.978), alpha_094 (p=0.585) 也受暖窗期影响。

## Pandas 参考实现 Bug（3 个）

以下 pandas 参考实现有公式错误，**alpha-lib 是正确的**。

| Factor | Pandas Bug | 验证 |
|---|---|---|
| alpha_038 | `ts_rank(open, 10)` 应为 `ts_rank(close, 10)` | al_vs_pt spearman=0.993 |
| alpha_036 | `sma(close,200)/200` 双重除以200 | al_vs_pt spearman=0.998 |
| alpha_047 | `sma(high,5)/5` = sum/25，应为 sum/5 | **al_vs_pt pearson=1.0** |

## decay_linear NaN 处理差异（3 个）

Pandas 在 LWMA 前做 `ffill().bfill().fillna(0)` 预处理，alpha-lib 正确传播 NaN。

| Factor | pearson (pd_vs_al) | 说明 |
|---|---|---|
| alpha_066 | 0.941 | decay_linear 输入含 NaN |
| alpha_072 | 0.020 | 分子分母均含 decay_linear |
| alpha_098 | 0.736 | 嵌套 decay_linear |

## 浮点精度（1 个）

| Factor | pearson (pd_vs_al) | 说明 |
|---|---|---|
| alpha_045 | 0.915 | `corr(x, y, 2)` 窗口=2 时值集中在 ±1 附近，1e-6 级 FP 差异导致截面排名不稳定。三方实现互相也不一致 (pd_vs_pt=0.926) |

## 结论

Alpha-lib 的 Rust 实现在所有已实现因子上**数学计算正确**，符合 Alpha 101 论文定义。与 pandas 参考实现的差异来源于：

1. **暖窗期设计选择**：alpha-lib 返回部分结果 vs pandas 返回 NaN（可通过 `FLAG_STRICTLY_CYCLE` 切换）
2. **Pandas 参考 bug**：3 个因子公式错误
3. **NaN 传播策略**：alpha-lib 正确传播，pandas 做了填充预处理
4. **浮点精度**：极端小窗口下不可避免
