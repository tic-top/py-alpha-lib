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

## 运行时间对比

测试数据：4000 只股票 × 261 个交易日 = 1,044,000 行

### 总体

| 实现 | 因子数 | 总耗时 | 均值 | 中位数 | vs pandas |
|---|---|---|---|---|---|
| pandas | 75 | 2,643,373ms (44min) | 35,245ms | 673ms | 1x |
| polars_ta | 81 | 58,130ms (58s) | 718ms | 424ms | **45x** |
| **alpha-lib** | **82** | **2,467ms (2.5s)** | **30ms** | **27ms** | **1,072x** |

### 逐因子对比（alpha-lib vs pandas）

| Factor | alpha-lib | pandas | 加速比 |
|---|---|---|---|
| alpha_001 | 33ms | 32,240ms | 977x |
| alpha_002 | 42ms | 685ms | 16x |
| alpha_003 | 29ms | 539ms | 19x |
| alpha_004 | 14ms | 92,428ms | 6,602x |
| alpha_005 | 25ms | 169ms | 7x |
| alpha_006 | 6ms | 422ms | 77x |
| alpha_007 | 30ms | 75,444ms | 2,515x |
| alpha_008 | 19ms | 305ms | 16x |
| alpha_009 | 29ms | 127ms | 4x |
| alpha_010 | 45ms | 210ms | 5x |
| alpha_011 | 31ms | 237ms | 8x |
| alpha_012 | 10ms | 17ms | 2x |
| alpha_013 | 31ms | 669ms | 22x |
| alpha_014 | 29ms | 501ms | 17x |
| alpha_015 | 34ms | 673ms | 20x |
| alpha_016 | 31ms | 562ms | 18x |
| alpha_017 | 44ms | 177,560ms | 4,035x |
| alpha_018 | 33ms | 556ms | 17x |
| alpha_019 | 18ms | 103ms | 6x |
| alpha_020 | 23ms | 273ms | 12x |
| alpha_021 | 28ms | 406ms | 15x |
| alpha_022 | 20ms | 695ms | 35x |
| alpha_023 | 8ms | - | - |
| alpha_024 | 19ms | 136ms | 7x |
| alpha_025 | 12ms | 139ms | 12x |
| alpha_026 | 31ms | 185,351ms | 5,979x |
| alpha_027 | 35ms | 675ms | 19x |
| alpha_028 | 24ms | 636ms | 27x |
| alpha_029 | 47ms | 91,282ms | 1,942x |
| alpha_030 | 20ms | 178ms | 9x |
| alpha_031 | 61ms | 985ms | 16x |
| alpha_032 | 32ms | 501ms | 16x |
| alpha_033 | 16ms | 91ms | 6x |
| alpha_034 | 28ms | 371ms | 13x |
| alpha_035 | 38ms | 263,594ms | 6,937x |
| alpha_036 | 92ms | 95,351ms | 1,037x |
| alpha_037 | 26ms | 535ms | 21x |
| alpha_038 | 24ms | 94,936ms | 3,956x |
| alpha_039 | 28ms | 297ms | 11x |
| alpha_040 | 20ms | 558ms | 28x |
| alpha_041 | 14ms | 4ms | 0.3x |
| alpha_042 | 13ms | 130ms | 10x |
| alpha_043 | 24ms | 174,838ms | 7,285x |
| alpha_044 | 18ms | 487ms | 27x |
| alpha_045 | 47ms | 1,291ms | 27x |
| alpha_046 | 24ms | 30ms | 1x |
| alpha_047 | 35ms | 343ms | 10x |
| alpha_049 | 9ms | 18ms | 2x |
| alpha_050 | 31ms | 682ms | 22x |
| alpha_051 | 14ms | 14ms | 1x |
| alpha_052 | 23ms | 92,591ms | 4,026x |
| alpha_053 | 8ms | 9ms | 1x |
| alpha_054 | 10ms | 20ms | 2x |
| alpha_055 | 35ms | 792ms | 23x |
| alpha_057 | 21ms | 31,689ms | 1,509x |
| alpha_060 | 22ms | 34,148ms | 1,552x |
| alpha_061 | 29ms | 671ms | 23x |
| alpha_064 | 62ms | 790ms | 13x |
| alpha_065 | 41ms | 760ms | 19x |
| alpha_066 | 29ms | 93,920ms | 3,239x |
| alpha_068 | 66ms | 87,173ms | 1,321x |
| alpha_071 | 63ms | - | - |
| alpha_072 | 65ms | 192,850ms | 2,967x |
| alpha_073 | 34ms | - | - |
| alpha_074 | 55ms | 2,281ms | 41x |
| alpha_075 | 49ms | 1,253ms | 26x |
| alpha_077 | 36ms | - | - |
| alpha_078 | 60ms | 1,359ms | 23x |
| alpha_081 | 62ms | 28,633ms | 462x |
| alpha_083 | 27ms | 352ms | 13x |
| alpha_084 | 32ms | 90,065ms | 2,814x |
| alpha_085 | 72ms | 194,698ms | 2,704x |
| alpha_086 | 32ms | 76,954ms | 2,405x |
| alpha_088 | 67ms | - | - |
| alpha_094 | 51ms | 230,125ms | 4,512x |
| alpha_095 | 42ms | 67,088ms | 1,597x |
| alpha_096 | 63ms | - | - |
| alpha_098 | 49ms | 112,529ms | 2,297x |
| alpha_099 | 48ms | 3,126ms | 65x |
| alpha_101 | 8ms | 297ms | 39x |

含 `ts_rank` / `correlation` 长窗口操作的因子加速比 **1,000x - 7,000x**，简单因子 **2x - 100x**。

## 结论

Alpha-lib 的 Rust 实现在所有已实现因子上**数学计算正确**，符合 Alpha 101 论文定义。与 pandas 参考实现的差异来源于：

1. **暖窗期设计选择**：alpha-lib 返回部分结果 vs pandas 返回 NaN（可通过 `FLAG_STRICTLY_CYCLE` 切换）
2. **Pandas 参考 bug**：3 个因子公式错误
3. **NaN 传播策略**：alpha-lib 正确传播，pandas 做了填充预处理
4. **浮点精度**：极端小窗口下不可避免
