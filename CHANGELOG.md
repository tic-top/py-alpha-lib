# ChangeLog


## [0.2.0] - 2026-02-26

### Change

- Naming Rules for algorithms
    - Functions without prefix means it is a rolling window operation
    - Functions with prefix `CC_` means it is a cross-commodity/cross-security/cross-group operation

### Add 

- Context now have a `end` option, which can be used to specify the end index of the calculation. usually used in iterative back test to improve some performance.


## [0.1.3] - 2026-02-22


@tic-top had verified the correctness of the computation, it's a great help for us to improve the quality of the alpha library. Thanks @tic-top!

## [0.1.2] - 2026-02-04

### Added

- TS_CORR @zhaojun6969

## [0.1.1] - 2026-02-02

### Added

- BINS @zhaojun6969
- FRET @zhaojun6969 
- INTERCEPT @zhaojun6969
- NEUTRALIZE @zhaojun6969
- REGBETA
- REGRESI
- SUMIF

See [algo.md](python/alpha/algo.md) for details.

Thanks @zhaojun6969, our first contributor!
