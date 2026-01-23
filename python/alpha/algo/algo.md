# Algo
## barslast
Bars since last condition true

Ref: https://www.amibroker.com/guide/afl/barslast.html

## barssince
Bars since first condition true

Ref: https://www.amibroker.com/guide/afl/barssince.html

## count
Count periods where condition is true

Ref: https://www.amibroker.com/guide/afl/count.html

## cross
CROSS(A, B): Previous A < B, Current A >= B

Ref: https://www.amibroker.com/guide/afl/cross.html

## dma
Exponential Moving Average

https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

current = alpha * current + (1 - alpha) * previous


## hhv
Highest High Value

Ref: https://www.amibroker.com/guide/afl/hhv.html

## hhvbars
Bars since Highest High Value

Ref: https://www.amibroker.com/guide/afl/hhvbars.html

## llv
Lowest Low Value

Ref: https://www.amibroker.com/guide/afl/llv.html

## llvbars
Bars since Lowest Low Value

Ref: https://www.amibroker.com/guide/afl/llvbars.html

## longcross
LONGCROSS(A,B,N): Previous N A < B, Current A >= B

## ma
Moving Average

https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average



## rank
rank by group dim

## rcross
RCROSE(A, B): Previous A > B, Current A <= B

## ref
Reference to value N periods ago

Ref: https://www.amibroker.com/guide/afl/ref.html

## rlongcross
RLONGCROSS(A,B,N): Previous N A > B, Current A <= B

## sma
Exponential Moving Average (variant of EMA)

alpha = m / n

https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average


## sum
Sum of value N periods ago

If periods is 0, it calculates the cumulative sum from the first valid value.

Ref: https://www.amibroker.com/guide/afl/sum.html

## sumbars
Sums X backwards until the sum is greater than or equal to A

Returns the number of periods (bars) passed.

Ref: https://www.amibroker.com/guide/afl/sumbars.html

## ts_rank
rank by ts dim

