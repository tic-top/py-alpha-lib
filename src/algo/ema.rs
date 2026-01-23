use num_traits::Float;

use super::{Context, Error, is_normal};
use rayon::prelude::*;

/// Exponential Moving Average (variant of well-known EMA) weight = 2 / (n + 1)
///
/// Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
///
pub fn ta_ema<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  let alpha = NumT::from(2.0).unwrap() / NumT::from(periods + 1).unwrap();

  ema_impl(ctx, r, input, alpha, periods)
}

/// Exponential Moving Average (variant of well-known EMA) weight = m / n
///
/// Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
///
pub fn ta_sma<NumT: Float + Send + Sync>(
  _ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  n: usize,
  m: usize,
) -> Result<(), Error> {
  let alpha = NumT::from(m).unwrap() / NumT::from(n).unwrap();
  ta_dma(_ctx, r, input, alpha)
}

/// Exponential Moving Average
/// current = weight * current + (1 - weight) * previous
///
/// Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
///
pub fn ta_dma<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  weight: NumT,
) -> Result<(), Error> {
  ema_impl(ctx, r, input, weight, 0)
}

pub fn ema_impl<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  weight: NumT,
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if weight < NumT::zero() || weight > NumT::one() {
    return Err(Error::InvalidParameter(
      "alpha must be between 0 and 1".to_string(),
    ));
  }

  let k = NumT::one() - weight;

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, i)| {
      let mut prev = i[0];
      let total = r.len();
      for (n, (r, c)) in r
        .iter_mut()
        .zip(i.iter())
        .enumerate()
        .skip(ctx.start(total))
      {
        if ctx.is_skip_nan() && !is_normal(c) {
          *r = NumT::nan();
          continue;
        }
        if ctx.is_strictly_cycle() && n < periods {
          *r = NumT::nan();
        } else {
          *r = weight * *c + k * prev;
        }
        prev = *r;
      }
    });
  Ok(())
}
