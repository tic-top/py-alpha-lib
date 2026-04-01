// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

//! Rolling mean absolute deviation (AVEDEV).
//!
//! AVEDEV(x, d) = mean(|x[i] - mean(x)|) over a window of d periods.

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Rolling mean absolute deviation over a window of `periods`.
///
/// AVEDEV(x, d) = (1/d) * sum(|x[i] - mean(x, d)|) for i in window.
pub fn ta_avedev<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  let r = ctx.align_end_mut(r);
  let input = ctx.align_end(input);

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());

      for i in start..r.len() {
        let win_start = if i + 1 >= periods { i + 1 - periods } else { 0 };

        if ctx.is_strictly_cycle() && (i - win_start + 1) < periods {
          continue;
        }

        // Collect valid values and compute mean
        let mut sum = NumT::zero();
        let mut count = 0usize;
        for j in win_start..=i {
          if is_normal(&x[j]) {
            sum = sum + x[j];
            count += 1;
          }
        }

        if count == 0 {
          continue;
        }

        let mean = sum / NumT::from(count).unwrap();

        // Compute mean absolute deviation
        let mut abs_dev_sum = NumT::zero();
        for j in win_start..=i {
          if is_normal(&x[j]) {
            abs_dev_sum = abs_dev_sum + (x[j] - mean).abs();
          }
        }

        r[i] = abs_dev_sum / NumT::from(count).unwrap();
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn ctx() -> Context {
    Context::new(0, 1, 0)
  }

  #[test]
  fn test_avedev_basic() {
    // x = [1, 2, 3, 4, 5], window=3
    // i=2: [1,2,3], mean=2, avedev = (|1-2|+|2-2|+|3-2|)/3 = 2/3
    // i=3: [2,3,4], mean=3, avedev = (1+0+1)/3 = 2/3
    // i=4: [3,4,5], mean=4, avedev = (1+0+1)/3 = 2/3
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = [0.0f64; 5];

    ta_avedev(&ctx(), &mut r, &x, 3).unwrap();
    assert!((r[2] - 2.0 / 3.0).abs() < 1e-9);
    assert!((r[3] - 2.0 / 3.0).abs() < 1e-9);
    assert!((r[4] - 2.0 / 3.0).abs() < 1e-9);
  }

  #[test]
  fn test_avedev_with_nan() {
    let x = [1.0, f64::NAN, 3.0, 4.0, 5.0];
    let mut r = [0.0f64; 5];

    // i=2: valid=[1,3], mean=2, avedev=(1+1)/2=1.0
    ta_avedev(&ctx(), &mut r, &x, 3).unwrap();
    assert!((r[2] - 1.0).abs() < 1e-9);
  }

  #[test]
  fn test_avedev_constant() {
    // All same values → avedev = 0
    let x = [3.0, 3.0, 3.0, 3.0];
    let mut r = [0.0f64; 4];

    ta_avedev(&ctx(), &mut r, &x, 3).unwrap();
    assert!((r[2] - 0.0).abs() < 1e-9);
    assert!((r[3] - 0.0).abs() < 1e-9);
  }
}
