// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

//! Rolling median over a moving window.

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Calculate median over a moving window.
///
/// For each position, collects valid (non-NaN) values in the window,
/// sorts them, and returns the middle value (average of two middle values
/// if the count is even).
pub fn ta_median<NumT: Float + Send + Sync>(
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

      let mut buf: Vec<NumT> = Vec::with_capacity(periods);

      for i in start..r.len() {
        let win_start = if i + 1 >= periods { i + 1 - periods } else { 0 };

        if ctx.is_strictly_cycle() && (i - win_start + 1) < periods {
          continue;
        }

        buf.clear();
        for j in win_start..=i {
          if is_normal(&x[j]) {
            buf.push(x[j]);
          }
        }

        if buf.is_empty() {
          continue;
        }

        buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = buf.len();
        r[i] = if n % 2 == 1 {
          buf[n / 2]
        } else {
          (buf[n / 2 - 1] + buf[n / 2]) / NumT::from(2).unwrap()
        };
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
  fn test_median_odd_window() {
    let x = [1.0, 5.0, 3.0, 2.0, 4.0];
    let mut r = [0.0f64; 5];

    // window=3:
    // i=0: [1] → 1.0
    // i=1: [1,5] → 3.0
    // i=2: [1,5,3] → 3.0
    // i=3: [5,3,2] → 3.0
    // i=4: [3,2,4] → 3.0
    ta_median(&ctx(), &mut r, &x, 3).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-9);
    assert!((r[1] - 3.0).abs() < 1e-9);
    assert!((r[2] - 3.0).abs() < 1e-9);
    assert!((r[3] - 3.0).abs() < 1e-9);
    assert!((r[4] - 3.0).abs() < 1e-9);
  }

  #[test]
  fn test_median_even_window() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let mut r = [0.0f64; 4];

    // window=4:
    // i=3: [1,2,3,4] → (2+3)/2 = 2.5
    ta_median(&ctx(), &mut r, &x, 4).unwrap();
    assert!((r[3] - 2.5).abs() < 1e-9);
  }

  #[test]
  fn test_median_with_nan() {
    let x = [1.0, f64::NAN, 3.0, 4.0, 2.0];
    let mut r = [0.0f64; 5];

    // window=3:
    // i=2: valid=[1,3] → (1+3)/2 = 2.0
    // i=3: valid=[3,4] → (3+4)/2 = 3.5
    // i=4: valid=[3,4,2] → 3.0
    ta_median(&ctx(), &mut r, &x, 3).unwrap();
    assert!((r[2] - 2.0).abs() < 1e-9);
    assert!((r[3] - 3.5).abs() < 1e-9);
    assert!((r[4] - 3.0).abs() < 1e-9);
  }

  #[test]
  fn test_median_strictly_cycle() {
    use crate::algo::context::FLAG_STRICTLY_CYCLE;
    let ctx = Context::new(0, 1, FLAG_STRICTLY_CYCLE);
    let x = [1.0, 5.0, 3.0, 2.0, 4.0];
    let mut r = [0.0f64; 5];

    // window=3, strict: first 2 should be NaN
    ta_median(&ctx, &mut r, &x, 3).unwrap();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
    assert!((r[2] - 3.0).abs() < 1e-9);
  }
}
