// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

//! Cutting operators (切割算子)
//!
//! Within a rolling window of `d` periods, sort by `y`, then aggregate `x`:
//! - `selmean_btm`: mean of the smallest `n` x-values (ranked by y)
//! - `selmean_top`: mean of the largest `n` x-values (ranked by y)
//! - `selmean_diff`: top mean − btm mean

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Collect valid (x, y) pairs in the window, sort by y, return mean of selected x values.
///
/// `pick_top = true`  → take the largest  n by y
/// `pick_top = false` → take the smallest n by y
#[inline]
fn selmean_window<NumT: Float>(
  x: &[NumT],
  y: &[NumT],
  start: usize,
  end: usize, // inclusive
  n: usize,
  pick_top: bool,
) -> NumT {
  // Gather valid pairs
  let mut pairs: Vec<(NumT, NumT)> = Vec::with_capacity(end - start + 1);
  for i in start..=end {
    if is_normal(&x[i]) && is_normal(&y[i]) {
      pairs.push((x[i], y[i]));
    }
  }

  if pairs.is_empty() || n == 0 {
    return NumT::nan();
  }

  let take = n.min(pairs.len());

  // Sort by y-value
  pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

  let selected = if pick_top {
    &pairs[pairs.len() - take..]
  } else {
    &pairs[..take]
  };

  let sum: NumT = selected.iter().fold(NumT::zero(), |acc, p| acc + p.0);
  sum / NumT::from(take).unwrap()
}

/// Mean of x for the bottom-n rows ranked by y over a rolling window of `periods`.
///
/// selmean_btm(x, y, d, n): over the past d periods, sort by y ascending, take
/// the bottom n rows, return mean(x) of those rows.
pub fn ta_selmean_btm<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  x: &[NumT],
  y: &[NumT],
  periods: usize,
  n: usize,
) -> Result<(), Error> {
  selmean_core(ctx, r, x, y, periods, n, false, false)
}

/// Mean of x for the top-n rows ranked by y over a rolling window of `periods`.
///
/// selmean_top(x, y, d, n): over the past d periods, sort by y ascending, take
/// the top n rows, return mean(x) of those rows.
pub fn ta_selmean_top<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  x: &[NumT],
  y: &[NumT],
  periods: usize,
  n: usize,
) -> Result<(), Error> {
  selmean_core(ctx, r, x, y, periods, n, true, false)
}

/// Difference between top-n mean and bottom-n mean of x ranked by y.
///
/// selmean_diff(x, y, d, n) = selmean_top(x, y, d, n) − selmean_btm(x, y, d, n)
pub fn ta_selmean_diff<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  x: &[NumT],
  y: &[NumT],
  periods: usize,
  n: usize,
) -> Result<(), Error> {
  selmean_core(ctx, r, x, y, periods, n, false, true)
}

fn selmean_core<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  x: &[NumT],
  y: &[NumT],
  periods: usize,
  n: usize,
  pick_top: bool,
  diff_mode: bool,
) -> Result<(), Error> {
  if r.len() != x.len() || x.len() != y.len() {
    return Err(Error::LengthMismatch(r.len(), x.len()));
  }

  let r = ctx.align_end_mut(r);
  let x = ctx.align_end(x);
  let y = ctx.align_end(y);

  let chunk_size = ctx.chunk_size(r.len());

  r.par_chunks_mut(chunk_size)
    .zip(x.par_chunks(chunk_size))
    .zip(y.par_chunks(chunk_size))
    .for_each(|((r, x), y)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());

      for i in start..r.len() {
        let win_start = if i + 1 >= periods { i + 1 - periods } else { 0 };

        if ctx.is_strictly_cycle() && (i - win_start + 1) < periods {
          continue;
        }

        if diff_mode {
          let top_val = selmean_window(x, y, win_start, i, n, true);
          let btm_val = selmean_window(x, y, win_start, i, n, false);
          if is_normal(&top_val) && is_normal(&btm_val) {
            r[i] = top_val - btm_val;
          }
        } else {
          r[i] = selmean_window(x, y, win_start, i, n, pick_top);
        }
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
  fn test_selmean_btm_basic() {
    // x = [10, 20, 30, 40, 50], y = [5, 1, 4, 2, 3]
    // window=5, n=2: sort by y → y=[1,2,3,4,5] → x=[20,40,50,30,10]
    // btm 2 by y: y=1→x=20, y=2→x=40 → mean = 30.0
    let x = [10.0, 20.0, 30.0, 40.0, 50.0];
    let y = [5.0, 1.0, 4.0, 2.0, 3.0];
    let mut r = [0.0f64; 5];

    ta_selmean_btm(&ctx(), &mut r, &x, &y, 5, 2).unwrap();
    assert!((r[4] - 30.0).abs() < 1e-9);
  }

  #[test]
  fn test_selmean_top_basic() {
    // Same data, top 2 by y: y=5→x=10, y=4→x=30 → mean = 20.0
    let x = [10.0, 20.0, 30.0, 40.0, 50.0];
    let y = [5.0, 1.0, 4.0, 2.0, 3.0];
    let mut r = [0.0f64; 5];

    ta_selmean_top(&ctx(), &mut r, &x, &y, 5, 2).unwrap();
    assert!((r[4] - 20.0).abs() < 1e-9);
  }

  #[test]
  fn test_selmean_diff_basic() {
    // diff = top(20.0) - btm(30.0) = -10.0
    let x = [10.0, 20.0, 30.0, 40.0, 50.0];
    let y = [5.0, 1.0, 4.0, 2.0, 3.0];
    let mut r = [0.0f64; 5];

    ta_selmean_diff(&ctx(), &mut r, &x, &y, 5, 2).unwrap();
    assert!((r[4] - (-10.0)).abs() < 1e-9);
  }

  #[test]
  fn test_selmean_with_nan() {
    let x = [10.0, f64::NAN, 30.0, 40.0, 50.0];
    let y = [5.0, 1.0, 4.0, 2.0, 3.0];
    let mut r = [0.0f64; 5];

    // NaN pair is excluded; 4 valid pairs remain
    // sorted by y: y=2→x=40, y=3→x=50, y=4→x=30, y=5→x=10
    // btm 2: y=2→x=40, y=3→x=50 → mean = 45.0
    ta_selmean_btm(&ctx(), &mut r, &x, &y, 5, 2).unwrap();
    assert!((r[4] - 45.0).abs() < 1e-9);
  }

  #[test]
  fn test_selmean_rolling() {
    // Verify rolling window behavior
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [5.0, 4.0, 3.0, 2.0, 1.0];
    let mut r = [0.0f64; 5];

    // window=3, n=1
    // i=0: win=[0..0], 1 pair → btm by y: (1,5) → mean=1.0
    // i=1: win=[0..1], 2 pairs, sorted by y: [(2,4),(1,5)] → btm 1: x=2 → 2.0
    // i=2: win=[0..2], 3 pairs, sorted by y: [(3,3),(2,4),(1,5)] → btm 1: x=3 → 3.0
    // i=3: win=[1..3], 3 pairs, sorted by y: [(4,2),(3,3),(2,4)] → btm 1: x=4 → 4.0
    // i=4: win=[2..4], 3 pairs, sorted by y: [(5,1),(4,2),(3,3)] → btm 1: x=5 → 5.0
    ta_selmean_btm(&ctx(), &mut r, &x, &y, 3, 1).unwrap();

    assert!((r[0] - 1.0).abs() < 1e-9);
    assert!((r[1] - 2.0).abs() < 1e-9);
    assert!((r[2] - 3.0).abs() < 1e-9);
    assert!((r[3] - 4.0).abs() < 1e-9);
    assert!((r[4] - 5.0).abs() < 1e-9);
  }

  #[test]
  fn test_selmean_n_exceeds_window() {
    // n > available pairs → use all pairs
    let x = [10.0, 20.0, 30.0];
    let y = [3.0, 1.0, 2.0];
    let mut r = [0.0f64; 3];

    // window=2, n=5 at i=2: pairs = [(20,1),(30,2)], n clamped to 2 → mean(20,30)=25
    ta_selmean_btm(&ctx(), &mut r, &x, &y, 2, 5).unwrap();
    assert!((r[2] - 25.0).abs() < 1e-9);
  }
}
