// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::{cmp::Ordering, collections::BTreeMap, fmt::Debug};

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error};

#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
struct OrderedFloat<NumT: Float> {
  value: NumT,
}

impl<NumT: Float> OrderedFloat<NumT> {
  pub fn new(value: NumT) -> OrderedFloat<NumT> {
    OrderedFloat { value }
  }
}

impl<NumT: Float> Ord for OrderedFloat<NumT> {
  fn cmp(&self, other: &Self) -> Ordering {
    match self.value.partial_cmp(&other.value) {
      Some(ord) => ord,
      None => {
        // At least one value is NaN. Establish total order: NaN < everything.
        let self_nan = self.value.is_nan();
        let other_nan = other.value.is_nan();
        match (self_nan, other_nan) {
          (true, true) => Ordering::Equal,
          (true, false) => Ordering::Less,
          (false, true) => Ordering::Greater,
          (false, false) => Ordering::Equal, // unreachable in practice
        }
      }
    }
  }
}
impl<NumT: Float> Eq for OrderedFloat<NumT> {}

impl<NumT: Float> From<NumT> for OrderedFloat<NumT> {
  fn from(value: NumT) -> Self {
    OrderedFloat::new(value)
  }
}

impl<NumT: Float + Debug> Debug for OrderedFloat<NumT> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", self.value)
  }
}

/// Calculate rank in a sliding window with size `periods`
///
/// Uses min-rank method for ties (same as pandas rankdata method='min').
/// NaN values are treated as larger than all non-NaN values.
pub fn ta_rank<NumT: Float + Send + Sync>(
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

  if periods == 1 {
    r.fill(NumT::from(1.0).unwrap());
    return Ok(());
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());
      // Track counts per unique value to handle duplicates correctly.
      let mut rank_window: BTreeMap<OrderedFloat<NumT>, u32> = BTreeMap::new();
      let mut nan_count: usize = 0;
      let mut window_size: usize = 0;

      for i in start..x.len() {
        let val = x[i];

        // Add current value to window
        if val.is_nan() {
          nan_count += 1;
        } else {
          *rank_window.entry(val.into()).or_insert(0) += 1;
        }
        window_size += 1;

        // Remove oldest value if window exceeds periods
        if window_size > periods {
          let old_val = x[i - periods];
          if old_val.is_nan() {
            nan_count -= 1;
          } else {
            let old_key: OrderedFloat<NumT> = old_val.into();
            if let Some(count) = rank_window.get_mut(&old_key) {
              *count -= 1;
              if *count == 0 {
                rank_window.remove(&old_key);
              }
            }
          }
          window_size -= 1;
        }

        if ctx.is_strictly_cycle() && window_size < periods {
          continue;
        }

        // Calculate rank
        if val.is_nan() {
          // NaN gets the highest rank in the window
          r[i] = NumT::from(window_size).unwrap();
        } else {
          // Count all values strictly less than current (NaN treated as smallest)
          let val_key: OrderedFloat<NumT> = val.into();
          let mut less_count: usize = nan_count;
          for (key, count) in rank_window.iter() {
            if *key < val_key {
              less_count += *count as usize;
            } else {
              break;
            }
          }
          r[i] = NumT::from(less_count + 1).unwrap();
        }
      }
    });

  Ok(())
}

#[derive(Debug, Clone, Copy)]
struct UnsafePtr<NumT: Float> {
  ptr: *mut NumT,
  len: usize,
}

impl<NumT: Float> UnsafePtr<NumT> {
  pub fn new(ptr: *mut NumT, len: usize) -> Self {
    UnsafePtr { ptr, len }
  }

  pub fn get(&self) -> &mut [NumT] {
    unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
  }
}

unsafe impl<NumT: Float> Send for UnsafePtr<NumT> {}
unsafe impl<NumT: Float> Sync for UnsafePtr<NumT> {}

/// Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups
/// Same value are averaged
pub fn ta_cc_rank<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  let r = ctx.align_end_mut(r);
  let input = ctx.align_end(input);

  let group_size = ctx.chunk_size(r.len()) as usize;
  let groups = ctx.groups() as usize;

  if ctx.groups() < 2 {
    return ta_rank(ctx, r, input, 0);
  }

  if r.len() != group_size * groups {
    // ensure data is complete
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  let r = ctx.align_end_mut(r);
  let input = ctx.align_end(input);

  let r = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    // Separate NaN and valid values before sorting.
    // NaN never participates in sort — avoids Rust sort panics on broken total order.
    let mut valid_entries: Vec<(OrderedFloat<NumT>, usize)> = Vec::new();
    let r = r.get();

    for i in 0..groups {
      let idx = i * group_size + j;
      if input[idx].is_nan() {
        r[idx] = NumT::nan();
      } else {
        valid_entries.push((input[idx].into(), idx));
      }
    }

    let valid_count = valid_entries.len();
    if valid_count == 0 {
      return;
    }

    valid_entries.sort_by(|a, b| a.0.cmp(&b.0));

    let total = NumT::from(valid_count).unwrap();

    // Rank valid values only — chunk by same value, assign averaged rank percentage
    let mut prev_rank_value = valid_entries[0].0.value;
    let mut s = 0;

    for e in 0..valid_entries.len() {
      if prev_rank_value == valid_entries[e].0.value {
        continue;
      }
      let rank_avg =
        NumT::from(e + s + 1).unwrap() / NumT::from(2usize).unwrap();
      for i in s..e {
        r[valid_entries[i].1] = rank_avg / total;
      }
      s = e;
      prev_rank_value = valid_entries[e].0.value;
    }

    // the last chunk of valid values
    let rank_avg =
      NumT::from(valid_count + s + 1).unwrap() / NumT::from(2usize).unwrap();
    for i in s..valid_entries.len() {
      r[valid_entries[i].1] = rank_avg / total;
    }
  });

  Ok(())
}

/// Discretize the input into n bins, the ctx.groups() is the number of groups
///
/// Bins are 0-based index.
/// Same value are assigned to the same bin.
pub fn ta_bins<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  bins: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  let r = ctx.align_end_mut(r);
  let input = ctx.align_end(input);

  let group_size = ctx.chunk_size(r.len()) as usize;
  let groups = ctx.groups() as usize;

  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  let r = ctx.align_end_mut(r);
  let input = ctx.align_end(input);

  // If bins is 0 or 1, everything is bin 0 (or error?)
  // If bins=1, all 0.
  if bins == 0 {
    // Return 0 or error? Let's return 0.
    r.fill(NumT::zero());
    return Ok(());
  }

  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    // Separate NaN and valid values before sorting.
    let mut valid_entries: Vec<(OrderedFloat<NumT>, usize)> = Vec::new();
    let r = r_ptr.get();

    for i in 0..groups {
      let idx = i * group_size + j;
      if input[idx].is_nan() {
        r[idx] = NumT::nan();
      } else {
        valid_entries.push((input[idx].into(), idx));
      }
    }

    let valid_count = valid_entries.len();
    if valid_count == 0 {
      return;
    }

    valid_entries.sort_by(|a, b| a.0.cmp(&b.0));

    let total = NumT::from(valid_count).unwrap();
    let bins_t = NumT::from(bins).unwrap();

    // Chunk by same value, assign bin index
    let mut prev_rank_value = valid_entries[0].0.value;
    let mut s = 0;

    for e in 0..valid_entries.len() {
      if prev_rank_value == valid_entries[e].0.value {
        continue;
      }
      let rank_avg = NumT::from(e + s + 1).unwrap() / NumT::from(2usize).unwrap();
      let b = ((rank_avg - NumT::one()) * bins_t / total).floor();
      let bin = if b >= bins_t { bins_t - NumT::one() } else { b };
      for i in s..e {
        r[valid_entries[i].1] = bin;
      }
      s = e;
      prev_rank_value = valid_entries[e].0.value;
    }

    // the last chunk
    let rank_avg = NumT::from(valid_count + s + 1).unwrap() / NumT::from(2usize).unwrap();
    let b = ((rank_avg - NumT::one()) * bins_t / total).floor();
    let bin = if b >= bins_t { bins_t - NumT::one() } else { b };
    for i in s..valid_entries.len() {
      r[valid_entries[i].1] = bin;
    }
  });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::assert_vec_eq_nan;

  #[test]
  fn test_ta_ts_rank_simple() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_rank(&ctx, &mut r, &input, 3).unwrap();
    // Position 0: [1] -> rank 1
    // Position 1: [1,2] -> rank of 2 is 2
    // Position 2: [1,2,3] -> rank of 3 is 3
    // Position 3: [2,3,4] -> rank of 4 is 3
    // Position 4: [3,4,5] -> rank of 5 is 3
    assert_vec_eq_nan(&r, &vec![1.0, 2.0, 3.0, 3.0, 3.0]);
  }

  #[test]
  fn test_ta_ts_rank_periods_one() {
    let input = vec![1.0, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_rank(&ctx, &mut r, &input, 1).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_ta_ts_rank_with_nan() {
    let input = vec![1.0, f64::NAN, 3.0, 4.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_rank(&ctx, &mut r, &input, 3).unwrap();
    // NaN gets the highest rank since NaN != NaN
    // Position 0: [1] -> 1
    // Position 1: [1, NaN] -> NaN rank 2 (highest)
    // Position 2: [1, NaN, 3] -> 3 rank 3
    // Position 3: [NaN, 3, 4] -> 4 rank 3
    assert_vec_eq_nan(&r, &vec![1.0, 2.0, 3.0, 3.0]);
  }

  #[test]
  fn test_ta_ts_rank_duplicates() {
    // Test that duplicate values in the window are handled correctly.
    // The old BTreeMap<key, index> approach would lose duplicates.
    let input = vec![1.0, 1.0, 2.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_rank(&ctx, &mut r, &input, 3).unwrap();
    // Position 0: [1] -> rank of 1 is 1
    // Position 1: [1, 1] -> rank of 1 is 1 (min rank for ties)
    // Position 2: [1, 1, 2] -> rank of 2 is 3 (two 1s below it)
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 3.0]);
  }

  #[test]
  fn test_ta_ts_rank_duplicates_sliding() {
    // Verify correct sliding window behavior when duplicates enter and leave.
    let input = vec![3.0, 1.0, 2.0, 1.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_rank(&ctx, &mut r, &input, 3).unwrap();
    // Position 0: [3] -> rank 1
    // Position 1: [3, 1] -> rank of 1 is 1
    // Position 2: [3, 1, 2] -> rank of 2 is 2
    // Position 3: [1, 2, 1] -> rank of 1 is 1 (min rank, two 1s and one 2)
    // Position 4: [2, 1, 3] -> rank of 3 is 3
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 2.0, 1.0, 3.0]);
  }

  #[test]
  fn test_ta_rank_same_value() {
    let input = vec![1.0, 2.0, 1.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_cc_rank(&ctx, &mut r, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![0.5, 1.0, 0.5]);
  }

  #[test]
  fn test_ta_rank_simple() {
    let input = vec![3.0, 1.0, 2.0, 4.0]; // groups=2, matrix [3,2; 1,4]
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 2, 0);

    ta_cc_rank(&ctx, &mut r, &input).unwrap();
    // j=0: values [3,2], sorted [2,3], ranks [2,1] at indices 0,2
    // j=1: values [1,4], sorted [1,4], ranks [1,2] at indices 1,3
    assert_vec_eq_nan(&r, &vec![1.0, 0.5, 0.5, 1.0]);
  }

  #[test]
  fn test_ta_rank_three_groups() {
    let input = vec![3.0, 1.0, 2.0, 5.0, 4.0, 6.0]; // groups=3, matrix [3,2; 1,5; 4,6]
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_cc_rank(&ctx, &mut r, &input).unwrap();
    // j=0: values [3,2,4], sorted [2,3,4], ranks [2,1,3] at 0,2,4
    // j=1: values [1,5,6], sorted [1,5,6], ranks [1,2,3] at 1,3,5
    assert_vec_eq_nan(
      &r,
      &vec![
        2.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
        2.0 / 3.0,
        3.0 / 3.0,
        3.0 / 3.0,
      ],
    );
  }

  #[test]
  fn test_ta_rank_with_nan() {
    // groups=3, matrix [3,NaN; 1,5; 4,6]
    let input = vec![3.0, f64::NAN, 1.0, 5.0, 4.0, 6.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_cc_rank(&ctx, &mut r, &input).unwrap();
    // j=0: values [3,1,4], all valid, sorted [1,3,4], ranks [2,1,3]
    // j=1: values [NaN,5,6], 2 valid, sorted [5,6,NaN]
    //   5 -> rank 1/2=0.5, 6 -> rank 2/2=1.0, NaN -> NaN
    assert_vec_eq_nan(
      &r,
      &vec![
        2.0 / 3.0,
        f64::NAN,
        1.0 / 3.0,
        1.0 / 2.0,
        3.0 / 3.0,
        2.0 / 2.0,
      ],
    );
  }
}
