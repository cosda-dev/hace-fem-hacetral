
extern crate alloc;

use alloc::vec::Vec;

use crate::core::tensor::{TensorView, TensorViewMut};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvError {
    OutOfBounds,
    ShapeMismatch,
    NonContiguous,
    CapacityExceeded,
}

pub struct KvCache<'a> {
    pub k: Vec<TensorViewMut<'a, f32>>,
    pub v: Vec<TensorViewMut<'a, f32>>,
    pub capacity: usize,
    pub seq_len: usize,
}

impl<'a> KvCache<'a> {
    pub fn new(
        k: Vec<TensorViewMut<'a, f32>>,
        v: Vec<TensorViewMut<'a, f32>>,
        capacity: usize,
    ) -> Self {
        Self {
            k,
            v,
            capacity,
            seq_len: 0,
        }
    }

    pub fn append(
        &mut self,
        layer: usize,
        k_new: TensorView<'a, f32>,
        v_new: TensorView<'a, f32>,
    ) -> Result<(), KvError> {
        if self.seq_len >= self.capacity {
            return Err(KvError::CapacityExceeded);
        }
        if layer >= self.k.len() || layer >= self.v.len() {
            return Err(KvError::OutOfBounds);
        }

        let seq_idx = self.seq_len;
        append_contiguous(&mut self.k[layer], &k_new, seq_idx)?;
        append_contiguous(&mut self.v[layer], &v_new, seq_idx)?;

        self.seq_len += 1;
        Ok(())
    }
}

fn append_contiguous(
    cache: &mut TensorViewMut<'_, f32>,
    token: &TensorView<'_, f32>,
    seq_idx: usize,
) -> Result<(), KvError> {
    if !cache.as_view().is_contiguous() || !token.is_contiguous() {
        return Err(KvError::NonContiguous);
    }

    let cache_shape = cache.shape();
    let token_shape = token.shape();

    if cache_shape.len() != token_shape.len() || cache_shape.is_empty() {
        return Err(KvError::ShapeMismatch);
    }

    if cache_shape[0] <= seq_idx || token_shape[0] != 1 {
        return Err(KvError::OutOfBounds);
    }

    for i in 1..cache_shape.len() {
        if cache_shape[i] != token_shape[i] {
            return Err(KvError::ShapeMismatch);
        }
    }

    let token_len = token.numel();
    let base = seq_idx
        .checked_mul(cache.strides()[0])
        .ok_or(KvError::OutOfBounds)?;
    let end = base.checked_add(token_len).ok_or(KvError::OutOfBounds)?;

    let dst = cache.data_mut();
    let src = token.data();

    if end > dst.len() || token_len > src.len() {
        return Err(KvError::OutOfBounds);
    }

    dst[base..end].copy_from_slice(&src[..token_len]);

    Ok(())
}
