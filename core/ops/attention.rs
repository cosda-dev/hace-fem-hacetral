
use crate::core::tensor::{TensorError, TensorView, TensorViewMut};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionError {
    ShapeMismatch,
    SeqTooLong,
    Tensor(TensorError),
    SoftmaxUnavailable,
}

impl From<TensorError> for AttentionError {
    fn from(err: TensorError) -> Self {
        AttentionError::Tensor(err)
    }
}

const MAX_SEQ: usize = 2048;

pub fn attention_with_kv(
    q: &TensorView<'_, f32>,
    k_cache: &TensorView<'_, f32>,
    v_cache: &TensorView<'_, f32>,
    out: &mut TensorViewMut<'_, f32>,
) -> Result<(), AttentionError> {
    let q_shape = q.shape();
    let k_shape = k_cache.shape();
    let v_shape = v_cache.shape();
    let o_shape = out.shape();

    if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 || o_shape.len() != 3 {
        return Err(AttentionError::ShapeMismatch);
    }

    let t = k_shape[0];
    let h = q_shape[1];
    let d = q_shape[2];

    if t > MAX_SEQ {
        return Err(AttentionError::SeqTooLong);
    }

    if q_shape[0] != 1 || o_shape[0] != 1 {
        return Err(AttentionError::ShapeMismatch);
    }
    if k_shape[0] != v_shape[0] || k_shape[1] != v_shape[1] || k_shape[2] != v_shape[2] {
        return Err(AttentionError::ShapeMismatch);
    }
    if k_shape[1] != h || k_shape[2] != d {
        return Err(AttentionError::ShapeMismatch);
    }
    if o_shape[1] != h || o_shape[2] != d {
        return Err(AttentionError::ShapeMismatch);
    }

    let scale = 1.0f32 / (d as f32).sqrt();

    for head in 0..h {
        let mut scores = [0f32; MAX_SEQ];

        for ti in 0..t {
            let mut dot = 0.0f32;
            for di in 0..d {
                let qv = unsafe { *q.get_unchecked(&[0, head, di]) };
                let kv = unsafe { *k_cache.get_unchecked(&[ti, head, di]) };
                dot += qv * kv;
            }
            scores[ti] = dot * scale;
        }

        #[cfg(feature = "libm")]
        {
            let mut max = scores[0];
            for ti in 1..t {
                if scores[ti] > max {
                    max = scores[ti];
                }
            }

            let mut sum = 0.0f32;
            for ti in 0..t {
                scores[ti] = libm::expf(scores[ti] - max);
                sum += scores[ti];
            }

            for ti in 0..t {
                scores[ti] /= sum;
            }
        }

        #[cfg(not(feature = "libm"))]
        {
            let _ = scores;
            return Err(AttentionError::SoftmaxUnavailable);
        }

        for di in 0..d {
            let mut val = 0.0f32;
            for ti in 0..t {
                let vv = unsafe { *v_cache.get_unchecked(&[ti, head, di]) };
                val += scores[ti] * vv;
            }
            unsafe {
                *out.get_unchecked_mut(&[0, head, di]) = val;
            }
        }
    }

    Ok(())
}
