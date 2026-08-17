
use crate::core::tensor::{TensorError, TensorView, TensorViewMut};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmsNormError {
    ShapeMismatch,
    Tensor(TensorError),
}

impl From<TensorError> for RmsNormError {
    fn from(err: TensorError) -> Self {
        RmsNormError::Tensor(err)
    }
}

pub fn rmsnorm_f32(
    x: &TensorView<'_, f32>,
    weight: &TensorView<'_, f32>,
    out: &mut TensorViewMut<'_, f32>,
    eps: f32,
) -> Result<(), RmsNormError> {
    let x_shape = x.shape();
    let w_shape = weight.shape();
    let o_shape = out.shape();

    if x_shape != o_shape || w_shape.len() != 1 || x_shape.is_empty() {
        return Err(RmsNormError::ShapeMismatch);
    }

    let dim = w_shape[0];
    if x_shape[x_shape.len() - 1] != dim {
        return Err(RmsNormError::ShapeMismatch);
    }

    let rows = x.numel() / dim;
    for r in 0..rows {
        let mut mean_sq = 0.0f32;
        for i in 0..dim {
            let idx = r * dim + i;
            let v = unsafe { *x.get_unchecked(&[idx]) };
            mean_sq += v * v;
        }
        mean_sq /= dim as f32;
        let inv = 1.0f32 / libm::sqrtf(mean_sq + eps);

        for i in 0..dim {
            let idx = r * dim + i;
            let v = unsafe { *x.get_unchecked(&[idx]) };
            let w = unsafe { *weight.get_unchecked(&[i]) };
            unsafe {
                *out.get_unchecked_mut(&[idx]) = v * inv * w;
            }
        }
    }

    Ok(())
}
