
use crate::core::tensor::{TensorError, TensorView, TensorViewMut};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeError {
    ShapeMismatch,
    Tensor(TensorError),
}

impl From<TensorError> for RopeError {
    fn from(err: TensorError) -> Self {
        RopeError::Tensor(err)
    }
}

pub fn rope_f32(
    x: &TensorView<'_, f32>,
    cos: &TensorView<'_, f32>,
    sin: &TensorView<'_, f32>,
    out: &mut TensorViewMut<'_, f32>,
) -> Result<(), RopeError> {
    let x_shape = x.shape();
    let c_shape = cos.shape();
    let s_shape = sin.shape();
    let o_shape = out.shape();

    if x_shape != o_shape || c_shape != s_shape || x_shape != c_shape {
        return Err(RopeError::ShapeMismatch);
    }

    if x_shape.is_empty() {
        return Ok(());
    }

    let last_dim = x_shape[x_shape.len() - 1];
    if last_dim % 2 != 0 {
        return Err(RopeError::ShapeMismatch);
    }

    let rows = x.numel() / last_dim;
    for r in 0..rows {
        let base = r * last_dim;
        for i in 0..(last_dim / 2) {
            let idx0 = base + 2 * i;
            let idx1 = idx0 + 1;

            let x0 = unsafe { *x.get_unchecked(&[idx0]) };
            let x1 = unsafe { *x.get_unchecked(&[idx1]) };
            let c0 = unsafe { *cos.get_unchecked(&[idx0]) };
            let s0 = unsafe { *sin.get_unchecked(&[idx0]) };

            unsafe {
                *out.get_unchecked_mut(&[idx0]) = x0 * c0 - x1 * s0;
                *out.get_unchecked_mut(&[idx1]) = x1 * c0 + x0 * s0;
            }
        }
    }

    Ok(())
}
