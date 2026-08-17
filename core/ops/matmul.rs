
use core::fmt;

use crate::core::tensor::{TensorError, TensorView, TensorViewMut};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatMulError {
    ShapeMismatch,
    Tensor(TensorError),
}

impl fmt::Display for MatMulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            MatMulError::ShapeMismatch => "shape_mismatch",
            MatMulError::Tensor(_) => "tensor_error",
        };
        f.write_str(label)
    }
}

impl From<TensorError> for MatMulError {
    fn from(err: TensorError) -> Self {
        MatMulError::Tensor(err)
    }
}

pub fn matmul_f32(
    a: &TensorView<'_, f32>,
    b: &TensorView<'_, f32>,
    out: &mut TensorViewMut<'_, f32>,
) -> Result<(), MatMulError> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let o_shape = out.shape();

    if a_shape.len() != 2 || b_shape.len() != 2 || o_shape.len() != 2 {
        return Err(MatMulError::ShapeMismatch);
    }

    let (m, k1) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);

    if k1 != k2 || o_shape[0] != m || o_shape[1] != n {
        return Err(MatMulError::ShapeMismatch);
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..k1 {
                let a_val = unsafe { *a.get_unchecked(&[i, k]) };
                let b_val = unsafe { *b.get_unchecked(&[k, j]) };
                acc += a_val * b_val;
            }
            unsafe {
                *out.get_unchecked_mut(&[i, j]) = acc;
            }
        }
    }

    Ok(())
}
