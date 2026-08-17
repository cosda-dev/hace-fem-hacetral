// Conformance Trial (NKC)
// NOTE: Adjust the crate name if the package is not `hacetral`.

extern crate hacetral;

use hacetral::core::ops::matmul::matmul_f32;
use hacetral::core::tensor::{TensorView, TensorViewMut};

fn reference_candle_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // TODO: Replace with Candle baseline output captured via soul_manifest.ail inputs.
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn matmul_2x2_conformance() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let expected = reference_candle_matmul(&a, &b, 2, 2, 2);

    let a_view = TensorView::from_contiguous(&a, &[2, 2]).expect("a view");
    let b_view = TensorView::from_contiguous(&b, &[2, 2]).expect("b view");

    let mut out_buf = [0.0f32; 4];
    let mut out_view = TensorViewMut::from_contiguous(&mut out_buf, &[2, 2]).expect("out view");

    matmul_f32(&a_view, &b_view, &mut out_view).expect("matmul");

    assert_eq!(&out_buf[..], expected.as_slice());
}

#[test]
fn matmul_non_contiguous_stride() {
    // 2x2 matrix stored with row stride = 3 (padding element per row)
    let a = [1.0f32, 2.0, 0.0, 3.0, 4.0, 0.0];
    let b = [5.0f32, 6.0, 0.0, 7.0, 8.0, 0.0];

    let a_view = TensorView::new(&a, &[2, 2], &[3, 1]).expect("a view");
    let b_view = TensorView::new(&b, &[2, 2], &[3, 1]).expect("b view");

    let mut out_buf = [0.0f32; 6];
    let mut out_view = TensorViewMut::new(&mut out_buf, &[2, 2], &[3, 1]).expect("out view");

    matmul_f32(&a_view, &b_view, &mut out_view).expect("matmul");

    let expected = [19.0f32, 22.0, 0.0, 43.0, 50.0, 0.0];
    assert_eq!(&out_buf[..], &expected[..]);
}
