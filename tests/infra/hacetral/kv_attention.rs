// KV cache + attention integration tests (feature-gated).

#[cfg(feature = "libm")]
mod tests {
    use hacetral::core::ops::attention::attention_with_kv;
    use hacetral::core::tensor::{TensorView, TensorViewMut};
    use hacetral::runtime::kv_cache::KvCache;

    #[test]
    fn kv_append_correctness() {
        let mut k_buf = [0.0f32; 8]; // [seq=2, h=1, d=4]
        let mut v_buf = [0.0f32; 8];

        let k_view = TensorViewMut::from_contiguous(&mut k_buf, &[2, 1, 4]).expect("k view");
        let v_view = TensorViewMut::from_contiguous(&mut v_buf, &[2, 1, 4]).expect("v view");

        let mut cache = KvCache::new(vec![k_view], vec![v_view], 2);

        let k_new = TensorView::from_contiguous(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]).expect("k new");
        let v_new = TensorView::from_contiguous(&[5.0f32, 6.0, 7.0, 8.0], &[1, 1, 4]).expect("v new");

        cache.append(0, k_new, v_new).expect("append");

        assert_eq!(cache.seq_len, 1);
    }

    #[test]
    fn attention_matches_cached_shapes() {
        let q = TensorView::from_contiguous(&[1.0f32, 0.0, 0.0, 0.0], &[1, 1, 4]).expect("q");
        let k = TensorView::from_contiguous(&[1.0f32, 0.0, 0.0, 0.0], &[1, 1, 4]).expect("k");
        let v = TensorView::from_contiguous(&[0.5f32, 0.0, 0.0, 0.0], &[1, 1, 4]).expect("v");

        let mut out_buf = [0.0f32; 4];
        let mut out = TensorViewMut::from_contiguous(&mut out_buf, &[1, 1, 4]).expect("out");

        attention_with_kv(&q, &k, &v, &mut out).expect("attention");
        assert!(out_buf[0] > 0.0);
    }
}
