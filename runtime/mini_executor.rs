
use crate::core::ops::matmul::matmul_f32;
use crate::core::tensor::TensorViewMut;
use crate::index::mto_index::MtoIndex;
use crate::rr_bind::blob_source::BlobSource;
use crate::rr_bind::memory_adapter::{AdapterError, MemoryAdapter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecError {
    MissingEntry,
    Adapter(AdapterError),
    MatMul,
}

impl From<AdapterError> for ExecError {
    fn from(err: AdapterError) -> Self {
        ExecError::Adapter(err)
    }
}

pub struct MiniExecutor<'a, B: BlobSource> {
    pub index: MtoIndex<'a>,
    pub adapter: MemoryAdapter<B>,
}

impl<'a, B: BlobSource> MiniExecutor<'a, B> {
    pub fn new(index: MtoIndex<'a>, adapter: MemoryAdapter<B>) -> Self {
        Self { index, adapter }
    }

    pub fn run_matmul(
        &self,
        a_hash: u64,
        b_hash: u64,
        a_shape: &[usize],
        a_strides: &[usize],
        b_shape: &[usize],
        b_strides: &[usize],
        out_shape: &[usize],
        out_strides: &[usize],
        out: &mut [f32],
    ) -> Result<(), ExecError> {
        let a_entry = self.index.get_by_hash(a_hash).ok_or(ExecError::MissingEntry)?;
        let b_entry = self.index.get_by_hash(b_hash).ok_or(ExecError::MissingEntry)?;

        let a = self
            .adapter
            .load_tensor_f32(a_entry, a_shape, a_strides)?;
        let b = self
            .adapter
            .load_tensor_f32(b_entry, b_shape, b_strides)?;

        let mut out_view = TensorViewMut::new(out, out_shape, out_strides)
            .map_err(|err| ExecError::Adapter(AdapterError::Tensor(err)))?;

        matmul_f32(&a, &b, &mut out_view).map_err(|_| ExecError::MatMul)
    }
}
