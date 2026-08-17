
use core::{mem, slice};

use crate::core::tensor::{TensorError, TensorView, TensorViewMut};
use crate::index::mto_index::{DType, MtoEntry};
use crate::rr_bind::blob_source::{BlobError, BlobRegion, BlobSource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterError {
    Blob(BlobError),
    Tensor(TensorError),
    Seal(SealError),
    Misaligned,
    InvalidLength,
    UnsupportedDType,
}

impl From<BlobError> for AdapterError {
    fn from(err: BlobError) -> Self {
        AdapterError::Blob(err)
    }
}

impl From<TensorError> for AdapterError {
    fn from(err: TensorError) -> Self {
        AdapterError::Tensor(err)
    }
}

impl From<SealError> for AdapterError {
    fn from(err: SealError) -> Self {
        AdapterError::Seal(err)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SealError {
    Missing,
    Invalid,
}

pub trait SealVerifier {
    fn verify(&self, entry: &MtoEntry, bytes: &[u8]) -> Result<(), SealError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoSeal;

impl SealVerifier for NoSeal {
    fn verify(&self, _entry: &MtoEntry, _bytes: &[u8]) -> Result<(), SealError> {
        Ok(())
    }
}

pub struct MemoryAdapter<B: BlobSource, S: SealVerifier = NoSeal> {
    pub source: B,
    pub seal: S,
}

impl<B: BlobSource> MemoryAdapter<B, NoSeal> {
    pub fn new(source: B) -> Self {
        Self {
            source,
            seal: NoSeal,
        }
    }
}

impl<B: BlobSource, S: SealVerifier> MemoryAdapter<B, S> {
    pub fn with_seal(source: B, seal: S) -> Self {
        Self { source, seal }
    }

    /// Zero-copy tensor load (f32 only for now).
    pub fn load_tensor_f32<'a>(
        &'a self,
        entry: &MtoEntry,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<TensorView<'a, f32>, AdapterError> {
        if entry.dtype != DType::F32 as u16 {
            return Err(AdapterError::UnsupportedDType);
        }

        let region = BlobRegion::new(entry.data_offset, entry.data_len);
        let bytes = self.source.map_region(region)?;
        self.seal.verify(entry, bytes)?;
        let data = bytes_as_f32(bytes)?;

        TensorView::new(data, shape, strides).map_err(AdapterError::from)
    }

    pub fn load_tensor_f32_mut<'a>(
        &'a self,
        entry: &MtoEntry,
        shape: &[usize],
        strides: &[usize],
        out: &'a mut [f32],
    ) -> Result<TensorViewMut<'a, f32>, AdapterError> {
        if entry.dtype != DType::F32 as u16 {
            return Err(AdapterError::UnsupportedDType);
        }

        let elements = bytes_len_to_f32_len(entry.data_len as usize)?;
        if out.len() < elements {
            return Err(AdapterError::InvalidLength);
        }

        TensorViewMut::new(out, shape, strides).map_err(AdapterError::from)
    }
}

fn bytes_as_f32(bytes: &[u8]) -> Result<&[f32], AdapterError> {
    let align = mem::align_of::<f32>();
    let size = mem::size_of::<f32>();

    if (bytes.as_ptr() as usize) % align != 0 {
        return Err(AdapterError::Misaligned);
    }
    if bytes.len() % size != 0 {
        return Err(AdapterError::InvalidLength);
    }

    let len = bytes.len() / size;
    let ptr = bytes.as_ptr() as *const f32;
    let data = unsafe { slice::from_raw_parts(ptr, len) };
    Ok(data)
}

fn bytes_len_to_f32_len(len: usize) -> Result<usize, AdapterError> {
    let size = mem::size_of::<f32>();
    if len % size != 0 {
        return Err(AdapterError::InvalidLength);
    }
    Ok(len / size)
}
