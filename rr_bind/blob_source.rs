//! RR-compatible blob reader abstraction.
//!
//! Implementors map offsets to a zero-copy or streamed source.

use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlobError {
    OutOfBounds,
    PermissionDenied,
    IoFault,
    Unsupported,
    Misaligned,
}

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            BlobError::OutOfBounds => "out_of_bounds",
            BlobError::PermissionDenied => "permission_denied",
            BlobError::IoFault => "io_fault",
            BlobError::Unsupported => "unsupported",
            BlobError::Misaligned => "misaligned",
        };
        f.write_str(label)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlobRegion {
    pub offset: u64,
    pub length: u32,
}

impl BlobRegion {
    pub const fn new(offset: u64, length: u32) -> Self {
        Self { offset, length }
    }
}

pub trait BlobSource {
    /// Reads `len` bytes at `offset` into `out`, returning bytes read.
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, BlobError>;

    /// Maps a region into a zero-copy window if supported.
    fn map_region<'a>(&'a self, region: BlobRegion) -> Result<&'a [u8], BlobError> {
        let _ = region;
        Err(BlobError::Unsupported)
    }

    /// Returns the total length if available.
    fn len(&self) -> Option<u64> {
        None
    }

    fn read_exact(&self, region: BlobRegion, out: &mut [u8]) -> Result<(), BlobError> {
        if out.len() != region.length as usize {
            return Err(BlobError::OutOfBounds);
        }

        let read = self.read_at(region.offset, out)?;
        if read != out.len() {
            return Err(BlobError::IoFault);
        }
        Ok(())
    }
}
