
use crate::index::mto_index::MtoEntry;
use crate::rr_bind::memory_adapter::{SealError, SealVerifier};

#[derive(Clone, Copy, Debug)]
pub struct MtoSealVerifier {
    pub allow_zero: bool,
}

impl MtoSealVerifier {
    pub const fn new(allow_zero: bool) -> Self {
        Self { allow_zero }
    }
}

impl SealVerifier for MtoSealVerifier {
    fn verify(&self, entry: &MtoEntry, bytes: &[u8]) -> Result<(), SealError> {
        let expected = entry.seal();
        if expected == 0 && !self.allow_zero {
            return Err(SealError::Missing);
        }
        let actual = fnv1a64(bytes);
        if actual != expected {
            return Err(SealError::Invalid);
        }
        Ok(())
    }
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
