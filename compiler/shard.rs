
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::compiler::mto_builder::{build_mto, BuildError as MtoBuildError, MtoDraftEntry};
use crate::index::mto_index::{DType, Device};
use crate::rr_bind::seal::fnv1a64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardError {
    TensorTooLarge,
    Mto(MtoBuildError),
}

impl From<MtoBuildError> for ShardError {
    fn from(err: MtoBuildError) -> Self {
        ShardError::Mto(err)
    }
}

#[derive(Clone, Debug)]
pub struct ShardBlob {
    pub shard_id: u16,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct ShardBinding {
    pub name_hash: u64,
    pub shard_id: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct TensorRef {
    pub name_hash: u64,
    pub shard_id: u16,
    pub offset: u64,
    pub len: u32,
}

pub struct ShardWriter {
    shard_size: usize,
    align_tensor: usize,
    align_shard: usize,
    current_id: u16,
    current: Vec<u8>,
    shards: Vec<ShardBlob>,
    entries: Vec<MtoDraftEntry>,
}

impl ShardWriter {
    pub fn new(shard_size: usize) -> Self {
        Self {
            shard_size,
            align_tensor: 32,
            align_shard: 4096,
            current_id: 0,
            current: Vec::new(),
            shards: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn with_alignment(mut self, tensor_align: usize, shard_align: usize) -> Self {
        self.align_tensor = tensor_align.max(1);
        self.align_shard = shard_align.max(1);
        self
    }

    pub fn write_tensor(
        &mut self,
        name: &str,
        bytes: &[u8],
        dtype: DType,
        device: Device,
    ) -> Result<TensorRef, ShardError> {
        if bytes.len() > self.shard_size {
            return Err(ShardError::TensorTooLarge);
        }

        let offset = align_up(self.current.len(), self.align_tensor);
        if offset + bytes.len() > self.shard_size {
            self.flush_current();
        }

        let offset = align_up(self.current.len(), self.align_tensor);
        self.current.resize(offset, 0u8);
        self.current.extend_from_slice(bytes);

        let name_hash = fnv1a64(name.as_bytes());
        let seal = fnv1a64(bytes);

        self.entries.push(MtoDraftEntry {
            name: String::from(name),
            name_hash,
            data_offset: offset as u64,
            data_len: bytes.len() as u32,
            dtype,
            device,
            shape_offset: 0,
            seal,
            shard_id: self.current_id,
        });

        Ok(TensorRef {
            name_hash,
            shard_id: self.current_id,
            offset: offset as u64,
            len: bytes.len() as u32,
        })
    }

    pub fn finalize(mut self) -> Result<(Vec<ShardBlob>, Vec<u8>, Vec<ShardBinding>), ShardError> {
        self.flush_current();
        let shard_map = self
            .entries
            .iter()
            .map(|entry| ShardBinding {
                name_hash: entry.name_hash,
                shard_id: entry.shard_id,
            })
            .collect();
        let mto = build_mto(&self.entries)?;
        Ok((self.shards, mto, shard_map))
    }

    fn flush_current(&mut self) {
        if self.current.is_empty() {
            return;
        }

        let aligned = align_up(self.current.len(), self.align_shard);
        self.current.resize(aligned, 0u8);

        let bytes = core::mem::take(&mut self.current);
        self.shards.push(ShardBlob {
            shard_id: self.current_id,
            bytes,
        });
        self.current_id = self.current_id.wrapping_add(1);
    }
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}
