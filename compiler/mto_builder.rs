
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::mem;

use crate::index::mto_index::{DType, Device, MtoEntry, MtoHeader};

#[derive(Clone, Debug)]
pub struct MtoDraftEntry {
    pub name: String,
    pub name_hash: u64,
    pub data_offset: u64,
    pub data_len: u32,
    pub dtype: DType,
    pub device: Device,
    pub shape_offset: u32,
    pub seal: u64,
    pub shard_id: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    Overflow,
}

pub fn build_mto(entries: &[MtoDraftEntry]) -> Result<Vec<u8>, BuildError> {
    let header_size = mem::size_of::<MtoHeader>();
    let entry_size = mem::size_of::<MtoEntry>();

    let entry_bytes = entry_size
        .checked_mul(entries.len())
        .ok_or(BuildError::Overflow)?;
    let entry_offset = header_size;
    let string_offset = entry_offset
        .checked_add(entry_bytes)
        .ok_or(BuildError::Overflow)?;

    let mut string_table = Vec::new();
    let mut encoded: Vec<MtoEntry> = Vec::with_capacity(entries.len());

    for draft in entries {
        let name_offset = string_table.len() as u32;
        let name_bytes = draft.name.as_bytes();
        let name_len = name_bytes.len() as u32;
        string_table.extend_from_slice(name_bytes);

        let entry = MtoEntry {
            name_hash: draft.name_hash,
            name_offset,
            name_len,
            data_offset: draft.data_offset,
            data_len: draft.data_len,
            dtype: draft.dtype as u16,
            device: draft.device as u16,
            shape_offset: draft.shape_offset,
            reserved: draft.seal,
        };
        encoded.push(entry);
    }

    let total = string_offset
        .checked_add(string_table.len())
        .ok_or(BuildError::Overflow)?;

    let mut out = Vec::with_capacity(total);
    out.resize(header_size, 0u8);

    for entry in &encoded {
        write_entry(&mut out, entry);
    }
    out.extend_from_slice(&string_table);

    let header = MtoHeader {
        magic: *b"MTO1",
        version: 0x0001,
        flags: 0,
        entry_count: entries.len() as u32,
        entry_offset: entry_offset as u64,
        string_offset: string_offset as u64,
        reserved: [0u64; 4],
    };

    write_header(&mut out, header);

    Ok(out)
}

fn write_header(buf: &mut [u8], header: MtoHeader) {
    let mut cursor = 0usize;
    buf[cursor..cursor + 4].copy_from_slice(&header.magic);
    cursor += 4;
    buf[cursor..cursor + 2].copy_from_slice(&header.version.to_le_bytes());
    cursor += 2;
    buf[cursor..cursor + 2].copy_from_slice(&header.flags.to_le_bytes());
    cursor += 2;
    buf[cursor..cursor + 4].copy_from_slice(&header.entry_count.to_le_bytes());
    cursor += 4;
    buf[cursor..cursor + 8].copy_from_slice(&header.entry_offset.to_le_bytes());
    cursor += 8;
    buf[cursor..cursor + 8].copy_from_slice(&header.string_offset.to_le_bytes());
    cursor += 8;
    for reserved in &header.reserved {
        buf[cursor..cursor + 8].copy_from_slice(&reserved.to_le_bytes());
        cursor += 8;
    }
}

fn write_entry(buf: &mut Vec<u8>, entry: &MtoEntry) {
    buf.extend_from_slice(&entry.name_hash.to_le_bytes());
    buf.extend_from_slice(&entry.name_offset.to_le_bytes());
    buf.extend_from_slice(&entry.name_len.to_le_bytes());
    buf.extend_from_slice(&entry.data_offset.to_le_bytes());
    buf.extend_from_slice(&entry.data_len.to_le_bytes());
    buf.extend_from_slice(&entry.dtype.to_le_bytes());
    buf.extend_from_slice(&entry.device.to_le_bytes());
    buf.extend_from_slice(&entry.shape_offset.to_le_bytes());
    buf.extend_from_slice(&entry.reserved.to_le_bytes());
}
