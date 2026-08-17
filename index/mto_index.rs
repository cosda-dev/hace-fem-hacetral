
extern crate alloc;

use core::{mem, slice, str};

const MAGIC: &[u8; 4] = b"MTO1";
const VERSION: u16 = 0x0001;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MtoHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub entry_count: u32,
    pub entry_offset: u64,
    pub string_offset: u64,
    pub reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MtoEntry {
    pub name_hash: u64,
    pub name_offset: u32,
    pub name_len: u32,
    pub data_offset: u64,
    pub data_len: u32,
    pub dtype: u16,
    pub device: u16,
    pub shape_offset: u32,
    pub reserved: u64,
}

impl MtoEntry {
    #[inline]
    pub fn seal(&self) -> u64 {
        self.reserved
    }
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    U8 = 4,
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    CPU = 0,
    GPU = 1,
    NPU = 2,
}

#[derive(Clone, Copy, Debug)]
pub struct MtoIndex<'a> {
    header: &'a MtoHeader,
    entries: &'a [MtoEntry],
    string_table: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtoError {
    InvalidMagic,
    UnsupportedVersion,
    OutOfBounds,
    Misaligned,
    Corrupt,
}

impl<'a> MtoIndex<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, MtoError> {
        if data.len() < mem::size_of::<MtoHeader>() {
            return Err(MtoError::OutOfBounds);
        }

        let header = unsafe { &*(data.as_ptr() as *const MtoHeader) };
        if &header.magic != MAGIC {
            return Err(MtoError::InvalidMagic);
        }
        if header.version != VERSION {
            return Err(MtoError::UnsupportedVersion);
        }

        let entry_start = header.entry_offset as usize;
        let entry_count = header.entry_count as usize;
        let entry_size = mem::size_of::<MtoEntry>();
        let entry_bytes = entry_size
            .checked_mul(entry_count)
            .ok_or(MtoError::OutOfBounds)?;
        let entry_end = entry_start
            .checked_add(entry_bytes)
            .ok_or(MtoError::OutOfBounds)?;

        if entry_end > data.len() {
            return Err(MtoError::OutOfBounds);
        }
        if (entry_start % mem::align_of::<MtoEntry>()) != 0 {
            return Err(MtoError::Misaligned);
        }

        let entries = unsafe {
            slice::from_raw_parts(
                data.as_ptr().add(entry_start) as *const MtoEntry,
                entry_count,
            )
        };

        let string_start = header.string_offset as usize;
        if string_start >= data.len() {
            return Err(MtoError::OutOfBounds);
        }
        let string_table = &data[string_start..];

        Ok(Self {
            header,
            entries,
            string_table,
        })
    }

    #[inline]
    pub fn header(&self) -> &MtoHeader {
        self.header
    }

    #[inline]
    pub fn entries(&self) -> &[MtoEntry] {
        self.entries
    }

    #[inline]
    pub fn get_by_hash(&self, hash: u64) -> Option<&MtoEntry> {
        for entry in self.entries {
            if entry.name_hash == hash {
                return Some(entry);
            }
        }
        None
    }

    #[inline]
    pub fn get_name(&self, entry: &MtoEntry) -> Result<&str, MtoError> {
        let start = entry.name_offset as usize;
        let end = start
            .checked_add(entry.name_len as usize)
            .ok_or(MtoError::OutOfBounds)?;

        if end > self.string_table.len() {
            return Err(MtoError::OutOfBounds);
        }
        let bytes = &self.string_table[start..end];
        str::from_utf8(bytes).map_err(|_| MtoError::Corrupt)
    }
}
