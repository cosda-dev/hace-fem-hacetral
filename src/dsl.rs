#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Action {
    pub name: String,
    pub params: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionMeta {
    pub mode: ExecutionMode,
    pub priority: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Sync,
    Async,
}

impl Default for ExecutionMeta {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Sync,
            priority: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorityDsl {
    pub aid: u64,
    pub domain: String,
    pub action: Action,
    pub constraints: Vec<String>,
    pub execution: ExecutionMeta,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorityCall {
    pub aid: u64,
    pub domain: String,
    pub action: Action,
    pub rc_hash: [u8; 32],
    pub invariants: Vec<String>,
    pub dim_snapshot_hash: [u8; 32],
    pub meta_timestamp_ms: u64,
    pub meta_nonce: u64,
}
