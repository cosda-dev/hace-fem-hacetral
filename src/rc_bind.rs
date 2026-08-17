use core::sync::atomic::{AtomicU64, Ordering};

use crate::dsl::{AuthorityCall, AuthorityDsl};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RcBindingError {
    InvalidDsl,
}

pub trait RcBinder {
    fn bind(&self, dsl: AuthorityDsl) -> Result<AuthorityCall, RcBindingError>;
}

static NONCE: AtomicU64 = AtomicU64::new(1);

fn now_ms() -> u64 {
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        if let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) {
            return dur.as_millis() as u64;
        }
    }
    0
}

fn hash64(input: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut i = 0usize;
    while i < input.len() {
        hash ^= input[i] as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        i += 1;
    }
    hash
}

fn rc_hash32(aid: u64, domain: &str, action: &str) -> [u8; 32] {
    let mut buf = [0u8; 32];
    let mut tmp = [0u8; 16];
    tmp[0..8].copy_from_slice(&aid.to_le_bytes());
    let h0 = hash64(&tmp);
    let h1 = hash64(domain.as_bytes());
    let h2 = hash64(action.as_bytes());
    let h3 = h0 ^ h1 ^ h2;
    buf[0..8].copy_from_slice(&h0.to_le_bytes());
    buf[8..16].copy_from_slice(&h1.to_le_bytes());
    buf[16..24].copy_from_slice(&h2.to_le_bytes());
    buf[24..32].copy_from_slice(&h3.to_le_bytes());
    buf
}

/// Deterministic binder: attaches RC hash + monotonic nonce metadata.
pub fn bind_rc(dsl: AuthorityDsl) -> Result<AuthorityCall, RcBindingError> {
    if dsl.domain.is_empty() || dsl.action.name.is_empty() || dsl.aid == 0 {
        return Err(RcBindingError::InvalidDsl);
    }

    let rc_hash = rc_hash32(dsl.aid, &dsl.domain, &dsl.action.name);
    let timestamp = now_ms();
    let nonce = NONCE.fetch_add(1, Ordering::Relaxed);

    Ok(AuthorityCall {
        aid: dsl.aid,
        domain: dsl.domain,
        action: dsl.action,
        rc_hash,
        invariants: dsl.constraints,
        dim_snapshot_hash: [0u8; 32],
        meta_timestamp_ms: timestamp,
        meta_nonce: nonce,
    })
}
