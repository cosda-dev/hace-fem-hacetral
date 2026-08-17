use crate::teleport::IntentFrame;

pub type Bitcode16 = u64;

pub fn resolve_bitcode16(frame: &IntentFrame, nonce: u32) -> Bitcode16 {
    ((frame.domain_id as u64) << 48) | ((frame.action_id as u64) << 32) | (nonce as u64)
}
