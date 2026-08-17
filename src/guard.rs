use crate::teleport::IntentFrame;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfidenceDecision {
    Allow,
    RequireHi,
    Reject,
}

pub struct ConfidenceGuard;

impl ConfidenceGuard {
    pub fn evaluate(frame: &IntentFrame) -> ConfidenceDecision {
        if frame.confidence >= 0.97 {
            ConfidenceDecision::Allow
        } else if frame.confidence >= 0.90 {
            ConfidenceDecision::RequireHi
        } else {
            ConfidenceDecision::Reject
        }
    }
}
