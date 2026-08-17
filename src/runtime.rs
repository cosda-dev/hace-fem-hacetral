use crate::guard::{ConfidenceDecision, ConfidenceGuard};
use crate::teleport::{teleport, IntentFrame, TeleportError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HacetralError {
    InvalidInput,
    TeleportFailed,
    LowConfidence,
}

pub trait HacetralEngine {
    fn process(&mut self, input: &str, aid: u64, domain: &str) -> Result<IntentFrame, HacetralError>;
}

#[derive(Default)]
pub struct DefaultHacetral;

impl HacetralEngine for DefaultHacetral {
    fn process(&mut self, input: &str, aid: u64, domain: &str) -> Result<IntentFrame, HacetralError> {
        let out = teleport(input, aid, domain).map_err(|err| match err {
            TeleportError::InvalidInput => HacetralError::InvalidInput,
            TeleportError::LowConfidence => HacetralError::LowConfidence,
            TeleportError::TokenizeFailed | TeleportError::GrammarFailed => HacetralError::TeleportFailed,
        })?;

        match ConfidenceGuard::evaluate(&out.frame) {
            ConfidenceDecision::Allow | ConfidenceDecision::RequireHi => Ok(out.frame),
            ConfidenceDecision::Reject => Err(HacetralError::LowConfidence),
        }
    }
}
