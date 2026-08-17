use crate::teleport::IntentFrame;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrammarError {
    InvalidDomain,
    InvalidAction,
}

pub struct GrammarGuard;

impl GrammarGuard {
    pub fn validate(frame: &IntentFrame) -> Result<(), GrammarError> {
        if frame.domain_id == 0 {
            return Err(GrammarError::InvalidDomain);
        }
        if frame.action_id == 0 {
            return Err(GrammarError::InvalidAction);
        }
        Ok(())
    }
}
