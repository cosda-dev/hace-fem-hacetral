use crate::dsl::AuthorityDsl;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DslValidationError {
    MissingAid,
    MissingDomain,
    MissingAction,
    MissingConstraints,
}

pub fn validate_dsl(dsl: &AuthorityDsl) -> Result<(), DslValidationError> {
    if dsl.aid == 0 {
        return Err(DslValidationError::MissingAid);
    }
    if dsl.domain.is_empty() {
        return Err(DslValidationError::MissingDomain);
    }
    if dsl.action.name.is_empty() {
        return Err(DslValidationError::MissingAction);
    }
    if dsl.constraints.is_empty() {
        return Err(DslValidationError::MissingConstraints);
    }
    Ok(())
}
