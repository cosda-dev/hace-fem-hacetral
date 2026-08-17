#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::dsl::{Action, AuthorityDsl, ExecutionMeta};
use crate::grammar::GrammarGuard;
use crate::guard::{ConfidenceDecision, ConfidenceGuard};
use crate::model::MistralCore;
use crate::resolver::{resolve_bitcode16, Bitcode16};
use crate::tokenizer::AuthorityTokenizer;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntentFrame {
    pub domain_id: u8,
    pub action_id: u16,
    pub params: [u8; 16],
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TeleportOutput {
    pub dsl: AuthorityDsl,
    pub frame: IntentFrame,
    pub bitcode16: Bitcode16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeleportError {
    InvalidInput,
    TokenizeFailed,
    GrammarFailed,
    LowConfidence,
}

fn select_domain(intent: &str, domain_hint: &str) -> (u8, String) {
    if domain_hint != "UNKNOWN" && !domain_hint.is_empty() {
        return (1, String::from(domain_hint));
    }
    let lower = intent.to_ascii_lowercase();
    if lower.contains("pay") || lower.contains("transfer") || lower.contains("send") {
        (11, String::from("FIN"))
    } else if lower.contains("audit") || lower.contains("verify") {
        (12, String::from("SEC"))
    } else if lower.contains("deploy") || lower.contains("build") || lower.contains("code") {
        (13, String::from("TECH"))
    } else {
        (1, String::from("SYS"))
    }
}

fn select_action(intent: &str) -> (u16, String, f32) {
    let lower = intent.to_ascii_lowercase();
    if lower.contains("pay") || lower.contains("transfer") || lower.contains("send") {
        (0x0050, String::from("transfer"), 0.98)
    } else if lower.contains("audit") || lower.contains("verify") {
        (0x00A1, String::from("audit"), 0.97)
    } else if lower.contains("deploy") || lower.contains("build") {
        (0x00C2, String::from("deploy"), 0.95)
    } else {
        (0x0001, String::from("unknown"), 0.92)
    }
}

/// Deterministic fn-teleport pipeline:
/// tokenize -> latent infer -> intent frame -> grammar -> confidence -> DSL.
pub fn teleport(intent: &str, aid: u64, domain: &str) -> Result<TeleportOutput, TeleportError> {
    if intent.trim().is_empty() {
        return Err(TeleportError::InvalidInput);
    }

    let tokens = AuthorityTokenizer::tokenize(intent).map_err(|_| TeleportError::TokenizeFailed)?;
    let latent = MistralCore::infer(&tokens);
    let (domain_id, domain_name) = select_domain(intent, domain);
    let (action_id, action_name, confidence) = select_action(intent);

    let mut params = [0u8; 16];
    let mut i = 0usize;
    while i < 16 {
        params[i] = (latent.data[i] as u8) ^ (i as u8);
        i += 1;
    }

    let frame = IntentFrame {
        domain_id,
        action_id,
        params,
        confidence,
    };

    GrammarGuard::validate(&frame).map_err(|_| TeleportError::GrammarFailed)?;
    if matches!(ConfidenceGuard::evaluate(&frame), ConfidenceDecision::Reject) {
        return Err(TeleportError::LowConfidence);
    }

    let action = Action {
        name: action_name,
        params: Vec::new(),
    };

    let mut constraints = Vec::new();
    constraints.push(String::from("auth_required"));

    let dsl = AuthorityDsl {
        aid,
        domain: domain_name,
        action,
        constraints,
        execution: ExecutionMeta::default(),
    };

    let bitcode16 = resolve_bitcode16(&frame, 1);
    Ok(TeleportOutput {
        dsl,
        frame,
        bitcode16,
    })
}
