#![cfg_attr(not(any(test, feature = "std")), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod dsl;
pub mod grammar;
pub mod guard;
pub mod model;
pub mod rc_bind;
pub mod resolver;
pub mod tokenizer;
pub mod runtime;
pub mod teleport;
pub mod validator;

#[cfg(any(test, feature = "std"))]
extern crate std;

pub use dsl::{Action, AuthorityCall, AuthorityDsl, ExecutionMeta};
pub use grammar::{GrammarError, GrammarGuard};
pub use guard::{ConfidenceDecision, ConfidenceGuard};
pub use model::{LatentVector, MistralCore};
pub use rc_bind::{bind_rc, RcBinder, RcBindingError};
pub use resolver::{resolve_bitcode16, Bitcode16};
pub use runtime::{HacetralEngine, HacetralError};
pub use teleport::{teleport, IntentFrame, TeleportError, TeleportOutput};
pub use tokenizer::{AuthorityTokenizer, TokenizeError};
pub use validator::{validate_dsl, DslValidationError};
