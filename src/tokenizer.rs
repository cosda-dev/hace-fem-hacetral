#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizeError {
    EmptyInput,
}

pub struct AuthorityTokenizer;

impl AuthorityTokenizer {
    pub fn tokenize(input: &str) -> Result<Vec<u16>, TokenizeError> {
        if input.trim().is_empty() {
            return Err(TokenizeError::EmptyInput);
        }

        let mut tokens = Vec::new();
        for word in input.split_whitespace() {
            let mut hash: u16 = 0;
            for b in word.as_bytes() {
                hash = hash.wrapping_mul(33).wrapping_add(*b as u16);
            }
            tokens.push(hash);
        }
        Ok(tokens)
    }
}
