/// Deterministic latent representation for fn-teleport classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentVector {
    pub data: [i32; 16],
}

pub struct MistralCore;

impl MistralCore {
    pub fn infer(tokens: &[u16]) -> LatentVector {
        let mut out = [0i32; 16];
        let mut i = 0usize;
        while i < tokens.len() {
            let idx = i % 16;
            out[idx] = out[idx].wrapping_add(tokens[i] as i32);
            i += 1;
        }
        LatentVector { data: out }
    }
}
