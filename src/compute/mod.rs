
// Compute Backend Trait Definitions
pub trait ComputeBackend: Send + Sync {
    fn rmsnorm(&self, input: &[f32], weight: &[f32], output: &mut [f32]) -> Result<(), String>;
    fn matmul(&self, a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), String>;
    fn softmax(&self, input: &[f32], output: &mut [f32]) -> Result<(), String>;
    fn silu(&self, input: &[f32], output: &mut [f32]) -> Result<(), String>;
}

