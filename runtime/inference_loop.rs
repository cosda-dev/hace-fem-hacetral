
extern crate alloc;

use alloc::vec::Vec;

use crate::core::tensor::TensorViewMut;
use crate::runtime::kv_cache::{KvCache, KvError};
use crate::runtime::static_graph_executor::{AuthorityChecker, ExecError, GraphExecutor, KernelRegistry};
use crate::runtime::token_sampler::Sampler;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceError {
    Exec(ExecError),
    Kv(KvError),
    MissingLogits,
    EmptyLogits,
}

impl From<ExecError> for InferenceError {
    fn from(err: ExecError) -> Self {
        InferenceError::Exec(err)
    }
}

impl From<KvError> for InferenceError {
    fn from(err: KvError) -> Self {
        InferenceError::Kv(err)
    }
}

pub struct InferenceRuntime<'a, K: KernelRegistry, A: AuthorityChecker> {
    pub executor: GraphExecutor<'a, K, A>,
    pub sampler: Sampler,
    pub kv_cache: KvCache<'a>,
}

impl<'a, K: KernelRegistry, A: AuthorityChecker> InferenceRuntime<'a, K, A> {
    pub fn run(
        &mut self,
        tokens: &mut Vec<u32>,
        max_new_tokens: usize,
        tensors: &mut [TensorViewMut<'a, f32>],
    ) -> Result<(), InferenceError> {
        for _ in 0..max_new_tokens {
            self.executor.run(&mut crate::runtime::static_graph_executor::TensorCtx { tensors, kv_cache: None })?;

            let logits_view = tensors.last().ok_or(InferenceError::MissingLogits)?;
            let logits = logits_view.as_view();
            let logits_len = logits.numel();
            let data = logits.data();
            if logits_len == 0 || data.len() < logits_len {
                return Err(InferenceError::EmptyLogits);
            }

            let next = self.sampler.sample(&data[..logits_len]).ok_or(InferenceError::EmptyLogits)?;
            tokens.push(next as u32);

            // TODO: wire kv_cache.append(...) once attention outputs are exposed.
        }
        Ok(())
    }
}
