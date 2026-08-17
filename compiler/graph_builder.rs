
extern crate alloc;

use alloc::vec::Vec;

use crate::forge::hgraph_builder::{BuildError, HGraphBuilder};
use crate::runtime::graph_format::OpCode;

pub type TensorId = u32;

#[derive(Clone, Copy, Debug)]
pub struct ConstBinding {
    pub tensor_id: TensorId,
    pub name_hash: u64,
}

pub struct GraphBuilder {
    inner: HGraphBuilder,
    next_tensor: TensorId,
    const_bindings: Vec<ConstBinding>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            inner: HGraphBuilder::new(),
            next_tensor: 0,
            const_bindings: Vec::new(),
        }
    }

    pub fn new_tensor(&mut self) -> TensorId {
        let id = self.next_tensor;
        self.next_tensor = self.next_tensor.wrapping_add(1);
        id
    }

    pub fn bind_const(&mut self, name_hash: u64) -> TensorId {
        let id = self.new_tensor();
        self.const_bindings.push(ConstBinding { tensor_id: id, name_hash });
        id
    }

    pub fn add_op(&mut self, op: OpCode, inputs: &[TensorId]) -> Result<TensorId, BuildError> {
        let output = self.new_tensor();
        self.inner.add_node(op, inputs, output, 0)?;
        Ok(output)
    }

    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, BuildError> {
        self.add_op(OpCode::MatMul, &[a, b])
    }

    pub fn rope(&mut self, x: TensorId) -> Result<TensorId, BuildError> {
        self.add_op(OpCode::Rope, &[x])
    }

    pub fn attention_kv(&mut self, q: TensorId, layer_id: usize) -> Result<TensorId, BuildError> {
        let output = self.new_tensor();
        self.inner.add_node(OpCode::AttentionKV, &[q], output, layer_id as u32)?;
        Ok(output)
    }

    pub fn rmsnorm(&mut self, x: TensorId) -> Result<TensorId, BuildError> {
        self.add_op(OpCode::RmsNorm, &[x])
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, BuildError> {
        self.add_op(OpCode::Add, &[a, b])
    }

    pub fn finish(self) -> (Vec<u8>, Vec<ConstBinding>) {
        (self.inner.finish(), self.const_bindings)
    }
}
