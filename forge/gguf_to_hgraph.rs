
extern crate alloc;

use alloc::vec::Vec;

use crate::forge::hgraph_builder::{BuildError, HGraphBuilder};
use crate::runtime::graph_format::OpCode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileError {
    Unsupported,
    Build(BuildError),
}

impl From<BuildError> for CompileError {
    fn from(err: BuildError) -> Self {
        CompileError::Build(err)
    }
}

/// Placeholder GGUF input descriptor.
/// Replace with a real GGUF reader (gguf-rs) in a std-enabled forge tool.
pub struct GgufModel<'a> {
    pub tensors: &'a [GgufTensor<'a>],
}

pub struct GgufTensor<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub dtype: u32,
}

/// Compile a minimal HGRAPH from a GGUF model description.
/// This builds a single MATMUL node as a proof-of-life scaffold.
pub fn compile_gguf_to_hgraph(model: &GgufModel<'_>) -> Result<Vec<u8>, CompileError> {
    let _ = model;

    let mut builder = HGraphBuilder::new();

    // TODO: Map GGUF layers to HGRAPH nodes.
    // Placeholder: a single MatMul node (inputs 0,1 -> output 2).
    builder.add_node(OpCode::MatMul, &[0, 1], 2, 0)?;

    Ok(builder.finish())
}
