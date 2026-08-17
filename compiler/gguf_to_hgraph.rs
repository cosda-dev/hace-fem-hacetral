
extern crate alloc;

use alloc::vec::Vec;

use crate::compiler::graph_builder::{ConstBinding, GraphBuilder, TensorId};
use crate::compiler::mapper::map_layer;
use crate::compiler::shard::{ShardBinding, ShardBlob, ShardError, ShardWriter};
use crate::forge::hgraph_builder::BuildError;
use crate::index::mto_index::{DType, Device};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileError {
    Unsupported,
    Shard(ShardError),
    GraphBuild(BuildError),
}

impl From<BuildError> for CompileError {
    fn from(err: BuildError) -> Self {
        CompileError::GraphBuild(err)
    }
}

pub struct CompileConfig {
    pub shard_size: usize,
}

pub struct CompileOutput {
    pub hgraph: Vec<u8>,
    pub mto_index: Vec<u8>,
    pub shards: Vec<ShardBlob>,
    pub const_bindings: Vec<ConstBinding>,
    pub shard_map: Vec<ShardBinding>,
}

pub struct GGUFModel<'a> {
    pub layers: &'a [Layer<'a>],
}

pub enum Layer<'a> {
    Attention(AttentionLayer<'a>),
    MLP(MLPLayer<'a>),
}

pub struct TensorData<'a> {
    pub name: &'a str,
    pub bytes: &'a [u8],
    pub dtype: DType,
    pub device: Device,
}

pub struct AttentionLayer<'a> {
    pub wq: TensorData<'a>,
    pub wk: TensorData<'a>,
    pub wv: TensorData<'a>,
    pub wo: TensorData<'a>,
}

pub struct MLPLayer<'a> {
    pub w1: TensorData<'a>,
    pub w2: TensorData<'a>,
    pub w3: TensorData<'a>,
}

pub fn compile_gguf(gguf: &GGUFModel<'_>, cfg: &CompileConfig) -> Result<CompileOutput, CompileError> {
    let mut graph = GraphBuilder::new();
    let mut shard = ShardWriter::new(cfg.shard_size);

    let mut current: TensorId = graph.new_tensor();

    for (layer_id, layer) in gguf.layers.iter().enumerate() {
        current = map_layer(layer, layer_id, current, &mut graph, &mut shard)?;
    }

    let (hgraph, const_bindings) = graph.finish();
    let (shards, mto_index, shard_map) = shard.finalize().map_err(CompileError::Shard)?;

    Ok(CompileOutput {
        hgraph,
        mto_index,
        shards,
        const_bindings,
        shard_map,
    })
}
