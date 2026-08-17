
extern crate alloc;

use crate::compiler::graph_builder::{GraphBuilder, TensorId};
use crate::compiler::shard::{ShardWriter, TensorRef};
use crate::compiler::gguf_to_hgraph::{AttentionLayer, CompileError, Layer, MLPLayer, TensorData};

pub fn map_layer(
    layer: &Layer<'_>,
    layer_id: usize,
    input: TensorId,
    graph: &mut GraphBuilder,
    shard: &mut ShardWriter,
) -> Result<TensorId, CompileError> {
    match layer {
        Layer::Attention(attn) => map_attention(attn, layer_id, input, graph, shard),
        Layer::MLP(mlp) => map_mlp(mlp, input, graph, shard),
    }
}

fn map_attention(
    attn: &AttentionLayer<'_>,
    layer_id: usize,
    input: TensorId,
    graph: &mut GraphBuilder,
    shard: &mut ShardWriter,
) -> Result<TensorId, CompileError> {
    let wq = bind_weight(&attn.wq, graph, shard)?;
    let wk = bind_weight(&attn.wk, graph, shard)?;
    let wv = bind_weight(&attn.wv, graph, shard)?;

    let q = graph.matmul(input, wq)?;
    let k = graph.matmul(input, wk)?;
    let _v = graph.matmul(input, wv)?;

    let q_rope = graph.rope(q)?;
    let _k_rope = graph.rope(k)?;

    // TODO: support transpose/attention score scaling via op params.
    let attn_out = graph.attention_kv(q_rope, layer_id)?;

    let wo = bind_weight(&attn.wo, graph, shard)?;
    let out = graph.matmul(attn_out, wo)?;

    Ok(out)
}

fn map_mlp(
    mlp: &MLPLayer<'_>,
    input: TensorId,
    graph: &mut GraphBuilder,
    shard: &mut ShardWriter,
) -> Result<TensorId, CompileError> {
    let w1 = bind_weight(&mlp.w1, graph, shard)?;
    let w2 = bind_weight(&mlp.w2, graph, shard)?;
    let w3 = bind_weight(&mlp.w3, graph, shard)?;

    let x1 = graph.matmul(input, w1)?;
    let x2 = graph.matmul(input, w3)?;

    let act = graph.add(x1, x1)?; // placeholder for SiLU
    let gated = graph.matmul(act, x2)?; // placeholder for elementwise mul

    let out = graph.matmul(gated, w2)?;
    Ok(out)
}

fn bind_weight(
    tensor: &TensorData<'_>,
    graph: &mut GraphBuilder,
    shard: &mut ShardWriter,
) -> Result<TensorId, CompileError> {
    let bytes = tensor.bytes;
    shard
        .write_tensor(tensor.name, bytes, tensor.dtype, tensor.device)
        .map(|TensorRef { name_hash, .. }| graph.bind_const(name_hash))
        .map_err(|err| CompileError::Shard(err))
}
