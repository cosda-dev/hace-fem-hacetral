
use crate::core::ops::attention::attention_with_kv;
use crate::core::ops::matmul::matmul_f32;
use crate::core::tensor::TensorViewMut;
use crate::runtime::graph_format::{GraphEdge, GraphError, GraphNode, GraphView, OpCode};
use crate::runtime::kv_cache::KvCache;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecError {
    Graph(GraphError),
    InvalidTensorIndex,
    InvalidInputCount,
    AliasViolation,
    UnsupportedOp,
    MissingKvCache,
    MatMul,
    Attention,
    AuthorityDenied,
}

impl From<GraphError> for ExecError {
    fn from(err: GraphError) -> Self {
        ExecError::Graph(err)
    }
}

pub struct TensorCtx<'a, 'b> {
    pub tensors: &'b mut [TensorViewMut<'a, f32>],
    pub kv_cache: Option<*mut KvCache<'a>>,
}

pub trait KernelRegistry {
    fn execute<'a, 'b>(
        &self,
        node: &GraphNode,
        inputs: &[GraphEdge],
        ctx: &mut TensorCtx<'a, 'b>,
    ) -> Result<(), ExecError>;
}

pub trait AuthorityChecker {
    fn check(&self, node: &GraphNode, const_pool: &[u8]) -> Result<(), ExecError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoAuthority;

impl AuthorityChecker for NoAuthority {
    fn check(&self, _node: &GraphNode, _const_pool: &[u8]) -> Result<(), ExecError> {
        Ok(())
    }
}

pub struct CoreKernels;

impl KernelRegistry for CoreKernels {
    fn execute<'a, 'b>(
        &self,
        node: &GraphNode,
        inputs: &[GraphEdge],
        ctx: &mut TensorCtx<'a, 'b>,
    ) -> Result<(), ExecError> {
        match node.op_code {
            x if x == OpCode::MatMul as u16 => exec_matmul(inputs, node.output, ctx),
            x if x == OpCode::AttentionKV as u16 => exec_attention_kv(inputs, node.authority_offset, node.output, ctx),
            _ => Err(ExecError::UnsupportedOp),
        }
    }
}

pub struct GraphExecutor<'a, K: KernelRegistry, A: AuthorityChecker> {
    pub graph: GraphView<'a>,
    pub kernels: K,
    pub authority: A,
}

impl<'a, K: KernelRegistry> GraphExecutor<'a, K, NoAuthority> {
    pub fn new(graph: GraphView<'a>, kernels: K) -> Self {
        Self {
            graph,
            kernels,
            authority: NoAuthority,
        }
    }
}

impl<'a, K: KernelRegistry, A: AuthorityChecker> GraphExecutor<'a, K, A> {
    pub fn with_authority(graph: GraphView<'a>, kernels: K, authority: A) -> Self {
        Self {
            graph,
            kernels,
            authority,
        }
    }

    pub fn run<'t, 'u>(&self, ctx: &mut TensorCtx<'t, 'u>) -> Result<(), ExecError> {
        let edges = self.graph.edges();
        for node in self.graph.nodes() {
            self.authority.check(node, self.graph.const_pool())?;

            let start = node.input_start as usize;
            let end = start
                .checked_add(node.input_len as usize)
                .ok_or(ExecError::InvalidInputCount)?;
            if end > edges.len() {
                return Err(ExecError::InvalidInputCount);
            }

            let inputs = &edges[start..end];
            self.kernels.execute(node, inputs, ctx)?;
        }
        Ok(())
    }
}

fn exec_matmul<'a, 'b>(
    inputs: &[GraphEdge],
    output: u32,
    ctx: &mut TensorCtx<'a, 'b>,
) -> Result<(), ExecError> {
    if inputs.len() < 2 {
        return Err(ExecError::InvalidInputCount);
    }

    let len = ctx.tensors.len();
    let a_idx = inputs[0].tensor_id as usize;
    let b_idx = inputs[1].tensor_id as usize;
    let out_idx = output as usize;

    if a_idx >= len || b_idx >= len || out_idx >= len {
        return Err(ExecError::InvalidTensorIndex);
    }
    if a_idx == out_idx || b_idx == out_idx {
        return Err(ExecError::AliasViolation);
    }

    let ptr = ctx.tensors.as_mut_ptr();
    let a = unsafe { &*ptr.add(a_idx) }.as_view();
    let b = unsafe { &*ptr.add(b_idx) }.as_view();
    let out = unsafe { &mut *ptr.add(out_idx) };

    matmul_f32(&a, &b, out).map_err(|_| ExecError::MatMul)
}

fn exec_attention_kv<'a, 'b>(
    inputs: &[GraphEdge],
    layer_id: u32,
    output: u32,
    ctx: &mut TensorCtx<'a, 'b>,
) -> Result<(), ExecError> {
    if inputs.len() < 1 {
        return Err(ExecError::InvalidInputCount);
    }

    let len = ctx.tensors.len();
    let q_idx = inputs[0].tensor_id as usize;
    let out_idx = output as usize;

    if q_idx >= len || out_idx >= len {
        return Err(ExecError::InvalidTensorIndex);
    }
    if q_idx == out_idx {
        return Err(ExecError::AliasViolation);
    }

    let kv_ptr = ctx.kv_cache.ok_or(ExecError::MissingKvCache)?;
    let kv = unsafe { &mut *kv_ptr };
    let layer = layer_id as usize;
    if layer >= kv.k.len() || layer >= kv.v.len() {
        return Err(ExecError::InvalidTensorIndex);
    }

    let ptr = ctx.tensors.as_mut_ptr();
    let q = unsafe { &*ptr.add(q_idx) }.as_view();
    let out = unsafe { &mut *ptr.add(out_idx) };

    let k_cache = kv.k[layer].as_view();
    let v_cache = kv.v[layer].as_view();

    attention_with_kv(&q, &k_cache, &v_cache, out).map_err(|_| ExecError::Attention)
}
