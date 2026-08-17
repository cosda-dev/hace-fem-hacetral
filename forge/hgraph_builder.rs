
extern crate alloc;

use alloc::vec::Vec;
use core::mem;

use crate::runtime::graph_format::{GraphEdge, GraphHeader, GraphNode, OpCode, GRAPH_MAGIC_U32};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    TooManyInputs,
    TooManyEdges,
}

pub struct HGraphBuilder {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    const_pool: Vec<u8>,
}

impl HGraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            const_pool: Vec::new(),
        }
    }

    pub fn push_const(&mut self, bytes: &[u8]) -> u32 {
        let offset = self.const_pool.len() as u32;
        self.const_pool.extend_from_slice(bytes);
        offset
    }

    pub fn add_node(
        &mut self,
        op: OpCode,
        inputs: &[u32],
        output: u32,
        authority_offset: u32,
    ) -> Result<u32, BuildError> {
        if inputs.len() > u16::MAX as usize {
            return Err(BuildError::TooManyInputs);
        }
        if self.edges.len() + inputs.len() > u32::MAX as usize {
            return Err(BuildError::TooManyEdges);
        }

        let input_start = self.edges.len() as u32;
        for &tensor_id in inputs {
            self.edges.push(GraphEdge { tensor_id });
        }

        let node = GraphNode {
            op_code: op as u16,
            input_start,
            input_len: inputs.len() as u16,
            output,
            authority_offset,
        };
        self.nodes.push(node);

        Ok((self.nodes.len() - 1) as u32)
    }

    pub fn finish(self) -> Vec<u8> {
        let header_size = mem::size_of::<GraphHeader>();
        let node_size = mem::size_of::<GraphNode>();
        let edge_size = mem::size_of::<GraphEdge>();

        let node_bytes = node_size * self.nodes.len();
        let edge_bytes = edge_size * self.edges.len();

        let mut const_offset = header_size + node_bytes + edge_bytes;
        let pad = align_up(const_offset, 4) - const_offset;
        const_offset += pad;

        let mut out = Vec::with_capacity(const_offset + self.const_pool.len());

        write_header(
            &mut out,
            GraphHeader {
                magic: GRAPH_MAGIC_U32,
                version: 0x0001,
                node_count: self.nodes.len() as u32,
                edge_count: self.edges.len() as u32,
                const_offset: const_offset as u64,
            },
        );

        for node in &self.nodes {
            write_node(&mut out, node);
        }
        for edge in &self.edges {
            write_edge(&mut out, edge);
        }

        out.resize(const_offset, 0u8);
        out.extend_from_slice(&self.const_pool);

        out
    }
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

fn write_header(buf: &mut Vec<u8>, header: GraphHeader) {
    buf.extend_from_slice(&header.magic.to_le_bytes());
    buf.extend_from_slice(&header.version.to_le_bytes());
    buf.extend_from_slice(&[0u8; 2]); // padding
    buf.extend_from_slice(&header.node_count.to_le_bytes());
    buf.extend_from_slice(&header.edge_count.to_le_bytes());
    buf.extend_from_slice(&header.const_offset.to_le_bytes());
}

fn write_node(buf: &mut Vec<u8>, node: &GraphNode) {
    buf.extend_from_slice(&node.op_code.to_le_bytes());
    buf.extend_from_slice(&[0u8; 2]); // padding
    buf.extend_from_slice(&node.input_start.to_le_bytes());
    buf.extend_from_slice(&node.input_len.to_le_bytes());
    buf.extend_from_slice(&[0u8; 2]); // padding
    buf.extend_from_slice(&node.output.to_le_bytes());
    buf.extend_from_slice(&node.authority_offset.to_le_bytes());
}

fn write_edge(buf: &mut Vec<u8>, edge: &GraphEdge) {
    buf.extend_from_slice(&edge.tensor_id.to_le_bytes());
}
