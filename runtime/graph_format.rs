
use core::{mem, slice};

pub const GRAPH_MAGIC_U32: u32 = u32::from_le_bytes(*b"HGRF");
const GRAPH_VERSION: u16 = 0x0001;

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpCode {
    MatMul = 1,
    Add = 2,
    RmsNorm = 3,
    Rope = 4,
    AttentionKV = 5,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GraphHeader {
    pub magic: u32,
    pub version: u16,
    pub node_count: u32,
    pub edge_count: u32,
    pub const_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GraphNode {
    pub op_code: u16,
    pub input_start: u32,
    pub input_len: u16,
    pub output: u32,
    pub authority_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GraphEdge {
    pub tensor_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    InvalidMagic,
    UnsupportedVersion,
    OutOfBounds,
    Misaligned,
    InvalidLayout,
}

#[derive(Clone, Copy, Debug)]
pub struct GraphView<'a> {
    header: &'a GraphHeader,
    nodes: &'a [GraphNode],
    edges: &'a [GraphEdge],
    const_pool: &'a [u8],
}

impl<'a> GraphView<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, GraphError> {
        if data.len() < mem::size_of::<GraphHeader>() {
            return Err(GraphError::OutOfBounds);
        }

        let header = unsafe { &*(data.as_ptr() as *const GraphHeader) };
        if header.magic != GRAPH_MAGIC_U32 {
            return Err(GraphError::InvalidMagic);
        }
        if header.version != GRAPH_VERSION {
            return Err(GraphError::UnsupportedVersion);
        }

        let header_size = mem::size_of::<GraphHeader>();
        let node_size = mem::size_of::<GraphNode>();
        let edge_size = mem::size_of::<GraphEdge>();

        let node_count = header.node_count as usize;
        let edge_count = header.edge_count as usize;

        let node_bytes = node_size
            .checked_mul(node_count)
            .ok_or(GraphError::OutOfBounds)?;
        let edge_bytes = edge_size
            .checked_mul(edge_count)
            .ok_or(GraphError::OutOfBounds)?;

        let node_start = header_size;
        let node_end = node_start
            .checked_add(node_bytes)
            .ok_or(GraphError::OutOfBounds)?;

        let edge_start = node_end;
        let edge_end = edge_start
            .checked_add(edge_bytes)
            .ok_or(GraphError::OutOfBounds)?;

        if edge_end > data.len() {
            return Err(GraphError::OutOfBounds);
        }
        if node_start % mem::align_of::<GraphNode>() != 0 {
            return Err(GraphError::Misaligned);
        }
        if edge_start % mem::align_of::<GraphEdge>() != 0 {
            return Err(GraphError::Misaligned);
        }

        let const_start = header.const_offset as usize;
        if const_start < edge_end {
            return Err(GraphError::InvalidLayout);
        }
        if const_start > data.len() {
            return Err(GraphError::OutOfBounds);
        }

        let nodes = unsafe {
            slice::from_raw_parts(
                data.as_ptr().add(node_start) as *const GraphNode,
                node_count,
            )
        };
        let edges = unsafe {
            slice::from_raw_parts(
                data.as_ptr().add(edge_start) as *const GraphEdge,
                edge_count,
            )
        };
        let const_pool = &data[const_start..];

        Ok(Self {
            header,
            nodes,
            edges,
            const_pool,
        })
    }

    #[inline]
    pub fn header(&self) -> &GraphHeader {
        self.header
    }

    #[inline]
    pub fn nodes(&self) -> &[GraphNode] {
        self.nodes
    }

    #[inline]
    pub fn edges(&self) -> &[GraphEdge] {
        self.edges
    }

    #[inline]
    pub fn const_pool(&self) -> &[u8] {
        self.const_pool
    }
}
