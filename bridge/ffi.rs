
use core::slice;

use crate::runtime::graph_format::GraphView;
use crate::runtime::static_graph_executor::{CoreKernels, GraphExecutor};

#[no_mangle]
pub extern "C" fn hacedle_run(
    graph_ptr: *const u8,
    graph_len: usize,
    tensor_ptr: *mut f32,
    tensor_len: usize,
    token_ptr: *mut u32,
    token_len: usize,
) {
    if graph_ptr.is_null() || graph_len == 0 {
        return;
    }

    let graph_bytes = unsafe { slice::from_raw_parts(graph_ptr, graph_len) };
    let graph = match GraphView::parse(graph_bytes) {
        Ok(view) => view,
        Err(_) => return,
    };

    if tensor_ptr.is_null() || tensor_len == 0 {
        return;
    }
    if token_ptr.is_null() || token_len == 0 {
        return;
    }

    // TODO: Build TensorViewMut slices from tensor_ptr + layout metadata.
    // TODO: Execute inference loop and update tokens.
    let _ = GraphExecutor::new(graph, CoreKernels);
}
