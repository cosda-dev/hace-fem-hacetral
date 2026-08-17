import ctypes
from typing import List


def run_inference(lib_path: str, graph_bytes: bytes, tokens: List[int]) -> List[int]:
    lib = ctypes.CDLL(lib_path)

    graph_buf = (ctypes.c_ubyte * len(graph_bytes)).from_buffer_copy(graph_bytes)
    token_buf = (ctypes.c_uint32 * len(tokens))(*tokens)

    # Placeholder tensor buffer; real binding should pass a mapped tensor pool.
    tensor_buf = (ctypes.c_float * 1)()

    lib.hacedle_run(
        graph_buf,
        len(graph_bytes),
        ctypes.cast(tensor_buf, ctypes.POINTER(ctypes.c_float)),
        1,
        token_buf,
        len(tokens),
    )

    return list(token_buf)
