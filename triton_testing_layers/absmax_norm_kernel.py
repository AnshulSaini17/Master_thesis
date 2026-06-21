import torch
import triton
import triton.language as tl


@triton.jit
def absmax_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    n: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * n
    y_row = y_ptr + row * n

    # Pass 1: find max(abs(x)) over the row
    _max = tl.zeros([block_size], dtype=tl.float32)
    for off in range(0, n, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n
        x_vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        _max = tl.maximum(_max, tl.abs(x_vals))
    absmax = tl.max(_max, axis=0)

    # Pass 2: normalize and scale by weight
    for off in range(0, n, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n
        x_vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y_vals = (x_vals / (absmax + eps)) * w_vals
        tl.store(y_row + cols, y_vals, mask=mask)
