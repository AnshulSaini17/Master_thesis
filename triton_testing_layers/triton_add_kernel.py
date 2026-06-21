"""Simplest Triton kernel: y = x + x  (element-wise doubling)."""

import triton
import triton.language as tl


@triton.jit
def triton_double_kernel(
    x_ptr,
    y_ptr,
    n: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * n
    y_row = y_ptr + row * n

    for off in range(0, n, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n
        x_vals = tl.load(x_row + cols, mask=mask, other=0)
        y_vals = x_vals + x_vals
        tl.store(y_row + cols, y_vals, mask=mask)
