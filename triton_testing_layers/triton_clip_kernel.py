"""Triton kernel: y = clip(x, lo, hi) = max(min(x, hi), lo)."""

import triton
import triton.language as tl


@triton.jit
def triton_clip_kernel(
    x_ptr,
    y_ptr,
    n: tl.constexpr,
    block_size: tl.constexpr,
    lo: tl.constexpr,
    hi: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * n
    y_row = y_ptr + row * n

    for off in range(0, n, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n
        x_vals = tl.load(x_row + cols, mask=mask, other=0)
        clamped = tl.minimum(x_vals, hi)
        y_vals = tl.maximum(clamped, lo)
        tl.store(y_row + cols, y_vals, mask=mask)
