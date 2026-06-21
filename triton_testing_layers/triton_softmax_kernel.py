"""Triton kernel: y = softmax(x) along the last axis.

For each row: subtract the row max for numerical stability, exponentiate,
normalize by the sum.
"""

import triton
import triton.language as tl


@triton.jit
def triton_softmax_kernel(
    x_ptr,
    y_ptr,
    n: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * n
    y_row = y_ptr + row * n

    cols = tl.arange(0, block_size)
    mask = cols < n

    # Load the row (n must be <= block_size for this simple form).
    x_vals = tl.load(x_row + cols, mask=mask, other=-float("inf"))

    # Numerical stability: subtract the row max.
    row_max = tl.max(x_vals, axis=0)
    shifted = x_vals - row_max

    # Exp and normalize.
    exp_vals = tl.exp(shifted)
    denom = tl.sum(exp_vals, axis=0)
    y_vals = exp_vals / denom

    tl.store(y_row + cols, y_vals, mask=mask)
