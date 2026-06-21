"""Triton kernel: sum all rows together. y[j] = sum_i x[i, j]."""

import triton
import triton.language as tl


@triton.jit
def triton_row_accumulate_kernel(
    x_ptr,
    y_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    col_block = tl.program_id(0)
    cols = col_block * block_size + tl.arange(0, block_size)
    mask = cols < n_cols

    acc = tl.zeros([block_size], dtype=tl.float32)
    for row in range(n_rows):
        x_vals = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0)
        acc += x_vals

    tl.store(y_ptr + cols, acc, mask=mask)
