"""Triton kernel: y[j] = max_i x[i, j] — element-wise max reduction across rows."""

import triton
import triton.language as tl


@triton.jit
def triton_max_reduction_kernel(
    x_ptr,
    y_ptr,
    num_rows: tl.constexpr,
    n: tl.constexpr,
    block_size: tl.constexpr,
):
    # Process one tile of columns per program
    for off in range(0, n, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n

        # Initialize accumulator with first row
        acc = tl.load(x_ptr + 0 * n + cols, mask=mask, other=0)

        # Take elementwise max with each subsequent row
        for row in range(1, num_rows):
            row_vals = tl.load(x_ptr + row * n + cols, mask=mask, other=0)
            acc = tl.maximum(acc, row_vals)

        tl.store(y_ptr + cols, acc, mask=mask)
