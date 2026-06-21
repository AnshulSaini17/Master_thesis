import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements y[j] = max_i x[i, j]:
    - input_type shape: (N, n_cols)
    - output shape:     (1, n_cols)
    Each Load() pops the next FIFO row; N-1 chained Max ops reduce to one row.
    """
    assert len(input_type.shape) == 2, "Expected 2-D input (N, n_cols)"
    n_rows, n_cols = input_type.shape
    dtype = input_type.dtype

    # Row type: one FIFO row — all in-kernel ops use this shape.
    row_type = ir.Type(dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load all N rows (each Load pops the next row from the FIFO).
        loads = [
            vpu_kernel.insert(
                pyxis.vpu.Load(),
                args=(input_0,),
                output_type=row_type,
            )
            for _ in range(n_rows)
        ]

        # Chain Max reductions: acc = max(acc, next_row) for each subsequent row.
        acc = loads[0]
        for nxt in loads[1:]:
            acc = vpu_kernel.insert(
                pyxis.vpu.Max(),
                args=(acc, nxt),
                output_type=row_type,
            )

    # Output collapses the row axis: (N, n_cols) -> (1, n_cols).
    graph.insert(ir.instructions.Output(), args=(acc,), output_type=row_type)

    return graph