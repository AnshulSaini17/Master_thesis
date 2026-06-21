import tensordyne.ir as ir
import tensordyne.ir.instructions as inst


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements y[j] = max_i x[i, j] — vertical max reduction across N rows.

    Compute shape: REDUCTION (N rows -> 1 row, element-wise max across rows).
    Strategy: unrolled reduction — N Loads, N-1 chained Max ops.
    Kernel count: 1 (single-pass reduction, no broadcast-back needed).
    """
    N = input_type.shape[0]       # number of rows to reduce (4 for our test)
    n_cols = input_type.shape[1]  # number of columns (16 for our test)

    # All in-kernel ops produce a single FIFO row: shape (1, n_cols).
    row_type = ir.Type(input_type.dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load all N rows — each Load() pops the next row from the FIFO.
        loads = [
            vpu_kernel.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=row_type,
            )
            for _ in range(N)
        ]

        # Chain N-1 element-wise Max ops: result = max(row_0, row_1, ..., row_{N-1})
        acc = loads[0]
        for nxt in loads[1:]:
            acc = vpu_kernel.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, nxt),
                output_type=row_type,
            )

    # Output collapses the row dimension: (N, n_cols) -> (1, n_cols)
    graph.insert(ir.instructions.Output(), args=(acc,), output_type=row_type)
    return graph
