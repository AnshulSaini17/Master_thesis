import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements y[j] = max_i x[i, j]:
    element-wise max reduction across rows of a (N, n_cols) input,
    producing a (1, n_cols) output.
    """
    # Derive dimensions from input_type
    shape = input_type.shape          # e.g. (N, n_cols)
    N = shape[0]
    n_cols = shape[1]
    dtype = input_type.dtype          # e.g. ir.Rfp16(...) or ir.Int16() or ir.UInt16()

    # Row type: a single row, used for all in-kernel ops
    row_type = ir.Type(dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Unrolled reduction: Load each row (each Load pops the next FIFO row),
        # then chain Max ops to compute the running element-wise maximum.
        loads = [
            vpu_kernel.insert(
                pyxis.vpu.Load(),
                args=(input_0,),
                output_type=row_type,
            )
            for _ in range(N)
        ]

        # Chain Max reductions: acc = max(acc, next_row)
        acc = loads[0]
        for nxt in loads[1:]:
            acc = vpu_kernel.insert(
                pyxis.vpu.Max(),
                args=(acc, nxt),
                output_type=row_type,
            )

    # Output collapses the row dimension: (N, n_cols) -> (1, n_cols)
    graph.insert(ir.instructions.Output(), args=(acc,), output_type=row_type)

    return graph