import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements y[j] = max_i x[i, j]: vertical max-reduction across rows.

    input_type: ir.Type with dtype in {Int16, Rfp16, UInt16} and shape (N, n_cols).
    output:     ir.Type with same dtype and shape (1, n_cols).
    """
    assert len(input_type.shape) == 2, "Expected 2-D input (N, n_cols)"
    N, n_cols = input_type.shape
    assert N >= 1, "Need at least one row"

    dtype    = input_type.dtype
    row_type = ir.Type(dtype, (1, n_cols))

    graph   = ir.RootGraph()
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

        # Chain Max reductions: acc = max(acc, next_row)
        acc = loads[0]
        for nxt in loads[1:]:
            acc = vpu_kernel.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, nxt),
                output_type=row_type,
            )

    # Output collapses the row axis: (N, n_cols) -> (1, n_cols)
    graph.insert(
        ir.instructions.Output(),
        args=(acc,),
        output_type=row_type,
    )

    return graph