import tensordyne.ir as ir
from tensordyne.ir import instructions
from tensordyne.ir.instructions import pyxis


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements y[j] = max_i x[i, j] — column-wise max reduction across rows.

    Compute shape: vertical reduction (N rows -> 1 row).
    Strategy: unrolled — N Load()s (each pops the next FIFO row) + N-1 chained
    pyxis.vpu.Max ops.  No RepeatGraph needed because N=4 is small and fixed.

    All in-kernel output_types use row_type (1, n_cols) per vpu_rule_in_kernel_row_shape.
    Output also uses row_type since the row axis is collapsed by the reduction.
    """
    N = input_type.shape[0]          # 4
    n_cols = input_type.shape[1]     # 16
    row_type = ir.Type(input_type.dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Pop all N rows from the FIFO — each Load() advances to the next row.
        loads = [
            vpu_kernel.insert(
                pyxis.vpu.Load(),
                args=(input_0,),
                output_type=row_type,
            )
            for _ in range(N)
        ]

        # Chain element-wise Max: mirrors the Triton accumulator init + loop.
        # loads[0] acts as the accumulator seed (row 0), then Max with rows 1..N-1.
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
