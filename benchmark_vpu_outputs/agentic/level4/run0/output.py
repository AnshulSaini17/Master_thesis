
import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N = input_type.shape[0]       # 4 rows
    n_cols = input_type.shape[1]  # 16 columns

    # Row shape: each in-kernel op produces one FIFO row (vpu_rule_in_kernel_row_shape)
    row_type = ir.Type(input_type.dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load all N rows — each Load() pops the next row from the FIFO
        loads = [
            vpu_kernel.insert(
                pyxis.vpu.Load(),
                args=(input_0,),
                output_type=row_type,
            )
            for _ in range(N)
        ]

        # Chain N-1 element-wise Max ops to compute column-wise maximum
        acc = loads[0]
        for nxt in loads[1:]:
            acc = vpu_kernel.insert(
                pyxis.vpu.Max(),
                args=(acc, nxt),
                output_type=row_type,
            )

    # Output collapses (N, n_cols) -> (1, n_cols): the reduced max row
    graph.insert(ir.instructions.Output(), args=(acc,), output_type=row_type)
    return graph
