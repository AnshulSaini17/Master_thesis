
from tensordyne import ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N      = input_type.shape[0]          # number of rows  (4)
    n_cols = input_type.shape[-1]         # columns         (16)

    # Full unrolled shape — required by vpu_rule_repeatgraph_output_shape
    full_type = ir.Type(input_type.dtype, (N, n_cols))

    graph   = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # RepeatGraph streams all N rows through the FIFO one at a time,
        # but the IR annotates every op with the full unrolled shape.
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            load_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=full_type,   # full (N, n_cols), not row (1, n_cols)
            )
            abs_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(load_0,),
                output_type=full_type,
            )

    graph.insert(ir.instructions.Output(), args=(abs_0,), output_type=full_type)
    return graph
