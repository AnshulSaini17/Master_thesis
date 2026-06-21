
import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type: ir.Type(rfp16, (4, 16))
    dtype = input_type.dtype
    shape = tuple(input_type.shape)          # (4, 16)
    N = shape[0]                             # 4 rows

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Iterate over all N rows; each iteration pops one row from the FIFO.
        # Per vpu_rule_repeatgraph_output_shape: output_type inside RepeatGraph
        # must use the FULL unrolled shape (N, n_cols), not the per-row shape.
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            out = repeat.insert(
                ir.instructions.pyxis.vpu.Negate(),
                args=(input_0,),
                output_type=input_type,   # full shape (4, 16)
            )

    graph.insert(ir.instructions.Output(), args=(out,), output_type=input_type)
    return graph
