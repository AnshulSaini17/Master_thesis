
import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type carries the full shape, e.g. (4, 16) and dtype rfp16
    n_rows = input_type.shape[0]

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Iterate over all N rows; inside the loop each op pops one row per iteration.
        # output_type is the FULL input shape (RepeatGraph convention).
        with ir.RepeatGraph(parent=vpu_kernel, times=n_rows) as repeat:
            out_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(input_0,),
                output_type=input_type,
            )

    graph.insert(ir.instructions.Output(), args=(out_0,), output_type=input_type)
    return graph
