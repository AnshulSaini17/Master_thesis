
import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    n_rows = input_type.shape[0]
    n_cols = input_type.shape[-1]
    full_type = ir.Type(input_type.dtype, (n_rows, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        with ir.RepeatGraph(parent=vpu_kernel, times=n_rows) as repeat:
            load_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=full_type,
            )
            abs_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(load_0,),
                output_type=full_type,
            )

    graph.insert(ir.instructions.Output(), args=(abs_0,), output_type=full_type)
    return graph
