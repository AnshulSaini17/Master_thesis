import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))
    row_type = ir.Type(input_type.dtype, (1, input_type.shape[-1]))
    n_rows = input_type.shape[0]

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        with ir.RepeatGraph(parent=vpu_kernel, times=n_rows) as repeat:
            # Load one row from the FIFO
            load_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=vpu_type,
            )
            # Insert scalar literal 0.0 for the ReLU lower bound
            zero = repeat.insert(
                ir.instructions.pyxis.vpu.LoadZero(),
                args=(),
                output_type=vpu_type,
            )
            # y = max(x, 0)
            relu_out = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(load_0, zero),
                output_type=vpu_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(relu_out,),
        output_type=input_type,
    )
    return graph