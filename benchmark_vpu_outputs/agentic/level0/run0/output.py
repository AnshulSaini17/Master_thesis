import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type: shape=(4, 16), dtype=rfp16
    N = input_type.shape[0]  # number of rows = 4

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Each iteration pops one row from the FIFO and negates it.
            # output_type is the FULL input shape per the vpu_elementwise_multi_row pattern.
            out_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Negate(),
                args=(input_0,),
                output_type=input_type,
            )

    graph.insert(ir.instructions.Output(), args=(out_0,), output_type=input_type)
    return graph
