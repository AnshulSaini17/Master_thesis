import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type has shape (N, n_cols), e.g. (N, block_size)
    N = input_type.shape[0]

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Each iteration pops one row and applies absolute value.
            # output_type is the FULL input shape (not row_type) — per RepeatGraph convention.
            out_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(input_0,),
                output_type=input_type,
            )

    graph.insert(ir.instructions.Output(), args=(out_0,), output_type=input_type)
    return graph