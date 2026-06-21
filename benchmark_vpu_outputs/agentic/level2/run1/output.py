import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N = input_type.shape[0]       # e.g. 4
    n_cols = input_type.shape[1]  # e.g. 16
    dtype = input_type.dtype

    row_type  = ir.Type(dtype, (1, n_cols))
    full_type = ir.Type(dtype, (N, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Initialize a zero-filled register (register-shape, outside RepeatGraph).
        zero = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=row_type)
        )

        # Element-wise ReLU: max(x, 0) for every row.
        # RepeatGraph unrolls over all N rows; ops inside use the full unrolled shape.
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            x_load = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=full_type,
            )
            relu = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(x_load, zero),
                output_type=full_type,
            )

    graph.insert(ir.instructions.Output(), args=(relu,), output_type=full_type)
    return graph
