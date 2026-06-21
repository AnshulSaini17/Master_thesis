import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N = input_type.shape[0]
    n_cols = input_type.shape[-1]
    row_type = ir.Type(input_type.dtype, (1, n_cols))
    full_type = ir.Type(input_type.dtype, (N, n_cols))  # same as input_type

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # LoadZero once outside RepeatGraph — broadcasts a zero register into every row
        # without popping any FIFO rows (scalar broadcast pattern).
        zero = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=row_type),
            output_type=row_type,
        )
        # RepeatGraph drives the VPU over all N rows.
        # All output_types inside use the full unrolled shape (N, n_cols).
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            load_x = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=full_type,
            )
            # Sub(0, x) == -x  (runner-supported; Negate() is tlm_only)
            neg_x = repeat.insert(
                ir.instructions.pyxis.vpu.Sub(),
                args=(zero, load_x),
                output_type=full_type,
            )

    graph.insert(ir.instructions.Output(), args=(neg_x,), output_type=full_type)
    return graph
