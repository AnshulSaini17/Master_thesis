import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N = input_type.shape[0]        # 4
    n_cols = input_type.shape[-1]  # 16
    dtype = input_type.dtype       # Rfp16(-15)

    # Row-level type for ops that live outside the RepeatGraph
    row_type = ir.Type(dtype, (1, n_cols))
    # Full unrolled shape — required for every op INSIDE the RepeatGraph
    # (vpu_rule_repeatgraph_output_shape)
    full_type = ir.Type(dtype, (N, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # LoadZero lives OUTSIDE the RepeatGraph; produces a single (1, n_cols)
        # row of zeros.  Used as the constant 0 comparand in max(x, 0).
        zero = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=row_type),
            args=(),
            output_type=row_type,
        )

        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Each iteration pops one FIFO row from input_0 (implicit FIFO pop).
            # Max(x, 0) == ReLU(x) — single HW-native op, no composition needed.
            relu = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(input_0, zero),
                output_type=full_type,
            )

    graph.insert(ir.instructions.Output(), args=(relu,), output_type=full_type)
    return graph
