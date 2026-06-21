
import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N = input_type.shape[0]
    n_cols = input_type.shape[1]
    dtype = input_type.dtype

    # Register-shape (outside RepeatGraph): single row
    reg_type  = ir.Type(dtype, (1, n_cols))
    # Full unrolled shape (inside RepeatGraph): all N rows annotated together
    full_type = ir.Type(dtype, (N, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # LoadZero OUTSIDE the RepeatGraph — creates a (1, n_cols) zero register.
        # This acts as a broadcast constant, not a FIFO pop.
        zero = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=reg_type)
        )

        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Load one row of x from the FIFO per iteration.
            # output_type = full_type per vpu_rule_repeatgraph_output_shape.
            x_load = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=full_type,
            )
            # Element-wise max(x, 0) — HW-native ReLU.
            # zero (reg_type) from outside loop is broadcast across all N rows.
            # Both operands share the same EB → satisfies vpu_rule_eb_alignment.
            out = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(x_load, zero),
                output_type=full_type,
            )

    graph.insert(ir.instructions.Output(), args=(out,), output_type=full_type)
    return graph
