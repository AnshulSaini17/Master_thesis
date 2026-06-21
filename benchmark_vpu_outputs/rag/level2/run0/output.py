import tensordyne.ir as ir
import tensordyne.ir.instructions.pyxis.vpu as vpu_ops


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type is e.g. ir.Type(ir.Rfp16(eb), (N, n_cols))
    dtype = input_type.dtype
    shape = input_type.shape          # (N, n_cols)
    N      = shape[0]
    n_cols = shape[-1]

    # Row-level type used for per-row ops inside VpuKernel / RepeatGraph.
    row_type  = ir.Type(dtype, (1, n_cols))
    # Full-shape type used as the RepeatGraph body output_type (unrolled).
    full_type = input_type

    # Zero input: caller supplies a zero-filled tensor of the same shape.
    zero_type = ir.Type(dtype, (1, n_cols))   # broadcast scalar row of zeros

    graph = ir.RootGraph()

    # Primary data input.
    input_x = graph.insert(ir.instructions.Input(output_type=input_type))

    # Zero input — a single broadcast row of zeros, shape (1, n_cols).
    # This is loaded once and reused for every row comparison.
    input_zero = graph.insert(ir.instructions.Input(output_type=zero_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load the zero row ONCE outside the loop so it is not re-popped
        # per iteration (vpu_scalar_broadcast_into_row pattern).
        load_zero = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(input_zero,),
            output_type=zero_type,
        )

        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Each iteration pops one row of x from the FIFO.
            # output_type is the FULL shape (N, n_cols) per
            # vpu_rule_repeatgraph_output_shape.
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(input_x, load_zero),
                output_type=full_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(result,),
        output_type=full_type,
    )
    return graph