import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # Derive shape info from input_type
    dtype = input_type.dtype
    shape = input_type.shape          # e.g. (N, n_cols)
    N = shape[0]
    n_cols = shape[-1]

    row_type  = ir.Type(dtype, (1, n_cols))   # single-row type (register shape)
    full_type = input_type                    # (N, n_cols) — used for RepeatGraph body

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # Build a zero-row constant to broadcast as the lower bound for ReLU.
    zeros = graph.insert(
        ir.instructions.Constant(value=0),
        output_type=row_type,
    )

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load the zero constant ONCE, outside the loop — one FIFO pop.
        load_zeros = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(zeros,),
            output_type=row_type,
        )

        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Each iteration pops one row of x from the FIFO.
            # output_type is the FULL shape per vpu_rule_repeatgraph_output_shape.
            relu_out = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(input_0, load_zeros),
                output_type=full_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(relu_out,),
        output_type=full_type,
    )
    return graph