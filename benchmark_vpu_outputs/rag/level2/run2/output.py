import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """y = max(x, 0) — ReLU — implemented via pyxis.vpu.Max against a zero row."""

    dtype = input_type.dtype          # e.g. Rfp16(eb) or Int16
    shape = input_type.shape          # (N, n_cols)
    N     = shape[0]
    n_cols = shape[-1]

    row_type    = ir.Type(dtype, (1, n_cols))   # single-row type for ops inside VpuKernel
    zero_type   = ir.Type(dtype, (1, n_cols))   # zero scalar row (broadcast)

    graph = ir.RootGraph()

    # Primary input: the multi-row tensor x, shape (N, n_cols).
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # Zero input: a single row of zeros, shape (1, n_cols), broadcast across all N rows.
    zero_input = graph.insert(ir.instructions.Input(output_type=zero_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load the zero row ONCE, outside the loop — it is the same for every row.
        # Per vpu_scalar_broadcast_into_row: load the broadcast operand before the RepeatGraph.
        load_zero = vpu_kernel.insert(
            pyxis.vpu.Load(),
            args=(zero_input,),
            output_type=zero_type,
        )

        # Iterate over all N rows of x.
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Per vpu_elementwise_multi_row: output_type uses the FULL input shape (N, n_cols)
            # inside the RepeatGraph body, not the single-row shape.
            result = repeat.insert(
                pyxis.vpu.Max(),
                args=(input_0, load_zero),
                output_type=input_type,   # full shape (N, n_cols) inside RepeatGraph
            )

    # Output preserves the full input shape.
    graph.insert(
        ir.instructions.Output(),
        args=(result,),
        output_type=input_type,
    )

    return graph