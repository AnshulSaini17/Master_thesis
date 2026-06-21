import tensordyne.ir as ir
import tensordyne.ir.instructions.pyxis.vpu as pyxis_vpu


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements y = clip(x, lo, hi) = max(min(x, hi), lo).

    Uses pyxis.vpu.Min then pyxis.vpu.Max element-wise over every row.
    input_type is expected to be e.g. ir.Type(ir.Rfp16(eb), (N, n_cols)).
    """
    graph = ir.RootGraph()

    # Primary input: the tensor x to be clipped.
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # lo and hi are broadcast constants — same row shape, repeated N times.
    # They are passed in as separate inputs of the same type (each is a
    # (N, n_cols) tensor filled with the scalar, or a (1, n_cols) broadcast).
    input_lo = graph.insert(ir.instructions.Input(output_type=input_type))
    input_hi = graph.insert(ir.instructions.Input(output_type=input_type))

    dtype   = input_type.dtype
    N       = input_type.shape[0]
    n_cols  = input_type.shape[1]

    # Inside a VpuKernel, ops consume one FIFO row at a time: shape (1, n_cols).
    row_type = ir.Type(dtype, (1, n_cols))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Pop one row from each input FIFO per iteration.
            # min(x, hi)  →  clamp the upper bound first.
            clamped_hi = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(input_0, input_hi),
                output_type=input_type,   # RepeatGraph body uses FULL shape
            )
            # max(result, lo)  →  clamp the lower bound.
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(clamped_hi, input_lo),
                output_type=input_type,   # RepeatGraph body uses FULL shape
            )

    graph.insert(
        ir.instructions.Output(),
        args=(result,),
        output_type=input_type,
    )
    return graph