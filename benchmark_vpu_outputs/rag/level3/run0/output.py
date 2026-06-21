import tensordyne.ir as ir
import tensordyne.ir.instructions.pyxis.vpu as pyxis_vpu


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements clip(x, lo, hi) = max(min(x, hi), lo) via VPU Min + Max.

    input_type is expected to be e.g. ir.Type(ir.Rfp16(eb), (N, n_cols))
    or ir.Type(ir.Int16(), (N, n_cols)) etc.
    """
    graph = ir.RootGraph()

    # Primary input: the values to be clipped.
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # lo and hi are per-element broadcast scalars; they share the same dtype
    # but are single-row tensors that will be loaded once and reused each iteration.
    dtype = input_type.dtype
    n_cols = input_type.shape[-1]
    N = input_type.shape[0]

    row_type = ir.Type(dtype, (1, n_cols))

    # lo_input and hi_input are (1, n_cols) constant tensors passed in.
    lo_input = graph.insert(ir.instructions.Input(output_type=row_type))
    hi_input = graph.insert(ir.instructions.Input(output_type=row_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load the lo and hi bounds once, outside the loop.
        # They are (1, n_cols) scalars broadcast across all N rows.
        load_lo = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(lo_input,),
            output_type=row_type,
        )
        load_hi = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(hi_input,),
            output_type=row_type,
        )

        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Pop one row of x per iteration.
            # Step 1: min(x, hi)  — clamp from above.
            clamped_hi = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(input_0, load_hi),
                output_type=input_type,   # full unrolled shape inside RepeatGraph
            )
            # Step 2: max(min(x, hi), lo)  — clamp from below.
            clipped = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(clamped_hi, load_lo),
                output_type=input_type,   # full unrolled shape inside RepeatGraph
            )

    graph.insert(
        ir.instructions.Output(),
        args=(clipped,),
        output_type=input_type,
    )
    return graph