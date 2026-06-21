
import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements y = clip(x, lo=-5.0, hi=5.0) = max(min(x, hi), lo).

    Compute shape: pointwise / elementwise
    Kernel count:  1 VpuKernel with 1 RepeatGraph(times=N_rows)
    HW ops used:   pyxis.vpu.Scalar, pyxis.vpu.Min, pyxis.vpu.Max
    """
    dtype     = input_type.dtype          # Rfp16(-15) from runner
    n_rows    = input_type.shape[0]       # 4
    n_cols    = input_type.shape[-1]      # 16

    # Shape aliases
    full_type   = ir.Type(dtype, (n_rows, n_cols))   # (4, 16) — used inside RepeatGraph
    scalar_type = ir.Type(dtype, (1, 1))              # (1, 1) — scalar literal

    # Clip constants matching the reference: lo=-5.0, hi=5.0
    lo_val = -5.0
    hi_val =  5.0

    graph   = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:

        # ── Scalar constants (outside RepeatGraph — no FIFO pop per iteration) ──
        hi_lit    = ir.TensorLiteral.from_values(dtype=dtype, values=[hi_val], is_raw=False)
        hi_scalar = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Scalar(value=hi_lit),
            output_type=scalar_type,
        )

        lo_lit    = ir.TensorLiteral.from_values(dtype=dtype, values=[lo_val], is_raw=False)
        lo_scalar = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Scalar(value=lo_lit),
            output_type=scalar_type,
        )

        # ── Row loop: process each of the n_rows rows ──
        with ir.RepeatGraph(parent=vpu_kernel, times=n_rows) as repeat:
            # Pop one row per iteration from the FIFO
            loaded = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=full_type,   # full unrolled shape per vpu_rule_repeatgraph_output_shape
            )
            # min(x, hi)  — clamp from above
            clamped = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(loaded, hi_scalar),
                output_type=full_type,
            )
            # max(clamped, lo) — clamp from below  →  final clip result
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(clamped, lo_scalar),
                output_type=full_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(result,),
        output_type=full_type,
    )
    return graph
