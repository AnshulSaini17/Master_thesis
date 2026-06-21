
import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N = input_type.shape[0]       # e.g. 4
    n_cols = input_type.shape[1]  # e.g. 16
    dtype = input_type.dtype      # Rfp16(-15)

    # Full unrolled shape used inside RepeatGraph (per vpu_rule_repeatgraph_output_shape)
    loop_type   = ir.Type(dtype, (N, n_cols))
    scalar_type = ir.Type(dtype, (1, 1))

    # clip bounds: lo = -5.0, hi = 5.0  (matching TritonClipRef defaults)
    lo_val = -5.0
    hi_val =  5.0

    graph   = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Embed compile-time scalar constants OUTSIDE the loop (one FIFO pop each)
        hi_literal = ir.TensorLiteral.from_values(dtype=dtype, values=[hi_val], is_raw=False)
        hi_scalar  = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Scalar(value=hi_literal),
            output_type=scalar_type,
        )

        lo_literal = ir.TensorLiteral.from_values(dtype=dtype, values=[lo_val], is_raw=False)
        lo_scalar  = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Scalar(value=lo_literal),
            output_type=scalar_type,
        )

        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Pop one row of x from the FIFO each iteration
            x_row = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=loop_type,
            )
            # min(x, hi) — clamp upper bound
            clamped_hi = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(x_row, hi_scalar),
                output_type=loop_type,
            )
            # max(min(x, hi), lo) — clamp lower bound
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(clamped_hi, lo_scalar),
                output_type=loop_type,
            )

    graph.insert(ir.instructions.Output(), args=(result,), output_type=loop_type)
    return graph
