
import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N = input_type.shape[0]
    n_cols = input_type.shape[-1]

    dtype = input_type.dtype
    loop_type = ir.Type(dtype, (N, n_cols))
    scalar_type = ir.Type(dtype, (1, 1))

    # Clip bounds matching the reference: lo=-5.0, hi=5.0
    LO = -5.0
    HI = 5.0

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Embed compile-time scalar constants for hi and lo
        hi_literal = ir.TensorLiteral.from_values(
            dtype=dtype, values=[HI], is_raw=False
        )
        lo_literal = ir.TensorLiteral.from_values(
            dtype=dtype, values=[LO], is_raw=False
        )

        hi_scalar = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Scalar(value=hi_literal),
            output_type=scalar_type,
        )
        lo_scalar = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Scalar(value=lo_literal),
            output_type=scalar_type,
        )

        # Process all N rows: min(x, hi) then max(result, lo)
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            x_row = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=loop_type,
            )
            # Step 1: clamp to upper bound: min(x, hi)
            clamped = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(x_row, hi_scalar),
                output_type=loop_type,
            )
            # Step 2: clamp to lower bound: max(clamped, lo)
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(clamped, lo_scalar),
                output_type=loop_type,
            )

    graph.insert(ir.instructions.Output(), args=(result,), output_type=loop_type)
    return graph
