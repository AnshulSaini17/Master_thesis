import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))
    row_type = ir.Type(input_type.dtype, (1, input_type.shape[-1]))
    n_rows = input_type.shape[0]

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        with ir.RepeatGraph(parent=vpu_kernel, times=n_rows) as repeat:
            # Load one row of x from FIFO
            load_x = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=vpu_type,
            )

            # Insert scalar literal row for hi
            lit_hi = repeat.insert(
                ir.instructions.pyxis.vpu.insert_literal(ir.Literal(input_type.dtype, "hi")),
                args=(),
                output_type=vpu_type,
            )

            # Insert scalar literal row for lo
            lit_lo = repeat.insert(
                ir.instructions.pyxis.vpu.insert_literal(ir.Literal(input_type.dtype, "lo")),
                args=(),
                output_type=vpu_type,
            )

            # clamped = min(x, hi)
            min_op = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(load_x, lit_hi),
                output_type=vpu_type,
            )

            # y = max(clamped, lo)
            max_op = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(min_op, lit_lo),
                output_type=vpu_type,
            )

        result = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(max_op,),
            output_type=vpu_type,
        )

    graph.insert(
        ir.instructions.Output(),
        args=(result,),
        output_type=vpu_type,
    )
    return graph