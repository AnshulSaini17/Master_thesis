import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))
    row_type = ir.Type(input_type.dtype, (1, input_type.shape[-1]))
    n_rows = input_type.shape[0]

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        with ir.RepeatGraph(parent=vpu_kernel, times=n_rows) as repeat:
            # Load one row from the input FIFO
            load_0 = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=vpu_type,
            )

            # Insert scalar literal row for hi
            hi_lit = repeat.insert(
                ir.instructions.pyxis.vpu.insert_literal(hi),
                args=(),
                output_type=vpu_type,
            )

            # Insert scalar literal row for lo
            lo_lit = repeat.insert(
                ir.instructions.pyxis.vpu.insert_literal(lo),
                args=(),
                output_type=vpu_type,
            )

            # clamped = min(x, hi)
            clamped = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(load_0, hi_lit),
                output_type=vpu_type,
            )

            # y = max(clamped, lo)
            y_vals = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(clamped, lo_lit),
                output_type=vpu_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(y_vals,),
        output_type=input_type,
    )
    return graph