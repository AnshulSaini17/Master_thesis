import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    n_rows, n_cols = input_type.shape[0], input_type.shape[-1]
    vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))
    row_type = ir.Type(input_type.dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        with ir.RepeatGraph(parent=vpu_kernel, times=n_rows) as repeat:
            # Load one row from the FIFO
            load_x = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=vpu_type,
            )

            # Broadcast scalar literal `hi` as a row, then apply Min(x, hi)
            hi_lit = repeat.insert(
                ir.instructions.pyxis.vpu.insert_literal(value="hi"),
                args=(),
                output_type=vpu_type,
            )
            clamped = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(load_x, hi_lit),
                output_type=vpu_type,
            )

            # Broadcast scalar literal `lo` as a row, then apply Max(clamped, lo)
            lo_lit = repeat.insert(
                ir.instructions.pyxis.vpu.insert_literal(value="lo"),
                args=(),
                output_type=vpu_type,
            )
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(clamped, lo_lit),
                output_type=vpu_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(result,),
        output_type=input_type,
    )
    return graph