import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    num_rows = input_type.shape[0]
    n_cols = input_type.shape[-1]

    vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))
    row_type = ir.Type(input_type.dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load the first row from the FIFO to seed the accumulator
        first_row = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(input_0,),
            output_type=row_type,
        )
        # Assign it so it can be used as the initial accumulator
        acc = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(first_row,),
            output_type=row_type,
        )

        # Repeat over the remaining (num_rows - 1) rows
        with ir.RepeatGraph(parent=vpu_kernel, times=num_rows - 1) as repeat:
            # Inside RepeatGraph, output_type uses full input shape per the rules
            row_val = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=row_type,
            )
            new_acc = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, row_val),
                output_type=row_type,
            )
            acc = repeat.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(new_acc,),
                output_type=row_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(acc,),
        output_type=row_type,
    )
    return graph