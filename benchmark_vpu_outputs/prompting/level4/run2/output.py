import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    num_rows = input_type.shape[0]
    n_cols = input_type.shape[-1]

    # Full input shape type and single-row type
    vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))
    row_type = ir.Type(input_type.dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Initialize accumulator to zero (matches Triton's other=0 seed)
        acc_init = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.LoadZero(),
            args=(),
            output_type=row_type,
        )

        # Assign the zero accumulator so it can be referenced across iterations
        acc = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(acc_init,),
            output_type=row_type,
        )

        # RepeatGraph: for each of the num_rows rows, load and take elementwise Max
        with ir.RepeatGraph(parent=vpu_kernel, times=num_rows) as repeat:
            # Load one row from the FIFO (pops from input stream)
            row_val = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=vpu_type,
            )

            # Elementwise max of accumulator and current row
            new_acc = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, row_val),
                output_type=vpu_type,
            )

            # Write back into accumulator
            repeat.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(new_acc,),
                output_type=vpu_type,
            )

        # After the loop, acc holds the per-column maximum (one row)
        result = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(acc,),
            output_type=row_type,
        )

    graph.insert(
        ir.instructions.Output(),
        args=(result,),
        output_type=row_type,
    )

    return graph