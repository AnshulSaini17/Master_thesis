import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    num_rows = input_type.shape[0]
    n_cols = input_type.shape[-1]

    # Full input shape type (used inside RepeatGraph)
    vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))
    # Single-row type (used outside RepeatGraph)
    row_type = ir.Type(input_type.dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Initialize the accumulator to zero (neutral-ish seed; all rows are
        # fed through Max, so the accumulator converges to the true row-max).
        acc_init = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.LoadZero(),
            args=(),
            output_type=row_type,
        )

        # Materialize the initial accumulator so RepeatGraph can update it.
        acc = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(acc_init,),
            output_type=row_type,
        )

        # RepeatGraph: iterate over every row, updating the running max.
        with ir.RepeatGraph(parent=vpu_kernel, times=num_rows) as repeat:
            # Load one row from the FIFO (pops a row each iteration).
            row_val = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=vpu_type,
            )

            # Elementwise max: running_max = max(running_max, current_row).
            new_acc = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, row_val),
                output_type=vpu_type,
            )

            # Write the updated max back into the accumulator register.
            repeat.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(new_acc,),
                output_type=vpu_type,
            )

        # After the loop, `acc` holds max over all rows — shape (1, n_cols).
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