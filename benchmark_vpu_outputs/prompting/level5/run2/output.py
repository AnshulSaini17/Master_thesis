import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type describes x: (n_rows, n_cols)
    dtype = input_type.dtype
    n_rows, n_cols = input_type.shape

    vpu_type = ir.Type(dtype, (n_rows, n_cols))
    row_type = ir.Type(dtype, (1, n_cols))

    eps = 1e-8  # matches typical usage; constant literal per row

    graph = ir.RootGraph()

    # Two inputs: x of shape (n_rows, n_cols), w of shape (1, n_cols)
    input_x = graph.insert(ir.instructions.Input(output_type=input_type))
    w_type = ir.Type(dtype, (1, n_cols))
    input_w = graph.insert(ir.instructions.Input(output_type=w_type))

    # ------------------------------------------------------------------ #
    # VpuKernel 1: Reduce each row of x to find absmax row-by-row,
    #              producing a single row of per-column-max then a global
    #              row-max scalar broadcast.  We emit one accumulator row
    #              that holds max(abs(x)) across all rows, per column,
    #              then that is already a (1, n_cols) row which the second
    #              kernel will use.
    # ------------------------------------------------------------------ #
    with ir.VpuKernel(parent=graph) as vpu_kernel_1:
        # Accumulator initialised to zero
        acc = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.LoadZero(),
            args=(),
            output_type=row_type,
        )

        # Repeat over n_rows: load a row of x, abs, max into accumulator
        with ir.RepeatGraph(parent=vpu_kernel_1, times=n_rows) as repeat_1:
            load_x = repeat_1.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_x,),
                output_type=vpu_type,
            )
            abs_x = repeat_1.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(load_x,),
                output_type=vpu_type,
            )
            # acc is broadcast into the repeat body shape for Max
            new_acc = repeat_1.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, abs_x),
                output_type=vpu_type,
            )
            assign_acc = repeat_1.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(new_acc,),
                output_type=vpu_type,
            )

        # After the loop, assign_acc holds the per-column absmax as (1, n_cols)
        # Materialise the final accumulator row
        absmax_row = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(assign_acc,),
            output_type=row_type,
        )

        # Add eps to absmax: need an eps literal row
        eps_lit = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.insert_literal(value=eps),
            args=(),
            output_type=row_type,
        )
        absmax_eps = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.Add(),
            args=(absmax_row, eps_lit),
            output_type=row_type,
        )
        # Compute scale = 1 / (absmax + eps) as a (1, n_cols) row
        one_lit = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.insert_literal(value=1.0),
            args=(),
            output_type=row_type,
        )
        scale = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.Div(),
            args=(one_lit, absmax_eps),
            output_type=row_type,
        )
        # Multiply scale by w (both are row_type)
        load_w = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(input_w,),
            output_type=row_type,
        )
        scale_w = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.Mul(),
            args=(scale, load_w),
            output_type=row_type,
        )
        # Assign scale_w so it can be used in kernel 2
        scale_w_assigned = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(scale_w,),
            output_type=row_type,
        )

    # ------------------------------------------------------------------ #
    # VpuKernel 2: Stream each row of x again, multiply by scale_w,
    #              producing (n_rows, n_cols) output.
    # ------------------------------------------------------------------ #
    with ir.VpuKernel(parent=graph) as vpu_kernel_2:
        with ir.RepeatGraph(parent=vpu_kernel_2, times=n_rows) as repeat_2:
            load_x2 = repeat_2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_x,),
                output_type=vpu_type,
            )
            # scale_w_assigned is (1, n_cols); broadcast across repeat body
            y_vals = repeat_2.insert(
                ir.instructions.pyxis.vpu.Mul(),
                args=(load_x2, scale_w_assigned),
                output_type=vpu_type,
            )
            out_row = repeat_2.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(y_vals,),
                output_type=vpu_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(out_row,),
        output_type=vpu_type,
    )

    return graph