import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    n_rows, n_cols = input_type.shape
    dtype = input_type.dtype

    vpu_type = ir.Type(dtype, (n_rows, n_cols))
    row_type = ir.Type(dtype, (1, n_cols))

    graph = ir.RootGraph()

    # Two inputs: x and w
    input_x = graph.insert(ir.instructions.Input(output_type=input_type))
    input_w = graph.insert(ir.instructions.Input(output_type=input_type))

    # ------------------------------------------------------------------ #
    # VpuKernel 1: Pass 1 — compute absmax = max(abs(x)) across all rows
    # Result shape: (1, n_cols) — one row holding the absmax value
    # ------------------------------------------------------------------ #
    with ir.VpuKernel(parent=graph) as vpu_kernel_1:

        # Accumulator initialised to zero: shape (1, n_cols)
        acc = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.LoadZero(),
            args=(),
            output_type=row_type,
        )

        # RepeatGraph: iterate over all n_rows, accumulating max(abs(x))
        with ir.RepeatGraph(parent=vpu_kernel_1, times=n_rows) as rg1:
            # Load one row of x — inside RepeatGraph output_type = vpu_type
            x_row = rg1.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_x,),
                output_type=vpu_type,
            )

            # abs(x_row)
            abs_x = rg1.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(x_row,),
                output_type=vpu_type,
            )

            # acc = max(acc, abs(x_row))
            new_acc = rg1.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, abs_x),
                output_type=vpu_type,
            )

            # Write back to accumulator
            acc = rg1.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(new_acc,),
                output_type=vpu_type,
            )

        # After the loop, acc holds max(abs(x)) per column: shape (1, n_cols)
        absmax = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(acc,),
            output_type=row_type,
        )

        # Add eps to absmax: insert a literal row of eps
        eps_literal = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.insert_literal(1e-6),
            args=(),
            output_type=row_type,
        )

        absmax_eps = vpu_kernel_1.insert(
            ir.instructions.pyxis.vpu.Add(),
            args=(absmax, eps_literal),
            output_type=row_type,
        )

    # ------------------------------------------------------------------ #
    # VpuKernel 2: Pass 2 — compute y = (x / (absmax + eps)) * w
    # ------------------------------------------------------------------ #
    with ir.VpuKernel(parent=graph) as vpu_kernel_2:

        with ir.RepeatGraph(parent=vpu_kernel_2, times=n_rows) as rg2:
            # Load one row of x
            x_row2 = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_x,),
                output_type=vpu_type,
            )

            # Load one row of w
            w_row = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_w,),
                output_type=vpu_type,
            )

            # Broadcast absmax_eps (1, n_cols) into (n_rows, n_cols) for division
            absmax_eps_bcast = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(absmax_eps,),
                output_type=vpu_type,
            )

            # x / (absmax + eps)
            x_normed = rg2.insert(
                ir.instructions.pyxis.vpu.Div(),
                args=(x_row2, absmax_eps_bcast),
                output_type=vpu_type,
            )

            # (x / (absmax + eps)) * w
            y_row = rg2.insert(
                ir.instructions.pyxis.vpu.Mul(),
                args=(x_normed, w_row),
                output_type=vpu_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(y_row,),
        output_type=vpu_type,
    )

    return graph