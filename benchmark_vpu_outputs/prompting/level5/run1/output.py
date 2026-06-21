import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    dtype = input_type.dtype
    n_rows, n_cols = input_type.shape[0], input_type.shape[-1]

    vpu_type   = ir.Type(dtype, (n_rows, n_cols))   # full shape, used inside RepeatGraph
    row_type   = ir.Type(dtype, (1, n_cols))         # single row, used outside RepeatGraph
    scalar_row = ir.Type(dtype, (1, n_cols))         # reused name for clarity

    graph = ir.RootGraph()

    # Two inputs: x and w
    input_x = graph.insert(ir.instructions.Input(output_type=input_type))
    input_w = graph.insert(ir.instructions.Input(output_type=input_type))

    # ------------------------------------------------------------------ #
    # VpuKernel 1: compute absmax over each row of x                      #
    # Result shape: (n_rows, n_cols) but every column holds the same      #
    # per-row absmax value — the HW reduces to a broadcast row.           #
    # ------------------------------------------------------------------ #
    with ir.VpuKernel(parent=graph) as vpu1:
        # Accumulator: start at zero, shape = one row
        acc = vpu1.insert(
            ir.instructions.pyxis.vpu.LoadZero(),
            args=(),
            output_type=row_type,
        )

        with ir.RepeatGraph(parent=vpu1, times=n_rows) as rg1:
            # Load one row of x
            x_row = rg1.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_x,),
                output_type=vpu_type,
            )
            # |x|
            abs_x = rg1.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(x_row,),
                output_type=vpu_type,
            )
            # running max with accumulator
            new_acc = rg1.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, abs_x),
                output_type=vpu_type,
            )
            # write back to accumulator
            acc_assign = rg1.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(new_acc,),
                output_type=vpu_type,
            )

        # absmax_row holds the per-row maximum |x| value (shape: row_type)
        absmax_row = vpu1.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(acc_assign,),
            output_type=row_type,
        )

    # ------------------------------------------------------------------ #
    # VpuKernel 2: normalize x by absmax and scale by w                  #
    # y = (x / (absmax + eps)) * w                                        #
    # ------------------------------------------------------------------ #
    eps_val = 1e-6  # matches typical usage; adjust if eps is a param

    with ir.VpuKernel(parent=graph) as vpu2:
        with ir.RepeatGraph(parent=vpu2, times=n_rows) as rg2:
            # Load current row of x
            x_row2 = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_x,),
                output_type=vpu_type,
            )
            # Load corresponding row of w
            w_row = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_w,),
                output_type=vpu_type,
            )
            # Broadcast absmax_row to full shape for elementwise ops
            absmax_broad = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(absmax_row,),
                output_type=vpu_type,
            )
            # eps literal row
            eps_lit = rg2.insert(
                ir.instructions.pyxis.vpu.insert_literal(value=eps_val),
                args=(),
                output_type=vpu_type,
            )
            # absmax + eps
            denom = rg2.insert(
                ir.instructions.pyxis.vpu.Add(),
                args=(absmax_broad, eps_lit),
                output_type=vpu_type,
            )
            # 1 / (absmax + eps)  — represented as x * (1/denom) via reciprocal
            # Use Mul with reciprocal literal approach:
            # x / denom = x * (1.0 / denom). Since there's no Div op listed,
            # emit a ones literal and divide by multiplying x by inv_denom.
            # The HW provides Mul; compute inv via ones/denom pattern:
            ones_lit = rg2.insert(
                ir.instructions.pyxis.vpu.insert_literal(value=1.0),
                args=(),
                output_type=vpu_type,
            )
            # inv_denom = 1.0 / denom  (Sub-based Newton or direct Recip if available)
            # Use pyxis.vpu.Reciprocal if present; otherwise express as Mac pattern.
            # Per the native ops list, emit Mul(ones, inv) — but the cleanest
            # available path without a Div primitive: use Sub trick is forbidden.
            # We emit pyxis.vpu.Reciprocal as the HW-native op for 1/x.
            inv_denom = rg2.insert(
                ir.instructions.pyxis.vpu.Reciprocal(),
                args=(denom,),
                output_type=vpu_type,
            )
            # x * inv_denom
            x_norm = rg2.insert(
                ir.instructions.pyxis.vpu.Mul(),
                args=(x_row2, inv_denom),
                output_type=vpu_type,
            )
            # x_norm * w
            y_row = rg2.insert(
                ir.instructions.pyxis.vpu.Mul(),
                args=(x_norm, w_row),
                output_type=vpu_type,
            )
            out_assign = rg2.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(y_row,),
                output_type=vpu_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(out_assign,),
        output_type=input_type,
    )
    return graph