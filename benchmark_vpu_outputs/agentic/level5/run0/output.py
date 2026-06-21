
from tensordyne import ir, emu
from tensordyne.conversion.lowering.vpu import (
    prepare_for_vpu_processing,
    restore_from_vpu_processing,
)
from tensordyne.conversion.pipeline.passes.lut_values import get_lut_values


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """Absmax-normalisation kernel in VPU IR.

    Implements (for each batch row i):
        absmax_i   = max_j( |x[i, j]| )
        y[i, j]    = x[i, j] / absmax_i        (* weight is 1 → no-op)

    Strategy
    --------
    Horizontal max over C columns is mapped to a *vertical* reduction in the
    VPU's FIFO domain by transposing x via prepare_for_vpu_processing.

      prepare_for_vpu_processing(x (B, C), reduction_dim=-1)
        → x_T of shape (C, B)     [C FIFO rows, each a (B,) vector]

    Kernel 1: Absolute + Max accumulation over C FIFO rows
              → per-row absmax, shape (1, B); then RECIP LUT → scale (1, B).

    Kernel 2: Multiply each of the C columns of x_T by the (1, B) scale.
              restore_from_vpu_processing → output (B, C).
    """
    B, C = input_type.shape   # e.g. (4, 64)
    dtype = input_type.dtype  # Rfp16
    eb    = dtype.eb

    # ---- Types in the transposed (C, B) FIFO domain -------------------------
    # Outside a RepeatGraph → register shape (1, B)
    # Inside  a RepeatGraph → full-unroll shape (C, B)
    acc_reg_type   = ir.Type(dtype, (1, B))   # (1, B)
    loop_full_type = ir.Type(dtype, (C, B))   # (C, B)

    # ---- Reciprocal LUT: 1 / absmax -----------------------------------------
    recip_out_eb = emu.ops.vpu.recip_get_out_eb(eb)
    recip_dtype  = ir.Rfp16(recip_out_eb)
    recip_table  = get_lut_values(
        ir.instructions.Reciprocal,
        input_dtype=dtype,
        output_dtype=recip_dtype,
    )

    # ---- Graph skeleton ------------------------------------------------------
    graph   = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # Two independent transposed views of x (one per kernel pass)
    x_T  = prepare_for_vpu_processing(graph, input_0, reduction_dim=-1)
    x_T2 = prepare_for_vpu_processing(graph, input_0, reduction_dim=-1)

    # =========================================================================
    # Kernel 1 — per-row absmax, then reciprocal
    # =========================================================================
    with ir.VpuKernel(parent=graph) as k1:

        # Accumulator register initialised to 0: shape (1, B)
        acc = k1.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=acc_reg_type)
        )

        # Vertical max-of-abs reduction across C feature columns
        with ir.RepeatGraph(parent=k1, times=C) as repeat:
            x_col = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_T,),
                output_type=loop_full_type,        # full-unroll (C, B)
            )
            x_abs = repeat.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(x_col,),
                output_type=loop_full_type,
            )
            new_max = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, x_abs),
                output_type=loop_full_type,
            )
            acc = repeat.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(acc, new_max),
                output_type=loop_full_type,
            )

        # Post-loop: acc register holds per-row absmax, shape (1, B).
        # Apply RECIP LUT → scale = 1 / absmax, shape (1, B).
        scale = k1.insert(
            ir.instructions.pyxis.vpu.LUT(
                lut_kind=ir.instructions.pyxis.vpu.LUT.LutKind.RECIP,
                table=recip_table,
                output_eb=recip_out_eb,
            ),
            args=(acc,),
            output_type=ir.Type(recip_dtype, (1, B)),  # register shape
        )

    # =========================================================================
    # Kernel 2 — multiply every column of x by the per-row scale
    # =========================================================================
    mul_full_type = ir.Type(dtype, (C, B))

    with ir.VpuKernel(parent=graph) as k2:
        # Load scale ONCE outside the loop; it is reused across all C iterations
        load_scale = k2.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(scale,),
            output_type=ir.Type(recip_dtype, (1, B)),
        )

        with ir.RepeatGraph(parent=k2, times=C) as repeat:
            x_col2 = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_T2,),
                output_type=loop_full_type,        # full-unroll (C, B)
            )
            y_col = repeat.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(x_col2, load_scale),
                output_type=mul_full_type,
            )

    # ---- Restore transposed result (C, B) → (B, C) --------------------------
    out_type = ir.Type(dtype, input_type.shape)
    result = restore_from_vpu_processing(
        graph,
        y_col,
        pre_vpu_shape=input_type.shape,
        output_type=out_type,
        reduction_dim=-1,
    )

    graph.insert(ir.instructions.Output(), args=(result,), output_type=out_type)
    return graph
