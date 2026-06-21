
import tensordyne.ir as ir
from tensordyne import emu
from tensordyne.conversion.lowering.vpu import (
    prepare_for_vpu_processing,
    restore_from_vpu_processing,
)
from tensordyne.conversion.pipeline.passes.lut_values import get_lut_values


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    N      = input_type.shape[0]   # e.g. 4  (rows)
    n_cols = input_type.shape[1]   # e.g. 64 (columns)
    dtype  = input_type.dtype      # Rfp16(-15)
    eb     = dtype.eb              # -15

    # LUT RECIP helpers
    lut_table  = get_lut_values(ir.instructions.Reciprocal)
    lut_kind   = ir.instructions.pyxis.vpu.LUT.LutKind.RECIP
    recip_eb   = emu.ops.vpu.recip_get_out_eb(eb)
    recip_dtype = ir.Rfp16(recip_eb)

    graph   = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # ══════════════════════════════════════════════════════════════════════
    # PASS 1 — Per-row absmax, computed as a vertical max in transposed space
    # ══════════════════════════════════════════════════════════════════════
    #
    # Transpose x (N, n_cols) → x_t (n_cols, N) so the VPU can reduce vertically.
    # Each "row" of x_t contains ONE element per original row:
    #   x_t[j, i] = x[i, j]
    # After reducing all n_cols rows of x_t we get the per-row absmax as a (1, N) row.
    x_t = prepare_for_vpu_processing(graph, input_0, reduction_dim=-1)
    # x_t shape: (n_cols, N) = (64, 4)

    # Type annotations:
    #   outside RepeatGraph → register-shape  (1, N)
    #   inside  RepeatGraph → full-unrolled   (n_cols, N)
    acc_reg_type    = ir.Type(dtype,       (1, N))
    loop_t          = ir.Type(dtype,       (n_cols, N))
    recip_reg_type  = ir.Type(recip_dtype, (1, N))

    with ir.VpuKernel(parent=graph) as kernel_1:
        # Running-max accumulator, initialised to 0 (safe since |x| ≥ 0).
        max_acc = kernel_1.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=acc_reg_type)
        )

        with ir.RepeatGraph(parent=kernel_1, times=n_cols) as repeat:
            row = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_t,),
                output_type=loop_t,
            )
            abs_row = repeat.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(row,),
                output_type=loop_t,
            )
            new_max = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(max_acc, abs_row),
                output_type=loop_t,
            )
            max_acc = repeat.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(max_acc, new_max),
                output_type=loop_t,
            )
        # max_acc: (1, N) = (1, 4) after the loop — one absmax per original row.

        # Fuse LUT RECIP here: compute 1/absmax while still inside kernel_1.
        # Output is recip_reg_type (1, N) rfp16(recip_eb).
        recip_absmax = kernel_1.insert(
            ir.instructions.pyxis.vpu.LUT(
                lut_kind=lut_kind,
                table=lut_table,
                output_eb=recip_eb,
            ),
            args=(max_acc,),
            output_type=recip_reg_type,
        )
    # recip_absmax: (1, N) = (1, 4), rfp16(recip_eb)
    # This is kernel_1's output; it will be written to AMEM for kernel_2.

    # ══════════════════════════════════════════════════════════════════════
    # PASS 2 — Normalise in transposed space using "load-once-outside-loop"
    # ══════════════════════════════════════════════════════════════════════
    #
    # Fresh second transpose of x for kernel_2.
    # (Each VpuKernel has its own FIFO budget — vpu_rule_fifo_budget_per_kernel)
    x_t2 = prepare_for_vpu_processing(graph, input_0, reduction_dim=-1)
    # x_t2 shape: (n_cols, N) = (64, 4)

    # Result type for the normalised output (still in transposed space).
    loop_out_t = ir.Type(dtype, (n_cols, N))   # (64, 4)

    with ir.VpuKernel(parent=graph) as kernel_2:
        # Load recip_absmax ONCE, outside the loop.
        # recip_absmax has exactly 1 row → its FIFO is fully consumed here,
        # satisfying the FIFO-ordering rule before x_t2 starts.
        load_recip = kernel_2.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(recip_absmax,),
            output_type=recip_reg_type,   # (1, N) register
        )

        # For each of the n_cols rows of x_t2: multiply by the pre-loaded
        # recip_absmax register.  `x_t2` appears as a direct FIFO arg, so
        # each iteration implicitly pops one row (vpu_rule_fifo_pop).
        # `load_recip` is a register from outside the loop → broadcast.
        with ir.RepeatGraph(parent=kernel_2, times=n_cols) as repeat:
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(x_t2, load_recip),
                output_type=loop_out_t,   # (64, 4) full-unrolled
            )
    # result: (n_cols, N) = (64, 4) in transposed space

    # ══════════════════════════════════════════════════════════════════════
    # Un-transpose: (n_cols, N) → (N, n_cols)
    # restore_from_vpu_processing reverses the prepare permutation.
    # ══════════════════════════════════════════════════════════════════════
    out_type = ir.Type(dtype, (N, n_cols))
    final = restore_from_vpu_processing(
        graph,
        result,
        pre_vpu_shape=input_type.shape,   # (N, n_cols) = (4, 64)
        output_type=out_type,
        reduction_dim=-1,
    )

    graph.insert(ir.instructions.Output(), args=(final,), output_type=out_type)
    return graph
