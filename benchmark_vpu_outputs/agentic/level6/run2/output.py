
import tensordyne.ir as ir
from tensordyne import emu
from tensordyne.conversion.lowering.vpu import (
    prepare_for_vpu_processing,
    restore_from_vpu_processing,
)
from tensordyne.conversion.pipeline.passes.lut_values import get_lut_values


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Row-wise softmax: y[i,j] = exp(x[i,j] - max_i) / sum_k(exp(x[i,k] - max_i))

    Strategy:
      1. prepare_for_vpu_processing: transpose (N, M) → (M, N) so the
         feature/softmax axis (M=16) becomes the VPU row axis.
         Vertical reductions now give per-batch-row statistics.
      2. K1 (Max): unrolled vertical Max over M rows → (1, N) per-row max.
      3. K2 (ExpSum): RepeatGraph(times=M): load x_T row, Sub max, Exp,
         ExtendedAdd into Rfp21 accumulator → Cast to (1, N) sum.
      4. K3 (Norm): LUT RECIP of sum (output_eb from recip_get_out_eb),
         then RepeatGraph(times=M): load x_T row, Sub max, Exp,
         Mul by recip (output_eb=eb) → (M, N) normalised rows.
      5. restore_from_vpu_processing: transpose (M, N) → (N, M) back.
    """
    N = input_type.shape[0]   # batch rows  (4)
    M = input_type.shape[1]   # feature cols / softmax dim  (16)
    dtype = input_type.dtype  # Rfp16(-15)
    eb = dtype.eb             # -15

    # Compute the hardware-mandated output EB for RECIP of Rfp16(eb)
    recip_out_eb = emu.ops.vpu.recip_get_out_eb(eb)   # -17 for eb=-15
    recip_dtype  = ir.Rfp16(recip_out_eb)

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # ── Transpose (N, M) → (M, N) so reduction axis becomes VPU row axis ──
    x_t = prepare_for_vpu_processing(graph, input_0, reduction_dim=-1)
    # x_t.output_type.shape == (M, N) == (16, 4)

    # Shared type helpers
    row_t       = ir.Type(dtype,        (1, N))   # (1, 4)  – Rfp16(-15) register shape
    loop_t      = ir.Type(dtype,        (M, N))   # (16, 4) – Rfp16(-15) full-unrolled
    recip_row_t = ir.Type(recip_dtype,  (1, N))   # (1, 4)  – Rfp16(-17) for RECIP output
    rfp21_reg   = ir.Type(ir.Rfp21(eb), (1, N))   # (1, 4)  – Rfp21 accumulator register
    rfp21_loop  = ir.Type(ir.Rfp21(eb), (M, N))   # (16, 4) – Rfp21 full-unrolled

    # ══════════════════════════════════════════════════════════════════════
    # Kernel 1 – vertical Max over M=16 rows of x_t → per-batch-row max
    #   Unrolled: avoids RepeatGraph shape ambiguity for the accumulator.
    # ══════════════════════════════════════════════════════════════════════
    with ir.VpuKernel(parent=graph) as k_max:
        max_acc = k_max.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(x_t,),
            output_type=row_t,
        )
        for _ in range(M - 1):
            nxt = k_max.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_t,),
                output_type=row_t,
            )
            max_acc = k_max.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(max_acc, nxt),
                output_type=row_t,
            )
    # max_acc: (1, N) = [max_row0, max_row1, max_row2, max_row3]

    # ══════════════════════════════════════════════════════════════════════
    # Kernel 2 – sum of exp(x - max), using Rfp21 accumulator for precision
    # ══════════════════════════════════════════════════════════════════════
    with ir.VpuKernel(parent=graph) as k_sum:
        # Load max once OUTSIDE the loop (broadcast-scalar pattern)
        max_s = k_sum.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(max_acc,),
            output_type=row_t,
        )
        # Initialise Rfp21 accumulator to zero
        sum_acc = k_sum.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=rfp21_reg)
        )
        with ir.RepeatGraph(parent=k_sum, times=M) as r_sum:
            xr = r_sum.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_t,),
                output_type=loop_t,
            )
            sh = r_sum.insert(
                ir.instructions.pyxis.vpu.Sub(),
                args=(xr, max_s),
                output_type=loop_t,
            )
            ev = r_sum.insert(
                ir.instructions.pyxis.vpu.Exp(output_eb=eb),
                args=(sh,),
                output_type=loop_t,
            )
            ns = r_sum.insert(
                ir.instructions.pyxis.vpu.ExtendedAdd(),
                args=(ev, sum_acc),
                output_type=rfp21_loop,
            )
            sum_acc = r_sum.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(sum_acc, ns),
                output_type=rfp21_loop,
            )
        # Bridge: Rfp21 (M, N) full-unrolled → Rfp16(eb) (1, N) register shape
        sum_rfp16 = k_sum.insert(
            ir.instructions.pyxis.vpu.Cast(output_dtype=ir.Rfp16(eb)),
            args=(sum_acc,),
            output_type=row_t,
        )
    # sum_rfp16: (1, N) = [sum_row0, sum_row1, sum_row2, sum_row3]

    # ── Build reciprocal LUT table (output EB = recip_out_eb) ───────────
    lut_table = get_lut_values(
        ir.instructions.Reciprocal,
        input_dtype=ir.Rfp16(eb),
        output_dtype=recip_dtype,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Kernel 3 – normalise: exp(x - max) / sum = exp(x - max) * recip(sum)
    # ══════════════════════════════════════════════════════════════════════
    with ir.VpuKernel(parent=graph) as k_norm:
        # Load max and sum ONCE outside the loop
        max_n = k_norm.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(max_acc,),
            output_type=row_t,
        )
        sum_n = k_norm.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(sum_rfp16,),
            output_type=row_t,
        )
        # Reciprocal of sum via hardware LUT (RECIP kind)
        # Output EB is recip_out_eb (hardware-mandated for RECIP)
        recip = k_norm.insert(
            ir.instructions.pyxis.vpu.LUT(
                lut_kind=ir.instructions.pyxis.vpu.LUT.LutKind.RECIP,
                table=lut_table,
                output_eb=recip_out_eb,
            ),
            args=(sum_n,),
            output_type=recip_row_t,
        )
        with ir.RepeatGraph(parent=k_norm, times=M) as r_norm:
            xr_n = r_norm.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_t,),
                output_type=loop_t,
            )
            sh_n = r_norm.insert(
                ir.instructions.pyxis.vpu.Sub(),
                args=(xr_n, max_n),
                output_type=loop_t,
            )
            ev_n = r_norm.insert(
                ir.instructions.pyxis.vpu.Exp(output_eb=eb),
                args=(sh_n,),
                output_type=loop_t,
            )
            # exp(eb) * recip(recip_out_eb); override output to Rfp16(eb)
            norm = r_norm.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(ev_n, recip),
                output_type=loop_t,
            )
    # norm: (M, N) = (16, 4) normalised softmax in transposed layout

    # ── Restore layout: (M, N) → (N, M) = (4, 16) ──────────────────────
    out_type = ir.Type(dtype, (N, M))
    result = restore_from_vpu_processing(
        graph,
        norm,
        pre_vpu_shape=input_type.shape,
        output_type=out_type,
        reduction_dim=-1,
    )

    graph.insert(ir.instructions.Output(), args=(result,), output_type=out_type)
    return graph
