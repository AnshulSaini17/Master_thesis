
import math

import tensordyne.ir as ir
from tensordyne import emu
from tensordyne.conversion.lowering.vpu import (
    prepare_for_vpu_processing,
    restore_from_vpu_processing,
)
from tensordyne.conversion.pipeline.passes.lut_values import get_lut_values


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """Row-wise softmax: y = softmax(x, dim=-1) for x of shape (B, N).

    Strategy:
      1. Transpose (B, N) → (N, B) so the per-row reduction becomes a
         VPU-native vertical reduction.
      2. Triplicate the transposed tensor (N, B) → (3*N, B) so a single
         VpuKernel can make three sequential passes:
           Phase 1 – running Max       → max register  (1, B)
           Phase 2 – exp(x-max) + sum  → sum register  (1, B)
           Phase 3 – exp(x-max) / sum  → output        (N, B)
      3. Transpose (N, B) → (B, N) to recover the original layout.
    """
    dtype = input_type.dtype    # Rfp16(-15) typically
    eb    = dtype.eb            # -15
    B     = input_type.shape[0] # batch  = number of independent softmax rows (4)
    N     = input_type.shape[1] # length = softmax dimension (16)

    # --- EB arithmetic ---------------------------------------------------
    exp_eb  = eb            # exp(x - max) ∈ [0, 1]  → Rfp16(-15) fine
    # sum ≤ N  →  we need  (2^k * 2^eb) ≥ N, i.e. k = ceil(log2(N+1))
    k       = math.ceil(math.log2(N + 1))   # = 5 for N=16
    sum_eb  = eb + k                         # = -10  (Rfp16(-10) max ≈ 32 ✓)
    # reciprocal output EB: emu helper knows the HW rule
    recip_out_eb = emu.ops.vpu.recip_get_out_eb(sum_eb)
    result_eb = eb          # final softmax output ∈ [0, 1] → Rfp16(-15)

    # --- LUT table for 1/x -----------------------------------------------
    recip_table = get_lut_values(ir.instructions.Reciprocal)

    # --- Graph + input ---------------------------------------------------
    graph   = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # --- Pre-VPU: transpose (B, N) → (N, B) for horizontal→vertical reduction
    x_t = prepare_for_vpu_processing(graph, input_0, reduction_dim=-1)
    # x_t.output_type.shape == (N, B) == (16, 4)

    # --- Pre-VPU: triplicate (N, B) → (3*N, B) for three single-pass phases
    x_2t = graph.insert(
        ir.instructions.compiler.Concat(axis=-2),
        args=(x_t, x_t),
        output_type=ir.Type(dtype, (2 * N, B)),
    )
    x_3t = graph.insert(
        ir.instructions.compiler.Concat(axis=-2),
        args=(x_2t, x_t),
        output_type=ir.Type(dtype, (3 * N, B)),
    )

    # --- Type shorthands -------------------------------------------------
    reg16    = ir.Type(ir.Rfp16(eb),           (1,     B))  # register outside RepeatGraph
    reg21    = ir.Type(ir.Rfp21(eb),           (1,     B))  # Rfp21 accumulator register
    reg_sum  = ir.Type(ir.Rfp16(sum_eb),       (1,     B))  # sum cast register
    reg_rcp  = ir.Type(ir.Rfp16(recip_out_eb), (1,     B))  # reciprocal register

    lp_nm1   = ir.Type(dtype,                  (N - 1, B))  # inside RepeatGraph(N-1)
    lp_n     = ir.Type(dtype,                  (N,     B))  # inside RepeatGraph(N)
    lp21_n   = ir.Type(ir.Rfp21(eb),           (N,     B))  # Rfp21 inside RepeatGraph(N)

    # =====================================================================
    # Single VpuKernel – three sequential phases over x_3t
    # =====================================================================
    with ir.VpuKernel(parent=graph) as vpu_kernel:

        # ── Phase 1: running Max (rows 0 … N-1 of x_3t) ─────────────────
        # Seed with the first row instead of LoadZero so negative-valued
        # inputs are handled correctly.
        max_acc = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(x_3t,),
            output_type=reg16,
        )
        with ir.RepeatGraph(parent=vpu_kernel, times=N - 1) as rg1:
            row1    = rg1.insert(ir.instructions.pyxis.vpu.Load(),
                                 args=(x_3t,), output_type=lp_nm1)
            new_max = rg1.insert(ir.instructions.pyxis.vpu.Max(),
                                 args=(max_acc, row1), output_type=lp_nm1)
            max_acc = rg1.insert(ir.instructions.pyxis.vpu.Assign(),
                                 args=(max_acc, new_max), output_type=lp_nm1)
        # max_acc : (1, B) — per-batch max, broadcast to each row in later loops

        # ── Phase 2: sum_j exp(x_j − max) (rows N … 2N-1 of x_3t) ──────
        sum_acc = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=reg21)
        )
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as rg2:
            row2     = rg2.insert(ir.instructions.pyxis.vpu.Load(),
                                  args=(x_3t,), output_type=lp_n)
            shifted2 = rg2.insert(ir.instructions.pyxis.vpu.Sub(),
                                  args=(row2, max_acc), output_type=lp_n)
            exp2     = rg2.insert(ir.instructions.pyxis.vpu.Exp(output_eb=exp_eb),
                                  args=(shifted2,), output_type=lp_n)
            new_sum  = rg2.insert(ir.instructions.pyxis.vpu.ExtendedAdd(),
                                  args=(exp2, sum_acc), output_type=lp21_n)
            sum_acc  = rg2.insert(ir.instructions.pyxis.vpu.Assign(),
                                  args=(sum_acc, new_sum), output_type=lp21_n)

        # Cast Rfp21 → Rfp16(sum_eb) so it can feed the RECIP LUT
        sum_rfp16 = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Cast(output_dtype=ir.Rfp16(sum_eb)),
            args=(sum_acc,),
            output_type=reg_sum,
        )
        # Compute 1/sum via hardware RECIP LUT
        recip_sum = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.LUT(
                lut_kind=ir.instructions.pyxis.vpu.LUT.LutKind.RECIP,
                table=recip_table,
                output_eb=recip_out_eb,
            ),
            args=(sum_rfp16,),
            output_type=reg_rcp,
        )

        # ── Phase 3: exp(x − max) × (1/sum) (rows 2N … 3N-1 of x_3t) ───
        lp_result = ir.Type(ir.Rfp16(result_eb), (N, B))
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as rg3:
            row3     = rg3.insert(ir.instructions.pyxis.vpu.Load(),
                                  args=(x_3t,), output_type=lp_n)
            shifted3 = rg3.insert(ir.instructions.pyxis.vpu.Sub(),
                                  args=(row3, max_acc), output_type=lp_n)
            exp3     = rg3.insert(ir.instructions.pyxis.vpu.Exp(output_eb=exp_eb),
                                  args=(shifted3,), output_type=lp_n)
            result   = rg3.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=result_eb),
                args=(exp3, recip_sum),
                output_type=lp_result,
            )

    # --- Post-VPU: transpose (N, B) → (B, N) to recover original layout ---
    out_type = ir.Type(ir.Rfp16(result_eb), input_type.shape)
    restored = restore_from_vpu_processing(
        graph,
        result,
        pre_vpu_shape=input_type.shape,
        output_type=out_type,
        reduction_dim=-1,
    )

    graph.insert(ir.instructions.Output(), args=(restored,), output_type=out_type)
    return graph
