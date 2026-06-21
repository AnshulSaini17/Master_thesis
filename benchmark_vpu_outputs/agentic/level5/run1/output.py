
import tensordyne.ir as ir
from tensordyne import emu
from tensordyne.conversion.lowering.vpu import (
    prepare_for_vpu_processing,
    restore_from_vpu_processing,
)
from tensordyne.conversion.pipeline.passes.lut_values import get_lut_values


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """Absmax normalisation: y = (x / (max|x| + eps)) * w

    Maps the Triton absmax_norm_kernel to three sequential VpuKernels:
      K1 – horizontal max-abs reduction per row  (prepare+transpose trick)
      K2 – reciprocal LUT × normalise            (still in transposed domain)
      K3 – multiply by weight w (= ones)         (back in original domain)

    The horizontal reduction (max across n_cols elements inside one VPU row)
    is made vertical via prepare_for_vpu_processing(reduction_dim=-1):
        (N, C) ──prepare──> (C, N)
    Then a standard LoadZero + RepeatGraph(C) + Max + Assign accumulator
    reduces the C VPU rows of width N → one row of N per-row absmax values.
    restore_from_vpu_processing inverts the transpose for the output.
    """
    eb     = input_type.dtype.eb     # exponent bias (e.g. -15 for Rfp16(-15))
    n_rows = input_type.shape[0]     # N  (e.g. 4)
    n_cols = input_type.shape[1]     # C  (e.g. 64)
    dtype  = input_type.dtype        # Rfp16(eb)

    # Output EB for the reciprocal LUT
    eb_recip = emu.ops.vpu.recip_get_out_eb(eb)

    # ── Root graph ───────────────────────────────────────────────────────────
    graph   = ir.RootGraph()
    # Single forward input: x of shape (N, C).
    # The module weight is a Parameter initialised to ones — embedded as a
    # compile-time constant in Kernel 3.
    input_x = graph.insert(ir.instructions.Input(output_type=input_type))

    # ── Transpose x for horizontal max reduction: (N, C) → (C, N) ──────────
    x_prep      = prepare_for_vpu_processing(graph, input_x, reduction_dim=-1)
    prep_shape  = x_prep.output_type.shape   # (C, N)
    n_prep_rows = prep_shape[0]              # C  (= n_cols)
    n_prep_cols = prep_shape[1]              # N  (= n_rows)

    prep_row_type  = ir.Type(dtype, (1,           n_prep_cols))  # (1, N)
    prep_full_type = ir.Type(dtype, (n_prep_rows, n_prep_cols))  # (C, N)

    # ── Kernel 1: max(abs(x)) per original row ───────────────────────────────
    # Each VPU row of x_prep is (1, N): column i carries one element from
    # original row i.  We accumulate element-wise Max across C VPU rows →
    # resulting (1, N) register holds absmax[i] in position i.
    with ir.VpuKernel(parent=graph) as k1:
        acc = k1.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=prep_row_type)
        )
        with ir.RepeatGraph(parent=k1, times=n_prep_rows) as rg1:
            x_row = rg1.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_prep,),
                output_type=prep_full_type,   # full unrolled (C, N)
            )
            abs_row = rg1.insert(
                ir.instructions.pyxis.vpu.Absolute(),
                args=(x_row,),
                output_type=prep_full_type,
            )
            new_acc = rg1.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc, abs_row),
                output_type=prep_full_type,
            )
            acc = rg1.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(acc, new_acc),
                output_type=prep_full_type,
            )
        # Cast the RepeatGraph-typed accumulator back to register shape (1, N)
        absmax_out = k1.insert(
            ir.instructions.pyxis.vpu.Cast(output_dtype=dtype),
            args=(acc,),
            output_type=prep_row_type,
        )
    # absmax_out: shape (1, N), dtype Rfp16(eb)

    # ── Kernel 2: recip(absmax + eps) × x  (still in transposed domain) ─────
    recip_row_type = ir.Type(ir.Rfp16(eb_recip), (1, n_prep_cols))  # (1, N)

    recip_table = get_lut_values(
        ir.instructions.Reciprocal,
        input_dtype=dtype,
        output_dtype=ir.Rfp16(eb_recip),
    )

    with ir.VpuKernel(parent=graph) as k2:
        # Load absmax (1, N) — one FIFO pop, stays in register
        absmax_loaded = k2.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(absmax_out,),
            output_type=prep_row_type,
        )
        # eps = 1e-6 embedded as a (1, N) constant register
        eps_lit = ir.TensorLiteral.from_values(
            dtype=dtype,
            values=[1e-6] * n_prep_cols,
            is_raw=False,
        )
        eps_reg = k2.insert(
            ir.instructions.pyxis.vpu.Scalar(value=eps_lit),
            output_type=prep_row_type,
        )
        absmax_eps = k2.insert(
            ir.instructions.pyxis.vpu.Add(),
            args=(absmax_loaded, eps_reg),
            output_type=prep_row_type,
        )
        # recip(absmax + eps) via hardware LUT
        recip = k2.insert(
            ir.instructions.pyxis.vpu.LUT(
                lut_kind=ir.instructions.pyxis.vpu.LUT.LutKind.RECIP,
                table=recip_table,
                output_eb=eb_recip,
            ),
            args=(absmax_eps,),
            output_type=recip_row_type,   # (1, N)
        )
        # Normalise: load each of the C rows of x_prep and multiply by recip.
        # recip stays in register (loaded once outside RepeatGraph).
        with ir.RepeatGraph(parent=k2, times=n_prep_rows) as rg2:
            x_row2 = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_prep,),
                output_type=prep_full_type,   # (C, N) unrolled
            )
            x_norm = rg2.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(x_row2, recip),
                output_type=prep_full_type,
            )
    # x_norm: (C, N) in transposed domain

    # ── Restore to original domain: (C, N) → (N, C) ─────────────────────────
    out_full_type = ir.Type(dtype, input_type.shape)   # (N, C)
    x_norm_full = restore_from_vpu_processing(
        graph,
        x_norm,
        pre_vpu_shape=input_type.shape,
        output_type=out_full_type,
        reduction_dim=-1,
    )

    # ── Kernel 3: multiply by weight w (= ones, nn.Parameter) ───────────────
    row_type  = ir.Type(dtype, (1,     n_cols))   # (1, C)
    full_type = ir.Type(dtype, (n_rows, n_cols))  # (N, C)

    # The reference module initialises self.weight = torch.ones(C).
    # Embed as a compile-time Scalar constant so no extra graph Input is needed.
    w_lit = ir.TensorLiteral.from_values(
        dtype=dtype,
        values=[1.0] * n_cols,
        is_raw=False,
    )

    with ir.VpuKernel(parent=graph) as k3:
        # Load w once — it is reused across all N row iterations.
        w_reg = k3.insert(
            ir.instructions.pyxis.vpu.Scalar(value=w_lit),
            output_type=row_type,            # (1, C)
        )
        with ir.RepeatGraph(parent=k3, times=n_rows) as rg3:
            norm_row = rg3.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_norm_full,),
                output_type=full_type,       # (N, C) unrolled
            )
            out_row = rg3.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(norm_row, w_reg),
                output_type=full_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(out_row,),
        output_type=full_type,
    )
    return graph
