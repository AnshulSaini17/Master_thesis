
import tensordyne.ir as ir
from tensordyne import emu
from tensordyne.conversion.lowering.vpu import (
    prepare_for_vpu_processing,
    restore_from_vpu_processing,
)
from tensordyne.conversion.pipeline.passes.lut_values import get_lut_values


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Softmax along axis=-1 for input shape (N_rows, n_cols).

    Strategy (transpose → vertical-reduce → transpose back):
      1. Transpose (N_rows, n_cols) → (n_cols, N_rows)   [prepare_for_vpu_processing]
      2. K1: Max reduction over n_cols rows  → (1, N_rows) [per-input-row max]
      3. K2: exp(x_T - max)                 → (n_cols, N_rows)
      4. K3: Sum of exp via Rfp21 acc        → (1, N_rows)
      5. K4: exp * recip(sum)                → (n_cols, N_rows)
      6. Transpose (n_cols, N_rows) → (N_rows, n_cols)    [restore_from_vpu_processing]

    Justify 4 kernels:
      K1→K2 boundary: max must be fully reduced before exp-subtract can begin.
      K2→K3 boundary: exp values must be written to AMEM so K3 can sum them.
      K3→K4 boundary: sum must be fully reduced before normalization.
      K3 and K4 both read K2's exp output independently (fresh FIFO per kernel).
    """
    N_rows = input_type.shape[0]   # 4
    n_cols = input_type.shape[1]   # 16
    dtype  = input_type.dtype      # Rfp16(-15)
    eb     = -15                   # exponent bias

    # After transpose: n_cols becomes the row axis, N_rows becomes the col axis.
    T_rows = n_cols   # 16
    T_cols = N_rows   # 4

    T_type     = ir.Type(dtype, (T_rows, T_cols))   # (16, 4) — full unrolled (inside RepeatGraph)
    T_row_type = ir.Type(dtype, (1,     T_cols))    # (1,  4) — single register row

    # Rfp21 accumulator types for the sum kernel
    fp21_dtype    = ir.Rfp21(eb)
    acc_reg_type  = ir.Type(fp21_dtype, (1,     T_cols))   # register shape (outside loop)
    loop_acc_type = ir.Type(fp21_dtype, (T_rows, T_cols))  # full unrolled (inside loop)

    # LUT for reciprocal
    recip_eb    = emu.ops.vpu.recip_get_out_eb(eb)
    recip_table = get_lut_values(ir.instructions.Reciprocal)
    recip_type  = ir.Type(ir.Rfp16(recip_eb), (1, T_cols))

    # ------------------------------------------------------------------ graph
    graph   = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # Step 0 — Transpose (N_rows, n_cols) → (n_cols, N_rows)
    x_T = prepare_for_vpu_processing(graph, input_0, reduction_dim=-1)

    # ------------------------------------------------------------------ K1: max reduction
    # Unrolled: 16 Loads + 15 Max chains  →  (1, 4) row_max
    # All ops at VpuKernel level  →  use T_row_type = (1, 4)
    with ir.VpuKernel(parent=graph) as k1:
        ld0 = k1.insert(
            ir.instructions.pyxis.vpu.Load(), args=(x_T,), output_type=T_row_type
        )
        max_acc = ld0
        for _ in range(T_rows - 1):
            ld_n = k1.insert(
                ir.instructions.pyxis.vpu.Load(), args=(x_T,), output_type=T_row_type
            )
            max_acc = k1.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(max_acc, ld_n),
                output_type=T_row_type,
            )
    row_max = max_acc   # (1, 4) Rfp16(-15)

    # ------------------------------------------------------------------ K2: exp(x_T - max)
    # Load max once (outside loop), then RepeatGraph(16) produces (16, 4) exp tensor.
    with ir.VpuKernel(parent=graph) as k2:
        load_max = k2.insert(
            ir.instructions.pyxis.vpu.Load(), args=(row_max,), output_type=T_row_type
        )
        with ir.RepeatGraph(parent=k2, times=T_rows) as rg2:
            x_row   = rg2.insert(
                ir.instructions.pyxis.vpu.Load(), args=(x_T,), output_type=T_type
            )
            shifted = rg2.insert(
                ir.instructions.pyxis.vpu.Sub(),
                args=(x_row, load_max),
                output_type=T_type,
            )
            exp_row = rg2.insert(
                ir.instructions.pyxis.vpu.Exp(),
                args=(shifted,),
                output_type=T_type,
            )
    exp_vals = exp_row   # (16, 4) Rfp16(-15)

    # ------------------------------------------------------------------ K3: sum of exp (Rfp21 acc)
    # Rfp21 accumulator; Cast back to Rfp16(-15) after loop.
    with ir.VpuKernel(parent=graph) as k3:
        sum_reg = k3.insert(
            ir.instructions.pyxis.vpu.LoadZero(output_type=acc_reg_type)
        )
        with ir.RepeatGraph(parent=k3, times=T_rows) as rg3:
            e_load  = rg3.insert(
                ir.instructions.pyxis.vpu.Load(), args=(exp_vals,), output_type=T_type
            )
            new_sum = rg3.insert(
                ir.instructions.pyxis.vpu.ExtendedAdd(),
                args=(e_load, sum_reg),
                output_type=loop_acc_type,
            )
            next_acc = rg3.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(sum_reg, new_sum),
                output_type=loop_acc_type,
            )
        sum_rfp16 = k3.insert(
            ir.instructions.pyxis.vpu.Cast(output_dtype=dtype),
            args=(next_acc,),
            output_type=T_row_type,
        )
    sum_exp = sum_rfp16   # (1, 4) Rfp16(-15)

    # ------------------------------------------------------------------ K4: normalize
    # recip(sum_exp) once, then multiply each exp row.
    with ir.VpuKernel(parent=graph) as k4:
        load_sum = k4.insert(
            ir.instructions.pyxis.vpu.Load(), args=(sum_exp,), output_type=T_row_type
        )
        recip = k4.insert(
            ir.instructions.pyxis.vpu.LUT(
                lut_kind=ir.instructions.pyxis.vpu.LUT.LutKind.RECIP,
                table=recip_table,
                output_eb=recip_eb,
            ),
            args=(load_sum,),
            output_type=recip_type,
        )
        with ir.RepeatGraph(parent=k4, times=T_rows) as rg4:
            e_load = rg4.insert(
                ir.instructions.pyxis.vpu.Load(), args=(exp_vals,), output_type=T_type
            )
            normed = rg4.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(e_load, recip),
                output_type=T_type,
            )
    softmax_T = normed   # (16, 4) Rfp16(-15)

    # Step 6 — Transpose back (n_cols, N_rows) → (N_rows, n_cols)
    result = restore_from_vpu_processing(
        graph, softmax_T,
        pre_vpu_shape=input_type.shape,
        output_type=input_type,
        reduction_dim=-1,
    )

    graph.insert(ir.instructions.Output(), args=(result,), output_type=input_type)
    return graph
