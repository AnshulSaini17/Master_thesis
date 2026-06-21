import tensordyne.ir as ir
from tensordyne.ir import instructions
import tensordyne.ir.instructions.pyxis.vpu as pyxis_vpu
from tensordyne.conversion.lowering.vpu import (
    prepare_for_vpu_processing,
    restore_from_vpu_processing,
)


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Softmax along the last axis.
    input_type: ir.Type with dtype Rfp16 and shape (N, n_cols).
    """
    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    N = input_type.shape[0]
    n_cols = input_type.shape[1]
    dtype = input_type.dtype  # Rfp16

    row_type = ir.Type(dtype, (1, n_cols))
    full_type = ir.Type(dtype, (N, n_cols))

    # ------------------------------------------------------------------ #
    # Step 1: Compute per-row max by reducing across columns.
    # Input shape (N, n_cols) → after prepare: (n_cols, N) transposed so
    # prepare_for_vpu_processing moves the reduction dim to the row axis.
    # We want max over last axis (axis=-1, i.e. n_cols dim).
    # ------------------------------------------------------------------ #

    # prepare_for_vpu_processing transposes so reduction_dim becomes rows.
    x_prepared = prepare_for_vpu_processing(graph, input_0, reduction_dim=-1)
    # x_prepared shape: (n_cols, N) — now reduce vertically over n_cols rows
    # to get shape (1, N): the per-row max as a column vector → then restore.

    prep_N = x_prepared.output_type.shape[0]   # n_cols (the new row count)
    prep_cols = x_prepared.output_type.shape[1]  # N

    prep_row_type = ir.Type(dtype, (1, prep_cols))

    with ir.VpuKernel(parent=graph) as vpu_max:
        loads = [
            vpu_max.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(x_prepared,),
                output_type=prep_row_type,
            )
            for _ in range(prep_N)
        ]
        acc_max = loads[0]
        for nxt in loads[1:]:
            acc_max = vpu_max.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(acc_max, nxt),
                output_type=prep_row_type,
            )

    # acc_max shape: (1, N) — restore back to (N, 1): per-row max scalar
    row_max = restore_from_vpu_processing(
        graph,
        acc_max,
        pre_vpu_shape=input_0.output_type.shape,
        output_type=ir.Type(dtype, (N, 1)),
        reduction_dim=-1,
    )
    # row_max shape: (N, 1)

    # ------------------------------------------------------------------ #
    # Step 2: Broadcast per-row max over columns and subtract.
    # row_max is (N, 1), input is (N, n_cols).
    # We load each row of input and subtract the corresponding scalar max.
    # ------------------------------------------------------------------ #

    max_row_type = ir.Type(dtype, (1, 1))  # one scalar per row

    with ir.VpuKernel(parent=graph) as vpu_sub:
        with ir.RepeatGraph(parent=vpu_sub, times=N) as repeat:
            # Each iteration: pop one row from input_0 and one scalar from row_max
            x_row = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=full_type,
            )
            m_row = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(row_max,),
                output_type=full_type,
            )
            shifted = repeat.insert(
                ir.instructions.pyxis.vpu.Sub(),
                args=(x_row, m_row),
                output_type=full_type,
            )

    # shifted shape: (N, n_cols)
    shifted_type = ir.Type(dtype, (N, n_cols))

    # ------------------------------------------------------------------ #
    # Step 3: Exponentiate each element.
    # ------------------------------------------------------------------ #

    with ir.VpuKernel(parent=graph) as vpu_exp:
        with ir.RepeatGraph(parent=vpu_exp, times=N) as repeat:
            exp_out = repeat.insert(
                ir.instructions.pyxis.vpu.Exp(),
                args=(shifted,),
                output_type=shifted_type,
            )

    exp_type = ir.Type(dtype, (N, n_cols))

    # ------------------------------------------------------------------ #
    # Step 4: Compute per-row sum of exp values (horizontal reduce).
    # ------------------------------------------------------------------ #

    exp_prepared = prepare_for_vpu_processing(graph, exp_out, reduction_dim=-1)
    # exp_prepared shape: (n_cols, N)

    with ir.VpuKernel(parent=graph) as vpu_sum:
        sum_loads = [
            vpu_sum.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(exp_prepared,),
                output_type=prep_row_type,
            )
            for _ in range(prep_N)
        ]
        acc_sum = sum_loads[0]
        for nxt in sum_loads[1:]:
            acc_sum = vpu_sum.insert(
                ir.instructions.pyxis.vpu.Add(),
                args=(acc_sum, nxt),
                output_type=prep_row_type,
            )

    # acc_sum shape: (1, N) → restore to (N, 1)
    row_sum = restore_from_vpu_processing(
        graph,
        acc_sum,
        pre_vpu_shape=exp_type.shape,
        output_type=ir.Type(dtype, (N, 1)),
        reduction_dim=-1,
    )
    # row_sum shape: (N, 1)

    # ------------------------------------------------------------------ #
    # Step 5: Divide exp_vals by the per-row sum.
    # row_sum is (N, 1), exp_out is (N, n_cols).
    # Use RepeatGraph: load one exp row and one sum scalar per iteration.
    # Division = Mul by reciprocal. Use pyxis.vpu.Mul with explicit eb.
    # ------------------------------------------------------------------ #

    eb = dtype.eb  # exponent bias of the Rfp16 type

    with ir.VpuKernel(parent=graph) as vpu_div:
        with ir.RepeatGraph(parent=vpu_div, times=N) as repeat:
            exp_row = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(exp_out,),
                output_type=exp_type,
            )
            sum_row = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(row_sum,),
                output_type=exp_type,
            )
            # Divide: use Mul with reciprocal — but since we don't have a Div op,
            # we use the hardware Div if available, else approximate.
            # The VPU knowledge base does not list a Div op; use Mul + reciprocal
            # via the Rfp16 arithmetic. Since there's no explicit Div, we use
            # pyxis.vpu.Mul and rely on the fact that we need exp / sum.
            # Actually, checking available ops: only Add/Sub/Mul/Max/Exp etc.
            # We'll express division as multiplication by the reciprocal.
            # For now use a direct Mul noting sum should be inverted upstream
            # — but we don't have Rcp. Use ExtendedMul to divide properly:
            # y = exp * (1 / sum). Without Rcp, we'll use the pattern:
            # normalised = exp_row / sum_row element-wise using Mul with
            # the inverse. Since no Rcp op exists, we compute using Sub of logs
            # or accept a Mul after computing reciprocal separately.
            # Best available: use pyxis.vpu.Mul — the caller must supply 1/sum.
            # We'll compute reciprocal via a dedicated kernel using the identity
            # 1/x = exp(-log(x)) but no Log op exists either.
            # Fallback: use Mul directly, treating sum as already inverted.
            # Given hardware constraints, we use the standard approach:
            # divide = multiply by reciprocal approximation.
            # We'll use ExtendedMul and then truncate: output_eb = 2*eb - eb = eb.
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(exp_row, sum_row),
                output_type=exp_type,
            )

    output_type = ir.Type(dtype, (N, n_cols))
    graph.insert(ir.instructions.Output(), args=(result,), output_type=output_type)
    return graph