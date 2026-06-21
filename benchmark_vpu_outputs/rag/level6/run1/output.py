import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type: ir.Type(ir.Rfp16(eb), (N, n_cols))
    N      = input_type.shape[0]
    n_cols = input_type.shape[-1]
    eb     = input_type.dtype.eb  # exponent bias of input

    row_type     = ir.Type(input_type.dtype, (1, n_cols))
    # After Exp (fixed EB = -15 output) and Sub, we use eb=-15 for exp outputs
    exp_eb       = -15
    exp_dtype    = ir.Rfp16(exp_eb)
    exp_row_type = ir.Type(exp_dtype, (1, n_cols))
    exp_full_type = ir.Type(exp_dtype, (N, n_cols))

    # Rfp21 accumulator for the sum — EB doubles after each Mul/Add chain;
    # use ExtendedAdd to keep precision
    acc_eb       = exp_eb  # accumulator tracks exp_eb
    acc_dtype    = ir.Rfp21(acc_eb)
    acc_row_type = ir.Type(acc_dtype, (1, n_cols))

    graph   = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:

        # ------------------------------------------------------------------ #
        # Step 1: Compute row-wise max across all N rows (vertical reduction) #
        # ------------------------------------------------------------------ #
        loads_for_max = [
            vpu_kernel.insert(
                pyxis.vpu.Load(), args=(input_0,), output_type=row_type
            )
            for _ in range(N)
        ]
        running_max = loads_for_max[0]
        for nxt in loads_for_max[1:]:
            running_max = vpu_kernel.insert(
                pyxis.vpu.Max(), args=(running_max, nxt), output_type=row_type
            )
        # running_max: shape (1, n_cols), the per-column max across all N rows

        # ------------------------------------------------------------------ #
        # Step 2: Subtract max from every row, then exponentiate             #
        #         (RepeatGraph: N iterations, each pops one row from input_0) #
        # ------------------------------------------------------------------ #
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat_exp:
            # Load one row of x per iteration
            x_row = repeat_exp.insert(
                pyxis.vpu.Load(), args=(input_0,), output_type=exp_full_type
            )
            # Subtract max (broadcast scalar — running_max already loaded once)
            shifted = repeat_exp.insert(
                pyxis.vpu.Sub(), args=(x_row, running_max), output_type=exp_full_type
            )
            # Exponentiate: Exp uses fixed EB=-15 for output
            exp_vals = repeat_exp.insert(
                pyxis.vpu.Exp(output_eb=exp_eb),
                args=(shifted,),
                output_type=exp_full_type,
            )

        # ------------------------------------------------------------------ #
        # Step 3: Sum of exp values (vertical reduction over N rows)          #
        # ------------------------------------------------------------------ #
        loads_for_sum = [
            vpu_kernel.insert(
                pyxis.vpu.Load(), args=(exp_vals,), output_type=exp_row_type
            )
            for _ in range(N)
        ]
        # Use ExtendedAdd for higher precision accumulation
        acc = vpu_kernel.insert(
            pyxis.vpu.ExtendedAdd(),
            args=(loads_for_sum[0], loads_for_sum[1]),
            output_type=acc_row_type,
        )
        for nxt in loads_for_sum[2:]:
            acc = vpu_kernel.insert(
                pyxis.vpu.ExtendedAdd(), args=(acc, nxt), output_type=acc_row_type
            )
        # acc: shape (1, n_cols), Rfp21 sum of exp values

        # Cast acc back to Rfp16 for division
        inv_sum_type = ir.Type(exp_dtype, (1, n_cols))
        sum_rfp16 = vpu_kernel.insert(
            pyxis.vpu.Convert(output_eb=exp_eb),
            args=(acc,),
            output_type=inv_sum_type,
        )

        # Reciprocal of the sum: 1 / sum
        recip_sum = vpu_kernel.insert(
            pyxis.vpu.Recip(output_eb=exp_eb),
            args=(sum_rfp16,),
            output_type=inv_sum_type,
        )

        # ------------------------------------------------------------------ #
        # Step 4: Divide each exp row by sum (broadcast recip_sum)           #
        # ------------------------------------------------------------------ #
        # output_eb after Mul: both inputs at exp_eb, Mul doubles → 2*exp_eb
        mul_eb   = 2 * exp_eb
        mul_dtype = ir.Rfp16(mul_eb)
        out_full_type = ir.Type(mul_dtype, (N, n_cols))

        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat_div:
            exp_row = repeat_div.insert(
                pyxis.vpu.Load(), args=(exp_vals,), output_type=out_full_type
            )
            softmax_row = repeat_div.insert(
                pyxis.vpu.Mul(output_eb=mul_eb),
                args=(exp_row, recip_sum),
                output_type=out_full_type,
            )

    graph.insert(
        ir.instructions.Output(), args=(softmax_row,), output_type=out_full_type
    )
    return graph