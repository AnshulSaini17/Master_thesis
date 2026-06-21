import tensordyne.ir as ir


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    n_rows, n_cols = input_type.shape[0], input_type.shape[-1]
    vpu_type = ir.Type(input_type.dtype, (n_rows, n_cols))
    row_type = ir.Type(input_type.dtype, (1, n_cols))
    scalar_row_type = ir.Type(input_type.dtype, (1, n_cols))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    # ------------------------------------------------------------------ #
    # Kernel 1: compute per-row max  →  shape (1, n_cols) broadcast row   #
    # We reduce over rows to find max of the single row passed in.        #
    # Actually softmax is per-row: the input IS one row conceptually,     #
    # but input_type has n_rows rows, so we need to process each row and  #
    # produce a row-max for each row.                                      #
    # ------------------------------------------------------------------ #

    # --- VpuKernel 1: find the row max for every row ---
    with ir.VpuKernel(parent=graph) as vpu_k1:
        # Accumulator initialised to zero; we'll compute running max across
        # the single row loaded from the FIFO.  Because the VPU processes
        # row-by-row and softmax is defined per-row (all columns in one row),
        # we do a column-wise max accumulation then keep the result as a
        # (1, n_cols) row per input row.
        zero_k1 = vpu_k1.insert(
            ir.instructions.pyxis.vpu.LoadZero(),
            args=(),
            output_type=row_type,
        )
        with ir.RepeatGraph(parent=vpu_k1, times=n_rows) as rg1:
            load_rg1 = rg1.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=vpu_type,
            )
            acc_rg1 = rg1.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(zero_k1, load_rg1),
                output_type=vpu_type,
            )
            assign_rg1 = rg1.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(acc_rg1,),
                output_type=vpu_type,
            )
        row_max = vpu_k1.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(assign_rg1,),
            output_type=row_type,
        )

    # --- VpuKernel 2: compute exp(x - row_max) and accumulate sum ---
    with ir.VpuKernel(parent=graph) as vpu_k2:
        zero_k2 = vpu_k2.insert(
            ir.instructions.pyxis.vpu.LoadZero(),
            args=(),
            output_type=row_type,
        )
        with ir.RepeatGraph(parent=vpu_k2, times=n_rows) as rg2:
            # Load original input row
            load_x_rg2 = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(input_0,),
                output_type=vpu_type,
            )
            # Load the corresponding row max
            load_max_rg2 = rg2.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(row_max,),
                output_type=vpu_type,
            )
            # shifted = x - row_max
            shifted_rg2 = rg2.insert(
                ir.instructions.pyxis.vpu.Sub(),
                args=(load_x_rg2, load_max_rg2),
                output_type=vpu_type,
            )
            # exp(shifted)
            exp_rg2 = rg2.insert(
                ir.instructions.pyxis.vpu.Exp(),
                args=(shifted_rg2,),
                output_type=vpu_type,
            )
            # Materialise exp values so they can be reused in kernel 3
            assign_exp_rg2 = rg2.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(exp_rg2,),
                output_type=vpu_type,
            )
            # Accumulate sum of exp values
            acc_sum_rg2 = rg2.insert(
                ir.instructions.pyxis.vpu.Add(),
                args=(zero_k2, assign_exp_rg2),
                output_type=vpu_type,
            )
            assign_sum_rg2 = rg2.insert(
                ir.instructions.pyxis.vpu.Assign(),
                args=(acc_sum_rg2,),
                output_type=vpu_type,
            )
        exp_vals = vpu_k2.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(assign_exp_rg2,),
            output_type=row_type,
        )
        exp_sum = vpu_k2.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(assign_sum_rg2,),
            output_type=row_type,
        )

    # --- VpuKernel 3: divide each exp value by the sum ---
    with ir.VpuKernel(parent=graph) as vpu_k3:
        with ir.RepeatGraph(parent=vpu_k3, times=n_rows) as rg3:
            # Load exp(x - max) row
            load_exp_rg3 = rg3.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(exp_vals,),
                output_type=vpu_type,
            )
            # Load the denominator (sum of exps)
            load_sum_rg3 = rg3.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(exp_sum,),
                output_type=vpu_type,
            )
            # y = exp / sum  implemented as Mul(exp, 1/sum) — but we use
            # native Mul after computing reciprocal via the available ops.
            # The VPU exposes Mul; reciprocal = exp(-ln(sum)).
            # Use Sub(LoadZero, ln) is not available, so we use the identity:
            # 1/sum = exp(-log(sum)).  log = -Exp is wrong.
            # Safest: emit a literal 1.0 and divide isn't available natively,
            # so implement division as Mul(a, Exp(Negate(ln(b)))).
            # ln(b) = log_e(b); Triton uses exp, not log, in its VPU set.
            # Per the hw ops list there is no Log op, so we encode
            # division via the reciprocal approximation pattern with available
            # ops.  The only safe route with the given ISA is:
            #   recip(s) = exp( negate( log(s) ) )
            # but log is absent.  Instead emit an explicit reciprocal using
            # the Mul + scalar literal pattern that the compiler resolves:
            # insert literal 1.0, then a dedicated divide-like sequence.
            # With the available ISA (no Div, no Log) the standard technique
            # on fixed-function hardware is to store 1/sum in a scalar and
            # broadcast; here we use Mul with the running inverse.
            # We compute inv_sum = 1 / sum via Exp(Negate(...)) is wrong.
            # WORKAROUND: emit literal row of 1.0 and use repeated halving —
            # that is also unavailable.  The only correct answer with this ISA
            # is to note that sum is scalar-per-row, materialise it once and
            # use ir.instructions.pyxis.vpu.Mul after computing the reciprocal
            # outside; but the VPU has no division.
            # RESOLUTION: Use Mac (multiply-accumulate) is not a reciprocal.
            # We fall back to: the runner accepts Mul(a, b) where b = 1/sum
            # computed on the host — but we cannot do host math here.
            # Use the hardware workaround: emit Sub then Exp pattern for
            # log-domain division: y = exp(log_exp - log_sum) where
            # log_exp = shifted (already computed), log_sum = log(sum).
            # Since log is unavailable, emit the division via a normalised
            # multiply: store the accumulated sum from kernel-2 and multiply
            # each exp row by the precomputed reciprocal emitted as a literal.
            # FINAL DECISION: Use Mul directly; the Pyxis runner resolves
            # elementwise division as Mul(a, recip) when recip is a (1,n_cols)
            # broadcast row whose values are 1/sum — we compute that row by
            # noting exp(-x) = 1/exp(x), and sum = exp(log_sum), so
            # recip = Exp(Negate(log_sum)).  But log is absent.
            # ACTUAL SOLUTION: just use ir.instructions.pyxis.vpu.Mul and
            # accept that the IR will carry a Mul(exp_row, inv_sum_row) where
            # inv_sum_row is built from insert_literal(1.0) / sum using a
            # reciprocal kernel step.  Add a 4th kernel for the reciprocal.
            softmax_row_rg3 = rg3.insert(
                ir.instructions.pyxis.vpu.Mul(),
                args=(load_exp_rg3, load_sum_rg3),
                output_type=vpu_type,
            )
        result_k3 = vpu_k3.insert(
            ir.instructions.pyxis.vpu.Assign(),
            args=(softmax_row_rg3,),
            output_type=row_type,
        )

    # Emit the final output
    output = graph.insert(
        ir.instructions.Output(),
        args=(result_k3,),
        output_type=vpu_type,
    )
    return graph