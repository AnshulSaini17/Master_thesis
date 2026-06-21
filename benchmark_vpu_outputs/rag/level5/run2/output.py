import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type: ir.Type(ir.Rfp16(eb), (N, n_cols))
    # We also need weight w: shape (1, n_cols), and eps_input: shape (1, n_cols)
    N = input_type.shape[0]
    n_cols = input_type.shape[-1]
    dtype = input_type.dtype
    eb = dtype.eb if hasattr(dtype, 'eb') else -15

    row_type     = ir.Type(ir.Rfp16(eb),  (1, n_cols))
    weight_type  = ir.Type(ir.Rfp16(eb),  (1, n_cols))
    # eps_type: a (1, n_cols) row holding broadcast eps value
    eps_type     = ir.Type(ir.Rfp16(eb),  (1, n_cols))

    graph = ir.RootGraph()
    # input_0: x of shape (N, n_cols)
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))
    # input_1: w of shape (1, n_cols)
    input_1 = graph.insert(ir.instructions.Input(output_type=weight_type))
    # input_2: eps broadcast row, shape (1, n_cols), all values = eps
    input_2 = graph.insert(ir.instructions.Input(output_type=eps_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # ---- Pass 1: compute absmax over N rows ----
        # Load each row, take abs, then reduce with Max
        abs_loads = []
        for i in range(N):
            load_i = vpu_kernel.insert(
                pyxis.vpu.Load(), args=(input_0,), output_type=row_type
            )
            abs_i = vpu_kernel.insert(
                pyxis.vpu.Absolute(), args=(load_i,), output_type=row_type
            )
            abs_loads.append(abs_i)

        # Reduce abs rows with Max
        absmax = abs_loads[0]
        for nxt in abs_loads[1:]:
            absmax = vpu_kernel.insert(
                pyxis.vpu.Max(), args=(absmax, nxt), output_type=row_type
            )
        # absmax: shape (1, n_cols) — element-wise max of abs(x) across rows

        # ---- absmax + eps ----
        load_eps = vpu_kernel.insert(
            pyxis.vpu.Load(), args=(input_2,), output_type=eps_type
        )
        denom = vpu_kernel.insert(
            pyxis.vpu.Add(), args=(absmax, load_eps), output_type=row_type
        )

        # ---- Pass 2: reload x rows, normalize, scale by w ----
        # Load w once (broadcast scalar)
        load_w = vpu_kernel.insert(
            pyxis.vpu.Load(), args=(input_1,), output_type=weight_type
        )

        # For each of the N rows: y_i = (x_i / denom) * w
        # Division: x / denom = x * (1/denom).
        # We approximate using Mul with right_eb to absorb the EB shift.
        # We reload x rows (a second pass through input_0 — requires N more pops).
        # NOTE: This requires the FIFO to be refilled; in practice the caller
        # feeds x twice, or the compiler re-streams. We model it as a second
        # sequence of Loads from input_0.
        #
        # Since VPU has no Reciprocal/Div op, we pre-compute reciprocal outside
        # and pass it in via input_3 (1/denom is data-dependent so caller must
        # provide it). HOWEVER to keep the graph self-contained and faithful,
        # we instead pass a fourth input: inv_absmax_eps of shape (1, n_cols).
        # The caller computes 1/(absmax+eps) on the host side.
        #
        # For a fully self-contained graph we accept this as a design constraint.
        pass

    # Because VPU lacks Div/Reciprocal, we need the caller to provide
    # inv_denom = 1/(absmax+eps).  Add that as input_3.
    inv_denom_type = ir.Type(ir.Rfp16(eb), (1, n_cols))
    input_3 = graph.insert(ir.instructions.Input(output_type=inv_denom_type))

    output_shape = input_type.shape  # (N, n_cols)
    output_type  = ir.Type(ir.Rfp16(eb), output_shape)

    with ir.VpuKernel(parent=graph) as vpu_kernel2:
        # Load inv_denom once (shape (1, n_cols))
        load_inv = vpu_kernel2.insert(
            pyxis.vpu.Load(), args=(input_3,), output_type=inv_denom_type
        )
        # Load w once
        load_w2 = vpu_kernel2.insert(
            pyxis.vpu.Load(), args=(input_1,), output_type=weight_type
        )

        # For N rows: y_i = x_i * inv_denom * w
        # Use RepeatGraph to iterate over N rows of x
        results = []
        with ir.RepeatGraph(parent=vpu_kernel2, times=N) as repeat:
            # Pop one row of x per iteration
            x_row = repeat.insert(
                pyxis.vpu.Load(), args=(input_0,), output_type=input_type
            )
            # x_i * inv_denom  (eb doubles: eb + eb = 2*eb; use output_eb=eb)
            scaled = repeat.insert(
                pyxis.vpu.Mul(output_eb=eb),
                args=(x_row, load_inv),
                output_type=ir.Type(ir.Rfp16(eb), output_shape),
            )
            # * w  (eb doubles again; use output_eb=eb)
            out_row = repeat.insert(
                pyxis.vpu.Mul(output_eb=eb),
                args=(scaled, load_w2),
                output_type=ir.Type(ir.Rfp16(eb), output_shape),
            )

    graph.insert(
        ir.instructions.Output(), args=(out_row,), output_type=output_type
    )
    return graph