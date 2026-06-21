import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    """
    Implements absmax-norm + weight scaling per row:
        y = x / (absmax(|x|) + eps) * w

    Because VPU has no Reciprocal instruction, the graph takes three inputs:
        input_0 : x          – (N, n_cols) Rfp16
        input_1 : w          – (1, n_cols) Rfp16  (broadcast weight row)
        input_2 : inv_scale  – (1, 1)      Rfp16  = 1 / (absmax(|x|) + eps),
                                                    pre-computed by the host or a
                                                    preceding reduction kernel.

    The VpuKernel fuses the normalization + weight multiply in a single pass:
        for each row r of x:
            y[r] = x[r] * inv_scale * w
    """
    N      = input_type.shape[0]
    n_cols = input_type.shape[-1]
    eb     = input_type.dtype.eb if hasattr(input_type.dtype, 'eb') else -15
    dtype  = ir.Rfp16(eb)

    full_type    = ir.Type(dtype, (N, n_cols))   # x / y shape
    row_type     = ir.Type(dtype, (1, n_cols))   # w shape  (one row)
    scalar_type  = ir.Type(dtype, (1, 1))        # inv_scale shape

    graph = ir.RootGraph()
    inp_x         = graph.insert(ir.instructions.Input(output_type=full_type))
    inp_w         = graph.insert(ir.instructions.Input(output_type=row_type))
    inp_inv_scale = graph.insert(ir.instructions.Input(output_type=scalar_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # ── load broadcast values ONCE, outside the row loop ──────────────
        inv_s  = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(inp_inv_scale,),
            output_type=scalar_type,
        )
        load_w = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(inp_w,),
            output_type=row_type,
        )

        # ── iterate over every row of x ────────────────────────────────────
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # Each iteration pops the next row from x's FIFO.
            x_row = repeat.insert(
                ir.instructions.pyxis.vpu.Load(),
                args=(inp_x,),
                output_type=full_type,   # RepeatGraph body uses FULL shape
            )

            # x_row * inv_scale  (shape-broadcast: (N,n_cols) × (1,1))
            # Mul doubles EB: output_eb pins result back to `eb`.
            normed = repeat.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(x_row, inv_s),
                output_type=full_type,
            )

            # normed * w  (shape-broadcast: (N,n_cols) × (1,n_cols))
            result = repeat.insert(
                ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                args=(normed, load_w),
                output_type=full_type,
            )

    graph.insert(
        ir.instructions.Output(),
        args=(result,),
        output_type=full_type,
    )
    return graph