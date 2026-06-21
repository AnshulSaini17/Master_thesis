import tensordyne.ir as ir
import tensordyne.ir.instructions.pyxis.vpu as pyxis_vpu


def build_graph(input_type: ir.Type) -> ir.RootGraph:
    # input_type is e.g. ir.Type(ir.Rfp16(eb), (N, n_cols))
    dtype = input_type.dtype
    shape = input_type.shape          # (N, n_cols)
    N = shape[0]
    n_cols = shape[-1]

    row_type = ir.Type(dtype, (1, n_cols))  # single-row register shape

    graph = ir.RootGraph()

    # Main input: the tensor x to clip, shape (N, n_cols)
    input_x = graph.insert(ir.instructions.Input(output_type=input_type))

    # lo and hi are provided as constant single-row tensors (1, n_cols),
    # broadcast across all rows.
    lo_type = ir.Type(dtype, (1, n_cols))
    hi_type = ir.Type(dtype, (1, n_cols))
    input_lo = graph.insert(ir.instructions.Input(output_type=lo_type))
    input_hi = graph.insert(ir.instructions.Input(output_type=hi_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # Load the scalar lo and hi rows ONCE outside the loop — one FIFO pop each.
        load_lo = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(input_lo,),
            output_type=row_type,
        )
        load_hi = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(input_hi,),
            output_type=row_type,
        )

        # Iterate over the N rows of x.
        with ir.RepeatGraph(parent=vpu_kernel, times=N) as repeat:
            # clip(x, lo, hi) = max(min(x, hi), lo)
            # Step 1: tmp = min(x_row, hi)  — pops one row of input_x per iteration
            tmp = repeat.insert(
                ir.instructions.pyxis.vpu.Min(),
                args=(input_x, load_hi),
                output_type=input_type,  # full unrolled shape inside RepeatGraph
            )
            # Step 2: out = max(tmp, lo)
            out = repeat.insert(
                ir.instructions.pyxis.vpu.Max(),
                args=(tmp, load_lo),
                output_type=input_type,  # full unrolled shape inside RepeatGraph
            )

    graph.insert(ir.instructions.Output(), args=(out,), output_type=input_type)
    return graph