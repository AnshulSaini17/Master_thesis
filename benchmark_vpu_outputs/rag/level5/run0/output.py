import tensordyne.ir as ir
from tensordyne.ir.instructions import pyxis
from tensordyne.conversion.lowering.vpu import prepare_for_vpu_processing, restore_from_vpu_processing

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    dtype = input_type.dtype
    shape = input_type.shape        # (1, n_cols)
    n_cols = shape[-1]
    eb = dtype.eb if hasattr(dtype, 'eb') else -15

    row_type    = ir.Type(dtype, (1, n_cols))
    scalar_type = ir.Type(dtype, (1, 1))

    graph = ir.RootGraph()
    input_x   = graph.insert(ir.instructions.Input(output_type=input_type))
    input_w   = graph.insert(ir.instructions.Input(output_type=input_type))
    input_eps = graph.insert(ir.instructions.Input(output_type=scalar_type))

    # Transpose x to (n_cols, 1) so vertical reduction = horizontal absmax
    x_T = prepare_for_vpu_processing(graph, input_x, reduction_dim=-1)
    prepared_type    = ir.Type(dtype, (n_cols, 1))
    prepared_row_t   = ir.Type(dtype, (1, 1))

    # Pass 1: absmax over the row  →  scalar (1,1)
    with ir.VpuKernel(parent=graph) as vk1:
        abs_vals = []
        for _ in range(n_cols):
            ld = vk1.insert(ir.instructions.pyxis.vpu.Load(),
                            args=(x_T,), output_type=prepared_row_t)
            ab = vk1.insert(ir.instructions.pyxis.vpu.Absolute(),
                            args=(ld,), output_type=prepared_row_t)
            abs_vals.append(ab)
        mx = abs_vals[0]
        for nxt in abs_vals[1:]:
            mx = vk1.insert(ir.instructions.pyxis.vpu.Max(),
                            args=(mx, nxt), output_type=prepared_row_t)
    # mx : (1,1) absmax scalar

    # Pass 2: y = x * (1/(absmax+eps)) * w  element-wise over (1, n_cols)
    with ir.VpuKernel(parent=graph) as vk2:
        ld_mx  = vk2.insert(ir.instructions.pyxis.vpu.Load(),
                            args=(mx,), output_type=scalar_type)
        ld_eps = vk2.insert(ir.instructions.pyxis.vpu.Load(),
                            args=(input_eps,), output_type=scalar_type)
        denom  = vk2.insert(ir.instructions.pyxis.vpu.Add(),
                            args=(ld_mx, ld_eps), output_type=scalar_type)
        recip  = vk2.insert(ir.instructions.pyxis.vpu.Reciprocal(),
                            args=(denom,), output_type=scalar_type)
        ld_x   = vk2.insert(ir.instructions.pyxis.vpu.Load(),
                            args=(input_x,), output_type=row_type)
        ld_w   = vk2.insert(ir.instructions.pyxis.vpu.Load(),
                            args=(input_w,), output_type=row_type)
        xn     = vk2.insert(ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                            args=(ld_x, recip), output_type=row_type)
        result = vk2.insert(ir.instructions.pyxis.vpu.Mul(output_eb=eb),
                            args=(xn, ld_w), output_type=row_type)

    graph.insert(ir.instructions.Output(), args=(result,), output_type=row_type)
    return graph