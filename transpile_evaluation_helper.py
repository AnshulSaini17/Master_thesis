#!/usr/bin/env python3
"""Helper functions to integrate transpiled models with evaluation.py"""

import inspect
import re

import torch

import tensordyne.ir as ir
import tensordyne.nn as nn


def _instantiate(cls, verbose: bool, label: str, config_data: dict | None = None):
    """Instantiate cls from config_data.

    HuggingFace models: config_data contains '_hf_config' (a real PreTrainedConfig).
    Tensordyne models:  config_data is a plain dict matched against __init__ params.
    """
    if config_data is not None and "_hf_config" in config_data:
        instance = cls(config_data["_hf_config"])
        if verbose:
            print(f"✅ Found {label} model: {cls.__name__}(config=<PreTrainedConfig>)")
        return instance

    # Common default values — used as fallback when no config file is given
    # and the parameter has no default in __init__.
    _KNOWN_ARGS = {
        "embed_dim": 64,
        "hidden_size": 64,
        "d_model": 64,
        "num_heads": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_layers": 2,
        "vocab_size": 1000,
        "seq_len": 10,
        "max_position_embeddings": 128,
        "head_dim": 16,
        "dropout": 0.0,
        "eps": 1e-6,
        "rms_norm_eps": 1e-6,
        "in_features": 64,
        "out_features": 64,
        "bias": True,
        "hidden_dim": 256,
    }

    sig = inspect.signature(cls.__init__)
    accepted = sig.parameters
    # Build kwargs: config_data takes priority, then _KNOWN_ARGS for any
    # remaining required parameters that have no default value.
    merged = {**_KNOWN_ARGS, **(config_data or {})}
    kwargs = {k: merged[k] for k in accepted if k != "self" and k in merged}
    instance = cls(**kwargs)
    if verbose:
        kw_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"✅ Found {label} model: {cls.__name__}({kw_str})")
    return instance


def run_transpiled_evaluation(
    input_file: str,
    output_file: str,
    input_shape: tuple,
    test_name: str | None = None,
    verbose: bool = True,
    config_file: str = "",
    target_class: str = "",
):
    """Run evaluation comparing original PyTorch and transpiled Tensordyne.

    Args:
        input_file: Path to original PyTorch model (.py file)
        output_file: Path to transpiled Tensordyne model (.py file)
        input_shape: Shape of test input (e.g., (1, 64))
        test_name: Optional test name
        verbose: Print detailed results

    Returns
    -------
        TestResult object from evaluation.py
    """
    from evaluation import print_results, run_td_test

    if verbose:
        print("\n" + "=" * 80)
        print("🧪 PREPARING EVALUATION")
        print("=" * 80)

    # Load config_data once — used for both PyTorch and Tensordyne instantiation.
    config_data: dict | None = None
    if config_file:
        import json

        with open(config_file) as f:
            config_data = json.load(f)
        if verbose:
            print(f"✅ Loaded config from {config_file}: {config_data}")

    # Just exec the files to get the model classes - simple!
    # PyTorch model — inject __name__ so HuggingFace-style files that check
    # `if __name__ == "__main__"` or reference __name__ at module level don't crash.
    torch_globals = {"__name__": "__main__", "__builtins__": __builtins__}
    with open(input_file) as f:
        source = f.read()

    # Check for relative imports (e.g. `from ...activations import ...`).
    # These cannot be resolved by bare exec() — fall back to importing the
    # target class from the installed `transformers` package instead.
    # Also treat auto-generated transformers files (no relative imports but still
    # require PreTrainedConfig) the same way — detect by path.
    _has_relative_imports = bool(re.search(r"^from \.[.\w]", source, re.MULTILINE))
    _is_hf_file = _has_relative_imports or "site-packages/transformers" in input_file.replace("\\", "/")
    if _has_relative_imports:
        try:
            import transformers as _tf

            if target_class:
                torch_globals[target_class] = getattr(_tf, target_class)
            else:
                for _name in dir(_tf):
                    _obj = getattr(_tf, _name)
                    if isinstance(_obj, type) and issubclass(_obj, torch.nn.Module):
                        torch_globals[_name] = _obj
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"{input_file} uses relative imports and cannot be exec'd standalone. "
                f"Tried falling back to `transformers` but failed: {e}. "
                "Provide a standalone file without relative imports."
            ) from e
    else:
        exec(source, torch_globals)

    # For HuggingFace models imported via transformers, config_data must be
    # passed as a PreTrainedConfig
    # Build a separate torch_config_data so Tensordyne still gets the plain dict.
    torch_config_data = config_data
    if _is_hf_file and config_data is not None:
        from pathlib import Path as _Path

        _model_name = _Path(input_file).stem  # e.g. "modeling_qwen3_next" → look for Qwen3NextConfig
        _snake = _model_name.replace("modeling_", "")
        # Convert snake_case to CamelCase: "qwen3_next" → "Qwen3Next"
        _camel = "".join(w.capitalize() for w in _snake.split("_"))
        _config_cls_name = f"{_camel}Config"
        _config_cls = getattr(__import__("transformers"), _config_cls_name, None)
        _init_data = {k: v for k, v in config_data.items() if k != "model_type"}
        if _config_cls is not None:
            torch_config_data = {"_hf_config": _config_cls(**_init_data)}
        elif "model_type" in config_data:
            # Fallback: AutoConfig.for_model handles names CamelCase mangles (e.g. gpt2 → GPT2Config)
            from transformers import AutoConfig as _AutoConfig

            torch_config_data = {"_hf_config": _AutoConfig.for_model(config_data["model_type"], **_init_data)}

    # Find the model class.
    # If target_class is given, use that class by name.
    # Otherwise pick the LAST nn.Module subclass (leaf modules are defined first).
    torch_model = None
    if target_class and target_class in torch_globals:
        torch_model = _instantiate(torch_globals[target_class], verbose, "PyTorch", torch_config_data)
    else:
        for obj in torch_globals.values():
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj != torch.nn.Module:
                torch_model = _instantiate(obj, verbose, "PyTorch", torch_config_data)

    if torch_model is None:
        raise ValueError(f"No PyTorch model found in {input_file}")

    # Tensordyne model
    td_globals = {}
    with open(output_file) as f:
        exec(f.read(), td_globals)

    td_model = None
    if target_class and target_class in td_globals:
        td_model = _instantiate(td_globals[target_class], verbose, "Tensordyne", torch_config_data)
    else:
        for obj in td_globals.values():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
                td_model = _instantiate(obj, verbose, "Tensordyne", torch_config_data)

    if td_model is None:
        raise ValueError(f"No Tensordyne model found in {output_file}")

    # Decide input dtype by inspecting the model rather than relying on exception messages.
    # If the model's primary input is an embedding lookup (e.g. Llama, GPT, BERT — anything
    # whose entry point is nn.Embedding), the first forward arg must be integer token IDs.
    # Otherwise we feed a float tensor (the model takes activations directly).
    def _expects_int_input(model: torch.nn.Module) -> bool:
        # HuggingFace convention: get_input_embeddings() returns the entry embedding.
        if hasattr(model, "get_input_embeddings"):
            try:
                emb = model.get_input_embeddings()
                if isinstance(emb, torch.nn.Embedding):
                    return True
            except Exception:
                pass
        # Generic fallback: any nn.Embedding inside the model implies token-id input.
        for m in model.modules():
            if isinstance(m, torch.nn.Embedding):
                return True
        return False

    _use_int_input = _expects_int_input(torch_model)
    if _use_int_input:
        concrete_inputs = (torch.randint(0, 1000, input_shape),)
    else:
        concrete_inputs = (torch.randn(*input_shape),)

    # Some HuggingFace models accept use_cache; pass False to avoid empty-cache crashes.
    _call_kwargs: dict = {}
    import inspect as _inspect

    _fwd_sig = _inspect.signature(torch_model.forward)
    if "use_cache" in _fwd_sig.parameters:
        _call_kwargs["use_cache"] = False
    torch_model.eval()
    with torch.no_grad():
        expected_output = torch_model(*concrete_inputs, **_call_kwargs)

    # HuggingFace models return dataclass outputs (e.g. CausalLMOutputWithPast).
    # Unwrap to the primary tensor (logits) for numeric comparison.
    if not isinstance(expected_output, torch.Tensor):
        expected_output = getattr(
            expected_output,
            "logits",
            getattr(expected_output, "last_hidden_state", next(iter(expected_output), expected_output)),
        )

    if verbose:
        print(f"✅ Created test input with shape: {input_shape} (int={_use_int_input})")
        print(f"✅ PyTorch output shape: {expected_output.shape}")

    # Create symbolic input.
    # Only batch is symbolic (varies at runtime). seq and all feature dims are
    # kept concrete so that attention softmax (which reduces over seq) and matmul
    # shape checks always resolve to integers during tracing.
    batch_sym = ir.Symbol("batch")
    sym_shape = [batch_sym] + list(input_shape[1:])  # concrete from dim 1 onward
    concrete_symbols = {"batch": input_shape[0]}

    _sym_dtype = ir.Int32() if _use_int_input else ir.Fp32()
    x_sym = nn.Tensor(shape=tuple(sym_shape), dtype=_sym_dtype)
    symbolic_args = (x_sym,)

    if verbose:
        print("\n" + "=" * 80)
        print("🚀 RUNNING EVALUATION")
        print("=" * 80)

    # Run the test — catch weight-name mismatches and report them clearly
    try:
        result = run_td_test(
            name=test_name or "Transpiled Model",
            td_module=td_model,
            symbolic_args=symbolic_args,
            concrete_inputs=concrete_inputs,
            concrete_symbols=concrete_symbols,
            torch_model=torch_model,
            expected_output=expected_output,
            atol=1e-5,
        )
    except Exception as _e:
        if "url_dict" in str(_e) or "resource information" in str(_e):
            torch_names = [n for n, _ in torch_model.named_parameters()]
            td_names = [n for n, _ in td_model.named_parameters()]
            raise type(_e)(
                f"{_e}\n\n"
                f"PYTORCH parameter names:   {torch_names}\n"
                f"TENSORDYNE parameter names: {td_names}\n"
                f"Fix: make Tensordyne attribute names match PyTorch exactly."
            ) from None
        raise

    if verbose:
        print_results([result])

    # Attach the dtype used for input generation so callers (e.g. the agent
    # tool wrapper) can show it in error messages — saves the agent from
    # guessing what input the evaluator actually fed to PyTorch.
    result.used_input_dtype = "int" if _use_int_input else "float"  # type: ignore[attr-defined]
    return result
