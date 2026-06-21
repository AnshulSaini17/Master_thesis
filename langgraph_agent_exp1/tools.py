"""Tools the agent can call during generation and fixing."""

import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

# Set up import paths once at module level
_TESTS_DIR = Path(__file__).resolve().parent.parent  # Path object — used for / operator
_CHATBOT_DIR = _TESTS_DIR / "new_chatbot_template"
for _p in [str(_CHATBOT_DIR), str(_TESTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


_API_KB_PATH = _TESTS_DIR / "data" / "databases" / "api_kb"

_api_db = None
_kb_path: Path = _API_KB_PATH  # can be changed via configure_run
_kb_cache: dict[str, str] = {}  # query → result, cleared per run
_output_path: str = ""  # set by configure_run; write_output writes here only
_allowed_read_paths: set[str] = set()  # absolute paths the agent may read


def configure_run(
    output_path: str,
    allowed_read_paths: list[str],
    kb_path: str | Path | None = None,
) -> None:
    """Call this before each agent run to lock down I/O paths and clear the KB cache.

    kb_path: optional path to a FAISS KB directory. Defaults to the curated API KB.
             Pass one of:
               - None / omit  → tests/data/databases/api_kb  (curated, default)
               - "token_600"  → tests/data/databases/api_kb_token_600
               - "ast"        → tests/data/databases/api_kb_ast
               - any Path     → used directly
    """
    global _output_path, _allowed_read_paths, _kb_cache, _api_db, _kb_path
    _output_path = str(Path(output_path).resolve())
    _allowed_read_paths = {str(Path(p).resolve()) for p in allowed_read_paths}
    _kb_cache = {}
    _api_db = None  # force reload if KB changed

    if kb_path is None:
        _kb_path = _API_KB_PATH
    elif isinstance(kb_path, str) and not Path(kb_path).exists():
        # shorthand alias
        _kb_path = _TESTS_DIR / "data" / "databases" / f"api_kb_{kb_path}"
    else:
        _kb_path = Path(kb_path)


def _get_db():
    """Return the FAISS KB (lazy-loaded)."""
    global _api_db
    if _api_db is None:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(model="text-embedding-3-large")
        _api_db = FAISS.load_local(str(_kb_path), embeddings=emb, allow_dangerous_deserialization=True)
    return _api_db


@tool
def lookup_kb(query: str) -> str:
    """Search the Tensordyne public API knowledge base for signatures, descriptions, and examples.

    Use short keyword queries — the KB uses semantic similarity search.

    Good:  "nn.repeat"  |  "arange recipe"  |  "RoPE rotary embedding"
    Bad:   "how does reshape work?"
    """
    if query in _kb_cache:
        return f"[cached]\n{_kb_cache[query]}"
    db = _get_db()
    docs = db.similarity_search(query, k=3)
    if not docs:
        return "No results found."
    chunks = []
    for i, d in enumerate(docs, 1):
        text = d.page_content.strip()
        if len(text) >= 40:
            chunks.append(f"--- Result {i} ---\n{text}")
    result = "\n\n".join(chunks) if chunks else "No useful results found."
    _kb_cache[query] = result
    return result


@tool
def run_evaluation(
    input_file: str,
    output_code: str,
    input_shape: str,
    config_file: str = "",
    target_class: str = "",
) -> str:
    """Run full evaluation: trace the Tensordyne model and compare numerically to PyTorch.

    Args:
        input_file: Path to original PyTorch model (.py) — relative paths are
                    resolved against the tests/ directory.
        output_code: The full Tensordyne Python code to evaluate
        input_shape: Comma-separated shape, e.g. "2,10,64"
        config_file: Optional path to a JSON file with constructor kwargs.
                     Use when the model __init__ takes a config object instead
                     of plain args, e.g. {"hidden_size": 64, "num_attention_heads": 4}.
                     Relative paths resolved against tests/.
        target_class: Optional class name to instantiate (e.g. "LlamaForCausalLM").
                      If omitted, the last nn.Module subclass in the file is used.

    Returns
    -------
        'PASSED' if all checks pass, or the error/traceback if something fails.
    """
    sys.path.insert(0, str(_TESTS_DIR))

    # Resolve relative paths against tests/
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_file = str(_TESTS_DIR / input_file)

    if config_file:
        cfg_path = Path(config_file)
        if not cfg_path.is_absolute():
            config_file = str(_TESTS_DIR / config_file)

    shape = tuple(int(x.strip()) for x in input_shape.split(","))

    # Write code to temp file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(output_code)
        tmp_path = f.name

    try:
        from transpile_evaluation_helper import run_transpiled_evaluation

        result = run_transpiled_evaluation(
            input_file=input_file,
            output_file=tmp_path,
            input_shape=shape,
            verbose=False,
            config_file=config_file,
            target_class=target_class,
        )

        # Surface what dtype/shape the helper used so the agent can debug
        # input-related failures without guessing.
        used_dtype = getattr(result, "used_input_dtype", "unknown")
        hint = f"\n[input used: dtype={used_dtype}, shape={shape}]"

        if result.passed:
            max_diff = f"{result.max_diff:.2e}" if result.max_diff is not None else "n/a"
            return f"PASSED (max_diff={max_diff}){hint}"
        return f"FAILED: {_trim_traceback(result.error or '')}{hint}"

    except Exception:
        import traceback

        return f"ERROR:\n{_trim_traceback(traceback.format_exc())}\n[input shape attempted: {shape}]"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Library paths whose frames are usually noise in tracebacks. Frames from these
# are collapsed; user code (anything under tests/ or /tmp temp files) is kept.
_NOISY_FRAME_PATHS = ("site-packages", "/staging/", "/usr/lib/python")


def _trim_traceback(tb_text: str, *, keep_last_noisy: int = 2) -> str:
    """Trim a Python traceback to keep user-code frames + the last few library frames + the final exception.

    Deep HuggingFace/PyTorch tracebacks waste agent context. The final exception line plus the bottom
    couple of library frames is almost always enough to diagnose what went wrong.
    """
    lines = tb_text.split("\n")
    if not lines or "Traceback" not in lines[0]:
        return tb_text

    out = [lines[0]]

    # Walk frames: each is "  File ..." followed by 1-2 code lines.
    frames: list[list[str]] = []
    i = 1
    while i < len(lines) and lines[i].startswith("  File "):
        frame = [lines[i]]
        j = i + 1
        while j < len(lines) and not lines[j].startswith("  File ") and not lines[j].startswith(("  ", "Traceback")) is False:
            # Take code lines (start with spaces) but stop at next File / blank / exception line
            if lines[j].startswith("  File ") or (j < len(lines) and not lines[j].startswith(" ") and lines[j]):
                break
            frame.append(lines[j])
            j += 1
        # Always grab the next 1 line after File (the code), and a possible caret line
        if len(frame) == 1 and i + 1 < len(lines):
            frame.append(lines[i + 1])
            if i + 2 < len(lines) and lines[i + 2].lstrip().startswith("^"):
                frame.append(lines[i + 2])
        frames.append(frame)
        i += len(frame)

    # Classify and keep
    is_noisy = [any(m in f[0] for m in _NOISY_FRAME_PATHS) for f in frames]
    keep_mask = [not n for n in is_noisy]  # always keep user frames

    # Keep the last N noisy frames too
    noisy_indices = [idx for idx, n in enumerate(is_noisy) if n]
    for idx in noisy_indices[-keep_last_noisy:]:
        keep_mask[idx] = True

    kept_frames = [f for f, keep in zip(frames, keep_mask) if keep]
    dropped = len(frames) - len(kept_frames)

    if dropped > 0:
        out.append(f"  [... {dropped} library frame(s) omitted ...]")
    for f in kept_frames:
        out.extend(f)

    # Append the exception line(s) at the bottom (everything after frames)
    if i < len(lines):
        out.extend(lines[i:])

    return "\n".join(out)


@tool
def read_pytorch_source(input_file: str) -> str:
    """Read the PyTorch source file to transpile.

    Args:
        input_file: Path to the PyTorch model (.py) — relative paths are
                    resolved against the tests/ directory.

    Returns
    -------
        The full source code as a string.
    """
    p = Path(input_file)
    if not p.is_absolute():
        p = _TESTS_DIR / input_file
    p = p.resolve()
    if _allowed_read_paths and str(p) not in _allowed_read_paths:
        allowed = sorted(_allowed_read_paths)
        return f"ERROR: access denied: {p}. You may only read: {allowed}"
    if not p.exists():
        return f"ERROR: file not found: {p}"
    if p.is_dir():
        return f"ERROR: {p} is a directory, not a file."
    return p.read_text()


@tool
def think(reasoning: str) -> str:
    """Use this tool to reason out loud before writing or fixing code.

    Call think() when you encounter:
      - A PyTorch pattern with no direct KB match
      - The same error appearing twice in a row
      - Uncertainty about input/output shapes
      - A complex op (MoE routing, fused projections, custom attention)

    Write out:
      1. What the PyTorch pattern is doing mathematically
      2. What Tensordyne primitives map to each step
      3. The expected shape at each step
      4. What you'll do differently if this is a retry

    Args:
        reasoning: Your step-by-step reasoning about the translation problem.

    Returns
    -------
        Your reasoning back, so it appears in context for your next step.
    """
    return f"[reasoning]\n{reasoning}"


@tool
def write_output(code: str) -> str:
    """Write the generated Tensordyne code to the designated output file.

    Call this once run_evaluation has returned PASSED.
    The output path is fixed for this run — you cannot choose it.

    Args:
        code: The complete Tensordyne Python source to write.

    Returns
    -------
        'OK: wrote <path>' or an error string.
    """
    if not _output_path:
        return "ERROR: output path not configured — this is a system bug, report it."
    if "import torch" in code and "import tensordyne" not in code:
        return "ERROR: this looks like PyTorch code, not Tensordyne code. Do not write PyTorch files."
    p = Path(_output_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        return f"OK: wrote {p}"
    except Exception as e:
        return f"ERROR: {e}"
