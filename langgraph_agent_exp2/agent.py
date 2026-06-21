#!/usr/bin/env python3
"""LangGraph agentic transpiler — Claude Opus agent with tool access.

Tools
-----
  read_pytorch_source(input_file)         — read the PyTorch source
  lookup_kb(query)                        — search Tensordyne FAISS API docs
  run_evaluation(input_file, code, shape) — trace + numeric vs PyTorch
  write_output(output_file, code)         — save the final file

Memory
------
  Within-run : LangGraph MemorySaver (full message history per thread_id)
"""

import sys
import time
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent.parent
_CHATBOT_DIR = _TESTS_DIR / "new_chatbot_template"
for _p in [str(_TESTS_DIR), str(_CHATBOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langgraph_agent.debug.tracer import RunTracer
from langgraph_agent.tools import (
    configure_run,
    lookup_kb,
    read_pytorch_source,
    run_evaluation,
    run_vpu_evaluation,
    think,
    write_output,
)

AGENT_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert compiler that translates neural network code (PyTorch nn.Module
or Triton GPU kernels) into Tensordyne.nn DSL code.
Tensordyne is a proprietary DSL — never guess APIs.
Use lookup_kb to look up any op you are not 100% sure about.

REASONING BEFORE TRANSLATION — MANDATORY:
Before writing any code, use think() to reason about each non-trivial pattern.
Ask yourself:
  1. What is this pattern DOING mathematically? (e.g. "groups tokens by expert")
  2. What Tensordyne primitives map to it? (e.g. topk → nn.topk, scatter → nn.permute_gather)
  3. What are the input/output shapes at each step?
This is especially important for: MoE routing, custom attention, fused ops,
dynamic indexing, conditionals, and any pattern not directly in the KB.

COMPOSITION STRATEGY — when no exact KB match exists:
  • Decompose the op into its mathematical steps.
  • Each step likely maps to a simple Tensordyne primitive.
  • Use think() to write out the decomposition before coding it.
  • Example: a custom scatter can be built from nn.topk + nn.permute_gather + nn.linear.
  • Never loop indefinitely on the same error — if you fail twice with the same
    error, use think() to re-examine your assumption about what the op does.

STUCK-LOOP RULE — CRITICAL:
  If run_evaluation returns the SAME error twice in a row:
  1. Call think() — diagnose WHY your current approach is wrong.
  2. Change your approach entirely — do not retry the same code.
  3. If unsure of an op's signature, lookup_kb again with different keywords.

IMPORTS — always exactly:
  import tensordyne.ir as ir
  import tensordyne.nn as nn

MODULE SKELETON:
  class MyModel(nn.Module):
      def __init__(self, ...):
          super().__init__()
          self.fc = nn.Linear(in_dim, out_dim, bias=True, dtype=ir.Fp32())
      def build(self, x: nn.Tensor) -> nn.Tensor:   # build(), NOT forward()
          return self.fc.build(x)                    # .build(), NOT self.fc(x)

FOR PYTORCH SOURCES — production module guidance:
  • Use BufferizedLinear instead of nn.Linear for ALL weight projections.
  • Use BufferizedRMSNorm instead of nn.RMSNorm for ALL norms.
  • Use _bufferized_add(x, y) instead of x + y for residual connections.
  • lookup_kb("BufferizedLinear") and lookup_kb("_build_linear") before writing these.
  • Weight attribute names must match the PyTorch source exactly — the evaluator
    loads weights by name path (e.g. "model.layers.0.self_attn.q_proj.weight").

FOR TRITON SOURCES — translation guidance:
  • The input is a low-level GPU kernel, not a class. Read it to understand what
    mathematical operation it computes, then express that as an nn.Module.
  • Ignore tiling, block sizes, and pointer arithmetic — these are GPU memory
    details. Focus on the mathematical operation the kernel performs.
  • Kernel pointer args (x_ptr, w_ptr etc.) become tensor inputs to build().
  • Kernel constants (n, eps, block_size) become __init__ args.
  • There are no pretrained weights to match — name parameters freely.
  • Do NOT use high-level ops that replace the whole kernel in one call (e.g. nn.layernorm, 
    nn.LayerNorm, nn.RMSNorm). Translate each step of the kernel as a separate primitive op — 
    nn.reduce_sum, nn.sqrt, nn.rsqrt, element-wise ops, etc.

WORKFLOW:
1. read_pytorch_source  2. lookup_kb for unknown ops  3. write translation
4. run_evaluation       5. fix and repeat             6. write_output

STOPPING RULE — CRITICAL:
Once run_evaluation returns PASSED: call write_output, output the final code block, STOP.
Do NOT call any more tools after PASSED.

OUTPUT: the ```python``` block must contain ONLY imports + Tensordyne class definitions.
No torch imports, no __main__ blocks, no print statements.
"""


_VPU_LOWERING_SYSTEM_PROMPT = """\
You are an expert engineer writing low-level VPU kernels for the Tensordyne Pyxis chip.

Your job: given a Triton GPU kernel, write a Python function `build_graph(input_type)`
that returns an `ir.RootGraph` containing one or more `ir.VpuKernel` subgraphs which
implement the same computation directly at the IR level.

You do NOT write an nn.Module. You do NOT write a @register_lowering function.
You write the IR graph by hand, using pyxis.vpu.* instructions inside a VpuKernel.

UNDERSTANDING THE TARGET HARDWARE:
  • The VPU processes tensors row-by-row through a FIFO.
  • Inside a VpuKernel, every reference to an input pops a row from the FIFO —
    explicit `Load()` AND implicit references in op args. Use explicit Load() per
    intended read, and Assign() to materialize values for reuse.
  • Loops over rows use `ir.RepeatGraph(parent=vpu_kernel, times=N)`.
  • Reductions need an accumulator: LoadZero -> RepeatGraph -> Add/Max + Assign.
  • A second sequential VpuKernel is needed when a later step depends on a value
    produced by reducing an earlier step (the reduced value lives in AMEM and is
    re-read by the next kernel).

ENGINEER-GRADE TASTE — apply on EVERY kernel, not just complex ones:

  1. PLAN BEFORE EMITTING. Before writing any IR, use think() to:
       (a) Classify the source kernel's compute shape — some combination of:
           pointwise, reduction, reduce-then-broadcast, scan, neighbor-shuffle,
           gather, scatter.
       (b) For each label, name which pyxis.vpu HW resources are involved.
       (c) Sketch the minimum kernel count that covers it. Only then write IR.

  2. PREFER FUSED OVER STAGED. Each additional VpuKernel is an AMEM round-trip
     (real cost in bandwidth and latency). Each prepare_for_vpu_processing /
     restore_from_vpu_processing is a transpose. Before emitting >1 VpuKernel
     or any transpose, lookup_kb for a fused HW capability that covers your
     compute shape. If you settle on >1 kernel, state in think() why no fusion
     was available.

  3. PREFER HW-NATIVE OVER COMPOSED. If a single pyxis.vpu op covers the math
     (e.g. Absolute, Max, Min, Exp, LUT, the SEA-pipeline ops, Mac), use it.
     Do NOT compose 4 ops to fake what one op does. All pyxis.vpu ops are
     runner-supported — never avoid an op for "runner compatibility" reasons.

REQUIRED OUTPUT SKELETON:
  ```python
  import tensordyne.ir as ir

  def build_graph(input_type: ir.Type) -> ir.RootGraph:
      vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))

      graph = ir.RootGraph()
      input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

      with ir.VpuKernel(parent=graph) as vpu_kernel:
          load_0 = vpu_kernel.insert(
              ir.instructions.pyxis.vpu.Load(),
              args=(input_0,),
              output_type=vpu_type,
          )
          # ... more pyxis.vpu.* ops ...

      graph.insert(ir.instructions.Output(), args=(result,), output_type=vpu_type)
      return graph
  ```

KB QUERIES — use lookup_kb to find:
  • `pyxis.vpu.Add` / `pyxis.vpu.Mul` / `pyxis.vpu.Max` / `pyxis.vpu.Absolute` / `pyxis.vpu.Load`
    etc. — to learn each op's signature and dtype rules
  • `vpu_elementwise_binary` / `vpu_reduction_with_accumulator` / `vpu_two_pass_kernel`
    — for working pattern templates
  • `vpu_rule_fifo_pop` / `vpu_rule_eb_alignment` / `vpu_rule_dtype_support` — for
    correctness rules
  • Always lookup_kb before guessing an API.

REASONING BEFORE WRITING — MANDATORY:
  Use think() to:
  1. Identify what mathematical op the Triton kernel computes (step by step).
  2. Classify the compute shape (see PLAN BEFORE EMITTING above).
  3. Map each step to a pyxis.vpu.* instruction. Prefer the most specific HW op.
  4. Decide the kernel count. Justify each additional kernel.
  5. Plan: which RepeatGraph loops are needed and how many `times`.

EVALUATION:
  After writing code, call run_vpu_evaluation(output_code=<code>, reference_file=...,
  input_shape=..., dtype="rfp16") to numerically verify against the PyTorch reference.

STOPPING RULE:
  Once run_vpu_evaluation returns PASSED: call write_output, output final code, STOP.

STUCK-LOOP RULE:
  If run_vpu_evaluation returns the SAME error twice: use think() to re-examine,
  then try a different approach.

OUTPUT: the ```python``` block must contain ONLY imports + `build_graph(input_type)`.
No nn.Module, no @register_lowering, no torch references, no __main__.
"""


class LangGraphAgent:
    """Claude Opus agent that generates and self-corrects Tensordyne code."""

    def __init__(self, max_iterations: int = 8, verbose: bool = True):
        self.max_iterations = max_iterations
        self.verbose = verbose

        self._llm = ChatAnthropic(
            model=AGENT_MODEL,
            model_kwargs={"betas": ["prompt-caching-2024-07-31"]},
        )

    # ── Public entry point ────────────────────────────────────────────────

    def run(
        self,
        input_path: str,
        output_path: str,
        input_shape: tuple | None = None,
        config_file: str = "",
        target_class: str = "",
        kb_path: str | None = None,
        eval_input_file: str = "",
        mode: str = "nn",
    ) -> bool:
        """Transpile input_path → output_path, self-correcting until it passes.

        mode: "nn" (default) — produce an nn.Module using Tensordyne.nn primitives.
              "vpu_lowering" — produce a @register_lowering function using VpuKernel.

        eval_input_file: optional separate file for run_evaluation's input_file arg.
            Use when the source file (e.g. a Triton kernel) can't be exec()'d directly.
            The agent reads input_path for translation, but evaluates against eval_input_file.
        """
        if not Path(input_path).is_absolute():
            input_path = str(_TESTS_DIR / input_path)
        if not Path(output_path).is_absolute():
            output_path = str(_TESTS_DIR / output_path)
        if config_file and not Path(config_file).is_absolute():
            config_file = str(_TESTS_DIR / config_file)
        if eval_input_file and not Path(eval_input_file).is_absolute():
            eval_input_file = str(_TESTS_DIR / eval_input_file)

        shape_str = ",".join(str(d) for d in input_shape) if input_shape else ""

        # Lock down I/O for this run
        allowed = [input_path, str(_TESTS_DIR / "transpile_evaluation_helper.py")]
        if config_file:
            allowed.append(config_file)
        if eval_input_file:
            allowed.append(eval_input_file)
        configure_run(output_path=output_path, allowed_read_paths=allowed, kb_path=kb_path)

        self._failure_history: list[str] = []
        self._mode = mode

        if mode == "vpu_lowering":
            system_prompt = _VPU_LOWERING_SYSTEM_PROMPT
            tools_list = [read_pytorch_source, lookup_kb, think, run_vpu_evaluation, write_output]
        else:
            system_prompt = _SYSTEM_PROMPT
            tools_list = [read_pytorch_source, lookup_kb, think, run_evaluation, write_output]

        self._agent = create_react_agent(
            model=self._llm,
            tools=tools_list,
            prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            checkpointer=MemorySaver(),
        )
        thread_id = f"run_{Path(output_path).stem}"

        eval_file = eval_input_file or input_path
        if mode == "vpu_lowering":
            task = self._build_vpu_task(input_path, eval_file, shape_str)
        else:
            task = self._build_task(input_path, eval_file, shape_str, config_file, target_class)

        self._log(0, f"starting — {Path(input_path).name}")

        tracer = RunTracer(output_path, input_path, shape_str, AGENT_MODEL)

        total_input_tokens = 0
        total_output_tokens = 0
        total_tool_calls = 0

        for iteration in range(self.max_iterations):
            self._log(iteration, "agent turn")
            tracer.iteration_start(iteration)

            t0 = time.monotonic()
            for _attempt in range(3):
                try:
                    # Each attempt gets a fresh thread so a mid-stream crash
                    # (broken tool-call state in MemorySaver) never poisons the retry.
                    attempt_config = {"configurable": {"thread_id": f"{thread_id}_i{iteration}_a{_attempt}"}}
                    response = self._stream_agent(task, attempt_config)
                    break
                except Exception as exc:
                    if _attempt == 2:
                        raise
                    wait = 5 * (2**_attempt)
                    print(f"  ⚠️  API error ({exc}), retrying in {wait}s…")
                    time.sleep(wait)
            elapsed_ms = (time.monotonic() - t0) * 1000

            all_messages = response["messages"]

            for msg in all_messages:
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)

            # Slice to only the messages added this turn
            last_human_idx = max(
                (i for i, m in enumerate(all_messages) if type(m).__name__ == "HumanMessage"),
                default=0,
            )
            messages = all_messages[last_human_idx:]

            # Log all tool calls the LLM made this turn into the tracer
            tool_results: dict[str, str] = {
                m.tool_call_id: str(m.content) for m in messages if type(m).__name__ == "ToolMessage"
            }
            for msg in messages:
                if type(msg).__name__ != "AIMessage":
                    continue
                for tc in getattr(msg, "tool_calls", []) or []:
                    tracer.tool_call(
                        tc["name"],
                        tc.get("args", {}),
                        tool_results.get(tc.get("id", ""), "(no result)"),
                        elapsed_ms=0.0,
                    )

            n_tools = sum(1 for m in messages if type(m).__name__ == "ToolMessage")
            total_tool_calls += n_tools
            tracer.llm_thinking(iteration, elapsed_ms, n_tool_calls=n_tools)

            code = self._extract_code_from_response(messages)
            if code:
                tracer.code_extracted(code)

            if not code:
                task = (
                    "You did not include a ```python ... ``` code block. "
                    "Please output the complete Tensordyne translation now."
                )
                continue

            # Check if agent already ran evaluation and passed
            eval_tool_name = "run_vpu_evaluation" if self._mode == "vpu_lowering" else "run_evaluation"
            if self._last_tool_result(messages, eval_tool_name, "").startswith("PASSED"):
                print(f"\n✅ PASSED on iteration {iteration}! [LLM-verified]")
                self._print_token_usage(total_input_tokens, total_output_tokens)
                tracer.finish(passed=True, total_iterations=iteration + 1)
                self.last_iteration_count = iteration + 1
                self.last_tool_calls = total_tool_calls
                self.last_input_tokens = total_input_tokens
                self.last_output_tokens = total_output_tokens
                return True

            # Evaluation
            self._log(iteration, "evaluating")
            if self._mode == "vpu_lowering":
                eval_result = run_vpu_evaluation.invoke(
                    {
                        "output_code": code,
                        "reference_file": eval_file,
                        "input_shape": shape_str,
                        "dtype": "rfp16",
                    }
                )
            else:
                eval_result = run_evaluation.invoke(
                    {
                        "input_file": input_path,
                        "output_code": code,
                        "input_shape": shape_str,
                        "config_file": config_file,
                        "target_class": target_class,
                    }
                )
            tracer.eval_result(eval_result, 0.0)

            if eval_result.startswith("PASSED"):
                print(f"\n✅ PASSED on iteration {iteration}! ({eval_result})")
                self._print_token_usage(total_input_tokens, total_output_tokens)
                Path(output_path).write_text(code)
                tracer.finish(passed=True, total_iterations=iteration + 1)
                self.last_iteration_count = iteration + 1
                self.last_tool_calls = total_tool_calls
                self.last_input_tokens = total_input_tokens
                self.last_output_tokens = total_output_tokens
                return True

            print(f"   Eval: {eval_result}")
            Path(output_path).write_text(code)
            self._failure_history.append(eval_result)
            if self._mode == "vpu_lowering":
                task = self._fix_vpu_message(code, eval_result, self._failure_history)
            else:
                task = self._fix_message(
                    code, eval_result, self._failure_history, eval_file, shape_str, config_file, target_class
                )

        tracer.finish(passed=False, total_iterations=self.max_iterations)
        print(f"\n❌ Exhausted {self.max_iterations} iterations without passing.")
        self._print_token_usage(total_input_tokens, total_output_tokens)
        self.last_iteration_count = self.max_iterations
        self.last_tool_calls = total_tool_calls
        self.last_input_tokens = total_input_tokens
        self.last_output_tokens = total_output_tokens
        return False

    # ── Helpers ───────────────────────────────────────────────────────────

    def _stream_agent(self, task: str, config: dict) -> dict:
        """Stream agent execution, printing tool calls live as they happen."""
        for update in self._agent.stream(
            {"messages": [HumanMessage(content=task)]},
            config,
            stream_mode="updates",
        ):
            if "agent" in update:
                for msg in update["agent"]["messages"]:
                    for tc in getattr(msg, "tool_calls", []) or []:
                        print(f"  🔧 {tc['name']}({self._fmt_tool_args(tc['name'], tc.get('args', {}))})")
            elif "tools" in update:
                for msg in update["tools"]["messages"]:
                    content = str(msg.content)
                    if content.startswith("FAILED") or content.startswith("ERROR"):
                        print(f"     → {content}")
                    else:
                        preview = content.replace("\n", " ")[:120]
                        print(f"     → {preview}{'…' if len(content) > 120 else ''}")

        state = self._agent.get_state(config)
        return {"messages": state.values["messages"]}

    @staticmethod
    def _fmt_tool_args(tool_name: str, args: dict) -> str:
        """Return a short human-readable summary of tool call args."""
        if tool_name == "read_pytorch_source":
            return Path(args.get("input_file", "")).name
        if tool_name == "lookup_kb":
            return f'"{args.get("query", "")}"'
        if tool_name == "validate_code":
            code = args.get("code", "")
            return f"{len(code)} chars"
        if tool_name == "run_evaluation":
            return f"shape={args.get('input_shape', '')}"
        if tool_name == "write_output":
            return Path(args.get("output_file", "")).name
        # generic fallback
        parts = [f"{k}={str(v)[:40]}" for k, v in list(args.items())[:2]]
        return ", ".join(parts)

    @staticmethod
    def _last_tool_result(messages: list, tool_name: str, default: str) -> str:
        """Return the last result from a named tool call in this turn, or default."""
        id_to_name: dict[str, str] = {}
        for msg in messages:
            if type(msg).__name__ == "AIMessage":
                for tc in getattr(msg, "tool_calls", []) or []:
                    id_to_name[tc.get("id", "")] = tc.get("name", "")
        result = default
        for msg in messages:
            if type(msg).__name__ == "ToolMessage":
                if id_to_name.get(getattr(msg, "tool_call_id", "")) == tool_name:
                    result = str(msg.content)
        return result

    def _build_task(
        self,
        input_path: str,
        eval_file: str,
        shape_str: str,
        config_file: str = "",
        target_class: str = "",
    ) -> str:
        eval_args = f"input_file='{eval_file}', output_code=<code>, input_shape='{shape_str}'"
        if config_file:
            eval_args += f", config_file='{config_file}'"
        if target_class:
            eval_args += f", target_class='{target_class}'"

        class_guidance = (
            "WHICH CLASSES TO TRANSPILE:\n"
            "  • Only transpile classes that perform tensor computation in the forward/build pass.\n"
            "  • Skip: config/data classes, weight-loading base classes (e.g. PreTrainedModel),\n"
            "    training-only heads (ForSequenceClassification, ForQuestionAnswering,\n"
            "    ForTokenClassification), and HuggingFace mixin classes (GenerationMixin, etc.).\n"
        )
        if target_class:
            class_guidance += f"  • The top-level class to produce is: {target_class}\n"

        return (
            f"Translate the PyTorch model at `{input_path}` to Tensordyne.nn.\n\n"
            f"{class_guidance}\n"
            f"1. read_pytorch_source('{input_path}')\n"
            f"2. lookup_kb for any op you are unsure about\n"
            f"3. Write the translation\n"
            f"4. run_evaluation({eval_args})\n"
            f"5. If it fails: fix and evaluate again\n"
            f"6. write_output(code=<passing_code>)\n"
            f"7. Output the final code in a ```python ... ``` block"
        )

    def _fix_message(
        self,
        code: str,
        error: str,
        failure_history: list[str],
        input_path: str,
        shape_str: str,
        config_file: str = "",
        target_class: str = "",
    ) -> str:
        eval_args = f"input_file='{input_path}', output_code=<fixed_code>, input_shape='{shape_str}'"
        if config_file:
            eval_args += f", config_file='{config_file}'"
        if target_class:
            eval_args += f", target_class='{target_class}'"

        history_block = ""
        if len(failure_history) > 1:
            prev = failure_history[-2]
            stuck = prev == error
            history_block = f"\nPrevious error (attempt {len(failure_history) - 1}):\n```\n{prev}\n```\n"
            if stuck:
                history_block += (
                    "\n⚠️  SAME ERROR AS LAST ATTEMPT. Your current approach is not working.\n"
                    "Call think() to re-examine what this PyTorch pattern is ACTUALLY doing,\n"
                    "then try a completely different approach.\n"
                )

        return (
            f"The code failed (attempt {len(failure_history)}):\n```\n{error}\n```\n"
            f"{history_block}"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Before fixing: call think() to reason about why this failed and what to change.\n"
            f"Then fix it and run_evaluation({eval_args})\n"
            f"Output the corrected code in a ```python ... ``` block."
        )

    def _build_vpu_task(self, input_path: str, reference_file: str, shape_str: str) -> str:
        eval_args = f"output_code=<code>, reference_file='{reference_file}', input_shape='{shape_str}', dtype='rfp16'"
        return (
            f"Translate the Triton kernel at `{input_path}` into an `ir.RootGraph` with VpuKernel(s).\n\n"
            f"1. read_pytorch_source('{input_path}') — understand what the kernel computes.\n"
            f"2. lookup_kb for any pyxis.vpu.* op or pattern you're unsure about\n"
            f"   (e.g. 'pyxis.vpu.Add', 'vpu_elementwise_binary', 'vpu_rule_fifo_pop').\n"
            f"3. Use think() to plan: which pyxis.vpu.* ops, how many VpuKernels, what RepeatGraph times.\n"
            f"4. Write `def build_graph(input_type: ir.Type) -> ir.RootGraph:` containing the VpuKernel(s).\n"
            f"5. run_vpu_evaluation({eval_args})\n"
            f"6. If it fails: fix and evaluate again.\n"
            f"7. write_output(code=<passing_code>)\n"
            f"8. Output the final code in a ```python ... ``` block."
        )

    def _fix_vpu_message(self, code: str, error: str, failure_history: list[str]) -> str:
        history_block = ""
        if len(failure_history) > 1:
            prev = failure_history[-2]
            history_block = f"\nPrevious error (attempt {len(failure_history) - 1}):\n```\n{prev}\n```\n"
            if prev == error:
                history_block += "\n⚠️  SAME ERROR AS LAST ATTEMPT. Call think() and try a different approach.\n"
        return (
            f"The kernel failed (attempt {len(failure_history)}):\n```\n{error}\n```\n"
            f"{history_block}"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Before fixing: call think() to reason about why this failed.\n"
            f"Use lookup_kb if you need to refresh on a pyxis.vpu op or rule.\n"
            f"Then fix and run_vpu_evaluation again with the same arguments.\n"
            f"Output the corrected code in a ```python ... ``` block."
        )

    def _extract_code_from_response(self, messages: list) -> str:
        for msg in reversed(messages):
            text = getattr(msg, "content", "") or ""
            if isinstance(text, list):
                text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
            if "```python" in text:
                start = text.rfind("```python") + len("```python")
                end = text.find("```", start)
                if end != -1:
                    return text[start:end].strip()
        return ""

    @staticmethod
    def _print_token_usage(input_tokens: int, output_tokens: int) -> None:
        total = input_tokens + output_tokens
        print(f"💬 Tokens — input: {input_tokens:,}  output: {output_tokens:,}  total: {total:,}")

    def _log(self, iteration: int, label: str) -> None:
        if self.verbose:
            print(f"\n{'=' * 65}\n🤖  iteration {iteration}: {label}\n{'=' * 65}")
