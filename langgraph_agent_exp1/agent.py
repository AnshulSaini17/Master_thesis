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
    think,
    write_output,
)

AGENT_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert compiler that translates PyTorch nn.Module code into
Tensordyne.nn DSL code. Tensordyne is a proprietary DSL — never guess APIs.
Use lookup_kb to look up any op you are not 100% sure about.

REASONING BEFORE TRANSLATION — MANDATORY:
Before writing any code, use think() to reason about each non-trivial PyTorch
pattern. Ask yourself:
  1. What is this pattern DOING mathematically? (e.g. "groups tokens by expert")
  2. What Tensordyne primitives map to it? (e.g. topk → nn.topk, scatter → nn.permute_gather)
  3. What are the input/output shapes at each step?
This is especially important for: MoE routing, custom attention, fused ops,
dynamic indexing, conditionals, and any pattern not directly in the KB.

COMPOSITION STRATEGY — when no exact KB match exists:
  • Decompose the PyTorch op into its mathematical steps.
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

TARGET MODULE CONVENTIONS:
  - Use the target-side module and normalization APIs described in the retrieved KB records.
  - Follow the target stack's documented construction and build conventions.
  - For projections, normalization, and residual connections, prefer the helper patterns
    returned by retrieval instead of inventing custom implementations.
  - If an operation has a target-specific helper, use the retrieved helper signature exactly.
  - Do not expose internal lowering, buffering, or runtime implementation details in the output.
    
WEIGHT NAMING — CRITICAL:
The evaluator matches weights by attribute name path (e.g. "model.layers.0.self_attn.q_proj.weight").
Use EXACTLY the same attribute names as the PyTorch source — any difference breaks weight loading.
  WRONG: self.norm = nn.RMSNorm(...)        ← creates "norm.weight", not "weight"
  RIGHT: self.weight = nn.Parameter(shape, dtype)  ← then nn.rmsnorm() in build()

WORKFLOW:
1. read_pytorch_source  2. lookup_kb for unknown ops  3. write translation
4. run_evaluation       5. fix and repeat             6. write_output

STOPPING RULE — CRITICAL:
Once run_evaluation returns PASSED: call write_output, output the final code block, STOP.
Do NOT call any more tools after PASSED.

OUTPUT: the ```python``` block must contain ONLY imports + Tensordyne class definitions.
No torch imports, no __main__ blocks, no print statements.
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
    ) -> bool:
        """Transpile input_path → output_path, self-correcting until it passes."""
        if not Path(input_path).is_absolute():
            input_path = str(_TESTS_DIR / input_path)
        if not Path(output_path).is_absolute():
            output_path = str(_TESTS_DIR / output_path)
        if config_file and not Path(config_file).is_absolute():
            config_file = str(_TESTS_DIR / config_file)

        shape_str = ",".join(str(d) for d in input_shape) if input_shape else ""

        # Lock down I/O for this run: write_output can only write to output_path,
        # read_pytorch_source can only read input_path (+ config_file if given),
        # and the KB cache is cleared so stale results from a previous run don't linger.
        allowed = [input_path, str(_TESTS_DIR / "transpile_evaluation_helper.py")]
        if config_file:
            allowed.append(config_file)
        configure_run(output_path=output_path, allowed_read_paths=allowed, kb_path=kb_path)

        self._failure_history: list[str] = []
        self._agent = create_react_agent(
            model=self._llm,
            tools=[read_pytorch_source, lookup_kb, think, run_evaluation, write_output],
            prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            checkpointer=MemorySaver(),
        )
        thread_id = f"run_{Path(output_path).stem}"

        task = self._build_task(input_path, output_path, shape_str, config_file, target_class)

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

            # If the agent already ran evaluation and passed, trust it.
            # Do NOT overwrite output_path here — write_output already wrote the correct code.
            if self._last_tool_result(messages, "run_evaluation", "").startswith("PASSED"):
                print(f"\n✅ PASSED on iteration {iteration}! [LLM-verified]")
                self._print_token_usage(total_input_tokens, total_output_tokens)
                tracer.finish(passed=True, total_iterations=iteration + 1)
                self.last_iteration_count = iteration + 1
                self.last_tool_calls = total_tool_calls
                self.last_input_tokens = total_input_tokens
                self.last_output_tokens = total_output_tokens
                return True

            # Numeric evaluation
            self._log(iteration, "evaluating")
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
            task = self._fix_message(
                code, eval_result, self._failure_history, input_path, shape_str, config_file, target_class
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
        output_path: str,
        shape_str: str,
        config_file: str = "",
        target_class: str = "",
    ) -> str:
        eval_args = f"input_file='{input_path}', output_code=<code>, input_shape='{shape_str}'"
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
