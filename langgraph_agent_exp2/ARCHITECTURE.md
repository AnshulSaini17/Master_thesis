# agent.py — Architecture Deep Dive

## Overview

`LangGraphAgent` wraps a LangGraph `create_react_agent` in an outer correction loop.
There are **two nested loops** — the inner ReAct loop (managed by LangGraph) and the
outer correction loop (managed by `agent.py` itself). Understanding both is the key to
seeing what is and isn't necessary.

---

## Initialization (`__init__`)

```
LangGraphAgent.__init__()
│
├── ChatAnthropic(model="claude-opus-4-6")          ← the LLM
│
├── OpenAIEmbeddings(model="text-embedding-3-small") ← for cross-run memory search
├── InMemoryStore(index={dims:1536, embed})          ← cross-run memory (survives across .run() calls)
│
├── MemorySaver()                                    ← within-run message history (THROWN AWAY in run())
│
└── create_react_agent(...)                          ← also THROWN AWAY immediately in run()
```

**Note:** The agent and checkpointer created in `__init__` are never used.
`run()` always recreates them. The only things that survive from `__init__` into `run()` are
`self._llm` and `self._store`.

---

## `run()` — Main Entry Point

```
run(input_path, output_path, input_shape, config_file, target_class)
│
├── 1. Normalize all paths to absolute
│
├── 2. configure_run()
│       Sets global allowed read paths in tools.py
│       Clears the KB cache (so stale results from prev model don't linger)
│
├── 3. Recreate agent with fresh MemorySaver()
│       Why: Each model gets a completely clean conversation history.
│       The self._store (cross-run memory) is intentionally NOT reset.
│
├── 4. _recall_similar(filename)
│       Semantic search of InMemoryStore for past successful transpilations.
│       Returns up to 2 code snippets (first 400 chars each).
│       Injected at the top of the initial task prompt.
│
├── 5. _build_task(...)
│       Builds the initial HumanMessage prompt. Includes:
│         - Past similar transpilations (if any)
│         - Which classes to transpile / which to skip
│         - Step-by-step instructions (read → lookup → write → eval → fix → write_output)
│
└── 6. Outer correction loop  ← see below
```

---

## The Outer Correction Loop

This loop exists because the ReAct agent sometimes gives up early — it outputs a final answer
without having passed eval. The outer loop is the safety net that catches this and forces another
agent turn with explicit error feedback.

```
for iteration in range(max_iterations):   # default: 8
│
├── A. Inner API retry loop (3 attempts, exponential backoff: 5s, 10s)
│       Each retry gets a FRESH thread_id (e.g. run_model_a0, run_model_a1)
│       Why: A mid-stream API crash can leave corrupted state in MemorySaver.
│            A fresh thread_id = fresh conversation slate for the retry.
│
│       _stream_agent(task, config)
│       └── self._agent.stream(...)     ← THE INNER REACT LOOP (see below)
│           Streams until agent gives final answer (no more tool calls).
│           Prints tool calls + result previews live as they happen.
│           Returns final state via get_state().
│
├── B. Message slicing
│       Finds the index of the last HumanMessage in the full history.
│       Slices to only messages from THIS turn.
│       Why: Used by the tracer and by _last_tool_result().
│
├── C. Tracer logging  (RunTracer)
│       Logs every tool call + result from this turn to a debug .md file.
│       Not needed for correctness — pure debug infrastructure.
│
├── D. _extract_code_from_response(messages)
│       Scans messages in reverse for the last ```python ... ``` block.
│       If none found → inject "please output code" message, continue to next iteration.
│
├── E. EVAL PATH 1 — "Did the agent already call run_evaluation and get PASSED?"
│       _last_tool_result(messages, "run_evaluation") → checks tool result strings
│       If the agent's own run_evaluation call returned "PASSED...":
│           → write file, save to cross-run memory, return True
│       This is an OPTIMIZATION — avoids calling eval again when we know it passed.
│
├── F. No-shape shortcut
│       If input_shape was not provided → write file and return True without eval.
│       (Numeric comparison requires a shape to create dummy input tensors.)
│
├── G. EVAL PATH 2 — Authoritative external eval call
│       run_evaluation.invoke({input_file, output_code=extracted_code, input_shape, ...})
│       This is the REAL check. Runs even if the agent already called eval (and failed).
│       Why separate from agent's own call: guarantees we evaluate the EXACT code
│       in the final ```python``` block, not whatever intermediate code the agent tested.
│       If PASSED → write file, save to cross-run memory, return True
│       If FAILED → build _fix_message, go to next iteration
│
└── After max_iterations: return False
```

---

## The Inner ReAct Loop (inside `_stream_agent`)

This is managed entirely by LangGraph. In a single call to `_stream_agent`, the agent
can call tools multiple times before giving its final answer.

```
Agent receives HumanMessage (task or fix_message)
│
└── ReAct loop (LangGraph internal):
    │
    ├── Agent decides: call a tool or give final answer?
    │
    ├── If tool call → execute tool → result fed back as ToolMessage → repeat
    │
    └── If final answer → streaming ends
    
    Typical happy path within one outer iteration:
      read_pytorch_source → lookup_kb (×N) → [run_evaluation → fix → run_evaluation again] → write_output → final answer
    
    The agent CAN call run_evaluation multiple times internally in one outer iteration.
    Each internal run_evaluation call is in addition to the outer eval (Eval Path 2).
```

---

## Evaluation — How Many Times and When

This is the most important thing to understand:

```
Per outer iteration, run_evaluation can be called:

  [Inner] Agent calls run_evaluation via tool: 0, 1, or many times
              ↓
  [Outer] Eval Path 1 check (line 224): reads the agent's tool result — NO new eval call
              ↓ (only if agent didn't pass)
  [Outer] Eval Path 2 (line 239): 1 mandatory external call

So per outer iteration: up to (N internal + 1 external) eval calls.
Across all iterations: up to max_iterations × (N+1) total eval calls.
```

In practice, on a clean first attempt:
- Iteration 0: agent calls `run_evaluation` once internally → outer also calls it → **2 total**
- If it fails, iteration 1: same → **2 more**
- etc.

If the agent is smart and self-corrects internally (calls eval, sees failure, fixes, calls eval again
before giving a final answer), you could have 3–4 eval calls in a single outer iteration.

---

## All Functions — Quick Reference

### `agent.py`

| Function | What it does |
|---|---|
| `__init__` | Creates LLM, cross-run memory store, embeddings. Also creates an agent that gets thrown away. |
| `run()` | Entry point. Normalizes paths, recreates agent, runs outer correction loop. |
| `_stream_agent()` | Runs one ReAct turn, streaming tool calls to stdout. Returns full message state. |
| `_fmt_tool_args()` | Formats tool args for pretty stdout printing. Display only, no functional role. |
| `_last_tool_result()` | Walks message history to find the last result from a named tool call. Used for Eval Path 1. |
| `_build_task()` | Builds the initial HumanMessage prompt with instructions + past examples. |
| `_fix_message()` | Builds the "here's the error, fix it" prompt for subsequent iterations. Includes error history. |
| `_extract_code_from_response()` | Finds the last ```python block in AI messages. |
| `_save_to_memory()` | Stores a successful transpilation snippet in InMemoryStore for future recall. |
| `_recall_similar()` | Semantic search of InMemoryStore to find past similar transpilations. |
| `_log()` | Prints a banner line for each iteration. |

### `tools.py`

| Tool | What it does |
|---|---|
| `configure_run()` | Sets allowed read paths and clears KB cache. Called once per `run()`. |
| `read_pytorch_source()` | Reads the PyTorch file. Enforces path whitelist. |
| `lookup_kb()` | FAISS similarity search on Tensordyne API docs. Results cached per run. |
| `run_evaluation()` | Writes code to temp file, runs `run_transpiled_evaluation`, returns `PASSED (max_diff=...)` or `FAILED: ...`. |
| `write_output()` | Writes code to the configured output path. Path is fixed — agent cannot choose it. |

---

## What You Actually Need vs What You Can Drop

### Definitely needed
- Cross-run memory (`_store`, `_recall_similar`, `_save_to_memory`)
- Agent recreation with fresh `MemorySaver` per `run()` call
- Outer correction loop (safety net for early agent exits)
- `_extract_code_from_response` (need to know if agent produced code)
- Eval Path 2 (the authoritative external eval call)
- `_build_task` and `_fix_message`
- API retry with backoff

### Can be dropped / simplified
- **`RunTracer`** — all of lines 163, 167, 202–214, 248, 262. Pure debug infrastructure.
  Removing it also eliminates the need for message slicing (lines 188–192) and `_last_tool_result`.
- **Eval Path 1** (line 224) — it's an optimization shortcut to avoid re-running eval when the
  agent already passed. If you drop it, Eval Path 2 still catches the pass on the next check.
  Slightly wasteful but correct and much simpler.
- **`error_history`** in `_fix_message` — the current error is already in the prompt. The history
  of previous errors adds marginal value.
- **Agent creation in `__init__`** — it's thrown away immediately. Just don't create it there.
- **Fresh thread_id per API retry attempt** — simplify to one retry with the same thread.
- **`_fmt_tool_args`** — nice display helper, not functional.
