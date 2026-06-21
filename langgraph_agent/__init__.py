"""LangGraph agentic transpiler package.

Adds tests/ and tests/new_chatbot_template/ to sys.path on import
so that transpile_cli, transpile_evaluation_helper, and chatbot_backend
are always importable regardless of the notebook's working directory.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent.parent
_CHATBOT_DIR = _TESTS_DIR / "new_chatbot_template"

for _p in [str(_CHATBOT_DIR), str(_TESTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
