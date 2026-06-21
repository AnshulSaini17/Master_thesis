#!/usr/bin/env python3
"""Build a FAISS KB from tensordyne-nn source using fixed 600-token chunks.

Usage:
    cd /workspaces/rock
    python tests/build_kb_token.py [--dry-run]
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_TESTS_DIR = Path(__file__).resolve().parent
load_dotenv(_TESTS_DIR / "config" / ".env")

SOURCE_DIR = _TESTS_DIR.parent / "python" / "tensordyne-nn" / "tensordyne"
OUT_PATH = _TESTS_DIR / "data" / "databases" / "api_kb_token_600"


def main(dry_run: bool = False) -> None:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600, chunk_overlap=50
    )

    docs = []
    for py_file in sorted(SOURCE_DIR.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": str(py_file.relative_to(SOURCE_DIR)), "chunk": i},
            ))

    print(f"{len(docs)} chunks from {SOURCE_DIR}")

    if dry_run:
        for d in docs[:3]:
            print(f"\n[{d.metadata}]\n{d.page_content[:200]}")
        return

    emb = OpenAIEmbeddings(model="text-embedding-3-large")
    db = None
    for i in range(0, len(docs), 20):
        batch = docs[i: i + 20]
        print(f"  batch {i // 20 + 1}/{(len(docs) + 19) // 20}")
        if db is None:
            db = FAISS.from_documents(batch, emb)
        else:
            db.add_documents(batch)

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    assert db is not None
    db.save_local(str(OUT_PATH))
    (OUT_PATH / "api_kb_summary.json").write_text(
        json.dumps({"chunking": "token_600", "chunk_size": 600, "chunk_overlap": 50, "total_chunks": len(docs)}, indent=2)
    )
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
