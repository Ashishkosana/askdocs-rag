"""Command-line entry point: ``askdocs index|query|serve``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="askdocs", description="RAG document Q&A with an evaluation harness"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index a file or directory")
    p_index.add_argument("path", help="path to a document or folder")

    p_query = sub.add_parser("query", help="ask a question against the index")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=None)

    p_serve = sub.add_parser("serve", help="run the FastAPI server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "index":
        from .pipeline import RagPipeline

        added = RagPipeline().index(Path(args.path))
        print(f"Indexed {added} chunks.")

    elif args.command == "query":
        from .pipeline import RagPipeline

        ans = RagPipeline().answer(args.question, top_k=args.top_k)
        print(ans.text)
        print("\n--- sources ---")
        for i, h in enumerate(ans.contexts, start=1):
            print(f"[{i}] {h.source} (score {h.score:.3f})")
        print(f"\n{json.dumps(ans.usage.as_dict())}")

    elif args.command == "serve":
        import uvicorn

        uvicorn.run("askdocs.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
