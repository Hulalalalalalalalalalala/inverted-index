"""Command line entry point: ``python3 -m inverted_index --root <dir> <subcommand>``.

Exit codes: 0 success, 1 a storage or verification error, 2 a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import DOMAIN, SOURCE_CATEGORIES, __version__
from .core import InvertedIndex, SnapshotError

USAGE_ERROR = 2

def _tags() -> list[str]:
    """Tags this domain claims: the comma-separated line that follows each named category heading."""
    import pathlib
    corpus = pathlib.Path(__file__).resolve().parent.parent / "corpus.md"
    if not corpus.is_file():
        return []
    wanted, tags, collect = set(SOURCE_CATEGORIES), [], False
    for line in corpus.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped in wanted:
            collect = True
            continue
        if not collect or not stripped:
            continue
        for token in stripped.split(","):
            token = token.strip().replace("\\", "")
            if token and token not in tags:
                tags.append(token)
        collect = False
    return tags


def _report(component_names: list[str], readiness: dict[str, bool]) -> str:
    import json
    return json.dumps({"domain": DOMAIN, "version": __version__, "sourceCategories": list(SOURCE_CATEGORIES),
                       "tags": _tags(), "components": component_names, "readiness": readiness},
                      ensure_ascii=False, sort_keys=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inverted_index", description="倒排索引与词项检索")
    parser.add_argument("--root", required=True, help="working directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create an empty store")
    add = sub.add_parser("add"); add.add_argument("doc_id"); add.add_argument("text")
    get = sub.add_parser("get"); get.add_argument("doc_id")
    update = sub.add_parser("update", help="replace a document's text"); update.add_argument("doc_id"); update.add_argument("text")
    delete = sub.add_parser("delete"); delete.add_argument("doc_id")
    apply_cmd = sub.add_parser("apply", help="atomically apply a JSON array of add/update/delete operations")
    apply_cmd.add_argument("operations", help="a JSON array of operation objects")
    query = sub.add_parser("query"); query.add_argument("terms", nargs="+")
    rank = sub.add_parser("rank", help="score documents against terms"); rank.add_argument("terms", nargs="+")
    phrase = sub.add_parser("phrase", help="find documents containing an exact token sequence"); phrase.add_argument("text")
    near = sub.add_parser("near", help="find documents where two token fragments occur close together")
    near.add_argument("left")
    near.add_argument("right")
    near.add_argument("max_gap", type=int, help="maximum number of tokens strictly between the fragments")
    search = sub.add_parser("search", help="evaluate a boolean expression of terms, phrases, AND, OR, NOT")
    search.add_argument("expression")
    sub.add_parser("terms", help="list every term")
    sub.add_parser("stats", help="print document, term and posting counts")
    sub.add_parser("reload", help="reload the index snapshot")
    sub.add_parser("report", help="print this domain's report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    index = InvertedIndex(args.root)
    try:
        if args.command == "init":
            index.init(); print(f"initialised {index.path}")
        elif args.command == "add":
            print(index.add(args.doc_id, args.text))
        elif args.command == "get":
            text = index.get(args.doc_id); print("" if text is None else text)
        elif args.command == "update":
            print(index.update(args.doc_id, args.text))
        elif args.command == "delete":
            print("true" if index.delete(args.doc_id) else "false")
        elif args.command == "apply":
            operations = json.loads(args.operations)
            print(json.dumps(index.apply(operations)))
        elif args.command == "query":
            print(json.dumps(index.query(args.terms), ensure_ascii=False))
        elif args.command == "rank":
            print(json.dumps(index.rank(args.terms), ensure_ascii=False))
        elif args.command == "phrase":
            print(json.dumps(index.phrase(args.text), ensure_ascii=False))
        elif args.command == "near":
            print(json.dumps(index.near(args.left, args.right, args.max_gap), ensure_ascii=False))
        elif args.command == "search":
            print(json.dumps(index.search(args.expression), ensure_ascii=False))
        elif args.command == "terms":
            print(json.dumps(index.terms()))
        elif args.command == "stats":
            print(json.dumps(index.stats(), sort_keys=True))
        elif args.command == "reload":
            index.reload(); print("ok")
        elif args.command == "report":
            print(_report(["postings", "tokenizer", "near", "search"],
                          {"add": True, "query": True, "delete": True, "ranking": True, "near": True,
                           "search": True}))
        return 0
    except SnapshotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
