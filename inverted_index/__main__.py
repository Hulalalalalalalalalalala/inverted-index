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
    bm25 = sub.add_parser("bm25", help="score documents against terms with BM25")
    bm25.add_argument("terms", nargs="+")
    bm25.add_argument("--filter", dest="expression", default=None,
                      help="optional boolean filter expression (same syntax as search)")
    bm25.add_argument("--k1", type=float, default=1.2, help="BM25 k1 (finite positive number)")
    bm25.add_argument("--b", type=float, default=0.75, help="BM25 b (a number within [0, 1])")
    phrase = sub.add_parser("phrase", help="find documents containing an exact token sequence"); phrase.add_argument("text")
    sloppy = sub.add_parser("sloppy-phrase", help="find documents containing an ordered token sequence with bounded gaps")
    sloppy.add_argument("text")
    sloppy.add_argument("--slop", type=int, default=0,
                        help="maximum total tokens sandwiched between matched positions (integer 0..2147483647, default 0)")
    window = sub.add_parser("window", help="find unordered minimal covering windows of a token multiset")
    window.add_argument("text")
    window.add_argument("--max-gap", dest="max_gap", type=int, default=0,
                        help="maximum extra tokens inside a covering window (integer 0..2147483647, default 0)")
    near = sub.add_parser("near", help="find documents where two token fragments occur close together")
    near.add_argument("left")
    near.add_argument("right")
    near.add_argument("max_gap", type=int, help="maximum number of tokens strictly between the fragments")
    search = sub.add_parser("search", help="evaluate a boolean expression of terms, wildcards, quoted phrases, NEAR(...) conditions, AND, OR, NOT")
    search.add_argument("expression")
    highlight = sub.add_parser("highlight", help="like search, returning the matched text spans in each document")
    highlight.add_argument("expression")
    snippets = sub.add_parser("snippets", help="like highlight, returning merged context fragments around the matched spans")
    snippets.add_argument("expression")
    snippets.add_argument("--context", type=int, default=20,
                          help="code points of context kept on each side of a highlight (non-negative integer)")
    snippets.add_argument("--max-fragments", dest="max_fragments", type=int, default=3,
                          help="maximum number of fragments per document (positive integer)")
    expand = sub.add_parser("expand", help="list every term matching a wildcard pattern")
    expand.add_argument("pattern")
    fuzzy = sub.add_parser("fuzzy", help="list every term within an edit distance of a term")
    fuzzy.add_argument("term")
    fuzzy.add_argument("--max-distance", dest="max_distance", type=int, default=1,
                       help="maximum Levenshtein edit distance (integer 0..2, default 1)")
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
        elif args.command == "bm25":
            print(json.dumps(index.bm25(args.terms, args.expression, args.k1, args.b), ensure_ascii=False))
        elif args.command == "phrase":
            print(json.dumps(index.phrase(args.text), ensure_ascii=False))
        elif args.command == "sloppy-phrase":
            print(json.dumps(index.sloppy_phrase(args.text, args.slop), ensure_ascii=False))
        elif args.command == "window":
            print(json.dumps(index.window(args.text, args.max_gap), ensure_ascii=False))
        elif args.command == "near":
            print(json.dumps(index.near(args.left, args.right, args.max_gap), ensure_ascii=False))
        elif args.command == "search":
            print(json.dumps(index.search(args.expression), ensure_ascii=False))
        elif args.command == "highlight":
            print(json.dumps(index.highlight(args.expression), ensure_ascii=False))
        elif args.command == "snippets":
            print(json.dumps(index.snippets(args.expression, args.context, args.max_fragments),
                             ensure_ascii=False))
        elif args.command == "expand":
            print(json.dumps(index.expand(args.pattern), ensure_ascii=False))
        elif args.command == "fuzzy":
            print(json.dumps(index.fuzzy(args.term, args.max_distance), ensure_ascii=False))
        elif args.command == "terms":
            print(json.dumps(index.terms()))
        elif args.command == "stats":
            print(json.dumps(index.stats(), sort_keys=True))
        elif args.command == "reload":
            index.reload(); print("ok")
        elif args.command == "report":
            print(_report(["postings", "tokenizer", "near"],
                          {"add": True, "query": True, "delete": True, "ranking": True, "near": True}))
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
