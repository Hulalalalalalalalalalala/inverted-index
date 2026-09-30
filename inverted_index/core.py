"""An inverted index over tokenised documents, persisted as one JSON snapshot."""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["InvertedIndex"]

INDEX_FILE = "index.json"
TOKEN = re.compile(r"[0-9A-Za-z_]+")


def tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics and lower-case; no stemming and no stop words."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return [match.group(0).lower() for match in TOKEN.finditer(text)]


class InvertedIndex:
    """A single-process inverted index rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.directory = Path(root)
        self.path = self.directory / INDEX_FILE

    def init(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write({"documents": {}, "postings": {}})

    def _read(self) -> dict:
        if not self.path.is_file():
            raise FileNotFoundError(f"no index at {self.path}; run init first")
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, document: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")

    def add(self, doc_id: str, text: str) -> int:
        if not doc_id:
            raise ValueError("doc_id must be non-empty")
        document = self._read()
        if doc_id in document["documents"]:
            raise ValueError(f"document {doc_id!r} already exists")
        document["documents"][doc_id] = text
        for token in set(tokenize(text)):
            document["postings"].setdefault(token, {})[doc_id] = tokenize(text).count(token)
        self._write(document)
        return len(document["documents"])

    def get(self, doc_id: str) -> str | None:
        return self._read()["documents"].get(doc_id)

    def delete(self, doc_id: str) -> bool:
        document = self._read()
        if doc_id not in document["documents"]:
            return False
        del document["documents"][doc_id]
        for token, postings in list(document["postings"].items()):
            postings.pop(doc_id, None)
            if not postings:
                del document["postings"][token]
        self._write(document)
        return True

    def query(self, terms: list[str]) -> list[dict]:
        if not terms:
            raise ValueError("query needs at least one term")
        wanted = [term.lower() for term in terms]
        postings = self._read()["postings"]
        sets = [set(postings.get(term, {})) for term in wanted]
        hits = set.intersection(*sets) if sets else set()
        return [{"id": doc_id, "matched": sum(1 for term in wanted if doc_id in postings.get(term, {}))}
                for doc_id in sorted(hits, key=lambda value: (-sum(1 for t in wanted if value in postings.get(t, {})), value))]

    def terms(self) -> list[str]:
        return sorted(self._read()["postings"])

    def stats(self) -> dict:
        document = self._read()
        return {"documents": len(document["documents"]), "terms": len(document["postings"]),
                "postings": sum(len(entries) for entries in document["postings"].values())}

    def reload(self) -> None:
        self._read()
