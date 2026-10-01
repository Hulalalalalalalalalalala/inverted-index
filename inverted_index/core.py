"""An inverted index over tokenised documents, persisted as one JSON snapshot."""

from __future__ import annotations

import json
import math
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

    def phrase(self, text: str) -> list[dict]:
        """Find documents where the token sequence of ``text`` occurs consecutively.

        Returns ``[{"id": doc_id, "positions": [...]}, ...]`` sorted by document
        id; each position is the 0-based index of the phrase's first token in the
        document's token sequence, ascending, with repeats kept.
        """
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("phrase needs at least one token")
        documents = self._read()["documents"]
        results = []
        for doc_id in sorted(documents):
            sequence = tokenize(documents[doc_id])
            positions = [start for start in range(len(sequence) - len(tokens) + 1)
                         if sequence[start:start + len(tokens)] == tokens]
            if positions:
                results.append({"id": doc_id, "positions": positions})
        return results

    def _snapshot(self) -> dict:
        """Read the persisted snapshot and validate its shape and field types."""
        if not self.path.is_file():
            raise FileNotFoundError(f"no index at {self.path}; run init first")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid index at {self.path}: {error}") from error
        if not isinstance(data, dict):
            raise ValueError("index snapshot must be a JSON object")
        documents, postings = data.get("documents"), data.get("postings")
        if not isinstance(documents, dict) or not isinstance(postings, dict):
            raise ValueError("index snapshot needs object 'documents' and 'postings'")
        for doc_id, text in documents.items():
            if not isinstance(doc_id, str) or not isinstance(text, str):
                raise ValueError("document ids and texts must be strings")
        for term, entries in postings.items():
            if not isinstance(term, str) or not isinstance(entries, dict):
                raise ValueError("postings must map string terms to per-document objects")
            for doc_id, tf in entries.items():
                if not isinstance(doc_id, str) or isinstance(tf, bool) or not isinstance(tf, int) or tf < 0:
                    raise ValueError("posting frequencies must be non-negative integers")
        return data

    def rank(self, terms: list[str]) -> list[dict]:
        """Score documents against ``terms`` with a tf-idf sum, best first."""
        if not terms or any(not isinstance(term, str) for term in terms):
            raise ValueError("rank needs at least one string term")
        wanted = sorted({term.lower() for term in terms})
        snapshot = self._snapshot()
        total = len(snapshot["documents"])
        postings = snapshot["postings"]
        scored: dict[str, dict] = {}
        for term in wanted:
            entries = {doc_id: tf for doc_id, tf in postings.get(term, {}).items() if tf > 0}
            if not entries:
                continue
            idf = math.log((1 + total) / (1 + len(entries))) + 1
            for doc_id, tf in entries.items():
                hit = scored.setdefault(doc_id, {"matched": 0, "score": 0.0})
                hit["matched"] += 1
                hit["score"] += (1 + math.log(tf)) * idf
        results = [{"id": doc_id, "matched": hit["matched"], "score": round(hit["score"], 6)}
                   for doc_id, hit in scored.items()]
        results.sort(key=lambda item: (-item["score"], item["id"]))
        return results

    def terms(self) -> list[str]:
        return sorted(self._read()["postings"])

    def stats(self) -> dict:
        document = self._read()
        return {"documents": len(document["documents"]), "terms": len(document["postings"]),
                "postings": sum(len(entries) for entries in document["postings"].values())}

    def reload(self) -> None:
        self._read()
