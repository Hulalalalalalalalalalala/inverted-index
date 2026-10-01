"""An inverted index over tokenised documents, persisted as one JSON snapshot."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

__all__ = ["InvertedIndex", "SnapshotError", "tokenize"]

INDEX_FILE = "index.json"
TOKEN = re.compile(r"[0-9A-Za-z_]+")


class SnapshotError(ValueError):
    """The persisted snapshot fails structural or consistency validation."""


def tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics and lower-case; no stemming and no stop words."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return [match.group(0).lower() for match in TOKEN.finditer(text)]


def _require_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_terms(terms: object, name: str) -> None:
    if isinstance(terms, str) or not terms:
        raise ValueError(f"{name} needs at least one term")
    if any(not isinstance(term, str) or not term for term in terms):
        raise ValueError(f"{name} needs non-empty string terms")


def _sequence_starts(sequence: list[str], fragment: list[str]) -> list[int]:
    """All 0-based starts where ``fragment`` occurs consecutively in ``sequence``."""
    length = len(fragment)
    return [start for start in range(len(sequence) - length + 1)
            if sequence[start:start + length] == fragment]


class InvertedIndex:
    """A single-process inverted index rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.directory = Path(root)
        self.path = self.directory / INDEX_FILE

    def init(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write({"documents": {}, "postings": {}})

    def _read(self) -> dict:
        """Read the persisted snapshot and validate structure, types and consistency."""
        if not self.path.is_file():
            raise FileNotFoundError(f"no index at {self.path}; run init first")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SnapshotError(f"invalid index at {self.path}: {error}") from error
        return self._validate(data)

    @staticmethod
    def _validate(data: object) -> dict:
        if not isinstance(data, dict):
            raise SnapshotError("index snapshot must be a JSON object")
        documents, postings = data.get("documents"), data.get("postings")
        if not isinstance(documents, dict) or not isinstance(postings, dict):
            raise SnapshotError("index snapshot needs object 'documents' and 'postings'")
        for doc_id, text in documents.items():
            if not isinstance(doc_id, str) or not isinstance(text, str):
                raise SnapshotError("document ids and texts must be strings")
        for term, entries in postings.items():
            if not isinstance(term, str) or not isinstance(entries, dict):
                raise SnapshotError("postings must map string terms to per-document objects")
            for doc_id, tf in entries.items():
                if not isinstance(doc_id, str) or isinstance(tf, bool) or not isinstance(tf, int) or tf < 0:
                    raise SnapshotError("posting frequencies must be non-negative integers")
                if doc_id not in documents:
                    raise SnapshotError(f"posting for {term!r} references unknown document {doc_id!r}")
        expected: dict[str, dict[str, int]] = {}
        for doc_id, text in documents.items():
            for token, frequency in Counter(tokenize(text)).items():
                expected.setdefault(token, {})[doc_id] = frequency
        if postings != expected:
            raise SnapshotError("postings are inconsistent with the stored documents")
        return data

    def _write(self, document: dict) -> None:
        """Atomically replace the snapshot: either the old or the new state survives."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._validate(document)
        payload = json.dumps(document, sort_keys=True, indent=2)
        handle, temporary = tempfile.mkstemp(dir=self.directory, prefix=".index-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        try:  # durability of the rename itself; not part of the commit contract
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass

    @staticmethod
    def _add_document(document: dict, doc_id: str, text: str) -> int:
        if doc_id in document["documents"]:
            raise ValueError(f"document {doc_id!r} already exists")
        document["documents"][doc_id] = text
        for token, frequency in Counter(tokenize(text)).items():
            document["postings"].setdefault(token, {})[doc_id] = frequency
        return len(document["documents"])

    @staticmethod
    def _update_document(document: dict, doc_id: str, text: str) -> int:
        if doc_id not in document["documents"]:
            raise KeyError(doc_id)
        document["documents"][doc_id] = text
        postings = document["postings"]
        for token in list(postings):
            postings[token].pop(doc_id, None)
            if not postings[token]:
                del postings[token]
        for token, frequency in Counter(tokenize(text)).items():
            postings.setdefault(token, {})[doc_id] = frequency
        return len(document["documents"])

    @staticmethod
    def _delete_document(document: dict, doc_id: str) -> bool:
        if doc_id not in document["documents"]:
            return False
        del document["documents"][doc_id]
        for token, postings in list(document["postings"].items()):
            postings.pop(doc_id, None)
            if not postings:
                del document["postings"][token]
        return True

    def add(self, doc_id: str, text: str) -> int:
        _require_non_empty_string(doc_id, "doc_id")
        _require_non_empty_string(text, "text")
        document = self._read()
        result = self._add_document(document, doc_id, text)
        self._write(document)
        return result

    def update(self, doc_id: str, text: str) -> int:
        """Replace ``doc_id``'s text and rebuild its postings, frequencies and positions.

        The document count is unchanged; the new total is returned.
        """
        _require_non_empty_string(doc_id, "doc_id")
        _require_non_empty_string(text, "text")
        document = self._read()
        result = self._update_document(document, doc_id, text)
        self._write(document)
        return result

    def get(self, doc_id: str) -> str | None:
        _require_non_empty_string(doc_id, "doc_id")
        return self._read()["documents"].get(doc_id)

    def delete(self, doc_id: str) -> bool:
        _require_non_empty_string(doc_id, "doc_id")
        document = self._read()
        result = self._delete_document(document, doc_id)
        self._write(document)
        return result

    @staticmethod
    def _parse_operation(operation: object, index: int) -> tuple[str, str, str | None]:
        """Validate one batch element's shape, types and field set."""
        label = f"operation {index}"
        if not isinstance(operation, dict):
            raise ValueError(f"{label} must be an object")
        op = operation.get("op")
        if op not in ("add", "update", "delete"):
            raise ValueError(f"{label} has an unknown op {op!r}; expected add, update or delete")
        expected = {"op", "doc_id"} if op == "delete" else {"op", "doc_id", "text"}
        keys = set(operation)
        if keys != expected:
            details = []
            if expected - keys:
                details.append(f"missing {sorted(expected - keys)}")
            if keys - expected:
                details.append(f"unexpected {sorted(keys - expected)}")
            raise ValueError(f"{label} has invalid fields ({'; '.join(details)})")
        doc_id = operation["doc_id"]
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"{label} doc_id must be a non-empty string")
        text = operation.get("text")
        if op != "delete" and (not isinstance(text, str) or not text):
            raise ValueError(f"{label} text must be a non-empty string")
        return op, doc_id, text

    def apply(self, operations: list[dict]) -> list[int | bool]:
        """Run add/update/delete operations in order and commit them as one batch.

        Returns one result per operation: the post-operation document count for
        ``add``, the unchanged count for ``update``, and a boolean for ``delete``
        (``False`` for an unknown id). Every element is validated and applied in
        sequence against the same in-memory snapshot, so later operations see the
        effects of earlier ones; the snapshot is written exactly once. Any error
        leaves the persisted state untouched. Structural problems and, within the
        batch, adds of existing documents or updates of missing ones raise
        ``ValueError``.
        """
        if not isinstance(operations, list):
            raise ValueError("operations must be a list")
        if not operations:
            raise ValueError("operations must not be empty")
        document = self._read()
        results: list[int | bool] = []
        for index, operation in enumerate(operations):
            op, doc_id, text = self._parse_operation(operation, index)
            if op == "add":
                results.append(self._add_document(document, doc_id, text))
            elif op == "update":
                try:
                    results.append(self._update_document(document, doc_id, text))
                except KeyError:
                    raise ValueError(f"operation {index} updates missing document {doc_id!r}") from None
            else:
                results.append(self._delete_document(document, doc_id))
        self._write(document)
        return results

    def query(self, terms: list[str]) -> list[dict]:
        _require_terms(terms, "query")
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
            positions = _sequence_starts(sequence, tokens)
            if positions:
                results.append({"id": doc_id, "positions": positions})
        return results

    def near(self, left: str, right: str, max_gap: int) -> list[dict]:
        """Find documents where the token sequences of ``left`` and ``right`` are close.

        Either fragment may occur first; each must be a consecutive, non-overlapping
        match. ``max_gap`` bounds the number of tokens strictly between the two
        fragments. Returns ``[{"id": doc_id, "occurrences": [[left_start, right_start], ...]}, ...]``
        sorted by document id; each pair gives the 0-based starts of the left and
        right fragment's first token (fields stay fixed even when right comes first),
        sorted by ``(left_start, right_start)`` with repeats kept.
        """
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError("near fragments must be strings")
        if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 0:
            raise ValueError("max_gap must be a non-negative integer")
        left_tokens = tokenize(left)
        right_tokens = tokenize(right)
        if not left_tokens or not right_tokens:
            raise ValueError("near fragments need at least one token each")
        documents = self._read()["documents"]
        results = []
        for doc_id in sorted(documents):
            sequence = tokenize(documents[doc_id])
            occurrences = []
            for left_start in _sequence_starts(sequence, left_tokens):
                left_end = left_start + len(left_tokens)
                for right_start in _sequence_starts(sequence, right_tokens):
                    right_end = right_start + len(right_tokens)
                    if right_start >= left_end and right_start - left_end <= max_gap:
                        occurrences.append([left_start, right_start])
                    elif left_start >= right_end and left_start - right_end <= max_gap:
                        occurrences.append([left_start, right_start])
            if occurrences:
                occurrences.sort()
                results.append({"id": doc_id, "occurrences": occurrences})
        return results

    def rank(self, terms: list[str]) -> list[dict]:
        """Score documents against ``terms`` with a tf-idf sum, best first."""
        _require_terms(terms, "rank")
        wanted = sorted({term.lower() for term in terms})
        snapshot = self._read()
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
