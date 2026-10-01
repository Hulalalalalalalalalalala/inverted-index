"""An inverted index over tokenised documents, persisted as one JSON snapshot.

The snapshot is committed atomically: a writer serialises the full state to a
temporary file in the same directory, fsyncs it and replaces ``index.json`` in
one ``os.replace`` call, so an interrupted write can only ever leave the old
snapshot or the new snapshot in place -- never a partial one.  Every read goes
through structural validation and a consistency check that recomputes the
postings from the stored documents with the public tokeniser.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

__all__ = ["InvertedIndex", "SnapshotError", "tokenize"]

INDEX_FILE = "index.json"
TOKEN = re.compile(r"[0-9A-Za-z_]+")


def tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics and lower-case; no stemming and no stop words."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return [match.group(0).lower() for match in TOKEN.finditer(text)]


class SnapshotError(ValueError):
    """The persisted snapshot is missing, unreadable, malformed or inconsistent."""


def _require_doc_id(doc_id: object) -> str:
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("doc_id must be a non-empty string")
    return doc_id


def _require_terms(terms: object) -> list[str]:
    if not isinstance(terms, (list, tuple)) or not terms:
        raise ValueError("at least one term is required")
    wanted: list[str] = []
    for term in terms:
        if not isinstance(term, str) or not term:
            raise ValueError("terms must be non-empty strings")
        wanted.append(term.lower())
    return wanted


class InvertedIndex:
    """A single-process inverted index rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.directory = Path(root)
        self.path = self.directory / INDEX_FILE

    def init(self) -> None:
        """Create a valid empty index, rebuilding over a missing or broken snapshot."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write({"documents": {}, "postings": {}})

    def _snapshot(self) -> dict:
        """Read the persisted snapshot, validating shape, types and consistency.

        Recomputes the postings from every stored document using the public
        tokeniser and demands an exact match, so no corrupt or inconsistent
        snapshot can ever feed a read computation.
        """
        if not self.path.is_file():
            raise FileNotFoundError(f"no index at {self.path}; run init first")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SnapshotError(f"invalid index at {self.path}: {error}") from error
        except UnicodeDecodeError as error:
            raise SnapshotError(f"invalid index at {self.path}: {error}") from error
        if not isinstance(data, dict) or set(data) != {"documents", "postings"}:
            raise SnapshotError(
                "index snapshot must be an object with 'documents' and 'postings'"
            )
        documents, postings = data["documents"], data["postings"]
        if not isinstance(documents, dict) or not isinstance(postings, dict):
            raise SnapshotError("'documents' and 'postings' must be JSON objects")
        for doc_id, text in documents.items():
            if not isinstance(doc_id, str) or not isinstance(text, str):
                raise SnapshotError("document ids and texts must be strings")
        for term, entries in postings.items():
            if not isinstance(term, str) or not isinstance(entries, dict):
                raise SnapshotError("postings must map string terms to objects")
            for doc_id, frequency in entries.items():
                if not isinstance(doc_id, str):
                    raise SnapshotError("posting document ids must be strings")
                if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 0:
                    raise SnapshotError("posting frequencies must be non-negative integers")
        expected: dict[str, dict[str, int]] = {}
        for doc_id, text in documents.items():
            frequencies: dict[str, int] = {}
            for token in tokenize(text):
                frequencies[token] = frequencies.get(token, 0) + 1
            for token, frequency in frequencies.items():
                expected.setdefault(token, {})[doc_id] = frequency
        if postings != expected:
            raise SnapshotError("postings are inconsistent with the stored documents")
        return data

    def _write(self, snapshot: dict) -> None:
        """Commit the full snapshot atomically; failures leave the old file intact."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._purge_staging()
        payload = json.dumps(snapshot, sort_keys=True, indent=2)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{INDEX_FILE}.", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
            self._sync_directory()
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name)
            raise

    def _purge_staging(self) -> None:
        """Remove staging files left behind by a hard-killed earlier write.

        Writers are single-process (see README), so every ``.index.json.*.tmp``
        can only be a remnant of a dead writer -- never a live rename source.
        """
        pattern = f".{INDEX_FILE}.*.tmp"
        for remnant in self.directory.glob(pattern):
            with contextlib.suppress(OSError):
                remnant.unlink()

    def _sync_directory(self) -> None:
        """Best-effort fsync of the directory so the rename is durable."""
        try:
            descriptor = os.open(self.directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def add(self, doc_id: str, text: str) -> int:
        doc_id = _require_doc_id(doc_id)
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        snapshot = self._snapshot()
        if doc_id in snapshot["documents"]:
            raise ValueError(f"document {doc_id!r} already exists")
        snapshot["documents"][doc_id] = text
        frequencies: dict[str, int] = {}
        for token in tokenize(text):
            frequencies[token] = frequencies.get(token, 0) + 1
        for token, frequency in frequencies.items():
            snapshot["postings"].setdefault(token, {})[doc_id] = frequency
        self._write(snapshot)
        return len(snapshot["documents"])

    def get(self, doc_id: str) -> str | None:
        doc_id = _require_doc_id(doc_id)
        return self._snapshot()["documents"].get(doc_id)

    def delete(self, doc_id: str) -> bool:
        doc_id = _require_doc_id(doc_id)
        snapshot = self._snapshot()
        if doc_id not in snapshot["documents"]:
            return False
        del snapshot["documents"][doc_id]
        for token, entries in list(snapshot["postings"].items()):
            entries.pop(doc_id, None)
            if not entries:
                del snapshot["postings"][token]
        self._write(snapshot)
        return True

    def query(self, terms: list[str]) -> list[dict]:
        wanted = _require_terms(terms)
        postings = self._snapshot()["postings"]
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
        documents = self._snapshot()["documents"]
        results = []
        for doc_id in sorted(documents):
            sequence = tokenize(documents[doc_id])
            positions = [start for start in range(len(sequence) - len(tokens) + 1)
                         if sequence[start:start + len(tokens)] == tokens]
            if positions:
                results.append({"id": doc_id, "positions": positions})
        return results

    def rank(self, terms: list[str]) -> list[dict]:
        """Score documents against ``terms`` with a tf-idf sum, best first."""
        wanted = sorted(set(_require_terms(terms)))
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
        return sorted(self._snapshot()["postings"])

    def stats(self) -> dict:
        snapshot = self._snapshot()
        return {"documents": len(snapshot["documents"]), "terms": len(snapshot["postings"]),
                "postings": sum(len(entries) for entries in snapshot["postings"].values())}

    def reload(self) -> None:
        self._snapshot()
