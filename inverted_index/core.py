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
#: An unquoted search fragment: a plain term or a wildcard pattern.
SEARCH_WORD = re.compile(r"[0-9A-Za-z_*?]+")
#: A lower-cased wildcard pattern consists solely of term characters and wildcards.
PATTERN_CHARS = re.compile(r"[0-9a-z_*?]+")


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


def _compile_pattern(pattern: object, *, require_literal: bool = False) -> re.Pattern:
    """Validate a wildcard pattern and return an anchored, full-match regex.

    ``*`` matches zero or more characters and ``?`` exactly one; anything else
    must be a term character (ASCII letter, digit or underscore), matched
    literally after lower-casing. Empty, non-string and illegal-character
    patterns raise ``ValueError``. With ``require_literal``, a pattern without
    at least one literal character is rejected too.
    """
    if not isinstance(pattern, str):
        raise ValueError("pattern must be a string")
    lowered = pattern.lower()
    if not lowered or not PATTERN_CHARS.fullmatch(lowered):
        raise ValueError("pattern must be a non-empty string of term characters, '*' and '?'")
    if require_literal and not any(char not in "*?" for char in lowered):
        raise ValueError("wildcard pattern needs at least one literal character")
    pieces = ["^"]
    for char in lowered:
        if char == "*":
            pieces.append(".*")
        elif char == "?":
            pieces.append(".")
        else:
            pieces.append(re.escape(char))
    pieces.append(r"\Z")
    return re.compile("".join(pieces), re.DOTALL)


_SEARCH_OPERATORS = ("AND", "OR", "NOT")


def _lex_search(expression: str) -> list[tuple[str, object]]:
    """Split a search expression into ``(kind, value)`` tokens.

    Kinds are ``TERM`` (value the lower-cased term), ``PHRASE`` (value the
    tokenised phrase), ``LPAREN``, ``RPAREN`` and the operators themselves.
    Operators are case-sensitive; anything outside terms, quotes, parentheses
    and whitespace is rejected.
    """
    tokens: list[tuple[str, object]] = []
    position = 0
    while position < len(expression):
        char = expression[position]
        if char.isspace():
            position += 1
        elif char == "(":
            tokens.append(("LPAREN", char))
            position += 1
        elif char == ")":
            tokens.append(("RPAREN", char))
            position += 1
        elif char == '"':
            end = expression.find('"', position + 1)
            if end == -1:
                raise ValueError("unclosed double quote in search expression")
            phrase_tokens = tokenize(expression[position + 1:end])
            if not phrase_tokens:
                raise ValueError("quoted phrase needs at least one token")
            tokens.append(("PHRASE", phrase_tokens))
            position = end + 1
        else:
            match = SEARCH_WORD.match(expression, position)
            if match is None:
                raise ValueError(f"unsupported character {char!r} in search expression")
            word = match.group(0)
            if "*" not in word and "?" not in word:
                if word in _SEARCH_OPERATORS:
                    tokens.append((word, word))
                else:
                    tokens.append(("TERM", word.lower()))
            else:
                # Operators are plain words; a wildcard fragment can never be one.
                if not any(each not in "*?" for each in word):
                    raise ValueError("wildcard term needs at least one literal character")
                tokens.append(("WILDCARD", _compile_pattern(word)))
            position = match.end()
    return tokens


def _parse_search(tokens: list[tuple[str, object]]):
    """Parse lexed tokens into a tree; NOT binds tightest, then AND, then OR."""
    if not tokens:
        raise ValueError("search expression must not be empty")
    node, position = _parse_search_or(tokens, 0)
    if position != len(tokens):
        kind = tokens[position][0]
        if kind == "RPAREN":
            raise ValueError("unbalanced parentheses in search expression")
        raise ValueError("missing AND or OR between operands in search expression")
    return node


def _parse_search_or(tokens, position):
    node, position = _parse_search_and(tokens, position)
    while position < len(tokens) and tokens[position][0] == "OR":
        right, position = _parse_search_and(tokens, position + 1)
        node = ("OR", node, right)
    return node, position


def _parse_search_and(tokens, position):
    node, position = _parse_search_not(tokens, position)
    while position < len(tokens) and tokens[position][0] == "AND":
        right, position = _parse_search_not(tokens, position + 1)
        node = ("AND", node, right)
    return node, position


def _parse_search_not(tokens, position):
    if position < len(tokens) and tokens[position][0] == "NOT":
        operand, position = _parse_search_not(tokens, position + 1)
        return ("NOT", operand), position
    return _parse_search_primary(tokens, position)


def _parse_search_primary(tokens, position):
    if position >= len(tokens):
        raise ValueError("missing operand in search expression")
    kind, value = tokens[position]
    if kind == "TERM":
        return ("TERM", value), position + 1
    if kind == "WILDCARD":
        return ("WILDCARD", value), position + 1
    if kind == "PHRASE":
        return ("PHRASE", value), position + 1
    if kind == "LPAREN":
        node, position = _parse_search_or(tokens, position + 1)
        if position >= len(tokens) or tokens[position][0] != "RPAREN":
            raise ValueError("unbalanced parentheses in search expression")
        return node, position + 1
    if kind == "RPAREN":
        raise ValueError("unbalanced parentheses in search expression")
    raise ValueError(f"missing operand in search expression before {kind}")


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
    def _index_text(document: dict, doc_id: str, text: str) -> None:
        for token, frequency in Counter(tokenize(text)).items():
            document["postings"].setdefault(token, {})[doc_id] = frequency

    @staticmethod
    def _drop_postings(document: dict, doc_id: str) -> None:
        postings = document["postings"]
        for token in list(postings):
            postings[token].pop(doc_id, None)
            if not postings[token]:
                del postings[token]

    def add(self, doc_id: str, text: str) -> int:
        _require_non_empty_string(doc_id, "doc_id")
        _require_non_empty_string(text, "text")
        document = self._read()
        if doc_id in document["documents"]:
            raise ValueError(f"document {doc_id!r} already exists")
        document["documents"][doc_id] = text
        self._index_text(document, doc_id, text)
        self._write(document)
        return len(document["documents"])

    def update(self, doc_id: str, text: str) -> int:
        """Replace ``doc_id``'s text and rebuild its postings, frequencies and positions.

        The document count is unchanged; the new total is returned.
        """
        _require_non_empty_string(doc_id, "doc_id")
        _require_non_empty_string(text, "text")
        document = self._read()
        if doc_id not in document["documents"]:
            raise KeyError(doc_id)
        document["documents"][doc_id] = text
        self._drop_postings(document, doc_id)
        self._index_text(document, doc_id, text)
        self._write(document)
        return len(document["documents"])

    def get(self, doc_id: str) -> str | None:
        _require_non_empty_string(doc_id, "doc_id")
        return self._read()["documents"].get(doc_id)

    def delete(self, doc_id: str) -> bool:
        _require_non_empty_string(doc_id, "doc_id")
        document = self._read()
        if doc_id not in document["documents"]:
            return False
        del document["documents"][doc_id]
        self._drop_postings(document, doc_id)
        self._write(document)
        return True

    def apply(self, operations: list[dict]) -> list[int | bool]:
        """Apply a non-empty batch of add/update/delete operations atomically.

        Operations run in order on one in-memory snapshot, so later operations
        see earlier ones (e.g. delete then add the same id), and the snapshot is
        rewritten exactly once: any failure leaves the committed state untouched.
        Each result mirrors the matching single-document call -- add and update
        return the document total at that point, delete returns whether the id
        existed. Within a batch an unknown update id is a ``ValueError`` (the
        single ``update`` still raises ``KeyError``); deleting an unknown id
        still returns ``False``.
        """
        if not isinstance(operations, list):
            raise ValueError("operations must be a list of operation objects")
        if not operations:
            raise ValueError("operations must contain at least one operation")
        document = self._read()
        results: list[int | bool] = []
        for position, operation in enumerate(operations):
            label = f"operation {position}"
            if not isinstance(operation, dict):
                raise ValueError(f"{label}: each operation must be an object")
            op = operation.get("op")
            if op not in ("add", "update", "delete"):
                raise ValueError(f"{label}: op must be one of add, update, delete")
            allowed = {"op", "doc_id", "text"} if op != "delete" else {"op", "doc_id"}
            if set(operation) != allowed:
                raise ValueError(f"{label}: {op} allows exactly the fields {sorted(allowed)}")
            doc_id = operation["doc_id"]
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"{label}: doc_id must be a non-empty string")
            if op != "delete":
                text = operation["text"]
                if not isinstance(text, str) or not text:
                    raise ValueError(f"{label}: text must be a non-empty string")
            if op == "add":
                if doc_id in document["documents"]:
                    raise ValueError(f"{label}: document {doc_id!r} already exists")
                document["documents"][doc_id] = text
                self._index_text(document, doc_id, text)
                results.append(len(document["documents"]))
            elif op == "update":
                if doc_id not in document["documents"]:
                    raise ValueError(f"{label}: document {doc_id!r} does not exist")
                document["documents"][doc_id] = text
                self._drop_postings(document, doc_id)
                self._index_text(document, doc_id, text)
                results.append(len(document["documents"]))
            else:
                existed = doc_id in document["documents"]
                if existed:
                    del document["documents"][doc_id]
                    self._drop_postings(document, doc_id)
                results.append(existed)
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

    def expand(self, pattern: str) -> list[str]:
        """Return every indexed term matching the wildcard pattern, sorted.

        ``*`` matches zero or more characters and ``?`` exactly one; the
        pattern follows the term rules otherwise (ASCII letters, digits and
        underscores, lower-cased). The result is sorted by Unicode code point
        and de-duplicated; an unmatched pattern gives an empty list. Empty,
        non-string or illegal-character patterns raise ``ValueError``.
        """
        matcher = _compile_pattern(pattern)
        postings = self._read()["postings"]
        return sorted(term for term in postings if matcher.match(term))

    def search(self, expression: str) -> list[str]:
        """Evaluate a boolean search expression and return matching document ids.

        The expression combines plain terms, double-quoted phrases, parentheses
        and the case-sensitive operators AND, OR, NOT; precedence is NOT, then
        AND, then OR, with left-to-right associativity at each level. Terms hit
        documents containing the term, phrases hit documents containing the
        consecutive token sequence, AND/OR intersect/union the two sides and
        NOT complements against every document in the index. Returns the
        matching document ids sorted ascending; the index is only read, never
        written. Malformed expressions raise ``ValueError``.
        """
        if not isinstance(expression, str):
            raise ValueError("expression must be a string")
        tree = _parse_search(_lex_search(expression))
        snapshot = self._read()
        documents = snapshot["documents"]
        postings = snapshot["postings"]
        universe = set(documents)

        def evaluate(node) -> set[str]:
            kind = node[0]
            if kind == "TERM":
                return set(postings.get(node[1], {}))
            if kind == "WILDCARD":
                matcher = node[1]
                hits: set[str] = set()
                for term, entries in postings.items():
                    if matcher.match(term):
                        hits.update(entries)
                return hits
            if kind == "PHRASE":
                return {doc_id for doc_id, text in documents.items()
                        if _sequence_starts(tokenize(text), node[1])}
            if kind == "AND":
                return evaluate(node[1]) & evaluate(node[2])
            if kind == "OR":
                return evaluate(node[1]) | evaluate(node[2])
            return universe - evaluate(node[1])

        return sorted(evaluate(tree))

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
