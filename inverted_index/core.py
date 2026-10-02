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
FRAGMENT = re.compile(r"[0-9A-Za-z_*?]+")
_PATTERN_CHARS = re.compile(r"[0-9A-Za-z_*?]+\Z")


class SnapshotError(ValueError):
    """The persisted snapshot fails structural or consistency validation."""


def tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics and lower-case; no stemming and no stop words."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return [match.group(0).lower() for match in TOKEN.finditer(text)]


def _compile_pattern(pattern: str):
    """Validate a wildcard pattern and return a full-match predicate over terms.

    The pattern is lower-cased like any term; only ASCII letters, digits,
    underscore and the wildcards ``*`` (zero or more characters) and ``?``
    (exactly one character) are allowed -- anything else, an empty pattern or
    a non-string raises ``ValueError``.
    """
    if not isinstance(pattern, str):
        raise ValueError("pattern must be a string")
    if not pattern:
        raise ValueError("pattern must not be empty")
    if not _PATTERN_CHARS.match(pattern):
        raise ValueError(f"pattern contains unsupported characters: {pattern!r}")
    lowered = pattern.lower()
    regex = "".join(".*" if char == "*" else "." if char == "?" else re.escape(char)
                    for char in lowered)
    return re.compile(regex + r"\Z").match


def _require_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_terms(terms: object, name: str) -> None:
    if isinstance(terms, str) or not terms:
        raise ValueError(f"{name} needs at least one term")
    if any(not isinstance(term, str) or not term for term in terms):
        raise ValueError(f"{name} needs non-empty string terms")


def _validate_bm25_k1(k1: object) -> None:
    """Validate BM25 ``k1``: int/float, booleans rejected, positive and finite.

    Floats must be finite positive numbers; arbitrarily large positive
    Python ints are also accepted (they never round-trip through float here,
    so no ``OverflowError`` escapes).
    """
    if isinstance(k1, bool) or not isinstance(k1, (int, float)):
        raise ValueError("k1 must be a finite positive number")
    if isinstance(k1, int):
        if k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
    elif not math.isfinite(k1) or k1 <= 0:
        raise ValueError("k1 must be a finite positive number")


def _validate_bm25_b(b: object) -> None:
    """Validate BM25 ``b``: int/float, booleans rejected, finite and in [0, 1].

    Only the integers 0 and 1 pass the integer path; out-of-range giant
    integers raise ``ValueError`` rather than overflowing inside ``float``.
    """
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise ValueError("b must be a finite number within [0, 1]")
    if isinstance(b, int):
        if not 0 <= b <= 1:
            raise ValueError("b must be a finite number within [0, 1]")
    elif not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("b must be a finite number within [0, 1]")


def _bm25_term_ratio(tf: int, k1: float, length_norm: float) -> float:
    """Return ``(k1 + 1) / (tf + k1 * length_norm)`` without float overflow.

    ``k1`` is a non-negative float; a value beyond the float range (from a
    huge Python int argument) is mapped to infinity by the caller and the
    ratio becomes ``1 / length_norm``. For finite ``k1 >= 1`` numerator and
    denominator are divided by ``k1`` first (both stay at most
    ``max(2, tf)``), so a large ``k1`` can never make the denominator
    infinity; for ``0 < k1 < 1`` the direct form is used -- the numerator is
    then at most 2 and ``k1 * length_norm`` stays at the length norm's
    scale, so tiny subnormal ``k1`` keeps its exact ordinary-form score.
    """
    if math.isinf(k1):
        return 1.0 / length_norm
    if k1 >= 1.0:
        inv_k1 = 1.0 / k1
        return (1.0 + inv_k1) / (length_norm + tf * inv_k1)
    return (1.0 + k1) / (tf + k1 * length_norm)


def _sequence_starts(sequence: list[str], fragment: list[str]) -> list[int]:
    """All 0-based starts where ``fragment`` occurs consecutively in ``sequence``."""
    length = len(fragment)
    return [start for start in range(len(sequence) - length + 1)
            if sequence[start:start + length] == fragment]


_SEARCH_OPERATORS = ("AND", "OR", "NOT")
_NEAR_NAME = "NEAR"
_FUZZY_NAME = "FUZZY"
_ASCII_DIGITS = frozenset("0123456789")
_NEAR_MAX_GAP = 2147483647
_FUZZY_MAX_DISTANCE = 2
_FUZZY_TERM_CHARS = re.compile(r"[0-9A-Za-z_]+\Z")


def _levenshtein(left: str, right: str, limit: int) -> int:
    """Levenshtein distance between two strings, capped at ``limit + 1``.

    Insertions, deletions and substitutions of a single character each count
    as one operation; swapping adjacent characters is not an operation.
    """
    if left == right:
        return 0
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for row, char in enumerate(left, 1):
        current = [row]
        for column, other in enumerate(right, 1):
            current.append(min(previous[column] + 1, current[column - 1] + 1,
                               previous[column - 1] + (char != other)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _validate_fuzzy_term(term: object) -> str:
    """Validate a fuzzy term and return it lower-cased.

    Only a non-empty string of ASCII letters, digits and underscores is
    accepted; the term is compared as-is (lower-cased), never tokenised,
    stripped or wildcard-expanded.
    """
    if not isinstance(term, str):
        raise ValueError("term must be a string")
    if not term:
        raise ValueError("term must not be empty")
    if not _FUZZY_TERM_CHARS.match(term):
        raise ValueError(f"term contains unsupported characters: {term!r}")
    return term.lower()


def _validate_max_distance(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or not 0 <= value <= _FUZZY_MAX_DISTANCE:
        raise ValueError("max_distance must be an integer within 0..2")
    return value


def _skip_spaces(expression: str, position: int) -> int:
    while position < len(expression) and expression[position].isspace():
        position += 1
    return position


def _read_near_fragment(expression: str, position: int) -> tuple[list[str], int]:
    """Read one double-quoted NEAR fragment starting at ``position``.

    Commas and parentheses inside the quotes are plain text; double quotes
    cannot be embedded or escaped. The raw text is tokenised with the usual
    :func:`tokenize` rules and must yield at least one token.
    """
    if position >= len(expression) or expression[position] != '"':
        raise ValueError("NEAR fragments must be double-quoted")
    end = expression.find('"', position + 1)
    if end == -1:
        raise ValueError("unclosed double quote in search expression")
    fragment_tokens = tokenize(expression[position + 1:end])
    if not fragment_tokens:
        raise ValueError("NEAR fragments need at least one token each")
    return fragment_tokens, end + 1


def _lex_near_call(expression: str, position: int) -> tuple[tuple[str, object], int]:
    """Lex a ``NEAR("left", "right", distance)`` call; ``position`` is at ``(``."""
    position = _skip_spaces(expression, position + 1)
    left_tokens, position = _read_near_fragment(expression, position)
    position = _skip_spaces(expression, position)
    if position >= len(expression) or expression[position] != ",":
        raise ValueError("missing comma between NEAR arguments")
    position = _skip_spaces(expression, position + 1)
    right_tokens, position = _read_near_fragment(expression, position)
    position = _skip_spaces(expression, position)
    if position >= len(expression) or expression[position] != ",":
        raise ValueError("NEAR needs exactly three arguments")
    position = _skip_spaces(expression, position + 1)
    digits_start = position
    while position < len(expression) and expression[position] in _ASCII_DIGITS:
        position += 1
    if position == digits_start:
        raise ValueError("NEAR distance must be an ASCII decimal integer")
    distance_text = expression[digits_start:position]
    distance = int(distance_text)
    if distance > _NEAR_MAX_GAP:
        raise ValueError("NEAR distance must be within 0..2147483647")
    position = _skip_spaces(expression, position)
    if position >= len(expression) or expression[position] != ")":
        raise ValueError("unbalanced parentheses in search expression")
    return ("NEAR", (left_tokens, right_tokens, distance)), position + 1


def _lex_fuzzy_call(expression: str, position: int) -> tuple[tuple[str, object], int]:
    """Lex a ``FUZZY("term", distance)`` call; ``position`` is at ``(``.

    The term must be double-quoted and follow the same rules as
    :meth:`InvertedIndex.fuzzy` (non-empty ASCII letters, digits, underscore;
    compared lower-cased, wildcards not special); the distance is an ASCII
    decimal integer from 0 to 2, leading zeros allowed. Exactly two arguments.
    """
    position = _skip_spaces(expression, position + 1)
    if position >= len(expression) or expression[position] != '"':
        raise ValueError("FUZZY term must be double-quoted")
    end = expression.find('"', position + 1)
    if end == -1:
        raise ValueError("unclosed double quote in search expression")
    term = _validate_fuzzy_term(expression[position + 1:end])
    position = _skip_spaces(expression, end + 1)
    if position >= len(expression) or expression[position] != ",":
        raise ValueError("FUZZY needs exactly two arguments")
    position = _skip_spaces(expression, position + 1)
    digits_start = position
    while position < len(expression) and expression[position] in _ASCII_DIGITS:
        position += 1
    if position == digits_start:
        raise ValueError("FUZZY distance must be an ASCII decimal integer")
    distance = int(expression[digits_start:position])
    if distance > _FUZZY_MAX_DISTANCE:
        raise ValueError("FUZZY distance must be within 0..2")
    position = _skip_spaces(expression, position)
    if position < len(expression) and expression[position] == ",":
        raise ValueError("FUZZY needs exactly two arguments")
    if position >= len(expression) or expression[position] != ")":
        raise ValueError("unbalanced parentheses in search expression")
    return ("FUZZY", (term, distance)), position + 1


def _lex_search(expression: str) -> list[tuple[str, object]]:
    """Split a search expression into ``(kind, value)`` tokens.

    Kinds are ``TERM`` (value the lower-cased term), ``WILDCARD`` (value the
    lower-cased pattern), ``PHRASE`` (value the tokenised phrase), ``NEAR``
    (value ``(left_tokens, right_tokens, max_gap)``), ``FUZZY`` (value
    ``(term, max_distance)``), ``LPAREN``, ``RPAREN``
    and the operators themselves. Operators are case-sensitive; a bare
    ``NEAR`` or ``FUZZY`` not directly followed by ``(`` is an ordinary term.
    Anything outside terms, wildcards, quotes, parentheses and whitespace is
    rejected. A wildcard fragment needs at least one literal character.
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
            match = FRAGMENT.match(expression, position)
            if match is None:
                raise ValueError(f"unsupported character {char!r} in search expression")
            word = match.group(0)
            if word == _NEAR_NAME:
                probe = _skip_spaces(expression, match.end())
                if probe < len(expression) and expression[probe] == "(":
                    token, position = _lex_near_call(expression, probe)
                    tokens.append(token)
                    continue
            if word == _FUZZY_NAME:
                probe = _skip_spaces(expression, match.end())
                if probe < len(expression) and expression[probe] == "(":
                    token, position = _lex_fuzzy_call(expression, probe)
                    tokens.append(token)
                    continue
            if word in _SEARCH_OPERATORS:
                tokens.append((word, word))
            elif "*" in word or "?" in word:
                if all(char in "*?" for char in word):
                    raise ValueError("wildcard term needs at least one literal character")
                tokens.append(("WILDCARD", word.lower()))
            else:
                tokens.append(("TERM", word.lower()))
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
    if kind == "NEAR":
        return ("NEAR", value), position + 1
    if kind == "FUZZY":
        return ("FUZZY", value), position + 1
    if kind == "LPAREN":
        node, position = _parse_search_or(tokens, position + 1)
        if position >= len(tokens) or tokens[position][0] != "RPAREN":
            raise ValueError("unbalanced parentheses in search expression")
        return node, position + 1
    if kind == "RPAREN":
        raise ValueError("unbalanced parentheses in search expression")
    raise ValueError(f"missing operand in search expression before {kind}")


def _near_match(sequence: list[str], left: list[str], right: list[str], max_gap: int) -> bool:
    """Whether ``left`` and ``right`` each occur once, non-overlapping, in either order.

    Uses the same gap rule as :meth:`InvertedIndex.near`: the number of tokens
    strictly between the two fragments is at most ``max_gap``.
    """
    left_length, right_length = len(left), len(right)
    for left_start in _sequence_starts(sequence, left):
        left_end = left_start + left_length
        for right_start in _sequence_starts(sequence, right):
            right_end = right_start + right_length
            if right_start >= left_end and right_start - left_end <= max_gap:
                return True
            if left_start >= right_end and left_start - right_end <= max_gap:
                return True
    return False


def _evaluate_search(tree, documents: dict[str, str], postings: dict[str, dict]) -> set[str]:
    """Evaluate a parsed search tree against a snapshot, returning matching doc ids."""
    universe = set(documents)

    def evaluate(node) -> set[str]:
        kind = node[0]
        if kind == "TERM":
            return set(postings.get(node[1], {}))
        if kind == "WILDCARD":
            matcher = _compile_pattern(node[1])
            return {doc_id for term, entries in postings.items() if matcher(term)
                    for doc_id in entries}
        if kind == "PHRASE":
            return {doc_id for doc_id, text in documents.items()
                    if _sequence_starts(tokenize(text), node[1])}
        if kind == "NEAR":
            left, right, max_gap = node[1]
            return {doc_id for doc_id, text in documents.items()
                    if _near_match(tokenize(text), left, right, max_gap)}
        if kind == "FUZZY":
            term, max_distance = node[1]
            return {doc_id for candidate, entries in postings.items()
                    if _levenshtein(term, candidate, max_distance) <= max_distance
                    for doc_id in entries}
        if kind == "AND":
            return evaluate(node[1]) & evaluate(node[2])
        if kind == "OR":
            return evaluate(node[1]) | evaluate(node[2])
        return universe - evaluate(node[1])

    return evaluate(tree)


def _merge_spans(spans: list[tuple[int, int]]) -> list[list[int]]:
    """Deduplicate half-open ``[start, end)`` spans and merge touching/overlapping ones.

    Spans are Unicode code point offsets; two spans that overlap or abut
    (``end == next_start``) become one span. Sorted by start ascending.
    """
    if not spans:
        return []
    ordered = sorted(set(spans))
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def _evaluate_highlight(tree, words: list[str], bounds: list[tuple[int, int]],
                        matchers: dict[str, object]) -> tuple[bool, list[tuple[int, int]]]:
    """Evaluate a parsed tree for one document, returning ``(matched, spans)``.

    ``words`` are the document's lower-cased tokens and ``bounds`` their
    half-open Unicode code point offsets. Matching is identical to
    :func:`_evaluate_search` (terms come from the same tokenisation), but a
    matched node also yields the spans that actually participate: every term
    or expanded-wildcard occurrence, each phrase match from the first token's
    start to the last token's end, both fragments of every NEAR pair that
    satisfies the gap and non-overlap rules, and every occurrence of every
    term within a FUZZY condition's edit distance. NOT only filters -- it never
    yields spans, so a double negation matches without restoring highlights.
    """

    def char_span(token_start: int, token_end: int) -> tuple[int, int]:
        return bounds[token_start][0], bounds[token_end - 1][1]

    def evaluate(node) -> tuple[bool, list[tuple[int, int]]]:
        kind = node[0]
        if kind == "TERM":
            term = node[1]
            spans = [bounds[index] for index, token in enumerate(words) if token == term]
            return bool(spans), spans
        if kind == "WILDCARD":
            pattern = node[1]
            matcher = matchers.get(pattern)
            if matcher is None:
                matcher = _compile_pattern(pattern)
                matchers[pattern] = matcher
            spans = [bounds[index] for index, token in enumerate(words) if matcher(token)]
            return bool(spans), spans
        if kind == "PHRASE":
            wanted = node[1]
            spans = [char_span(start, start + len(wanted))
                     for start in _sequence_starts(words, wanted)]
            return bool(spans), spans
        if kind == "NEAR":
            left, right, max_gap = node[1]
            spans: set[tuple[int, int]] = set()
            for left_start in _sequence_starts(words, left):
                left_end = left_start + len(left)
                for right_start in _sequence_starts(words, right):
                    right_end = right_start + len(right)
                    if right_start >= left_end and right_start - left_end <= max_gap:
                        pass
                    elif left_start >= right_end and left_start - right_end <= max_gap:
                        pass
                    else:
                        continue
                    spans.add(char_span(left_start, left_end))
                    spans.add(char_span(right_start, right_end))
            ordered = sorted(spans)
            return bool(ordered), ordered
        if kind == "FUZZY":
            term, max_distance = node[1]
            spans = [bounds[index] for index, token in enumerate(words)
                     if _levenshtein(term, token, max_distance) <= max_distance]
            return bool(spans), spans
        if kind == "AND":
            left_match, left_spans = evaluate(node[1])
            if not left_match:
                return False, []
            right_match, right_spans = evaluate(node[2])
            if not right_match:
                return False, []
            return True, left_spans + right_spans
        if kind == "OR":
            left_match, left_spans = evaluate(node[1])
            right_match, right_spans = evaluate(node[2])
            spans = list(left_spans) if left_match else []
            if right_match:
                spans.extend(right_spans)
            return left_match or right_match, spans
        matched, _ = evaluate(node[1])
        return not matched, []

    return evaluate(tree)


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

    def search(self, expression: str) -> list[str]:
        """Evaluate a boolean search expression and return matching document ids.

        The expression combines plain terms, wildcard terms, double-quoted
        phrases, ``NEAR("left", "right", distance)`` conditions, parentheses
        and the case-sensitive operators AND, OR, NOT; precedence is NOT, then
        AND, then OR, with left-to-right associativity at each level. An
        unquoted fragment containing ``*`` (zero or more characters) or ``?``
        (exactly one character) is a wildcard term that hits every document
        containing at least one expanded term; it needs at least one literal
        character. Terms hit documents containing the term, phrases hit
        documents containing the consecutive token sequence (the quoted text
        keeps ``tokenize`` semantics, wildcards are not special there), a NEAR
        condition hits documents where both quoted fragments (wildcards not
        expanded there) occur consecutively, non-overlapping, in either order
        with at most ``distance`` tokens strictly between them (an ASCII
        decimal integer from 0 to 2147483647, leading zeros allowed); a bare
        ``NEAR`` without a following parenthesis is still an ordinary term. A
        ``FUZZY("term", distance)`` condition hits every document containing
        at least one indexed term within Levenshtein distance ``distance`` of
        the double-quoted term (insertions, deletions and substitutions of a
        single character count one each; adjacent swaps are not an
        operation); the term follows the same rules as :meth:`fuzzy`
        (non-empty ASCII letters, digits, underscore, compared lower-cased,
        wildcards not special) and the distance is an ASCII decimal integer
        from 0 to 2, leading zeros allowed; the function name only accepts
        upper-case ``FUZZY``, whitespace is allowed between the name and the
        parenthesis and around the arguments, and a bare ``FUZZY`` without a
        following parenthesis is still an ordinary term.
        AND/OR intersect/union the two sides and NOT complements against every
        document in the index. Returns the matching document ids sorted
        ascending; the index is only read, never written. Malformed
        expressions raise ``ValueError``.
        """
        if not isinstance(expression, str):
            raise ValueError("expression must be a string")
        tree = _parse_search(_lex_search(expression))
        snapshot = self._read()
        return sorted(_evaluate_search(tree, snapshot["documents"], snapshot["postings"]))

    def highlight(self, expression: str) -> list[dict]:
        """Return matching documents with the text spans that actually matched.

        Uses the same syntax and filtering semantics as :meth:`search`, so the
        document set is identical; items are sorted by document id and each is
        ``{"id": doc_id, "text": text, "spans": [[start, end], ...]}`` with the
        full stored text and half-open ``[start, end)`` offsets counted in
        Unicode code points from zero. A plain term highlights every
        occurrence of the complete term, a wildcard every occurrence of every
        expanded term, a phrase the whole interval of each consecutive match
        (from the first token's start through the last token's end, keeping the
        punctuation and whitespace in between), and a NEAR condition both
        fragments of every fragment pair satisfying the existing gap and
        non-overlap rules -- never the gap between them -- and a FUZZY
        condition every occurrence of every expanded term within the edit
        distance. AND collects both
        sides, OR only the branches that hold in the document, NOT only
        filters (its interior yields no spans and double negation does not
        restore them), so a document matched solely through a negation still
        returns its text with empty spans. Collected spans are deduplicated and
        overlapping or abutting spans merged, sorted by start. The index is
        only read, never written. Non-strings, blank or otherwise malformed
        expressions raise ``ValueError`` before the store is touched, so the
        expression is validated even on an empty index.
        """
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a non-blank string")
        tree = _parse_search(_lex_search(expression))
        snapshot = self._read()
        documents = snapshot["documents"]
        postings = snapshot["postings"]
        matching = _evaluate_search(tree, documents, postings)
        results: list[dict] = []
        matchers: dict[str, object] = {}
        for doc_id in sorted(matching):
            text = documents[doc_id]
            matches = list(TOKEN.finditer(text))
            words = [match.group(0).lower() for match in matches]
            bounds = [(match.start(), match.end()) for match in matches]
            _, spans = _evaluate_highlight(tree, words, bounds, matchers)
            results.append({"id": doc_id, "text": text, "spans": _merge_spans(spans)})
        return results

    def snippets(self, expression: str, context: int = 20, max_fragments: int = 3) -> list[dict]:
        """Return matching documents with context fragments around the matched spans.

        Uses the same syntax and filtering semantics as :meth:`search` and the
        same highlight spans as :meth:`highlight`, so the matching document set
        and the highlighted intervals are identical; items are sorted by
        document id and each is ``{"id": doc_id, "fragments": [...]}``. Every
        highlight span is widened by up to ``context`` Unicode code points on
        each side, clipped to the text bounds; widened windows that overlap or
        abut merge (transitively), and the first ``max_fragments`` windows by
        start are kept. Each fragment is ``{"start", "end", "text", "spans"}``:
        ``start``/``end`` the window's half-open offsets in the stored text,
        ``text`` the corresponding slice (original case, punctuation and
        whitespace kept, nothing inserted), and ``spans`` every highlight span
        intersecting the window, clipped to it, shifted to window-relative
        half-open offsets and deduplicated/merged as in :meth:`highlight`. A
        document matched without any highlight span (only through negation)
        yields a single fragment covering the first ``2 * context + 1`` code
        points with empty ``spans``; an empty stored text yields no fragments.
        A valid expression with no hits returns an empty array; the whole call
        reads one snapshot and never writes. ``context`` must be a
        non-negative integer and ``max_fragments`` a positive integer (booleans
        rejected); bad parameters, a non-string or blank expression or
        malformed syntax raise ``ValueError`` before the store is touched, even
        on an empty index.
        """
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a non-blank string")
        if isinstance(context, bool) or not isinstance(context, int) or context < 0:
            raise ValueError("context must be a non-negative integer")
        if isinstance(max_fragments, bool) or not isinstance(max_fragments, int) or max_fragments < 1:
            raise ValueError("max_fragments must be a positive integer")
        tree = _parse_search(_lex_search(expression))
        snapshot = self._read()
        documents = snapshot["documents"]
        postings = snapshot["postings"]
        matching = _evaluate_search(tree, documents, postings)
        results: list[dict] = []
        matchers: dict[str, object] = {}
        for doc_id in sorted(matching):
            text = documents[doc_id]
            matches = list(TOKEN.finditer(text))
            words = [match.group(0).lower() for match in matches]
            bounds = [(match.start(), match.end()) for match in matches]
            _, spans = _evaluate_highlight(tree, words, bounds, matchers)
            merged = _merge_spans(spans)
            fragments: list[dict] = []
            if merged:
                windows = _merge_spans([(max(0, start - context), min(len(text), end + context))
                                        for start, end in merged])
                for window_start, window_end in windows[:max_fragments]:
                    local = _merge_spans([(max(start, window_start) - window_start,
                                           min(end, window_end) - window_start)
                                          for start, end in merged
                                          if start < window_end and end > window_start])
                    fragments.append({"start": window_start, "end": window_end,
                                      "text": text[window_start:window_end], "spans": local})
            elif text:
                end = min(len(text), 2 * context + 1)
                fragments.append({"start": 0, "end": end, "text": text[:end], "spans": []})
            results.append({"id": doc_id, "fragments": fragments})
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

    def bm25(self, terms: list[str], expression: str | None = None,
             k1: float = 1.2, b: float = 0.75) -> list[dict]:
        """Score documents against ``terms`` with BM25, best first.

        Terms are only lower-cased and de-duplicated -- never tokenised or
        wildcard-expanded. ``N`` is every document in the index (including
        documents without tokens), ``dl`` a document's token count,
        ``avgdl`` the mean ``dl`` over all documents, ``df`` the number of
        documents containing a term and ``tf`` its frequency in a document.
        Each matched term contributes
        ``ln(1 + (N - df + 0.5) / (df + 0.5)) * tf * (k1 + 1)
        / (tf + k1 * (1 - b + b * dl / avgdl))``; statistics always cover the
        whole index, while ``expression``, using the full :meth:`search`
        syntax (``None`` means no filter), only restricts which documents are
        returned. Every returned document matches at least one term and the
        filter; items are ``{"id", "matched", "score"}`` with ``matched`` the
        number of distinct matched terms, sorted by rounded score descending
        then id ascending. Scores are always finite: arithmetic is scaled so
        even ``k1`` near ``float('inf')`` (or a positive int beyond the float
        range, e.g. ``10 ** 1000``) yields the limiting score rather than
        infinity. Read-only; raises ``ValueError`` for bad terms, ``k1``/``b``
        or an invalid expression, even when nothing could match.
        """
        if not isinstance(terms, list) or not terms:
            raise ValueError("bm25 needs a non-empty list of terms")
        if any(not isinstance(term, str) or not term for term in terms):
            raise ValueError("bm25 terms must be non-empty strings")
        _validate_bm25_k1(k1)
        _validate_bm25_b(b)
        tree = None
        if expression is not None:
            if not isinstance(expression, str):
                raise ValueError("expression must be a string or None")
            tree = _parse_search(_lex_search(expression))
        wanted = list(dict.fromkeys(term.lower() for term in terms))
        # A giant Python int overflows float conversion; map it to +inf.
        # Arithmetic below never multiplies by the raw int, so no
        # OverflowError can escape.
        try:
            k1_float = float(k1)
        except OverflowError:
            k1_float = math.inf
        snapshot = self._read()
        documents = snapshot["documents"]
        postings = snapshot["postings"]
        n_docs = len(documents)
        lengths = {doc_id: len(tokenize(text)) for doc_id, text in documents.items()}
        total_length = sum(lengths.values())
        candidates: set[str] | None = None
        if tree is not None:
            candidates = _evaluate_search(tree, documents, postings)
        if n_docs == 0 or total_length == 0:
            return []
        avgdl = total_length / n_docs
        scored: dict[str, dict] = {}
        for term in wanted:
            entries = postings.get(term, {})
            df = len(entries)
            idf = math.log1p((n_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in entries.items():
                if candidates is not None and doc_id not in candidates:
                    continue
                length_norm = 1.0 - b + b * lengths[doc_id] / avgdl
                contribution = idf * tf * _bm25_term_ratio(tf, k1_float, length_norm)
                hit = scored.setdefault(doc_id, {"matched": 0, "score": 0.0})
                hit["matched"] += 1
                hit["score"] += contribution
        results = [{"id": doc_id, "matched": hit["matched"], "score": round(hit["score"], 6)}
                   for doc_id, hit in scored.items()]
        results.sort(key=lambda item: (-item["score"], item["id"]))
        return results

    def expand(self, pattern: str) -> list[str]:
        """List every indexed term matching the wildcard ``pattern``.

        ``*`` matches zero or more characters, ``?`` exactly one; the pattern
        is lower-cased like any term and may only contain ASCII letters,
        digits, underscore and the two wildcards. Returns the matching terms
        sorted by Unicode code point, without duplicates; an empty list when
        nothing matches. The index is only read, never written. An empty
        pattern, a non-string pattern or one with unsupported characters
        raises ``ValueError``.
        """
        matcher = _compile_pattern(pattern)
        return sorted(term for term in self._read()["postings"] if matcher(term))

    def fuzzy(self, term: str, max_distance: int = 1) -> list[dict]:
        """List indexed terms within Levenshtein distance ``max_distance`` of ``term``.

        The term is lower-cased like any term and may only contain ASCII
        letters, digits and underscores; it is compared as-is -- never
        tokenised, stripped or wildcard-expanded. The distance counts
        insertions, deletions and substitutions of a single character as one
        operation each; swapping adjacent characters is not an operation.
        ``max_distance`` must be an integer from 0 to 2 (booleans rejected).
        Returns ``[{"term": term, "distance": distance}, ...]`` sorted by
        distance then term (Unicode code points), without duplicates and
        including an exact match at distance 0; an empty index or no
        candidates yields an empty array. All arguments are validated before
        the snapshot is read; the index is only read, never written.
        """
        wanted = _validate_fuzzy_term(term)
        limit = _validate_max_distance(max_distance)
        postings = self._read()["postings"]
        results = []
        for candidate in postings:
            distance = _levenshtein(wanted, candidate, limit)
            if distance <= limit:
                results.append({"term": candidate, "distance": distance})
        results.sort(key=lambda item: (item["distance"], item["term"]))
        return results

    def terms(self) -> list[str]:
        return sorted(self._read()["postings"])

    def stats(self) -> dict:
        document = self._read()
        return {"documents": len(document["documents"]), "terms": len(document["postings"]),
                "postings": sum(len(entries) for entries in document["postings"].values())}

    def reload(self) -> None:
        self._read()
