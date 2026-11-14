"""Six inspectable chunking strategies with exact original-text provenance.

Lengths and offsets are Python Unicode characters, not model tokens. Semantic
chunking uses the supplied embedding function; it never substitutes a lexical
heuristic when a model is unavailable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SourceSpan:
    """A chunk's intersection with one clause (or an unlabelled source gap).

    All offsets are absolute in Document.text. clause_start/clause_end describe
    the entire original clause, not just the retrieved intersection. A clause ID
    being present therefore does not mean that all of its words were retrieved.
    """

    start: int
    end: int
    clause_id: str
    clause_start: int
    clause_end: int


@dataclass(frozen=True)
class Section:
    title: str
    path: str
    start: int
    end: int
    level: int = 0
    content_start: int = 0


@dataclass(frozen=True)
class Clause:
    id: str
    start: int
    end: int
    section: str


@dataclass
class Document:
    text: str
    sections: list[Section]
    clauses: dict[str, Clause]
    title: str = "员工手册"


@dataclass
class Chunk:
    id: str
    text: str
    index_text: str
    context: str
    source_spans: list[SourceSpan]
    section: str
    parent_id: str | None = None
    parent_text: str = ""
    parent_spans: list[SourceSpan] = field(default_factory=list)
    start: int = 0
    end: int = 0
    metadata: dict = field(default_factory=dict)


DEFAULT_SETTINGS = {
    "chunk_size": 280,
    "chunk_overlap": 50,
    "parent_size": 900,
    "semantic_percentile": 75.0,
}
STRATEGIES = ("fixed", "recursive", "structure", "semantic", "contextual", "parent_child")
_SEPARATORS = (r"\n[ \t]*\n", r"\n", r"[。！？!?；;]", r"[，,：:]", r"\s+")
_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*\r?$")
_CLAUSE = re.compile(r"(?m)^[ \t]*\[([A-Za-z][A-Za-z0-9_-]*)\][ \t]*")


def parse_document(text: str) -> Document:
    """Parse Markdown headings and [F01]-style clauses without changing text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("员工手册原文不能为空。")
    headings = list(_HEADING.finditer(text))
    sections: list[Section] = []
    title = headings[0].group(2).strip() if headings else "员工手册"
    stack: list[tuple[int, str]] = []
    if not headings:
        sections.append(Section(title, title, 0, len(text), 0, 0))
    else:
        if headings[0].start() > 0:
            sections.append(Section("文档开头", "文档开头", 0, headings[0].start(), 0, 0))
        for index, heading in enumerate(headings):
            level, name = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            sections.append(Section(name, " / ".join(item[1] for item in stack), heading.start(), end, level, heading.end()))
    clauses: dict[str, Clause] = {}
    markers = list(_CLAUSE.finditer(text))
    for index, marker in enumerate(markers):
        clause_id = marker.group(1)
        if clause_id in clauses:
            raise ValueError(f"原文存在重复条款编号：{clause_id}")
        limit = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        # A labelled paragraph ends at a blank line or a new heading. This keeps
        # unrelated later paragraphs out of the clause's evidence range.
        blank = re.search(r"\n[ \t]*\r?\n", text[marker.end():limit])
        if blank:
            limit = min(limit, marker.end() + blank.start())
        next_heading = next((h.start() for h in headings if marker.start() < h.start() < limit), None)
        if next_heading is not None:
            limit = next_heading
        end = marker.start() + len(text[marker.start():limit].rstrip())
        section = _section_for(sections, marker.start()).path
        clauses[clause_id] = Clause(clause_id, marker.start(), end, section)
    return Document(text, sections, clauses, title)


def _section_for(sections: Sequence[Section], offset: int) -> Section:
    return next((section for section in sections if section.start <= offset < section.end), sections[-1])


def source_spans(doc: Document, start: int, end: int) -> list[SourceSpan]:
    """Project a contiguous source range to clauses, including unlabelled gaps."""
    if not 0 <= start <= end <= len(doc.text):
        raise ValueError("原文范围不合法。")
    result: list[SourceSpan] = []
    cursor = start
    for clause in sorted(doc.clauses.values(), key=lambda item: item.start):
        if clause.end <= start or clause.start >= end:
            continue
        left, right = max(start, clause.start), min(end, clause.end)
        if cursor < left:
            result.append(SourceSpan(cursor, left, "", cursor, left))
        result.append(SourceSpan(left, right, clause.id, clause.start, clause.end))
        cursor = right
    if cursor < end:
        result.append(SourceSpan(cursor, end, "", cursor, end))
    return result


def _settings(settings: Mapping | None) -> dict:
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("切块设置必须为键值对象。")
    supplied = dict(settings or {})
    if "overlap" in supplied and "chunk_overlap" not in supplied:
        supplied["chunk_overlap"] = supplied["overlap"]
    result = {**DEFAULT_SETTINGS, **supplied}
    for name in ("chunk_size", "chunk_overlap", "parent_size"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是整数。")
    if not 8 <= result["chunk_size"] <= 20000:
        raise ValueError("每块长度须在 8 到 20000 字符之间。")
    if not 0 <= result["chunk_overlap"] < result["chunk_size"]:
        raise ValueError("重叠字符数须大于等于 0 且小于每块长度。")
    if not result["chunk_size"] <= result["parent_size"] <= 100000:
        raise ValueError("父块长度须不小于子块长度，且不超过 100000 字符。")
    percentile = result["semantic_percentile"]
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)) or not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("语义分位数须在 0 到 100 之间。")
    return result


def _fixed_intervals(start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    intervals = []
    cursor = start
    while cursor < end:
        stop = min(cursor + size, end)
        intervals.append((cursor, stop))
        if stop == end:
            break
        cursor = stop - overlap
    return intervals


def _recursive_units(text: str, start: int, end: int, size: int, separators=_SEPARATORS) -> list[tuple[int, int]]:
    """Recursively split only oversized pieces; retain separator characters."""
    if end <= start:
        return []
    if end - start <= size:
        return [(start, end)]
    for index, separator in enumerate(separators):
        matches = list(re.finditer(separator, text[start:end]))
        if not matches:
            continue
        units, cursor = [], start
        for match in matches:
            stop = start + match.end()
            if stop > cursor:
                units.extend(_recursive_units(text, cursor, stop, size, separators[index + 1:]))
                cursor = stop
        if cursor < end:
            units.extend(_recursive_units(text, cursor, end, size, separators[index + 1:]))
        return units
    return _fixed_intervals(start, end, size, 0)


def _merge_units(units: Iterable[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Pack adjacent complete units; overlap is a maximum target, not a promise."""
    windows: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    for unit in units:
        if current and unit[1] - current[0][0] > size:
            windows.append((current[0][0], current[-1][1]))
            while current and (current[-1][1] - current[0][0] > overlap or unit[1] - current[0][0] > size):
                current.pop(0)
        current.append(unit)
    if current:
        windows.append((current[0][0], current[-1][1]))
    return windows


def _recursive_intervals(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    return _merge_units(_recursive_units(text, start, end, size), size, overlap)


def _make_chunk(doc: Document, strategy: str, number: int, start: int, end: int, *, index_prefix: str = "", metadata: dict | None = None) -> Chunk:
    body = doc.text[start:end]
    paths = list(dict.fromkeys(section.path for section in doc.sections if section.start < end and section.end > start))
    if not body.strip():
        # A source containing more than a full chunk of whitespace still keeps
        # exact provenance. Index its real heading rather than submit empty API
        # input; no policy content is invented for this empty source interval.
        index_prefix = index_prefix or paths[0]
    return Chunk(
        id=f"{strategy}:{number:04d}:{start}-{end}",
        text=body,
        index_text=(index_prefix + "\n" + body) if index_prefix else body,
        context="",
        source_spans=source_spans(doc, start, end),
        section=" | ".join(paths),
        start=start,
        end=end,
        metadata={"strategy": strategy, "length_unit": "characters", **(metadata or {})},
    )


def _sentence_units(text: str, max_size: int) -> list[tuple[int, int]]:
    cuts = [0]
    # Newlines keep headings and paragraph identifiers in their real positions.
    for match in re.finditer(r"[。！？!?；;]+[”’\"）)]*|\n+", text):
        if match.end() > cuts[-1]:
            cuts.append(match.end())
    if cuts[-1] != len(text):
        cuts.append(len(text))
    units: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        # Merge a whitespace-only unit into the preceding source interval so an
        # embeddings endpoint never receives empty input for a standalone gap.
        if not text[left:right].strip() and units and right - units[-1][0] <= max_size:
            old_start, _ = units[-1]
            units[-1] = (old_start, right)
        else:
            units.extend(_recursive_intervals(text, left, right, max_size, 0))
    return units


def _coerce_vectors(payload, expected: int) -> list[list[float]]:
    values = payload.get("vectors") if isinstance(payload, Mapping) else payload
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise ValueError("语义切块的 embedding 数量与句子数量不一致。")
    vectors, dimension = [], None
    for value in values:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("语义切块需要非空 embedding 向量。")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
            raise ValueError("embedding 向量中存在无效数值。")
        vector = [float(item) for item in value]
        if not any(vector):
            raise ValueError("embedding 向量不能全为零。")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError("embedding 向量维数不一致。")
        vectors.append(vector)
    return vectors


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return max(-1.0, min(1.0, numerator / denominator))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _semantic_intervals(doc: Document, config: dict, embed: Callable) -> tuple[list[tuple[int, int]], dict]:
    size = config["chunk_size"]
    minimum, maximum = max(1, size // 2), size * 2
    units = _sentence_units(doc.text, maximum)
    if not units:
        return [], {"semantic_threshold": 0.0, "semantic_sentence_count": 0}
    # Each sentence is represented with one adjacent sentence on each side.
    # The callback receives actual source text only, never questions or gold.
    inputs = [doc.text[units[max(0, index - 1)][0]:units[min(len(units) - 1, index + 1)][1]].strip() for index in range(len(units))]
    # Long whitespace runs can become their own bounded units. They have no
    # semantic content, so skip empty API inputs and carry the nearest actual
    # sentence vector across them. Their physical length is still respected.
    nonempty = [index for index, value in enumerate(inputs) if value]
    actual = _coerce_vectors(embed([inputs[index] for index in nonempty]), len(nonempty))
    by_index = dict(zip(nonempty, actual))
    vectors = [by_index[index] if index in by_index else by_index[min(nonempty, key=lambda other: abs(other - index))] for index in range(len(inputs))]
    distances = [1.0 - _cosine(left, right) for left, right in zip(vectors, vectors[1:])]
    threshold = _percentile(distances, config["semantic_percentile"])
    intervals: list[tuple[int, int]] = []
    start, stop = units[0]
    for index, (left, right) in enumerate(units[1:], 1):
        semantic_break = stop - start >= minimum and distances[index - 1] > threshold
        forced_break = right - start > maximum
        if semantic_break or forced_break:
            intervals.append((start, stop))
            start = left
        stop = right
    intervals.append((start, stop))
    return intervals, {
        "semantic_threshold": round(threshold, 6),
        "semantic_percentile": config["semantic_percentile"],
        "semantic_sentence_count": len(units),
        "semantic_embedding_input_count": len(nonempty),
        "semantic_min_chars": minimum,
        "semantic_max_chars": maximum,
    }


def build_chunks(doc: Document, strategy: str, settings: Mapping | None = None, embed: Callable | None = None) -> list[Chunk]:
    """Build one strategy; no generated context or simulated vectors are added.

    Contextual chunks share recursive source boundaries. The caller may later
    populate Chunk.context and prepend that model-generated background to
    index_text. Parent retrieval indexes children and returns their parent_text.
    """
    if not isinstance(doc, Document):
        raise ValueError("请先使用 parse_document 解析原文。")
    if strategy == "fixed_overlap":
        strategy = "fixed"
    if strategy not in STRATEGIES:
        raise ValueError(f"未知切块方式：{strategy}")
    config = _settings(settings)
    size, overlap = config["chunk_size"], config["chunk_overlap"]
    if strategy == "parent_child":
        chunks: list[Chunk] = []
        parent_number = 0
        for section in doc.sections:
            for parent_start, parent_end in _recursive_intervals(doc.text, section.start, section.end, config["parent_size"], 0):
                parent_number += 1
                parent_id = f"parent:{parent_number:04d}:{parent_start}-{parent_end}"
                parent_body = doc.text[parent_start:parent_end]
                spans = source_spans(doc, parent_start, parent_end)
                for left, right in _recursive_intervals(doc.text, parent_start, parent_end, size, overlap):
                    chunk = _make_chunk(doc, strategy, len(chunks) + 1, left, right)
                    chunk.parent_id = parent_id
                    chunk.parent_text = parent_body
                    chunk.parent_spans = list(spans)
                    chunks.append(chunk)
        return chunks
    if strategy == "structure":
        chunks = []
        for section in doc.sections:
            for left, right in _recursive_intervals(doc.text, section.start, section.end, size, overlap):
                chunks.append(_make_chunk(doc, strategy, len(chunks) + 1, left, right, index_prefix=section.path))
        return chunks
    metadata = None
    if strategy == "fixed":
        intervals = _fixed_intervals(0, len(doc.text), size, overlap)
    elif strategy == "semantic":
        if embed is None:
            raise ValueError("语义切块需要真实 embedding 服务；请配置硅基流动 API。")
        intervals, metadata = _semantic_intervals(doc, config, embed)
    else:
        intervals = _recursive_intervals(doc.text, 0, len(doc.text), size, overlap)
    return [_make_chunk(doc, strategy, index + 1, left, right, metadata=metadata) for index, (left, right) in enumerate(intervals)]
