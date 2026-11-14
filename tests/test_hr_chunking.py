"""Source-provenance and algorithm tests; fake vectors test boundaries, not quality."""

import unittest
from dataclasses import asdict
from pathlib import Path

from labs.hr_chunking_lab.chunking import build_chunks, parse_document, source_spans


MANUAL = """# 员工手册

## 公司福利假

[F01] 公司福利假是额外福利。与法定年休假分开记录，不可以混用。

[F02] 本公司满五年，总额八天。周年当天晋档，已用天数继续扣除。

## 请假流程

[L01] 超过两天要两级审批。直属主管后由部门负责人批准。HR仅备案。
"""


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.doc = parse_document(MANUAL)

    def assert_complete_provenance(self, chunks, original=MANUAL):
        covered = set()
        for chunk in chunks:
            self.assertEqual(original[chunk.start:chunk.end], chunk.text)
            self.assertTrue(chunk.start < chunk.end)
            positions = set()
            for span in chunk.source_spans:
                self.assertTrue(chunk.start <= span.start < span.end <= chunk.end)
                self.assertTrue(span.clause_start <= span.start < span.end <= span.clause_end)
                positions.update(range(span.start, span.end))
                covered.update(range(span.start, span.end))
            self.assertEqual(positions, set(range(chunk.start, chunk.end)))
        self.assertEqual(covered, set(range(len(original))))

    def test_parse_preserves_exact_text_clauses_and_nonoverlapping_sections(self):
        self.assertEqual(self.doc.text, MANUAL)
        self.assertEqual(set(self.doc.clauses), {"F01", "F02", "L01"})
        f02 = self.doc.clauses["F02"]
        self.assertTrue(MANUAL[f02.start:f02.end].startswith("[F02]"))
        self.assertTrue(MANUAL[f02.start:f02.end].endswith("扣除。"))
        self.assertIn("公司福利假", f02.section)
        self.assertEqual(self.doc.sections[0].start, 0)
        self.assertEqual(self.doc.sections[-1].end, len(MANUAL))
        for left, right in zip(self.doc.sections, self.doc.sections[1:]):
            self.assertEqual(left.end, right.start)

    def test_fixed_overlap_and_all_source_characters(self):
        chunks = build_chunks(self.doc, "fixed", {"chunk_size": 40, "chunk_overlap": 9})
        self.assert_complete_provenance(chunks)
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left.end - right.start, 9)
        self.assertTrue(all(len(chunk.text) <= 40 for chunk in chunks))

    def test_recursive_preserves_boundaries_and_falls_back_for_long_units(self):
        doc = parse_document("甲" * 10 + "。" + "乙" * 10 + "。" + "丙" * 10 + "。")
        chunks = build_chunks(doc, "recursive", {"chunk_size": 25, "chunk_overlap": 0})
        self.assertEqual([chunk.text for chunk in chunks], ["甲" * 10 + "。" + "乙" * 10 + "。", "丙" * 10 + "。"])
        self.assert_complete_provenance(chunks, doc.text)
        long_doc = parse_document("无标点文本" * 50)
        pieces = build_chunks(long_doc, "recursive", {"chunk_size": 21, "chunk_overlap": 4})
        self.assertTrue(all(len(piece.text) <= 21 for piece in pieces))
        self.assert_complete_provenance(pieces, long_doc.text)

    def test_structure_does_not_cross_sections_and_indexes_heading_context(self):
        chunks = build_chunks(self.doc, "structure", {"chunk_size": 60, "chunk_overlap": 5})
        self.assert_complete_provenance(chunks)
        for chunk in chunks:
            sections = [s for s in self.doc.sections if s.start <= chunk.start < s.end]
            self.assertEqual(len(sections), 1)
            self.assertLessEqual(chunk.end, sections[0].end)
            self.assertTrue(chunk.index_text.startswith(sections[0].path + "\n"))

    def test_contextual_has_exact_recursive_body_and_no_fabricated_context(self):
        settings = {"chunk_size": 50, "chunk_overlap": 7}
        recursive = build_chunks(self.doc, "recursive", settings)
        contextual = build_chunks(self.doc, "contextual", settings)
        self.assertEqual([(c.start, c.end, c.text) for c in recursive], [(c.start, c.end, c.text) for c in contextual])
        self.assertTrue(all(not c.context and c.index_text == c.text for c in contextual))

    def test_parent_children_have_exact_parent_sources_and_child_coverage(self):
        chunks = build_chunks(self.doc, "parent_child", {"chunk_size": 40, "chunk_overlap": 5, "parent_size": 80})
        self.assert_complete_provenance(chunks)
        parents = {}
        for chunk in chunks:
            self.assertTrue(chunk.parent_id)
            parents.setdefault(chunk.parent_id, chunk.parent_text)
            self.assertEqual(parents[chunk.parent_id], chunk.parent_text)
            self.assertLessEqual(len(chunk.parent_text), 80)
            self.assertIn(chunk.text, chunk.parent_text)
            self.assertEqual("".join(MANUAL[s.start:s.end] for s in chunk.parent_spans), chunk.parent_text)
        self.assertLess(len(parents), len(chunks))

    def test_partial_clause_hit_is_not_whole_clause(self):
        clause = self.doc.clauses["F02"]
        spans = source_spans(self.doc, clause.start + 5, clause.end - 4)
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.clause_id, "F02")
        self.assertEqual((span.clause_start, span.clause_end), (clause.start, clause.end))
        self.assertLess(span.end - span.start, span.clause_end - span.clause_start)
        self.assertEqual(asdict(span)["start"], clause.start + 5)

    def test_semantic_uses_given_embeddings_to_find_topic_boundary(self):
        text = "假期额度由任职年限决定。" * 4 + "交通报销需要附上支付凭证。" * 4
        doc = parse_document(text)
        recorded = []
        def embed(inputs):
            recorded.extend(inputs)
            return [[1.0, 0.0] if item.count("假期") >= item.count("交通") else [0.0, 1.0] for item in inputs]
        chunks = build_chunks(doc, "semantic", {"chunk_size": 65, "chunk_overlap": 0, "semantic_percentile": 50}, embed)
        self.assertTrue(recorded)
        self.assertTrue(all(part in text for part in recorded))
        self.assert_complete_provenance(chunks, text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.text) <= 130 for c in chunks))
        self.assertTrue(all("semantic_threshold" in c.metadata for c in chunks))
        self.assertEqual(chunks[0].end, len("假期额度由任职年限决定。" * 4))

    def test_semantic_requires_service_and_rejects_invalid_vectors(self):
        with self.assertRaisesRegex(ValueError, "embedding"):
            build_chunks(self.doc, "semantic")
        with self.assertRaisesRegex(ValueError, "数量"):
            build_chunks(self.doc, "semantic", embed=lambda inputs: [])
        with self.assertRaisesRegex(ValueError, "全为零"):
            build_chunks(self.doc, "semantic", embed=lambda inputs: [[0, 0] for _ in inputs])

    def test_long_whitespace_remains_bounded_without_empty_embedding_inputs(self):
        original = "# 标题\n\n[F01] 一条规则。" + "\n" * 100
        recorded = []
        def embed(inputs):
            recorded.extend(inputs)
            return [[1.0, 0.0] for _ in inputs]
        chunks = build_chunks(parse_document(original), "semantic", {"chunk_size": 8, "chunk_overlap": 0}, embed)
        self.assert_complete_provenance(chunks, original)
        self.assertTrue(all(recorded))
        self.assertTrue(all(len(chunk.text) <= 16 for chunk in chunks))
        self.assertTrue(all(chunk.index_text.strip() for chunk in chunks))

    def test_settings_and_duplicate_ids_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            parse_document("[F01] 一条。\n\n[F01] 另一条。")
        with self.assertRaisesRegex(ValueError, "重叠"):
            build_chunks(self.doc, "fixed", {"chunk_size": 10, "chunk_overlap": 10})
        with self.assertRaisesRegex(ValueError, "分位数"):
            build_chunks(self.doc, "semantic", {"semantic_percentile": float("nan")}, lambda x: x)

    def test_supplied_employee_manual_is_covered_by_all_structural_algorithms(self):
        path = Path(__file__).resolve().parents[1] / "labs/hr_chunking_lab/data/handbook.md"
        doc = parse_document(path.read_text(encoding="utf-8"))
        for strategy in ("fixed", "recursive", "structure", "contextual", "parent_child"):
            with self.subTest(strategy=strategy):
                self.assert_complete_provenance(build_chunks(doc, strategy), doc.text)


if __name__ == "__main__":
    unittest.main()
