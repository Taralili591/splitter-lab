"""Integration checks with recording providers; no cloud calls or fake lab mode."""

import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from labs.hr_chunking_lab import engine
from labs.hr_chunking_lab.chunking import source_spans


SOURCE = """# 星桥测试手册

## 苹果福利

[F01] """ + "苹果福利适用本公司任职年限，必须查清日期和扣除已用。" * 7 + """

[F02] """ + "苹果福利申请需要主管审批，并保存已经完成的交接记录。" * 7 + """

## 香蕉补休

[O01] """ + "香蕉补休需要先核对加班类型，再判断是否可以补休。" * 8 + """

[O02] """ + "香蕉补休账户按小时记账，使用前应确认实际剩余时数。" * 7 + "\n"

GOLD_ANSWER = "GOLD_REFERENCE_SHOULD_NEVER_LEAK"
GOLD_POINT = "GOLD_REQUIRED_POINT_SHOULD_NEVER_LEAK"
QUERY = "苹果福利需要怎样计算？"


class RecordingEmbedder:
    calls = []

    def __init__(self, settings, cache_dir):
        self.settings = settings

    @staticmethod
    def vector(text):
        apples, bananas = text.count("苹果"), text.count("香蕉")
        return [float(apples), float(bananas), 0.0] if apples or bananas else [0.0, 0.0, 1.0]

    def embed(self, texts, role="document"):
        self.calls.append({"role": role, "texts": list(texts), "model": self.settings.get("siliconflow_embedding_model")})
        return [self.vector(text) for text in texts]


class RecordingChat:
    calls = []

    def __init__(self, settings):
        self.settings = settings

    def chat(self, messages, max_tokens=1000):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        context_generation = "<document>" in messages[-1]["content"]
        return {
            "content": "这是从测试手册提取的定位背景。" * 8 if context_generation else "现有材料不足以确定，请补充所需条件。",
            "model": self.settings.get("deepseek_model", "test-chat"),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class HREngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.write_source(SOURCE)
        (self.data_dir / "quiz.json").write_text(json.dumps({"questions": [{
            "id": "Q01", "question": QUERY, "answer": GOLD_ANSWER,
            "required_points": [GOLD_POINT], "evidence_ids": ["F01"],
            "type": "可答", "case_facts": {"used_days": 2},
        }]}, ensure_ascii=False))
        (self.data_dir / "assistant_prompt.txt").write_text("只依据检索原文回答；没有证据就说明不足。")
        self.lab = engine.LabEngine(self.data_dir, self.root / "runtime")
        self.settings = {
            "siliconflow_base_url": "https://example.invalid/v1",
            "siliconflow_embedding_model": "test-embedding-a",
            "deepseek_base_url": "https://chat.invalid/v1",
            "deepseek_model": "test-chat-a",
        }
        RecordingEmbedder.calls = []
        RecordingChat.calls = []
        self.enterContext(patch.object(engine, "SiliconFlowEmbedder", RecordingEmbedder))
        self.enterContext(patch.object(engine, "DeepSeekChat", RecordingChat))

    def write_source(self, text):
        (self.data_dir / "handbook.md").write_text(text, encoding="utf-8")

    def params(self, **overrides):
        return {
            "chunk_size": 80, "chunk_overlap": 10, "parent_size": 280,
            "semantic_percentile": 75, "top_k": 3, "context_budget_chars": 1800,
            "retrieval_mode": "vector", **overrides,
        }

    def test_vector_ranking_changes_with_actual_query_vector(self):
        banana = self.lab.retrieve(["fixed"], "香蕉补休", self.params(top_k=1), self.settings)
        apple = self.lab.retrieve(["fixed"], "苹果福利", self.params(top_k=1), self.settings)
        banana_top = banana["results"][0]["candidates"][0]
        apple_top = apple["results"][0]["candidates"][0]
        self.assertIn("香蕉", banana_top["text"])
        self.assertIn("苹果", apple_top["text"])
        self.assertNotEqual(banana_top["id"], apple_top["id"])
        self.assertAlmostEqual(banana_top["score"], 1.0)
        self.assertEqual(banana["results"][0]["score_type"], "cosine")
        self.assertTrue(any(c["role"] == "document" for c in RecordingEmbedder.calls))
        self.assertEqual(sum(c["role"] == "query" for c in RecordingEmbedder.calls), 2)

    def test_keyword_bm25_uses_no_cloud_api(self):
        run = self.lab.retrieve(["fixed", "recursive", "structure", "parent_child"], "香蕉补休", self.params(retrieval_mode="keyword"), self.settings)
        self.assertEqual(RecordingEmbedder.calls, [])
        self.assertEqual(RecordingChat.calls, [])
        for result in run["results"]:
            self.assertEqual(result["score_type"], "BM25")
            self.assertTrue(result["included_count"])
            self.assertIn("香蕉", result["context"])

    def test_parent_retrieval_deduplicates_and_preserves_returned_provenance(self):
        run = self.lab.retrieve(["parent_child"], "苹果福利", self.params(top_k=6, context_budget_chars=4000), self.settings)
        result = run["results"][0]
        included = [c for c in result["candidates"] if c["included"]]
        self.assertEqual(len({c["parent_id"] for c in included}), len(included))
        self.assertTrue(any("去重" in c["excluded_reason"] for c in result["candidates"]))
        for card in included:
            source = "".join(SOURCE[s["start"]:s["end"]] for s in card["returned_spans"])
            self.assertEqual(card["returned_text"], source)
            self.assertEqual(card["returned_text"], card["parent_text"])
            self.assertIn(card["text"], card["returned_text"])
            self.assertTrue(set(card["clause_ids"]).issubset(card["returned_clause_ids"]))

    def test_budget_counts_background_headers_and_never_truncates_original_blocks(self):
        source = "# 测试手册\n\n[F01] " + "甲" * 620
        self.write_source(source)
        params = self.params(chunk_size=280, parent_size=900, retrieval_mode="keyword", context_budget_chars=400)
        run = self.lab.retrieve(["contextual"], "甲", params, self.settings)
        result = run["results"][0]
        self.assertEqual(result["context_chars"], len(result["context"]))
        self.assertLessEqual(result["context_chars"], 400)
        included = [c for c in result["candidates"] if c["included"]]
        self.assertTrue(included)
        self.assertGreater(result["context_chars"], sum(len(c["returned_text"]) for c in included))
        excluded = [c for c in result["candidates"] if "预算" in c["excluded_reason"]]
        self.assertTrue(excluded)
        self.assertTrue(any(len(c["text"]) < 400 < len(c["returned_text"]) for c in excluded))
        for card in result["candidates"]:
            self.assertEqual(card["text"], source[card["start"]:card["end"]])
            self.assertTrue(card["returned_text"].endswith(card["text"]))
            self.assertIn("背景说明", card["returned_text"])
        for card in included:
            self.assertIn(card["returned_text"], result["context"])

    def test_reference_coverage_requires_full_clause_text_not_just_id(self):
        doc = self.lab.document()
        clause = doc.clauses["F01"]
        midpoint = (clause.start + clause.end) // 2
        first = [asdict(s) for s in source_spans(doc, clause.start, midpoint)]
        partial = engine.evidence_coverage(doc, first, {"evidence_ids": ["F01"]})
        self.assertFalse(partial["all_complete"])
        self.assertEqual(partial["clauses"][0]["status"], "partial")
        self.assertLess(partial["clauses"][0]["ratio"], 1)
        second = [asdict(s) for s in source_spans(doc, midpoint - 4, clause.end)]
        full = engine.evidence_coverage(doc, first + second, {"evidence_ids": ["F01"]})
        self.assertTrue(full["all_complete"])
        self.assertEqual(full["clauses"][0]["ratio"], 1.0)
        self.assertEqual(full["clauses"][0]["covered_chars"], clause.end - clause.start)

    def test_gold_answers_never_enter_context_generation_or_answer_messages(self):
        params = self.params(retrieval_mode="keyword")
        run = self.lab.retrieve(["contextual"], QUERY, params, self.settings, question_id="Q01")
        self.assertEqual(run["reference"]["answer"], GOLD_ANSWER)
        self.lab.answer(run["run_id"], ["contextual"], self.settings)
        self.assertTrue(any("<document>" in c["messages"][-1]["content"] for c in RecordingChat.calls))
        self.assertTrue(any("<retrieved_material>" in c["messages"][-1]["content"] for c in RecordingChat.calls))
        for record in RecordingChat.calls:
            outgoing = json.dumps(record["messages"], ensure_ascii=False)
            self.assertNotIn(GOLD_ANSWER, outgoing)
            self.assertNotIn(GOLD_POINT, outgoing)
        for record in RecordingEmbedder.calls:
            outgoing = "\n".join(record["texts"])
            self.assertNotIn(GOLD_ANSWER, outgoing)
            self.assertNotIn(GOLD_POINT, outgoing)
        answer_message = RecordingChat.calls[-1]["messages"][-1]["content"]
        self.assertIn('"used_days": 2', answer_message)

    def test_edited_question_does_not_inherit_gold_or_old_case_facts(self):
        run = self.lab.retrieve(["fixed"], QUERY + "我已经改变个人情况。", self.params(retrieval_mode="keyword"), self.settings, question_id="Q01")
        self.assertIsNone(run["question_id"])
        self.assertIsNone(run["reference"])
        self.assertEqual(run["case_facts"], {})
        self.assertIsNone(run["results"][0]["coverage"])
        self.lab.answer(run["run_id"], ["fixed"], self.settings)
        outgoing = RecordingChat.calls[-1]["messages"][-1]["content"]
        self.assertNotIn(GOLD_ANSWER, outgoing)
        self.assertNotIn('"used_days": 2', outgoing)

    def test_index_cache_invalidates_when_embedding_model_base_or_chunk_params_change(self):
        params = self.params()
        first = self.lab.build_index("fixed", params, self.settings)
        document_calls = len([c for c in RecordingEmbedder.calls if c["role"] == "document"])
        same = self.lab.build_index("fixed", params, self.settings)
        self.assertEqual(first["index_id"], same["index_id"])
        self.assertEqual(document_calls, len([c for c in RecordingEmbedder.calls if c["role"] == "document"]))
        changed_model = self.lab.build_index("fixed", params, {**self.settings, "siliconflow_embedding_model": "test-embedding-b"})
        changed_base = self.lab.build_index("fixed", params, {**self.settings, "siliconflow_base_url": "https://other.invalid/v1"})
        changed_overlap = self.lab.build_index("fixed", {**params, "chunk_overlap": 11}, self.settings)
        self.assertEqual(len({first["index_id"], changed_model["index_id"], changed_base["index_id"], changed_overlap["index_id"]}), 4)
        self.assertEqual(document_calls + 3, len([c for c in RecordingEmbedder.calls if c["role"] == "document"]))

    def test_empty_retrieval_allows_insufficiency_answer_without_handbook_fallback(self):
        run = self.lab.retrieve(["fixed"], "zzzzzz_unmatched", self.params(retrieval_mode="keyword"), self.settings)
        result = run["results"][0]
        self.assertEqual(result["context"], "")
        self.assertEqual(result["included_count"], 0)
        self.assertTrue(result["warnings"])
        response = self.lab.answer(run["run_id"], ["fixed"], self.settings)
        self.assertIn("不足", response["answers"][0]["answer"])
        message = RecordingChat.calls[-1]["messages"][-1]["content"]
        self.assertIn("本次没有取回资料。", message)
        self.assertNotIn(SOURCE, message)
        self.assertNotIn("[F01]", message)


if __name__ == "__main__":
    unittest.main()
