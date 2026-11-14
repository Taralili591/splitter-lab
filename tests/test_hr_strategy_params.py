"""Method-specific settings stay independent while retrieval controls stay shared."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from labs.hr_chunking_lab import engine
from tests.test_hr_engine import SOURCE, RecordingChat, RecordingEmbedder


class HRStrategyParamTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        data = self.root / 'data'
        data.mkdir()
        (data / 'handbook.md').write_text(SOURCE, encoding='utf-8')
        (data / 'quiz.json').write_text('{"questions": []}', encoding='utf-8')
        self.lab = engine.LabEngine(data, self.root / 'runtime')
        self.shared = {'retrieval_mode': 'keyword', 'top_k': 1, 'context_budget_chars': 600}
        self.enterContext(patch.object(engine, 'SiliconFlowEmbedder', RecordingEmbedder))
        self.enterContext(patch.object(engine, 'DeepSeekChat', RecordingChat))
        RecordingEmbedder.calls = []
        RecordingChat.calls = []

    def test_build_uses_each_methods_settings_and_only_changed_index_invalidates(self):
        overrides = {'fixed': {'chunk_size': 160, 'chunk_overlap': 20},
                     'recursive': {'chunk_size': 280, 'chunk_overlap': 0}}
        built = self.lab.build(['fixed', 'recursive'], self.shared, {}, strategy_params=overrides)
        first = {index['strategy']: index for index in built['indexes']}
        self.assertEqual(first['fixed']['stats']['max_chars'], 160)
        self.assertLessEqual(first['recursive']['stats']['max_chars'], 280)
        self.assertEqual(first['recursive']['params']['chunk_overlap'], 0)
        changed = {**overrides, 'fixed': {'chunk_size': 200, 'chunk_overlap': 30}}
        again = self.lab.build(['fixed', 'recursive'], self.shared, {}, strategy_params=changed)
        second = {index['strategy']: index for index in again['indexes']}
        self.assertNotEqual(first['fixed']['index_id'], second['fixed']['index_id'])
        self.assertEqual(first['recursive']['index_id'], second['recursive']['index_id'])
        self.assertEqual(RecordingEmbedder.calls, [])

    def test_retrieval_and_export_record_each_effective_method_and_shared_limits(self):
        overrides = {'fixed': {'chunk_size': 160, 'chunk_overlap': 20},
                     'recursive': {'chunk_size': 280, 'chunk_overlap': 0}}
        run = self.lab.retrieve(['fixed', 'recursive'], '苹果福利', self.shared, {}, strategy_params=overrides)
        self.assertEqual(run['strategy_params'], overrides)
        self.assertEqual(self.lab.load_run(run['run_id'])['strategy_params'], overrides)
        for result in run['results']:
            for key, value in overrides[result['strategy']].items():
                self.assertEqual(result['params'][key], value)
            self.assertEqual(result['params']['top_k'], 1)
            self.assertEqual(result['params']['retrieval_mode'], 'keyword')
            self.assertLessEqual(result['included_count'], 1)
            self.assertLessEqual(result['context_chars'], 600)

    def test_unused_parent_size_does_not_restrict_large_fixed_chunks(self):
        (self.lab.data_dir / 'handbook.md').write_text('# 测试手册\n\n[F01] ' + '苹果福利条款。' * 300, encoding='utf-8')
        built = self.lab.build(['fixed'], self.shared, {},
                               strategy_params={'fixed': {'chunk_size': 1200, 'chunk_overlap': 50}})
        self.assertEqual(built['indexes'][0]['params']['chunk_size'], 1200)
        self.assertGreater(built['indexes'][0]['stats']['max_chars'], 900)
        preview = self.lab.preview('fixed', {'chunk_size': 1200, 'parent_size': 900}, {})
        self.assertTrue(preview['chunks'])

    def test_semantic_does_not_inherit_unused_overlap_or_parent_constraint(self):
        _, resolved = engine.resolve_strategy_params(['semantic'],
            {'chunk_size': 1200, 'chunk_overlap': 500, 'parent_size': 280},
            {'semantic': {'chunk_size': 80, 'semantic_percentile': 90}})
        self.assertEqual(resolved['semantic']['chunk_overlap'], 0)
        self.assertEqual(resolved['semantic']['semantic_percentile'], 90)
        self.assertGreaterEqual(resolved['semantic']['parent_size'], 80)

    def test_invalid_overrides_fail_before_cloud_work(self):
        invalid = [[], {'unknown': {}}, {'fixed': []}, {'fixed': {'top_k': 8}},
                   {'fixed': {'parent_size': 1000}}, {'fixed': {'chunk_size': 79}},
                   {'fixed': {'chunk_size': 80, 'chunk_overlap': 80}},
                   {'parent_child': {'chunk_size': 600, 'parent_size': 300}},
                   {'semantic': {'semantic_percentile': True}},
                   {'contextual': {'chunk_size': 79}}]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.lab.retrieve(['fixed'], '苹果福利', {'retrieval_mode': 'vector'}, {},
                                  strategy_params=overrides)
        self.assertEqual(RecordingEmbedder.calls, [])
        self.assertEqual(RecordingChat.calls, [])

    def test_flat_params_remain_compatible_fallback(self):
        run = self.lab.retrieve(['fixed', 'recursive'], '苹果福利',
            {**self.shared, 'chunk_size': 180, 'chunk_overlap': 15}, {})
        self.assertEqual(run['strategy_params']['fixed'], {'chunk_size': 180, 'chunk_overlap': 15})
        self.assertEqual(run['strategy_params']['recursive'], {'chunk_size': 180, 'chunk_overlap': 15})


if __name__ == '__main__':
    unittest.main()
