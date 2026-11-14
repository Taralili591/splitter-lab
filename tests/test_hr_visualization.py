"""Exact source-offset checks for the colored original/chunk comparison.

The rendering metadata must follow actual intervals, not repeated text matches,
the configured overlap target, generated context, or UTF-16 string offsets.
"""
from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from labs.hr_chunking_lab.chunking import build_chunks, parse_document
from labs.hr_chunking_lab.engine import LabEngine
from labs.hr_chunking_lab.visualization import build_visualization


def chunk(text, name, start, end, **extra):
    result = {
        'id': name, 'start': start, 'end': end, 'text': text[start:end],
        'index_text': text[start:end], 'context': '', 'source_spans': [],
        'section': '', 'parent_id': None, 'parent_text': '', 'parent_spans': [],
    }
    result.update(extra)
    return result


class VisualizationTests(unittest.TestCase):
    def check_partition(self, text, chunks, result):
        self.assertEqual(result['unit'], 'unicode_codepoint')
        segments = result['segments']
        cursor = 0
        expected_overlaps = 0
        indices = {item['id']: item['index'] for item in result['chunks']}
        for segment in segments:
            self.assertEqual(segment['start'], cursor)
            self.assertLess(segment['start'], segment['end'])
            self.assertLessEqual(segment['end'], len(text))
            for offset in range(segment['start'], segment['end']):
                actual_ids = {item['id'] for item in chunks if item['start'] <= offset < item['end']}
                self.assertEqual(set(segment['chunk_ids']), actual_ids)
                self.assertEqual(set(segment['chunk_indices']), {indices[i] for i in actual_ids})
                self.assertEqual(segment['is_overlap'], len(actual_ids) > 1)
                expected_overlaps += len(actual_ids) > 1
            cursor = segment['end']
        self.assertEqual(cursor, len(text))
        self.assertEqual(''.join(text[s['start']:s['end']] for s in segments), text)
        self.assertEqual(result['overlap_chars'], expected_overlaps)

        # Each chunk's gold-colored ranges identify the other owners precisely,
        # even when a two-way overlap changes into a three-way overlap.
        for item in result['chunks']:
            expected = {
                offset for offset in range(item['start'], item['end'])
                if sum(c['start'] <= offset < c['end'] for c in chunks) > 1
            }
            found = set()
            for overlap in item['overlap_ranges']:
                self.assertEqual(overlap['char_count'], overlap['end'] - overlap['start'])
                for offset in range(overlap['start'], overlap['end']):
                    self.assertNotIn(offset, found)
                    found.add(offset)
                    other_ids = {
                        c['id'] for c in chunks
                        if c['id'] != item['id'] and c['start'] <= offset < c['end']
                    }
                    self.assertEqual(set(overlap['with_ids']), other_ids)
            self.assertEqual(found, expected)

    def test_two_way_overlap_and_uncovered_source_tail(self):
        text = 'abcdefghijklm'
        chunks = [chunk(text, 'a', 0, 8), chunk(text, 'b', 5, 12)]
        result = build_visualization(text, chunks)
        self.check_partition(text, chunks, result)
        self.assertEqual(result['overlap_chars'], 3)
        self.assertEqual(result['segments'][-1]['chunk_ids'], [])
        self.assertEqual(result['segments'][-1]['start'], 12)

    def test_three_way_overlap_preserves_all_owners(self):
        text = 'abcdefghijklm'
        chunks = [chunk(text, 'a', 0, 8), chunk(text, 'b', 3, 11), chunk(text, 'c', 6, 13)]
        result = build_visualization(text, chunks)
        self.check_partition(text, chunks, result)
        self.assertEqual(result['overlap_chars'], 8)
        triple = next(s for s in result['segments'] if s['start'] <= 6 < s['end'])
        self.assertEqual(set(triple['chunk_ids']), {'a', 'b', 'c'})

    def test_recursive_uses_actual_overlap_instead_of_target(self):
        text = '短句。' * 40
        settings = {'chunk_size': 25, 'chunk_overlap': 7, 'parent_size': 80}
        chunks = [asdict(c) for c in build_chunks(parse_document(text), 'recursive', settings)]
        actual = [max(0, a['end'] - b['start']) for a, b in zip(chunks, chunks[1:])]
        self.assertTrue(actual)
        self.assertTrue(any(n != settings['chunk_overlap'] for n in actual))
        self.check_partition(text, chunks, build_visualization(text, chunks))

    def test_no_overlap_and_repeated_content_are_not_marked_as_overlap(self):
        text = '相同文字相同文字'
        chunks = [chunk(text, 'first', 0, 4), chunk(text, 'second', 4, 8)]
        self.assertEqual(chunks[0]['text'], chunks[1]['text'])
        result = build_visualization(text, chunks)
        self.check_partition(text, chunks, result)
        self.assertEqual(result['overlap_chars'], 0)
        self.assertTrue(all(not s['is_overlap'] for s in result['segments']))
        self.assertTrue(all(not item['overlap_ranges'] for item in result['chunks']))

    def test_emoji_offsets_are_python_unicode_codepoints(self):
        text = '甲😀乙🧑‍💻丙'
        chunks = [chunk(text, 'a', 0, 4), chunk(text, 'b', 2, len(text))]
        result = build_visualization(text, chunks)
        self.check_partition(text, chunks, result)
        overlap = next(s for s in result['segments'] if s['is_overlap'])
        self.assertEqual(text[overlap['start']:overlap['end']], '乙🧑')
        self.assertEqual(result['overlap_chars'], 2)

    def test_parent_ranges_come_from_source_spans_and_keep_child_mapping(self):
        text = '甲乙丙丁甲乙丙丁'
        spans = lambda start, end: [{'start': start, 'end': end, 'clause_id': '', 'clause_start': start, 'clause_end': end}]
        chunks = [
            chunk(text, 'a', 0, 3, parent_id='p1', parent_text=text[:4], parent_spans=spans(0, 4)),
            chunk(text, 'b', 2, 4, parent_id='p1', parent_text=text[:4], parent_spans=spans(0, 4)),
            chunk(text, 'c', 4, 8, parent_id='p2', parent_text=text[4:], parent_spans=spans(4, 8)),
        ]
        result = build_visualization(text, chunks)
        self.check_partition(text, chunks, result)
        parents = {p['id']: p for p in result['parents']}
        self.assertEqual((parents['p1']['start'], parents['p1']['end']), (0, 4))
        self.assertEqual((parents['p2']['start'], parents['p2']['end']), (4, 8))
        self.assertEqual(set(parents['p1']['child_ids']), {'a', 'b'})
        self.assertEqual(set(parents['p2']['child_ids']), {'c'})
        self.assertNotEqual(parents['p1']['color_index'], parents['p2']['color_index'])

    def test_generated_background_does_not_create_source_intervals(self):
        text = '原文只有这几个字。'
        c = chunk(text, 'a', 0, len(text), context='原文中不存在的背景解释', index_text='额外背景\n' + text)
        result = build_visualization(text, [c])
        self.check_partition(text, [c], result)
        self.assertEqual(result['overlap_chars'], 0)
        self.assertEqual(result['segments'][0]['end'], len(text))

    def test_empty_chunks_leave_original_uncolored_without_fabricated_cut(self):
        text = '# 原文\n\n尚未调用语义模型。'
        result = build_visualization(text, [])
        self.check_partition(text, [], result)
        self.assertEqual(result['chunks'], [])
        self.assertEqual(result['parents'], [])
        self.assertTrue(all(not s['chunk_ids'] and not s['is_overlap'] for s in result['segments']))

    def test_structure_and_parent_children_share_exact_original_text(self):
        text = '# 手册\n\n## 假期\n\n[F01] 满五年有额外假。\n\n## 申请\n\n[L01] 提前申请，由主管批准。'
        doc = parse_document(text)
        for strategy in ['fixed', 'recursive', 'structure', 'contextual', 'parent_child']:
            with self.subTest(strategy=strategy):
                chunks = [asdict(c) for c in build_chunks(doc, strategy, {'chunk_size': 20, 'chunk_overlap': 4, 'parent_size': 40})]
                self.check_partition(text, chunks, build_visualization(text, chunks))

    def test_invalid_bounds_are_rejected(self):
        text = 'abcdefgh'
        for start, end in [(-1, 3), (0, 9), (5, 4), (3, 3), (True, 3), (0, 3.5)]:
            bad = {'id': 'bad', 'start': start, 'end': end, 'text': 'bad'}
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                build_visualization(text, [bad])

    def test_preview_returns_original_and_visualization_without_cloud_requests(self):
        text = '# 试验\n\n[F01] ' + '公司福利假按年限计算。😀' * 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'data/handbook.md').write_text(text)
            lab = LabEngine(root / 'data', root / 'runtime')
            with patch('requests.post') as remote:
                preview = lab.preview('fixed', {'chunk_size': 80, 'chunk_overlap': 20}, {})
                self.assertEqual(preview['document']['text'], text)
                self.assertEqual(preview['document']['char_count'], len(text))
                self.check_partition(text, preview['chunks'], preview['visualization'])
                semantic = lab.preview('semantic', {'chunk_size': 80, 'chunk_overlap': 20}, {})
                self.assertTrue(semantic['needs_build'])
                self.assertEqual(semantic['document']['text'], text)
                self.assertEqual(semantic['visualization']['chunks'], [])
                remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
