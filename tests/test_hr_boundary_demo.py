"""Controlled boundary comparisons use real source coordinates and no models."""
import unittest
from unittest.mock import patch

from labs.hr_chunking_lab.boundary_demo import boundary_demo


class BoundaryDemoTests(unittest.TestCase):
    def test_short_chapters_fit_together_but_structure_keeps_them_separate(self):
        data = boundary_demo()
        self.assertLess(data['document']['char_count'], 280)
        self.assertEqual(len(data['document']['sections']), 2)
        recursive, structure = data['methods']
        self.assertEqual((recursive['chunk_count'], recursive['cross_section_count']), (1, 1))
        self.assertEqual((structure['chunk_count'], structure['cross_section_count']), (2, 0))
        self.assertEqual(recursive['chunks'][0]['section_indices'], [0, 1])
        self.assertEqual([chunk['section_indices'] for chunk in structure['chunks']], [[0], [1]])

    def test_tight_limit_can_make_the_two_methods_agree(self):
        # The demo must report real outcomes, rather than force every setting
        # to produce a dramatic difference.
        data = boundary_demo('short', 80)
        ranges = [[(c['start'], c['end']) for c in method['chunks']]
                  for method in data['methods']]
        self.assertEqual(ranges[0], ranges[1])
        self.assertEqual(data['comparison']['same_range_count'], len(ranges[0]))

    def test_long_chapters_are_split_inside_each_section(self):
        data = boundary_demo('long', 280)
        structure = data['methods'][1]
        self.assertEqual(structure['cross_section_count'], 0)
        self.assertGreater(structure['chunk_count'], len(data['document']['sections']))
        for section_index in range(2):
            self.assertGreater(sum(c['section_indices'] == [section_index]
                                   for c in structure['chunks']), 1)

    def test_every_displayed_character_projects_to_exact_original_source(self):
        for sample in ('short', 'long'):
            for size in (80, 140, 280, 600):
                with self.subTest(sample=sample, size=size):
                    data = boundary_demo(sample, size)
                    text = data['document']['text']
                    self.assertEqual(data['params'], {'chunk_size': size, 'chunk_overlap': 0})
                    for method in data['methods']:
                        self.assertEqual(''.join(c['text'] for c in method['chunks']), text)
                        for chunk in method['chunks']:
                            self.assertEqual(chunk['text'], text[chunk['start']:chunk['end']])
                            self.assertEqual(chunk['char_count'], len(chunk['text']))
                            self.assertLessEqual(chunk['char_count'], size)
                            self.assertEqual(''.join(part['text'] for part in chunk['parts']), chunk['text'])
                            self.assertNotIn('index_text', chunk)
                            for part in chunk['parts']:
                                self.assertEqual(part['text'], text[part['start']:part['end']])
                                section = data['document']['sections'][part['section_index']]
                                self.assertGreaterEqual(part['start'], section['start'])
                                self.assertLessEqual(part['end'], section['end'])

    def test_bad_samples_and_lengths_are_rejected_without_remote_work(self):
        with patch('requests.post') as remote:
            for sample in ('missing', None, [], 1):
                with self.subTest(sample=sample), self.assertRaises(ValueError):
                    boundary_demo(sample)
            for size in (True, False, None, '280', 280.0, 79, 601):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    boundary_demo('short', size)
            data = boundary_demo()
            self.assertEqual(data['controls'], {
                'local_only': True, 'source_only': True, 'title_prefix_added': False})
            remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
