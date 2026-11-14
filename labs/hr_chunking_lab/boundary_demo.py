"""A local, source-only comparison of recursive and section-first boundaries.

Both methods use the real chunker, the same character limit and zero overlap.
This teaching view deliberately stops before indexing, retrieval and generation:
the section title prefix used by the full retrieval experiment is not displayed
as source text and cannot account for any boundary difference in this view.
"""
from __future__ import annotations

from pathlib import Path

from .chunking import build_chunks, parse_document


DATA_DIR = Path(__file__).parent / 'data'
SAMPLES = {
    'short': {
        'filename': 'boundary_demo_short.md',
        'label': '短章节：合起来也装得下',
        'description': '两章合计不足 280 字符。观察递归是否合并两章，以及结构切块是否仍在章节边界分开。',
    },
    'long': {
        'filename': 'boundary_demo_long.md',
        'label': '长章节：章内仍需继续拆',
        'description': '每章都超过默认块长。结构切块先限定章节，再在章内使用同一套递归规则。',
    },
}


def boundary_demo(sample='short', chunk_size=280, *, data_dir=None):
    """Return exact source ranges for two genuine, controlled chunking runs."""
    if not isinstance(sample, str) or sample not in SAMPLES:
        raise ValueError('请选择 short 或 long 对照样例。')
    if type(chunk_size) is not int or not 80 <= chunk_size <= 600:
        raise ValueError('对照块长须为 80–600 字符之间的整数。')
    info = SAMPLES[sample]
    text = (Path(data_dir or DATA_DIR) / info['filename']).read_text(encoding='utf-8')
    document = parse_document(text)
    sections = [
        {'title': section.title, 'start': section.start, 'end': section.end,
         'color_index': index}
        for index, section in enumerate(document.sections)
    ]
    params = {'chunk_size': chunk_size, 'chunk_overlap': 0}
    methods = []
    for strategy, label in [('recursive', '递归切块'), ('structure', '文档结构切块')]:
        chunks = []
        for chunk in build_chunks(document, strategy, params):
            parts = []
            for index, section in enumerate(sections):
                start, end = max(chunk.start, section['start']), min(chunk.end, section['end'])
                if start < end:
                    parts.append({'start': start, 'end': end, 'text': text[start:end],
                                  'section_index': index, 'title': section['title']})
            section_indices = [part['section_index'] for part in parts]
            chunks.append({
                'id': chunk.id, 'start': chunk.start, 'end': chunk.end,
                'text': chunk.text, 'char_count': len(chunk.text),
                'parts': parts, 'section_indices': section_indices,
                'crosses_sections': len(section_indices) > 1,
            })
        methods.append({
            'strategy': strategy, 'label': label, 'chunks': chunks,
            'chunk_count': len(chunks),
            'cross_section_count': sum(chunk['crosses_sections'] for chunk in chunks),
        })
    ranges = [{(chunk['start'], chunk['end']) for chunk in method['chunks']}
              for method in methods]
    return {
        'sample': sample, 'label': info['label'], 'description': info['description'],
        'document': {'text': text, 'char_count': len(text), 'sections': sections},
        'params': params, 'methods': methods,
        'comparison': {'same_range_count': len(ranges[0] & ranges[1])},
        'controls': {'local_only': True, 'source_only': True, 'title_prefix_added': False},
    }
