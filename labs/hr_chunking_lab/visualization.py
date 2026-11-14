"""Exact source ranges for chunk colours; presentation never changes the text.

Offsets use Python Unicode code points (start inclusive, end exclusive). The
browser must slice Array.from(document.text), rather than UTF-16 string indices.
Gold overlap means simultaneous coverage at the same original position, not two
matching strings somewhere in the document.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

PALETTE_SIZE = 6


def _range(value: dict, text_length: int, label: str) -> tuple[int,int]:
    start,end=value.get('start'),value.get('end')
    if type(start) is not int or type(end) is not int or not 0<=start<end<=text_length:
        raise ValueError(f'{label} 的原文范围无效。')
    return start,end


def build_visualization(document_text: str, chunks: list[dict[str,Any]]) -> dict:
    if not isinstance(document_text,str) or not isinstance(chunks,list):
        raise ValueError('颜色对照需要原文文字与资料块列表。')
    length=len(document_text)
    by_id={}
    starts=defaultdict(set)
    ends=defaultdict(set)
    boundaries={0,length}
    visual_chunks=[]
    parents={}
    for index,chunk in enumerate(chunks):
        if not isinstance(chunk,dict):
            raise ValueError('资料块必须为对象。')
        cid=chunk.get('id')
        if not isinstance(cid,str) or not cid or cid in by_id:
            raise ValueError('资料块编号为空或重复。')
        start,end=_range(chunk,length,'资料块')
        if 'text' in chunk and chunk['text']!=document_text[start:end]:
            raise ValueError('资料块文字与对应原文范围不一致，不能绘制错误颜色。')
        by_id[cid]=index
        starts[start].add(index)
        ends[end].add(index)
        boundaries.update((start,end))
        visual_chunks.append({'id':cid,'index':index,'start':start,'end':end,
                              'color_index':index%PALETTE_SIZE,'overlap_ranges':[],
                              'parent_id':chunk.get('parent_id')})
        parent_id=chunk.get('parent_id')
        if parent_id:
            pspans=chunk.get('parent_spans',[])
            if not isinstance(pspans,list) or not pspans:
                raise ValueError('父块缺少精确原文范围。')
            intervals=sorted(_range(span,length,'父块') for span in pspans)
            left,right=intervals[0]
            for a,b in intervals[1:]:
                if a>right:
                    raise ValueError('父块原文范围不连续。')
                right=max(right,b)
            if not left<=start<end<=right:
                raise ValueError('子块不在对应的父块范围内。')
            if 'parent_text' in chunk and chunk['parent_text']!=document_text[left:right]:
                raise ValueError('父块文字与对应原文范围不一致。')
            if parent_id not in parents:
                parents[parent_id]={'id':parent_id,'start':left,'end':right,
                                    'child_ids':[],'color_index':len(parents)%PALETTE_SIZE}
            elif (parents[parent_id]['start'],parents[parent_id]['end'])!=(left,right):
                raise ValueError('同一个父块出现不一致的范围。')
            parents[parent_id]['child_ids'].append(cid)

    segments=[]
    active=set()
    cuts=sorted(boundaries)
    overlap_chars=0
    for start,end in zip(cuts,cuts[1:]):
        active.difference_update(ends[start])
        active.update(starts[start])
        if end<=start:
            continue
        indices=sorted(active)
        ids=[visual_chunks[i]['id'] for i in indices]
        overlap=len(indices)>1
        segments.append({'start':start,'end':end,'chunk_ids':ids,
                         'chunk_indices':indices,'is_overlap':overlap})
        if not overlap:
            continue
        overlap_chars+=end-start
        for i in indices:
            with_ids=[visual_chunks[j]['id'] for j in indices if j!=i]
            ranges=visual_chunks[i]['overlap_ranges']
            if ranges and ranges[-1]['end']==start and ranges[-1]['with_ids']==with_ids:
                ranges[-1]['end']=end
                ranges[-1]['char_count']+=end-start
            else:
                ranges.append({'start':start,'end':end,'with_ids':with_ids,
                               'char_count':end-start})
    return {'segments':segments,'chunks':visual_chunks,'parents':list(parents.values()),
            'overlap_chars':overlap_chars,'unit':'unicode_codepoint'}
