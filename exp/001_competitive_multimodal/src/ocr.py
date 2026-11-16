from __future__ import annotations

import gc
import re

from .common import ROOT, digest, file_hash, versions
from .image_processing import load_image


def group_lines(lines):
    """Retain reading order within each orientation and separate distant panels."""
    blocks = []
    for angle in sorted({x['angle'] for x in lines}):
        ordered = sorted((x for x in lines if x['angle'] == angle),
                         key=lambda x: (min(p[1] for p in x['box']), min(p[0] for p in x['box'])))
        rows = []
        for line in ordered:
            top = min(p[1] for p in line['box'])
            bottom = max(p[1] for p in line['box'])
            centre = (top + bottom) / 2
            height = max(1, bottom - top)
            if rows and abs(centre - rows[-1]['centre']) < .5 * max(height, rows[-1]['height']):
                rows[-1]['items'].append(line)
                rows[-1]['bottom'] = max(rows[-1]['bottom'], bottom)
            else:
                rows.append({'items': [line], 'top': top, 'bottom': bottom, 'centre': centre, 'height': height})
        current = []
        previous = None
        for row in rows:
            row['items'].sort(key=lambda x: min(p[0] for p in x['box']))
            if previous and row['top'] - previous['bottom'] > 3 * max(row['height'], previous['height']):
                blocks.append(current)
                current = []
            text = ' '.join(x['text'] for x in row['items'])
            current.append({'text': text, 'confidence': min(x['confidence'] for x in row['items']),
                            'parts': row['items']})
            previous = row
        if current:
            blocks.append(current)
    result = []
    for rows in blocks:
        total = sum(max(len(x['text']), 1) for x in rows)
        result.append({'text': '\n'.join(x['text'] for x in rows), 'lines': rows,
                       'confidence': sum(x['confidence'] * max(len(x['text']), 1) for x in rows) / total})
    return result


class OCR:
    def __init__(self, cfg, cache, budget, max_images):
        self.cfg, self.cache, self.budget = cfg, cache, budget
        self.max_images = max_images
        self.engine = None

    def run(self, path):
        key = digest([file_hash(path), self.cfg, versions(),
                      file_hash(ROOT / 'src/ocr.py'), file_hash(ROOT / 'src/image_processing.py')])
        cached = self.cache.get('ocr', key)
        if cached is not None:
            return cached
        if self.budget.state['ocr_images'] >= self.max_images:
            return {'lines': [], 'raw': [], 'blocks': [], 'skipped': 'stage_limit'}
        self.budget.check()
        self.cache.attempts('ocr', key, 1 + self.budget.cfg['retry_count'])
        if self.engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        import numpy as np
        image = load_image(path, self.cfg['max_side'], self.cfg['contrast'])
        raw, accepted, seen = [], [], set()
        for angle in self.cfg['rotations']:
            if angle and sum(len(x['text']) for x in accepted) >= self.cfg['rotate_if_chars_below']:
                break
            # RapidOCR's ndarray interface expects BGR, rather than PIL's RGB.
            result, elapsed = self.engine(np.asarray(image.rotate(angle, expand=True))[:, :, ::-1])
            for box, text, confidence in result or []:
                item = {'text': str(text), 'confidence': float(confidence), 'angle': angle,
                        'box': np.asarray(box).tolist(), 'elapsed': elapsed,
                        'ambiguous': bool(confidence < self.cfg['min_confidence'] or
                                          re.search(r'\d', text) and confidence < .95)}
                raw.append(item)
                normal = ' '.join(text.split())
                if confidence >= self.cfg['min_confidence'] and normal not in seen:
                    accepted.append({**item, 'text': normal})
                    seen.add(normal)
        value = {'lines': accepted, 'raw': raw, 'blocks': group_lines(raw), 'image_hash': file_hash(path)}
        self.budget.state['ocr_images'] += 1
        self.cache.put('ocr', key, value)
        self.budget.save()
        return value

    def close(self):
        self.engine = None
        gc.collect()
