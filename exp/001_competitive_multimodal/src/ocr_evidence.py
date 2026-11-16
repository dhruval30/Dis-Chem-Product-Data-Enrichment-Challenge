from __future__ import annotations

import re

from .common import digest


def expand_spans(evidence):
    """Reconstruct existing detections, not OCR or new image crops."""
    for parent in list(evidence.sources.values()):
        if parent.kind != 'ocr' or parent.metadata.get('derived_ocr'):
            continue
        path = parent.location.split('/block/', 1)[0]
        parts = {digest([p.get('text'), p.get('box'), p.get('angle')]): p
                 for line in parent.metadata.get('lines', []) for p in line.get('parts', [])
                 if p.get('text') and p.get('box')}
        for key, part in parts.items():
            role = 'nutrition_basis' if re.fullmatch(r'100\s*(?:g|ml)', part['text'], re.I) and re.search(
                r'NUTRITION|PER\s*100|PER\s*25', parent.text, re.I) else None
            meta = {'derived_ocr': 'span', 'image_path': path, 'parent_source_id': parent.id,
                    'box': part['box'], 'angle': part.get('angle', 0), 'panel_role': role}
            evidence.source('ocr', path + '/span/' + key[:16], part['text'], part.get('confidence', 0),
                            exact=parent.exact_product, eligible=parent.eligible, metadata=meta)
            net = re.search(r'\b\d+(?:[.,]\d+)?\s*(?:ml|g|kg)\s*net', part['text'], re.I)
            if net:
                if (part.get('confidence') or 0) >= .95:
                    evidence.source('ocr', path + '/net/' + key[:16], net.group(), part['confidence'],
                        exact=parent.exact_product, eligible=parent.eligible,
                        metadata={**meta, 'derived_ocr': 'net_quantity', 'quote': net.group()})
                else:
                    issue = {'kind': 'low_confidence_net_quantity', 'image': path,
                             'text': net.group(), 'confidence': part.get('confidence')}
                    if issue not in evidence.issues:
                        evidence.issues.append(issue)
            if role:
                for source in evidence.sources.values():
                    if source.kind == 'ocr' and source.location == path and source.text == part['text']:
                        source.metadata['panel_role'] = role
        # Storage columns often sit beside nutrition tables. Follow geometry, not merged row order.
        for anchor in parts.values():
            if not re.match(r'^\s*(?:STORE\s+IN\s*A?\b|KEEP REFRIGERATED)', anchor['text'], re.I):
                continue
            box = anchor['box']
            cx = sum(p[0] for p in box) / len(box)
            cy = sum(p[1] for p in box) / len(box)
            width = max(p[0] for p in box) - min(p[0] for p in box)
            height = max(p[1] for p in box) - min(p[1] for p in box)
            candidates = [p for p in parts.values() if p.get('angle', 0) == anchor.get('angle', 0) and
                          abs(sum(b[0] for b in p['box']) / len(p['box']) - cx) < width * .65 and
                          sum(b[1] for b in p['box']) / len(p['box']) >= cy - height * .2]
            candidates.sort(key=lambda p: (sum(b[1] for b in p['box']) / len(p['box']), min(b[0] for b in p['box'])))
            chosen, previous_y = [], cy
            for part in candidates[:14]:
                y = sum(b[1] for b in part['box']) / len(part['box'])
                if y - previous_y > height * 3 or re.search(r'Important\s*Notice|Warnings?|Distribution|Nutrition', part['text'], re.I):
                    break
                if (part.get('confidence') or 0) < .9:
                    break
                chosen.append(part)
                previous_y = y
            if not chosen:
                continue
            text = '\n'.join(p['text'] for p in chosen)
            key = digest([(p['text'], p['box']) for p in chosen])[:16]
            evidence.source('ocr', path + '/column/' + key, text, min(p['confidence'] for p in chosen),
                exact=parent.exact_product, eligible=parent.eligible,
                metadata={'derived_ocr': 'column', 'image_path': path, 'parent_source_id': parent.id,
                          'panel_role': 'storage', 'lines': [{'text': p['text'], 'confidence': p['confidence'],
                                                              'parts': [p]} for p in chosen]})


def formulation_disagreement(row, sources):
    text = ' '.join(s.text for s in sources if s.eligible and (s.confidence or 0) >= .95)
    if not re.search(r'standard\s*fib(?:re|er)', text, re.I):
        return None
    target = row.get('Product Name', '')
    if re.search(r'\bstandard\b', target, re.I) and not re.search(r'fib(?:re|er)', target, re.I):
        description = row.get('Short Description', '') + ' ' + row.get('Product Description', '')
        if re.search(r'without\b.{0,45}dietary\s+fib(?:re|er)|fib(?:re|er)[ -]free', description, re.I):
            return 'label_fibre_formulation_contradicts_supplied_description'
        return 'additional_fibre_formulation_requires_review'
    return None
