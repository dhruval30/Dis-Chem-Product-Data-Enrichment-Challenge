from __future__ import annotations

import re

from .evidence import normalise
from .text_extraction import extract, extract_nutrition_text, literal_pack_quantity, quantity_relation, pack_quantity
from .verification import canonical, quote_confidence
from rapidfuzz import fuzz
from .common import ROOT

VERSION = 'visual-evidence-v2'
OBSERVATIONS = {'product_form', 'colour', 'pattern', 'decoration', 'shape', 'packaging_form', 'design', 'visible_count'}
FORBIDDEN = re.compile(r'\b(?:cotton|polyester|silicone|latex|leather|wool|silk|pearls?|diamonds?|sterile|safe|suitable|hypoallergenic|antibacterial|organic|non.?toxic|waterproof|medical|therapeutic|prevents?|treats?|cures?|efficacy|ingredient|allergen)\b|\d+(?:[.,]\d+)?\s*(?:cm|mm|kg|mg|ml|inch|%)\b', re.I)
UNSEEN_PROPERTIES = re.compile(r'\b(?:soft|absorbent|breathable|durable|comfortable|washable|newborn|sterilisable|bpa.free|non.?slip)\b', re.I)
PLACEHOLDER = re.compile(r'\b(?:brief observed feature|specific visual support|example value|insert (?:text|value)|your (?:answer|description) here)\b', re.I)


def validate_response(response):
    from jsonschema import Draft202012Validator
    errors = list(Draft202012Validator(response_schema()).iter_errors(response))
    if errors:
        raise ValueError('Invalid visual evidence response: ' + errors[0].message)
    for item in response['images']:
        texts = [item['identity_evidence']]
        texts += [b['text'] for b in item['printed']]
        texts += [s for o in item['observations'] for s in (o['value'], o['evidence'])]
        if any(PLACEHOLDER.search(text) for text in texts):
            raise ValueError('Visual response contains prompt placeholders instead of image evidence')


def instructions():
    return (ROOT / 'prompts/image_extraction.txt').read_text()


def response_schema():
    string = {'type': 'string'}
    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
    shape = obj({'schema_version': {'const': VERSION}, 'images': {'type': 'array', 'minItems': 1, 'items': obj({
        'image_index': {'type': 'integer', 'minimum': 0},
        'identity': {'enum': ['confirmed', 'unverifiable', 'mismatch']}, 'identity_evidence': string,
        'visible_brand': string, 'visible_name': string, 'visible_variant': string,
        'visible_quantity': string, 'target_variant_quote': string,
        'printed': {'type': 'array', 'items': obj({'kind': {'enum': ['identity', 'quantity', 'ingredients_declaration', 'highlighted_ingredient', 'claim', 'storage', 'usage', 'warning', 'nutrition', 'specification']}, 'text': string})},
        'observations': {'type': 'array', 'items': obj({'attribute': {'enum': sorted(OBSERVATIONS)}, 'value': string, 'evidence': string})}
    })}})
    shape['properties']['images']['items']['required'] = ['image_index', 'identity', 'identity_evidence', 'printed', 'observations']
    return shape


def aligned_ocr(evidence, path):
    path = str(path)
    return [s for s in evidence.sources.values() if s.kind == 'ocr' and
            (s.location == path or s.location.startswith(path + '/'))]


def corroborated(text, sources, minimum=.95):
    numbers = re.findall(r'[<>≤≥]?\s*\d+(?:[.,]\d+)?', text)
    compact = lambda value: re.sub(r'\W+', '', canonical(value))
    for source in sources:
        if not source.eligible:
            continue
        if not compact(text) or compact(text) not in compact(source.text):
            continue
        # Percentages and mass/volume units must agree beside the same number.
        measurements = re.findall(r'(\d+(?:[.,]\d+)?)\s*(%|(?:mg|kg|ml|g|kcal|kj)(?!\w))', text, re.I)
        if any(not re.search(r'(?<![\d.,])' + re.escape(value).replace(r'\.', '[.,]').replace(',', '[.,]') +
                             r'\s*' + re.escape(unit) + (r'(?:(?!\w)|(?=net\b))' if unit != '%' else ''), source.text, re.I)
               for value, unit in measurements):
            continue
        confidence = quote_confidence(source, text)
        if confidence is None or confidence < minimum:
            lines = source.metadata.get('lines', [])
            matching = [(i, line) for i, line in enumerate(lines) if compact(line['text']) and
                        (compact(line['text']) in compact(text) or compact(text) in compact(line['text']))]
            joined = ''.join(compact(line['text']) for _, line in matching)
            consecutive = bool(matching) and [i for i, _ in matching] == list(range(matching[0][0], matching[-1][0] + 1))
            if not consecutive or compact(text) not in joined or any((line['confidence'] or 0) < minimum for _, line in matching):
                continue
        if all(re.search(r'(?<![\d.,])' + re.escape(re.sub(r'\s+', '', n)).replace(r'\.', '[.,]').replace(',', '[.,]') + r'(?![\d.,])', source.text) for n in numbers):
            return source.id
    return None


def printed_role(text, kind, row):
    if kind in ('identity', 'ingredients_declaration', 'storage', 'usage', 'warning', 'nutrition', 'highlighted_ingredient'):
        return kind
    if literal_pack_quantity(text, row.get('Product Name', '')):
        return 'quantity'
    if kind == 'quantity':
        return 'strength' if quantity_relation(text, row.get('Product Name', '')) == 'strength' else 'quantity'
    identity_text = canonical(text).replace('fragranced', 'fragrance')
    if identity_text and f' {identity_text} ' in f" {canonical(row.get('Product Name', ''))} ":
        return 'identity_fragment'
    if re.match(r'^No need to remove\b', text, re.I):
        return 'usage'
    if re.match(r'^(?:experience|discover|explore|unleash|indulge|welcome to)\b', text.strip(), re.I):
        return 'slogan'
    if kind == 'specification':
        return 'specification'
    return 'claim'


def import_response(response, paths, row, evidence, verification, contradiction):
    validate_response(response)
    seen = set()
    for item in response['images']:
        index = item['image_index']
        if index >= len(paths) or index in seen:
            evidence.issues.append({'kind': 'invalid_visual_image_index', 'image_index': index})
            continue
        seen.add(index)
        path = str(paths[index])
        checked_item = dict(item)
        if not checked_item.get('visible_quantity'):
            quantities = [b['text'] for b in item['printed'] if b['kind'] == 'quantity']
            if len(quantities) == 1:
                checked_item['visible_quantity'] = quantities[0]
        conflict = contradiction(checked_item, row)
        if conflict == 'quantity' and not corroborated(checked_item.get('visible_quantity', ''), aligned_ocr(evidence, path)):
            evidence.issues.append({'kind': 'unconfirmed_visual_quantity_conflict', 'image': path,
                                    'text': checked_item.get('visible_quantity', '')})
            conflict = None
        if any(s.metadata.get('identity_status') == 'mismatch' for s in aligned_ocr(evidence, path)):
            conflict = 'prior_image_mismatch'
        if any(s.metadata.get('identity_hold_reason') for s in aligned_ocr(evidence, path)):
            evidence.issues.append({'kind': 'image_formulation_requires_review', 'image': path})
            continue
        forms = {'sock', 'towel', 'cream', 'powder', 'soother', 'brush', 'cloth', 'shoe', 'blanket'}
        target_forms = {word.rstrip('s') for word in canonical(row['Product Name']).split()} & forms
        visible_forms = {word.rstrip('s') for obs in item['observations'] if obs['attribute'] == 'product_form'
                         for word in canonical(obs['value']).split()} & forms
        printed_names = item.get('visible_name', '') + ' ' + ' '.join(b['text'] for b in item['printed'] if b['kind'] == 'identity')
        visible_forms |= {word.rstrip('s') for word in canonical(printed_names).split()} & forms
        if target_forms and visible_forms and not target_forms & visible_forms:
            conflict = 'product_form'
        state = item['identity']
        explanation = item['identity_evidence']
        uncertainty = re.search(r'no (?:printed|readable|visible|identif)|unreadable|cannot (?:read|confirm)|not enough', explanation, re.I)
        reported_conflict = (state == 'mismatch' and not uncertainty and
            bool(item['printed'] or item['observations']) and
            bool(re.search(r'different|contradict|mismatch|expected|instead|whereas|does not match', explanation, re.I)))
        if conflict or reported_conflict:
            for source in aligned_ocr(evidence, path):
                source.eligible = False
                source.metadata['identity_status'] = 'mismatch'
            evidence.issues.append({'kind': 'image_product_mismatch', 'image': path,
                                    'attribute': conflict or 'reported_visual_identity', 'evidence': item['identity_evidence']})
            continue
        printed_identity = ' '.join(b['text'] for b in item['printed'] if b['kind'] == 'identity')
        name = canonical(item.get('visible_name') or printed_identity)
        brand = canonical(item.get('visible_brand', ''))
        target_brand = canonical(row.get('T_BRAND', ''))
        brand_matches = bool(target_brand and (brand and (fuzz.ratio(brand, target_brand) >= 85 or
                             f' {target_brand} ' in f' {brand} ' or f' {brand} ' in f' {target_brand} ') or
                             f' {target_brand} ' in f' {name} '))
        anchors = set(name.split()) & set(canonical(row['Product Name']).split()) - {'ml', 'g', 'kg', 'pack', 'and'}
        ocr_confirmed = any(s.eligible and s.exact_product and s.metadata.get('identity_status') == 'confirmed'
                            for s in aligned_ocr(evidence, path))
        confirmed = state == 'confirmed' and brand_matches and len(anchors) >= 2 or ocr_confirmed
        state = 'confirmed' if confirmed else 'unverifiable'
        for source in aligned_ocr(evidence, path):
            source.metadata['vlm_identity_status'] = state
            if source.metadata.get('identity_status') != 'confirmed':
                source.metadata['identity_status'] = state
        base = {'visual_schema': VERSION, 'image_index': index, 'image_path': path,
                'identity_status': state, 'identity_evidence': item['identity_evidence'],
                'reliability': 'direct_visible', 'identity_checked': confirmed,
                'mapped_image': True, 'confidence_basis': 'rule_gate_not_probability',
                'identity_basis': 'matching_ocr_identity' if ocr_confirmed else 'visible_brand_and_product',
                'vlm_identity_status': item['identity']}
        if ocr_confirmed and not item['printed']:
            evidence.issues.append({'kind': 'visual_printed_text_omitted_despite_ocr', 'image': path})
        for block in item['printed']:
            text, kind = block['text'].strip(), block['kind']
            if not text or '[MASKED]' in text.upper():
                continue
            role = printed_role(text, kind, row)
            literal = literal_pack_quantity(text, row.get('Product Name', ''))
            relation = quantity_relation(text, row.get('Product Name', '')) if literal else 'unverifiable'
            regulated = bool(re.search(r'VITAMIN|HERBAL|MEDIC|BABY.*SAFETY|BABY HEALTH', row.get('category_group', ''), re.I))
            minimum = .9 if literal and confirmed and not regulated else .95
            ocr_id = corroborated(text, aligned_ocr(evidence, path), minimum)
            high_risk = role in {'ingredients_declaration', 'storage', 'usage', 'warning', 'nutrition'}
            high_risk = high_risk or bool(re.search(r'VITAMIN|HERBAL|MEDIC|BABY.*SAFETY|BABY HEALTH', row.get('category_group', ''), re.I))
            high_risk = high_risk or bool(re.search(r'weight|height|width|depth|nutrition|per\s*100|per\s+serv|safe|suitable|hypoallergenic|antibacterial|clinically|\bSPF\b|allergen|may contain|\bfree.from\b|cures?|treats?|prevents?|medical|dosage|pregnan|children', text, re.I))
            high_risk = high_risk or bool(re.search(r'MATERNITY|BABY HEALTH', row.get('category_group', ''), re.I)
                and re.search(r'\b(?:soothe|protect|breastfeeding)\b', text, re.I))
            high_risk = high_risk or role in ('claim', 'specification', 'highlighted_ingredient') and bool(re.search(r'\d', text))
            high_risk = high_risk or role == 'claim' and bool(re.search(r'[À-ÿ]|\b(?:SENZA|SANS|OHNE)\b', text))
            eligible = bool(ocr_id or confirmed and not high_risk)
            scope = 'component' if relation == 'component' else 'whole'
            if role == 'quantity' and relation == 'different':
                eligible = False
                scope = 'conflicting'
            elif role == 'quantity' and (not literal or relation == 'unverifiable' and pack_quantity(row.get('Product Name', ''))):
                eligible = False
                scope = 'unresolved'
            if kind == 'ingredients_declaration' and ocr_id:
                complete = []
                for ocr_source in aligned_ocr(evidence, path):
                    if '/block/' not in ocr_source.location:
                        continue
                    parsed = extract({'Product Description': ocr_source.text}, kind='ocr')
                    complete.extend(parsed.sources[f.source_id].text for f in parsed.facts
                                    if f.field == 'ingredient_information')
                eligible = any(canonical(declaration) == canonical(text) for declaration in complete)
            meta = {**base, 'evidence_type': 'printed', 'printed_kind': kind,
                    'ocr_source_id': ocr_id, 'high_risk': high_risk,
                    'ocr_minimum_confidence': minimum, 'content_role': role,
                    'quantity_scope': scope,
                    'reliability': 'ocr_agreement' if ocr_id else 'direct_visible'}
            if kind == 'nutrition' and ocr_id:
                ocr_source = evidence.sources[ocr_id]
                meta['lines'] = [{'text': line, 'confidence': quote_confidence(ocr_source, line)}
                                 for line in text.splitlines() if line.strip()]
            source_id = evidence.source('visual_printed', path, text, None, exact=confirmed,
                                        eligible=eligible, metadata=meta)
            if not eligible:
                evidence.issues.append({'kind': 'visual_quantity_conflict' if scope == 'conflicting' else 'printed_evidence_requires_review', 'source_id': source_id})
                continue
            if role == 'quantity':
                if literal and scope == 'whole':
                    existing = [f.value for f in evidence.facts if f.field == 'product_specifications' and f.path == ['pack_size']]
                    if not any(quantity_relation(text, p['value'] + ' ' + p['unit']) == 'same' for p in existing):
                        evidence.add('product_specifications', ['pack_size'], literal[0], source_id,
                                     text, method='visual_printed')
            if kind == 'ingredients_declaration':
                evidence.add('ingredient_information', [], [text], source_id, text, method='visual_printed')
            elif kind == 'storage':
                evidence.add('storage_instructions', [], [text], source_id, text, method='visual_printed')
            elif kind == 'nutrition':
                extract_nutrition_text(evidence, source_id)
        for observation in item['observations']:
            attribute = observation['attribute']
            value = observation['value'].strip()
            support = observation['evidence'].strip()
            eligible = bool(value and support and not FORBIDDEN.search(value + ' ' + support)
                            and not UNSEEN_PROPERTIES.search(value + ' ' + support))
            if attribute == 'visible_count' and not re.fullmatch(r'\d+\s+(?:individual\s+)?(?:visible\s+)?[A-Za-z -]+', value):
                eligible = False
            if attribute == 'visible_count' and re.search(r'\b(?:pairs?|packs?|sets?)\b', value, re.I):
                eligible = False
            meta = {**base, 'evidence_type': 'observation', 'attribute': attribute,
                    'observation_value': value, 'visible_support': support,
                    'reliability': 'mapped_direct_observation'}
            source_id = evidence.source('visual_observation', path, value, None, exact=confirmed,
                eligible=eligible, metadata=meta)
            if not eligible:
                evidence.issues.append({'kind': 'unsupported_visual_observation', 'source_id': source_id})
