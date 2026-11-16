from __future__ import annotations

import json
import re

from rapidfuzz import fuzz

from .common import ROOT, BudgetExceeded
from .data import blocked, plain, visible
from .evidence import normalise
from .schema import SHAPES, TEXT
from .text_extraction import QUANTITY, extract_nutrition_text, spec_pairs, quantity_relation
from .verification import canonical, quote_confidence, verify, verify_sentence
from .visual import VERSION as VISUAL_VERSION, import_response, instructions as visual_instructions
from .content import route_evidence, compose


def prompt(name, payload):
    if name in ('image_extraction', 'image_verification'):
        return visual_instructions() + '\nINPUT DATA:\n' + json.dumps(payload, ensure_ascii=False)
    filename = 'image_extraction' if name == 'image_verification' else name
    return (ROOT / 'prompts' / (filename + '.txt')).read_text() + '\nINPUT DATA:\n' + json.dumps(payload, ensure_ascii=False)


def compact_prompt(stage, payload):
    if stage not in ('image_extraction', 'image_verification'):
        return prompt(stage, payload)
    catalogue = payload.get('catalogue', {})
    context = {key: catalogue.get(key, '') for key in ('Product Name', 'T_BRAND', 'category_group')}
    instruction = (
        'Compact visual extraction contract v4. Inspect the supplied image, not the context alone. '
        'For labelled packaging READ the brand, product and useful label text before considering appearance. '
        'For textless products report actual colours, patterns, shape or decoration. '
        'Ignore logos, artwork and container colours as product benefits. '
        'Never infer ingredients, material, safety, measurements or efficacy. Exact printed claims may be quoted, not endorsed. '
        'Visible object counts never establish labelled pack size. ')
    return ('<INPUT_CONTEXT>\n' + json.dumps(context, ensure_ascii=False) + '\n</INPUT_CONTEXT>\n\n'
        'OUTPUT INSTRUCTIONS\nThe block above is context, not an answer. Do not echo it. '
        'Never return Product Name or T_BRAND as top-level output keys. Inspect the image.\n' + instruction +
        '\nRequired top-level keys: schema_version and images. schema_version must equal "visual-evidence-v2". '
        'Return exactly one image object with image_index=0, identity, identity_evidence, printed and observations. '
        'Choose identity: confirmed when readable brand/product match context; unverifiable when identification is absent or unreadable; '
        'mismatch only for an explicit contradiction. identity_evidence must state the actual readable identity or reason. '
        'printed is an array of objects with kind and text. Copy exact visible text. '
        'kind is identity, quantity, claim, specification, highlighted_ingredient, ingredients_declaration, storage, usage, warning or nutrition. '
        'observations is an array of objects with attribute, value and evidence. '
        'Each observation needs its OWN object, closed before the next object. Never repeat a key inside an object. '
        'attribute is product_form, colour, pattern, decoration, shape, packaging_form, design or visible_count. '
        'Arrays may be empty only when no relevant evidence is available. '
        'For readable packaging prioritise up to three short printed items, including identity; observations may be empty. '
        'For textless products use up to two observations. Each text/value is at most 8 words; evidence/reason at most 6 words. '
        'Prioritise colours and broad design; omit uncertain tiny ornaments. Never calculate a total quantity. '
        'Optional visible_brand and visible_name may contain exact readable strings; omit other optional keys. '
        'No example answers are supplied: every evidence string must come from this image. Finish the complete JSON within 256 output tokens. '
        'No markdown, preamble, explanation or input-context object. Start with {"schema_version":"visual-evidence-v2","images":[')


def sources_payload(evidence, purpose='fields'):
    sources = [s for s in evidence.sources.values() if s.eligible and
               (s.exact_product or s.kind in ('family_shared', 'visual_printed', 'visual_observation')) and '/fallback' not in s.location]
    sources = [s for s in sources if not (s.kind == 'visual_printed' and
               s.metadata.get('content_role') in ('slogan', 'identity_fragment', 'strength'))]
    blocks = [s for s in sources if s.kind == 'ocr' and '/block/' in s.location]
    if purpose == 'descriptions':
        sources = [s for s in sources if s.kind in ('grounded_text', 'visual_observation', 'visual_printed') or
                   s.kind == 'record' and s.location == 'Product Name']
    unique = {}
    priority = {'grounded_text': 0, 'record': 1, 'exact_catalogue': 2, 'ocr': 3, 'visual_printed': 3, 'visual_observation': 8}
    for source in sorted(sources, key=lambda s: (priority.get(s.kind, 5), '#' in s.location, len(s.text))):
        if source.kind == 'ocr' and '/block/' not in source.location and any(
                b.location.startswith(source.location + '/block/') and canonical(source.text) in canonical(b.text) for b in blocks):
            continue
        meta = {k: v for k, v in source.metadata.items() if k != 'lines'}
        # Coordinates and raw OCR alternatives stay in artifacts, not model context.
        unique.setdefault((source.kind, source.text), {**source.__dict__, 'metadata': meta})
    return list(unique.values())


def add_model_facts(response, evidence):
    if not isinstance(response.get('facts'), list):
        raise ValueError('Field extraction response requires facts array')
    for fact in response['facts']:
        try:
            if not isinstance(fact, dict) or fact['field'] in TEXT or fact['field'] not in SHAPES:
                raise ValueError('Invalid structured field')
            if not isinstance(fact['path'], list) or not all(isinstance(x, str) for x in fact['path']):
                raise ValueError('Invalid fact path')
            source = evidence.sources.get(fact.get('source_id'))
            is_visual = source is not None and source.kind in ('visual_printed', 'visual_observation')
            confidence = 1. if is_visual else float(fact['confidence'])
            if not 0 <= confidence <= 1 or not isinstance(fact['quote'], str):
                raise ValueError('Invalid confidence or quote')
            evidence.add(fact['field'], fact['path'], fact['value'], fact['source_id'],
                         fact['quote'], confidence, method='model_extraction')
        except (KeyError, ValueError, TypeError) as error:
            evidence.issues.append({'kind': 'invalid_model_fact', 'error': str(error)})


def image_identity(item, row, confirmed=None):
    if item.get('match') is not True:
        return False
    try:
        if not .94 <= float(item.get('match_confidence', 0)) <= 1:
            return False
    except (ValueError, TypeError):
        return False
    name = canonical(item.get('visible_name', ''))
    target = canonical(row['Product Name'])
    anchors = set(name.split()) & set(target.split()) - {'ml', 'g', 'l', 'kg', 'pack', 'the', 'and'}
    linked_index = item.get('identity_image_index')
    linked = (type(linked_index) is int and linked_index in (confirmed or {}) and
              linked_index != item.get('image_index') and float(item.get('match_confidence', 0)) >= .98)
    if len(anchors) < 2 and not linked:
        return False
    brand = canonical(item.get('visible_brand', ''))
    target_brand = canonical(row.get('T_BRAND', ''))
    if target_brand and (not brand and not linked or brand and fuzz.ratio(brand, target_brand) < 85):
        return False
    variant = canonical(item.get('visible_variant', ''))
    target_text = canonical(row['Product Name'] + ' ' + plain(row.get('Specifications', '')))
    if variant and f' {variant} ' not in f' {target_text} ':
        return False
    printed = str(item.get('visible_quantity', ''))
    if printed and quantity_relation(printed, row['Product Name']) == 'different':
        return False
    return True


def image_contradiction(item, row):
    brand = canonical(item.get('visible_brand', ''))
    target_brand = canonical(row.get('T_BRAND', ''))
    if (brand and target_brand and fuzz.ratio(brand, target_brand) < 85 and
            f' {brand} ' not in f' {target_brand} ' and f' {target_brand} ' not in f' {brand} '):
        return 'brand'
    printed = str(item.get('visible_quantity', ''))
    if printed and quantity_relation(printed, row['Product Name']) == 'different':
        return 'quantity'
    variant = canonical(item.get('visible_variant', ''))
    variants = [canonical(value) for key, value in spec_pairs(row.get('Specifications', ''))
                if key.casefold() in ('shade', 'colour', 'color', 'flavour', 'flavor', 'fragrance')]
    target_quote = canonical(item.get('target_variant_quote', ''))
    target_text = canonical(row['Product Name'] + ' ' + plain(row.get('Specifications', '')))
    if target_quote and f' {target_quote} ' in f' {target_text} ':
        variants.append(target_quote)
    if variant and variants and not any(f' {variant} ' in f' {v} ' or f' {v} ' in f' {variant} ' for v in variants):
        return 'variant'
    return None


def visual_high_risk(text, row):
    return bool(re.search(r'ingredient|allergen|\bcontains?\b|nutrition|per\s+100|per\s+serv|weight|height|width|depth|warning|dose|dosage|contraindicat|\bsafe\b|suitable|clinically|hypoallergenic|antibacterial|\bSPF\b|pregnan', text, re.I)
                or re.search(r'VITAMIN|HERBAL|MEDIC|BABY HEALTH|BABY.*SAFETY', row.get('category_group', ''), re.I))


def visual_images(response):
    items = response.get('images', []) if isinstance(response, dict) else []
    return [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []


def visual_blocks(item):
    blocks = item.get('transcriptions', [])
    return [x for x in blocks if isinstance(x, dict)] if isinstance(blocks, list) else []


def confirmed_images(response, row, count):
    return {item['image_index']: item for item in visual_images(response)
            if type(item.get('image_index')) is int and 0 <= item['image_index'] < count and image_identity(item, row)}


def numeric_spans(text):
    return [re.sub(r'\s+', '', x).replace(',', '.') for x in
            re.findall(r'[<>≤≥]?\s*\d+(?:[.,]\d+)?\s*%?', str(text))]


def confident(item, threshold):
    try:
        return threshold <= float(item.get('confidence', 0)) <= 1
    except (ValueError, TypeError):
        return False


def agrees_with_ocr(text, source):
    if canonical(text) not in canonical(source.text) or quote_confidence(source, text) < .95:
        return False
    expected, observed = numeric_spans(text), numeric_spans(source.text)
    return not expected or any(observed[i:i + len(expected)] == expected for i in range(len(observed)))


def add_visual(response, paths, evidence, cfg, row, second=None):
    if response.get('schema_version') == VISUAL_VERSION:
        return import_response(response, paths, row, evidence, cfg, image_contradiction)
    items = response.get('images', [])
    if not isinstance(items, list):
        raise ValueError('Image response requires images array')
    second_images = {x.get('image_index'): x for x in visual_images(second) if type(x.get('image_index')) is int}
    confirmed = confirmed_images(response, row, len(paths))
    second_confirmed = confirmed_images(second, row, len(paths))
    identity = {}
    contradictions = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        index = item.get('image_index')
        if type(index) is not int or not 0 <= index < len(paths):
            continue
        contradiction = image_contradiction(item, row)
        if contradiction:
            contradictions[index] = contradiction
            evidence.issues.append({'kind': 'image_product_mismatch', 'image': str(paths[index]), 'attribute': contradiction})
            continue
        matches = image_identity(item, row, confirmed)
        identity[index] = matches
        if not matches:
            evidence.issues.append({'kind': 'image_identity_not_confirmed', 'image': str(paths[index])})
            continue
        for block in visual_blocks(item):
            if not isinstance(block, dict) or not isinstance(block.get('text'), str):
                continue
            text = block['text'].strip()
            try:
                confidence = float(block.get('confidence', 0))
            except (ValueError, TypeError):
                continue
            if not text or not 0 <= confidence <= 1:
                continue
            agreement = any(agrees_with_ocr(text, s)
                            for s in evidence.sources.values() if s.kind == 'ocr' and str(paths[index]) in s.location)
            other = second_images.get(index, {})
            second_read = (image_identity(other, row, second_confirmed) and any(
                canonical(b.get('text', '')) == canonical(text) and
                numeric_spans(b.get('text', '')) == numeric_spans(text) and
                confident(b, cfg.get('visual_high_risk_confidence', .98))
                for b in visual_blocks(other)))
            eligible = confidence >= cfg.get('visual_min_confidence', .94) and (
                agreement or cfg.get('allow_uncorroborated_visual', True))
            meta = {'identity_checked': True, 'identity': {k: v for k, v in item.items() if k != 'transcriptions'},
                    'ocr_agreement': agreement, 'second_read_agreement': bool(second_read),
                    'high_risk': visual_high_risk(text, row), 'image_index': index}
            source = evidence.source('visual', str(paths[index]), text, confidence, eligible=eligible, metadata=meta)
            if eligible:
                extract_nutrition_text(evidence, source)
            else:
                evidence.issues.append({'kind': 'visual_requires_review', 'source_id': source})
    for index, path in enumerate(paths):
        if index in contradictions:
            for source in evidence.sources.values():
                if source.kind in ('ocr', 'visual') and (source.location == str(path) or source.location.startswith(str(path) + '/')):
                    source.eligible = False
                    source.metadata['identity_status'] = 'mismatch'
        elif identity.get(index) is not True:
            for source in evidence.sources.values():
                if source.kind == 'ocr' and (source.location == str(path) or source.location.startswith(str(path) + '/')):
                    source.metadata['identity_status'] = 'uncertain'


def source_descriptions(row, evidence):
    route_evidence(row, evidence, {})
    return compose(row, evidence)

def should_use_vlm(row, evidence, paths, cfg):
    policy = cfg['policy']
    mode = policy['vlm_selection']
    if not policy['vlm'] or mode == 'off':
        return False
    if mode == 'all':
        return True
    if mode != 'adaptive':
        raise ValueError('Unknown VLM selection policy')
    missing_copy = not visible(row.get('Short Description', '')) or not visible(row.get('Product Description', ''))
    recoverable = bool(paths) or any(s.eligible and s.kind in ('ocr', 'family_shared', 'exact_catalogue') for s in evidence.sources.values())
    markers = [('ingredient_information', (), r'ingredients?\s*:'),
               ('storage_instructions', (), r'storage\s*:'),
               ('variant_attributes', ('shade',), r'(?:shade|colou?r)\s*:'),
               ('variant_attributes', ('flavour',), r'flavou?r\s*:'),
               ('variant_attributes', ('fragrance',), r'fragrance\s*:'),
               ('product_specifications', ('application_area',), r'application(?: area)?\s*:'),
               ('product_specifications', ('packaging_type',), r'packaging(?: type)?\s*:'),
               ('product_specifications', ('other',), r'(?:material|usage instructions)\s*:')]
    populated = {(f.field, tuple(f.path)) for f in evidence.facts}
    unavailable = blocked(row)
    structured_need = (policy.get('field_enrichment', True) and policy.get('enrich_sparse_structured', True) and
                       len(evidence.facts) < policy.get('min_supported_facts', 5) and any(
                           (field, path) not in populated and field not in unavailable and
                           any(s.eligible and re.search(pattern, s.text, re.I) for s in evidence.sources.values())
                           for field, path, pattern in markers))
    ocr_chars = sum(len(s.text) for s in evidence.sources.values() if s.kind == 'ocr')
    return missing_copy and recoverable or structured_need or bool(paths and ocr_chars < policy['min_ocr_chars'])


def visual_description_fallback(row, evidence, fallback, cfg):
    route_evidence(row, evidence, cfg)
    return compose(row, evidence)

def finish(row, evidence, paths, candidates, vlm, cfg, use_model):
    calls, stages, rejected = 0, {}, []
    generation_before = getattr(vlm, 'generation_count', 0)
    blocked_fields = blocked(row)
    route_evidence(row, evidence, cfg)

    def ask(stage, payload, images=()):
        nonlocal calls
        try:
            if calls >= cfg['policy']['max_calls_per_record']:
                raise RuntimeError('Per-record stage allowance exhausted')
            calls += 1
            emit = getattr(vlm, 'log', lambda *args, **kwargs: None)
            emit('prompt_assembly_start', stage=stage, compact=cfg['model'].get('compact_prompts', False))
            assembled = compact_prompt(stage, payload) if cfg['model'].get('compact_prompts', False) else prompt(stage, payload)
            save_prompt = getattr(vlm, 'save_prompt', None)
            if save_prompt is not None:
                save_prompt(stage, assembled)
            emit('prompt_assembly_complete', stage=stage, prompt_chars=len(assembled), image_count=len(images))
            response = vlm.ask(stage, assembled, images)
            stages[stage] = 'complete'
            return response
        except BudgetExceeded:
            raise
        except Exception as error:
            stages[stage] = 'failed'
            evidence.issues.append({'kind': 'model_stage_failed', 'stage': stage, 'error': str(error)})
            return None

    if use_model:
        ocr_chars = sum(len(s.text) for s in evidence.sources.values() if s.kind == 'ocr')
        useful_ocr = sum(s.kind == 'grounded_text' and s.metadata.get('claim_scope') == 'label_claim'
                         for s in evidence.sources.values()) >= 2
        need_vision = (cfg['policy']['vlm_selection'] == 'all' or not useful_ocr and
                       (not visible(row.get('Product Description', '')) or ocr_chars < cfg['policy']['min_ocr_chars']))
        if paths and need_vision:
            payload = {'catalogue': row, 'image_count': len(paths)}
            if cfg['policy']['pass_ocr']:
                payload['ocr'] = [{**s, 'image_index': index} for index, path in enumerate(paths)
                    for s in sources_payload(evidence) if s['kind'] == 'ocr' and
                    (s['location'] == str(path) or s['location'].startswith(str(path) + '/'))]
            response = ask('image_extraction', payload, paths)
            if response:
                second = None
                confirmed = confirmed_images(response, row, len(paths))
                risk_blocks = [b for item in visual_images(response) if image_identity(item, row, confirmed)
                               for b in visual_blocks(item) if visual_high_risk(str(b.get('text', '')), row)
                               and not any(agrees_with_ocr(str(b.get('text', '')), s)
                                           for s in evidence.sources.values() if s.kind == 'ocr')]
                if (response.get('schema_version') != VISUAL_VERSION and risk_blocks and cfg['policy'].get('image_verification', True) and
                        cfg['verification'].get('visual_require_second_read_high_risk', True)):
                    # No first-read text or OCR is passed back for the independent read.
                    second = ask('image_verification', {'catalogue': row, 'image_count': len(paths),
                                 'task': 'Read label panels afresh, especially ingredients, warnings and numeric tables.'}, paths)
                try:
                    add_visual(response, paths, evidence, cfg['verification'], row, second)
                except (ValueError, TypeError, KeyError) as error:
                    stages['image_extraction'] = 'failed'
                    evidence.issues.append({'kind': 'invalid_image_response', 'error': str(error)})
        route_evidence(row, evidence, cfg)
        if cfg['policy']['field_enrichment']:
            context = [s.__dict__ for s in evidence.sources.values() if s.kind == 'family_context'][:3]
            response = ask('field_enrichment', {'sources': sources_payload(evidence), 'category': row.get('category_group', ''),
                           'related_context_not_evidence': context, 'blocked': sorted(blocked_fields),
                           'shapes': {k: v for k, v in SHAPES.items() if k not in TEXT}})
            if response:
                try:
                    add_model_facts(response, evidence)
                except (ValueError, TypeError, KeyError) as error:
                    stages['field_enrichment'] = 'failed'
                    evidence.issues.append({'kind': 'invalid_field_response', 'error': str(error)})
    values, accepted, rejected = verify(evidence, cfg['_schema'], cfg['verification'], blocked_fields)
    fallback = compose(row, evidence)
    response = None
    need_description = (cfg['policy']['vlm_selection'] == 'all' or
                        not visible(row.get('Short Description', '')) or
                        not visible(row.get('Product Description', '')))
    if use_model and cfg['policy']['descriptions'] and need_description:
        response = ask('description_generation', {'sources': sources_payload(evidence, 'descriptions'),
                       'category': row.get('category_group', ''), 'structured_fields': values,
                       'blocked': sorted(blocked_fields), 'full_product_name': row['Product Name']})
    prose_trace = {}
    for field in ('short_description', 'long_description'):
        if field in blocked_fields:
            continue
        kept = []
        sentences = (response or {}).get(field, [])
        if not isinstance(sentences, list):
            stages['description_generation'] = 'failed'
            evidence.issues.append({'kind': 'invalid_description_response', 'field': field})
            sentences = []
        for sentence in sentences:
            if not isinstance(sentence, dict) or not isinstance(sentence.get('text'), str):
                continue
            reason = verify_sentence(sentence['text'], sentence.get('source_ids', []), evidence, cfg['verification'])
            if not reason and re.search(r'VITAMIN|HERBAL|MEDICINE|BABY HEALTH', row.get('category_group', ''), re.I):
                if re.search(r'\b(dose|dosage|daily|take|contraindication|treat|relieve|may help|warning)\b', sentence['text'], re.I):
                    if not any(normalise(sentence['text']) in normalise(evidence.sources[s].text)
                               for s in sentence.get('source_ids', [])):
                        reason = 'regulated_wording_changed'
            if reason:
                rejected.append({'reason': reason, 'field': field, 'sentence': sentence})
            elif canonical(sentence['text']) not in {canonical(x['text']) for x in kept}:
                kept.append({**sentence, 'mode': 'generated'})
        # Select by evidence use rather than rewarding longer generated copy.
        if not kept:
            kept = fallback[field]
        if field == 'long_description' and kept and not kept[0]['text'].casefold().startswith(plain(row['Product Name']).casefold()):
            kept = fallback[field]
        prose_trace[field] = kept
        values[field] = ' '.join(x['text'] for x in kept)
    if (values['short_description'] and values['long_description'].startswith(values['short_description'])
            and canonical(values['short_description']) != canonical(row['Product Name'])):
        source = next((s for s in evidence.sources.values() if s.kind == 'record' and s.location == 'Product Name'), None)
        if source:
            prefix = {'text': source.text + '.', 'source_ids': [source.id], 'mode': 'source'}
            prose_trace['long_description'] = [prefix] + prose_trace.get('long_description', [])
            values['long_description'] = ' '.join(x['text'] for x in prose_trace['long_description'])
    # Claims marked for source review cannot be restored by generation or source reuse.
    for field in ('short_description', 'long_description'):
        filtered = []
        for sentence in prose_trace.get(field, []):
            reason = verify_sentence(sentence['text'], sentence['source_ids'], evidence, cfg['verification'])
            if reason:
                rejected.append({'reason': reason, 'field': field, 'sentence': sentence})
            else:
                filtered.append(sentence)
        prose_trace[field] = filtered
        values[field] = ' '.join(x['text'] for x in filtered)
    errors = cfg['_schema'].errors(values, blocked_fields)
    if errors:
        raise ValueError('; '.join(errors))
    succeeded = any(status == 'complete' for status in stages.values())
    failed = any(status == 'failed' for status in stages.values())
    return {'values': values, 'evidence': evidence.payload(), 'accepted_facts': [f.__dict__ for f in accepted],
            'rejected': rejected, 'description_trace': prose_trace, 'retrieval': candidates,
            'model_stages': calls, 'stage_status': stages, 'degraded': failed,
            'model_attempted': calls > 0, 'model_loaded': bool(use_model and getattr(vlm, 'loaded', False)),
            'model_succeeded': succeeded, 'model_failed': failed, 'used_model': succeeded,
            'model_generation_calls': getattr(vlm, 'generation_count', 0) - generation_before}
