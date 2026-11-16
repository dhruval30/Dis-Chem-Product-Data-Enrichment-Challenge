from __future__ import annotations

import copy
import json
import re
from decimal import Decimal

from jsonschema import Draft202012Validator

from .evidence import normalise
from .schema import SHAPES, TEXT, leaves

ALIASES = {'millilitre': 'ml', 'millilitres': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
           'grams': 'g', 'gram': 'g', 'kilogram': 'kg', 'kilograms': 'kg',
           'litre': 'l', 'litres': 'l', 'liter': 'l', 'liters': 'l', 'pcs': 'pieces'}
CRITICAL = {'ingredient_information', 'nutritional_information', 'weight', 'dimensions'}
STOP = {'a', 'an', 'the', 'and', 'of', 'for', 'with', 'in', 'is', 'are', 'to', 'it', 'this', 'its', 'as', 'on', 'from'}
RISK_WORDS = {'safe', 'safety', 'proven', 'clinically', 'cures', 'cure', 'prevents', 'prevention',
              'treats', 'treatment', 'guaranteed', 'hypoallergenic', 'antibacterial', 'pregnancy',
              'children', 'babies', 'infants', 'free', 'reduces', 'effective', 'suitable'}


def canonical(text):
    # Split attached measurement units without splitting decimal values.
    text = re.sub(r'(\d)\s*(ml|kg|g)net\b', r'\1 \2 net', str(text), flags=re.I)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)', ' ', str(text))
    words = re.findall(r'\d+(?:[.,]\d+)?|[\w]+', text.casefold())
    result = []
    for word in words:
        if re.fullmatch(r'\d+(?:[.,]\d+)?', word):
            word = format(Decimal(word.replace(',', '.')).normalize(), 'f')
        result.append(ALIASES.get(word, word))
    return ' '.join(result)


def allowed_path(field, path):
    if field in TEXT or field in ('ingredient_information', 'storage_instructions', 'weight'):
        return path == []
    if field == 'variant_attributes':
        return len(path) == 1 and path[0] in ('shade', 'flavour', 'fragrance', 'other')
    if field == 'product_specifications':
        return len(path) == 1 and path[0] in ('pack_size', 'application_area', 'packaging_type', 'packaging_material', 'other')
    if field == 'dimensions':
        return len(path) == 1 and path[0] in ('height', 'width', 'depth', 'unit')
    if field == 'nutritional_information':
        return path == ['serving_size'] or len(path) == 2 and path[0] in ('per_100g', 'per_serving')
    return False


def comparable(value):
    if isinstance(value, dict):
        result = {}
        for key, scalar in value.items():
            if key == 'unit' and isinstance(scalar, str):
                words = canonical(scalar).split()
                result[key] = ' '.join(dict.fromkeys(words))
            else:
                result[key] = comparable(scalar)
        return result
    if isinstance(value, list):
        return [comparable(x) for x in value]
    return canonical(value)


def deduplicate(value):
    if isinstance(value, dict):
        return {k: deduplicate(v) for k, v in value.items()}
    if isinstance(value, list):
        result, seen = [], set()
        for item in value:
            item = deduplicate(item)
            key = json.dumps(comparable(item), sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
    return value


def supported(value, quote):
    """Conservative lexical support; this is not a semantic entailment oracle."""
    q = canonical(quote)
    for _, scalar in leaves(value):
        if isinstance(scalar, (int, float)):
            numbers = [float(x.replace(',', '.')) for x in re.findall(r'(?<![\w.])\d+(?:[.,]\d+)?', quote)]
            if float(scalar) not in numbers:
                return False
        else:
            v = canonical(scalar)
            if f' {v} ' not in f' {q} ':
                # Values such as 'millilitre (ml)' contain redundant unit notation.
                tokens = v.split()
                if not tokens or not all(x in q.split() for x in tokens):
                    return False
    return True


def fact_reason(fact, evidence, cfg, blocked):
    if fact.field not in SHAPES or fact.field in blocked:
        return 'blocked_or_unknown_field'
    if not allowed_path(fact.field, fact.path):
        return 'invalid_field_path'
    source = evidence.sources.get(fact.source_id)
    if source is None or not source.eligible:
        return 'ineligible_source'
    if fact.field == 'variant_attributes':
        from .text_extraction import valid_variant
        category = next((s.text for s in evidence.sources.values() if s.kind == 'record' and
                         s.location.split('#')[0] == 'category_group'), '')
        if not isinstance(fact.value, list) or any(not valid_variant(fact.path[0], value,
                {'category_group': category}) for value in fact.value):
            return 'invalid_variant_value'
    if source.kind == 'grounded_text':
        from .content import grounded_reason
        reason = grounded_reason(source, evidence)
        if reason:
            return reason
        if source.metadata.get('mapped_storage_only') and fact.field != 'storage_instructions':
            return 'mapped_panel_storage_only'
        usage_spec = (source.metadata.get('role') == 'usage' and fact.field == 'product_specifications' and
                      fact.path == ['other'] and fact.value == [{'attribute': 'usage_instructions', 'value': source.text}])
        if fact.field not in ('variant_attributes', 'storage_instructions') and not usage_spec:
            return 'descriptive_text_is_not_a_structured_measurement'
    elif source.kind in ('visual_printed', 'visual_observation'):
        reason = visual_evidence_reason(source, evidence, fact)
        if reason:
            return reason
    elif not source.exact_product:
        meta = source.metadata
        if (source.kind != 'family_shared' or fact.field != meta.get('allowed_field') or
            fact.path != meta.get('allowed_path') or fact.value != meta.get('allowed_value') or
            len(meta.get('consensus_ids', [])) < 2 or not (
                meta.get('target_anchor_ids') or meta.get('size_only_application_area') and fact.path == ['application_area'])):
            return 'ineligible_family_transfer'
    if not fact.quote or normalise(fact.quote) not in normalise(source.text):
        return 'quote_not_in_source'
    threshold = cfg['field_thresholds'].get(fact.field, cfg['minimum_confidence'])
    if source.kind not in ('visual_printed', 'visual_observation'):
        confidence = quote_confidence(source, fact.quote, fact)
        if fact.confidence is None or confidence is None or min(fact.confidence, confidence) < threshold:
            return 'low_confidence'
    support_value = fact.value
    if source.kind in ('visual_printed', 'visual_observation', 'grounded_text') and fact.field == 'product_specifications' and fact.path == ['other']:
        if isinstance(fact.value, list) and all(isinstance(item, dict) and 'value' in item for item in fact.value):
            support_value = [item['value'] for item in fact.value]
    if not supported(support_value, fact.quote):
        return 'value_not_in_quote'
    if fact.field == 'product_specifications' and fact.path == ['pack_size']:
        from .text_extraction import pack_quantity
        parsed = pack_quantity(fact.quote, source.text)
        complete = pack_quantity(source.text)
        if complete and '+' in complete[0]['value']:
            parsed = complete
        if not parsed or comparable(parsed[0]) != comparable(fact.value):
            return 'strength_or_incomplete_pack_quantity'
    if source.kind == 'visual':
        reason = visual_reason(source, cfg, fact.field in CRITICAL)
        if reason:
            return reason
    # Require explicit labels for high-risk placements rather than merely matching a number.
    context = (source.location + ' ' + fact.quote).casefold()
    if fact.field == 'weight' and not re.search(r'\b(?:net\s+|gross\s+)?weight\b', context):
        return 'unlabelled_weight'
    if fact.field == 'weight' and isinstance(fact.value, dict):
        pair = re.escape(str(fact.value.get('value', '')))
        if not re.search(r'weight\s*[:=]?\s*' + pair + r'(?![\d.])', fact.quote, re.I):
            return 'weight_value_label_mismatch'
    if fact.field == 'dimensions':
        if not fact.path or fact.path[0] not in ('height', 'width', 'depth', 'unit'):
            return 'unlabelled_dimension_axis'
        if fact.path[0] != 'unit' and fact.path[0] not in context:
            return 'unlabelled_dimension_axis'
        if fact.path[0] != 'unit' and not re.search(
                re.escape(fact.path[0]) + r'\s*[:=]?\s*' + re.escape(str(fact.value)) + r'(?![\d.])', fact.quote, re.I):
            return 'dimension_value_label_mismatch'
    if fact.field == 'nutritional_information' and fact.path:
        if fact.path[0] == 'per_100g' and not re.search(r'(?:per_100g|100\s*g\b)', context):
            return 'missing_nutrition_basis'
        if fact.path[0] == 'per_serving' and not re.search(r'(?:per_serving|per\s+serv)', context):
            return 'missing_nutrition_basis'
        if fact.method == 'model_extraction' and fact.path[0] in ('per_100g', 'per_serving'):
            # A matching number elsewhere in a nutrition panel is not evidence for this cell.
            try:
                from .text_extraction import NUTRIENTS
                data = json.loads(source.text)
                matches = [values.get(fact.path[0]) for nutrient, values in data.get('nutrients', {}).items()
                           if NUTRIENTS.get(nutrient.casefold(), nutrient) == fact.path[1]]
                if not any(re.fullmatch(r'\d+(?:[.,]\d+)?', str(x)) and
                           float(str(x).replace(',', '.')) == float(fact.value) for x in matches):
                    return 'nutrition_cell_mismatch'
            except (ValueError, TypeError, IndexError, AttributeError):
                cells = source.metadata.get('nutrition_cells', [])
                if not any(c['basis'] == fact.path[0] and c['nutrient'] == fact.path[1] and
                           c['value'] == fact.value and c['confidence'] >= threshold for c in cells):
                    return 'nutrition_cell_not_verified'
    if fact.field == 'ingredient_information' and not re.search(r'ingredient|contains?|may contain', context):
        return 'unlabelled_ingredient'
    if fact.field == 'ingredient_information' and isinstance(fact.value, list):
        declaration = ' '.join(str(x) for x in fact.value)
        for qualifier in ('may contain', 'does not contain', 'contains no', 'free from', 'no added'):
            if qualifier in fact.quote.casefold() and qualifier not in declaration.casefold():
                return 'ingredient_qualifier_lost'
        position = 0
        for item in fact.value:
            found = canonical(fact.quote).find(canonical(item), position)
            if found < 0:
                return 'ingredient_order_changed'
            position = found + len(canonical(item))
    if '[MASKED]' in str(fact.value).upper():
        return 'masked_value'
    return None


def assign(root, path, value):
    if not path:
        return copy.deepcopy(value)
    node = root
    for key in path[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise ValueError('Invalid field path')
        node = node[key]
    if not isinstance(node, dict):
        raise ValueError('Only named object paths are accepted')
    node[path[-1]] = copy.deepcopy(value)
    return root


def verify(evidence, schema, cfg, blocked):
    output = schema.blank()
    rejected, accepted, groups = [], [], {}
    for fact in evidence.facts:
        reason = fact_reason(fact, evidence, cfg, blocked)
        if not reason:
            try:
                trial = assign(copy.deepcopy(output[fact.field]), fact.path, fact.value)
                Draft202012Validator(SHAPES[fact.field]).validate(trial)
            except Exception:
                reason = 'invalid_fact_shape'
        if reason:
            rejected.append({'reason': reason, 'fact': fact.__dict__})
        else:
            groups.setdefault((fact.field, tuple(fact.path)), []).append(fact)
    conflicted_fields = set()
    for (field, path), facts in groups.items():
        if field in TEXT:
            # The target name has priority over an equivalent catalogue name.
            facts.sort(key=lambda f: evidence.sources[f.source_id].kind != 'record')
        first = facts[0]
        if isinstance(first.value, list):
            merged = []
            # Ingredient declarations are ordered alternatives, never union different lists.
            if field == 'ingredient_information':
                unique = {canonical(' '.join(str(x) for x in f.value)) for f in facts}
                if len(unique) > 1:
                    rejected.extend({'reason': 'conflicting_ingredient_declarations', 'fact': f.__dict__} for f in facts)
                    continue
                output[field] = copy.deepcopy(first.value)
                accepted.extend(facts)
                continue
            for fact in facts:
                for item in fact.value:
                    if item not in merged:
                        merged.append(item)
            output[field] = assign(output[field], path, merged)
            accepted.extend(facts)
        else:
            unique = {json.dumps(comparable(f.value), sort_keys=True) for f in facts}
            if len(unique) > 1:
                rejected.extend({'reason': 'conflicting_values', 'fact': f.__dict__} for f in facts)
                if field == 'dimensions':
                    conflicted_fields.add(field)
                continue
            output[field] = assign(output[field], path, first.value)
            accepted.extend(facts)
    if 'dimensions' in conflicted_fields or output['dimensions']['unit'] == '' and any(output['dimensions'][k] for k in ('height', 'width', 'depth')):
        output['dimensions'] = copy.deepcopy(schema.empty['dimensions'])
        accepted = [f for f in accepted if f.field != 'dimensions']
        rejected.append({'reason': 'dimension_unit_missing_or_conflicting'})
    for field in output:
        if field not in TEXT and field != 'ingredient_information':
            output[field] = deduplicate(output[field])
    specifications = output['product_specifications']
    pair = specifications['pack_size']
    if pair['value'] and pair['unit']:
        from .text_extraction import quantity
        other = []
        for item in specifications['other']:
            parsed = quantity(item['value'])
            if (canonical(item['attribute']) in {'pack size', 'pack_size'} and parsed and
                    comparable(parsed[0]) == comparable(pair)):
                continue
            other.append(item)
        specifications['other'] = other
    # Recheck combined objects, not just isolated facts.
    errors = schema.errors(output, blocked)
    if errors:
        raise ValueError('Combined output failed validation: ' + '; '.join(errors))
    return output, accepted, rejected


def verify_sentence(sentence, source_ids, evidence, cfg):
    if not isinstance(source_ids, list) or any(not isinstance(s, str) for s in source_ids):
        return 'invalid_sentence_sources'
    sources = [evidence.sources.get(s) for s in source_ids]
    if not sources or any(s is None or not s.eligible or not (
            s.exact_product or s.kind in ('visual_printed', 'visual_observation')) for s in sources):
        return 'invalid_sentence_sources'
    for source in sources:
        if source.kind in ('visual_printed', 'visual_observation'):
            reason = visual_evidence_reason(source, evidence)
            if reason:
                return reason
    from .content import rendered_sentence_reason
    rendering = rendered_sentence_reason(sentence, sources, evidence)
    if rendering != 'not_closed_rendering':
        return rendering
    if len(sources) == 1:
        source = sources[0]
        if source.kind == 'visual_observation':
            prefix = 'The photograph shows ' if source.metadata.get('attribute') == 'visible_count' else 'Visible details include '
            if normalise(sentence) == normalise(prefix + source.text + '.'):
                return None
        if source.kind == 'visual_printed' and source.metadata.get('printed_kind') in ('claim', 'highlighted_ingredient'):
            if normalise(sentence) == normalise('The label states: ' + source.text + '.'):
                return None
    if len(sources) == 1 and sources[0].kind == 'record' and sources[0].location in {
            'T_CATEGORY/classification', 'T_DEPARTMENT/classification', 'category_group/classification'}:
        if normalise(sentence) == normalise('Catalogue category: ' + sources[0].text + '.'):
            return 'catalogue_metadata_filler'
    for source in sources:
        if source.kind == 'visual_observation' and source.metadata.get('attribute') == 'visible_count':
            return 'visible_count_requires_photo_attribution'
        if source.kind == 'visual_printed' and source.metadata.get('printed_kind') in ('claim', 'highlighted_ingredient'):
            if not re.search(r'\b(?:label|packaging)\b', sentence, re.I):
                return 'unattributed_label_claim'
        if source.kind == 'record' and ('Description' in source.location or '/fallback' in source.location):
            if re.search(r'\b(?:may|might|could|designed to|safe|effective|soothe|soothes|helps?|supports?|reduces?)\b', sentence, re.I):
                return 'unreviewed_source_claim'
        if source.kind == 'visual':
            reason = visual_reason(source, cfg, source.metadata.get('high_risk', False))
            if reason:
                return reason
    text = ' '.join(s.text for s in sources)
    if '[MASKED]' in sentence.upper():
        return 'masked_sentence'
    if re.search(r'\b(?:best ever|miracle|guaranteed|sure to|perfect for|unparalleled|will|clinically proven|completely safe)\b', sentence, re.I):
        return 'speculative_sentence'
    tokens = set(canonical(sentence).split()) - STOP
    source_tokens = set(canonical(text).split())
    if not tokens or len(tokens & source_tokens) / len(tokens) < cfg['description_token_overlap']:
        return 'low_sentence_support'
    if any(s.kind in ('grounded_text', 'visual_observation', 'visual_printed') for s in sources):
        framing = {'product', 'label', 'labelled', 'packaging', 'states', 'contains', 'containing',
                   'features', 'comes', 'has', 'design', 'details', 'include', 'photograph', 'shows'}
        if tokens - source_tokens - framing:
            return 'unsupported_description_content'
        if re.search(r'\bno\b', text, re.I) and not re.search(r'\b(?:no|not|without)\b', sentence, re.I):
            return 'lost_source_negation'
    for token in tokens & RISK_WORDS:
        if token not in source_tokens:
            return 'unsupported_claim_word'
    if not supported(re.findall(r'\d+(?:[.,]\d+)?', sentence), text):
        return 'unsupported_sentence_number'
    from .text_extraction import QUANTITY
    for match in QUANTITY.finditer(sentence):
        if f' {canonical(match.group())} ' not in f' {canonical(text)} ':
            return 'sentence_quantity_unit_mismatch'
    # Do not strengthen hedged claims, including claims supported elsewhere in the same source.
    if re.search(r'\b(may|might|can help)\b', text, re.I) and not re.search(r'\b(may|might|can help)\b', sentence, re.I):
        if tokens & {'help', 'helps', 'reduce', 'reduces', 'support', 'supports', 'relieve', 'relieves'}:
            return 'lost_claim_qualifier'
    return None


def visual_evidence_reason(source, evidence, fact=None):
    from .visual import FORBIDDEN, UNSEEN_PROPERTIES, OBSERVATIONS, VERSION, corroborated, PLACEHOLDER
    meta = source.metadata
    if PLACEHOLDER.search(source.text + ' ' + meta.get('visible_support', '')):
        return 'visual_prompt_placeholder'
    if (meta.get('visual_schema') != VERSION or not meta.get('mapped_image') or
            meta.get('identity_status') not in ('confirmed', 'unverifiable') or not source.eligible):
        return 'invalid_visual_provenance'
    if source.kind == 'visual_observation':
        attribute = meta.get('attribute')
        if (attribute not in OBSERVATIONS or not meta.get('visible_support') or
            FORBIDDEN.search(source.text + ' ' + meta['visible_support']) or
            UNSEEN_PROPERTIES.search(source.text + ' ' + meta['visible_support'])):
            return 'unsupported_visual_observation'
        if fact:
            return 'observation_cannot_establish_commercial_attribute'
    elif source.kind == 'visual_printed':
        ocr_id = meta.get('ocr_source_id')
        if ocr_id:
            ocr = evidence.sources.get(ocr_id)
            minimum = max(.9, float(meta.get('ocr_minimum_confidence', .95)))
            if ocr is None or not ocr.eligible or not corroborated(source.text, [ocr], minimum):
                return 'invalid_ocr_corroboration'
        elif meta.get('identity_status') != 'confirmed' or meta.get('high_risk'):
            return 'printed_evidence_requires_review'
        kind = meta.get('printed_kind')
        role = meta.get('content_role', kind)
        if role in ('slogan', 'identity_fragment', 'strength'):
            return 'printed_context_not_product_content'
        if fact and kind in ('claim', 'highlighted_ingredient') and role != 'quantity':
            return 'printed_claim_is_descriptive_only'
        if fact and fact.field == 'product_specifications' and fact.path == ['pack_size']:
            if role != 'quantity' or meta.get('quantity_scope', 'whole') != 'whole':
                return 'quantity_is_not_whole_pack'
        if fact and fact.field == 'ingredient_information' and kind != 'ingredients_declaration':
            return 'not_a_complete_ingredient_declaration'
        if fact:
            allowed_fields = {'identity': set(), 'quantity': {'product_specifications'},
                'ingredients_declaration': {'ingredient_information'}, 'storage': {'storage_instructions'},
                'usage': {'product_specifications'}, 'warning': {'product_specifications'},
                'nutrition': {'nutritional_information'}, 'specification': {'product_specifications', 'variant_attributes', 'weight', 'dimensions'}}
            allowed_kind = 'quantity' if role == 'quantity' else kind
            if allowed_kind in allowed_fields and fact.field not in allowed_fields[allowed_kind]:
                return 'printed_evidence_wrong_field'
            if kind == 'quantity' and fact.path != ['pack_size']:
                return 'quantity_is_not_an_unrelated_specification'
            if fact.field == 'product_specifications' and fact.path == ['packaging_material']:
                if 'packaging material' not in canonical(source.text):
                    return 'product_material_is_not_packaging_material'
            if fact.field == 'variant_attributes':
                labels = {'shade': r'\b(?:shade|colou?r)\b', 'flavour': r'\bflavou?r\b',
                          'fragrance': r'\bfragrance\b', 'other': r'\bvariant\b'}
                if not fact.path or not re.search(labels.get(fact.path[0], r'(?!)'), source.text, re.I):
                    return 'unlabelled_printed_variant'
            if kind == 'specification' and fact.field == 'product_specifications' and fact.path == ['other']:
                if not isinstance(fact.value, list) or any(
                    not isinstance(item, dict) or canonical(str(item.get('attribute', '')).replace('_', ' ')) not in canonical(source.text)
                    for item in fact.value):
                    return 'specification_label_not_in_printed_text'
    return None


def visual_reason(source, cfg, high_risk=False):
    meta = source.metadata
    if not meta.get('identity_checked'):
        return 'visual_identity_unverified'
    if source.confidence < cfg.get('visual_min_confidence', .94):
        return 'low_visual_confidence'
    if not meta.get('ocr_agreement') and not cfg.get('allow_uncorroborated_visual', False):
        return 'uncorroborated_visual'
    if high_risk or meta.get('high_risk'):
        if source.confidence < cfg.get('visual_high_risk_confidence', .98):
            return 'low_high_risk_visual_confidence'
        if cfg.get('visual_require_second_read_high_risk', True) and not (
                meta.get('ocr_agreement') or meta.get('second_read_agreement')):
            return 'high_risk_visual_not_confirmed'
    return None


def quote_confidence(source, quote, fact=None):
    if source.kind != 'ocr' or not source.metadata.get('lines'):
        return source.confidence
    if fact and fact.field == 'nutritional_information' and len(fact.path) == 2:
        cells = [x for x in source.metadata.get('nutrition_cells', []) if
                 x['basis'] == fact.path[0] and x['nutrient'] == fact.path[1] and x['value'] == fact.value]
        if cells:
            return min(x['confidence'] for x in cells)
    if fact:
        values = [canonical(value) for path, value in leaves(fact.value)
                  if not path or path[-1] != 'attribute']
        matching = [line['confidence'] for line in source.metadata['lines']
                    if canonical(line['text']) in canonical(quote) and values and
                    all(f' {value} ' in f" {canonical(line['text'])} " for value in values)]
        if matching:
            return max(matching)
    used = [line['confidence'] for line in source.metadata['lines']
            if canonical(line['text']) in canonical(quote) or canonical(quote) in canonical(line['text'])]
    return min(used) if used else 0.
