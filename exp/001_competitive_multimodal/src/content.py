from __future__ import annotations

import re
from rapidfuzz.distance import Levenshtein

from .data import plain
from .evidence import normalise
from .verification import canonical, quote_confidence

ARTWORK = re.compile(r'\b(?:logos?|illustrations?|cartoons?|artwork|graphics?|depicted|pictured|printed image|grain image|image on|picture of|holding baby|on (?:the )?box|box front|package front)\b', re.I)
CLAIM = re.compile(r'\b(?:helps?|may help|soothe|protect|hypoallergenic|natural|preservatives?|colou?rants?|flavou?rings?|free.from|no need|before breastfeeding)\b', re.I)
HYPE = re.compile(r'\b(?:miracle|best ever|guaranteed|sure to|unparalleled|cures?|clinically proven|completely safe|'
                  r'radiate power|unforgettable presence|bustling souks|transport you|captivating|opulent|'
                  r'exotic warmth|(?:warm|rich) and inviting atmosphere|brightness of the sun|sumptuousness)\b', re.I)
VERSION = 'product-content-v2'


def clean_text(text):
    for pattern, value in [(r'\bSTORE INA\b', 'STORE IN A'), (r'\bDRYPLACEBEFORE\b', 'DRY PLACE BEFORE'),
                           (r'\bANDAFTEROPENING\b', 'AND AFTER OPENING')]:
        text = re.sub(pattern, value, text, flags=re.I)
    text = re.sub(r'(?<=[.!?])(?=[A-Za-z])', ' ', text)
    text = re.sub(r'(?m)^\s*[✓✔√Vv]\s*(?=NO\b)', '', text)
    text = re.sub(r'&\s*', ' and ', text)
    text = re.sub(r'(?<=\w),(?=\w)', ', ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=%)(?=[A-Za-z])', ' ', text)
    return ' '.join(text.split()).strip(' •*')


def image_identity(row, sources):
    spans = []
    for source in sources:
        if not source.eligible:
            continue
        spans.extend(source.metadata.get('lines', [{'text': source.text, 'confidence': source.confidence}]))
    text = canonical(' '.join(s['text'] for s in spans if (s['confidence'] or 0) >= .9))
    brand_text = canonical(' '.join(s['text'] for s in spans if (s['confidence'] or 0) >= .85))
    brand = canonical(row.get('T_BRAND', ''))
    name = set(canonical(row['Product Name']).split())
    ignored = set(brand.split()) | {'the', 'and', 'for', 'from', 'with', 'ml', 'g', 'months', 'size', 'professional'}
    anchors = {w for w in name - ignored if len(w) >= 3 and not w.isdigit() and w in text.split()}
    # OCR can join product-name words. This only identifies the image, never repairs a value.
    words = canonical(row['Product Name']).split()
    for a, b in zip(words, words[1:]):
        if a not in ignored and b not in ignored and len(a + b) >= 5 and a + b in text.split():
            anchors.add(a + b)
    tokens = brand_text.split()
    target = ''.join(brand.split())
    brand_matches = bool(brand and f' {brand} ' in f' {brand_text} ')
    if not brand_matches and len(target) >= 5:
        tolerance = 2 if len(target) >= 12 else 1
        brand_matches = any(Levenshtein.distance(target, ''.join(tokens[i:i + length])) <= tolerance
                            for i in range(len(tokens)) for length in range(1, len(brand.split()) + 2))
    if brand_matches and len(anchors) >= 2:
        return True
    from .text_extraction import literal_pack_quantity, quantity_relation
    quantity_match = any((s.confidence or 0) >= .95 and literal_pack_quantity(s.text, row['Product Name']) and
                         quantity_relation(s.text, row['Product Name']) == 'same' for s in sources if s.eligible)
    return len(anchors) >= 4 and quantity_match


def ocr_quantity_conflict(row, sources):
    from .text_extraction import literal_pack_quantity, quantity_relation
    name = row.get('Product Name', '')
    for source in sources:
        if not source.eligible:
            continue
        if source.metadata.get('panel_role') in ('nutrition_basis', 'serving_quantity'):
            continue
        for line in source.metadata.get('lines', [{'text': source.text, 'confidence': source.confidence}]):
            if (line['confidence'] or 0) < .95:
                continue
            candidate = literal_pack_quantity(line['text'], name)
            if not candidate:
                continue
            if quantity_relation(line['text'], name) == 'different':
                return True
    return False


def appearance_reason(source, row):
    if source.kind != 'visual_observation':
        return None
    support = source.text + ' ' + source.metadata.get('visible_support', '')
    from .visual import PLACEHOLDER
    if PLACEHOLDER.search(support):
        return 'visual_prompt_placeholder'
    if re.search(r'\b(?:packaging|pouch|container|box|pump top|hang hole)\b', support, re.I):
        return 'incidental_packaging_artwork'
    if ARTWORK.search(support):
        is_garment = re.search(r'CLOTHING|TEXTILE|BEDDING', row.get('category_group', '') + ' ' + row.get('T_CATEGORY', ''), re.I)
        if not is_garment or re.search(r'\b(?:box|packaging|package|label|logo)\b', support, re.I):
            return 'incidental_packaging_artwork'
    if source.metadata.get('attribute') in ('visible_count', 'packaging_form'):
        return 'non_descriptive_photo_detail'
    # Packaging decoration is not the product's own design. Textless garments/accessories can use appearance.
    category = row.get('category_group', '') + ' ' + row.get('T_CATEGORY', '')
    if re.search(r'FOOD|NUTRITION|VITAMIN|HERBAL|MEDIC|PERFUM|COSMETIC|BEAUTY|HEALTH|HAIR SALON|HOME SCENTS|MATERNITY', category, re.I):
        return 'appearance_low_relevance_for_packaged_product'
    return None


def grounded(evidence, parent, quote, role, text=None, confidence=None):
    text = clean_text(quote) if text is None else text
    value = parent.confidence if confidence is None else confidence
    return evidence.source('grounded_text', parent.location, text, value, exact=parent.exact_product,
        eligible=parent.eligible, metadata={'content_version': VERSION, 'role': role,
            'parent_source_id': parent.id, 'quote': quote, 'rendered_text': text,
            'image_path': parent.metadata.get('image_path', parent.location.split('/block/', 1)[0])
                          if parent.kind in ('ocr', 'visual_printed', 'visual_observation', 'visual') else None,
            'identity_status': parent.metadata.get('identity_status', 'supplied_record'),
            'claim_scope': 'label_claim' if parent.kind in ('ocr', 'visual_printed') and role in ('claim', 'benefit', 'usage', 'highlighted_ingredient') else 'supplied_text'})


def specification_sentence(label, value):
    if not value or '[MASKED]' in value.upper():
        return None
    if label.casefold() == 'skin type':
        value = value.lower().removesuffix(' skin')
        return 'For ' + value.replace(', ', ' and ') + ' skin.'
    if label.casefold() in ('make-up finish', 'makeup finish', 'finish'):
        return value[:1].upper() + value[1:].lower() + ' finish.'
    if label.casefold() == 'waterproof' and value.casefold() == 'yes':
        return 'Listed as waterproof.'
    if label.casefold() == 'spf' and re.fullmatch(r'\d+\+?', value):
        return 'SPF ' + value + '.'
    return None


def fragrance_notes(quote):
    matches = list(re.finditer(r'\b(?:blend of|scents of|notes of|base combines)\s+(.+?)(?=,?\s+while\b|[.!?]|$)', quote, re.I))
    if not matches:
        return None
    values = [match[1].strip(' ,') for match in matches]
    if any(len(value.split()) > 16 or HYPE.search(value) for value in values):
        return None
    return 'Fragrance notes include ' + '; '.join(values) + '.'


def rendering_supported(source, parent):
    rendering = source.metadata.get('rendering')
    quote = source.metadata.get('quote', '')
    if rendering == 'specification':
        label, separator, value = quote.partition(':')
        return bool(separator and quote == parent.text and source.text == specification_sentence(label.strip(), value.strip()))
    if rendering == 'fragrance_notes':
        return source.text == fragrance_notes(quote)
    return False


def route_ocr_facts(row, evidence, parent, line):
    from .text_extraction import pack_quantity
    if (line['confidence'] or 0) < .95:
        return
    raw, text = line['text'], clean_text(line['text'])
    pair = pack_quantity(raw, row.get('Product Name', ''))
    if pair and canonical(raw).removesuffix(' net').strip() != canonical(pair[1]) and not re.match(
            r'^(?:pack size|net content|net contents|net volume|contents)\b', canonical(raw)):
        pair = None
    if pair and parent.metadata.get('panel_role') not in ('nutrition_basis', 'serving_quantity'):
        existing = [f.value for f in evidence.facts if f.field == 'product_specifications' and f.path == ['pack_size']]
        count_units = {'pack', 'capsule', 'tablet', 'piece', 'sachet', 'softgel', 'wipe', 'pair'}
        equivalent = any(canonical(p['value']) == canonical(pair[0]['value']) and
            (canonical(p['unit']) == canonical(pair[0]['unit']) or
             canonical(p['unit']).rstrip('s') in count_units and canonical(pair[0]['unit']).rstrip('s') in count_units)
            for p in existing)
        if not equivalent:
            evidence.add('product_specifications', ['pack_size'], pair[0], parent.id, raw,
                         confidence=line['confidence'], method='ocr_pack')
    if re.search(r'COSMETIC', row.get('category_group', ''), re.I):
        match = re.fullmatch(r'\d{2,4}\s+([A-Za-z][A-Za-z -]{1,35})', text)
        if match and canonical(match[1]) not in {'ml', 'g', 'hrs', 'hours', 'spf', 'pack'}:
            value = match[1].strip()
            if f' {canonical(value)} ' in f" {canonical(row['Product Name'])} ":
                evidence.add('variant_attributes', ['shade'], [value], parent.id, raw,
                             confidence=line['confidence'], method='ocr_labelled_shade')
    for pattern in (r'\b(?:matte|styling|texturi[sz]ing)\s+paste\b',
                    r'\b(?:medium|strong|light|firm|extra strong)\s+(?:control|hold)\b'):
        match = re.search(pattern, text, re.I)
        if match and re.search(r'HAIR|STYLING', row.get('category_group', '') + ' ' + row.get('T_BMC', ''), re.I):
            grounded(evidence, parent, raw, 'description', match.group(), line['confidence'])
    category = row.get('category_group', '')
    features = []
    if re.search(r'COSMETIC|BEAUTY|PERSONAL CARE|HAIR', category, re.I):
        features += [m.group() for m in re.finditer(r'\b(?:mattifying|strong hold hairspray|medium control|gel\s*to\s*powder)\b', text, re.I)]
    if re.search(r'TOYS', category, re.I):
        features += [m.group() for m in re.finditer(r'\b(?:one|two|three|four|\d+)\s+body patterns\b', text, re.I)]
        if re.fullmatch(r'[3-9]\+', text):
            features.append(text)
    for value in features:
        grounded(evidence, parent, raw, 'claim', value, line['confidence'])
    match = re.search(r'\bSPF\s*(\d+\+?)\b', text, re.I)
    if match:
        evidence.add('product_specifications', ['other'], [{'attribute': 'SPF', 'value': match[1]}],
                     parent.id, raw, confidence=line['confidence'], method='ocr_labelled_spf')


def route_evidence(row, evidence, cfg):
    from .text_extraction import NUTRIENTS
    from .visual import printed_role
    from .ocr_evidence import expand_spans, formulation_disagreement
    expand_spans(evidence)
    for source in evidence.sources.values():
        if source.kind == 'visual_printed':
            source.metadata['content_role'] = printed_role(source.text, source.metadata.get('printed_kind'), row)
    for fact in evidence.facts:
        if fact.field == 'nutritional_information' and len(fact.path) == 2:
            fact.path[1] = NUTRIENTS.get(fact.path[1].casefold(), fact.path[1])
    groups = {}
    for source in evidence.sources.values():
        if source.kind == 'ocr':
            path = source.metadata.get('image_path', source.location.split('/block/', 1)[0])
            groups.setdefault(path, []).append(source)
    for path, sources in groups.items():
        formulation = formulation_disagreement(row, sources)
        if formulation:
            for source in sources:
                source.eligible = False
                source.exact_product = False
                source.metadata.update(identity_status='unverifiable', identity_hold_reason=formulation)
            evidence.issues.append({'kind': 'formulation_disagreement', 'image': path, 'reason': formulation})
            continue
        conflict = ocr_quantity_conflict(row, sources)
        confirmed = image_identity(row, sources) or any(
            s.kind == 'visual_printed' and s.eligible and s.metadata.get('identity_status') == 'confirmed' and
            s.metadata.get('printed_kind') == 'identity' and s.location == path
            for s in evidence.sources.values())
        for source in sources:
            if conflict or source.metadata.get('identity_status') == 'mismatch':
                source.eligible = False
                source.metadata['identity_status'] = 'mismatch'
                continue
            source.exact_product = confirmed
            brand = canonical(row.get('T_BRAND', ''))
            brand_panel = bool(brand and any((s.confidence or 0) >= .8 and
                f' {brand} ' in f' {canonical(s.text)} ' for s in sources if s.eligible))
            source.metadata.update({'identity_status': 'confirmed' if confirmed else 'unverifiable',
                'association_basis': 'provided_image_mapping',
                'identity_basis': 'ocr_brand_and_product_anchors' if confirmed else 'mapping_only',
                'mapped_brand_panel': brand_panel})
        if conflict:
            evidence.issues.append({'kind': 'ocr_quantity_mismatch', 'image': path})
            for source in evidence.sources.values():
                if source.kind in ('visual_printed', 'visual_observation') and source.location == path:
                    source.eligible = False
                    source.metadata['identity_status'] = 'mismatch'
    evidence.facts = [f for f in evidence.facts if evidence.sources[f.source_id].kind != 'visual_observation']
    for source in list(evidence.sources.values()):
        reason = appearance_reason(source, row)
        if reason:
            source.eligible = False
            source.metadata['excluded_reason'] = reason
            evidence.issues.append({'kind': reason, 'source_id': source.id})
    # Both panels and reliable individual spans may contain recoverable facts.
    parents = [s for s in evidence.sources.values() if s.eligible and (s.exact_product or
        s.kind == 'ocr' and s.metadata.get('mapped_brand_panel') and s.metadata.get('panel_role') == 'storage') and
        (s.kind == 'ocr' or s.kind == 'record' and
         s.location.split('#')[0] in ('Short Description', 'Product Description') or s.kind == 'visual_printed')]
    for parent in parents:
        if parent.kind == 'ocr':
            if parent.metadata.get('panel_role') == 'storage' and (parent.confidence or 0) >= .9:
                identifier = grounded(evidence, parent, parent.text, 'storage', confidence=parent.confidence)
                evidence.sources[identifier].metadata['mapped_storage_only'] = not parent.exact_product
                evidence.add('storage_instructions', [], [evidence.sources[identifier].text], identifier)
                if not parent.exact_product:
                    continue
            lines = parent.metadata.get('lines', [{'text': x, 'confidence': parent.confidence} for x in parent.text.splitlines()])
            for index, line in enumerate(lines):
                raw = line['text']
                normal = clean_text(raw)
                confidence = line['confidence']
                if confidence is None or confidence < .9:
                    continue
                route_ocr_facts(row, evidence, parent, line)
                # Recover explicit negative declarations, retaining their scope and not inventing a free-from claim.
                for match in re.finditer(r'\bno\s+(?:artificial\s+)?(?:preservatives?|colou?rants?|flavou?rings?)\b', normal, re.I):
                    identifier = grounded(evidence, parent, raw, 'claim', match.group(), confidence)
                    evidence.sources[identifier].metadata['normalised_line'] = normal
                for match in re.finditer(r'\b(?:100\s*%\s*natural|hypoallergenic)\b', normal, re.I):
                    grounded(evidence, parent, raw, 'claim', match.group(), confidence)
                # Join short consecutive physical rows for sentences wrapped on packaging.
                for length in (1, 2, 3):
                    window = lines[index:index + length]
                    if len(window) != length or any(x['confidence'] is None for x in window):
                        continue
                    joined = '\n'.join(x['text'] for x in window)
                    cleaned = clean_text(joined)
                    minimum = min(x['confidence'] for x in window)
                    for pattern, role in [
                        (r'^(?:may\s+help|helps?)\b.+?(?=\bNo need\b|\b\d+\s*(?:ml|g)\b|$)', 'benefit'),
                        (r'\bNo need to remove\s+before breastfeeding\b', 'usage'),
                        (r'\b(?:store\s+(?:in|below|at|away)|keep refrigerated)\b.+?(?=\bWarnings?\b|$)', 'storage'),
                    ]:
                        match = re.search(pattern, cleaned, re.I)
                        if match and minimum >= (.95 if role in ('benefit', 'usage') else .9):
                            identifier = grounded(evidence, parent, joined, role, match.group().strip(' ,;&'), minimum)
                            evidence.sources[identifier].metadata['normalised_line'] = cleaned
                            if role == 'storage':
                                evidence.add('storage_instructions', [], [evidence.sources[identifier].text], identifier)
            # Exact printed variant names can be corroborated by the supplied product name.
            for line in lines:
                value = clean_text(line['text'])
                if (line['confidence'] or 0) >= .95:
                    for match in re.finditer(r'\b(?:Original|Unscented|Unflavoured)\b', value, re.I):
                        if canonical(match.group()) in canonical(row['Product Name']):
                            identifier = grounded(evidence, parent, line['text'], 'variant', match.group(), line['confidence'])
                            evidence.add('variant_attributes', ['other'], [match.group()], identifier)
        elif parent.kind == 'visual_printed':
            role = {'claim': 'claim', 'highlighted_ingredient': 'highlighted_ingredient',
                    'usage': 'usage', 'storage': 'storage'}.get(parent.metadata.get('content_role'))
            if role:
                grounded(evidence, parent, parent.text, role)
        else:
            role_hint = None
            for sentence in re.split(r'(?<=[.!?])\s+|\n+', parent.text):
                if not sentence.strip():
                    continue
                heading = sentence.strip().rstrip(':').casefold()
                if heading in ('warnings', 'warning', 'contraindications', 'usage instructions', 'directions', 'key features'):
                    role_hint = 'warning' if heading in ('warnings', 'warning', 'contraindications') else 'usage' if heading in ('usage instructions', 'directions') else None
                    continue
                if re.match(r'^\s*Specifications\s*:', sentence, re.I):
                    continue
                if re.match(r'^\s*(?:Store\s+(?:in|below|at|away)|Keep refrigerated)\b', sentence, re.I):
                    continue
                if re.search(r'HOME SCENTS|PERFUM', row.get('category_group', ''), re.I):
                    notes = fragrance_notes(sentence)
                    if notes:
                        identifier = grounded(evidence, parent, sentence, 'description', notes)
                        evidence.sources[identifier].metadata['rendering'] = 'fragrance_notes'
                        if HYPE.search(sentence):
                            evidence.issues.append({'kind': 'speculative_source_claim_removed', 'source_id': parent.id, 'text': sentence})
                        continue
                if HYPE.search(sentence):
                    evidence.issues.append({'kind': 'speculative_source_claim_removed', 'source_id': parent.id, 'text': sentence})
                    continue
                role = role_hint or ('benefit' if CLAIM.search(sentence) else 'description')
                clauses = re.split(r',\s*(?=it (?:offers|provides|gives)\b)', sentence.strip(), flags=re.I)
                for clause in clauses:
                    grounded(evidence, parent, sentence.strip(), role, clean_text(clause))
    for source in list(evidence.sources.values()):
        if source.kind == 'record' and source.eligible and source.location.startswith('Specifications/'):
            label, separator, value = source.text.partition(':')
            sentence = specification_sentence(label.strip(), value.strip()) if separator else None
            if sentence:
                identifier = grounded(evidence, source, source.text, 'specification', sentence)
                evidence.sources[identifier].metadata['rendering'] = 'specification'
    for source in evidence.sources.values():
        if source.kind == 'grounded_text':
            parent = evidence.sources.get(source.metadata.get('parent_source_id'))
            if parent is None or not parent.eligible or not (parent.exact_product or source.metadata.get('mapped_storage_only')):
                source.eligible = False
                source.metadata['excluded_reason'] = 'parent_evidence_not_eligible'


def grounded_reason(source, evidence):
    meta = source.metadata
    parent = evidence.sources.get(meta.get('parent_source_id'))
    quote = meta.get('quote', '')
    storage_panel = bool(parent and parent.kind == 'ocr' and meta.get('role') == 'storage' and
        meta.get('mapped_storage_only') and parent.metadata.get('mapped_brand_panel') and
        parent.metadata.get('panel_role') == 'storage' and not parent.metadata.get('identity_hold_reason'))
    if parent is None or not parent.eligible or not (parent.exact_product or storage_panel) or not source.eligible:
        return 'unverified_product_text'
    if normalise(quote) not in normalise(parent.text):
        return 'grounded_quote_not_in_source'
    cleaned = clean_text(quote)
    # Normalisation may remove a tick or join a wrapped sentence, but must not add facts.
    if (canonical(source.text) not in canonical(cleaned) and not rendering_supported(source, parent)) or HYPE.search(source.text):
        return 'unsupported_text_normalisation'
    if parent.kind == 'ocr':
        required = .95 if (meta.get('role') in ('benefit', 'usage') or
                           meta.get('role') == 'storage' and re.search(r'\d', source.text)) else .9
        original_confidence = quote_confidence(parent, quote)
        if source.confidence is None or original_confidence is None or min(source.confidence, original_confidence) < required:
            return 'low_ocr_claim_confidence'
    if parent.kind == 'visual_printed':
        from .verification import visual_evidence_reason
        reason = visual_evidence_reason(parent, evidence)
        if reason:
            return reason
    return None


def claim_group(sources):
    values = [s.text.strip().rstrip('.!?') for s in sources]
    if all(re.match(r'^no\s+', value, re.I) for value in values):
        values = [re.sub(r'^no\s+', '', value, flags=re.I).lower() for value in values]
        joined = values[0] if len(values) == 1 else ', '.join(values[:-1]) + ' or ' + values[-1]
        return 'The packaging states: No ' + joined + '.'
    return 'The label states: ' + '; '.join(values) + '.'


def short_identity(name, brand):
    core = name
    brand_key = re.sub(r'\W', '', brand).casefold()
    words = list(re.finditer(r'\S+', name))
    for index in range(min(len(words), 5)):
        if re.sub(r'\W', '', name[:words[index].end()]).casefold() == brand_key:
            core = name[words[index].end():].strip()
            break
    return core or name


def redundant(text, previous):
    def claim_key(value):
        value = re.sub(r'\b(support|offer|help|provide)s\b', r'\1', canonical(value))
        value = re.sub(r'\belastici[sz]ed\b|\belasticated\b', 'elastic', value)
        return re.sub(r'\bpremium\s+', '', value)
    key = claim_key(text)
    words = set(key.split()) - {'the', 'a', 'and', 'of', 'for', 'with', 'this', 'is', 'it'}
    for other in previous:
        existing = claim_key(other)
        if key in existing or existing in key:
            return True
        other_words = set(existing.split()) - {'the', 'a', 'and', 'of', 'for', 'with', 'this', 'is', 'it'}
        if words and other_words and len(words & other_words) / len(words | other_words) >= .72:
            return True
    return False


def appearance_text(source):
    # Use the observation itself. Its supporting explanation is not a second product claim.
    value = re.sub(r'^(?:one|two|three|four|five|\d+)\s+', '', source.text, flags=re.I)
    if re.search(r'\b(?:individual|visible|\d+)\b', value, re.I):
        return None
    return value.strip().rstrip('.')


def appearance_sentence(source, name):
    value = appearance_text(source)
    if not value:
        return None
    if source.metadata.get('attribute') == 'colour':
        noun = re.search(r'\b(?:socks|crayons|towels?|blankets?|shoes|bibs?|rompers?|balls?|brushes)\b', name, re.I)
        if noun and canonical(noun.group()) not in canonical(value):
            noun = noun.group().lower()
            verb = 'are' if noun.endswith('s') else 'is'
            return f'The {noun} {verb} {value.lower()}.'
    return 'Design details: ' + value + '.'


def compose(row, evidence):
    name_source = next((s for s in evidence.sources.values() if s.kind == 'record' and s.location == 'Product Name'), None)
    if name_source is None:
        return {'short_description': [], 'long_description': []}
    name = name_source.text
    short = short_identity(name, row.get('T_BRAND', ''))
    result = {'short_description': [{'text': short.rstrip('.') + '.', 'source_ids': [name_source.id], 'mode': 'identity_span'}],
              'long_description': [{'text': name.rstrip('.') + '.', 'source_ids': [name_source.id], 'mode': 'source'}]}
    supplied_short = [s for s in evidence.sources.values() if s.kind == 'grounded_text' and
        s.metadata.get('role') in ('description', 'benefit') and
        not s.metadata.get('rendering') and
        evidence.sources[s.metadata['parent_source_id']].location.startswith('Short Description') and
        not grounded_reason(s, evidence)]
    supplied_short = [s for s in supplied_short if len(s.text) <= 240]
    if supplied_short:
        result['short_description'] = [{'text': s.text.rstrip('.') + '.', 'source_ids': [s.id], 'mode': 'product_text'}
                                       for s in supplied_short[:1]]
    short_texts = [s['text'] for s in result['short_description']]
    selected = []
    ranks = {'benefit': 0, 'specification': 1, 'description': 2, 'claim': 3, 'highlighted_ingredient': 3, 'usage': 4}
    for source in sorted((s for s in evidence.sources.values() if s.kind == 'grounded_text'),
                         key=lambda s: (ranks.get(s.metadata.get('role'), 9), -len(s.text))):
        if source.metadata.get('role') not in ranks or grounded_reason(source, evidence):
            continue
        if supplied_short and canonical(source.text).startswith(canonical(name) + ' '):
            continue
        if canonical(source.text) in canonical(name) or redundant(source.text, short_texts + [s.text for s in selected]):
            continue
        selected.append(source)
    # Actual product information wins over appearance. Do not add artwork to consume space.
    if selected:
        first = selected[0]
        if first.metadata.get('role') in ('benefit', 'specification') and not supplied_short:
            text = short.rstrip('.') + '. ' + first.text.rstrip('.') + '.'
            result['short_description'] = [{'text': text, 'source_ids': [name_source.id, first.id], 'mode': 'product_summary'}]
            selected = selected[1:]
        claims = [s for s in selected if s.metadata.get('role') == 'claim']
        if claims:
            result['long_description'].append({'text': claim_group(claims[:4]),
                'source_ids': [s.id for s in claims[:4]], 'mode': 'product_text'})
        paste = next((s for s in selected if re.fullmatch(r'(?:matte|styling|texturi[sz]ing) paste', s.text, re.I)), None)
        hold = next((s for s in selected if re.fullmatch(r'(?:medium|strong|light|firm|extra strong) (?:control|hold)', s.text, re.I)), None)
        if paste and hold:
            result['long_description'].append({'text': paste.text + ' with ' + hold.text.lower() + '.',
                'source_ids': [paste.id, hold.id], 'mode': 'product_text'})
            selected = [s for s in selected if s.id not in (paste.id, hold.id)]
        for source in [s for s in selected if s.metadata.get('role') != 'claim'][:3]:
            prefix = ('The label highlights: ' if source.metadata.get('role') == 'highlighted_ingredient' else
                      'The label states: ' if source.metadata.get('claim_scope') == 'label_claim' else '')
            result['long_description'].append({'text': prefix + source.text.rstrip('.!?') + '.',
                'source_ids': [source.id], 'mode': 'product_text'})
    observations = [s for s in evidence.sources.values() if s.kind == 'visual_observation' and s.eligible
                    and not appearance_reason(s, row) and s.metadata.get('attribute') != 'visible_count']
    details = []
    for source in observations:
        value = appearance_text(source)
        if not value or canonical(value) in canonical(name) or redundant(value, [s.text for s in selected]):
            continue
        if redundant(value, [s.text for s in details]):
            continue
        details.append(source)
    for source in details[:2]:
        sentence = appearance_sentence(source, name)
        if sentence:
            result['long_description'].append({'text': sentence,
                'source_ids': [name_source.id, source.id], 'mode': 'product_appearance'})
    return result


def rendered_sentence_reason(sentence, sources, evidence):
    """Recognise closed, grounded renderings without treating token overlap as entailment."""
    if not sources:
        return 'invalid_sentence_sources'
    for source in sources:
        if source.kind == 'grounded_text':
            reason = grounded_reason(source, evidence)
            if reason:
                return reason
    text = sentence.strip().rstrip('.')
    if len(sources) == 1 and sources[0].kind == 'record' and sources[0].location == 'Product Name':
        if canonical(text) in canonical(sources[0].text):
            return None
    if all(s.kind == 'grounded_text' and s.metadata.get('role') == 'claim' for s in sources):
        if canonical(sentence) == canonical(claim_group(sources)):
            return None
    if len(sources) == 2 and sources[0].kind == 'record' and sources[0].location == 'Product Name' and sources[1].kind == 'grounded_text':
        parts = sentence.split('. ', 1)
        if len(parts) == 2 and canonical(parts[0]) in canonical(sources[0].text) and canonical(parts[1]) == canonical(sources[1].text):
            return None
    if (len(sources) == 2 and sources[0].kind == 'record' and sources[0].location == 'Product Name' and
            sources[1].kind == 'visual_observation' and sources[1].metadata.get('attribute') != 'visible_count'):
        expected = appearance_sentence(sources[1], sources[0].text)
        if expected and canonical(sentence) == canonical(expected):
            return None
    if len(sources) == 1 and sources[0].kind == 'visual_observation' and sources[0].eligible:
        value = appearance_text(sources[0])
        if value and canonical(sentence) == canonical('Design details: ' + value + '.'):
            return None
    if len(sources) == 1 and sources[0].kind == 'grounded_text':
        source = sources[0]
        if canonical(text) in (canonical(source.text), canonical('The label states: ' + source.text),
                              canonical('The label highlights: ' + source.text)):
            return None
    return 'not_closed_rendering'
