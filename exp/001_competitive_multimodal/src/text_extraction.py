from __future__ import annotations

import json
import re
from decimal import Decimal

from bs4 import BeautifulSoup

from .data import plain, visible
from .evidence import Evidence

NUMBER = r'\d+(?:[.,]\d+)?'
UNIT = r'(?:millilitre\s*\(ml\)|milliliter\s*\(ml\)|gram\s*\(g\)|kilogram\s*\(kg\)|ml|cl|litres?|liters?|l|kg|mg|grams?|g|tablets?|capsules?|softgels?|sachets?|pieces?|pcs|wipes?|pads?|pack|pairs?)'
QUANTITY = re.compile(rf'(?<![\w.])(?:(\d+)\s*[x×]\s*)?({NUMBER})\s*({UNIT})(?:(?!\w)|(?=net\b))', re.I)
MEASURE = re.compile(rf'^\s*({NUMBER})\s*(mm|cm|m|inch(?:es)?|kg|mg|g|grams?|kilograms?|ml|l)\s*$', re.I)
NUTRIENTS = {'energy (kj)': 'energy_kj', 'energy (kcal)': 'energy_kcal',
             'carbohydrate (g)': 'carbohydrate_g', 'of which total sugar (g)': 'total_sugar_g',
             'dietary fibre (g)': 'dietary_fibre_g', 'total fat (g)': 'total_fat_g',
             'of which saturated fat (g)': 'saturated_fat_g', 'protein (g)': 'protein_g',
             'total sodium (mg)': 'sodium_mg', 'salt (g)': 'salt_g'}
SPEC_ALIASES = {'shade': ('variant_attributes', ['shade']),
                'colour': ('variant_attributes', ['shade']),
                'color': ('variant_attributes', ['shade']),
                'flavour': ('variant_attributes', ['flavour']),
                'flavor': ('variant_attributes', ['flavour']),
                'fragrance': ('variant_attributes', ['fragrance']),
                'application area': ('product_specifications', ['application_area']),
                'packaging material': ('product_specifications', ['packaging_material']),
                'packaging type': ('product_specifications', ['packaging_type'])}
NUTRIENTS.update({'polyunsaturated fat (g)': 'polyunsaturated_fat_g',
                  'monounsaturated fat (g)': 'monounsaturated_fat_g',
                  'saturated fat (g)': 'saturated_fat_g', 'sodium (mg)': 'sodium_mg',
                  'total sugar (g)': 'total_sugar_g', 'dietary fiber (g)': 'dietary_fibre_g',
                  'cholesterol (mg)': 'cholesterol_mg'})


def spec_pairs(raw):
    soup = BeautifulSoup(raw, 'html.parser')
    result = []
    for tr in soup.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        if len(cells) == 2:
            result.append((cells[0].get_text(' ', strip=True), cells[1].get_text(' ', strip=True)))
    if not result:
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                result = [(str(k), str(v)) for k, v in value.items()
                          if isinstance(v, (str, int, float))]
        except (ValueError, TypeError):
            for line in raw.splitlines():
                if ':' in line:
                    result.append(tuple(line.split(':', 1)))
    return [(k.strip(), v.strip()) for k, v in result if visible(k) and visible(v)]


def quantity(text):
    match = QUANTITY.search(text)
    if not match:
        return None
    count, value, unit = match.groups()
    return {'value': f'{count} x {value}' if count else value, 'unit': unit}, match.group()


def pack_quantity(text, context=''):
    """Keep component amounts; a dosage or strength is not a pack quantity."""
    matches = list(QUANTITY.finditer(text))
    if not matches:
        count = re.search(r"(?:(?<!\w)[x×]\s*(\d+)|\b(\d+)['’]s)\s*$", text, re.I)
        noun = list(re.finditer(r'\b(wipes|pads|capsules|tablets|softgels|sachets|pieces|pairs)\b', text + ' ' + context, re.I))
        if count and noun and not re.search(r'\b(?:model|code|series)\s*$', text[:count.start()], re.I):
            value = count[1] or count[2]
            # Include the supplied count noun in the quote when it is present in this source.
            preceding = [m for m in noun if m.end() <= count.start()]
            quote = text[preceding[-1].start():count.end()] if preceding else count.group()
            return {'value': value, 'unit': noun[0][1]}, quote
        return None
    counts = [m for m in matches if re.fullmatch(
        r'tablets?|capsules?|softgels?|sachets?|pieces?|pcs|wipes?|pads?|pack|pairs?', m[3], re.I)]
    if counts:
        matches = counts
    elif re.search(r'\b(?:tablets?|capsules?|softgels?|vitamin|strength|dosage)\b', text + ' ' + context, re.I):
        matches = [m for m in matches if m[3].casefold() != 'mg']
    if not matches:
        return None
    if len(matches) == 1:
        match = matches[0]
        count, value, unit = match.groups()
        return {'value': f'{count} x {value}' if count else value, 'unit': unit}, match.group()
    # Never choose just the first component or add up an unprovided total.
    if any(not re.fullmatch(r'\s*\+\s*', text[a.end():b.start()])
           for a, b in zip(matches, matches[1:])):
        return None
    units = [m[3].casefold() for m in matches]
    if len(set(units)) != 1 or any(m[1] for m in matches):
        return None
    return {'value': ' + '.join(m[2] for m in matches), 'unit': matches[0][3]}, text[matches[0].start():matches[-1].end()]


def quantity_unit(unit):
    unit = unit.casefold().strip()
    if unit.startswith(('millilitre', 'milliliter')):
        return 'ml'
    if unit.startswith('kilogram'):
        return 'kg'
    if unit.startswith('gram'):
        return 'g'
    if unit in ('litre', 'litres', 'liter', 'liters'):
        return 'l'
    if unit == 'pcs':
        return 'piece'
    return unit.rstrip('s')


def quantity_relation(printed, target):
    """Compare like quantities without summing components or converting units."""
    visible = pack_quantity(printed, target) or quantity(printed)
    if not visible:
        return 'unverifiable'
    pair = visible[0]
    unit = quantity_unit(pair['unit'])
    def value_key(value):
        return re.sub(NUMBER, lambda m: format(Decimal(m.group().replace(',', '.')).normalize(), 'f'), value).replace(' ', '').lower()
    if unit == 'mg' and re.search(r'\b(?:vitamin|capsules?|tablets?|softgels?|strength)\b', target, re.I):
        strengths = [m[2] for m in QUANTITY.finditer(target) if quantity_unit(m[3]) == 'mg']
        return 'strength' if value_key(pair['value']) in {value_key(s) for s in strengths} else 'different' if strengths else 'unverifiable'
    expected = pack_quantity(target)
    if not expected:
        return 'unverifiable'
    expected = expected[0]
    other_unit = quantity_unit(expected['unit'])
    count_units = {'capsule', 'tablet', 'softgel', 'piece', 'wipe', 'pad', 'sachet', 'pair'}
    if unit != other_unit:
        generic_count = (unit in ('pack', 'piece') and other_unit in count_units or other_unit in ('pack', 'piece') and unit in count_units)
        if not generic_count:
            return 'unverifiable'
        if other_unit in ('pack', 'piece'):
            forms = {quantity_unit(w) for w in re.findall(r'\b(?:capsules?|tablets?|softgels?|pieces?|wipes?|pads?|sachets?|pairs?)\b', target, re.I)}
            if forms and unit not in forms and unit not in ('pack', 'piece'):
                return 'different'
    if 'x' in expected['value']:
        numbers = re.findall(r'x\s*(' + NUMBER + ')', expected['value'])
    else:
        numbers = re.findall(NUMBER, expected['value'])
    if value_key(pair['value']) == value_key(expected['value']):
        return 'same'
    if '+' in expected['value'] or 'x' in expected['value']:
        return 'component' if value_key(pair['value']) in {value_key(n) for n in numbers} else 'different'
    return 'different'


def literal_pack_quantity(text, context=''):
    parsed = pack_quantity(text, context)
    if not parsed:
        return None
    before, _, after = text.partition(parsed[1])
    if before.strip().casefold() not in ('', 'pack size:', 'net contents:', 'net content:', 'net volume:', 'contents:'):
        return None
    if after.strip().casefold() not in ('', 'net', 'net.', 'e', '℮'):
        return None
    return parsed


def name_attributes(row, evidence, source_id):
    name = row.get('Product Name', '')
    match = re.search(r'\b(elbow|knee|ankle|wrist|thumb)\s+(?:support|brace|sleeve)\b', name, re.I)
    if match:
        evidence.add('product_specifications', ['application_area'], [match[1]], source_id, match.group())
    for match in re.finditer(r'\bSize\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:months?|years?)\b', name, re.I):
        evidence.add('variant_attributes', ['other'], [match.group()], source_id, match.group())
    match = re.search(r'\b(?:Extra\s+Small|Extra\s+Large|Small|Medium|Large|XXL|XL)\s*$', name, re.I)
    if match:
        evidence.add('variant_attributes', ['other'], [match.group()], source_id, match.group())
    category = row.get('category_group', '')
    if re.search(r'HOME SCENTS', category, re.I) and not any(
            f.field == 'variant_attributes' and f.path == ['fragrance'] for f in evidence.facts):
        core = re.sub(r'\s+\d+(?:[.,]\d+)?\s*(?:ml|g|kg|l)\s*$', '', name, flags=re.I)
        match = re.search(r'\b(?:Candles?|Diffuser|Room Spray|Room And Linen Mist)\s+(.+)$', core, re.I)
        value = match[1].strip() if match else ''
        if not value:
            match = re.search(r'^(.*?)\s+(?:Fragrance\s+)?(?:Diffuser|Candle)\s*$', core, re.I)
            if match:
                value = re.sub(r'^' + re.escape(row.get('T_BRAND', '')) + r'\s+', '', match[1], flags=re.I).strip()
        if value and value.casefold() != row.get('T_BRAND', '').casefold() and valid_variant('fragrance', value, row):
            evidence.add('variant_attributes', ['fragrance'], [value], source_id, value)
    if re.search(r'\bAssorted\b', name, re.I):
        match = re.search(r'\bAssorted\b', name, re.I)
        evidence.add('variant_attributes', ['other'], [match.group()], source_id, match.group())


def valid_variant(path, value, row=None):
    value = str(value).strip()
    if not value or '[MASKED]' in value.upper():
        return False
    if path in ('shade', 'flavour', 'fragrance') and literal_pack_quantity(value):
        return False
    if path == 'fragrance' and re.fullmatch(r'[\d\s.,+×x]+', value, re.I):
        return False
    if path == 'flavour' and row and re.search(r'HOME SCENTS|PERFUM', row.get('category_group', ''), re.I):
        return False
    return True


def nutrition(raw, evidence, source):
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        evidence.issues.append({'kind': 'malformed_nutrition'})
        return
    if not isinstance(data, dict):
        return
    serving = data.get('serving_size', '')
    if isinstance(serving, str) and visible(serving):
        q = quantity(serving)
        if q:
            evidence.add('nutritional_information', ['serving_size'], q[0], source, raw)
    labels = data.get('source_column_labels', {})
    if not isinstance(labels, dict):
        labels = {}
    nutrients = data.get('nutrients', {})
    if not isinstance(nutrients, dict):
        evidence.issues.append({'kind': 'malformed_nutrition'})
        return
    for nutrient, values in nutrients.items():
        if not isinstance(values, dict):
            continue
        key = NUTRIENTS.get(nutrient.casefold())
        if key is None:
            # Preserve the declared nutrient label and unit for unfamiliar nutrients.
            key = nutrient
        for basis in ('per_100g', 'per_serving'):
            value = values.get(basis)
            label = str(labels.get(basis, ''))
            # The schema has no per_100ml slot: do not silently relabel that basis.
            if basis == 'per_100g' and (not re.search(r'100\s*g\b', label, re.I)):
                continue
            if basis == 'per_serving' and not re.search(r'per\s+serv', label, re.I):
                continue
            if not visible(value) or not re.fullmatch(NUMBER, str(value)):
                if visible(value):
                    evidence.issues.append({'kind': 'unrepresentable_nutrition_value',
                                            'nutrient': nutrient, 'basis': basis})
                continue
            numeric = float(str(value).replace(',', '.'))
            evidence.add('nutritional_information', [basis, key], numeric, source, raw)


def extract_nutrition_text(evidence, source_id):
    source = evidence.sources[source_id]
    rows = source.metadata.get('lines', [{'text': x, 'confidence': source.confidence}
                                          for x in source.text.splitlines()])
    bases, header, header_confidence = [], '', 0.
    cells = []
    for row in rows:
        text = row['text'].strip()
        headings = list(re.finditer(r'per\s*100\s*(g|ml)\b|per\s+serv(?:ing|e)\b', text, re.I))
        if headings:
            bases = [('per_100g' if m[1] and m[1].casefold() == 'g' else
                      'unsupported_100ml' if m[1] else 'per_serving') for m in headings]
            header, header_confidence = text, row['confidence']
            continue
        # A fully labelled row and unambiguous explicit unit are required.
        m = re.fullmatch(r'\s*([A-Za-z][A-Za-z /-]*?)\s*\((kJ|kcal|g|mg)\)\s*[:|]?\s*(.+)', text, re.I)
        if not m or not bases or header_confidence < .95 or row['confidence'] < .95:
            continue
        values = re.findall(r'[<>≤≥]?\s*\d+(?:[.,]\d+)?', m[3])
        remainder = re.sub(r'[<>≤≥]?\s*\d+(?:[.,]\d+)?|[|\s]', '', m[3])
        if remainder or len(values) != len(bases):
            evidence.issues.append({'kind': 'ambiguous_nutrition_columns', 'source_id': source_id})
            continue
        label = f'{m[1].strip()} ({m[2]})'
        key = NUTRIENTS.get(label.casefold(), label)
        for basis, raw_value in zip(bases, values):
            if basis == 'unsupported_100ml' or not re.fullmatch(NUMBER, raw_value.strip()):
                continue
            value = float(raw_value.replace(',', '.'))
            cells.append({'basis': basis, 'nutrient': key, 'value': value, 'row': text,
                          'header': header, 'confidence': min(header_confidence, row['confidence'])})
            evidence.add('nutritional_information', [basis, key], value, source_id,
                         source.text, confidence=min(header_confidence, row['confidence']), method='nutrition_table')
    if cells:
        source.metadata['nutrition_cells'] = cells


def extract(row, kind='record', exact=True, eligible=True):
    evidence = Evidence()
    name_source = None
    for field, raw in row.items():
        if field == 'do_not_predict_fields' or not raw:
            continue
        # Preserve readable sections around masks, but never join the two sides into a fact.
        for index, fragment in enumerate(re.split(r'\[MASKED\]', str(raw), flags=re.I)):
            text = plain(fragment)
            if text:
                evidence.source(kind, f'{field}#{index}', text, exact=exact, eligible=eligible)
    name = row.get('Product Name', '')
    if visible(name):
        src = evidence.source(kind, 'Product Name', plain(name), exact=exact, eligible=eligible)
        name_source = src
        evidence.add('online_name', [], plain(name), src)
        q = pack_quantity(name)
        if q:
            evidence.add('product_specifications', ['pack_size'], q[0], src, q[1])
        elif len(list(QUANTITY.finditer(name))) > 1:
            evidence.issues.append({'kind': 'ambiguous_pack_components', 'source_id': src})
    for label, value in spec_pairs(row.get('Specifications', '')):
        text = f'{label}: {value}'
        src = evidence.source(kind, f'Specifications/{label}', text, exact=exact, eligible=eligible)
        key = label.casefold()
        if key == 'pack size':
            q = pack_quantity(value)
            if q:
                evidence.add('product_specifications', ['pack_size'], q[0], src)
        elif key in SPEC_ALIASES:
            field, path = SPEC_ALIASES[key]
            if field == 'variant_attributes':
                if path == ['flavour'] and re.search(r'HOME SCENTS|PERFUM', row.get('category_group', ''), re.I):
                    path = ['fragrance']
                if not valid_variant(path[0], value, row):
                    evidence.issues.append({'kind': 'invalid_variant_value', 'source_id': src})
                    continue
            evidence.add(field, path, value if key == 'packaging type' else [value], src)
        elif key in ('weight', 'net weight', 'gross weight'):
            m = MEASURE.fullmatch(value)
            if m and re.fullmatch(r'kg|mg|g|grams?|kilograms?', m[2], re.I):
                evidence.add('weight', [], {'value': m[1], 'unit': m[2]}, src)
            else:
                evidence.issues.append({'kind': 'weight_without_explicit_unit', 'label': label})
        elif key in ('height', 'width', 'depth'):
            m = MEASURE.fullmatch(value)
            if m and re.fullmatch(r'mm|cm|m|inch(?:es)?', m[2], re.I):
                evidence.add('dimensions', [key], m[1], src)
                evidence.add('dimensions', ['unit'], m[2], src)
            else:
                evidence.issues.append({'kind': 'dimension_without_explicit_unit', 'label': label})
        elif key in ('ingredients', 'ingredient information'):
            evidence.add('ingredient_information', [], [value], src)
        elif key in ('storage', 'storage instructions'):
            evidence.add('storage_instructions', [], [value], src)
        elif key not in ('serving size',):
            evidence.add('product_specifications', ['other'], [{'attribute': label, 'value': value}], src)
    if name_source:
        name_attributes(row, evidence, name_source)
    for field in ('Short Description', 'Product Description'):
        raw = row.get(field, '')
        # Labelled sections only. Keep whole declarations, including allergens and order.
        text = BeautifulSoup(raw, 'html.parser').get_text('\n', strip=True)
        for label, target in [('ingredients?', 'ingredient_information'),
                              ('storage(?: instructions)?', 'storage_instructions')]:
            pattern = rf'(?:^|\n|(?<=[.!?])\s+)\s*({label})\s*[:,]\s*(.+?)(?=\n\s*(?:[A-Za-z /]+:|Warnings|Usage Instructions|Key Features)|\s+(?:Storage|Warnings|Usage Instructions|Specifications)\s*[:,]|\Z)'
            for m in re.finditer(pattern, text, re.I | re.S):
                declaration = m[2].strip()
                if visible(declaration):
                    src = evidence.source(kind, field + '/' + m[1], m.group().strip(),
                                          exact=exact, eligible=eligible)
                    evidence.add(target, [], [declaration], src)
        for match in re.finditer(r'\b(?:Store\s+(?:in|below|at|away)|Keep refrigerated)\b[^.!?]*(?:[.!?]|$)', text, re.I):
            direction = match.group().strip()
            if visible(direction):
                src = evidence.source(kind, field + '/storage_text', direction, exact=exact, eligible=eligible)
                evidence.add('storage_instructions', [], [direction], src)
    raw = row.get('Nutritional Information', '')
    if raw and raw != '{}':
        src = evidence.source(kind, 'Nutritional Information/raw', raw, exact=exact, eligible=eligible)
        nutrition(raw, evidence, src)
    return evidence
