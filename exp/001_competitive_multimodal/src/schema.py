from __future__ import annotations

import copy
import json
import math
import re

from jsonschema import Draft202012Validator

TEXT = ('online_name', 'short_description', 'long_description')
STRING = {'type': 'string'}
STRINGS = {'type': 'array', 'items': STRING}


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


PAIR = obj({'value': STRING, 'unit': STRING})
SHAPES = {
    **{f: STRING for f in TEXT},
    'ingredient_information': STRINGS, 'storage_instructions': STRINGS,
    'variant_attributes': obj({k: STRINGS for k in ('shade', 'flavour', 'fragrance', 'other')}),
    'product_specifications': obj({
        'pack_size': PAIR, 'application_area': STRINGS, 'packaging_type': STRING,
        'packaging_material': STRINGS,
        'other': {'type': 'array', 'items': obj({'attribute': STRING, 'value': STRING})}}),
    'weight': PAIR,
    'dimensions': obj({k: STRING for k in ('height', 'width', 'depth', 'unit')}),
    'nutritional_information': obj({
        'serving_size': PAIR,
        'per_100g': {'type': 'object', 'additionalProperties': {'type': 'number'}},
        'per_serving': {'type': 'object', 'additionalProperties': {'type': 'number'}}}),
}


def leaves(value, prefix=()):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, prefix + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, prefix + (i,))
    elif value != '' and value is not None:
        yield prefix, value


class Schema:
    def __init__(self, path):
        self.raw = json.loads(path.read_text())
        self.columns = self.raw['required_columns']
        self.formats = self.raw['field_formats']
        if len(self.columns) != len(set(self.columns)) or set(self.columns) != {'article_id', *SHAPES}:
            raise ValueError('Official schema changed; update experiment before inference')
        self.empty = {f: copy.deepcopy(v['empty_value']) for f, v in self.formats.items()}
        for f, shape in SHAPES.items():
            if self.formats[f]['type'] != shape['type']:
                raise ValueError(f'Official type changed: {f}')
            Draft202012Validator(shape).validate(self.empty[f])
            if 'required_keys' in self.formats[f]:
                if set(self.formats[f]['required_keys']) != set(shape['properties']):
                    raise ValueError(f'Official object keys changed: {f}')

    def blank(self):
        return copy.deepcopy(self.empty)

    def decode(self, row):
        return {f: (copy.deepcopy(self.empty[f]) if row[f] == '' else row[f] if f in TEXT
                    else json.loads(row[f])) for f in self.empty}

    def errors(self, values, blocked_fields=()):
        errors = []
        if set(values) != set(self.empty):
            errors.append('Wrong output fields')
        for f in self.empty:
            if f not in values:
                continue
            for error in Draft202012Validator(SHAPES[f]).iter_errors(values[f]):
                errors.append(f'{f}: {error.message}')
            if f in blocked_fields and values[f] != self.empty[f]:
                errors.append(f'{f}: populated do_not_predict field')
            for path, value in leaves(values[f]):
                if isinstance(value, float) and not math.isfinite(value):
                    errors.append(f'{f}{path}: non-finite number')
                if isinstance(value, str) and (re.search(r'\[MASKED\]', value, re.I)
                    or value.strip().casefold() in {'n/a', 'none', 'unknown', 'nan', 'null'}):
                    errors.append(f'{f}{path}: forbidden placeholder')
        return errors
