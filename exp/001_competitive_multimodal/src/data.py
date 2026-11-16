from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from openpyxl import load_workbook

INPUT_COLUMNS = ['record_id', 'Product Name', 'category_group', 'Short Description',
                 'Product Description', 'Specifications', 'Nutritional Information', 'T_BRAND']


def csv_rows(path: Path, id_column: str | None = None):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'Missing or duplicate CSV headers: {path}')
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'CSV row width mismatch: {path}')
    if id_column:
        ids = [row[id_column] for row in rows]
        if any(not x or x != x.strip() for x in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'Invalid or duplicate identifiers: {path}')
    return rows


def clean(value) -> str:
    if value is None or isinstance(value, float) and math.isnan(value):
        return ''
    return str(value).strip()


def visible(value) -> bool:
    return bool(clean(value)) and '[MASKED]' not in clean(value).upper()


def plain(value) -> str:
    soup = BeautifulSoup(clean(value), 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    return soup.get_text(' ', strip=True)


def blocked(row) -> set[str]:
    raw = row.get('do_not_predict_fields', '')
    if not raw:
        return set()
    value = json.loads(raw)
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError('do_not_predict_fields must be a JSON array of strings')
    return set(value)


def load_inputs(data_dir: Path):
    result = {}
    for split, filename, id_col in [
        ('test', 'Test_Masked.csv', 'record_id'),
        ('reference', 'unlabelled_data.csv', 'record_id'),
        ('template', 'template_input.csv', 'record_id'),
        ('labels', 'template.csv', 'article_id'),
        ('sample', 'SampleSubmission.csv', 'article_id'),
    ]:
        rows = csv_rows(data_dir / filename, id_col)
        if split in ('test', 'reference', 'template'):
            if not rows or not set(INPUT_COLUMNS) <= rows[0].keys():
                raise ValueError(f'Unexpected input schema in {filename}')
            for row in rows:
                blocked(row)
        result[split] = rows
    if {r['record_id'] for r in result['test']} != {r['article_id'] for r in result['sample']}:
        raise ValueError('Sample and test ID sets differ')
    if {r['record_id'] for r in result['template']} != {r['article_id'] for r in result['labels']}:
        raise ValueError('Template input and label ID sets differ')
    return result


def mapping_path(data_dir: Path, image_dir: Path):
    for path in (data_dir / 'image_mapping.csv', image_dir / 'image_mapping.csv'):
        if path.is_file():
            return path
    return data_dir / 'image_mapping.xlsx'


def image_mapping(data_dir: Path, image_dir: Path):
    mapping = defaultdict(list)
    warnings = []
    source = mapping_path(data_dir, image_dir)
    wb = None
    csv_file = None
    try:
        if source.suffix == '.csv':
            csv_file = source.open(newline='', encoding='utf-8-sig')
            rows = csv.reader(csv_file)
        else:
            wb = load_workbook(source, read_only=True, data_only=True)
            rows = wb.active.iter_rows(values_only=True)
        headers = [clean(x) for x in next(rows, ())]
        if len(headers) != len(set(headers)):
            raise ValueError('Duplicate image mapping headers')
        if not {'record_id', 'img_filename'} <= set(headers):
            raise ValueError('Image mapping requires record_id and img_filename columns')
        id_idx, image_idx = headers.index('record_id'), headers.index('img_filename')
        for row in rows:
            raw_id = row[id_idx] if id_idx < len(row) else None
            raw_name = row[image_idx] if image_idx < len(row) else None
            name = clean(raw_name)
            if not name:
                continue
            # Excel stores these identifiers numerically; CSV identifiers remain untouched.
            if isinstance(raw_id, (int, float)):
                if isinstance(raw_id, bool) or not math.isfinite(raw_id) or raw_id != int(raw_id) or abs(raw_id) > 2**53:
                    raise ValueError('Excel mapping contains an ambiguous numeric identifier')
                key = str(int(raw_id))
            else:
                key = clean(raw_id)
            if not key:
                continue
            candidate = (image_dir / name).resolve()
            if not candidate.is_relative_to(image_dir.resolve()):
                raise ValueError('Image mapping escapes configured image directory')
            if not candidate.is_file():
                relative_to_dataset = (data_dir / name).resolve()
                if relative_to_dataset.is_relative_to(image_dir.resolve()) and relative_to_dataset.is_file():
                    candidate = relative_to_dataset
            if not candidate.is_file():
                warnings.append({'kind': 'missing_image', 'record_id': key, 'image': name})
                continue
            if candidate not in mapping[key]:
                mapping[key].append(candidate)
    finally:
        if wb is not None:
            wb.close()
        if csv_file is not None:
            csv_file.close()
    return dict(mapping), warnings
