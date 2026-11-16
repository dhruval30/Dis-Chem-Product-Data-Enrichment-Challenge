from pathlib import Path

import pytest
import yaml

from src import data
from src.common import ROOT
from src.enrichment import add_visual, finish
from src.schema import Schema
from src.submission import validate_traces
from src.text_extraction import extract


@pytest.fixture
def cfg():
    value = yaml.safe_load((ROOT / 'config.yaml').read_text())
    value['_schema'] = Schema(ROOT.parents[1] / 'dataset/submission_format.json')
    return value


def product(**changes):
    return {'record_id': '001', 'Product Name': 'Example Hand Balm 50ml', 'T_BRAND': 'Example',
            'category_group': 'PERSONAL CARE', 'T_CATEGORY': 'HAND CARE',
            'Short Description': '', 'Product Description': '', 'Specifications': '',
            'Nutritional Information': '{}', **changes}


def test_csv_mapping_preferred_and_blank_filenames_excluded(tmp_path, monkeypatch):
    image_dir = tmp_path / 'product_images'
    nested = image_dir / 'images' / 'nested'
    nested.mkdir(parents=True)
    image = nested / 'photo.jpg'
    image.write_bytes(b'fixture')
    (tmp_path / 'image_mapping.csv').write_text(
        'record_id,img_filename\n001,images/nested/photo.jpg\n001,images/nested/photo.jpg\n'
        '002,\n003,   \n004,images/missing.jpg\n005,product_images/images/nested/photo.jpg\n')
    def reject_excel(*args, **kwargs):
        raise AssertionError('CSV should take precedence over Excel')
    monkeypatch.setattr(data, 'load_workbook', reject_excel)
    mapping, warnings = data.image_mapping(tmp_path, image_dir)
    assert mapping == {'001': [image.resolve()], '005': [image.resolve()]}
    assert len(warnings) == 1
    assert warnings[0]['record_id'] == '004'
    assert sum(len(paths) for paths in mapping.values()) == 2


def test_csv_mapping_rejects_path_escape(tmp_path):
    image_dir = tmp_path / 'product_images'
    image_dir.mkdir()
    (tmp_path / 'image_mapping.csv').write_text('record_id,img_filename\n001,../outside.jpg\n')
    with pytest.raises(ValueError, match='escapes'):
        data.image_mapping(tmp_path, image_dir)


def test_uncertain_secondary_image_keeps_ocr(cfg):
    row = product()
    evidence = extract(row)
    source = evidence.source('ocr', '/synthetic/back.jpg', 'Ingredients: Water.', .99)
    add_visual({'images': [{'image_index': 0, 'match': False, 'match_confidence': .2,
                           'visible_name': '', 'visible_brand': '', 'transcriptions': []}]},
               [Path('/synthetic/back.jpg')], evidence, cfg['verification'], row)
    assert evidence.sources[source].eligible
    assert evidence.sources[source].metadata['identity_status'] == 'uncertain'


@pytest.mark.parametrize('identity', [
    {'visible_brand': 'Different Brand'}, {'visible_quantity': '100ml'},
    {'visible_variant': 'Red', 'target_variant_quote': 'Blue'},
])
def test_explicit_image_contradiction_rejects_ocr(cfg, identity):
    row = product(Specifications='Colour: Blue')
    evidence = extract(row)
    source = evidence.source('ocr', '/synthetic/back.jpg', 'Packaging Type: Tube', .99)
    add_visual({'images': [{'image_index': 0, 'match': False, **identity}]},
               [Path('/synthetic/back.jpg')], evidence, cfg['verification'], row)
    assert not evidence.sources[source].eligible
    assert evidence.sources[source].metadata['identity_status'] == 'mismatch'


def test_no_images_or_descriptions_still_produces_grounded_identity(cfg):
    row = product()
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['short_description'] == 'Hand Balm 50ml.'
    assert result['values']['long_description'] == row['Product Name'] + '.'
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_name_only_fallback_does_not_invent_properties(cfg):
    row = product(T_CATEGORY='', category_group='')
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['short_description'] == 'Hand Balm 50ml.'
    assert result['values']['long_description'] == row['Product Name'] + '.'
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_speculative_source_claim_is_removed_and_flagged(cfg):
    row = product(**{'Product Description': 'This balm will cure dry skin.'})
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert 'cure' not in result['values']['long_description']
    assert any(x['kind'] == 'speculative_source_claim_removed' for x in result['evidence']['issues'])
