import json
from pathlib import Path

import pytest
import yaml

from src.common import ROOT, digest, read_json, write_json
from src.enrichment import finish
from src.re_enrich import re_enrich
from src.schema import Schema
from src.submission import validate_traces
from src.text_extraction import extract
from src.verification import verify_sentence
from src.visual import VERSION


@pytest.fixture
def cfg():
    value = yaml.safe_load((ROOT / 'config.yaml').read_text())
    value['_schema'] = Schema(ROOT.parents[1] / 'dataset/submission_format.json')
    return value


def product(name, brand='Example', **changes):
    return {'record_id': '001', 'Product Name': name, 'T_BRAND': brand,
            'category_group': 'PERSONAL CARE', 'Short Description': '', 'Product Description': '',
            'Specifications': '', 'Nutritional Information': '{}', **changes}


def ocr(evidence, lines, confidence=.98, block=0):
    rows = [{'text': text, 'confidence': score} for text, score in
            ((line if isinstance(line, tuple) else (line, confidence)) for line in lines)]
    return evidence.source('ocr', f'/synthetic/pack.jpg/block/{block}', '\n'.join(x['text'] for x in rows),
        confidence, metadata={'lines': rows, 'identity_status': 'unverifiable'})


def artwork(evidence, text, support):
    return evidence.source('visual_observation', '/synthetic/pack.jpg', text, None, exact=False,
        metadata={'visual_schema': VERSION, 'mapped_image': True, 'identity_status': 'unverifiable',
                  'attribute': 'decoration', 'visible_support': support, 'observation_value': text})


def test_porridge_pack_claims_beat_packaging_art(cfg):
    row = product('Example From 6 Months Original Baby Porridge 350g', category_group='BABY FOOD AND FEEDING')
    evidence = extract(row)
    ocr(evidence, ['EXAMPLE', 'Baby Porridge', 'FROM', '6 Original', 'MONTHS',
                   ('VNO PRESERVATIVES', .916), ('VNO COLOURANTS', .946), ('VNO ARTIFICIAL FLAVOURINGS', .954)])
    artwork(evidence, 'blue oval logo', 'logo at the top of the package')
    artwork(evidence, 'grain image', 'grain illustration on the bag')
    result = finish(row, evidence, [], [], None, cfg, False)
    text = result['values']['short_description'] + ' ' + result['values']['long_description']
    assert 'preservatives' in text.lower()
    assert 'colourants' in text.lower()
    assert 'artificial flavourings' in text.lower()
    assert 'logo' not in text and 'grain image' not in text
    assert not any(x['attribute'].startswith('visible_') for x in result['values']['product_specifications']['other'])
    assert 'Original' in result['values']['variant_attributes']['other']
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_wrapped_nipple_cream_claims_are_preserved_without_strengthening(cfg):
    row = product('Example HPA Lanolin Nipple Cream 40ml')
    evidence = extract(row)
    identity = ocr(evidence, ['Example', ('HPA Lanolin', .918), ('Nipple Cream', .950)])
    ocr(evidence, [('Helps soothe &protect sore,cracked nipples', .960),
        ('and dry skin No need to remove', .955), ('before breastfeeding', .979),
        ('hypoallergenic 100%natural&', .978), ('40mle', .984)], block=1)
    artwork(evidence, 'hand holding baby', 'image on box front')
    result = finish(row, evidence, [], [], None, cfg, False)
    text = result['values']['short_description'] + ' ' + result['values']['long_description']
    assert 'Helps soothe and protect sore, cracked nipples and dry skin' in text
    assert 'No need to remove before breastfeeding' in text
    assert 'hypoallergenic' in text and '100% natural' in text
    assert 'holding baby' not in text and 'tube on box' not in text
    assert evidence.sources[identity].metadata['identity_status'] == 'confirmed'
    assert evidence.sources[identity].exact_product
    assert result['values']['ingredient_information'] == []
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_unidentified_ocr_cannot_supply_health_claims(cfg):
    row = product('Example Lanolin Cream')
    evidence = extract(row)
    identifier = ocr(evidence, ['Helps soothe cracked skin'])
    result = finish(row, evidence, [], [], None, cfg, False)
    assert 'soothe' not in result['values']['long_description']
    assert not evidence.sources[identifier].exact_product


def test_materials_and_pack_quantities_are_not_inferred_from_socks(cfg):
    row = product("Mother's Choice Lace Socks With Bows Size 6-12 months Assorted", brand='Mothers Choice',
                  category_group='BABY CLOTHING')
    evidence = extract(row)
    artwork(evidence, 'three bows in red, yellow and pink', 'each sock has a bow with beads')
    result = finish(row, evidence, [], [], None, cfg, False)
    assert result['values']['short_description'] == 'Lace Socks With Bows Size 6-12 months Assorted.'
    assert 'with three bows' not in result['values']['short_description']
    assert result['values']['product_specifications']['other'] == []
    assert result['values']['product_specifications']['pack_size'] == {'value': '', 'unit': ''}
    assert 'bows in red, yellow and pink' in result['values']['long_description']
    assert 'beads' not in result['values']['long_description']
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_grounded_claim_verifier_rejects_strengthened_wording(cfg):
    row = product('Example Lanolin Nipple Cream')
    evidence = extract(row)
    ocr(evidence, ['Example', 'Lanolin Nipple Cream', 'May help soothe skin'])
    finish(row, evidence, [], [], None, cfg, False)
    source = next(s for s in evidence.sources.values() if s.kind == 'grounded_text' and s.metadata['role'] == 'benefit')
    assert verify_sentence('Will soothe skin.', [source.id], evidence, cfg['verification'])


def test_do_not_predict_applies_after_content_routing(cfg):
    row = product('Example Original Baby Porridge', do_not_predict_fields='["short_description","variant_attributes"]')
    evidence = extract(row)
    ocr(evidence, ['Example', 'Original Baby Porridge', 'Original', 'NO PRESERVATIVES'])
    result = finish(row, evidence, [], [], None, cfg, False)
    assert result['values']['short_description'] == ''
    assert result['values']['variant_attributes'] == cfg['_schema'].empty['variant_attributes']


def test_re_enrich_requires_no_model_and_does_not_change_source_run(tmp_path, cfg):
    cfg = {k: v for k, v in cfg.items() if k != '_schema'}
    cfg.update({'data_dir': str(tmp_path / 'data'), 'output_dir': str(tmp_path / 'outputs'),
                'report_dir': str(tmp_path / 'reports')})
    data = Path(cfg['data_dir'])
    data.mkdir()
    schema_path = ROOT.parents[1] / 'dataset/submission_format.json'
    (data / 'submission_format.json').write_bytes(schema_path.read_bytes())
    row = product('Example Baby Porridge 350g')
    import csv
    with (data / 'Test_Masked.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    source = Path(cfg['output_dir']) / 'runs/test/saved'
    evidence = extract(row)
    ocr(evidence, ['Example', 'Baby Porridge', 'NO PRESERVATIVES'])
    name_source = next(s for s in evidence.sources.values() if s.location == 'Product Name')
    evidence.add('product_specifications', ['pack_size'], {'value': '1000', 'unit': 'mg'}, name_source.id, '1000mg')
    record = {'record_id': '001', 'evidence': evidence.payload(), 'retrieval': [], 'degraded': False}
    path = source / 'records' / (digest('001') + '.json')
    write_json(path, record)
    original_bytes = path.read_bytes()
    write_json(source / 'manifest.json', {'split': 'test', 'expected_ids': ['001'],
        'metadata': {'inputs': {}, 'source': {}, 'images': {}, 'config': cfg}})
    write_json(source / 'budget.json', {'calls': 1, 'ocr_images': 1})
    result_dir = re_enrich(source, cfg, ['001'])
    result = read_json(result_dir / 'records' / path.name)
    assert result['re_enriched_without_inference']
    assert result['model_generation_calls'] == 0
    assert 'preservatives' in result['values']['long_description'].lower()
    assert result['values']['product_specifications']['pack_size'] == {'value': '350', 'unit': 'g'}
    assert path.read_bytes() == original_bytes
