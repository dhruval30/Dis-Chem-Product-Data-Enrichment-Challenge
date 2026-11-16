import csv
import json
from pathlib import Path

import pytest
import yaml

from src.common import ROOT, digest, read_json, write_json
from src.content import route_evidence
from src.enrichment import add_visual, finish, image_contradiction
from src.re_enrich import re_enrich
from src.schema import Schema
from src.submission import validate_traces
from src.text_extraction import extract, quantity_relation
from src.verification import verify
from src.visual import VERSION, validate_response, corroborated
from src.vlm import parse_json


@pytest.fixture
def cfg():
    value = yaml.safe_load((ROOT / 'config.yaml').read_text())
    value['_schema'] = Schema(ROOT.parents[1] / 'dataset/submission_format.json')
    return value


def product(name='Example Daily Hand Cream 50ml', **changes):
    return {'record_id': '001', 'Product Name': name, 'T_BRAND': 'Example',
            'category_group': 'PERSONAL CARE', 'Short Description': '', 'Product Description': '',
            'Specifications': '', 'Nutritional Information': '{}', **changes}


def response(printed, observations=None):
    return {'schema_version': VERSION, 'images': [{'image_index': 0, 'identity': 'confirmed',
        'identity_evidence': 'Matching brand and product text', 'printed': printed,
        'observations': observations or []}]}


def ocr(evidence, lines, confidence=.99):
    return evidence.source('ocr', '/synthetic/front.jpg/block/0', '\n'.join(lines), confidence,
        metadata={'lines': [{'text': line, 'confidence': confidence} for line in lines]})


def test_repeated_observation_triples_are_recovered_without_losing_values():
    raw = ('{"schema_version":"visual-evidence-v2","images":[{"image_index":0,'
        '"identity":"unverifiable","identity_evidence":"No printed identity","printed":[],"observations":['
        '{"attribute":"colour","value":"white","evidence":"white socks",'
        '"attribute":"decoration","value":"bows","evidence":"bows on socks"}]}]}')
    with pytest.raises(ValueError, match='Duplicate JSON keys'):
        parse_json(raw)
    events = []
    parsed = parse_json(raw, recover_observations=True, parser_events=events)
    validate_response(parsed)
    assert [o['value'] for o in parsed['images'][0]['observations']] == ['white', 'bows']
    assert events == [{'kind': 'observation_objects_recovered', 'image_index': 0, 'count': 2}]
    assert raw.count('"attribute"') == 2


@pytest.mark.parametrize('raw', [
    '{"facts":[],"facts":[1]}',
    '{"images":[{"printed":[{"kind":"quantity","text":"50ml","text":"100ml"}]}]}',
    '{"images":[{"observations":[{"attribute":"colour","value":"red","evidence":"red","value":"blue"}]}]}',
    '{"value":NaN}',
])
def test_ambiguous_or_nonfinite_json_is_never_silently_coerced(raw):
    with pytest.raises(ValueError):
        parse_json(raw, recover_observations=True)


def test_ocr_agreement_can_join_spaces_but_cannot_change_a_number():
    evidence = extract(product())
    identifier = evidence.source('ocr', '/synthetic/front.jpg', '100mLnet', .92)
    assert corroborated('100 ml net', [evidence.sources[identifier]], .9) == identifier
    identifier = evidence.source('ocr', '/synthetic/front.jpg', '2100ml', .99)
    assert corroborated('100ml', [evidence.sources[identifier]]) is None


@pytest.mark.parametrize('printed,target,relation', [
    ('30 CAPSULES', 'Example 1000mg Vitamin C Capsules 30 Pack', 'same'),
    ('1000 mg', 'Example 1000mg Vitamin C Capsules 30 Pack', 'strength'),
    ('500 mg', 'Example 1000mg Vitamin C Capsules 30 Pack', 'different'),
    ('60 CAPSULES', 'Example 1000mg Vitamin C Capsules 30 Pack', 'different'),
    ('30 TABLETS', 'Example 1000mg Vitamin C Capsules 30 Pack', 'different'),
    ('10ml', 'Example Fragrance Set 90ml + 10ml', 'component'),
    ('100ml', 'Example Fragrance Set 90ml + 10ml', 'different'),
    ('90ml + 10ml', 'Example Fragrance Set 90ml + 10ml', 'same'),
    ('25ml', 'Example Cream 2 x 25ml', 'component'),
    ('2ml', 'Example Cream 2 x 25ml', 'different'),
    ('30.0 capsules', 'Example Capsules 30 Pack', 'same'),
])
def test_quantity_comparisons_respect_scope(printed, target, relation):
    assert quantity_relation(printed, target) == relation


def test_count_wording_does_not_invalidate_a_matching_supplement_image(cfg):
    row = product('Example 1000mg Vitamin C Capsules 30 Pack', category_group='VITAMINS / HERBAL')
    assert image_contradiction({'visible_quantity': '30 CAPSULES'}, row) is None
    evidence = extract(row)
    source = ocr(evidence, ['EXAMPLE', 'Vitamin C Capsules', '1000 mg Capsules', '30 CAPSULES'])
    route_evidence(row, evidence, cfg)
    add_visual(response([{'kind': 'identity', 'text': 'Example Vitamin C Capsules'},
                         {'kind': 'quantity', 'text': '30 CAPSULES'}]),
               [Path('/synthetic/front.jpg')], evidence, cfg['verification'], row)
    result = finish(row, evidence, [], [], None, cfg, False)
    assert evidence.sources[source].eligible
    assert not any(i['kind'] == 'image_product_mismatch' for i in evidence.issues)
    assert result['values']['product_specifications']['pack_size'] == {'value': '30', 'unit': 'Pack'}


def test_components_and_unverified_total_cannot_erase_the_declared_set(cfg):
    row = product('Example Gold Fragrance Set 90ml + 10ml')
    evidence = extract(row)
    add_visual(response([{'kind': 'identity', 'text': 'Example Gold Fragrance Set'},
                         {'kind': 'quantity', 'text': '100ml'}, {'kind': 'quantity', 'text': '10ml'}]),
               [Path('/synthetic/front.jpg')], evidence, cfg['verification'], row)
    values, _, rejected = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['product_specifications']['pack_size'] == {'value': '90 + 10', 'unit': 'ml'}
    assert not any(r['reason'] == 'conflicting_values' for r in rejected)
    component = next(s for s in evidence.sources.values() if s.text == '10ml')
    evidence.add('product_specifications', ['pack_size'], {'value': '10', 'unit': 'ml'}, component.id)
    _, _, rejected = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert any(r['reason'] == 'quantity_is_not_whole_pack' for r in rejected)


def test_quantity_in_specification_is_used_with_ocr_agreement(cfg):
    row = product('Example Professional Mess Up Paste', category_group='HAIR SALON PROFESSIONAL')
    evidence = extract(row)
    ocr(evidence, ['EXAMPLE', 'Mess Up Paste'])
    evidence.source('ocr', '/synthetic/front.jpg', '100mLnet', .92)
    route_evidence(row, evidence, cfg)
    add_visual(response([{'kind': 'identity', 'text': 'Example Mess Up Paste'},
                         {'kind': 'specification', 'text': '100 mL net'}]),
               [Path('/synthetic/front.jpg')], evidence, cfg['verification'], row)
    result = finish(row, evidence, [], [], None, cfg, False)
    pair = result['values']['product_specifications']['pack_size']
    assert pair['value'] == '100' and pair['unit'].lower() == 'ml'
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_slogans_and_identity_fragments_are_context_not_specifications(cfg):
    row = product('Example Fragrance Candle Moroccan Spice', category_group='HOME SCENTS')
    evidence = extract(row)
    add_visual(response([{'kind': 'identity', 'text': 'Example Fragrance Candle'},
                         {'kind': 'claim', 'text': 'EXPERIENCE AFRICA'},
                         {'kind': 'claim', 'text': '-Fragranced Candle-'},
                         {'kind': 'claim', 'text': 'MOROCCAN SPICE'}]),
               [Path('/synthetic/front.jpg')], evidence, cfg['verification'], row)
    result = finish(row, evidence, [], [], None, cfg, False)
    assert result['values']['product_specifications']['other'] == []
    assert 'EXPERIENCE AFRICA' not in result['values']['long_description']
    assert 'Fragranced Candle' not in result['values']['long_description']
    assert result['values']['variant_attributes']['fragrance'] == ['Moroccan Spice']


def test_actual_crayon_colours_survive_alongside_a_printed_claim(cfg):
    row = product('Example Bath Crayons 6 Pieces', category_group='BABY TOYS')
    evidence = extract(row)
    ocr(evidence, ['EXAMPLE', 'Bath Crayons', 'Stimulating creativity during bath time!'])
    route_evidence(row, evidence, cfg)
    add_visual(response([{'kind': 'identity', 'text': 'Example Bath Crayons'},
                         {'kind': 'claim', 'text': 'Stimulating creativity during bath time!'}],
        [{'attribute': 'colour', 'value': 'Yellow, orange, red, purple, blue, green',
          'evidence': 'Crayons visible through transparent pack'}]),
        [Path('/synthetic/front.jpg')], evidence, cfg['verification'], row)
    result = finish(row, evidence, [], [], None, cfg, False)
    assert 'yellow, orange, red, purple, blue, green' in result['values']['long_description']
    assert '!.' not in result['values']['long_description']
    assert result['values']['product_specifications']['other'] == []
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_pearl_material_and_changed_safety_claim_remain_unverified(cfg):
    row = product('Example Lace Socks With Bows', category_group='BABY CLOTHING')
    evidence = extract(row)
    add_visual(response([{'kind': 'identity', 'text': 'Example Lace Socks'}],
        [{'attribute': 'decoration', 'value': 'bows with pearls', 'evidence': 'pearls on bows'},
         {'attribute': 'colour', 'value': 'white', 'evidence': 'white socks'}]),
        [Path('/synthetic/front.jpg')], evidence, cfg['verification'], row)
    result = finish(row, evidence, [], [], None, cfg, False)
    assert 'pearls' not in result['values']['long_description']
    assert 'The socks are white.' in result['values']['long_description']
    row = product('Example Bath Crayons', category_group='BABY TOYS')
    evidence = extract(row)
    ocr(evidence, ['EXAMPLE', 'Bath Crayons', 'Safe for use on skin'])
    route_evidence(row, evidence, cfg)
    add_visual(response([{'kind': 'identity', 'text': 'Example Bath Crayons'},
                         {'kind': 'claim', 'text': 'Safe for use on kids'}]),
               [Path('/synthetic/front.jpg')], evidence, cfg['verification'], row)
    result = finish(row, evidence, [], [], None, cfg, False)
    assert 'safe' not in result['values']['long_description'].lower()


def test_replay_reparses_raw_and_uses_previsual_ocr_without_inference(tmp_path, cfg):
    cfg = {k: v for k, v in cfg.items() if k != '_schema'}
    cfg.update(data_dir=str(tmp_path / 'data'), output_dir=str(tmp_path / 'outputs'), report_dir=str(tmp_path / 'reports'))
    data = Path(cfg['data_dir'])
    data.mkdir()
    (data / 'submission_format.json').write_bytes((ROOT.parents[1] / 'dataset/submission_format.json').read_bytes())
    row = product('Example 1000mg Vitamin C Capsules 30 Pack', category_group='VITAMINS / HERBAL')
    with (data / 'Test_Masked.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    source = Path(cfg['output_dir']) / 'runs/test/saved'
    evidence = extract(row)
    ocr_id = ocr(evidence, ['EXAMPLE', 'Vitamin C Capsules', '30 CAPSULES'])
    filename = digest('001') + '.json'
    write_json(source / 'prepared' / filename, {'evidence': evidence.payload(), 'paths': ['/synthetic/front.jpg']})
    evidence.sources[ocr_id].eligible = False
    evidence.sources[ocr_id].metadata['identity_status'] = 'mismatch'
    write_json(source / 'records' / filename, {'record_id': '001', 'degraded': True,
        'stage_status': {'image_extraction': 'failed'}, 'evidence': evidence.payload()})
    write_json(source / 'manifest.json', {'split': 'test', 'expected_ids': ['001'],
        'metadata': {'inputs': {}, 'config': cfg, 'images': {}, 'source': {}}})
    write_json(source / 'budget.json', {'calls': 1})
    call_key = 'a' * 64
    raw_path = Path(cfg['output_dir']) / 'cache/vlm_raw' / (call_key + '.json')
    raw = json.dumps(response([{'kind': 'identity', 'text': 'Example Vitamin C Capsules'},
                               {'kind': 'quantity', 'text': '30 CAPSULES'}]))
    raw = raw.replace('"observations": []', '"observations": [{"attribute":"colour","value":"white",'
                      '"evidence":"white bottle","evidence":"orange label"}]')
    write_json(raw_path, {'text': raw})
    (source / 'diagnostics').mkdir()
    (source / 'diagnostics/vlm.jsonl').write_text(json.dumps({'record_id': '001', 'stage': 'image_extraction',
        'event': 'parsing_start', 'call_key': call_key}) + '\n')
    old_record = (source / 'records' / filename).read_bytes()
    old_raw = raw_path.read_bytes()
    destination = re_enrich(source, cfg)
    result = read_json(destination / 'records' / filename)
    assert result['visual_replay']['status'] == 'replayed'
    assert result['model_generation_calls'] == 0 and not result['degraded']
    assert result['recovered_stages'] == ['image_extraction']
    assert result['visual_replay']['parser_events'][0]['kind'] == 'ambiguous_observation_discarded'
    assert result['values']['product_specifications']['pack_size'] == {'value': '30', 'unit': 'Pack'}
    assert next(s for s in result['evidence']['sources'] if s['id'] == ocr_id)['eligible']
    assert (source / 'records' / filename).read_bytes() == old_record
    assert raw_path.read_bytes() == old_raw
