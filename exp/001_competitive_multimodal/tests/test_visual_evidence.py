import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.common import ROOT, Cache, digest, write_json
from src.enrichment import add_visual, finish
from src.evidence import Evidence
from src.inspection import export_inspection
from src.schema import Schema
from src.text_extraction import extract
from src.verification import verify, verify_sentence
from src.visual import VERSION, response_schema
from src.vlm import VLM, parse_json


@pytest.fixture
def cfg():
    value = yaml.safe_load((ROOT / 'config.yaml').read_text())
    value['_schema'] = Schema(ROOT.parents[1] / 'dataset/submission_format.json')
    value['policy']['vlm_selection'] = 'all'
    return value


def row(name='Example Baby Socks', **changes):
    return {'record_id': '001', 'Product Name': name, 'T_BRAND': 'Example',
            'category_group': 'BABY CLOTHING', 'Specifications': '',
            'Short Description': '', 'Product Description': '', 'Nutritional Information': '{}', **changes}


def response(state='unverifiable', printed=None, observations=None, **fields):
    return {'schema_version': VERSION, 'images': [{'image_index': 0, 'identity': state,
        'identity_evidence': 'no printed identification' if state == 'unverifiable' else 'readable brand and product name',
        'printed': printed or [], 'observations': observations or [], **fields}]}


def observation(attribute='decoration', value='decorative bows', evidence='bows attached to the socks'):
    return {'attribute': attribute, 'value': value, 'evidence': evidence}


def import_image(cfg, product, output):
    evidence = extract(product)
    add_visual(output, [Path('/synthetic/image.jpg')], evidence, cfg['verification'], product)
    return evidence


def test_textless_image_keeps_observations_without_claiming_exact_identity(cfg):
    product = row()
    evidence = import_image(cfg, product, response(observations=[observation()]))
    source = next(s for s in evidence.sources.values() if s.kind == 'visual_observation')
    assert source.metadata['identity_status'] == 'unverifiable'
    assert source.exact_product is False
    assert source.confidence is None
    assert source.metadata['reliability'] == 'mapped_direct_observation'
    values, accepted, _ = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['product_specifications']['other'] == []
    assert not any(f.source_id == source.id for f in accepted)
    assert verify_sentence('Visible details include decorative bows.', [source.id], evidence, cfg['verification']) is None


def test_observations_reach_short_and_long_copy_without_extra_model_call(cfg):
    output = response(observations=[observation(), observation('colour', 'white socks', 'white sock fabric is visible')])
    class MockModel:
        def ask(self, stage, *args):
            return output if stage == 'image_extraction' else {'facts': []} if stage == 'field_enrichment' else {}
    product = row()
    result = finish(product, extract(product), [Path('/synthetic/image.jpg')], [], MockModel(), cfg, True)
    assert 'decorative bows' in result['values']['long_description']
    assert result['values']['short_description']
    assert cfg['_schema'].errors(result['values']) == []


def test_appearance_rendering_accepts_name_and_observation_without_loosening_claims(cfg):
    product = row()
    evidence = import_image(cfg, product, response(observations=[observation()]))
    name = next(s for s in evidence.sources.values() if s.location == 'Product Name')
    visual = next(s for s in evidence.sources.values() if s.kind == 'visual_observation')
    sources = [name.id, visual.id]
    assert verify_sentence('Design details: decorative bows.', sources, evidence, cfg['verification']) is None
    assert verify_sentence('Design details: waterproof decorative bows.', sources, evidence, cfg['verification'])
    visual.eligible = False
    assert verify_sentence('Design details: decorative bows.', sources, evidence, cfg['verification'])


def test_visible_count_cannot_become_pack_size(cfg):
    product = row()
    evidence = import_image(cfg, product, response(observations=[observation('visible_count', '3 individual socks', 'three separate socks are shown')]))
    source = next(s for s in evidence.sources.values() if s.kind == 'visual_observation')
    evidence.add('product_specifications', ['pack_size'], {'value': '3', 'unit': 'socks'}, source.id, source.text)
    values, _, rejected = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['product_specifications']['pack_size'] == {'value': '', 'unit': ''}
    assert any(x['reason'] == 'observation_cannot_establish_commercial_attribute' for x in rejected)
    assert verify_sentence('The photograph shows 3 individual socks.', [source.id], evidence, cfg['verification']) is None


def test_appearance_colour_is_not_an_official_shade(cfg):
    product = row()
    evidence = import_image(cfg, product, response(observations=[observation('colour', 'white', 'white socks are shown')]))
    source = next(s for s in evidence.sources.values() if s.kind == 'visual_observation')
    evidence.add('variant_attributes', ['shade'], ['white'], source.id, source.text)
    assert verify(evidence, cfg['_schema'], cfg['verification'], set())[0]['variant_attributes']['shade'] == []


def test_confirmed_packaging_recovers_printed_quantity_and_claims(cfg):
    product = row('Example Intensive Hydrating Cream 50ml', category_group='BEAUTY')
    output = response('confirmed', printed=[{'kind': 'identity', 'text': 'Example Intensive Hydrating Cream'},
        {'kind': 'quantity', 'text': '50 ml'}, {'kind': 'claim', 'text': 'All Skin Types'},
        {'kind': 'highlighted_ingredient', 'text': 'Crosspolymer Hyaluronic Acid'}],
        visible_brand='Example', visible_name='Intensive Hydrating Cream', visible_quantity='50 ml')
    evidence = import_image(cfg, product, output)
    values, _, _ = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['product_specifications']['pack_size']['value'] == '50'
    assert values['ingredient_information'] == []
    assert values['product_specifications']['other'] == []
    source = next(s for s in evidence.sources.values() if s.metadata.get('printed_kind') == 'claim')
    assert verify_sentence('The label states: All Skin Types.', [source.id], evidence, cfg['verification']) is None


def test_partial_identity_is_unverifiable_not_mismatch(cfg):
    product = row()
    evidence = import_image(cfg, product, response('confirmed', visible_name='Socks', observations=[observation()]))
    source = next(s for s in evidence.sources.values() if s.kind == 'visual_observation')
    assert source.metadata['identity_status'] == 'unverifiable'
    assert source.eligible


def test_missing_branding_does_not_establish_a_mismatch(cfg):
    product = row()
    output = response('mismatch', observations=[observation()])
    output['images'][0]['identity_evidence'] = 'no readable printed identification'
    evidence = import_image(cfg, product, output)
    source = next(s for s in evidence.sources.values() if s.kind == 'visual_observation')
    assert source.metadata['identity_status'] == 'unverifiable'
    assert source.eligible


def test_explicit_mismatch_quarantines_ocr_and_observations(cfg):
    product = row('Example Cream 50ml')
    evidence = extract(product)
    ocr_id = evidence.source('ocr', '/synthetic/image.jpg', '100 ml', .99)
    output = response('confirmed', printed=[{'kind': 'quantity', 'text': '100 ml'}], observations=[observation()])
    add_visual(output, [Path('/synthetic/image.jpg')], evidence, cfg['verification'], product)
    assert not evidence.sources[ocr_id].eligible
    assert not any(s.kind == 'visual_observation' for s in evidence.sources.values())


def test_highlighted_ingredient_cannot_become_full_declaration(cfg):
    product = row('Example Hydrating Cream', category_group='BEAUTY')
    output = response('confirmed', visible_brand='Example', visible_name='Example Hydrating Cream',
                      printed=[{'kind': 'highlighted_ingredient', 'text': 'Hyaluronic acid'}])
    evidence = import_image(cfg, product, output)
    source = next(s for s in evidence.sources.values() if s.kind == 'visual_printed')
    evidence.add('ingredient_information', [], ['Hyaluronic acid'], source.id, source.text)
    assert verify(evidence, cfg['_schema'], cfg['verification'], set())[0]['ingredient_information'] == []


def test_partial_ingredient_panel_is_not_a_complete_declaration(cfg):
    product = row('Example Cream', category_group='BEAUTY')
    evidence = extract(product)
    evidence.source('ocr', '/synthetic/image.jpg/block/0',
        'Ingredients: Water, Glycerin. May contain milk.', .99,
        metadata={'lines': [{'text': 'Ingredients: Water, Glycerin. May contain milk.', 'confidence': .99}]})
    add_visual(response(printed=[{'kind': 'ingredients_declaration', 'text': 'Ingredients: Water'}]),
        [Path('/synthetic/image.jpg')], evidence, cfg['verification'], product)
    assert not any(s.eligible for s in evidence.sources.values() if s.kind == 'visual_printed')


def test_recovery_and_safe_profile_still_bound_image_work():
    from src.common import config
    safe = config(str(ROOT / 'profiles/m2_safe.yaml'), [])
    assert safe['model']['max_images'] == 1
    assert safe['model']['image_max_side'] == 512
    assert safe['model']['stage_max_tokens']['image_extraction'] == 256
    assert safe['model']['visual_schema_version'] == VERSION
    assert not safe['policy']['image_verification']


def test_ocr_and_printed_storage_merge_without_duplicate_values(cfg):
    product = row('Example Balm')
    evidence = extract(product)
    ocr_id = evidence.source('ocr', '/synthetic/image.jpg', 'Storage: Keep dry.', .99)
    evidence.add('storage_instructions', [], ['Storage: Keep dry.'], ocr_id)
    output = response(printed=[{'kind': 'storage', 'text': 'Storage: Keep dry.'}])
    add_visual(output, [Path('/synthetic/image.jpg')], evidence, cfg['verification'], product)
    assert verify(evidence, cfg['_schema'], cfg['verification'], set())[0]['storage_instructions'] == ['Storage: Keep dry.']


def test_unsupported_visual_material_is_not_emitted(cfg):
    product = row()
    evidence = import_image(cfg, product, response(observations=[observation('design', '100% cotton', 'looks like cotton')]))
    assert not any(s.eligible for s in evidence.sources.values() if s.kind == 'visual_observation')


def test_non_applicable_output_wins_over_visual_fact(cfg):
    product = row(do_not_predict_fields='["product_specifications"]')
    evidence = import_image(cfg, product, response(observations=[observation()]))
    assert verify(evidence, cfg['_schema'], cfg['verification'], {'product_specifications'})[0]['product_specifications'] == cfg['_schema'].empty['product_specifications']


def test_visual_schema_and_json_wrapper_parser():
    from jsonschema import Draft202012Validator
    value = response(observations=[observation()])
    Draft202012Validator(response_schema()).validate(value)
    import json
    assert parse_json('Result: ' + json.dumps(value)) == value
    bad = copy.deepcopy(value)
    bad['images'][0]['identity'] = 'maybe'
    assert list(Draft202012Validator(response_schema()).iter_errors(bad))


def test_real_vlm_cache_identity_changes_with_prompt(tmp_path, cfg):
    class SpyCache(Cache):
        def __init__(self, path):
            super().__init__(path)
            self.keys = []
        def get(self, stage, key):
            self.keys.append(key)
            return None
    cache = SpyCache(tmp_path / 'cache')
    model = VLM(cfg['model'], cache, SimpleNamespace(), {}, 42, tmp_path / 'diagnostics')
    def stop_before_model_loading():
        raise RuntimeError('test stop before model load')
    model.load = stop_before_model_loading
    for text in ('visual schema v1', 'visual schema v2'):
        with pytest.raises(RuntimeError, match='test stop'):
            model._ask('image_extraction', text)
    assert len(cache.keys) == 2 and cache.keys[0] != cache.keys[1]


def test_inspection_export_does_not_need_inference(tmp_path, cfg):
    cfg['output_dir'] = str(tmp_path)
    run = tmp_path / 'runs/pseudo/example'
    product = row()
    result = finish(product, extract(product), [], [], None, cfg, False)
    result['record_id'] = '001'
    key = digest('001')
    write_json(run / 'manifest.json', {'split': 'pseudo', 'metadata': {'images': {}}})
    write_json(run / 'pseudo_inputs.json', {'001': product})
    write_json(run / 'records' / (key + '.json'), result)
    write_json(run / 'prepared' / (key + '.json'), {'paths': [], 'evidence': result['evidence']})
    destination = export_inspection(run, '001', cfg)
    assert (destination / 'enriched_fields.json').exists()
    assert (destination / 'images.json').exists()
    assert (destination / 'rejected.json').exists()
