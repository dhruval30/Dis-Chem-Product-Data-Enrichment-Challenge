import pytest
import yaml

from src.common import ROOT
from src.enrichment import compact_prompt, finish
from src.schema import Schema
from src.submission import validate_traces
from src.text_extraction import extract, pack_quantity
from src.verification import verify
from src.visual import VERSION, validate_response


@pytest.fixture
def cfg():
    value = yaml.safe_load((ROOT / 'config.yaml').read_text())
    value['_schema'] = Schema(ROOT.parents[1] / 'dataset/submission_format.json')
    return value


def product(name, **changes):
    return {'record_id': '001', 'Product Name': name, 'T_BRAND': 'Example',
            'category_group': 'PERSONAL CARE', 'Short Description': '', 'Product Description': '',
            'Specifications': '', 'Nutritional Information': '{}', **changes}


def ocr(evidence, lines, confidence=.99):
    rows = [{'text': line, 'confidence': confidence} for line in lines]
    return evidence.source('ocr', '/synthetic/photo.jpg/block/0', '\n'.join(lines), confidence,
                           metadata={'lines': rows, 'identity_status': 'unverifiable'})


@pytest.mark.parametrize('name,expected', [
    ('Example 1000mg Vitamin C Capsules 30 Pack', {'value': '30', 'unit': 'Pack'}),
    ('Example 500mg Tablets 60 Tablets', {'value': '60', 'unit': 'Tablets'}),
    ('Example Vitamin C Capsules 1000mg', None),
    ('Example Eau De Parfum 90ml + 10ml', {'value': '90 + 10', 'unit': 'ml'}),
    ('Example Cream 2 x 25 ml', {'value': '2 x 25', 'unit': 'ml'}),
    ('Example Set 50ml + 100g', None),
])
def test_pack_quantities_preserve_contents_and_exclude_strength(name, expected):
    parsed = pack_quantity(name)
    assert (parsed[0] if parsed else None) == expected


def test_old_or_model_strength_fact_is_rejected_without_losing_pack_count(cfg):
    row = product('Example 1000mg Vitamin C Capsules 30 Pack', category_group='VITAMINS / HERBAL')
    evidence = extract(row)
    source = next(s for s in evidence.sources.values() if s.location == 'Product Name')
    evidence.add('product_specifications', ['pack_size'], {'value': '1000', 'unit': 'mg'}, source.id, '1000mg')
    values, _, rejected = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['product_specifications']['pack_size'] == {'value': '30', 'unit': 'Pack'}
    assert any(r['reason'] == 'strength_or_incomplete_pack_quantity' for r in rejected)


def test_model_cannot_truncate_a_multi_component_set(cfg):
    row = product('Example Fragrance Set 90ml + 10ml')
    evidence = extract(row)
    source = next(s for s in evidence.sources.values() if s.location == 'Product Name')
    evidence.add('product_specifications', ['pack_size'], {'value': '90', 'unit': 'ml'}, source.id, '90ml')
    values, _, _ = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['product_specifications']['pack_size'] == {'value': '90 + 10', 'unit': 'ml'}


def test_specifications_and_ocr_shade_reach_output_without_generation(cfg):
    row = product('Example Longwear Espresso Foundation 30ml', category_group='COSMETICS',
                  Specifications='Skin Type: Combination, Oily\nMake-Up Finish: Matte\nWaterproof: Yes')
    evidence = extract(row)
    ocr(evidence, ['EXAMPLE', 'Longwear Foundation', '021 ESPRESSO', 'SPF15/FPS15', '30 ml'])
    result = finish(row, evidence, [], [], None, cfg, False)
    values = result['values']
    text = values['short_description'] + ' ' + values['long_description']
    assert 'combination and oily skin' in text
    assert 'Matte finish.' in text
    assert values['variant_attributes']['shade'] == ['ESPRESSO']
    assert {'attribute': 'SPF', 'value': '15'} in values['product_specifications']['other']
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_ocr_brand_typo_and_joined_name_do_not_erase_reliable_facts(cfg):
    row = product('Nature Complex Vitamin C Capsules 30 Pack', T_BRAND='Nature Complex', category_group='VITAMINS / HERBAL')
    evidence = extract(row)
    identifier = ocr(evidence, ['Nafure', 'COMPLEXO', 'Vitamin C', 'Capsules', '30 CAPSULES'])
    result = finish(row, evidence, [], [], None, cfg, False)
    assert evidence.sources[identifier].metadata['identity_status'] == 'confirmed'
    assert result['values']['product_specifications']['pack_size']['value'] == '30'
    assert not result['degraded']


def test_hair_label_features_and_quantity_beat_container_description(cfg):
    row = product('Example Professional Style+ Mess Up', category_group='HAIR SALON PROFESSIONAL')
    evidence = extract(row)
    ocr(evidence, ['EXAMPLE', 'Style+ MESSUP', 'Matte paste', 'Medium control', '100mLnet'])
    result = finish(row, evidence, [], [], None, cfg, False)
    text = result['values']['short_description'] + ' ' + result['values']['long_description']
    assert 'matte paste' in text.lower() and 'medium control' in text.lower()
    assert result['values']['product_specifications']['pack_size'] == {'value': '100', 'unit': 'mL'}
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_quantity_contradiction_quarantines_matching_brand_ocr(cfg):
    row = product('Example Daily Hand Cream 50ml')
    evidence = extract(row)
    identifier = ocr(evidence, ['EXAMPLE', 'Daily Hand Cream', '100 ml', 'Helps soothe skin'])
    result = finish(row, evidence, [], [], None, cfg, False)
    assert not evidence.sources[identifier].eligible
    assert evidence.sources[identifier].metadata['identity_status'] == 'mismatch'
    assert 'soothe' not in result['values']['long_description']


def test_nutrition_row_quantity_is_not_a_product_identity_mismatch(cfg):
    row = product('Example Daily Cereal 350g', category_group='NUTRITION CATEGORY')
    evidence = extract(row)
    identifier = ocr(evidence, ['EXAMPLE', 'Daily Cereal', 'Protein 10 g'])
    finish(row, evidence, [], [], None, cfg, False)
    assert evidence.sources[identifier].metadata['identity_status'] == 'confirmed'
    assert not any(f.method == 'ocr_pack' for f in evidence.facts)


def test_explicit_size_fragrance_and_applicability(cfg):
    row = product('Example Fragrance Candle Moroccan Spice', category_group='HOME SCENTS')
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['variant_attributes']['fragrance'] == ['Moroccan Spice']
    row = product('Example Elasticised Elbow Support Large', do_not_predict_fields='["variant_attributes"]')
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['variant_attributes'] == cfg['_schema'].empty['variant_attributes']
    assert result['values']['product_specifications']['application_area'] == ['Elbow']


def test_marketing_is_removed_but_explicit_fragrance_notes_survive(cfg):
    row = product('Example Candle Spice', category_group='HOME SCENTS', **{
        'Short Description': 'Fill your home with exotic warmth. This candle evokes bustling souks with its blend of cinnamon, ginger and cardamom.',
        'Product Description': 'Let the fragrance transport you to the bustling souks, with scents of cinnamon, ginger and cardamom.'})
    result = finish(row, extract(row), [], [], None, cfg, False)
    text = result['values']['short_description'] + ' ' + result['values']['long_description']
    assert 'souks' not in text and 'transport' not in text
    assert text.count('cinnamon, ginger and cardamom') == 1
    assert result['values']['ingredient_information'] == []
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_short_and_long_copy_do_not_repeat_the_same_source_claim(cfg):
    row = product('Example Vitamin C Capsules', category_group='VITAMINS / HERBAL', **{
        'Short Description': 'These capsules support immune function. Supports immune function.',
        'Product Description': 'These capsules support immune function. Take one capsule daily.'})
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['short_description'].lower().count('support') == 1
    assert 'support immune function' not in result['values']['long_description']


def test_prompt_contains_no_filled_example_and_prioritises_printed_evidence():
    text = compact_prompt('image_extraction', {'catalogue': product('Example Cream 50ml')})
    assert 'brief observed feature' not in text and 'specific visual support' not in text
    assert '"printed":[]' not in text and '"identity":"unverifiable"' not in text
    assert 'up to three short printed items' in text
    assert '256 output tokens' in text


def test_strict_visual_validation_rejects_copied_placeholders():
    output = {'schema_version': VERSION, 'images': [{'image_index': 0, 'identity': 'unverifiable',
        'identity_evidence': 'No readable identification', 'printed': [], 'observations': [
            {'attribute': 'decoration', 'value': 'brief observed feature', 'evidence': 'specific visual support'}]}]}
    with pytest.raises(ValueError, match='prompt placeholders'):
        validate_response(output)


def test_saved_placeholder_is_quarantined_during_postprocessing(cfg):
    row = product('Example Bath Crayons 6 Pieces', category_group='BABY TOYS')
    evidence = extract(row)
    evidence.source('visual_observation', '/synthetic/photo.jpg', 'brief observed feature', None, exact=False,
        metadata={'visual_schema': VERSION, 'mapped_image': True, 'identity_status': 'unverifiable',
                  'attribute': 'decoration', 'visible_support': 'specific visual support'})
    result = finish(row, evidence, [], [], None, cfg, False)
    assert 'brief observed feature' not in result['values']['long_description']
    assert any(issue['kind'] == 'visual_prompt_placeholder' for issue in result['evidence']['issues'])
