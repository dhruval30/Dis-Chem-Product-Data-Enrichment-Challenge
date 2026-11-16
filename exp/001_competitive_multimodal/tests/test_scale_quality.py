import pytest
import yaml

from src.common import ROOT, config
from src.enrichment import finish, add_visual
from src.schema import Schema
from src.submission import validate_traces
from src.text_extraction import extract, pack_quantity
from src.verification import verify
from src.visual import VERSION, corroborated
from pathlib import Path


@pytest.fixture
def cfg():
    value = yaml.safe_load((ROOT / 'config.yaml').read_text())
    value['_schema'] = Schema(ROOT.parents[1] / 'dataset/submission_format.json')
    return value


def product(name='Example Daily Cream 50ml', **changes):
    return {'record_id': '001', 'Product Name': name, 'T_BRAND': 'Example', 'category_group': 'PERSONAL CARE',
            'Short Description': '', 'Product Description': '', 'Specifications': '',
            'Nutritional Information': '{}', **changes}


def panel(evidence, lines):
    return evidence.source('ocr', '/synthetic/image.jpg/block/0', '\n'.join(lines), .99,
        metadata={'lines': [{'text': line, 'confidence': .99} for line in lines]})


def part(text, x, y, width=100, confidence=.99):
    return {'text': text, 'confidence': confidence, 'angle': 0,
            'box': [[x, y], [x + width, y], [x + width, y + 20], [x, y + 20]]}


@pytest.mark.parametrize('name,value,unit', [
    ('Example Cleansing Wipes X 70', '70', 'Wipes'),
    ('Example Cleansing Wipes ×70', '70', 'Wipes'),
    ("Example Breast Pads 30's", '30', 'Pads'),
    ('Example Breast Pads 30’s', '30', 'Pads'),
    ('Example Crayons 48 PCS', '48', 'PCS'),
])
def test_declared_count_notations_reach_pack_size(cfg, name, value, unit):
    row = product(name)
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['product_specifications']['pack_size'] == {'value': value, 'unit': unit}
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_model_code_is_not_a_count():
    assert pack_quantity('Example Wipes Model X70') is None


def test_diffuser_scent_is_not_a_volume_or_flavour(cfg):
    row = product('Example Cotton Flower And Freesia Fragrance Diffuser 250ml', category_group='HOME SCENTS',
                  Specifications='Flavour: Cotton Flower And Freesia')
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['variant_attributes']['fragrance'] == ['Cotton Flower And Freesia']
    assert result['values']['variant_attributes']['flavour'] == []
    evidence = extract(row)
    source = next(s for s in evidence.sources.values() if s.location == 'Product Name')
    evidence.add('variant_attributes', ['fragrance'], ['250ml'], source.id)
    values, _, rejected = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['variant_attributes']['fragrance'] == ['Cotton Flower And Freesia']
    assert any(r['reason'] == 'invalid_variant_value' for r in rejected)
    row['Specifications'] = ''
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['variant_attributes']['fragrance'] == ['Cotton Flower And Freesia']


def test_inline_ingredients_and_usage_labelled_storage_are_extracted(cfg):
    row = product(**{'Product Description': 'Clean the skin. Ingredients, Water, Glycerin, Citric Acid. May contain milk.',
                     'Short Description': 'Usage Instructions:\nStore in a cool, dry place, away from direct sunlight.'})
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert result['values']['ingredient_information'] == ['Water, Glycerin, Citric Acid. May contain milk.']
    assert result['values']['storage_instructions'] == ['Store in a cool, dry place, away from direct sunlight.']
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_nutrition_basis_span_is_not_pack_contents(cfg):
    row = product('Example Daily Cereal 250g', category_group='BABY FOOD AND FEEDING')
    evidence = extract(row)
    panel(evidence, ['EXAMPLE', 'Daily Cereal'])
    evidence.source('ocr', '/synthetic/image.jpg', '100g', .99)
    evidence.source('ocr', '/synthetic/image.jpg/block/1', 'NUTRITION PER100g', .99,
        metadata={'lines': [{'text': 'NUTRITION PER100g', 'confidence': .99,
                            'parts': [part('100g', 200, 40)]}]})
    result = finish(row, evidence, [], [], None, cfg, False)
    assert result['values']['product_specifications']['pack_size'] == {'value': '250', 'unit': 'g'}
    assert not any(i['kind'] == 'ocr_quantity_mismatch' for i in result['evidence']['issues'])


def test_back_panel_storage_column_does_not_become_variant_nutrition(cfg):
    row = product('Example Original Baby Cereal 250g', category_group='BABY FOOD AND FEEDING')
    evidence = extract(row)
    parts = [part('EXAMPLE', 0, 0), part('STORE INA COOL,', 300, 30, 140),
             part('DRYPLACEBEFORE', 300, 55, 140), part('ANDAFTEROPENING.', 300, 80, 140),
             part('SEAL BAG ONCE', 300, 105, 140), part('OPENED.DISCARD', 300, 130, 140),
             part('UNFINISHED', 300, 155, 140), part('PORTIONS.', 300, 180, 140),
             part('Important Notice:', 300, 210, 140), part('Protein (g) 5.5', 0, 80, 150)]
    evidence.source('ocr', '/synthetic/image.jpg/block/0', ' '.join(p['text'] for p in parts), .99,
        metadata={'lines': [{'text': ' '.join(p['text'] for p in parts), 'confidence': .99, 'parts': parts}]})
    result = finish(row, evidence, [], [], None, cfg, False)
    storage = ' '.join(result['values']['storage_instructions'])
    assert 'STORE IN A COOL' in storage and 'DRY PLACE BEFORE AND AFTER OPENING' in storage
    assert 'UNFINISHED PORTIONS' in storage and 'Protein' not in storage
    assert result['values']['nutritional_information']['per_100g'] == {}
    assert any(s.get('metadata', {}).get('mapped_storage_only') for s in result['evidence']['sources'])
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


@pytest.mark.parametrize('name,category,feature', [
    ('Example Magic Shine Eraser Gel To Powder 10g', 'COSMETICS', 'MATTIFYING'),
    ('Example Human Body Magnetic Puzzle Game 48 pieces', 'BABY TOYS', 'Four Body Patterns'),
])
def test_functional_ocr_is_used_when_descriptions_are_missing(cfg, name, category, feature):
    row = product(name, category_group=category)
    evidence = extract(row)
    panel(evidence, ['EXAMPLE', name, feature])
    result = finish(row, evidence, [], [], None, cfg, False)
    assert feature.lower() in result['values']['long_description'].lower()


def test_formulation_disagreement_blocks_image_evidence(cfg):
    row = product('Example Isosource Standard Neutral 500ml', category_group='NUTRITION CATEGORY',
                  **{'Product Description': 'Formula without the inclusion of dietary fiber.'})
    evidence = extract(row)
    identifier = panel(evidence, ['EXAMPLE', 'Isosource', 'standardfibre', '500ml'])
    finish(row, evidence, [], [], None, cfg, False)
    assert not evidence.sources[identifier].eligible
    assert evidence.sources[identifier].metadata['identity_hold_reason']
    output = {'schema_version': VERSION, 'images': [{'image_index': 0, 'identity': 'confirmed',
        'identity_evidence': 'Example Isosource standardfibre', 'printed': [
            {'kind': 'identity', 'text': 'Example Isosource standardfibre'}], 'observations': []}]}
    add_visual(output, [Path('/synthetic/image.jpg')], evidence, cfg['verification'], row)
    assert not any(s.kind == 'visual_printed' for s in evidence.sources.values())


def test_numeric_claims_cannot_change_units_or_evade_nutrition_checks(cfg):
    row = product('Example Protein Powder 500g', category_group='NUTRITION CATEGORY')
    evidence = extract(row)
    identifier = panel(evidence, ['EXAMPLE', 'Protein Powder', 'PROTEIN 22.1 g'])
    assert corroborated('PROTEIN 22.1%', [evidence.sources[identifier]]) is None
    output = {'schema_version': VERSION, 'images': [{'image_index': 0, 'identity': 'confirmed',
        'identity_evidence': 'Example Protein Powder', 'printed': [
            {'kind': 'identity', 'text': 'Example Protein Powder'},
            {'kind': 'claim', 'text': 'PROTEIN 22.1%'}], 'observations': []}]}
    add_visual(output, [Path('/synthetic/image.jpg')], evidence, cfg['verification'], row)
    result = finish(row, evidence, [], [], None, cfg, False)
    assert '22.1%' not in result['values']['long_description']


def test_full_m2_profile_includes_unpictured_records_and_keeps_safe_limits():
    value = config(str(ROOT / 'profiles/m2_full.yaml'), [])
    assert value['policy']['require_image'] is False
    assert value['policy']['vlm_selection'] == 'adaptive'
    assert value['model']['max_images'] == 1 and value['model']['image_max_side'] == 512
    assert value['model']['stage_max_tokens']['image_extraction'] == 256
    assert not value['policy']['field_enrichment'] and not value['policy']['descriptions']
