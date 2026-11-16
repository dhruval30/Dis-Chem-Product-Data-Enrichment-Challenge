import json
from pathlib import Path

import pytest

from src.common import ROOT
from src.data import blocked
from src.evidence import Evidence
from src.retrieval import Catalogue, duplicate_signatures
from src.schema import Schema
from src.text_extraction import extract, quantity
from src.verification import verify
from src.vlm import parse_json


@pytest.fixture
def schema():
    return Schema(ROOT.parents[1] / 'dataset' / 'submission_format.json')


@pytest.fixture
def rules():
    return {'minimum_confidence': .8, 'field_thresholds': {},
            'allow_uncorroborated_visual': False, 'description_token_overlap': .65}


def row(**changes):
    return {'record_id': '0001', 'Product Name': 'Example Balm 50ml', 'T_BRAND': 'Example',
            'category_group': 'BEAUTY', 'Specifications': '', 'Nutritional Information': '{}',
            'Short Description': '', 'Product Description': '', **changes}


@pytest.mark.parametrize('text,value,unit', [('Balm 50ml', '50', 'ml'),
    ('Balm 2 x 25 ml', '2 x 25', 'ml'), ('Food 100 g', '100', 'g'),
    ('Medicine 30 tablets', '30', 'tablets'), ('Cloths 2 pack', '2', 'pack')])
def test_quantities(text, value, unit):
    assert quantity(text)[0] == {'value': value, 'unit': unit}


def test_partial_masks_do_not_erase_visible_specs(schema, rules):
    evidence = extract(row(Specifications='<table><tr><td>Shade</td><td>Rose</td></tr><tr><td>Weight</td><td>[MASKED]</td></tr></table>'))
    values, _, _ = verify(evidence, schema, rules, set())
    assert values['variant_attributes']['shade'] == ['Rose']
    assert values['weight'] == schema.empty['weight']


def test_volume_is_not_weight_or_dimensions(schema, rules):
    evidence = extract(row(VOLUME='0.123', T_VOLUME_UNIT='Cubic decimeter', T_WEIGHT_UNIT='Gram'))
    values, _, _ = verify(evidence, schema, rules, set())
    assert values['weight'] == schema.empty['weight']
    assert values['dimensions'] == schema.empty['dimensions']


def test_axis_and_unit_required(schema, rules):
    evidence = extract(row(Specifications='Height: 12 cm\nWidth: 7 cm\nDepth: 3 cm'))
    values, _, _ = verify(evidence, schema, rules, set())
    assert values['dimensions'] == {'height': '12', 'width': '7', 'depth': '3', 'unit': 'cm'}
    evidence = extract(row(Specifications='Height: 12\nWidth: 7'))
    assert verify(evidence, schema, rules, set())[0]['dimensions'] == schema.empty['dimensions']


def test_blocked_output_overrides_explicit_fact(schema, rules):
    evidence = extract(row(Specifications='Flavour: Apple'))
    values, _, _ = verify(evidence, schema, rules, {'variant_attributes'})
    assert values['variant_attributes'] == schema.empty['variant_attributes']
    with pytest.raises(ValueError):
        blocked({'do_not_predict_fields': '{}'})


def test_conflicting_quantities_are_omitted(schema, rules):
    evidence = extract(row(Specifications='Pack Size: 100 ml'))
    values, _, rejected = verify(evidence, schema, rules, set())
    assert values['product_specifications']['pack_size'] == schema.empty['product_specifications']['pack_size']
    assert any(x['reason'] == 'conflicting_values' for x in rejected)


def test_fuzzy_match_never_transfers_variant():
    references = [row(record_id='2', **{'Product Name': 'Example Balm 100ml'})]
    catalogue = Catalogue(references, {'top_k': 5, 'min_score': 0, 'allow_exact_transfer': True})
    candidates = catalogue.find(row())
    assert not candidates[0]['copy_allowed']
    assert catalogue.find(references[0], excluded_signatures=duplicate_signatures(references[0])) == []


def test_nutrition_inequality_and_basis_are_not_changed(schema, rules):
    raw = {'nutrients': {'Protein (g)': {'per_100g': '<0.2', 'per_serving': '2'},
                         'Energy (kJ)': {'per_100g': '100', 'per_serving': '200'}},
           'source_column_labels': {'per_100g': 'Per 100ml', 'per_serving': 'Per serving'}}
    values = verify(extract(row(**{'Nutritional Information': json.dumps(raw)})), schema, rules, set())[0]
    assert values['nutritional_information']['per_100g'] == {}
    assert values['nutritional_information']['per_serving'] == {'protein_g': 2., 'energy_kj': 200.}


def test_invented_value_is_rejected(schema, rules):
    evidence = Evidence()
    source = evidence.source('record', 'Specifications', 'Weight: 20 g')
    evidence.add('weight', [], {'value': '30', 'unit': 'g'}, source)
    values, _, rejected = verify(evidence, schema, rules, set())
    assert values['weight'] == schema.empty['weight']
    assert rejected[0]['reason'] == 'value_not_in_quote'


def test_model_response_parser_rejects_ambiguous_output():
    assert parse_json('```json\n{"facts":[]}\n```') == {'facts': []}
    with pytest.raises(ValueError):
        parse_json('{"facts":[]} {"facts":[]}')
