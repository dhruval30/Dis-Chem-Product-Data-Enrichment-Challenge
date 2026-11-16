import copy
from pathlib import Path

import pytest
import yaml

from src.common import ROOT, Budget, BudgetExceeded, Cache
from src.enrichment import add_visual, agrees_with_ocr, finish, should_use_vlm
from src.evidence import Evidence
from src.ocr import group_lines
from src.retrieval import Catalogue
from src.schema import Schema
from src.submission import validate_traces, write_submission
from src.text_extraction import extract, extract_nutrition_text
from src.verification import quote_confidence, verify
from src.vlm import parse_json


def record(name='Example Daily Hand Balm 50ml', **changes):
    return {'record_id': '001', 'Product Name': name, 'T_BRAND': 'Example',
            'category_group': 'PERSONAL CARE', 'Short Description': '', 'Product Description': '',
            'Specifications': '', 'Nutritional Information': '{}', **changes}


@pytest.fixture
def cfg():
    value = yaml.safe_load((ROOT / 'config.yaml').read_text())
    value['_schema'] = Schema(ROOT.parents[1] / 'dataset/submission_format.json')
    return value


def test_monitoring_without_unconfirmed_limits(tmp_path):
    cfg = {'max_seconds': None, 'max_cost_usd': None, 'max_rss_gb': None, 'local_cost_per_hour_usd': 1.}
    budget = Budget(cfg, tmp_path / 'budget.json')
    budget.base = 50000
    budget.check()
    assert budget.state['seconds'] >= 50000
    assert budget.state['estimated_cost_usd'] > 0
    cfg['max_seconds'] = 1
    with pytest.raises(BudgetExceeded):
        budget.check()


def test_stage_failure_keeps_source_copy_and_facts(cfg):
    row = record(**{'Short Description': 'A daily balm for hands.',
                    'Product Description': 'A hand balm with a smooth texture. Apply to hands.',
                    'Specifications': 'Packaging Type: Tube'})
    class BrokenModel:
        def ask(self, *args):
            raise ValueError('truncated JSON')
    result = finish(row, extract(row), [], [], BrokenModel(), cfg, True)
    assert result['degraded']
    assert result['values']['short_description'] == row['Short Description']
    assert 'smooth texture' in result['values']['long_description']
    assert result['values']['long_description'].startswith(row['Product Name'])
    assert result['values']['product_specifications']['packaging_type'] == 'Tube'
    assert validate_traces(result, row, cfg['_schema'], cfg['verification']) == []


def test_partial_generation_falls_back_to_complete_source(cfg):
    row = record(**{'Short Description': 'A hand balm with a smooth texture.',
                    'Product Description': 'A hand balm with a smooth texture. Supplied in a tube.'})
    class PartialModel:
        def ask(self, stage, *args):
            return {'facts': []} if stage == 'field_enrichment' else {'long_description': []}
    result = finish(row, extract(row), [], [], PartialModel(), cfg, True)
    assert 'smooth texture' in result['values']['short_description']
    assert 'Supplied in a tube.' in result['values']['long_description']
    assert 'smooth texture' not in result['values']['long_description']


def test_image_mapping_handles_short_and_empty_excel_rows(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src import data

    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    image = image_dir / 'photo.jpg'
    image.write_bytes(b'fixture')
    rows = [('record_id', 'img_filename'), (), (2,), (None, 'photo.jpg'),
            ('001', 'photo.jpg'), ('001', 'photo.jpg')]
    closed = []
    workbook = SimpleNamespace(
        active=SimpleNamespace(iter_rows=lambda **kwargs: iter(rows)),
        close=lambda: closed.append(True))
    monkeypatch.setattr(data, 'load_workbook', lambda *args, **kwargs: workbook)
    mapping, warnings = data.image_mapping(tmp_path, image_dir)
    assert mapping == {'001': [image.resolve()]}
    assert warnings == []
    assert closed == [True]


def test_existing_description_does_not_itself_trigger_vlm(cfg):
    row = record(**{'Short Description': 'A simple balm.', 'Product Description': 'A simple hand balm. ' * 40})
    assert not should_use_vlm(row, extract(row), [], cfg)
    cfg['policy']['vlm_selection'] = 'all'
    assert should_use_vlm(row, extract(row), [], cfg)


def test_missing_labelled_subfield_can_trigger_recovery(cfg):
    row = record(**{'Short Description': 'A hand balm.',
                    'Product Description': 'A hand balm. Packaging Type: Tube'})
    assert should_use_vlm(row, extract(row), [], cfg)


def visual_response(text, confidence=.99, **identity):
    return {'images': [{'image_index': 0, 'match': True, 'match_confidence': .99,
                       'visible_name': 'Example Daily Hand Balm', 'visible_brand': 'Example',
                       'visible_variant': '', 'visible_quantity': '50ml',
                       'transcriptions': [{'text': text, 'confidence': confidence}], **identity}]}


def test_clear_image_only_fact_is_eligible(cfg):
    row = record()
    evidence = extract(row)
    add_visual(visual_response('Packaging Type: Tube'), [Path('/synthetic/image.jpg')],
               evidence, cfg['verification'], row)
    source = next(s for s in evidence.sources.values() if s.kind == 'visual')
    evidence.add('product_specifications', ['packaging_type'], 'Tube', source.id, source.text, .99)
    assert verify(evidence, cfg['_schema'], cfg['verification'], set())[0]['product_specifications']['packaging_type'] == 'Tube'


def test_high_risk_visual_requires_confirmation(cfg):
    row, evidence = record(), extract(record())
    response = visual_response('Ingredients: Water, Glycerin. May contain milk.')
    add_visual(response, [Path('/synthetic/image.jpg')], evidence, cfg['verification'], row)
    source = next(s for s in evidence.sources.values() if s.kind == 'visual')
    evidence.add('ingredient_information', [], ['Water, Glycerin. May contain milk.'], source.id, source.text, .99)
    values, _, rejected = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['ingredient_information'] == []
    assert any(x['reason'] == 'high_risk_visual_not_confirmed' for x in rejected)
    second_evidence = extract(row)
    add_visual(response, [Path('/synthetic/image.jpg')], second_evidence, cfg['verification'], row, response)
    source = next(s for s in second_evidence.sources.values() if s.kind == 'visual')
    second_evidence.add('ingredient_information', [], ['Water, Glycerin. May contain milk.'], source.id, source.text, .99)
    assert verify(second_evidence, cfg['_schema'], cfg['verification'], set())[0]['ingredient_information']


def test_image_quantity_mismatch_quarantines_ocr(cfg):
    row, evidence = record(), extract(record())
    source = evidence.source('ocr', '/synthetic/image.jpg', 'Packaging Type: Tube', .99)
    add_visual(visual_response('Packaging Type: Tube', visible_quantity='100ml'),
               [Path('/synthetic/image.jpg')], evidence, cfg['verification'], row)
    assert not evidence.sources[source].eligible
    assert not any(s.kind == 'visual' and s.eligible for s in evidence.sources.values())


def test_ocr_agreement_preserves_inequalities():
    evidence = Evidence()
    source = evidence.source('ocr', '/synthetic/image.jpg', 'Protein (g) 0.1', .99)
    assert not agrees_with_ocr('Protein (g) <0.1', evidence.sources[source])


def test_back_panel_can_link_to_confirmed_supplied_front(cfg):
    row = record()
    response = visual_response('Packaging Type: Tube')
    response['images'].append({'image_index': 1, 'match': True, 'match_confidence': .99,
        'identity_image_index': 0, 'visible_name': '', 'visible_brand': '',
        'visible_variant': '', 'visible_quantity': '',
        'transcriptions': [{'text': 'Ingredients: Water.', 'confidence': .99}]})
    evidence = extract(row)
    add_visual(response, [Path('/synthetic/front.jpg'), Path('/synthetic/back.jpg')],
               evidence, cfg['verification'], row, response)
    source = next(s for s in evidence.sources.values() if s.kind == 'visual' and 'back.jpg' in s.location)
    evidence.add('ingredient_information', [], ['Water.'], source.id, source.text, .99)
    assert verify(evidence, cfg['_schema'], cfg['verification'], set())[0]['ingredient_information'] == ['Water.']


def test_family_transfer_is_scoped_and_never_copies_quantity(cfg):
    rows = [record('Example Daily Hand Balm 100ml', record_id='2', Specifications='Application Area: Hands\nPack Size: 100 ml'),
            record('Example Daily Hand Balm 200ml', record_id='3', Specifications='Application Area: Hands\nPack Size: 200 ml')]
    catalogue = Catalogue(rows, cfg['retrieval'])
    evidence = extract(record())
    catalogue.enrich(evidence, catalogue.find(record()))
    values, _, _ = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['product_specifications']['application_area'] == ['Hands']
    assert values['product_specifications']['pack_size'] == {'value': '50', 'unit': 'ml'}
    family = next(s for s in evidence.sources.values() if s.kind == 'family_shared')
    evidence.add('ingredient_information', [], ['Hands'], family.id, family.text)
    assert verify(evidence, cfg['_schema'], cfg['verification'], set())[0]['ingredient_information'] == []


def test_family_disagreement_prevents_shared_transfer(cfg):
    rows = [record('Example Daily Hand Balm 100ml', record_id='2', Specifications='Application Area: Hands'),
            record('Example Daily Hand Balm 200ml', record_id='3', Specifications='Application Area: Face')]
    catalogue = Catalogue(rows, cfg['retrieval'])
    evidence = extract(record())
    catalogue.enrich(evidence, catalogue.find(record()))
    assert not any(s.kind == 'family_shared' for s in evidence.sources.values())


def test_nutrition_rows_keep_column_basis_and_ignore_unrelated_weak_line(cfg):
    evidence = Evidence()
    lines = [{'text': 'Per 100g Per serving', 'confidence': .99},
             {'text': 'Energy (kJ) 100 200', 'confidence': .99},
             {'text': 'Protein (g) <0.1 2', 'confidence': .99},
             {'text': 'unreadable slogan', 'confidence': .40}]
    source = evidence.source('ocr', '/synthetic/image.jpg/block/0', '\n'.join(x['text'] for x in lines),
                             .85, metadata={'lines': lines})
    extract_nutrition_text(evidence, source)
    values, _, _ = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['nutritional_information']['per_100g'] == {'energy_kj': 100.}
    assert values['nutritional_information']['per_serving'] == {'energy_kj': 200., 'protein_g': 2.}
    assert quote_confidence(evidence.sources[source], 'Energy (kJ) 100 200') == .99


def test_redundant_unit_notation_does_not_create_conflict(cfg):
    evidence = extract(record(Specifications='Pack Size: 50 millilitre (ml)'))
    values, _, rejected = verify(evidence, cfg['_schema'], cfg['verification'], set())
    assert values['product_specifications']['pack_size']['value'] == '50'
    assert not any(x['reason'] == 'conflicting_values' for x in rejected)


def test_cache_attempts_survive_reopening(tmp_path):
    Cache(tmp_path).attempts('vlm', 'same-key', 2)
    Cache(tmp_path).attempts('vlm', 'same-key', 2)
    with pytest.raises(RuntimeError):
        Cache(tmp_path).attempts('vlm', 'same-key', 2)


def test_wrapped_json_is_recovered_without_repairing_numbers():
    assert parse_json('Result:\n{"facts":[]}\nDone.') == {'facts': []}
    with pytest.raises(ValueError):
        parse_json('{"facts":[')


def test_incomplete_run_cannot_release_submission(cfg, tmp_path):
    from src.common import write_json
    cfg['output_dir'] = str(tmp_path)
    write_json(tmp_path / 'manifest.json', {'split': 'test', 'complete': False})
    with pytest.raises(ValueError, match='incomplete'):
        write_submission(tmp_path, cfg)
