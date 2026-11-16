import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.common import ROOT, Cache, digest, read_json, versions, write_json
from src.enrichment import finish
from src.pipeline import archive_degraded, reuse_ocr_cache
from src.schema import Schema
from src.text_extraction import extract
from src.verification import deduplicate
from src.vlm import check_processor


@pytest.fixture
def cfg():
    value = yaml.safe_load((ROOT / 'config.yaml').read_text())
    value['_schema'] = Schema(ROOT.parents[1] / 'dataset/submission_format.json')
    return value


def product(**changes):
    return {'record_id': '001', 'Product Name': 'Example Hand Balm 50ml', 'T_BRAND': 'Example',
            'category_group': 'PERSONAL CARE', 'Short Description': '', 'Product Description': '',
            'Specifications': '', 'Nutritional Information': '{}', **changes}


def test_failed_model_load_is_not_reported_as_model_use(cfg):
    class MissingDependency:
        loaded = False
        generation_count = 0
        def ask(self, *args):
            raise ImportError('torchvision is required by AutoVideoProcessor')
    row = product()
    result = finish(row, extract(row), [], [], MissingDependency(), cfg, True)
    assert result['model_attempted'] is True
    assert result['model_loaded'] is False
    assert result['model_succeeded'] is False
    assert result['model_failed'] is True
    assert result['used_model'] is False
    assert result['model_generation_calls'] == 0
    assert result['degraded'] is True
    assert result['values']['online_name'] == row['Product Name']


def test_successful_model_state_is_distinct_from_attempt(cfg):
    class WorkingModel:
        loaded = True
        generation_count = 0
        def ask(self, stage, *args):
            self.generation_count += 1
            return {'facts': []} if stage == 'field_enrichment' else {'short_description': [], 'long_description': []}
    row = product()
    result = finish(row, extract(row), [], [], WorkingModel(), cfg, True)
    assert result['model_loaded'] is True
    assert result['model_succeeded'] is True
    assert result['model_failed'] is False
    assert result['used_model'] is True
    assert result['model_generation_calls'] > 0


def test_processor_check_initialises_processor_only(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text('{}')
    calls = []
    def load_processor(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='2.8.0'))
    monkeypatch.setitem(sys.modules, 'torchvision', SimpleNamespace(__version__='0.23.0'))
    monkeypatch.setitem(sys.modules, 'mlx_vlm.utils', SimpleNamespace(load_processor=load_processor))
    result = check_processor(tmp_path)
    assert result['status'] == 'loaded'
    assert result['weights_loaded'] is False
    assert result['generation_performed'] is False
    assert calls == [(tmp_path.resolve(), {'local_files_only': True})]


def test_retry_archives_degraded_outputs_and_preserves_prepared(tmp_path):
    records = tmp_path / 'records'
    prepared = tmp_path / 'prepared'
    key = digest('001')
    write_json(records / (key + '.json'), {'record_id': '001', 'degraded': True})
    write_json(prepared / (key + '.json'), {'evidence': 'retained'})
    good_key = digest('002')
    write_json(records / (good_key + '.json'), {'record_id': '002', 'degraded': False})
    archive_degraded(records, tmp_path, [{'record_id': '001'}, {'record_id': '002'}])
    assert not (records / (key + '.json')).exists()
    history = list((tmp_path / 'attempt_history' / key).glob('*.json'))
    assert len(history) == 1 and read_json(history[0])['degraded']
    assert read_json(prepared / (key + '.json')) == {'evidence': 'retained'}
    assert (records / (good_key + '.json')).exists()


def test_torch_dependency_change_can_reuse_identical_ocr(tmp_path):
    cache = Cache(tmp_path)
    path = Path('/synthetic/photo.jpg')
    source = {'src/ocr.py': 'same-ocr', 'src/image_processing.py': 'same-images'}
    old_packages = {'numpy': '2', 'torch': 'not installed', 'torchvision': 'not installed'}
    new_packages = {**old_packages, 'torch': '2.8.0', 'torchvision': '0.23.0'}
    old = {'config': {'ocr': {'max_side': 2200}}, 'source': source, 'packages': old_packages,
           'images': {str(path): 'image-hash'}}
    new = {**old, 'packages': new_packages}
    old_key = digest(['image-hash', old['config']['ocr'], old_packages, 'same-ocr', 'same-images'])
    raw = {'image_hash': 'image-hash', 'raw': [{'text': 'original OCR'}]}
    cache.put('ocr', old_key, raw)
    assert reuse_ocr_cache(cache, path, old['config'], new, old)
    new_key = digest(['image-hash', old['config']['ocr'], new_packages, 'same-ocr', 'same-images'])
    assert cache.get('ocr', new_key) == raw
    new['source'] = {**source, 'src/ocr.py': 'changed-preprocessing'}
    assert not reuse_ocr_cache(cache, path, old['config'], new, old)


def test_supported_qualified_claim_is_not_discarded(cfg):
    row = product(**{'Product Description': 'This balm may help soothe skin.'})
    result = finish(row, extract(row), [], [], None, cfg, False)
    assert 'may help soothe' in (result['values']['short_description'] + result['values']['long_description']).lower()
    assert 'Catalogue category:' not in result['values']['long_description']
    assert any(s['kind'] == 'grounded_text' for s in result['evidence']['sources'])


def test_structured_duplicates_keep_first_declared_spelling():
    value = [{'attribute': 'Material', 'value': 'Cotton'}, {'attribute': 'material', 'value': 'cotton'}]
    assert deduplicate(value) == value[:1]
    assert deduplicate(['Blue', 'blue', 'BLUE']) == ['Blue']


def test_reproducibility_tracks_processor_dependencies():
    assert {'torch', 'torchvision'} <= versions().keys()
