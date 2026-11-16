import json
from types import SimpleNamespace

import pytest

from src.common import ROOT, Cache, config, digest, read_json, write_json
from src.enrichment import compact_prompt
from src.pipeline import archive_degraded
from src.vlm import VLM, limit_processor_pixels


def test_safe_profile_loads_base_paths_and_conservative_overrides():
    cfg = config(str(ROOT / 'profiles/m2_safe.yaml'), [])
    assert cfg['model']['profile'] == 'm2_safe'
    assert cfg['model']['max_images'] == 1
    assert cfg['model']['image_max_side'] == 512
    assert cfg['model']['batch_size'] == 1
    assert cfg['model']['stage_max_tokens']['image_extraction'] == 256
    assert cfg['policy']['vlm_selection'] == 'all'
    assert cfg['policy']['require_image']
    assert not cfg['policy']['field_enrichment']
    assert not cfg['policy']['descriptions']
    assert not cfg['policy']['image_verification']
    assert cfg['policy']['ocr'] and cfg['policy']['retrieval']
    assert cfg['budget']['retry_count'] == 0


def test_competitive_profile_remains_available():
    cfg = config(str(ROOT / 'config.yaml'), [])
    assert cfg['model']['profile'] == 'competitive'
    assert cfg['model']['max_images'] == 3
    assert cfg['policy']['field_enrichment'] and cfg['policy']['descriptions']


def test_safe_profile_rejects_multi_image_override():
    with pytest.raises(ValueError, match='M2 safe'):
        config(str(ROOT / 'profiles/m2_safe.yaml'), ['model.max_images=2'])


def test_safe_prompt_excludes_long_catalogue_text_and_ocr():
    payload = {'catalogue': {'Product Name': 'Example Powder 10g', 'T_BRAND': 'Example',
                            'Product Description': 'long catalogue copy ' * 2000},
               'ocr': ['large OCR block ' * 2000]}
    text = compact_prompt('image_extraction', payload)
    assert len(text) < config(str(ROOT / 'profiles/m2_safe.yaml'), [])['model']['max_input_chars']
    assert 'Example Powder 10g' in text
    assert 'long catalogue copy' not in text
    assert 'large OCR block' not in text


def test_pixel_controls_bound_processor_resizing():
    processor = SimpleNamespace(image_processor=SimpleNamespace(
        max_pixels=999999, min_pixels=200704,
        size={'longest_edge': 999999, 'shortest_edge': 200704}))
    limits = limit_processor_pixels(processor, {'processor_max_pixels': 262144, 'processor_min_pixels': 16384})
    assert limits['max_pixels'] == 262144
    assert processor.image_processor.size == {'longest_edge': 262144, 'shortest_edge': 16384}


def test_diagnostics_metadata_and_one_image_rejection_without_loading(tmp_path):
    cfg = config(str(ROOT / 'profiles/m2_safe.yaml'), [])
    budget = SimpleNamespace(cfg={'retry_count': 0})
    model = VLM(cfg['model'], Cache(tmp_path / 'cache'), budget, {}, 42, tmp_path / 'diagnostics')
    model.record_id = '001'
    model.record_key = digest('001')
    model.attempt_scope = 'manual-test'
    with pytest.raises(ValueError, match='Image count'):
        model.ask('image_extraction', '{}', ['first.jpg', 'second.jpg'])
    events = [json.loads(line) for line in (tmp_path / 'diagnostics/vlm.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'request_start'
    assert any(e['event'] == 'request_failed' and 'traceback' in e for e in events)
    assert events[-1]['event'] == 'cleanup_complete'
    assert all(e['record_id'] == '001' and e['profile'] == 'm2_safe' for e in events)
    assert all(e['attempt_scope'] == 'manual-test' for e in events)
    assert model.model is None


def test_retry_preserves_prepared_evidence_and_archives_degraded_result(tmp_path):
    key = digest('001')
    write_json(tmp_path / 'records' / (key + '.json'), {'record_id': '001', 'degraded': True})
    write_json(tmp_path / 'prepared' / (key + '.json'), {'paths': ['original.jpg'], 'evidence': {'facts': []}})
    archive_degraded(tmp_path / 'records', tmp_path, [{'record_id': '001'}])
    assert not (tmp_path / 'records' / (key + '.json')).exists()
    assert len(list((tmp_path / 'attempt_history' / key).glob('*.json'))) == 1
    assert read_json(tmp_path / 'prepared' / (key + '.json'))['paths'] == ['original.jpg']


def test_explicit_retry_scopes_preserve_previous_attempt_ledger(tmp_path):
    cache = Cache(tmp_path)
    original = digest(['call-key', 'original-session'])
    retry = digest(['call-key', 'manual-retry'])
    cache.attempts('vlm', original, 1)
    with pytest.raises(RuntimeError):
        cache.attempts('vlm', original, 1)
    cache.attempts('vlm', retry, 1)
    assert cache.get('vlm_attempts', original)['count'] == 1
    assert cache.get('vlm_attempts', retry)['count'] == 1
