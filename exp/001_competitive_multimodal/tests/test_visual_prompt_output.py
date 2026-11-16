import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from src.common import ROOT, Cache, config, digest
from src.enrichment import compact_prompt
from src.visual import VERSION, response_schema
from src.vlm import VLM, parse_json


def payload():
    return {'catalogue': {'Product Name': "Mother's Choice Lace Socks With Bows Size 6-12 months Assorted",
                          'T_BRAND': 'Mothers Choice', 'Product Description': 'irrelevant copy ' * 500}}


def test_input_context_precedes_output_contract():
    text = compact_prompt('image_extraction', payload())
    start, end = text.index('<INPUT_CONTEXT>'), text.index('</INPUT_CONTEXT>')
    context = json.loads(text[start + len('<INPUT_CONTEXT>'):end])
    assert set(context) == {'Product Name', 'T_BRAND', 'category_group'}
    assert text.index('OUTPUT INSTRUCTIONS') > end
    assert 'Do not echo it' in text[end:]
    assert 'Never return Product Name or T_BRAND as top-level output keys' in text[end:]
    assert 'schema_version must equal "visual-evidence-v2"' in text[end:]
    assert not text.rstrip().endswith(json.dumps(context, ensure_ascii=False))
    assert 'irrelevant copy' not in text


def test_textless_image_instructions_require_observation_schema_and_bound_output():
    text = compact_prompt('image_extraction', payload())
    for required in ('image_index=0', 'identity_evidence', 'printed and observations',
                     'attribute, value and evidence', 'unverifiable when identification is absent'):
        assert required in text
    assert 'up to two observations' in text
    assert 'brief observed feature' not in text
    assert 'specific visual support' not in text
    assert '"identity":"unverifiable"' not in text
    assert '"printed":[]' not in text
    assert '256 output tokens' in text
    safe = config(str(ROOT / 'profiles/m2_safe.yaml'), [])
    assert len(text) < safe['model']['max_input_chars']
    assert safe['model']['stage_max_tokens']['image_extraction'] == 256


def test_echoed_input_is_not_coerced_into_visual_evidence():
    echoed = parse_json(json.dumps(payload()['catalogue']))
    assert list(Draft202012Validator(response_schema()).iter_errors(echoed))
    complete = {'schema_version': VERSION, 'images': [{'image_index': 0,
        'identity': 'unverifiable', 'identity_evidence': 'No printed identification.', 'printed': [],
        'observations': [{'attribute': 'decoration', 'value': 'decorative bows',
                          'evidence': 'bows attached to visible socks'}]}]}
    assert not list(Draft202012Validator(response_schema()).iter_errors(complete))


def test_assembled_and_formatted_prompts_are_saved_separately(tmp_path):
    model = VLM({'profile': 'm2_safe'}, Cache(tmp_path / 'cache'), SimpleNamespace(), {}, 42,
                diagnostics_dir=tmp_path / 'diagnostics')
    model.record_id = '310290'
    model.record_key = digest('310290')
    assembled = compact_prompt('image_extraction', payload())
    first = model.save_prompt('image_extraction', assembled)
    formatted = model.save_prompt('image_extraction', '<image>' + assembled, kind='chat_formatted')
    assert first != formatted
    assert first.read_text() == assembled
    assert formatted.read_text() == '<image>' + assembled
    events = [json.loads(line) for line in (tmp_path / 'diagnostics/vlm.jsonl').read_text().splitlines()]
    assert {event['prompt_kind'] for event in events} == {'assembled', 'chat_formatted'}
    assert all(event['record_id'] == '310290' for event in events)


def test_changed_compact_prompt_cannot_reuse_echo_cache_key(tmp_path):
    class SpyCache(Cache):
        def __init__(self, path):
            super().__init__(path)
            self.keys = []
        def get(self, stage, key):
            self.keys.append(key)
            return None
    safe = config(str(ROOT / 'profiles/m2_safe.yaml'), [])
    cache = SpyCache(tmp_path / 'cache')
    model = VLM(safe['model'], cache, SimpleNamespace(), {}, 42, tmp_path / 'diagnostics')
    def stop_before_load():
        raise RuntimeError('mock stops before loading any model')
    model.load = stop_before_load
    old = 'Target: ' + json.dumps({key: value for key, value in payload()['catalogue'].items()
                                  if key in ('Product Name', 'T_BRAND')})
    new = compact_prompt('image_extraction', payload())
    for text in (old, new):
        with pytest.raises(RuntimeError, match='mock stops'):
            model._ask('image_extraction', text)
    assert len(cache.keys) == 2 and cache.keys[0] != cache.keys[1]
