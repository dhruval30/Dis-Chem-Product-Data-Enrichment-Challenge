import importlib.util
import sys

import pytest

from src.common import ROOT, write_json
from src import pipeline


@pytest.mark.parametrize('selection,complete,should_raise', [
    (['--record-id', '310290'], False, False),
    (['--limit', '1'], False, False),
    ([], False, True),
    ([], True, False),
])
def test_cli_distinguishes_selected_records_from_full_run(
        tmp_path, monkeypatch, selection, complete, should_raise):
    spec = importlib.util.spec_from_file_location('experiment001_cli', ROOT / 'run.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    write_json(tmp_path / 'manifest.json', {'complete': complete})
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return tmp_path

    monkeypatch.setattr(cli, 'config', lambda *args: {})
    monkeypatch.setattr(pipeline, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['run.py', 'infer', '--split', 'test', *selection])
    if should_raise:
        with pytest.raises(RuntimeError, match='Incomplete or degraded run'):
            cli.main()
    else:
        cli.main()
    assert calls[0]['record_ids'] == (['310290'] if '--record-id' in selection else [])
