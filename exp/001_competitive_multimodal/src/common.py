from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def read_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError('Cyclic configuration inheritance')
    value = yaml.safe_load(path.read_text())
    parent = value.pop('extends', None)
    base = read_config(path.parent / parent, seen + (path,)) if parent else {}
    def merge(target, update):
        for key, child in update.items():
            if isinstance(child, dict) and isinstance(target.get(key), dict):
                merge(target[key], child)
            else:
                target[key] = child
    merge(base, value)
    return base


def durable_event(path, event, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {'time_unix': time.time(), 'monotonic_seconds': time.monotonic(),
            'pid': os.getpid(), 'event': event, **metadata}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def config(path: str, overrides: list[str]):
    cfg = read_config(path)
    for item in overrides:
        key, raw = item.split('=', 1)
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in node:
                raise ValueError(f'Unknown configuration key: {key}')
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f'Unknown configuration key: {key}')
        node[parts[-1]] = yaml.safe_load(raw)
    for key in ('data_dir', 'image_dir', 'output_dir', 'report_dir'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    cfg['model']['path'] = str((ROOT / cfg['model']['path']).resolve())
    for key in ('output_dir', 'report_dir'):
        if not Path(cfg[key]).is_relative_to(ROOT):
            raise ValueError('All generated artifacts must stay inside the experiment')
    if cfg['model']['batch_size'] != 1:
        raise ValueError('This memory-bounded backend supports batch_size=1 only')
    if cfg['model'].get('profile') == 'm2_safe':
        if cfg['model']['max_images'] != 1 or not 0 < cfg['model']['image_max_side'] <= 512:
            raise ValueError('M2 safe mode requires one image and max side no greater than 512')
    if cfg['budget']['retry_count'] not in (0, 1):
        raise ValueError('Competition permits at most one retry')
    for key in ('max_seconds', 'max_cost_usd', 'max_rss_gb'):
        value = cfg['budget'][key]
        if value is not None and value <= 0:
            raise ValueError(f'budget.{key} must be positive or null')
    random.seed(cfg['seed'])
    import numpy as np
    np.random.seed(cfg['seed'])
    return cfg


def versions():
    names = ['numpy', 'Pillow', 'rapidocr-onnxruntime', 'onnxruntime', 'mlx', 'mlx-lm',
             'mlx-vlm', 'transformers', 'rapidfuzz', 'openpyxl', 'huggingface-hub',
             'tokenizers', 'jsonschema', 'opencv-python', 'PyYAML', 'beautifulsoup4', 'psutil',
             'torch', 'torchvision']
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not installed'
    return result


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, cfg, state_path):
        self.cfg = cfg
        self.path = state_path
        self.state = read_json(state_path) if state_path.exists() else {
            'seconds': 0., 'peak_rss_bytes': 0, 'calls': 0, 'ocr_images': 0}
        self.started = time.monotonic()
        self.base = self.state['seconds']

    def check(self):
        self.state['seconds'] = self.base + time.monotonic() - self.started
        rss = psutil.Process().memory_info().rss
        high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        high_water *= 1 if sys.platform == 'darwin' else 1024
        self.state['peak_rss_bytes'] = max(rss, high_water, self.state['peak_rss_bytes'])
        self.state['estimated_cost_usd'] = self.state['seconds'] / 3600 * self.cfg['local_cost_per_hour_usd']
        if self.cfg['max_seconds'] is not None and self.state['seconds'] >= self.cfg['max_seconds']:
            raise BudgetExceeded('Configured cumulative runtime budget exhausted')
        if self.cfg['max_rss_gb'] is not None and rss > self.cfg['max_rss_gb'] * 1024**3:
            raise BudgetExceeded('Configured RSS memory budget exceeded')
        if self.cfg['max_cost_usd'] is not None and self.state['estimated_cost_usd'] > self.cfg['max_cost_usd']:
            raise BudgetExceeded('Configured cost budget exceeded')

    def save(self):
        self.state['seconds'] = self.base + time.monotonic() - self.started
        write_json(self.path, self.state)


class Cache:
    def __init__(self, root):
        self.root = root

    def get(self, stage, key):
        p = self.root / stage / (key + '.json')
        return read_json(p) if p.exists() else None

    def put(self, stage, key, value):
        write_json(self.root / stage / (key + '.json'), value)

    def attempts(self, stage, key, maximum):
        old = self.get(stage + '_attempts', key) or {'count': 0}
        if old['count'] >= maximum:
            raise RuntimeError(f'{stage} retry allowance exhausted for cached call')
        self.put(stage + '_attempts', key, {'count': old['count'] + 1})
