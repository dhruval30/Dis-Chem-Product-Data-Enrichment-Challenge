from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field as dataclass_field

from .common import digest
from .schema import leaves


def normalise(text):
    return ' '.join(re.findall(r'[\w]+', str(text).casefold()))


@dataclass
class Source:
    id: str
    kind: str
    location: str
    text: str
    confidence: float | None = 1.0
    exact_product: bool = True
    eligible: bool = True
    metadata: dict = dataclass_field(default_factory=dict)


@dataclass
class Fact:
    field: str
    path: list
    value: object
    source_id: str
    quote: str
    confidence: float | None
    method: str = 'deterministic'
    qualifier: str = ''


class Evidence:
    def __init__(self):
        self.sources: dict[str, Source] = {}
        self.facts: list[Fact] = []
        self.issues: list[dict] = []

    def source(self, kind, location, text, confidence=1., exact=True, eligible=True, metadata=None):
        metadata = metadata or {}
        key = digest([kind, location, text, exact, eligible, metadata])[:20]
        self.sources[key] = Source(key, kind, location, text, confidence, exact, eligible, metadata)
        return key

    def add(self, field, path, value, source_id, quote=None, confidence=1., method='deterministic'):
        source = self.sources[source_id]
        fact = Fact(field, list(path), value, source_id,
                    source.text if quote is None else quote,
                    None if source.confidence is None else confidence if source.kind == 'ocr' and source.metadata.get('lines') else min(confidence, source.confidence), method)
        if fact not in self.facts and list(leaves(value)):
            self.facts.append(fact)

    def payload(self):
        return {'sources': [asdict(x) for x in self.sources.values()],
                'facts': [asdict(x) for x in self.facts], 'issues': self.issues}

    def absorb(self, other):
        self.sources.update(other.sources)
        for fact in other.facts:
            if fact not in self.facts:
                self.facts.append(fact)
        self.issues.extend(other.issues)
