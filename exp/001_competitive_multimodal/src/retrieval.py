from __future__ import annotations

import json
import re
from collections import defaultdict

from rapidfuzz import fuzz

from .evidence import normalise
from .text_extraction import QUANTITY, extract


def family_name(row):
    return normalise(QUANTITY.sub(' ', row.get('Product Name', '')))


def identity(row):
    return (normalise(row.get('Product Name', '')), normalise(row.get('T_BRAND', '')),
            normalise(row.get('category_group', '')))


def duplicate_signatures(row):
    signatures = {('name', identity(row))}
    for key in ('Short Description', 'Product Description'):
        text = normalise(row.get(key, ''))
        if len(text) >= 80:
            signatures.add((key, text))
    return signatures


class Catalogue:
    def __init__(self, rows, cfg):
        self.rows, self.cfg = rows, cfg
        self.by_brand = defaultdict(set)
        self.by_token = defaultdict(set)
        self.identities = []
        for index, row in enumerate(rows):
            name, brand, category = identity(row)
            self.identities.append((name, brand, category))
            self.by_brand[brand].add(index)
            for token in name.split():
                self.by_token[token].add(index)

    def find(self, row, excluded_ids=(), excluded_signatures=()):
        name, brand, category = identity(row)
        candidates = set(self.by_brand.get(brand, ())) if brand else set()
        if not candidates:
            for token in name.split():
                candidates.update(self.by_token.get(token, ()))
        found = []
        excluded_ids, excluded_signatures = set(excluded_ids), set(excluded_signatures)
        for index in candidates:
            candidate = self.rows[index]
            if candidate['record_id'] in excluded_ids or duplicate_signatures(candidate) & excluded_signatures:
                continue
            cn, cb, cc = self.identities[index]
            if category and cc != category:
                continue
            if brand and cb != brand:
                continue
            score = fuzz.ratio(name, cn)
            if score >= self.cfg['min_score']:
                exact = bool(name and brand and (name, brand, category) == (cn, cb, cc))
                target_core, candidate_core = family_name(row), family_name(candidate)
                same_line = bool(brand and cb == brand and category == cc and
                                 len(set(target_core.split()) - set(brand.split())) >=
                                 self.cfg.get('family_min_identity_tokens', 3) - 1)
                for key in ('T_DEPARTMENT', 'T_CATEGORY', 'T_BMC'):
                    if row.get(key) and candidate.get(key) and normalise(row[key]) != normalise(candidate[key]):
                        same_line = False
                size_only = same_line and target_core == candidate_core
                family = same_line and (size_only or fuzz.ratio(target_core, candidate_core) >= self.cfg.get('family_min_score', 88))
                found.append({'index': index, 'score': score, 'relation': 'exact' if exact else
                              'family' if family else 'related', 'size_only': size_only,
                              'record_id': candidate['record_id'], 'name': candidate['Product Name'],
                              'copy_allowed': exact and self.cfg['allow_exact_transfer']})
        return sorted(found, key=lambda x: (-x['score'], self.rows[x['index']]['record_id']))[:self.cfg['top_k']]

    def enrich(self, evidence, candidates):
        shared = defaultdict(list)
        for candidate in candidates:
            row = self.rows[candidate['index']]
            if candidate['copy_allowed']:
                extra = extract(row, kind='exact_catalogue', exact=True)
                # Keep reference provenance distinguishable from the target record.
                for source in extra.sources.values():
                    source.location = f"catalogue/{row['record_id']}/{source.location}"
                evidence.absorb(extra)
            elif candidate['relation'] == 'family':
                extra = extract(row, kind='family_catalogue', exact=False, eligible=False)
                for fact in extra.facts:
                    if fact.field != 'product_specifications' or len(fact.path) != 1:
                        continue
                    if fact.path[0] not in self.cfg.get('shared_attributes', []):
                        continue
                    source = extra.sources[fact.source_id]
                    key = (fact.path[0], json.dumps(fact.value, sort_keys=True))
                    shared[key].append((candidate, fact, source))
                # Context stays non-transferable. It is supplied separately to the model.
                for source in extra.sources.values():
                    if 'Description#' in source.location:
                        evidence.source('family_context', f"catalogue/{row['record_id']}/{source.location}",
                                        source.text, exact=False, eligible=False)
        if not self.cfg.get('allow_family_transfer', False):
            return
        target_sources = [s for s in evidence.sources.values() if s.kind == 'record']
        for (attribute, serialised), entries in shared.items():
            distinct = {x[0]['record_id'] for x in entries}
            distinct_products = {normalise(x[0]['name']) for x in entries}
            if min(len(distinct), len(distinct_products)) < self.cfg.get('family_consensus', 2):
                continue
            # Consensus cannot override a conflicting family declaration.
            if any(k[0] == attribute and k != (attribute, serialised) for k in shared):
                continue
            candidate, fact, source = entries[0]
            terms = fact.value if isinstance(fact.value, list) else [fact.value]
            anchors = [s.id for s in target_sources if all(
                f' {normalise(v)} ' in f' {normalise(s.text)} ' for v in terms)]
            size_invariant_area = (attribute == 'application_area' and all(x[0]['size_only'] for x in entries))
            category = ' '.join(s.text for s in target_sources if s.location.startswith('category_group'))
            if re.search(r'VITAMIN|HERBAL|MEDICINE|BABY HEALTH', category, re.I):
                size_invariant_area = False
            if not anchors and not size_invariant_area:
                continue
            # No quantities, regulated claims or variant attributes enter this route.
            if re.search(r'\d|safe|suitable|baby|child|pregnan|allergen|treat|dose', str(fact.value), re.I):
                continue
            identifier = evidence.source('family_shared', f"catalogue/{candidate['record_id']}/{source.location}",
                source.text, confidence=.93, exact=False, eligible=True,
                metadata={'allowed_field': fact.field, 'allowed_path': fact.path,
                          'allowed_value': fact.value, 'consensus_ids': sorted(distinct),
                          'target_anchor_ids': anchors, 'size_only_application_area': size_invariant_area,
                          'basis': 'target_corroboration' if anchors else 'same_named_product_different_size'})
            evidence.add(fact.field, fact.path, fact.value, identifier, source.text, .93, method='family_shared')
