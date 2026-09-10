#!/usr/bin/env python3
"""Verify released label files, checksums, distributions, and dialogue splits."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from source_data import DOMAINS, sha256

HEADER = ['example_id', 'domain', 'dialogue_id', 'turn_id', 'speaker', 'politeness_label', 'split', 'paper_subset']


def verify(dataset_dir):
    root = Path(dataset_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('format_version') != 1 or manifest.get('status') != 'complete':
        raise ValueError('Expected a complete format-version 1 annotation release')
    if Counter(info['domain'] for info in manifest['files']) != Counter(DOMAINS):
        raise ValueError('Release must contain exactly one label file for each of the four domains')
    seen, groups, totals = set(), {}, {}
    for info in manifest['files']:
        path = root / info['path']
        if path.parent.resolve() != root.resolve():
            raise ValueError('Release paths must be plain filenames')
        if sha256(path) != info['sha256'] or path.stat().st_size != info['bytes']:
            raise ValueError(f'{path.name}: checksum/size mismatch')
        counts, selected, splits = Counter(), Counter(), Counter()
        dialogue_ids = set()
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != HEADER:
                raise ValueError(f'{path.name}: unexpected schema')
            for row in reader:
                if None in row or any(value is None or value == '' for value in row.values()):
                    raise ValueError(f'{path.name}: malformed row')
                if row['domain'] != info['domain'] or row['politeness_label'] not in ('0', '1', '2', '3'):
                    raise ValueError(f'{path.name}: invalid domain/label')
                label = int(row['politeness_label'])
                if row['speaker'] not in ('agent', 'user') or int(row['turn_id']) < 0:
                    raise ValueError(f'{path.name}: invalid speaker/turn')
                if row['split'] not in ('train', 'validation', 'test') or row['paper_subset'] not in ('0', '1'):
                    raise ValueError(f'{path.name}: invalid split/selection')
                if row['example_id'] != f'{row["domain"]}:{row["dialogue_id"]}:{row["turn_id"]}':
                    raise ValueError(f'{path.name}: inconsistent identifier')
                if row['example_id'] in seen:
                    raise ValueError('Duplicate example ID')
                seen.add(row['example_id'])
                group = row['domain'], row['dialogue_id']
                dialogue_ids.add(row['dialogue_id'])
                if groups.get(group, row['split']) != row['split']:
                    raise ValueError('Dialogue leakage across partitions')
                groups[group] = row['split']
                counts[str(label)] += 1
                splits[row['split']] += 1
                if row['paper_subset'] == '1':
                    selected[str(label)] += 1
        expected = {str(i): counts[str(i)] for i in range(4)}
        subset = {str(i): selected[str(i)] for i in range(4)}
        if sum(counts.values()) != info['rows'] or expected != info['label_counts'] or dict(splits) != info['split_counts']:
            raise ValueError(f'{path.name}: measured counts do not match manifest')
        if len(dialogue_ids) != info.get('dialogues'):
            raise ValueError(f'{path.name}: dialogue count does not match manifest')
        if subset != info['paper_subset_counts']:
            raise ValueError(f'{path.name}: subset counts do not match manifest')
        if info['paper_target_matched'] != (subset == info['paper_target_counts']):
            raise ValueError(f'{path.name}: target-match flag is inaccurate')
        shortfalls = {str(i): max(0, info['paper_target_counts'][str(i)] - counts[str(i)])
                      for i in range(4)}
        if shortfalls != info.get('paper_target_shortfalls'):
            raise ValueError(f'{path.name}: manuscript target shortfalls do not match measured counts')
        totals[info['domain']] = dict(rows=info['rows'], labels=expected, paper_target_matched=info['paper_target_matched'])
    if set(totals) != {'movie', 'restaurant', 'taxi', 'dstc1'}:
        raise ValueError('Release must cover all four domains')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'datasets')
    print(json.dumps(verify(parser.parse_args().dataset_dir), indent=2))


if __name__ == '__main__':
    main()
