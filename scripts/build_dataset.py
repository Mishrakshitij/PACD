#!/usr/bin/env python3
"""Join released politeness labels with locally acquired dialogue sources."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import tempfile
import os
from source_data import DOMAINS, FIELDS, read_sources, sha256
from verify_dataset import HEADER


def build(source_dir, output_dir, dataset_dir, domains=DOMAINS, selection='full'):
    source_dir, output_dir, dataset_dir = map(Path, (source_dir, output_dir, dataset_dir))
    manifest = json.loads((dataset_dir / 'manifest.json').read_text())
    if manifest.get('format_version') != 1 or manifest.get('status') != 'complete':
        raise ValueError('Expected a complete format-version 1 annotation release')
    if any(Path(item['path']).name != item['path'] for item in manifest['files']):
        raise ValueError('Annotation paths must be plain filenames')
    required_sources = set()
    for domain in domains:
        if sum(item['domain'] == domain for item in manifest['files']) != 1:
            raise ValueError(f'Expected exactly one label file for {domain}')
        required_sources.update(['dstc1/train3.tgz', 'dstc1/test3.tgz'] if domain == 'dstc1'
                                else [f'mdc/data/{domain}_all.tsv'])
    covered_sources = {item['path'] for item in manifest['sources']
                       if set(item['domains']).intersection(domains)}
    if not required_sources <= covered_sources:
        raise ValueError('Manifest is missing required source checksums')
    input_paths = {(dataset_dir / 'manifest.json').resolve()}
    input_paths.update((dataset_dir / item['path']).resolve() for item in manifest['files'])
    input_paths.update((source_dir / item['path']).resolve() for item in manifest['sources'])
    if any((output_dir / f'{domain}.csv').resolve() in input_paths for domain in domains):
        raise ValueError('Output files must not overwrite source data or released annotations')
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in manifest['sources']:
        if set(source['domains']).intersection(domains):
            path = source_dir / source['path']
            if sha256(path) != source['sha256']:
                raise ValueError(f'Source checksum mismatch: {source["path"]}; use the release-linked version')
    output_counts = {}
    for domain in domains:
        info = next(item for item in manifest['files'] if item['domain'] == domain)
        if selection == 'paper' and not info['paper_target_matched']:
            raise ValueError(f'{domain}: manuscript class targets are not met; use --selection full')
        label_path = dataset_dir / info['path']
        if sha256(label_path) != info['sha256']:
            raise ValueError(f'Annotation checksum mismatch: {label_path}')
        with label_path.open(newline='', encoding='utf-8') as handle:
            labels, seen, group_splits = {}, set(), {}
            reader = csv.DictReader(handle)
            if reader.fieldnames != HEADER:
                raise ValueError(f'{label_path}: unexpected annotation schema')
            for label in reader:
                if None in label or any(value is None or value == '' for value in label.values()):
                    raise ValueError(f'{label_path}: malformed annotation row')
                key = label['example_id']
                if key in seen:
                    raise ValueError(f'Duplicate annotation ID: {key}')
                seen.add(key)
                if (label['domain'] != domain or label['politeness_label'] not in ('0', '1', '2', '3')
                        or label['split'] not in ('train', 'validation', 'test')
                        or label['paper_subset'] not in ('0', '1')
                        or label['speaker'] not in ('agent', 'user') or int(label['turn_id']) < 0):
                    raise ValueError(f'Invalid annotation values: {key}')
                if key != f"{domain}:{label['dialogue_id']}:{label['turn_id']}":
                    raise ValueError(f'Inconsistent annotation identifier: {key}')
                group = label['dialogue_id']
                if group_splits.get(group, label['split']) != label['split']:
                    raise ValueError('Dialogue leakage across annotation splits')
                group_splits[group] = label['split']
                if selection == 'full' or label['paper_subset'] == '1':
                    labels[key] = label
        if not labels:
            raise ValueError(f'No {selection} annotations available for {domain}')
        descriptor, temporary = tempfile.mkstemp(prefix=f'.{domain}-', suffix='.csv', dir=output_dir)
        count = 0
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, FIELDS)
                writer.writeheader()
                for row in read_sources(source_dir, (domain,)):
                    label = labels.pop(row['example_id'], None)
                    if label is None:
                        continue
                    for key in ('domain', 'dialogue_id', 'speaker'):
                        if row[key] != label[key]:
                            raise ValueError(f'Source/annotation mismatch for {row["example_id"]}: {key}')
                    if int(label['turn_id']) != row['turn_id']:
                        raise ValueError(f'Turn mismatch for {row["example_id"]}')
                    row['context'] = json.dumps(row['context'], ensure_ascii=False, separators=(',', ':'))
                    row['politeness_label'], row['split'] = label['politeness_label'], label['split']
                    writer.writerow(row)
                    count += 1
            if labels:
                raise ValueError(f'{domain}: {len(labels)} annotations could not be joined to source text')
            os.replace(temporary, output_dir / f'{domain}.csv')
            output_counts[domain] = count
        finally:
            Path(temporary).unlink(missing_ok=True)
    return output_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path,
                        help='Directory containing mdc/data/*_all.tsv and dstc1/train3.tgz, test3.tgz')
    parser.add_argument('--output-dir', type=Path, default=Path('.local/data'))
    parser.add_argument('--dataset-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'datasets')
    parser.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument('--selection', choices=['full', 'paper'], default='full',
                        help='Use every released label or the subset targeting manuscript class counts')
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir, args.dataset_dir, args.domains, args.selection), indent=2))


if __name__ == '__main__':
    main()
