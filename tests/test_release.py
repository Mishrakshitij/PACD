"""Check source identity preservation and release joins without upstream data."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from source_data import DOMAINS, read_mdc, read_dstc1, sha256
from build_dataset import build
from verify_dataset import HEADER, verify


class SourceTests(unittest.TestCase):
    def test_duplicate_source_turn_ids_preserve_both_utterances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'mdc/data').mkdir(parents=True)
            path = root / 'mdc/data/movie_all.tsv'
            with path.open('w', newline='', encoding='utf-8-sig') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['session.ID', 'Message.ID', 'Message.From', 'Message.Text'])
                writer.writerows([['1', '2', 'agent', 'Second'], ['1', '1', 'user', 'First'],
                                  ['1', '2', 'agent', 'Third'], ['2', '1', 'user', 'Fourth']])
            rows = list(read_mdc(root, 'movie'))
            self.assertEqual([row['text'] for row in rows], ['First', 'Second', 'Third', 'Fourth'])
            self.assertEqual(len({row['example_id'] for row in rows}), 4)
            self.assertEqual([turn['text'] for turn in rows[2]['context']], ['First', 'Second'])
            self.assertEqual(rows[3]['context'], [])

    def test_dstc_joins_transcriptions_without_extracting_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dstc1').mkdir()
            for subset in ('train3', 'test3'):
                with tarfile.open(root / 'dstc1' / f'{subset}.tgz', 'w:gz') as archive:
                    documents = {
                        f'{subset}/dialogue/dstc.log.json': {
                            'session-id': 'dialogue', 'turns': [
                                {'turn-index': 0, 'output': {'transcript': 'Where to?'},
                                 'input': {'live': {'asr-hyps': [{'asr-hyp': 'WRONG'}]}}}]},
                        f'{subset}/dialogue/dstc.labels.json': {
                            'session-id': 'dialogue', 'turns': [
                                {'turn-index': 0, 'transcription': 'Museum'}]},
                        '../never-extract.json': {'unsafe': True}}
                    for name, value in documents.items():
                        data = json.dumps(value).encode()
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
            rows = list(read_dstc1(root))
            self.assertEqual([row['text'] for row in rows], ['Where to?', 'Museum'] * 2)
            self.assertEqual(len({row['example_id'] for row in rows}), 4)
            self.assertEqual(rows[1]['context'][0]['turn_id'], 0)
            self.assertFalse((root / 'never-extract.json').exists())

    def test_hydration_checks_sources_and_does_not_publish_partial_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'mdc/data').mkdir(parents=True)
            data = root / 'dataset'
            data.mkdir()
            source = root / 'mdc/data/movie_all.tsv'
            source.write_text('session.ID\tMessage.ID\tMessage.From\tMessage.Text\n1\t1\tuser\tHello\n')
            labels = data / 'movie.csv'
            with labels.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, HEADER)
                writer.writeheader()
                writer.writerow(dict(example_id='movie:1:1', domain='movie', dialogue_id='1', turn_id=1,
                                     speaker='user', politeness_label=2, split='train', paper_subset=1))
            manifest = {'format_version': 1, 'status': 'complete', 'sources': [{'domains': ['movie'], 'path': 'mdc/data/movie_all.tsv', 'sha256': sha256(source)}],
                        'files': [{'domain': 'movie', 'path': 'movie.csv', 'sha256': sha256(labels)}]}
            (data / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'must not overwrite'):
                build(root, data, data, ['movie'])
            self.assertEqual(build(root, root / 'output', data, ['movie']), {'movie': 1})
            output = root / 'output/movie.csv'
            original = output.read_bytes()
            source.write_text(source.read_text().replace('Hello', 'Changed'))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                build(root, root / 'output', data, ['movie'])
            self.assertEqual(output.read_bytes(), original)


class VerificationTests(unittest.TestCase):
    def make_release(self, root):
        manifest = {'format_version': 1, 'status': 'complete', 'files': []}
        records = {}
        for domain in DOMAINS:
            rows = [dict(example_id=f'{domain}:{dialogue}:{turn}', domain=domain,
                         dialogue_id=dialogue, turn_id=str(turn), speaker=speaker,
                         politeness_label=str(label), split=split, paper_subset='0')
                    for dialogue, turn, speaker, label, split in (
                        ('a', 0, 'user', 0, 'train'), ('a', 1, 'agent', 1, 'train'),
                        ('b', 0, 'user', 2, 'test'), ('c', 0, 'user', 3, 'validation'))]
            records[domain] = rows
            manifest['files'].append({
                'domain': domain, 'path': f'{domain}.csv', 'rows': 4, 'dialogues': 3,
                'label_counts': {'0': 1, '1': 1, '2': 1, '3': 1},
                'split_counts': {'train': 2, 'test': 1, 'validation': 1},
                'paper_target_counts': {'0': 2, '1': 1, '2': 1, '3': 1},
                'paper_subset_counts': {'0': 0, '1': 0, '2': 0, '3': 0},
                'paper_target_matched': False,
                'paper_target_shortfalls': {'0': 1, '1': 0, '2': 0, '3': 0}})
        self.write_release(root, manifest, records)
        return manifest, records

    def write_release(self, root, manifest, records):
        for info in manifest['files']:
            path = root / info['path']
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, HEADER)
                writer.writeheader()
                writer.writerows(records[info['domain']])
            info['sha256'], info['bytes'] = sha256(path), path.stat().st_size
        (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def test_valid_release_counts_dialogues_and_available_class_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_release(root)
            totals = verify(root)
            self.assertEqual(set(totals), set(DOMAINS))
            self.assertTrue(all(info['rows'] == 4 for info in totals.values()))

    def test_rejects_duplicate_missing_and_unknown_domain_files(self):
        for mutation in ('duplicate', 'missing', 'unknown'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _ = self.make_release(root)
                if mutation == 'duplicate':
                    manifest['files'].append(dict(manifest['files'][0], path='another.csv'))
                elif mutation == 'missing':
                    manifest['files'].pop()
                else:
                    manifest['files'][0]['domain'] = 'unknown'
                (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'exactly one label file'):
                    verify(root)

    def test_rejects_noncanonical_label_strings_even_with_matching_checksum(self):
        for label in ('00', '+0', ' 0', '0.0'):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, records = self.make_release(root)
                records['movie'][0]['politeness_label'] = label
                self.write_release(root, manifest, records)
                with self.assertRaisesRegex(ValueError, 'invalid domain/label'):
                    verify(root)

    def test_rejects_incorrect_or_missing_dialogue_count(self):
        for declared in (4, None):
            with self.subTest(declared=declared), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, records = self.make_release(root)
                if declared is None:
                    del manifest['files'][0]['dialogues']
                else:
                    manifest['files'][0]['dialogues'] = declared
                self.write_release(root, manifest, records)
                with self.assertRaisesRegex(ValueError, 'dialogue count'):
                    verify(root)

    def test_rejects_shortfalls_computed_from_selected_instead_of_available_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, records = self.make_release(root)
            manifest['files'][0]['paper_target_shortfalls'] = {'0': 2, '1': 1, '2': 1, '3': 1}
            self.write_release(root, manifest, records)
            with self.assertRaisesRegex(ValueError, 'shortfalls'):
                verify(root)

    def test_rejects_missing_shortfall_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, records = self.make_release(root)
            del manifest['files'][0]['paper_target_shortfalls']
            self.write_release(root, manifest, records)
            with self.assertRaisesRegex(ValueError, 'shortfalls'):
                verify(root)


if __name__ == '__main__':
    unittest.main()
