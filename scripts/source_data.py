"""Read locally acquired MDC and DSTC1 sources without extracting archives."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import tarfile

DOMAINS = ('movie', 'restaurant', 'taxi', 'dstc1')
FIELDS = ['example_id', 'domain', 'dialogue_id', 'turn_id', 'speaker', 'text', 'context', 'politeness_label', 'split']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def with_context(domain, dialogue_id, turns):
    history = []
    for turn_id, speaker, text in turns:
        if not isinstance(text, str):
            raise ValueError(f'{dialogue_id}: text must be a string')
        record = dict(example_id=f'{domain}:{dialogue_id}:{turn_id}', domain=domain,
                      dialogue_id=dialogue_id, turn_id=turn_id, speaker=speaker,
                      text=text, context=list(history))
        # Non-speech/empty targets are not classification examples.
        if text.strip():
            yield record
            history.append(dict(turn_id=turn_id, speaker=speaker, text=text))


def read_mdc(root, domain):
    path = Path(root) / 'mdc' / 'data' / f'{domain}_all.tsv'
    dialogues = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            dialogue_id, turn_id = row['session.ID'], int(row['Message.ID'])
            speaker = row['Message.From']
            if speaker not in ('user', 'agent'):
                raise ValueError(f'Unknown MDC speaker: {speaker}')
            dialogues[dialogue_id].append((turn_id, speaker, row['Message.Text']))
    for dialogue_id, turns in dialogues.items():
        ordered = sorted(turns, key=lambda value: value[0])
        yield from with_context(domain, dialogue_id,
                                ((index, speaker, text) for index, (_, speaker, text) in enumerate(ordered, 1)))


def read_dstc1(root):
    for subset in ('train3', 'test3'):
        sessions = defaultdict(dict)
        with tarfile.open(Path(root) / 'dstc1' / f'{subset}.tgz', 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(('/dstc.log.json', '/dstc.labels.json')):
                    continue
                value = json.load(archive.extractfile(member))
                kind = 'labels' if member.name.endswith('/dstc.labels.json') else 'log'
                if kind == 'log':
                    value = dict(**{'session-id': value['session-id']}, turns=[
                        {'turn-index': turn['turn-index'], 'text': turn['output'].get('transcript', '')}
                        for turn in value['turns']])
                sessions[member.name.rsplit('/', 1)[0]][kind] = value
        for name, parts in sorted(sessions.items()):
            if set(parts) != {'log', 'labels'}:
                raise ValueError(f'Missing transcription or log: {name}')
            log, labels = parts['log'], parts['labels']
            if log['session-id'] != labels['session-id']:
                raise ValueError(f'Session mismatch: {name}')
            label_turns = {int(turn['turn-index']): turn for turn in labels['turns']}
            turns = []
            for turn in sorted(log['turns'], key=lambda value: value['turn-index']):
                index = int(turn['turn-index'])
                turns.append((2 * index, 'agent', turn['text']))
                transcription = label_turns[index].get('transcription', '')
                if transcription.strip().lower() in ('[silence]', '[noise]', '[unintelligible]', '[hang up]', '++noise++'):
                    transcription = ''
                turns.append((2 * index + 1, 'user', transcription))
            yield from with_context('dstc1', f'{subset}/{log["session-id"]}', turns)


def read_sources(root, domains=DOMAINS):
    for domain in domains:
        if domain not in DOMAINS:
            raise ValueError(f'Unknown domain {domain}')
        yield from read_dstc1(root) if domain == 'dstc1' else read_mdc(root, domain)
