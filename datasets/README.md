# PACD dataset

Four-class politeness annotations for movie, restaurant, and taxi conversations
from the Microsoft Dialogue Challenge (MDC), and bus-information conversations
from DSTC1. The labels use the scale in
[Predicting Politeness Variations in Goal-Oriented Conversations](https://doi.org/10.1109/TCSS.2022.3156580):
`0=impolite`, `1=somewhat_impolite`, `2=somewhat_polite`, `3=polite`.

This is a new annotation release. The paper's original manual annotations,
annotator agreement, and reported experiment scores are separate from these
files. The numeric manuscript tables are preserved in
[docs/results](../docs/results/README.md).

## Release files

The release distributes labels, source identifiers, and dialogue-disjoint split
assignments. Dialogue text and context are reconstructed locally using the loader
below. MDC's [source license](https://github.com/xiul-msr/e2e_dialog_challenge/blob/05de23b4488350cb963dfc5592f27821e5171470/Human%20Dialogue%20Dataset%20License%20Terms.txt)
permits noncommercial research use and modification but prohibits redistribution
of the source data. DSTC1's downloaded archives contain their own noncommercial
license and attribution notices. Retain and follow each source's terms when
working with the hydrated data.

**Release 1.0.0: 113,903 annotated utterances.**

| Domain | Annotated utterances | Dialogues | Impolite | Somewhat impolite | Somewhat polite | Polite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Movie | 21,656 | 2,890 | 3,576 | 3,088 | 9,299 | 5,693 |
| Restaurant | 29,718 | 4,103 | 6,764 | 5,323 | 11,207 | 6,424 |
| Taxi | 23,311 | 3,094 | 6,134 | 3,430 | 8,725 | 5,022 |
| DSTC1 | 39,218 | 1,708 | 13,622 | 4,650 | 14,735 | 6,211 |

![Measured label counts in this release](../docs/figures/dataset_release.png)

The `paper_subset` selections contain **31,647 utterances** and match all four domains' printed class-count targets exactly. These selections are made from the released labels; they are not the original gold examples.

`manifest.json` records file hashes, measured counts, source checksums, and the
status of manuscript class-count targets. `summary.csv` gives the measured counts
in a compact table. Run:

```bash
python scripts/verify_dataset.py
```

The verifier checks every record, label, source ID, checksum, distribution,
selection flag, and dialogue assignment. It does not require source corpora or
model dependencies.

## Obtain source text

Download the source data under its upstream terms. These commands create the
layout expected by the loader; they do not assign labels.

```bash
mkdir -p .local/dstc1

git clone https://github.com/xiul-msr/e2e_dialog_challenge.git .local/mdc
git -C .local/mdc checkout 05de23b4488350cb963dfc5592f27821e5171470

curl --fail --location \
  'https://download.microsoft.com/download/B/4/5/B4502E67-FBE2-417E-9530-E9A9A990E3F3/dstc_data_train3_v00.tgz' \
  --output .local/dstc1/train3.tgz
curl --fail --location \
  'https://download.microsoft.com/download/F/0/A/F0AF24D9-34BC-4E17-9B0E-19E985F73DB7/dstc_data_test3.tgz' \
  --output .local/dstc1/test3.tgz

python scripts/build_dataset.py --source-dir .local --output-dir .local/data
```

The source checksum must match the manifest. Existing source or annotation files
cannot be overwritten by the loader. Archives are read directly without
extracting files. To build only one domain, add `--domains movie` or another
domain name.

The result is `.local/data/movie.csv`, `restaurant.csv`, `taxi.csv`, and
`dstc1.csv`, each containing the full training schema below. `.local/` is ignored
by Git. Follow the [training commands](../docs/reproduction.md) to use these files.

## Schema

The committed CSVs contain these columns:

| Column | Meaning |
| --- | --- |
| `example_id` | Unique key `<domain>:<dialogue_id>:<turn_id>` |
| `domain` | `movie`, `restaurant`, `taxi`, or `dstc1` |
| `dialogue_id` | MDC session ID, or DSTC1 subset/session ID |
| `turn_id` | Stable integer identifying the utterance within its dialogue |
| `speaker` | `user` or `agent` |
| `politeness_label` | Integer 0–3 using the scale above |
| `split` | `train`, `test`, or `validation` |
| `paper_subset` | 1 for a selected manuscript-count subset, 0 otherwise |

Hydration adds `text` and `context`, and omits `paper_subset` from the model input.
`context` is a JSON array of preceding turns with `turn_id`, `speaker`, and `text`.
It contains no labels or future turns.

In MDC, turns are stably ordered by original `Message.ID` and assigned sequential
IDs starting at 1. Repeated source message IDs remain separate records. In DSTC1,
source turn index `i` maps to agent utterance `2*i` and user utterance `2*i+1`.
Agent text is the system transcript; user text is the human transcription in
`dstc.labels.json`, rather than an ASR hypothesis. Empty targets are excluded.

## Splits and manuscript-count subsets

Splits keep every dialogue together, including its context. A stable hash assigns
dialogues to training, test, and validation with probabilities 70%, 19.5%, and
10.5%. Actual utterance counts vary with dialogue length and are recorded in the
manifest. These are new partitions, not recovered original split IDs.

Manuscript class-count targets use the printed class table exactly:

| Domain | Impolite | Somewhat impolite | Somewhat polite | Polite | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movie | 1,136 | 533 | 2,367 | 3,110 | 7,146 |
| Restaurant | 1,301 | 866 | 1,824 | 3,141 | 7,132 |
| Taxi | 1,285 | 674 | 1,787 | 3,480 | 7,226 |
| DSTC1 | 2,570 | 708 | 4,374 | 2,491 | 10,143 |

For domains marked `paper_target_matched: true` in the manifest:

```bash
python scripts/build_dataset.py --source-dir .local \
  --output-dir .local/paper-data --domains movie --selection paper
```

The loader rejects `--selection paper` for a domain whose quotas cannot be met.
The complete released labels remain available through `--selection full`, the
default.

## Scope and attribution

Politeness is context-dependent. Low labels include direct task wording and
slot-value answers; they do not alone establish hostility or user dissatisfaction.
The paper's cultural, demographic, social-power, and individual-variation
limitations apply when interpreting this scale. Use these labels for research,
not as a factual assessment of a speaker's character.

Cite the [PACD paper](../CITATION.bib), the
[Microsoft Dialogue Challenge](https://github.com/xiul-msr/e2e_dialog_challenge),
and the [DSTC1 source](https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/dstc1-downloads/)
when using the corresponding domains. The source dialogues remain attributed to
their original providers. Record this release version and whether you used the
full corpus or a manuscript-count subset in any experimental report.
