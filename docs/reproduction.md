# Running PACD experiments

This repository implements the hierarchical politeness classifier described in *Predicting Politeness Variations in Goal-Oriented Conversations*. The committed result tables are transcribed from the manuscript. Training this implementation on the new annotations produces a separate experiment; agreement with the original results is not established.

## Inputs and context

Prepare local data with the instructions in the [dataset card](../datasets/README.md). The training interface accepts CSV files with a dialogue identifier, turn identifier, domain, speaker, text, politeness label, preceding context, and split. The context field is a JSON list of objects such as:

```json
[{"turn_id": 0, "speaker": "agent", "text": "How can I help?"}]
```

Context must contain only earlier turns of the same dialogue. Split assignments must keep entire dialogues together, so shared conversation history cannot leak between training and evaluation. The loader checks preceding-turn order, duplicate identifiers and turns, valid labels, and dialogue-disjoint splits. The class mapping is `0=impolite`, `1=somewhat_impolite`, `2=somewhat_polite`, `3=polite`.

## Hierarchical model

The model uses a shared word-level Transformer to encode individual utterances, takes the EOS representation of each utterance, adds utterance position information, and applies a context Transformer before the four-way classifier. It uses sinusoidal positions and class-weighted cross-entropy. The vocabulary is constructed from training data.

| Setting | [hierarchical.json](../configs/hierarchical.json) | [hierarchical_small.json](../configs/hierarchical_small.json) |
| --- | ---: | ---: |
| Word embedding dimensions | 300 | 300 |
| Hidden dimensions | 768 | 256 |
| Attention heads | 12 | 8 |
| Utterance encoder layers | 12 | 2 |
| Context encoder layers | 2 | 2 |
| Feed-forward dimensions | 3,072 | 1,024 |
| Maximum tokens per utterance, including EOS | 128 | 128 |
| Maximum turns, including the current utterance | 32 | 32 |

These are implementation configurations. The smaller configuration is useful for lower-resource runs. Neither is asserted to recover the paper's unspecified architecture settings or its reported parameter count. Training uses AdamW, and command-line options override configuration values.

```bash
python -m pacd train \
  --data .local/data/movie.csv \
  --config configs/hierarchical.json \
  --output runs/movie
```

To initialize word embeddings from a locally obtained GloVe text file, pass `--glove`:

```bash
python -m pacd train \
  --data .local/data/movie.csv \
  --config configs/hierarchical.json \
  --glove /path/to/glove.txt \
  --output runs/movie-glove
```

Use embeddings with the dimensionality selected in the configuration. Without `--glove`, embeddings are learned from random initialization. GloVe files are not bundled.

The manuscript selects learning rate `4e-5`, batch size `8`, and `2` epochs. Its printed search covers learning rates `1e-5` through `4e-5`, batch sizes `8`, `16`, `32`, and epoch counts `2`, `3`, `4`; only the 2- and 4-epoch tables are supplied. The corresponding [CSV](results/hyperparameters.csv) contains only the printed configurations.

## Pretrained baselines

Install baseline dependencies, then select one of the supported model names:

```bash
python -m pip install -e '.[baselines]'
python -m pacd train \
  --data .local/data/movie.csv \
  --model distilbert \
  --output runs/movie-distilbert
```

The supported names are `albert`, `distilbert`, and `distilroberta`. Model weights and tokenizers must be available locally or downloadable when first used. The manuscript uses ALBERT base v1, DistilBERT base uncased, and DistilRoBERTa base. These are utterance classification baselines. Speaker roles are retained in the dataset metadata; the models consume utterance text and, for the hierarchy, turn order. No separate speaker embedding is added.

## In-domain and transfer evaluation

The `evaluate` command selects the test split from the supplied corpus:

```bash
python -m pacd evaluate \
  --checkpoint runs/movie/best.pt \
  --data .local/data/restaurant.csv \
  --split test \
  --output runs/movie/restaurant.json
```

Supplying `movie.csv` evaluates within-domain; supplying another domain evaluates transfer without further training. To train on all three MDC domains, provide all three files:

```bash
python -m pacd train \
  --data .local/data/movie.csv .local/data/restaurant.csv .local/data/taxi.csv \
  --config configs/hierarchical.json \
  --output runs/mdc

python -m pacd evaluate \
  --checkpoint runs/mdc/best.pt \
  --data .local/data/dstc1.csv \
  --split test \
  --output runs/mdc/dstc1.json
```

For the reverse direction, train on DSTC1 and evaluate on the three MDC files.

## Five-fold evaluation

```bash
python -m pacd cross-validate \
  --data .local/data/movie.csv \
  --config configs/hierarchical.json \
  --folds 5 \
  --output runs/movie-cv
```

Folds are grouped by dialogue. The command combines the supplied rows and replaces their original split assignments with five outer folds; 10.5% of the remaining dialogues are reserved for validation within each fold. Cross-validation and the supplied fixed splits are different experiment protocols; the manuscript does not specify how its five-fold averaging interacts with the fixed split table. Record which protocol is used with every score.

## Saved outputs

Each training run saves `best.pt`, resolved `config.json`, example IDs in `splits.json`, per-epoch `history.json`, and `validation_metrics.json`. Baseline runs also save their tokenizer; keep `best.pt` alongside its `tokenizer/` directory when moving a baseline checkpoint. Use a new output directory for each run. Evaluation writes a metrics JSON file; add `--predictions /path/to/predictions.json` to save example IDs, predicted labels, and class probabilities. Cross-validation writes per-fold outputs plus `cross_validation.json` containing the fold scores, means, and standard deviations.

## Metric conventions

Evaluation reports accuracy, macro and support-weighted F1, the confusion matrix, and per-class precision, recall, specificity, geometric mean, and index-balanced accuracy. Aggregate GM and IBA are also exposed. For each class, treated as one-versus-rest:

```text
GM = sqrt(recall × specificity)
IBA = (1 + alpha × (recall − specificity)) × GM²
alpha = 0.1
```

Checkpoint selection uses validation macro F1. The manuscript does not identify the aggregate F1/GM/IBA convention or the IBA alpha value in its result tables. These explicit implementation conventions should therefore be reported with new experiments; the CSV column `f1` retains the manuscript's unqualified metric name.

## Manuscript ambiguities

The transcribed values are preserved exactly; the following issues affect a literal reproduction.

| Topic | What the manuscript says | Consequence |
| --- | --- | --- |
| Word encoder | Methodology specifies GloVe embeddings, sinusoidal positions, and EOS pooling; implementation details describe a 12-layer, 768-hidden, 12-head BERT model with 110M parameters. | The two descriptions do not determine one architecture. This implementation exposes its choices in configuration files. |
| Architecture details | Separate utterance/context layer counts, context-window length, and several training settings are unspecified. | Repository defaults are implementation choices, not recovered experimental settings. |
| Label counts | Restaurant classes sum to 7,132, versus Phase 2 total 7,133; DSTC1 classes sum to 10,143, versus Phase 2 total 10,144. Aggregated MDC class counts sum to 21,504, versus Phase 2 total 21,505. | Plots use the printed class totals. No extra label is inferred. |
| DSTC1 splits | Text says 70%/19.5%/10.5% train/test/validation for both datasets. The DSTC1 counts are approximately 70%/20%/10%. | Preserve the count table and report actual splits for new experiments. |
| Dominant DSTC1 class | Transfer discussion names class 1 as dominant; the distribution table and figure make class 2 the largest, at 4,374 instances. | Distribution plots follow the numeric table. |
| Cross-validation | Five-fold averages and fixed train/test/validation partitions are both described. Their relationship is unspecified. | Fixed-split and grouped cross-validation results are separate protocols here. |
| Metric aggregation | F1, GM, and IBA are named without complete aggregation settings. | New evaluation output names its averaging and IBA settings explicitly. |
| Search cost | The stated grid has 36 combinations, but the runtime calculation multiplies each epoch duration by 36. Another paragraph reports substantially shorter training times. | The repository preserves printed losses/F1; it does not claim those runtimes. |

The paper's original annotation procedure used classifier-assisted annotation followed by human verification and majority voting. Its annotator agreement and gold-standard claims describe that original study. They do not describe the new annotation release in this repository.

## Regenerate manuscript figures

```bash
python -m pip install matplotlib
python scripts/plot_results.py --formats png svg
```

The script reads only the committed [manuscript CSV tables](results/README.md). Every figure identifies its values as manuscript-reported. Use a separate output location for figures from newly trained models.
