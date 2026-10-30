# Experiment pipeline

All scripts are run from this directory with the virtual environment created from `requirements.txt`. Every setting is in `config.yaml`. Add `--quick` to a training script for a smoke test (few epochs, one fold).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Scripts in execution order

| Script | Output | Purpose |
|---|---|---|
| `01_prepare_data.py` | `data/processed/*.csv`, `results/data_report*.json` | download SIDER and Tox21 from MoleculeNet, RDKit standardisation, token statistics, trivial-baseline F1 |
| `02_run_baselines.py` | `results/baselines.csv` | majority and random predictors, logistic regression, SVM, random forest and XGBoost on ECFP4 and RDKit descriptors |
| `03_run_dl.py` | `results/dl_untuned.csv` | encoding ablation (index, one-hot, embedding), 12 recurrent configurations (L, CL, LD, CLD × 1–3 layers), BiLSTM, GRU and CNN-GRU variants |
| `04_run_hpo.py --method {random,tpe,ga,pso,mbo}` | `results/hpo_trace.csv`, `hpo_final.csv`, `hpo_best_config.json` | optimiser benchmark with an identical budget, cross-validated re-evaluation of the selected configurations (stratified split) |
| `04b_run_hpo_scaffold.py` | appends to `results/hpo_final.csv` | re-evaluation of the selected configurations on the scaffold split |
| `05_run_all_tasks.py` | `results/all_tasks.csv` | majority, RF-ECFP4, single-task and multi-label recurrent networks on the 27 organ classes |
| `06_interpret.py --top 12` | `figures/attribution/*.png`, `results/attributions_top.csv`, `token_importance.csv` | Integrated Gradients on the most confident predictions of the best configuration |
| `07_summarize.py` | `results/summary.md`, `figures/*.png` | summary tables, paired tests and diagnostic figures |
| `08_run_learning_curve.py --dataset {sider,tox21}` | `results/lc_*.csv` | paired learning curves for RF, XGBoost and the default recurrent network |
| `08b_training_curves.py` | `results/training_curves.csv` | per-epoch losses of the three encodings |
| `08c_run_lc_tuned.py` | appends to `results/lc_*.csv` | learning curve of the PSO-selected configuration |
| `09_run_pretrained.py --dataset ... --mode {primary,multitask,lc} --models ...` | `results/pretrained_*.csv`, appends to `lc_*.csv` | ChemBERTa-2, MoLFormer-XL (HuggingFace) and Chemprop v2 (CLI); needs a GPU, `transformers` and `chemprop` |

`scripts/pull_results.sh` copies results from a remote server (`status`, `pull`, `summary`), and `run_all.sh` is the tmux launcher used on the GPU instance.

## Evaluation protocol

* Outer splits: stratified 5-fold and Bemis–Murcko scaffold 5-fold, each repeated with three seeds (15 test estimates per model and split).
* Inside every training fold, 15 % of the molecules form a validation subset used for early stopping, the HPO objective and threshold selection. Test folds are used only for the final evaluation.
* Metrics: AUROC, AUPRC, MCC, balanced accuracy, accuracy, F1, precision, recall and specificity, at the threshold 0.5 and at the threshold that maximises the MCC on the validation subset (`*_tuned` columns).
* Recurrent networks: class-weighted binary cross-entropy, Adam, gradient clipping 1.0, early stopping with patience 10, packed sequences.
* Statistics: two-sided Wilcoxon signed-rank test paired on (seed, fold), paired t-test and Cohen's d, Holm correction within each family of comparisons (`src/adr/stats.py`).

## Library

```
src/adr/data.py        download, standardisation, stratified and scaffold splits
src/adr/features.py    SMILES tokeniser (atom-level regex or characters), ECFP4, RDKit descriptors
src/adr/models_dl.py   SeqClassifier (L/CL/LD/CLD, LSTM/GRU, bidirectional, three input encodings), training loop
src/adr/baselines.py   logistic regression, SVM, random forest, XGBoost
src/adr/hpo.py         search space, objective, random / TPE / GA / PSO / MBO wrappers
src/adr/metrics.py     metrics, trivial baselines, threshold selection
src/adr/stats.py       paired tests, Holm correction, summary tables
src/adr/lc.py          paired subsampling for learning curves
src/adr/pretrained.py  fine-tuning of chemical language models, Chemprop driver
src/adr/interpret.py   Integrated Gradients and atom highlighting
src/adr/utils.py       I/O helpers (schema-aligned CSV appends)
```

Changing `data.task` in `config.yaml` selects another organ class as the primary task, `hpo.schemes` selects the schemes to tune, and `src/adr/hpo.py` holds the search space.
