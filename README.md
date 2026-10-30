# SIDER ADR benchmark

Code, per-fold results and figure/table generators for the study

> **A controlled benchmark of SMILES-based recurrent networks, molecular fingerprints and pretrained chemical language models for adverse drug reaction prediction on SIDER**
> Ananda Fitri Karimah, Angel Metanosa Afinda, Isman Kurniawan (School of Computing, Telkom University). Manuscript in preparation.

The study compares, under one evaluation protocol (stratified and Bemis–Murcko scaffold 5-fold cross-validation repeated over three seeds, paired Wilcoxon tests with Holm correction):

* three input encodings (integer index, one-hot, learned embedding) and 15 recurrent architectures on SMILES token sequences,
* five hyper-parameter optimisers (random search, TPE, genetic algorithm, particle swarm optimisation, monarch butterfly optimisation) under an identical budget of 100 trainings and three search seeds, re-evaluated with cross-validation on both splits,
* classical learners (logistic regression, SVM, random forest, XGBoost) on ECFP4 fingerprints and RDKit descriptors,
* fine-tuned ChemBERTa-2 and MoLFormer-XL and the Chemprop D-MPNN,
* generalisation over the 27 SIDER organ classes, paired learning curves on SIDER and Tox21 (NR-AhR), threshold-dependent metrics, and Integrated Gradients attributions.

Primary task: *Blood and lymphatic system disorders* (SIDER, 1,347 molecules after standardisation, 855 positives).

## Repository layout

```
code/
  config.yaml              all settings (data, splits, models, optimisers, learning curves, pretrained models)
  requirements.txt         pinned Python dependencies (CPU pipeline)
  src/adr/                 library: data standardisation and splits, tokenisation and features, models,
                           baselines, HPO, metrics, statistics, learning curves, pretrained models, attribution
  scripts/01 ... 09        the experiment pipeline, in execution order (see code/README.md)
  data/processed/          standardised SIDER (27 organ classes) and Tox21 tables used in the study
  results/                 per-fold results of every training (CSV) and the data reports (JSON)
  figures/                 diagnostic figures produced by scripts 06 and 07
manuscript/
  make_tables.py           generates every table of the article (tables/*.tex) and the numbers quoted in the text
  make_figures.py          generates the publication figures (figures/*.pdf|png) from code/results
  tables/, figures/        the generated tables and figures
```

## Result files (code/results)

| File | Content | Rows |
|---|---|---|
| `baselines.csv` | trivial baselines and classical learners, both splits, all metrics at 0.5 and at the validation-tuned threshold | one per (model, split, seed, fold) |
| `dl_untuned.csv` | recurrent networks with default hyper-parameters (3 encodings, 12 configurations, 3 variants), both splits | 540 trainings |
| `hpo_trace.csv` | every evaluation of the five optimisers (3 seeds × 100 trainings each) | 1,500 trainings |
| `hpo_final.csv` | cross-validated re-evaluation of the 15 selected configurations on both splits | 450 trainings |
| `hpo_best_config.json` | configuration with the highest validation AUROC in the benchmark | |
| `pretrained_primary.csv`, `pretrained_tasks.csv` | ChemBERTa-2, MoLFormer-XL and Chemprop on the primary task and on the 27 classes | |
| `all_tasks.csv` | majority, RF-ECFP4, single-task and multi-label recurrent networks on all 27 organ classes | |
| `lc_sider.csv`, `lc_tox21.csv` | paired learning curves (7 models × sizes × 15 folds) | 1,200 trainings |
| `training_curves.csv` | per-epoch training and validation loss of the three encodings | |
| `attributions_top.csv`, `token_importance.csv` | Integrated Gradients results for the 12 most confident predictions | |
| `data_report.json`, `data_report_tox21.json` | standardisation statistics and class counts | |

Every CSV contains the split, seed and fold of each estimate, so all paired tests of the article can be recomputed.

## Reproducing the study

```bash
cd code
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_prepare_data.py            # downloads SIDER/Tox21 from MoleculeNet and standardises them
.venv/bin/python scripts/02_run_baselines.py
.venv/bin/python scripts/03_run_dl.py
.venv/bin/python scripts/04_run_hpo.py --method random  # likewise tpe, ga, pso, mbo
.venv/bin/python scripts/04b_run_hpo_scaffold.py
.venv/bin/python scripts/05_run_all_tasks.py
.venv/bin/python scripts/06_interpret.py --top 12
.venv/bin/python scripts/08_run_learning_curve.py --dataset sider   # and tox21
.venv/bin/python scripts/08b_training_curves.py
.venv/bin/python scripts/08c_run_lc_tuned.py
.venv/bin/python scripts/09_run_pretrained.py --dataset sider --mode primary --models chemberta molformer chemprop
.venv/bin/python scripts/07_summarize.py
```

Scripts 01 to 08 run on CPU (the study used an 8-core server). Script 09 needs a GPU and, in addition to `requirements.txt`, `transformers` and `chemprop` (versions 5.20 and 2.3.1 were used). Add `--quick` to any training script for a smoke test. Tables and figures of the article are regenerated with

```bash
cd manuscript && make tables && python make_figures.py
```

## Data

SIDER 4.1 (27 system organ classes) and Tox21 are downloaded by script 01 from the MoleculeNet distribution and standardised with RDKit (largest fragment, neutralisation, canonical SMILES, molecules longer than 200 tokens removed). The standardised tables are included under `code/data/processed` for convenience and remain subject to the licences of the original datasets.

## Licence and citation

Code is released under the MIT licence. Please cite the article (reference to be added on publication) and this repository (see `CITATION.cff`).
