#!/usr/bin/env bash
# Peluncur run penuh di instance GPU. Tiap job = satu sesi tmux, log di logs/<job>.log
cd /workspace/code
export PATH=/venv/main/bin:$PATH HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 PYTHONUNBUFFERED=1
job() { tmux new -d -s "$1" "python scripts/$2 > logs/$1.log 2>&1"; echo "started $1"; }
job lc_tox21      "08_run_learning_curve.py --dataset tox21"
job lc_sider      "08_run_learning_curve.py --dataset sider"
job pre_primary   "09_run_pretrained.py --dataset sider --mode primary   --models chemberta chemprop"
job pre_multitask "09_run_pretrained.py --dataset sider --mode multitask --models chemberta chemprop"
job pre_lc_tox21  "09_run_pretrained.py --dataset tox21 --mode lc        --models chemberta chemprop"
job pre_lc_sider  "09_run_pretrained.py --dataset sider --mode lc        --models chemberta chemprop"
# MoLFormer: satu sesi berantai (AMP off, lebih berat) agar tidak lebih dari 7 job GPU bersamaan
tmux new -d -s pre_molformer "python scripts/09_run_pretrained.py --dataset sider --mode primary   --models molformer > logs/pre_primary_mf.log 2>&1; \
python scripts/09_run_pretrained.py --dataset sider --mode multitask --models molformer > logs/pre_multitask_mf.log 2>&1; \
python scripts/09_run_pretrained.py --dataset sider --mode lc        --models molformer > logs/pre_lc_sider_mf.log 2>&1; \
python scripts/09_run_pretrained.py --dataset tox21 --mode lc        --models molformer > logs/pre_lc_tox21_mf.log 2>&1"
