"""Pretrained chemical language models (ChemBERTa-2, MoLFormer) fine-tuned with a masked multi-label
BCE head, and a thin wrapper around the Chemprop v2 CLI (directed message-passing GNN)."""
import copy, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np
import torch
import torch.nn.functional as F


def _device(pref="auto"):
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ HF fine-tuning
def finetune_lm(hf_id, smiles_tr, Y_tr, smiles_va, Y_va, max_len=160, lr=5e-5, batch_size=32,
                max_epochs=30, patience=4, seed=0, device="auto", class_weight=True, verbose=False, amp=True):
    """Returns (model, tokenizer, info). Y arrays: (N, n_tasks) float with NaN for missing labels."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device(device)
    Y_tr = np.asarray(Y_tr, np.float32).reshape(len(smiles_tr), -1); Y_va = np.asarray(Y_va, np.float32).reshape(len(smiles_va), -1)
    n_tasks = Y_tr.shape[1]
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=n_tasks, trust_remote_code=True,
                                                               problem_type="multi_label_classification").to(dev)
    enc = lambda s: tok(list(s), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    pos = np.nansum(Y_tr, 0); neg = np.sum(~np.isnan(Y_tr), 0) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.1, 10), dtype=torch.float32, device=dev) if class_weight else None

    def loss_fn(logits, y):
        mask = ~torch.isnan(y); l = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), pos_weight=pw, reduction="none")
        return (l * mask).sum() / mask.sum().clamp(min=1)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = int(np.ceil(len(smiles_tr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max_epochs * steps_per_epoch, pct_start=0.1)
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    rng = np.random.RandomState(seed); best, best_loss, bad, hist = None, np.inf, 0, []
    smiles_tr = np.asarray(smiles_tr); smiles_va = np.asarray(smiles_va); t0 = time.perf_counter()
    for ep in range(max_epochs):
        model.train(); order = rng.permutation(len(smiles_tr)); tr_loss = 0.0
        for i in range(0, len(order), batch_size):
            j = order[i:i + batch_size]; b = {k: v.to(dev) for k, v in enc(smiles_tr[j]).items()}
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(**b).logits.float()
            loss = loss_fn(logits, torch.as_tensor(Y_tr[j], device=dev))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tr_loss += loss.item() * len(j)
        va_logits = predict_lm(model, tok, smiles_va, max_len, batch_size, logits=True, amp=use_amp)
        va_loss = loss_fn(torch.as_tensor(va_logits, device=dev), torch.as_tensor(Y_va, device=dev)).item()
        hist.append({"epoch": ep, "train_loss": tr_loss / len(order), "val_loss": va_loss})
        if verbose:
            print(f"ep {ep} train {tr_loss/len(order):.4f} val {va_loss:.4f}")
        if va_loss < best_loss - 1e-4:
            best_loss, best, bad = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return model, tok, {"history": hist, "best_val_loss": best_loss, "epochs": len(hist), "train_time": time.perf_counter() - t0}


@torch.no_grad()
def predict_lm(model, tok, smiles, max_len=160, batch_size=64, logits=False, amp=True):
    model.eval(); dev = next(model.parameters()).device; out = []
    smiles = np.asarray(smiles)
    for i in range(0, len(smiles), batch_size):
        b = {k: v.to(dev) for k, v in tok(list(smiles[i:i + batch_size]), padding=True, truncation=True,
                                           max_length=max_len, return_tensors="pt").items()}
        with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
            z = model(**b).logits.float()
        out.append(z.cpu().numpy())
    z = np.concatenate(out)
    return z if logits else 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ Chemprop v2 CLI wrapper
def _chemprop_bin():
    cand = os.path.join(os.path.dirname(sys.executable), "chemprop")
    return cand if os.path.exists(cand) else (shutil.which("chemprop") or "chemprop")


def chemprop_fit_predict(smiles_tr, Y_tr, smiles_va, Y_va, smiles_te, task_names, epochs=60, patience=10,
                         hidden=300, depth=3, seed=0, workdir=None, gpu=True):
    """Train a Chemprop v2 (D-MPNN) multi-task classifier via its CLI using an explicit train/val split
    (--splits-column) and return (P_va, P_te, info); P arrays have shape (n, n_tasks)."""
    import pandas as pd
    from rdkit import Chem
    wd = tempfile.mkdtemp(prefix="chemprop_", dir=workdir); cp = _chemprop_bin()
    Y_tr = np.asarray(Y_tr, float).reshape(len(smiles_tr), -1); Y_va = np.asarray(Y_va, float).reshape(len(smiles_va), -1)
    ok = lambda arr: np.array([Chem.MolFromSmiles(str(x)) is not None for x in arr])
    k_tr, k_va, k_te = ok(smiles_tr), ok(smiles_va), ok(smiles_te)
    smiles_tr, Y_tr = np.asarray(smiles_tr)[k_tr], Y_tr[k_tr]
    smiles_va_all, Y_va_all = np.asarray(smiles_va), Y_va
    smiles_va, Y_va = smiles_va_all[k_va], Y_va[k_va]
    smiles_te_all = np.asarray(smiles_te); smiles_te = smiles_te_all[k_te]
    tn = list(task_names)
    d_tr = pd.DataFrame(Y_tr, columns=tn).assign(smiles=list(smiles_tr), split="train")
    d_va = pd.DataFrame(Y_va, columns=tn).assign(smiles=list(smiles_va), split="val")
    d_te = d_va.copy().assign(split="test")           # chemprop needs a test partition; real test preds come from `predict`
    pd.concat([d_tr, d_va, d_te])[["smiles", "split"] + tn].to_csv(f"{wd}/data.csv", index=False)
    pd.DataFrame({"smiles": list(smiles_va)}).to_csv(f"{wd}/val.csv", index=False)
    pd.DataFrame({"smiles": list(smiles_te)}).to_csv(f"{wd}/test.csv", index=False)
    acc = ["--accelerator", "gpu", "--devices", "1"] if (gpu and torch.cuda.is_available()) else ["--accelerator", "cpu"]
    t0 = time.perf_counter()
    cmd = [cp, "train", "--data-path", f"{wd}/data.csv", "--splits-column", "split", "--smiles-columns", "smiles",
           "--target-columns", *tn, "--task-type", "classification", "--save-dir", f"{wd}/out",
           "--epochs", str(epochs), "--patience", str(patience), "--message-hidden-dim", str(hidden), "--depth", str(depth),
           "--pytorch-seed", str(seed), "--data-seed", str(seed), *acc]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("chemprop train failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
    ckpts = [os.path.join(dp, f) for dp, _, fs in os.walk(f"{wd}/out") for f in fs if f == "best.pt"]
    if not ckpts:
        ckpts = [os.path.join(dp, f) for dp, _, fs in os.walk(f"{wd}/out") for f in fs if f.endswith(".pt") or f.endswith(".ckpt")]
    outs = []
    for name in ("val", "test"):
        r = subprocess.run([cp, "predict", "--test-path", f"{wd}/{name}.csv", "--smiles-columns", "smiles",
                            "--model-paths", ckpts[0], "--preds-path", f"{wd}/{name}_preds.csv", *acc], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("chemprop predict failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
        preds = pd.read_csv(f"{wd}/{name}_preds.csv")
        cols = [c for c in preds.columns if c != "smiles"]
        P = preds[cols].values.astype(float)
        keep = k_va if name == "val" else k_te                   # invalid SMILES get a non-informative 0.5
        full = np.full((len(keep), P.shape[1]), 0.5); full[keep] = P
        outs.append(full)
    shutil.rmtree(wd, ignore_errors=True)
    return outs[0], outs[1], {"train_time": time.perf_counter() - t0, "n_invalid": int((~k_tr).sum() + (~k_va).sum() + (~k_te).sum())}
