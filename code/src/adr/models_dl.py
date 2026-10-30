"""PyTorch sequence classifiers for SMILES (L / CL / LD / CLD schemes, LSTM/GRU, bidirectional)
with three input modes for the representation ablation:
  - 'index'     : raw integer token index as a single scalar feature (the original thesis pipeline, flawed)
  - 'onehot'    : one-hot vectors (vocab_size features per step)
  - 'embedding' : learnable embedding layer (recommended)
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


def get_device(pref="auto"):
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEFAULT_CFG = dict(
    scheme="CL",            # L | CL | LD | CLD
    input_mode="embedding", # index | onehot | embedding
    emb_dim=64,
    rnn_type="lstm",        # lstm | gru
    rnn_units=[64],         # one entry per recurrent layer
    bidirectional=False,
    conv_filters=64, kernel_size=5,
    dense_units=64,
    dropout=0.2,
    lr=1e-3, weight_decay=1e-5, batch_size=32,
    max_epochs=100, patience=10,
    class_weight=True,      # pos_weight = n_neg / n_pos in BCE
)


class SeqClassifier(nn.Module):
    def __init__(self, vocab_size, n_tasks=1, **cfg):
        super().__init__()
        c = {**DEFAULT_CFG, **cfg}
        self.cfg, self.vocab_size, self.n_tasks = c, vocab_size, n_tasks
        self.input_mode = c["input_mode"]
        scheme = c["scheme"].upper()
        self.use_conv, self.use_dense = scheme.startswith("C"), scheme.endswith("D")

        if self.input_mode == "embedding":
            self.emb = nn.Embedding(vocab_size, c["emb_dim"], padding_idx=0); in_dim = c["emb_dim"]
        elif self.input_mode == "onehot":
            in_dim = vocab_size
        elif self.input_mode == "index":
            in_dim = 1
        else:
            raise ValueError(c["input_mode"])

        if self.use_conv:
            self.conv = nn.Conv1d(in_dim, c["conv_filters"], c["kernel_size"], padding="same")
            in_dim = c["conv_filters"]

        rnn_cls = nn.LSTM if c["rnn_type"] == "lstm" else nn.GRU
        self.rnns = nn.ModuleList()
        for h in c["rnn_units"]:
            self.rnns.append(rnn_cls(in_dim, h, batch_first=True, bidirectional=c["bidirectional"]))
            in_dim = h * (2 if c["bidirectional"] else 1)
        self.drop = nn.Dropout(c["dropout"])
        if self.use_dense:
            self.dense = nn.Linear(in_dim, c["dense_units"]); in_dim = c["dense_units"]
        self.out = nn.Linear(in_dim, n_tasks)

    def embed(self, x):
        if self.input_mode == "embedding":
            return self.emb(x)
        if self.input_mode == "onehot":
            return F.one_hot(x, self.vocab_size).float()
        return (x.float() / self.vocab_size).unsqueeze(-1)   # 'index' mode (ablation)

    def forward(self, x, lengths, emb=None):
        h = self.embed(x) if emb is None else emb            # (B, T, F)
        if self.use_conv:
            h = F.relu(self.conv(h.transpose(1, 2))).transpose(1, 2)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        for i, rnn in enumerate(self.rnns):
            packed, state = rnn(packed)
            if i < len(self.rnns) - 1:
                packed = PackedSequence(self.drop(packed.data), packed.batch_sizes,
                                        packed.sorted_indices, packed.unsorted_indices)
        h_n = state[0] if isinstance(state, tuple) else state   # (dirs, B, H)
        feat = torch.cat([h_n[-2], h_n[-1]], 1) if self.cfg["bidirectional"] else h_n[-1]
        feat = self.drop(feat)
        if self.use_dense:
            feat = self.drop(F.relu(self.dense(feat)))
        return self.out(feat)


def _batches(X, L, Y, bs, shuffle, rng):
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield X[j], L[j], (Y[j] if Y is not None else None)


def train_model(X_tr, L_tr, Y_tr, X_va, L_va, Y_va, vocab_size, cfg, seed=0, device="auto", verbose=False):
    """Train with Adam + early stopping on validation loss. Y may be (N,) or (N, n_tasks); NaN = missing label."""
    cfg = {**DEFAULT_CFG, **cfg}
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    dev = get_device(device)
    Y_tr = np.asarray(Y_tr, dtype=np.float32).reshape(len(X_tr), -1)
    Y_va = np.asarray(Y_va, dtype=np.float32).reshape(len(X_va), -1)
    n_tasks = Y_tr.shape[1]
    model = SeqClassifier(vocab_size, n_tasks, **cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["class_weight"]:
        pos = np.nansum(Y_tr, 0); neg = np.sum(~np.isnan(Y_tr), 0) - pos
        pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.1, 10), dtype=torch.float32, device=dev)
    else:
        pos_weight = None

    def loss_fn(logits, y):
        mask = ~torch.isnan(y)
        y0 = torch.nan_to_num(y)
        l = F.binary_cross_entropy_with_logits(logits, y0, pos_weight=pos_weight, reduction="none")
        return (l * mask).sum() / mask.sum().clamp(min=1)

    def to_t(a, dtype=torch.long):
        return torch.as_tensor(a, dtype=dtype, device=dev)

    best, best_loss, bad, hist = None, np.inf, 0, []
    t0 = time.perf_counter()
    for ep in range(cfg["max_epochs"]):
        model.train(); tr_loss = 0.0
        for xb, lb, yb in _batches(X_tr, L_tr, Y_tr, cfg["batch_size"], True, rng):
            opt.zero_grad()
            loss = loss_fn(model(to_t(xb), to_t(lb)), to_t(yb, torch.float32))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(X_tr)
        va_loss = evaluate_loss(model, X_va, L_va, Y_va, loss_fn, cfg["batch_size"], dev)
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        if verbose:
            print(f"ep {ep:3d} train {tr_loss:.4f} val {va_loss:.4f}")
        if va_loss < best_loss - 1e-4:
            best_loss, best, bad = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best)
    model.eval()
    return model, {"history": hist, "best_val_loss": best_loss, "epochs": len(hist),
                   "train_time": time.perf_counter() - t0}


@torch.no_grad()
def evaluate_loss(model, X, L, Y, loss_fn, bs, dev):
    model.eval(); tot = 0.0
    for xb, lb, yb in _batches(X, L, Y, bs, False, None):
        tot += loss_fn(model(torch.as_tensor(xb, device=dev), torch.as_tensor(lb, device=dev)),
                       torch.as_tensor(yb, dtype=torch.float32, device=dev)).item() * len(xb)
    return tot / len(X)


@torch.no_grad()
def predict_proba(model, X, L, bs=256):
    model.eval(); dev = next(model.parameters()).device; out = []
    for xb, lb, _ in _batches(X, L, None, bs, False, None):
        out.append(torch.sigmoid(model(torch.as_tensor(xb, device=dev), torch.as_tensor(lb, device=dev))).cpu().numpy())
    return np.concatenate(out)


def config_from_name(name: str, base: dict) -> dict:
    """'CL1' -> scheme CL, 1 LSTM layer; 'L3' -> 3 layers etc. Uses base['rnn_units'][0] per layer."""
    import re
    m = re.fullmatch(r"(C?)L(\d)(D?)", name.upper())
    if not m:
        raise ValueError(name)
    n = int(m.group(2)); units = base.get("rnn_units", [64])[0]
    return {**base, "scheme": f"{m.group(1)}L{m.group(3)}", "rnn_units": [units] * n}
