"""Hyper-parameter optimisation benchmark under an EQUAL evaluation budget.

Methods: random, tpe (Optuna), mbo / ga / pso / de (NiaPy). All methods see the same
search space, the same objective (1 - validation AUROC) and the same number of model trainings.
Every evaluation is logged (params, fitness, wall-clock) so convergence curves can be drawn.
"""
import time, math
import numpy as np
from .models_dl import train_model, predict_proba
from .metrics import compute_metrics

# ------------------------------------------------------------------ search space
# Each entry: (type, low, high) ; type in {int, float, logfloat, cat}
SPACE = {
    "n_layers":     ("int", 1, 3),
    "units":        ("int", 32, 128),
    "conv_filters": ("int", 32, 128),
    "kernel_size":  ("int", 2, 20),
    "dense_units":  ("int", 32, 128),
    "emb_dim":      ("int", 16, 128),
    "dropout":      ("float", 0.0, 0.5),
    "lr":           ("logfloat", 1e-4, 1e-2),
    "batch_size":   ("cat", [16, 32, 64]),
    "bidirectional": ("cat", [False, True]),
}


def active_params(scheme: str):
    """Only parameters that exist for the given scheme are searched (fair dimensionality)."""
    keys = ["n_layers", "units", "emb_dim", "dropout", "lr", "batch_size", "bidirectional"]
    if scheme.startswith("C"):
        keys += ["conv_filters", "kernel_size"]
    if scheme.endswith("D"):
        keys += ["dense_units"]
    return keys


def decode(vec, keys):
    """Map a vector in [0,1]^d to a concrete parameter dict."""
    p = {}
    for v, k in zip(vec, keys):
        t = SPACE[k]
        v = float(np.clip(v, 0, 1 - 1e-9))
        if t[0] == "int":
            p[k] = int(t[1] + math.floor(v * (t[2] - t[1] + 1)))
        elif t[0] == "float":
            p[k] = t[1] + v * (t[2] - t[1])
        elif t[0] == "logfloat":
            p[k] = float(math.exp(math.log(t[1]) + v * (math.log(t[2]) - math.log(t[1]))))
        elif t[0] == "cat":
            p[k] = t[1][int(v * len(t[1]))]
    return p


def to_model_cfg(params: dict, scheme: str, base: dict) -> dict:
    cfg = {**base, "scheme": scheme}
    cfg["rnn_units"] = [params["units"]] * params["n_layers"]
    for k in ("conv_filters", "kernel_size", "dense_units", "emb_dim", "dropout", "lr",
              "batch_size", "bidirectional"):
        if k in params:
            cfg[k] = params[k]
    return cfg


# ------------------------------------------------------------------ objective
class Objective:
    """Callable fitness: trains the model on (train) and returns 1 - AUROC on (val). Logs everything."""

    def __init__(self, data, vocab_size, scheme, base_cfg, seed=0, device="cpu"):
        self.d, self.vocab_size, self.scheme, self.base, self.seed, self.device = data, vocab_size, scheme, base_cfg, seed, device
        self.keys = active_params(scheme)
        self.dim = len(self.keys)
        self.log = []          # one row per evaluation
        self.t0 = time.perf_counter()
        self.best = (np.inf, None)

    def __call__(self, vec) -> float:
        params = decode(vec, self.keys)
        cfg = to_model_cfg(params, self.scheme, self.base)
        X_tr, L_tr, y_tr, X_va, L_va, y_va = self.d
        t = time.perf_counter()
        model, info = train_model(X_tr, L_tr, y_tr, X_va, L_va, y_va, self.vocab_size, cfg,
                                  seed=self.seed, device=self.device)
        p = predict_proba(model, X_va, L_va)[:, 0]
        m = compute_metrics(y_va, p)
        fit = 1.0 - m["auroc"]
        if fit < self.best[0]:
            self.best = (fit, cfg)
        self.log.append({"eval": len(self.log) + 1, "fitness": fit, "val_auroc": m["auroc"],
                         "val_auprc": m["auprc"], "val_loss": info["best_val_loss"], "epochs": info["epochs"],
                         "eval_time": time.perf_counter() - t, "elapsed": time.perf_counter() - self.t0,
                         "best_so_far": self.best[0], **params})
        return fit


# ------------------------------------------------------------------ optimisers
def run_random(obj: Objective, budget: int, seed=0):
    rng = np.random.RandomState(seed)
    for _ in range(budget):
        obj(rng.rand(obj.dim))
    return obj.best


def run_tpe(obj: Objective, budget: int, seed=0):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda tr: obj([tr.suggest_float(k, 0, 1) for k in obj.keys]), n_trials=budget)
    return obj.best


def run_niapy(obj: Objective, budget: int, method: str, seed=0, population_size=20, **kw):
    from niapy.problems import Problem
    from niapy.task import Task
    from niapy.algorithms.basic import (MonarchButterflyOptimization, GeneticAlgorithm,
                                        ParticleSwarmAlgorithm, DifferentialEvolution)

    class P(Problem):
        def __init__(self):
            super().__init__(dimension=obj.dim, lower=0.0, upper=1.0)
        def _evaluate(self, x):
            return obj(x)

    algos = {
        "mbo": lambda: MonarchButterflyOptimization(population_size=population_size,
                                                    partition=kw.get("partition", 5.0 / 12.0),
                                                    period=kw.get("period", 1.2), seed=seed),
        "ga":  lambda: GeneticAlgorithm(population_size=population_size, crossover_rate=0.8, mutation_rate=0.1, seed=seed),
        "pso": lambda: ParticleSwarmAlgorithm(population_size=population_size, seed=seed),
        "de":  lambda: DifferentialEvolution(population_size=population_size, seed=seed),
    }
    task = Task(problem=P(), max_evals=budget)
    try:
        algos[method]().run(task)
    except StopIteration:
        pass
    return obj.best


def run_hpo(method: str, obj: Objective, budget: int, seed=0, **kw):
    method = method.lower()
    if method == "random":
        return run_random(obj, budget, seed)
    if method == "tpe":
        return run_tpe(obj, budget, seed)
    if method in ("mbo", "ga", "pso", "de"):
        return run_niapy(obj, budget, method, seed, **kw)
    raise ValueError(method)
