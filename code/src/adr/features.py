"""Molecular representations: SMILES tokenisation (char / atom-level), ECFP, RDKit descriptors."""
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# Standard SMILES regex (Schwaller et al. 2019) -> multi-character atoms kept intact (Cl, Br, [nH], @@ ...)
_SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")


def tokenize(smi: str, level="atom"):
    if level == "char":
        return list(smi)
    toks = _SMI_REGEX.findall(smi)
    assert "".join(toks) == smi, f"Regex tokeniser lost characters for {smi}"
    return toks


class SmilesTokenizer:
    """Maps SMILES to padded integer sequences. index 0 = PAD, 1 = UNK."""
    PAD, UNK = 0, 1

    def __init__(self, level="atom", max_len=None):
        self.level, self.max_len = level, max_len
        self.vocab = {"<pad>": 0, "<unk>": 1}

    def fit(self, smiles):
        lens = []
        for s in smiles:
            toks = tokenize(s, self.level); lens.append(len(toks))
            for t in toks:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
        if self.max_len is None:
            self.max_len = max(lens)
        self.itos = {i: t for t, i in self.vocab.items()}
        return self

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, smiles) -> np.ndarray:
        X = np.zeros((len(smiles), self.max_len), dtype=np.int64)
        for i, s in enumerate(smiles):
            ids = [self.vocab.get(t, self.UNK) for t in tokenize(s, self.level)][: self.max_len]
            X[i, : len(ids)] = ids
        return X

    def lengths(self, X):
        return (X != self.PAD).sum(1)


def ecfp(smiles, radius=2, n_bits=2048, counts=False) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = gen.GetCountFingerprintAsNumPy(mol) if counts else gen.GetFingerprintAsNumPy(mol)
        out[i] = fp
    return out


def rdkit_descriptors(smiles) -> tuple[np.ndarray, list]:
    names = [n for n, _ in Descriptors.descList]
    fns = [f for _, f in Descriptors.descList]
    out = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        for j, f in enumerate(fns):
            try:
                out[i, j] = f(mol)
            except Exception:
                pass
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1e6, 1e6).astype(np.float32)
    return out, names


def randomized_smiles(smi: str, n: int, seed=0) -> list:
    """SMILES enumeration (augmentation): n random non-canonical SMILES for the same molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi] * n
    rng = np.random.RandomState(seed)
    out = set()
    for _ in range(n * 5):
        out.add(Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True))
        if len(out) >= n:
            break
    out = list(out)
    return out + [smi] * (n - len(out)) if len(out) < n else out[:n]
