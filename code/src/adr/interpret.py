"""Token-level attribution (Integrated Gradients on the embedding) + atom highlighting."""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from .features import tokenize


def integrated_gradients(model, x, length, task=0, steps=32):
    """Returns attribution per token position (sum over embedding dims) for one molecule."""
    model.eval(); dev = next(model.parameters()).device
    x = torch.as_tensor(x[None], device=dev); L = torch.as_tensor([int(length)], device=dev)
    emb = model.embed(x).detach()
    base = torch.zeros_like(emb)
    total = torch.zeros_like(emb)
    for a in torch.linspace(0, 1, steps, device=dev):
        e = (base + a * (emb - base)).requires_grad_(True)
        logit = model(x, L, emb=e)[0, task]
        g, = torch.autograd.grad(logit, e)
        total += g
    attr = ((emb - base) * total / steps).sum(-1)[0].detach().cpu().numpy()
    return attr[: int(length)]


def atom_attributions(smiles, tokens, attr):
    """Map token attributions onto RDKit atom indices (atom tokens appear in SMILES order)."""
    mol = Chem.MolFromSmiles(smiles)
    atom_attr, ai = [], 0
    for t, a in zip(tokens, attr):
        if t[0].isalpha() or t.startswith("["):
            atom_attr.append(float(a)); ai += 1
    assert mol is None or ai == mol.GetNumAtoms(), f"token/atom mismatch for {smiles}"
    return mol, np.array(atom_attr)


def draw_attribution(smiles, tokens, attr, path, size=(450, 350)):
    mol, aa = atom_attributions(smiles, tokens, attr)
    if mol is None:
        return
    s = np.abs(aa).max() or 1.0
    colors = {i: ((1.0, 1 - v / s, 1 - v / s) if v > 0 else (1 - abs(v) / s, 1 - abs(v) / s, 1.0))
              for i, v in enumerate(aa)}
    d = Draw.MolDraw2DCairo(*size)
    d.DrawMolecule(mol, highlightAtoms=list(colors), highlightAtomColors=colors)
    d.FinishDrawing()
    with open(path, "wb") as f:
        f.write(d.GetDrawingText())
