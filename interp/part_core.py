"""ParT (Particle Transformer, JetClass, 10 classes) helpers for the interp experiments: the test-jet sample, the physics-edit
primitives on the jet dict (kin, build, jmass, sub, cat), the angular moments, and the attention hooks.

copied from distill-part/jld-features research/discovery (same author): the header of G192_part_physics_edits.py (PRE, files,
fof, PK, kin, build, jmass, sub, cat, read_root usage) and the attention hooks of G201_part_groups.py (forward(Jb, capture, M)
with the pre-hook that sets need_weights and adds a mask); _lund.py (lund) and _wintop.py (move_w, masses) sit next to this file.

Working command (the venv of jld-features, read-only: torch 2.14.1 + weaver-core 0.4.17, MPS works; see run_part.sh):
  cd /Users/anrunw/Documents/aaron-interp && \
  JETDISTILL_TAGGER=part KERAS_BACKEND=jax JETDISTILL_DEVICE=mps \
  JETDISTILL_DATA=/Users/anrunw/Documents/distill-part/jet-latent-distillation/data_part \
  JETDISTILL_MODELS=/Users/anrunw/Documents/distill-part/jet-latent-distillation/models_part \
  AI_RESULTS=results PYTHONPATH=/Users/anrunw/Documents/aaron-interp \
  /Users/anrunw/Documents/distill-part/jld-features/.venv/bin/python interp/<script>.py
Environment variables the package reads: jetdistill/config.py — JETDISTILL_TAGGER (must be 'part' here: CLASSES, NETWORKS and
the path defaults switch on it), JETDISTILL_DATA, JETDISTILL_MODELS, JETDISTILL_RESULTS, JETDISTILL_SMOKE;
jetdistill/part/network.py — JETDISTILL_DEVICE ('cpu' default; 'mps' or 'cuda'); jetdistill/part/data.py reads nothing else
(its read_root needs uproot + awkward). KERAS_BACKEND is not used by the ParT path (torch); it is set so that the JEDI modules
import cleanly when a script uses both. AI_RESULTS is this repo's results root (results/part/ for ParT).
The test ROOT files are listed in DATA/test/files.json with paths relative to the jld-features working directory
('data_part/raw/test/<sample>_100.root'); fof() resolves them against DATA.parent.

Jet dict (data.read_root): every part_* branch as (J, 128) float32, hardest first, zero-padded, plus 'mask', jet_pt, jet_eta,
jet_phi, jet_energy, 'label'. deta / dphi are relative to the jet axis.
"""
import json
import os
from pathlib import Path
import numpy as np
from jetdistill.config import DATA, CLASSES as CL
from jetdistill.part.data import read_root  # noqa: F401  (re-exported)
from jetdistill.part.network import ParTNetwork, inputs  # noqa: F401  (re-exported)

assert os.environ.get('JETDISTILL_TAGGER') == 'part', 'run with JETDISTILL_TAGGER=part (see the docstring / run_part.sh)'
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
PK = ['px', 'py', 'pz', 'energy', 'deta', 'dphi', 'd0val', 'd0err', 'dzval', 'dzerr', 'charge', 'isChargedHadron', 'isNeutralHadron',
      'isPhoton', 'isElectron', 'isMuon', 'mask']
N98 = 97            # the class-token W/Z neuron "n98" of the G-series (1-based there): 0-based index 97 of z. G5b's 13 active
#                     neurons 17, 98, 69, 19, 25, 105, 121, 116, 79, 53, 82, 84, 125 are the indices 16, 97, 68, … here; index 98 is
#                     constant (s.d. 0.0000). Wqq +2.06, Zqq −0.67 per s.d. through the last layer; falls with mass (G80).
ACTIVE = [16, 97, 68, 18, 24, 104, 120, 115, 78, 52, 81, 83, 124]   # 0-based, in G5b's strength order


def files():
    return json.loads((DATA / 'test' / 'files.json').read_text())


def fof(c):
    """the first test file of class c (…_100.root), resolved to an existing path"""
    f = next(x['file'] for x in files() if x['file'].split('/')[-1].startswith(PRE[c] + '_1'))
    for p in (Path(f), DATA.parent / f, DATA / 'raw' / 'test' / Path(f).name):
        if p.exists(): return p
    raise FileNotFoundError(f)


def sample(per_class=400, start=3000, classes=CL):
    """test jets: rows start .. start + per_class of each class's first test file (G192: 400 per class from row 3,000)"""
    return cat([read_root(fof(c), start=start, stop=start + per_class) for c in classes])


def kin(J):
    """(pT, Δη, Δφ, mass) of the particles, (J, 128) each, 0 in padded slots"""
    m = J['mask'] > 0; pt = np.hypot(J['px'], J['py']) * m; p2 = J['px'] ** 2 + J['py'] ** 2 + J['pz'] ** 2
    return pt, J['deta'] * m, J['dphi'] * m, np.sqrt(np.maximum(J['energy'] ** 2 - p2, 0)) * m


def build(J, pt, deta, dphi, mass, extra=None):
    """new jet dict from per-particle (pt, Δη, Δφ, mass) and the other per-particle arrays of J (or `extra`, same keys);
    sorted by pT, ≤ 128 slots, padded slots (pt = 0) zeroed. eta = jet_eta + sign(jet_eta)·Δη, phi = jet_phi + Δφ (JetClass
    stores part_deta with the jet's η sign factored out: checked on 1,000 test jets, max deviation 1.2e-7; G192's version used
    jet_eta + Δη, which mirrors the particles of jets with jet_eta < 0 in η and changed 0.9 % of ParT_full's calls on unedited
    jets); px, py, pz, E rebuilt; jet_pt = |Σ p_T| and jet_energy = Σ E recomputed from the particles."""
    F = {k: (extra[k] if extra is not None else J[k]).copy() for k in PK if k not in ('px', 'py', 'pz', 'energy', 'deta', 'dphi')}
    F['mask'] = (pt > 0).astype(np.float32); o = np.argsort(-pt, 1, kind='stable')[:, :128]; tk = lambda a: np.take_along_axis(a, o, 1)
    pt, deta, dphi, mass = tk(pt), tk(deta), tk(dphi), tk(mass); F = {k: tk(v) for k, v in F.items()}; m = F['mask'] > 0
    sgn = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]
    eta = J['jet_eta'][:, None] + sgn * deta; phi = J['jet_phi'][:, None] + dphi
    E = dict(F); E.update(px=np.where(m, pt * np.cos(phi), 0), py=np.where(m, pt * np.sin(phi), 0), pz=np.where(m, pt * np.sinh(eta), 0),
                          deta=np.where(m, deta, 0), dphi=np.where(m, dphi, 0))
    E['energy'] = np.where(m, np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2), 0)
    E = {k: v.astype(np.float32) for k, v in E.items()}
    E['jet_eta'], E['jet_phi'], E['label'] = J['jet_eta'], J['jet_phi'], J['label']
    E['jet_pt'] = np.hypot(E['px'].sum(1), E['py'].sum(1)).astype(np.float32); E['jet_energy'] = E['energy'].sum(1).astype(np.float32); return E


def jmass(J):
    """invariant mass of the real particles' four-vector sum [GeV]"""
    m = J['mask'] > 0; E = (J['energy'] * m).sum(1); px, py, pz = ((J[k] * m).sum(1) for k in ('px', 'py', 'pz'))
    return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))


def sub(J, rows):
    return {k: (v[rows] if np.ndim(v) else v) for k, v in J.items()}


def cat(Js):
    return {k: np.concatenate([J[k] for J in Js]) for k in Js[0]}


def moments(pt, deta, dphi):
    """angular moments about the given axis, as in E6: M0 = Σ pT, M1 = Σ pT r⃗, M2 = Σ pT r², |M1|, and √(M0·M2 − |M1|²)"""
    pt, deta, dphi = (np.asarray(a, np.float64) for a in (pt, deta, dphi))
    M0 = pt.sum(1); M1e, M1p = (pt * deta).sum(1), (pt * dphi).sum(1); M2 = (pt * (deta ** 2 + dphi ** 2)).sum(1)
    return dict(M0=M0, M1e=M1e, M1p=M1p, M1=np.hypot(M1e, M1p), M2=M2, M0M2=M0 * M2, m2a=M0 * M2 - M1e ** 2 - M1p ** 2)


class Attention:
    """attention weights of the 8 particle blocks and an additive attention mask (G201). forward(Jb, capture, M) ->
    (logits (J, 10), [A_block (J, P, P) averaged over the 8 heads] if capture else [], P). M: (J, 128, 128) float32 added to
    every block's attention logits (−1e9 blocks a pair), or None. remove() takes the hooks off."""

    def __init__(self, net, H=8):
        import torch
        self.torch, self.net, self.model, self.dev, self.H = torch, net, net.model, net.device, H
        self.CAP = dict(on=False, A=[]); self.MASK = dict(M=None); self.model.eval()
        self.hooks = [b.attn.register_forward_pre_hook(self._pre, with_kwargs=True) for b in self.model.blocks] + \
                     [b.attn.register_forward_hook(self._post) for b in self.model.blocks]

    def _pre(self, mod, args, kwargs):
        kwargs['need_weights'] = self.CAP['on']; kwargs['average_attn_weights'] = False
        if self.MASK['M'] is not None and kwargs.get('attn_mask') is not None:
            am = kwargs['attn_mask']; NH, P, _ = am.shape; N = NH // self.H
            kwargs['attn_mask'] = (am.view(N, self.H, P, P) + self.MASK['M'][:, None, :P, :P]).view(NH, P, P)
        return args, kwargs

    def _post(self, mod, args, out):
        if self.CAP['on']: self.CAP['A'].append(out[1].detach().float().mean(1).cpu().numpy())

    def forward(self, Jb, capture=False, M=None):
        torch = self.torch; x, v, m = inputs(Jb, self.net.n); self.CAP['on'] = capture; self.CAP['A'] = []
        self.MASK['M'] = None if M is None else torch.from_numpy(np.asarray(M, np.float32)).to(self.dev)
        with torch.no_grad(): L = self.model(*(torch.from_numpy(t).to(self.dev) for t in (x, v, m))).float().cpu().numpy()
        self.CAP['on'] = False; self.MASK['M'] = None; return L, self.CAP['A'], x.shape[-1]

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks = []
