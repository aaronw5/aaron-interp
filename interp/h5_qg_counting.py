"""H5 (claim D7): do trained quark-gluon taggers count constituents? Collinear split (Delta N = +1) at fixed IRC-safe shapes,
k splits, remove-softest (Delta N = -1) and a count-fixed control, against the Casimir-meets-Poisson prediction
(N_q ~ Poisson(lam_q), N_g ~ Poisson(lam_g), lam_g / lam_q = 1.589 from H2 -> Delta log LR = log 1.589 = 0.463 per particle)
and the empirical LR(N) slope.

Models (QG, Komiske-Metodiev-Thaler Pythia set, interp.data_qg):
  part_kin, part_kinpid, pnet_kin, pnet_kinpid  weaver checkpoints of the H2 trainings (net_best_epoch_state.pt at run time:
                                                interim if the training is still running; the epoch is recorded)
  lorentznet                                   released LorentzNet QG weights (sdogsq/LorentzNet-release logs/qg/pretrained)
  efn_t, pfn_t                                 torch EFN (sum z_i Phi(deta, dphi), IRC-safe) and PFN (sum Phi(z, deta, dphi)),
                                                trained in this job on 400k train jets (stand-ins until the H2 energyflow QG
                                                models exist)
  lr_cls, lr_shape                             logistic regressions on (N, IRC-safe shapes) and on the shapes alone (classical
                                                counting positive control / classical IRC-safe zero control)
  part_kin_perm                                ParT kin with every weight tensor's entries permuted (null)
Output logit = log p(gluon) - log p(quark) for every model (positive = toward gluon).

Steps:
  --step hw      (CPU) download the public Herwig 7.1 QG set (Zenodo 3066475, energyflow 'herwig') files 0-1 into data/qg/herwig
  --step run     (GPU, part_job.sh) all edits on data_qg.sample(5000, 701), statistics -> $AI_RESULTS/physics/h5_run.json
  --step herwig  (GPU) the same models (unedited + split rank 0 / k splits) on 5000 Herwig jets per class -> h5_herwig.json
  --step fig     (CPU) figure results/physics/h5_counting.png
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
QG = V / 'data' / 'qg'
HW = QG / 'herwig'
POISSON = float(np.log(1.589))
L_IN = 128           # weaver input length (qg_*.yaml)
L_LN = 148           # LorentzNet padding length = max multiplicity of the energyflow Pythia QG set (H2: train max 148)
DR_SPLIT = 5e-4      # daughter separation of the collinear split (< 1e-3)
MASS = {11: 0.000511, 13: 0.105658, 22: 0.0, 130: 0.497611, 211: 0.139570, 321: 0.493677, 2112: 0.939565, 2212: 0.938272}
CH_HAD = (211, 321, 2212)


# ------------------------------------------------------------------------------------------------------------------ Herwig
def step_hw():
    import urllib.request
    HW.mkdir(parents=True, exist_ok=True)
    urls = ['https://zenodo.org/records/3066475/files/QG_jets_herwig_{i}.npz?download=1',
            'https://zenodo.org/record/3066475/files/QG_jets_herwig_{i}.npz?download=1']
    got = []
    for i in (0, 1):
        dst = HW / f'QG_jets_herwig_{i}.npz'
        if dst.exists() and dst.stat().st_size > 1e6: got.append(str(dst)); continue
        for u in urls:
            u = u.format(i=i)
            try:
                log('download', u)
                urllib.request.urlretrieve(u, str(dst) + '.part')
                os.replace(str(dst) + '.part', dst); got.append(str(dst)); log('  ok', dst.stat().st_size, 'bytes'); break
            except Exception as e:
                log('  failed:', repr(e))
    if not got:
        (HW / 'status.json').write_text(json.dumps(dict(ok=False, note='download failed'))); return
    X, y = [], []
    for f in got:                                     # one file at a time, float32, pT-sorted (memory)
        with np.load(f, allow_pickle=False) as d:
            x = d['X'].astype(np.float32); y.append(d['y'])
        o = np.argsort(-x[..., 0], axis=1, kind='stable'); X.append(np.take_along_axis(x, o[..., None], 1)); del x, o
    L = max(x.shape[1] for x in X)
    out = np.lib.format.open_memmap(HW / 'raw.npy', mode='w+', dtype=np.float32, shape=(sum(len(x) for x in X), L, 4)); s = 0
    for x in X: out[s:s + len(x), :x.shape[1]] = x; s += len(x)
    out.flush(); X = out; y = np.concatenate(y).astype(np.int8); np.save(HW / 'label.npy', y)
    n = (X[..., 0] > 0).sum(1)
    pids = np.unique(np.abs(X[..., 3][X[..., 0] > 0]).astype(int))
    info = dict(ok=True, files=got, J=int(len(y)), L=int(L), quark_frac=float(y.mean()), pids=pids.tolist(),
                N_mean_q=float(n[y == 1].mean()), N_mean_g=float(n[y == 0].mean()),
                N_ratio=float(n[y == 0].mean() / n[y == 1].mean()))
    (HW / 'status.json').write_text(json.dumps(info, indent=1)); log(info)


# ------------------------------------------------------------------------------------------------------- jet representation
def from_test(S):
    """sample dict (interp.data_qg) -> jets: dict pt, eta, phi (J, L) float64 absolute, pid (J, L) int, n (J,)"""
    F = S['full'].astype(np.float64); px, py, pz = F[..., 0], F[..., 1], F[..., 2]
    pt = np.hypot(px, py); real = pt > 0
    eta = np.where(real, np.arcsinh(np.where(real, pz / np.maximum(pt, 1e-30), 0)), 0)
    phi = np.where(real, np.arctan2(py, px), 0)
    return dict(pt=pt, eta=eta, phi=phi, pid=S['pid'].astype(np.int64) * real, n=real.sum(1))


def from_raw(X):
    """energyflow (pt, y, phi, pid) rows, pT-sorted -> jets (massless, eta = y, as the particle_transformer conversion)"""
    X = X.astype(np.float64); real = X[..., 0] > 0
    return dict(pt=X[..., 0] * real, eta=X[..., 1] * real, phi=X[..., 2] * real, pid=X[..., 3].astype(np.int64) * real, n=real.sum(1))


def to_list(Jt):
    return [np.stack([Jt['pt'][j, :Jt['n'][j]], Jt['eta'][j, :Jt['n'][j]], Jt['phi'][j, :Jt['n'][j]],
                      Jt['pid'][j, :Jt['n'][j]].astype(np.float64)], 1) for j in range(len(Jt['n']))]


def from_list(L, Lpad=None):
    n = np.array([len(a) for a in L]); Lp = Lpad or int(n.max()); J = len(L)
    out = {k: np.zeros((J, Lp)) for k in ('pt', 'eta', 'phi')}; pid = np.zeros((J, Lp), np.int64)
    for j, a in enumerate(L):
        a = a[np.argsort(-a[:, 0], kind='stable')]; m = min(len(a), Lp)
        out['pt'][j, :m], out['eta'][j, :m], out['phi'][j, :m] = a[:m, 0], a[:m, 1], a[:m, 2]; pid[j, :m] = a[:m, 3]
    out['pid'] = pid; out['n'] = np.minimum(n, Lp); out['n_full'] = n
    return out


def kinematics(Jt):
    """massless four-vectors and the jet axis (as utils/convert_qg_datasets.py): px, py, pz, E, deta, dphi, jet (pt, eta, E)"""
    pt, eta, phi = Jt['pt'], Jt['eta'], Jt['phi']; real = pt > 0
    px, py, pz, E = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    for a in (px, py, pz, E): a *= real
    jx, jy, jz, jE = px.sum(1), py.sum(1), pz.sum(1), E.sum(1)
    jpt = np.hypot(jx, jy); jeta = np.arcsinh(jz / jpt); jphi = np.arctan2(jy, jx)
    sg = np.sign(jeta); sg[sg == 0] = 1
    deta = (eta - jeta[:, None]) * sg[:, None] * real
    dphi = ((phi - jphi[:, None] + np.pi) % (2 * np.pi) - np.pi) * real
    return dict(px=px, py=py, pz=pz, E=E, deta=deta, dphi=dphi, real=real, jpt=jpt, jeta=jeta, jphi=jphi, jE=jE)


# -------------------------------------------------------------------------------------------------------------------- edits
def _split(a, i, z, rng, pid2=None):
    p = a[i]; th = rng.uniform(0, 2 * np.pi); u = np.array([np.cos(th), np.sin(th)])
    d1 = p.copy(); d2 = p.copy()
    d1[0], d2[0] = z * p[0], (1 - z) * p[0]
    d1[1:3] = p[1:3] + (1 - z) * DR_SPLIT * u; d2[1:3] = p[1:3] - z * DR_SPLIT * u   # pT-weighted centroid unchanged
    if pid2 is not None: d2[3] = pid2
    return np.concatenate([np.delete(a, i, 0), d1[None], d2[None]])


def edit_list(L, kind, rng, **kw):
    """apply one edit to every jet; returns (new list, applied mask)"""
    out, ok = [], np.ones(len(L), bool)
    for j, a in enumerate(L):
        n = len(a)
        if kind in ('rmsoft', 'merge', 'merge_split') and n < 3:   # ΔN = -1 / merge edits need >= 3 particles (no empty jets)
            out.append(a); ok[j] = False; continue
        if kind == 'split':
            r = kw['rank']; i = n - 1 if r == 'soft' else r
            if i >= n: out.append(a); ok[j] = False; continue
            out.append(_split(a, i, kw['z'], rng))
        elif kind == 'ksplit':                # jets with N < 5 are left out of every k (same jet set for all k)
            if n < 5: out.append(a); ok[j] = False; continue
            idx = np.sort(rng.choice(n, kw['k'], replace=False))[::-1]; b = a
            for i in idx: b = _split(b, int(i), 0.5, rng)
            out.append(b)
        elif kind == 'pidsplit':           # mode: ch_copy, ch_photon (charged hadron -> same + photon), ph_copy
            ap = np.abs(a[:, 3]).astype(int)
            cand = np.flatnonzero(np.isin(ap, CH_HAD)) if kw['mode'].startswith('ch') else np.flatnonzero(ap == 22)
            if len(cand) == 0: out.append(a); ok[j] = False; continue
            out.append(_split(a, int(cand[0]), 0.5, rng, pid2=22 if kw['mode'] == 'ch_photon' else None))
        elif kind == 'rmsoft':
            out.append(a[:-1])
        elif kind == 'merge_split':        # merge the two softest (pT-weighted position, pid of the harder), split the hardest
            s1, s2 = a[-2], a[-1]; m = s1.copy(); m[0] = s1[0] + s2[0]
            m[1:3] = (s1[0] * s1[1:3] + s2[0] * s2[1:3]) / m[0]
            out.append(_split(np.concatenate([a[:-2], m[None]]), 0, 0.5, rng))
        elif kind == 'merge':
            s1, s2 = a[-2], a[-1]; m = s1.copy(); m[0] = s1[0] + s2[0]
            m[1:3] = (s1[0] * s1[1:3] + s2[0] * s2[1:3]) / m[0]
            out.append(np.concatenate([a[:-2], m[None]]))
        else:
            raise ValueError(kind)
    return out, ok


def edits_spec():
    E = []
    for r in (0, 1, 5, 'soft'):
        for z in (0.5, 0.2):
            E.append((f'split_r{r}_z{z}', 'split', dict(rank=r, z=z), +1))
    for k in range(1, 6): E.append((f'ksplit_{k}', 'ksplit', dict(k=k), k))
    for m in ('ch_copy', 'ch_photon', 'ph_copy'): E.append((f'pid_{m}', 'pidsplit', dict(mode=m), +1))
    E.append(('rmsoft', 'rmsoft', {}, -1))
    E.append(('merge_soft', 'merge', {}, -1))
    E.append(('merge_split', 'merge_split', {}, 0))
    return E


# ------------------------------------------------------------------------------------------------------------- observables
def shapes(Jt, K=None):
    """N, N_ch, pT^D, girth, C1^(beta) for beta in (0.5, 1, 2) (IRC-safe except N, N_ch, pT^D); angles about the E-scheme axis"""
    K = K or kinematics(Jt); pt = Jt['pt']; real = K['real']; z = pt / pt.sum(1, keepdims=True)
    dR = np.hypot(K['deta'], K['dphi'])
    out = dict(N=real.sum(1).astype(float), Nch=(np.isin(np.abs(Jt['pid']), CH_HAD + (11, 13)) & real).sum(1).astype(float),
               ptd=np.sqrt((pt ** 2).sum(1)) / pt.sum(1), girth=(z * dR).sum(1))
    c = {b: np.zeros(len(pt)) for b in (0.5, 1.0, 2.0)}
    for s in range(0, len(pt), 500):
        e, p, zz = K['deta'][s:s + 500], K['dphi'][s:s + 500], z[s:s + 500]
        d = np.hypot(e[:, :, None] - e[:, None, :], (p[:, :, None] - p[:, None, :] + np.pi) % (2 * np.pi) - np.pi)
        w = zz[:, :, None] * zz[:, None, :]
        for b in c: c[b][s:s + 500] = 0.5 * (w * d ** b).sum((1, 2))
    out.update({f'C1_{b}': c[b] for b in c})
    out['mass'] = np.sqrt(np.maximum(K['jE'] ** 2 - K['px'].sum(1) ** 2 - K['py'].sum(1) ** 2 - K['pz'].sum(1) ** 2, 0))
    return out


SHAPE_KEYS = ['girth', 'C1_0.5', 'C1_1.0', 'C1_2.0']


def shape_feats(O, with_n):
    X = [np.log(np.maximum(O[k], 1e-8)) for k in SHAPE_KEYS]
    if with_n: X = [O['N']] + X
    return np.stack(X, 1)


# ------------------------------------------------------------------------------------------------------------------ models
def weaver_inputs(Jt, pidfeat):
    K = kinematics(Jt); L = L_IN
    sl = lambda a: a[:, :L] if a.shape[1] >= L else np.pad(a, ((0, 0), (0, L - a.shape[1])))
    real = sl(K['real']).astype(np.float32)
    pt = sl(Jt['pt']); E = sl(K['E']); safe = lambda a: np.where(real > 0, a, 1.0)
    jpt, jE = K['jpt'][:, None], K['jE'][:, None]
    cl = lambda a, c, s: np.clip((a - c) * s, -5, 5)
    deta, dphi = sl(K['deta']), sl(K['dphi'])
    f = [cl(np.log(safe(pt)), 1.7, 0.7), cl(np.log(safe(E)), 2.0, 0.7), cl(np.log(safe(pt) / jpt), -4.7, 0.7),
         cl(np.log(safe(E) / jE), -4.7, 0.7), cl(np.hypot(deta, dphi), 0.2, 4.0)]
    if pidfeat:
        pid = sl(Jt['pid']); ap = np.abs(pid)
        chp = np.isin(pid, (211, 321, 2212)); chm = np.isin(pid, (-211, -321, -2212))
        charge = (chp.astype(float) + (pid == -11) + (pid == -13)) - (chm.astype(float) + (pid == 11) + (pid == 13))
        f += [charge, (chp | chm).astype(float), np.isin(ap, (130, 2112)).astype(float), (pid == 22).astype(float),
              (ap == 11).astype(float), (ap == 13).astype(float)]
    f += [deta, dphi]
    x = (np.stack(f, 1) * real[:, None]).astype(np.float32)
    pts = (np.stack([deta, dphi], 1) * real[:, None]).astype(np.float32)
    v = (np.stack([sl(K['px']), sl(K['py']), sl(K['pz']), E], 1) * real[:, None]).astype(np.float32)
    return pts, x, v, real[:, None, :]


def lnet_inputs(Jt):
    """LorentzNet QG inputs as qg/dataset.py: p4 from (pt, y, phi, pid) with PDG masses (energyflow.p4s_from_ptyphipids),
    one-hot of |pid| over [11, 13, 22, 130, 211, 321, 2112, 2212], padded to L_LN nodes"""
    L = L_LN; sl = lambda a: a[:, :L] if a.shape[1] >= L else np.pad(a, ((0, 0), (0, L - a.shape[1])))
    pt, y, phi, pid = sl(Jt['pt']), sl(Jt['eta']), sl(Jt['phi']), sl(Jt['pid']); real = pt > 0
    ap = np.abs(pid); m = np.zeros_like(pt)
    for k, v in MASS.items(): m[ap == k] = v
    mt = np.sqrt(pt ** 2 + m ** 2)
    p4 = np.stack([mt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y)], -1) * real[..., None]
    cats = [11, 13, 22, 130, 211, 321, 2112, 2212]
    oh = np.stack([(ap == c) & real for c in cats], -1).astype(np.float32)
    return p4.astype(np.float32), oh, real


class Models:
    def __init__(self, dev):
        import torch
        self.t = torch; self.dev = dev; self.m = {}; self.info = {}

    def _sd(self, path):
        sd = self.t.load(path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
        out = {}
        for k, v in sd.items():
            for p in ('module.', 'mod.'):
                if k.startswith(p): k = k[len(p):]
            out[k] = v
        return out

    def add_weaver(self, name, arch, pidfeat):
        from weaver.nn.model.ParticleTransformer import ParticleTransformer
        from weaver.nn.model.ParticleNet import ParticleNet
        import hashlib
        d = V / 'models' / 'qg' / name; best = d / 'net_best_epoch_state.pt'
        raw = best.read_bytes(); h = hashlib.md5(raw).hexdigest()
        ep = None
        for f in sorted(d.glob('net_epoch-*_state.pt')):
            if hashlib.md5(f.read_bytes()).hexdigest() == h: ep = int(f.name.split('-')[1].split('_')[0])
        n_ep = len(list(d.glob('net_epoch-*_state.pt')))
        done = (d / 'predict_output').exists() or n_ep >= 20
        nin = 13 if pidfeat else 7
        if arch == 'part':
            net = ParticleTransformer(input_dim=nin, num_classes=2, pair_input_dim=4, use_pre_activation_pair=False,
                                      embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8,
                                      num_cls_layers=2, block_params=None,
                                      cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                                      fc_params=[], activation='gelu', trim=True, for_inference=False)
        else:
            net = ParticleNet(input_dims=nin, num_classes=2, conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
                              fc_params=[(256, 0.1)], use_fusion=False, use_fts_bn=True, use_counts=True, for_inference=False)
        missing = net.load_state_dict(self._sd(best), strict=True)
        net.eval().to(self.dev)
        self.m[name] = ('weaver', arch, pidfeat, net)
        self.info[name] = dict(checkpoint=str(best), best_epoch=ep, epochs_written=n_ep, final=bool(done),
                               status='final' if done else 'interim', md5=h)
        log(f'{name}: best epoch {ep} of {n_ep} written ({self.info[name]["status"]})')
        return net

    def add_permuted(self, name, src):
        import copy
        kind, arch, pidfeat, net = self.m[src]; g = self.t.Generator().manual_seed(0)
        net2 = copy.deepcopy(net).cpu()
        with self.t.no_grad():
            for p in net2.parameters():
                flat = p.data.view(-1); flat.copy_(flat[self.t.randperm(flat.numel(), generator=g)])
        self.m[name] = (kind, arch, pidfeat, net2.eval().to(self.dev)); self.info[name] = dict(note=f'{src} with permuted weights')

    def add_lnet(self):
        import sys
        sys.path.insert(0, str(V / 'ext' / 'LorentzNet-release'))
        from models import LorentzNet
        sys.path.pop(0)
        net = LorentzNet(n_scalar=8, n_hidden=72, n_class=2, n_layers=6, c_weight=1e-3, dropout=0.2)
        net.load_state_dict(self._sd(V / 'models' / 'qg' / 'lorentznet' / 'best-val-model.pt'), strict=True)
        self.m['lorentznet'] = ('lnet', None, True, net.eval().to(self.dev))
        tr = json.loads((V / 'models' / 'qg' / 'lorentznet' / 'test-result.json').read_text())
        self.info['lorentznet'] = dict(status='released', released_test_acc=tr['test_acc'], released_test_auc=tr['test_auc'])

    def add_torch(self, name, net, info):
        self.m[name] = ('ef', name, False, net.eval().to(self.dev)); self.info[name] = info

    def logit(self, name, Jt, bs=500):
        """log p(g) - log p(q) per jet"""
        t = self.t; kind, arch, pidfeat, net = self.m[name]; out = []
        with t.no_grad():
            if kind == 'weaver':
                P, X, Vv, M = weaver_inputs(Jt, pidfeat)
                for s in range(0, len(X), bs):
                    a = [t.from_numpy(z[s:s + bs]).to(self.dev) for z in (P, X, Vv, M)]
                    o = net(a[1], v=a[2], mask=a[3].bool()) if arch == 'part' else net(a[0], a[1], a[3])
                    o = t.log_softmax(o.float(), -1); out.append((o[:, 1] - o[:, 0]).cpu().numpy())   # labels [isQ, isG]
            elif kind == 'lnet':
                p4, oh, real = lnet_inputs(Jt); L = p4.shape[1]; bs = 200
                for s in range(0, len(p4), bs):
                    b = len(p4[s:s + bs]); rm = t.from_numpy(real[s:s + bs])
                    em = (rm[:, None, :] & rm[:, :, None]) & ~t.eye(L, dtype=t.bool)[None]
                    bi, ii, jj = t.nonzero(em, as_tuple=True)
                    edges = [(bi * L + ii).to(self.dev), (bi * L + jj).to(self.dev)]
                    o = net(scalars=t.from_numpy(oh[s:s + bs]).view(b * L, -1).to(self.dev),
                            x=t.from_numpy(p4[s:s + bs]).view(b * L, -1).to(self.dev), edges=edges,
                            node_mask=rm.view(b * L, 1).float().to(self.dev), edge_mask=None, n_nodes=L)
                    o = t.log_softmax(o.float(), -1); out.append((o[:, 0] - o[:, 1]).cpu().numpy())  # label 1 = quark
            else:
                Z, A = ef_inputs(Jt)
                for s in range(0, len(Z), 2000):
                    o = net(t.from_numpy(Z[s:s + 2000]).to(self.dev), t.from_numpy(A[s:s + 2000]).to(self.dev))
                    o = t.log_softmax(o.float(), -1); out.append((o[:, 0] - o[:, 1]).cpu().numpy())  # label 1 = quark
        return np.concatenate(out).astype(np.float64)


def ef_inputs(Jt, L=L_IN):
    K = kinematics(Jt); sl = lambda a: a[:, :L] if a.shape[1] >= L else np.pad(a, ((0, 0), (0, L - a.shape[1])))
    pt = sl(Jt['pt']); z = pt / pt.sum(1, keepdims=True)
    return z.astype(np.float32), np.stack([sl(K['deta']), sl(K['dphi'])], -1).astype(np.float32) * (pt > 0)[..., None]


def make_ef(kind):
    import torch
    nn = torch.nn

    class EF(nn.Module):
        def __init__(self):
            super().__init__(); self.kind = kind; d = 2 if kind == 'efn' else 3
            self.phi = nn.Sequential(nn.Linear(d, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 256), nn.ReLU())
            self.F = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(),
                                   nn.Linear(100, 2))

        def forward(self, z, a):
            m = (z > 0).float()[..., None]
            a = a / 0.4                                   # angles in units of R (fixed scale; IRC safety unchanged)
            if self.kind == 'efn': lat = (z[..., None] * self.phi(a)).sum(1)
            else: lat = (m * self.phi(torch.cat([z[..., None], a], -1))).sum(1)
            return self.F(lat)
    return EF()


def train_ef(kind, dev, n_train=400_000, epochs=15, seed=0, save=True):
    """EFN / PFN on kin3 (pT, deta, dphi of the 128 hardest) of random non-val train rows; val = 50k is_val rows"""
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    d = QG / 'train'; y = np.load(d / 'label.npy'); isv = np.load(d / 'is_val.npy'); k3 = np.load(d / 'kin3.npy', mmap_mode='r')
    tr = np.sort(rng.choice(np.flatnonzero(~isv), n_train, replace=False)); va = np.sort(rng.choice(np.flatnonzero(isv), 50_000, replace=False))
    def prep(ix):
        a = np.asarray(k3[ix], np.float32); pt = a[..., 0]; z = pt / pt.sum(1, keepdims=True)
        return torch.from_numpy(z), torch.from_numpy(a[..., 1:3] * (pt > 0)[..., None]), torch.from_numpy(y[ix].astype(np.int64))
    Ztr, Atr, Ytr = prep(tr); Zva, Ava, Yva = prep(va)
    net = make_ef(kind).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); lf = torch.nn.CrossEntropyLoss()
    best, bst = 1e9, None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Ytr))
        for s in range(0, len(perm), 500):
            b = perm[s:s + 500]; opt.zero_grad()
            loss = lf(net(Ztr[b].to(dev), Atr[b].to(dev)), Ytr[b].to(dev)); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            o = torch.cat([net(Zva[s:s + 5000].to(dev), Ava[s:s + 5000].to(dev)).cpu() for s in range(0, len(Yva), 5000)])
            vl = float(lf(o, Yva)); va_acc = float((o.argmax(1) == Yva).float().mean())
        log(f'  {kind} epoch {ep}: val loss {vl:.4f} acc {va_acc:.4f}')
        if vl < best: best = vl; bst = {k: v.clone() for k, v in net.state_dict().items()}; best_ep = ep
    net.load_state_dict(bst)
    out = V / 'models' / 'qg' / f'h5_{kind}_torch.pt'
    if save: torch.save(bst, out)
    return net, dict(status='trained in H5', train_jets=n_train, epochs=epochs, best_epoch=best_ep, val_loss=best, path=str(out))


def fit_classical(seed=0, n=100_000):
    """logistic regressions on random non-val train rows (kin3): (N, log girth, log C1^b) and the shapes alone"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    d = QG / 'train'; rng = np.random.default_rng(seed); y = np.load(d / 'label.npy'); isv = np.load(d / 'is_val.npy')
    ix = np.sort(rng.choice(np.flatnonzero(~isv), n, replace=False)); k3 = np.asarray(np.load(d / 'kin3.npy', mmap_mode='r')[ix], np.float64)
    pt = k3[..., 0]; real = pt > 0
    Jt = dict(pt=pt, eta=k3[..., 1] * real, phi=k3[..., 2] * real, pid=np.zeros(pt.shape, np.int64), n=real.sum(1))
    O = shapes(Jt, dict(real=real, deta=k3[..., 1] * real, dphi=k3[..., 2] * real, jE=np.ones(len(pt)), px=pt, py=pt, pz=pt))
    g = (y[ix] == 0).astype(int)
    out = {}
    for name, wn in (('lr_cls', True), ('lr_shape', False)):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1e3)).fit(shape_feats(O, wn), g)
        out[name] = m
    coef_n = out['lr_cls'][1].coef_[0][0] / out['lr_cls'][0].scale_[0]
    return out, dict(train_jets=n, coef_N_per_particle=float(coef_n), features=['N'] + [f'log {k}' for k in SHAPE_KEYS])


def lr_n_table():
    """empirical log LR(N) = log p(N|g) - log p(N|q) on all non-val train jets (jet_nparticles)"""
    d = QG / 'train'; y = np.load(d / 'label.npy'); isv = np.load(d / 'is_val.npy'); n = np.load(d / 'jet.npy', mmap_mode='r')[:, 5]
    n = np.asarray(n)[~isv].astype(int); y = y[~isv]; M = 200
    hq = np.bincount(n[y == 1], minlength=M)[:M] + 0.5; hg = np.bincount(n[y == 0], minlength=M)[:M] + 0.5
    return np.log(hg / hg.sum()) - np.log(hq / hq.sum()), dict(n_train=int(len(y)), mean_q=float(n[y == 1].mean()), mean_g=float(n[y == 0].mean()))


# -------------------------------------------------------------------------------------------------------------- statistics
def boot_mean(d, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(d); s = np.empty(B)
    for b in range(0, B, 200):
        ix = rng.integers(0, n, (min(200, B - b), n)); s[b:b + len(ix)] = d[ix].mean(1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def signflip_p(d, P=5000, seed=0):
    """two-sided paired sign-flip permutation p for mean(d) = 0"""
    rng = np.random.default_rng(seed); obs = abs(d.mean()); cnt = 0
    for s in range(0, P, 500):
        sg = rng.choice([-1.0, 1.0], (min(500, P - s), len(d)))
        cnt += int((np.abs(sg @ d / len(d)) >= obs - 1e-15).sum())
    return (cnt + 1) / (P + 1)


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans (a = before call, b = after call)"""
    from scipy.stats import binomtest
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, float(p)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def roc_metrics(s_g, y_q):
    """s_g = gluon logit; quark is signal: score -s_g"""
    from sklearn.metrics import roc_auc_score
    sq = -s_g; auc = roc_auc_score(y_q, sq); r = {}
    for e in (0.3, 0.5):
        thr = np.quantile(sq[y_q == 1], 1 - e); eg = np.mean(sq[y_q == 0] >= thr); r[f'R{int(e * 100)}'] = float(1 / max(eg, 1e-9))
    return dict(AUC=float(auc), acc=float(((sq > 0).astype(int) == y_q).mean()), **r)


def boot_auc(s_g, y, B=200, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        ix = rng.integers(0, len(y), len(y)); v.append(roc_auc_score(y[ix], -s_g[ix]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def wp_effects(l0, l1, y):
    """thresholds frozen on unedited jets at eps_q 0.3 / 0.5 -> (eps_q, 1/eps_g) after the edit; McNemar at eps_q 0.5"""
    out = {}
    for e in (0.3, 0.5):
        thr = np.quantile(-l0[y == 1], 1 - e)
        c0, c1 = -l0 >= thr, -l1 >= thr
        eg0, eg1 = c0[y == 0].mean(), c1[y == 0].mean()
        out[f'wp{int(e * 100)}'] = dict(eps_q_after=float(c1[y == 1].mean()), R_before=float(1 / max(eg0, 1e-9)),
                                        R_after=float(1 / max(eg1, 1e-9)), dR=float(1 / max(eg1, 1e-9) - 1 / max(eg0, 1e-9)))
        if e == 0.5:
            n01, n10, p = mcnemar(c0, c1)
            out['calls_changed_pct'] = float(100 * (c0 != c1).mean())
            out['calls_changed_ci'] = boot_mean((c0 != c1).astype(float) * 100)
            out['mcnemar'] = dict(to_quark=n01, to_gluon=n10, p=p)
    return out


def c2st(Lu, Le, seed=0, cols=None):
    """classifier two-sample test: edited jets (one half) vs unedited jets (other half), GBDT on low-level summaries"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    def feats(L):
        F = []
        for a in L:
            pt = a[:, 0]; z = pt / pt.sum(); n = len(a)
            d = np.hypot(a[:, None, 1] - a[None, :, 1], (a[:, None, 2] - a[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
            d[np.diag_indices(n)] = 9; m6 = min(n, 6)
            zz = np.pad(z[:5], (0, max(0, 5 - len(z))))
            F.append(np.concatenate([[n, np.sqrt((z ** 2).sum()), np.log(d.min() + 1e-9), np.log(d[:m6, :m6].min() + 1e-9),
                                      (d < 1e-3).sum() / 2, z[-1] * pt.sum(), np.log(z[0] / max(z[1], 1e-9)) if n > 1 else 0], zz]))
        return np.array(F)
    rng = np.random.default_rng(seed); J = len(Lu); perm = rng.permutation(J); A, B = perm[: J // 2], perm[J // 2:]
    X = np.concatenate([feats([Le[i] for i in A]), feats([Lu[i] for i in B])]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    if cols is not None: X = X[:, cols]
    sp = rng.permutation(len(y)); tr, te = sp[: len(y) // 2], sp[len(y) // 2:]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr]); p = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p); bs = []
    for _ in range(200):
        ix = rng.integers(0, len(te), len(te)); bs.append(roc_auc_score(y[te][ix], p[ix]))
    return dict(auc=float(auc), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


# -------------------------------------------------------------------------------------------------------------------- run
def load_models(dev, with_ef=True, smoke=False):
    M = Models(dev); MI = {}
    for name, arch, pf in (('part_kin', 'part', False), ('part_kinpid', 'part', True), ('pnet_kin', 'pn', False), ('pnet_kinpid', 'pn', True)):
        try: M.add_weaver(name, arch, pf)
        except Exception as e: log(f'{name}: not loaded: {e!r}')
    M.add_lnet()
    if 'part_kin' in M.m: M.add_permuted('part_kin_perm', 'part_kin')
    if with_ef:
        for kind in ('efn', 'pfn'):
            p = V / 'models' / 'qg' / f'h5_{kind}_torch.pt'
            import torch
            if smoke:
                net, info = train_ef(kind, dev, n_train=20_000, epochs=1, save=False)
            elif p.exists():
                net = make_ef(kind); net.load_state_dict(torch.load(p, map_location='cpu'))
                info = json.loads((V / 'models' / 'qg' / f'h5_{kind}_torch.json').read_text())
            else:
                log(f'train {kind}'); net, info = train_ef(kind, dev)
                (V / 'models' / 'qg' / f'h5_{kind}_torch.json').write_text(json.dumps(info))
            M.add_torch(f'{kind}_t', net, info)
    return M


def step_run(a):
    import torch
    from interp import data_qg
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    S = data_qg.sample(a.per_class, a.seed); y = S['label'].astype(int); J = len(y)
    log(f'sample: {J} jets (seed {a.seed}), quark frac {y.mean():.3f}, device {dev}')
    Jt0 = from_test(S); L0 = to_list(Jt0); Ju = from_list(L0, Lpad=160)
    K = kinematics(Ju)
    chk = dict(max_d_deta=float(np.abs(K['deta'][:, :135] - S['full'][..., 4]).max()),
               max_d_dphi=float(np.abs(K['dphi'][:, :135] - S['full'][..., 5]).max()),
               max_rel_d_jetpt=float(np.abs(K['jpt'] / S['jet'][:, 0] - 1).max()))
    log('rebuild check vs stored arrays', chk)
    nn_dr = []
    for arr in L0[:2000]:
        d = np.hypot(arr[:, None, 1] - arr[None, :, 1], (arr[:, None, 2] - arr[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
        d[np.diag_indices(len(arr))] = 9; nn_dr.append(d.min())
    nn_dr = np.array(nn_dr)
    indist = dict(frac_pair_dR_lt_1em3=float((nn_dr < 1e-3).mean()), frac_pair_dR_lt_5em4=float((nn_dr < 5e-4).mean()),
                  median_min_pair_dR=float(np.median(nn_dr)), n_jets=len(nn_dr))
    log('in-distribution check (unedited, 2000 jets): min pair dR', indist)

    lrN, lrn_info = lr_n_table()
    clsm, cls_info = fit_classical(seed=a.seed, n=5000 if a.smoke else 100_000)
    log('classical: LR(N) table', lrn_info, '; lr_cls', cls_info)
    M = load_models(dev, smoke=a.smoke)
    names = list(M.m)
    O0 = shapes(Ju)
    base = {}
    for nm in names:
        base[nm] = M.logit(nm, Ju); log(f'{nm}: unedited', roc_metrics(base[nm], y))
    for nm, m in clsm.items(): base[nm] = m.decision_function(shape_feats(O0, nm == 'lr_cls')); names.append(nm)
    base['LR_N'] = lrN[np.minimum(O0['N'].astype(int), 199)]; names.append('LR_N')
    unedited = {nm: dict(**roc_metrics(base[nm], y), AUC_ci=boot_auc(base[nm], y)) for nm in names}
    for nm in names: log(f'unedited {nm}:', unedited[nm])

    rng = np.random.default_rng(a.seed + 1)
    E = edits_spec(); rows = {}; per_jet = {}; full = {}; okm = {}; ptests = []
    for en, kind, kw, dn in E:
        Le, ok = edit_list(L0, kind, rng, **kw); Je = from_list(Le, Lpad=170); Oe = shapes(Je)
        dshape = {k: dict(mean_abs=float(np.mean(np.abs(Oe[k] - O0[k])[ok])), mean_rel=float(np.mean((np.abs(Oe[k] - O0[k]) / np.maximum(np.abs(O0[k]), 1e-12))[ok])))
                  for k in ('N', 'Nch', 'ptd', 'girth', 'C1_0.5', 'C1_1.0', 'C1_2.0', 'mass')}
        overflow = int((Je['n_full'] > L_IN).sum())
        R = dict(kind=kind, kw={k: str(v) for k, v in kw.items()}, dN=dn, applied=int(ok.sum()), overflow_128=overflow, dshape=dshape, models={})
        for nm in names:
            if nm in M.m: le = M.logit(nm, Je)
            elif nm == 'LR_N': le = lrN[np.minimum(Oe['N'].astype(int), 199)]
            else: le = clsm[nm].decision_function(shape_feats(Oe, nm == 'lr_cls'))
            full[(en, nm)] = le - base[nm]; okm[en] = ok
            d = (le - base[nm])[ok]; per_jet[(en, nm)] = d
            r = dict(mean=float(d.mean()), ci=boot_mean(d), median=float(np.median(d)), sd_base=float(base[nm].std()),
                     per_dN=float(d.mean() / dn) if dn else None,
                     by_class=dict(q=float(d[y[ok] == 1].mean()), g=float(d[y[ok] == 0].mean())),
                     p0=signflip_p(d), p0_t=_tp(d), **wp_effects(base[nm][ok], le[ok], y[ok]))
            R['models'][nm] = r; ptests.append((en, nm, 'p0_t', r['p0_t']))
        rows[en] = R
        log(f'{en}: ' + '; '.join(f'{nm} {R["models"][nm]["mean"]:+.3f}' for nm in names))

    # edit vs count-fixed control (split hardest z 0.5 vs merge-soft + split-hardest), restore fraction
    ctrl = {}
    for nm in names:
        jm = okm['split_r0_z0.5'] & okm['merge_split']
        a1, c1 = full[('split_r0_z0.5', nm)][jm], full[('merge_split', nm)][jm]
        d = a1 - c1
        ctrl[nm] = dict(n=int(jm.sum()), edit=float(a1.mean()), edit_ci=boot_mean(a1), control=float(c1.mean()), control_ci=boot_mean(c1),
                        edit_minus_control=float(d.mean()), ci=boot_mean(d), p=_tp(d), p_perm=signflip_p(d),
                        restore_fraction=float(1 - c1.mean() / a1.mean()) if abs(a1.mean()) > 1e-9 else None,
                        restore_ci=_ratio_ci(a1, c1))
        ptests.append(('split_r0_z0.5 - merge_split', nm, 'p_ctrl', ctrl[nm]['p']))
    # linearity in k
    lin = {}
    for nm in names:
        D = np.stack([per_jet[(f'ksplit_{k}', nm)] for k in range(1, 6)], 1); ks = np.arange(1, 6)
        def fit(D):
            mu = D.mean(0); A = np.vstack([np.ones(5), ks, ks ** 2]).T
            c2 = np.linalg.lstsq(A, mu, rcond=None)[0]; c1 = np.polyfit(ks, mu, 1)
            return c1[0], c2[2], mu
        b, q, mu = fit(D); rb = np.random.default_rng(3); bs = []
        for _ in range(1000):
            ix = rb.integers(0, len(D), len(D)); bs.append(fit(D[ix])[:2])
        bs = np.array(bs)
        lin[nm] = dict(means=mu.tolist(), slope=float(b), slope_ci=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                       quad=float(q), quad_ci=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                       slope_over_poisson=float(b / POISSON))
    adj = holm([p[3] for p in ptests])
    holm_tab = [dict(edit=p[0], model=p[1], test=p[2], p=p[3], holm=float(h)) for p, h in zip(ptests, adj)]
    lrn_slope = dict(ksplit_1=rows['ksplit_1']['models']['LR_N']['mean'], ksplit_slope=lin['LR_N']['slope'])

    # off-manifold: classifier two-sample tests
    rng2 = np.random.default_rng(a.seed + 7); off = {}
    for en, kind, kw in (('split_r0_z0.5', 'split', dict(rank=0, z=0.5)), ('split_r5_z0.5', 'split', dict(rank=5, z=0.5)),
                         ('split_rsoft_z0.5', 'split', dict(rank='soft', z=0.5)), ('ksplit_1', 'ksplit', dict(k=1)),
                         ('ksplit_5', 'ksplit', dict(k=5)), ('rmsoft', 'rmsoft', {}), ('merge_split', 'merge_split', {})):
        Le, _ = edit_list(L0, kind, rng2, **kw)
        off[en] = dict(all=c2st(L0, Le), angle_only=c2st(L0, Le, cols=[2, 3, 4]), no_angle=c2st(L0, Le, cols=[0, 1, 5, 6, 7, 8, 9, 10, 11]))
        log('c2st', en, off[en])
    off['none (unedited vs unedited)'] = dict(all=c2st(L0, L0))

    # interim epochs (seed / training-stage spread): k=1 random split and rmsoft for every written epoch of each weaver model
    epochs = {}
    if not a.no_epochs:
        import torch as T
        Le1, ok1 = edit_list(L0, 'ksplit', np.random.default_rng(11), k=1); Je1 = from_list(Le1, Lpad=170)
        Ler, _ = edit_list(L0, 'rmsoft', np.random.default_rng(12)); Jer = from_list(Ler, Lpad=170)
        for nm in ('part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid'):
            if nm not in M.m: continue
            net = M.m[nm][3]; keep = {k: v.clone() for k, v in net.state_dict().items()}; epochs[nm] = []
            for f in sorted((V / 'models' / 'qg' / nm).glob('net_epoch-*_state.pt'), key=lambda p: int(p.name.split('-')[1].split('_')[0])):
                net.load_state_dict(M._sd(f)); l0 = M.logit(nm, Ju); l1 = M.logit(nm, Je1); l2 = M.logit(nm, Jer)
                epochs[nm].append(dict(epoch=int(f.name.split('-')[1].split('_')[0]), **roc_metrics(l0, y),
                                       ksplit1=float((l1 - l0).mean()), ksplit1_ci=boot_mean(l1 - l0, B=500),
                                       rmsoft=float((l2 - l0).mean()), rmsoft_ci=boot_mean(l2 - l0, B=500)))
                log(nm, epochs[nm][-1])
            net.load_state_dict(keep)

    out = dict(sample=dict(per_class=a.per_class, seed=a.seed, J=J, split='test'), rebuild_check=chk, in_distribution=indist,
               dr_split=DR_SPLIT, poisson_pred=POISSON, lr_n=lrn_info, lr_n_slope=lrn_slope, classical=cls_info,
               models=M.info, unedited=unedited, edits=rows, control=ctrl, linearity=lin, holm=holm_tab, offmanifold=off,
               epochs=epochs, wall_s=time.time() - T0)
    tag = '_smoke' if a.smoke else ''
    (RES / f'h5_run{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(RES / f'h5_perjet{tag}.npz', y=y, idx=S['idx'], **{f'{e}|{m}': v for (e, m), v in per_jet.items()},
                        **{f'base|{m}': v for m, v in base.items()})
    log('wrote', RES / f'h5_run{tag}.json')


def _tp(d):
    from scipy.stats import ttest_1samp
    return float(ttest_1samp(d, 0).pvalue) if d.std() > 0 else (1.0 if abs(d.mean()) < 1e-12 else 0.0)


def _ratio_ci(a1, c1, B=1000, seed=5):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        ix = rng.integers(0, len(a1), len(a1)); m = a1[ix].mean()
        if abs(m) > 1e-9: v.append(1 - c1[ix].mean() / m)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None


def step_herwig(a):
    import torch
    from interp import data_qg
    st = json.loads((HW / 'status.json').read_text()) if (HW / 'status.json').exists() else dict(ok=False)
    if not st.get('ok'):
        (RES / 'h5_herwig.json').write_text(json.dumps(dict(status='blocked', reason='Herwig download failed or absent', hw=st))); return
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = np.load(HW / 'raw.npy', mmap_mode='r'); yh = np.load(HW / 'label.npy'); rng = np.random.default_rng(a.seed)
    ix = np.sort(np.concatenate([rng.choice(np.flatnonzero(yh == c), a.per_class, replace=False) for c in (0, 1)]))
    Jh = from_raw(np.asarray(X[ix])); Lh = to_list(Jh); Jh = from_list(Lh, Lpad=170); y_h = yh[ix].astype(int)
    S = data_qg.sample(a.per_class, a.seed); y_p = S['label'].astype(int); Lp = to_list(from_test(S)); Jp = from_list(Lp, Lpad=170)
    M = load_models(dev); lrN, _ = lr_n_table(); res = dict(hw=st, models={})
    Oh, Op = shapes(Jh), shapes(Jp)
    res['N_mean'] = dict(herwig_q=float(Oh['N'][y_h == 1].mean()), herwig_g=float(Oh['N'][y_h == 0].mean()),
                         pythia_q=float(Op['N'][y_p == 1].mean()), pythia_g=float(Op['N'][y_p == 0].mean()))
    for nm in list(M.m) + ['LR_N']:
        r = {}
        for tag, Jt, Lx, yy, O in (('pythia', Jp, Lp, y_p, Op), ('herwig', Jh, Lh, y_h, Oh)):
            f = (lambda J_, O_: lrN[np.minimum(O_['N'].astype(int), 199)]) if nm == 'LR_N' else (lambda J_, O_: M.logit(nm, J_))
            l0 = f(Jt, O); Le, ok = edit_list(Lx, 'ksplit', np.random.default_rng(21), k=1); Je = from_list(Le, Lpad=170)
            l1 = f(Je, shapes(Je) if nm == 'LR_N' else None); l0, l1 = l0, np.where(ok, l1, l0)
            r[tag] = dict(**roc_metrics(l0, yy), AUC_ci=boot_auc(l0, yy), mean_logit_q=float(l0[yy == 1].mean()),
                          mean_logit_g=float(l0[yy == 0].mean()), ksplit1=float((l1 - l0).mean()), ksplit1_ci=boot_mean(l1 - l0))
        res['models'][nm] = r; log(nm, r)
    (RES / 'h5_herwig.json').write_text(json.dumps(res, indent=1, default=float)); log('wrote h5_herwig.json')


def step_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((RES / 'h5_run.json').read_text())
    order = [m for m in ('part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_t', 'efn_t', 'lr_cls', 'lr_shape', 'LR_N', 'part_kin_perm') if m in R['linearity']]
    lab = dict(part_kin='ParT kin', part_kinpid='ParT kinpid', pnet_kin='PN kin', pnet_kinpid='PN kinpid', lorentznet='LorentzNet',
               pfn_t='PFN', efn_t='EFN (IRC-safe)', lr_cls='classical LR(N, shapes)', lr_shape='classical LR(shapes)', LR_N='count-only LR(N) (data)',
               part_kin_perm='ParT kin, permuted weights')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ks = np.arange(1, 6); cmap = plt.get_cmap('tab10')
    for i, m in enumerate(order):
        if m in ('part_kin_perm',): continue
        mu = R['linearity'][m]['means']
        cis = np.array([R['edits'][f'ksplit_{k}']['models'][m]['ci'] for k in ks])
        ax[0].errorbar(ks + (i - 5) * 0.03, mu, yerr=[np.array(mu) - cis[:, 0], cis[:, 1] - np.array(mu)], marker='o', ms=4,
                       label=lab[m], color=cmap(i % 10), ls='--' if m in ('LR_N', 'lr_cls', 'lr_shape') else '-')
    ax[0].plot(ks, POISSON * ks, 'k:', lw=2, label='Casimir-meets-Poisson prediction (0.46 per particle)')
    ax[0].set_xlabel('number of random collinear splits k  (ΔN = +k, IRC-safe shapes fixed)'); ax[0].set_ylabel('mean Δ logit (gluon − quark)')
    ax[0].axhline(0, color='grey', lw=0.5)
    ax[0].legend(fontsize=7.5, loc='upper left')
    rows = [m for m in ('part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_t', 'efn_t', 'LR_N') if m in R['edits']['rmsoft']['models']]
    edits = [('split_r0_z0.5', 'split hardest particle (ΔN = +1)', '#d62728'),
             ('merge_split', 'merge 2 softest + split hardest (ΔN = 0, control)', '#7f7f7f'),
             ('rmsoft', 'remove softest particle (ΔN = −1)', '#1f77b4')]
    w = 0.26
    for i, (e, el, c) in enumerate(edits):
        mu = np.array([R['edits'][e]['models'][m]['mean'] for m in rows]); ci = np.array([R['edits'][e]['models'][m]['ci'] for m in rows])
        ax[1].barh(np.arange(len(rows)) + (i - 1) * w, mu, height=w, color=c, xerr=[mu - ci[:, 0], ci[:, 1] - mu], error_kw=dict(lw=0.7), label=el)
    ax[1].set_yticks(range(len(rows))); ax[1].set_yticklabels([lab[m] for m in rows], fontsize=8); ax[1].invert_yaxis()
    ax[1].axvline(0, color='k', lw=0.5)
    ax[1].set_xlabel('mean Δ logit (gluon − quark), 95 % CI'); ax[1].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.45, -0.13), ncol=1, frameon=False)
    ax[1].set_title('split hardest vs count-fixed control vs remove softest', fontsize=9)
    fig.suptitle(f'Quark–gluon taggers: response to adding collinear particles ({R["sample"]["J"]} test jets)', fontsize=11)
    fig.tight_layout(); fig.savefig(RES / 'h5_counting.png', dpi=130); log('wrote', RES / 'h5_counting.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['hw', 'run', 'herwig', 'fig'])
    ap.add_argument('--per_class', type=int, default=5000); ap.add_argument('--seed', type=int, default=701)
    ap.add_argument('--no_epochs', action='store_true'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    dict(hw=lambda: step_hw(), run=lambda: step_run(a), herwig=lambda: step_herwig(a), fig=lambda: step_fig(a))[a.step]()
