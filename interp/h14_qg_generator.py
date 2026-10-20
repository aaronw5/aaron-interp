"""H14 (loop D; claims D7, D5; absorbs H10): quark-gluon generator dependence (Pythia 8 vs Herwig 7.1) and the hard-core
function behind the hardest-split response.

Samples
  Pythia: interp.data_qg.sample(20000 per class, seed 9311) (Komiske-Metodiev-Thaler test split, random rows).
  Herwig: Zenodo 3066475 files 18-19 (= the Herwig test split of the 80/10/10 file split used for the Herwig retrains;
          disjoint from H5/H5v, which used files 0-1), 20000 random jets per class, seed 9311.
Models (logit = log p(gluon) - log p(quark))
  Pythia-trained: part_kin, part_kinpid, pnet_kin, pnet_kinpid (H2 weaver trainings, final), lorentznet (released),
                  efn_t, pfn_t (H5 torch stand-ins), part_kin_perm (permuted-weight null)
  Herwig-trained: hw_part_kin, hw_pnet_kin (models/qg_herwig, launched in H14 with interp/h14_hw_train.sh; the checkpoint
                  status, final or interim with its epoch, is recorded)
  Baselines (CPU step efp): LR(N) from negative binomial fits to the multiplicity per class, one per generator (lrn_py,
                  lrn_hw); GBDT on EFPs (d <= 5, beta = 1) + N (efpn_<gen>) and on EFPs only (efp_<gen>), trained on 60k
                  jets of each generator's train split.
Edits (every edit is generated deterministically from (generator, edit name), so the GPU and CPU steps see the same jets)
  add_soft       Delta N +1: a particle with the jet's softest pT at a random existing particle's position + 0.05 in a random
                 direction (H5v, on-manifold by C2ST)
  add_soft_swap  Delta N 0 control for add_soft: the same added particle, the original softest particle removed
  rmsoft         Delta N -1: remove the softest particle
  split_hard     Delta N +1: hardest particle -> two daughters, z = 0.5, Delta R = 0 (exact collinear split)
  merge_split    Delta N 0 control for split_hard: merge the two softest (pT-weighted position) and split the hardest (dR 0)
  ksplit_k       k = 1, 2, 3 random particles split at z = 0.5, dR 0 (jets with N >= 5); slope per particle
  ptd_up/ptd_dn  (H10 i) pT^D +-0.01 at fixed z1, z2 and positions: z_i of particles 3..N -> proportional to z_i^alpha,
                 renormalised to their sum, alpha solved per jet
  z1_up/z1_dn    (H10 ii) z1 +-0.015 at fixed pT^D, z_{>=5} and positions: z2..z4 re-shared (shape kept, spread solved)
  pair_dr{0,0.005,0.02}  (H10 iii) leading pair moved along its axis about its pT-weighted centroid to separation Delta R,
                 pT list fixed
  ptd34_up/dn    (H10 i, local; second batch) pT^D +-0.005 by moving momentum between particles 3 and 4 only
  pair_rot       control for (iii): leading pair rotated by 90 degrees about its centroid (separation and pT list fixed)
Steps
  --step hwdata  (CPU, conda) download Herwig files 0-19, write parquet (particle_transformer convert_qg_datasets.py columns)
                 for the retrains and per-file multiplicity/label arrays
  --step eval    (GPU) --gen py|hw --set pythia|herwig: per-jet logits, unedited and edited -> physics/h14_logits_<gen>_<set>.npz
  --step efp     (CPU, venv-ef) EFP/N baselines + LR(N) per generator on the same edits, C2ST checks -> h14_logits_<gen>_base.npz,
                 h14_c2st.json
  --step stats   (CPU) statistics over all logit files -> physics/h14_run.json
  --step fig     (CPU) physics/h14_generator.png
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path
import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
QG = V / 'data' / 'qg'
HW = QG / 'herwig'
HWP = HW / 'parquet'
SEED = 9311
CH_HAD = (211, 321, 2212)


# ------------------------------------------------------------------------------------------------------------- Herwig data
def _hw_file(i):
    return HW / f'QG_jets_herwig_{i}.npz'


def step_hwdata(a):
    import urllib.request
    import pyarrow as pa
    import pyarrow.parquet as pq
    HW.mkdir(parents=True, exist_ok=True); HWP.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        dst = _hw_file(i)
        if dst.exists() and dst.stat().st_size > 5e7: continue
        for u in (f'https://zenodo.org/records/3066475/files/QG_jets_herwig_{i}.npz?download=1',
                  f'https://zenodo.org/record/3066475/files/QG_jets_herwig_{i}.npz?download=1'):
            try:
                log('download', u); urllib.request.urlretrieve(u, str(dst) + '.part'); os.replace(str(dst) + '.part', dst)
                log('  ok', dst.stat().st_size); break
            except Exception as e:
                log('  failed', repr(e))
        assert dst.exists(), f'download of herwig file {i} failed'
    meta = dict(N=[], label=[], fid=[])
    for i in range(20):
        name = f'train_file_{i}.parquet' if i < 18 else f'test_file_{i - 18}.parquet'
        with np.load(_hw_file(i)) as d:
            X = d['X'].astype(np.float32); y = d['y'].astype(np.int64)
        o = np.argsort(-X[..., 0], axis=1, kind='stable'); X = np.take_along_axis(X, o[..., None], 1); del o
        mask = X[..., 0] > 0; n = mask.sum(1)
        pt, eta, phi, pid = (X[..., k][mask].astype(np.float64) for k in range(4))
        jid = np.repeat(np.arange(len(n)), n)
        px, py, pz, E = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
        S = lambda v: np.bincount(jid, v, minlength=len(n))
        jx, jy, jz, jE = S(px), S(py), S(pz), S(E)
        jpt = np.hypot(jx, jy); jeta = np.arcsinh(jz / jpt); jphi = np.arctan2(jy, jx)
        jm = np.sqrt(np.maximum(jE ** 2 - jx ** 2 - jy ** 2 - jz ** 2, 0))
        sg = np.sign(jeta); sg[sg == 0] = 1
        deta = (eta - jeta[jid]) * sg[jid]; dphi = (phi - jphi[jid] + np.pi) % (2 * np.pi) - np.pi
        pidi = pid.astype(np.int64)
        chp = np.isin(pidi, CH_HAD) | (pidi == -11) | (pidi == -13); chm = np.isin(pidi, [-c for c in CH_HAD]) | (pidi == 11) | (pidi == 13)
        off = pa.array(np.r_[0, np.cumsum(n)].astype(np.int32))
        L = lambda v: pa.ListArray.from_arrays(off, pa.array(np.asarray(v, np.float32)))
        cols = dict(label=pa.array(y), jet_pt=pa.array(jpt.astype(np.float32)), jet_eta=pa.array(jeta.astype(np.float32)),
                    jet_phi=pa.array(jphi.astype(np.float32)), jet_energy=pa.array(jE.astype(np.float32)),
                    jet_mass=pa.array(jm.astype(np.float32)), jet_nparticles=pa.array(n.astype(np.int64)),
                    part_px=L(px), part_py=L(py), part_pz=L(pz), part_energy=L(E), part_deta=L(deta), part_dphi=L(dphi),
                    part_pid=L(pid), part_isChargedHadron=L(np.isin(np.abs(pidi), CH_HAD)),
                    part_isNeutralHadron=L(np.isin(np.abs(pidi), (130, 2112))), part_isPhoton=L(pidi == 22),
                    part_isElectron=L(np.abs(pidi) == 11), part_isMuon=L(np.abs(pidi) == 13),
                    part_charge=L(chp.astype(np.float32) - chm.astype(np.float32)))
        pq.write_table(pa.table(cols), HWP / name, compression='lz4')
        meta['N'].append(n); meta['label'].append(y); meta['fid'].append(np.full(len(n), i))
        log(f'herwig file {i} -> {name}: {len(n)} jets, quark frac {y.mean():.3f}, <N> q {n[y == 1].mean():.2f} g {n[y == 0].mean():.2f}, '
            f'jet pT [{jpt.min():.1f}, {jpt.max():.1f}], |deta| max {np.abs(deta).max():.3f}')
        del X, mask, pt, eta, phi, pid, px, py, pz, E, cols
    np.savez(HW / 'meta.npz', **{k: np.concatenate(v) for k, v in meta.items()})
    (HWP / 'done').write_text('ok'); log('done')


# --------------------------------------------------------------------------------------------------------------- samples
def load_sample(gen, per_class=20000, seed=SEED):
    """-> (list of (n, 4) arrays [pT, eta(=y), phi, pid], pT-sorted; labels (1 = quark); info)"""
    from interp import h5_qg_counting as H
    if gen == 'py':
        from interp import data_qg
        S = data_qg.sample(per_class, seed); y = S['label'].astype(int)
        return H.to_list(H.from_test(S)), y, dict(source='data_qg test split', idx=S['idx'])
    X, y, fid = [], [], []
    for i in (18, 19):
        with np.load(_hw_file(i)) as d:
            X.append(d['X'].astype(np.float32)); y.append(d['y'].astype(int)); fid.append(np.full(len(d['y']), i))
    L = max(x.shape[1] for x in X)
    X = np.concatenate([np.pad(x, ((0, 0), (0, L - x.shape[1]), (0, 0))) for x in X]); y = np.concatenate(y); fid = np.concatenate(fid)
    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in (0, 1)]))
    x = X[idx]; o = np.argsort(-x[..., 0], axis=1, kind='stable'); x = np.take_along_axis(x, o[..., None], 1)
    return H.to_list(H.from_raw(x)), y[idx], dict(source='herwig files 18-19', idx=idx, file=fid[idx])


# ----------------------------------------------------------------------------------------------------------------- edits
def _rng(gen, name):
    return np.random.default_rng([SEED, zlib.crc32(gen.encode()), zlib.crc32(name.encode())])


def _split0(a, i, z=0.5):
    p = a[i]; d1 = p.copy(); d2 = p.copy(); d1[0] = z * p[0]; d2[0] = (1 - z) * p[0]     # Delta R = 0 (exact collinear)
    return np.concatenate([np.delete(a, i, 0), d1[None], d2[None]])


def _merge2soft(a):
    s1, s2 = a[-2], a[-1]; m = s1.copy(); m[0] = s1[0] + s2[0]
    d = (s2[2] - s1[2] + np.pi) % (2 * np.pi) - np.pi
    m[1] = (s1[0] * s1[1] + s2[0] * s2[1]) / m[0]; m[2] = s1[2] + s2[0] * d / m[0]
    return np.concatenate([a[:-2], m[None]])


def _ptd_edit(a, dptd):
    """(H10 i) pT^D -> pT^D + dptd with z1, z2, positions fixed: z_{3..N} ∝ z^alpha (alpha >= 0), sum kept"""
    pt = a[:, 0]; n = len(a)
    if n < 4: return None
    P = pt.sum(); z = pt / P; r = z[2:]; R = r.sum(); A = (z[:2] ** 2).sum()
    tgt = (np.sqrt((z ** 2).sum()) + dptd) ** 2 - A
    f = lambda al: (R * r ** al / (r ** al).sum()) ** 2
    lo, hi = 0.0, 40.0
    if not (f(lo).sum() <= tgt <= f(hi).sum()): return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid).sum() < tgt: lo = mid
        else: hi = mid
    rn = R * r ** hi / (r ** hi).sum()
    if rn[0] > z[1] or not np.isfinite(rn).all(): return None             # keep the leading pair the two hardest
    b = a.copy(); b[2:, 0] = rn * P
    return b


def _ptd34_edit(a, dptd):
    """(H10 i, local) pT^D -> pT^D + dptd by moving momentum between particles 3 and 4 only (z1, z2, z_{>=5}, positions
    fixed; order kept)"""
    pt = a[:, 0]; n = len(a)
    if n < 4: return None
    P = pt.sum(); z = pt / P; S2 = (z ** 2).sum(); dS2 = (np.sqrt(S2) + dptd) ** 2 - S2; g = z[2] - z[3]
    disc = g * g + 2 * dS2
    if disc < 0: return None
    d = (-g + np.sqrt(disc)) / 2                       # root with the smaller |delta| (2 d^2 + 2 g d - dS2 = 0)
    z3, z4 = z[2] + d, z[3] - d; z5 = z[4] if n > 4 else 0.0
    if not (z4 > 0 and z4 >= z5 and z3 <= z[1] and z3 >= z4): return None
    b = a.copy(); b[2, 0] = z3 * P; b[3, 0] = z4 * P
    return b


def _z1_edit(a, dz):
    """(H10 ii) z1 -> z1 + dz at fixed pT^D, z_{>=5}, positions: z2..z4 re-shared (shape kept, spread lambda solved)"""
    pt = a[:, 0]; n = len(a)
    if n < 4: return None
    P = pt.sum(); z = pt / P; s = z[1:4]; S = s.sum(); z1n = z[0] + dz; Sn = S - dz
    T = (z ** 2).sum() - z1n ** 2 - (z[4:] ** 2).sum(); Vs = ((s - S / 3) ** 2).sum()
    if Sn <= 0 or Vs <= 0: return None
    lam2 = (T - Sn ** 2 / 3) / Vs
    if lam2 < 0: return None
    sn = Sn / 3 + np.sqrt(lam2) * (s - S / 3)
    if (sn <= 0).any() or sn[0] > z1n: return None
    b = a.copy(); b[0, 0] = z1n * P; b[1:4, 0] = sn * P
    return b


def _pair_edit(a, dr=None, rot=False, rng=None):
    """(H10 iii) leading pair: separation set to dr about its pT-weighted centroid (direction kept), or rotated by 90 deg"""
    if len(a) < 2: return None
    p0, p1 = a[0], a[1]; w0, w1 = p0[0], p1[0]
    d = np.array([p0[1] - p1[1], (p0[2] - p1[2] + np.pi) % (2 * np.pi) - np.pi]); sep = np.hypot(*d)
    c = np.array([p1[1], p1[2]]) + w0 / (w0 + w1) * d
    u = d / sep if sep > 0 else np.array([1.0, 0.0])
    if rot: u = np.array([-u[1], u[0]]); dr = sep
    b = a.copy()
    b[0, 1:3] = c + dr * w1 / (w0 + w1) * u; b[1, 1:3] = c - dr * w0 / (w0 + w1) * u
    return b


EDITS = [  # name, dN, family
    ('add_soft', 1, 'count'), ('add_soft_swap', 0, 'count_ctrl'), ('rmsoft', -1, 'count'),
    ('split_hard', 1, 'split'), ('merge_split', 0, 'split_ctrl'),
    ('ksplit_1', 1, 'ksplit'), ('ksplit_2', 2, 'ksplit'), ('ksplit_3', 3, 'ksplit'),
    ('ptd_up', 0, 'h10i'), ('ptd_dn', 0, 'h10i'), ('z1_up', 0, 'h10ii'), ('z1_dn', 0, 'h10ii'),
    ('pair_dr0', 0, 'h10iii'), ('pair_dr0.005', 0, 'h10iii'), ('pair_dr0.02', 0, 'h10iii'), ('pair_rot', 0, 'h10iii_ctrl')]
EDITS_X = [('ptd34_up', 0, 'h10i_local'), ('ptd34_dn', 0, 'h10i_local')]   # second batch ('--edits extra')
DPTD, DZ1, DPTD34 = 0.01, 0.015, 0.005


def make_edits(L0, gen, which='main'):
    """-> dict name -> (edited list, applied mask); unapplied jets are left unedited"""
    out = {}
    if which == 'extra':
        def run1(name, fn):
            rng = _rng(gen, name); Le, ok = [], np.ones(len(L0), bool)
            for j, a in enumerate(L0):
                b = fn(a, rng)
                if b is None: Le.append(a); ok[j] = False
                else: Le.append(b)
            out[name] = ([b[np.argsort(-b[:, 0], kind='stable')] for b in Le], ok)
        run1('ptd34_up', lambda a, r: _ptd34_edit(a, +DPTD34)); run1('ptd34_dn', lambda a, r: _ptd34_edit(a, -DPTD34))
        return out
    def run(name, fn):
        rng = _rng(gen, name); Le, ok = [], np.ones(len(L0), bool)
        for j, a in enumerate(L0):
            b = fn(a, rng)
            if b is None: Le.append(a); ok[j] = False
            else: Le.append(b)
        out[name] = (Le, ok)
    # add_soft and its Delta N = 0 swap use the same added particle
    rng = _rng(gen, 'add_soft'); A, B, ok2 = [], [], np.ones(len(L0), bool)
    for j, a in enumerate(L0):
        i = rng.integers(len(a)); th = rng.uniform(0, 2 * np.pi)
        p = a[i].copy(); p[1] += 0.05 * np.cos(th); p[2] += 0.05 * np.sin(th); p[0] = a[-1, 0]
        A.append(np.concatenate([a, p[None]]))
        if len(a) >= 2: B.append(np.concatenate([a[:-1], p[None]]))
        else: B.append(a); ok2[j] = False
    out['add_soft'] = (A, np.ones(len(L0), bool)); out['add_soft_swap'] = (B, ok2)
    run('rmsoft', lambda a, r: a[:-1] if len(a) >= 3 else None)
    run('split_hard', lambda a, r: _split0(a, 0))
    run('merge_split', lambda a, r: _split0(_merge2soft(a), 0) if len(a) >= 3 else None)
    for k in (1, 2, 3):
        def ks(a, r, k=k):
            if len(a) < 5: return None
            b = a
            for i in np.sort(r.choice(len(a), k, replace=False))[::-1]: b = _split0(b, int(i))
            return b
        run(f'ksplit_{k}', ks)
    run('ptd_up', lambda a, r: _ptd_edit(a, +DPTD)); run('ptd_dn', lambda a, r: _ptd_edit(a, -DPTD))
    run('z1_up', lambda a, r: _z1_edit(a, +DZ1)); run('z1_dn', lambda a, r: _z1_edit(a, -DZ1))
    for d in (0.0, 0.005, 0.02): run(f'pair_dr{d:g}', lambda a, r, d=d: _pair_edit(a, dr=d))
    run('pair_rot', lambda a, r: _pair_edit(a, rot=True))
    for k, (Le, ok) in out.items():                       # keep every edited list pT-sorted
        out[k] = ([b[np.argsort(-b[:, 0], kind='stable')] for b in Le], ok)
    return out


def jet_obs(L):
    """per-jet N, pT^D, z1, z12, dR12, girth, C1^(1), mass (E-scheme axis)"""
    from interp import h5_qg_counting as H
    Jt = H.from_list(L, Lpad=max(len(a) for a in L)); O = H.shapes(Jt)
    z1 = np.array([a[0, 0] / a[:, 0].sum() for a in L]); z12 = np.array([a[:2, 0].sum() / a[:, 0].sum() for a in L])
    dr12 = np.array([np.hypot(a[0, 1] - a[1, 1], (a[0, 2] - a[1, 2] + np.pi) % (2 * np.pi) - np.pi) if len(a) > 1 else 0 for a in L])
    return dict(N=O['N'], ptd=O['ptd'], z1=z1, z12=z12, dr12=dr12, girth=O['girth'], C1=O['C1_1.0'], mass=O['mass'])


# ------------------------------------------------------------------------------------------------------------------ models
def add_weaver_dir(M, name, arch, pidfeat, d):
    """as h5 Models.add_weaver but from directory d (Herwig-trained models); records best epoch and final/interim"""
    import hashlib
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from weaver.nn.model.ParticleNet import ParticleNet
    best = d / 'net_best_epoch_state.pt'; h = hashlib.md5(best.read_bytes()).hexdigest(); ep = None
    eps = sorted(d.glob('net_epoch-*_state.pt'), key=lambda p: int(p.name.split('-')[1].split('_')[0]))
    for f in eps:
        if hashlib.md5(f.read_bytes()).hexdigest() == h: ep = int(f.name.split('-')[1].split('_')[0])
    final = (d / 'predict_output').exists()
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
    net.load_state_dict(M._sd(best), strict=True); net.eval().to(M.dev)
    M.m[name] = ('weaver', arch, pidfeat, net)
    M.info[name] = dict(checkpoint=str(best), best_epoch=ep, epochs_written=len(eps), status='final' if final else 'interim', md5=h)
    log(f'{name}: best epoch {ep} of {len(eps)} written ({M.info[name]["status"]})')


def add_keras_ef(M, kind):
    """H2 energyflow EFN / PFN (models/qg/<kind>/weights.npz, Keras Dense kernels (in, out)) ported into the torch EF module of
    h5_qg_counting.make_ef (same Phi = (100, 100, 256), F = (100, 100, 100), ReLU). The torch module divides angles by 0.4, so
    the angle columns of the first Phi kernel are multiplied by 0.4."""
    import torch
    from interp import h5_qg_counting as H
    d = V / 'models' / 'qg' / kind
    if not (d / 'info.json').exists(): log(f'{kind}_k: no finished H2 model'); return
    W = np.load(d / 'weights.npz'); ks = sorted(W.files)
    kern = [W[k] for k in ks if k.endswith('_0')]; bias = [W[k] for k in ks if k.endswith('_1')]
    net = H.make_ef(kind); lin = [m for m in list(net.phi) + list(net.F) if isinstance(m, torch.nn.Linear)]
    assert len(lin) == len(kern) == 7, (len(lin), len(kern))
    with torch.no_grad():
        for i, (l, k, b) in enumerate(zip(lin, kern, bias)):
            k = k.copy()
            if i == 0: k[-2:] *= 0.4                         # angle rows (efn: both rows; pfn: rows 1, 2 after z)
            assert l.weight.shape == k.T.shape, (i, l.weight.shape, k.shape)
            l.weight.copy_(torch.from_numpy(k.T)); l.bias.copy_(torch.from_numpy(b))
    info = json.loads((d / 'info.json').read_text()); info = dict(status='final (H2 energyflow, ported)', test=info['test'],
                                                                   best_epoch=info['best_epoch'], epochs_run=info['epochs_run'])
    M.add_torch(f'{kind}_k', net, info); log(f'{kind}_k loaded: H2 test {info["test"]}')


def load_set(dev, which):
    from interp import h5_qg_counting as H
    if which == 'keras':
        M = H.Models(dev)
        for kind in ('efn', 'pfn'): add_keras_ef(M, kind)
        return M
    if which == 'pythia':
        M = H.load_models(dev)                    # part/pnet kin/kinpid, lorentznet, part_kin_perm, efn_t, pfn_t
        return M
    M = H.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        d = V / 'models' / 'qg_herwig' / name
        if (d / 'net_best_epoch_state.pt').exists(): add_weaver_dir(M, 'hw_' + name, arch, False, d)
        else: log(f'hw_{name}: no checkpoint yet')
    return M


def step_eval(a):
    import torch
    from interp import h5_qg_counting as H
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    L0, y, info = load_sample(a.gen, a.per_class)
    log(f'{a.gen}: {len(y)} jets, quark frac {y.mean():.3f}, <N> q {np.mean([len(x) for x, c in zip(L0, y) if c == 1]):.2f} '
        f'g {np.mean([len(x) for x, c in zip(L0, y) if c == 0]):.2f}; device {dev}')
    E = make_edits(L0, a.gen, a.edits); log('edits made:', {k: int(v[1].sum()) for k, v in E.items()})
    M = load_set(dev, a.set); names = list(M.m)
    out = dict(y=y, idx=info['idx'])
    Ju = H.from_list(L0, Lpad=180)
    for nm in names:
        out[f'base|{nm}'] = M.logit(nm, Ju).astype(np.float32); log(nm, 'unedited', H.roc_metrics(out[f'base|{nm}'], y))
    for en, (Le, ok) in E.items():
        Je = H.from_list(Le, Lpad=180); out[f'ok|{en}'] = ok
        for nm in names:
            out[f'{en}|{nm}'] = M.logit(nm, Je).astype(np.float32)
        log(en, 'applied', int(ok.sum()), '; '.join(f'{nm} {np.mean((out[f"{en}|{nm}"] - out[f"base|{nm}"])[ok]):+.4f}' for nm in names))
    if a.set == 'pythia':                                   # per-jet observables, once per generator
        O0 = jet_obs(L0); out.update({f'obs|base|{k}': v.astype(np.float32) for k, v in O0.items()})
        for en, (Le, ok) in E.items():
            Oe = jet_obs(Le); out.update({f'obs|{en}|{k}': v.astype(np.float32) for k, v in Oe.items()})
        log('observables done')
    tag = ('_x' if a.edits == 'extra' else '') + ('_smoke' if a.per_class < 20000 else '')
    np.savez_compressed(RES / f'h14_logits_{a.gen}_{a.set}{tag}.npz', **out)
    (RES / f'h14_models_{a.gen}_{a.set}{tag}.json').write_text(json.dumps(M.info, indent=1, default=str))
    log('wrote', RES / f'h14_logits_{a.gen}_{a.set}{tag}.npz')


# --------------------------------------------------------------------------------------------------------- CPU baselines
def train_N(gen):
    """multiplicity and label of the generator's training jets (Pythia: non-val train rows; Herwig: files 0-17)"""
    if gen == 'py':
        d = QG / 'train'; y = np.load(d / 'label.npy'); isv = np.load(d / 'is_val.npy')
        n = np.asarray(np.load(d / 'jet.npy', mmap_mode='r')[:, 5]).astype(int)
        return n[~isv], y[~isv].astype(int)
    m = np.load(HW / 'meta.npz'); tr = m['fid'] < 18
    return m['N'][tr].astype(int), m['label'][tr].astype(int)


def nb_fit(n):
    """negative binomial MLE (r, p) for counts n, starting from moments"""
    from scipy.optimize import minimize
    from scipy.stats import nbinom
    mu, var = n.mean(), n.var(); p0 = min(mu / var, 0.99); r0 = mu * p0 / (1 - p0)
    nll = lambda th: -nbinom.logpmf(n, np.exp(th[0]), 1 / (1 + np.exp(-th[1]))).sum()
    th = minimize(nll, [np.log(r0), np.log(p0 / (1 - p0))], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-3, maxiter=2000)).x
    return float(np.exp(th[0])), float(1 / (1 + np.exp(-th[1])))


def lrn_tables():
    """per generator: NB fit per class -> log LR(N) = log NB_g(N) - log NB_q(N), N = 0..299; plus histogram LR"""
    from scipy.stats import nbinom
    T, info = {}, {}
    for gen in ('py', 'hw'):
        n, y = train_N(gen); Ns = np.arange(300)
        fq, fg = nb_fit(n[y == 1]), nb_fit(n[y == 0])
        T[gen] = nbinom.logpmf(Ns, *fg) - nbinom.logpmf(Ns, *fq)
        hq = np.bincount(n[y == 1], minlength=300)[:300] + 0.5; hg = np.bincount(n[y == 0], minlength=300)[:300] + 0.5
        T[gen + '_hist'] = np.log(hg / hg.sum()) - np.log(hq / hq.sum())
        info[gen] = dict(n_train=int(len(n)), nb_q=dict(r=fq[0], p=fq[1]), nb_g=dict(r=fg[0], p=fg[1]),
                         mean_q=float(n[y == 1].mean()), sd_q=float(n[y == 1].std()), mean_g=float(n[y == 0].mean()),
                         sd_g=float(n[y == 0].std()))
        log('LR(N)', gen, info[gen])
    return T, info


_EFPS = None


def _efp_chunk(L):
    global _EFPS
    import energyflow as ef
    if _EFPS is None: _EFPS = ef.EFPSet('d<=5', 'p==1', measure='hadr', beta=1, normed=True, coords='ptyphim', check_input=False)
    out = []
    for a in L:                       # recentre on the hardest particle (phi wrapped) so angular distances never cross 2 pi
        b = np.array(a[:, :3], np.float64); b[:, 1] -= b[0, 1]; b[:, 2] = (b[:, 2] - b[0, 2] + np.pi) % (2 * np.pi) - np.pi
        out.append(_EFPS.compute(b))
    return np.array(out)


def efps(L, pool):
    ch = [L[s:s + 200] for s in range(0, len(L), 200)]
    return np.concatenate(pool.map(_efp_chunk, ch))


def efp_train_lists(gen, n_per_class=30000, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    if gen == 'py':
        d = QG / 'train'; y = np.load(d / 'label.npy'); isv = np.load(d / 'is_val.npy')
        idx = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & ~isv), n_per_class, replace=False) for c in (0, 1)]))
        k3 = np.asarray(np.load(d / 'kin3.npy', mmap_mode='r')[idx], np.float64)
        N = np.asarray(np.load(d / 'jet.npy', mmap_mode='r')[idx, 5])
        L = [k[k[:, 0] > 0] for k in k3]                          # (pT, deta, dphi): EFPs are translation invariant
        return L, y[idx].astype(int), N.astype(float)
    m = np.load(HW / 'meta.npz'); tr = np.flatnonzero(m['fid'] < 18); y = m['label']
    idx = np.sort(np.concatenate([rng.choice(tr[y[tr] == c], n_per_class, replace=False) for c in (0, 1)]))
    L, ys, N = [], [], []
    for f in range(18):
        rows = idx[m['fid'][idx] == f] - int(np.flatnonzero(m['fid'] == f)[0])
        if len(rows) == 0: continue
        with np.load(_hw_file(f)) as d: X = d['X'][rows]; yy = d['y'][rows]
        for x in X: x = x[x[:, 0] > 0]; L.append(x[np.argsort(-x[:, 0])][:, :3])
        ys.append(yy); N.append((X[..., 0] > 0).sum(1))
    return L, np.concatenate(ys).astype(int), np.concatenate(N).astype(float)


def step_efp(a):
    """baselines on the same edits: LR(N) NB per generator, GBDT(EFP d<=5 beta 1 + N), GBDT(EFP); C2ST checks"""
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h5_qg_counting as H
    RES.mkdir(parents=True, exist_ok=True)
    T, lrinfo = lrn_tables(); pool = Pool(a.cpu)
    feat = lambda F, N, wn: np.c_[np.log(np.maximum(F, 1e-12)), N] if wn else np.log(np.maximum(F, 1e-12))
    clf = {}
    for gen in ('py', 'hw'):
        Lt, yt, Nt = efp_train_lists(gen, n_per_class=2000 if a.smoke else 30000)
        Ft = efps(Lt, pool); g = (yt == 0).astype(int)
        for wn in (True, False):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                                               random_state=0).fit(feat(Ft, Nt, wn), g)
            clf[('efpn' if wn else 'efp') + '_' + gen] = m
        log(f'EFP GBDTs trained on {gen} ({len(yt)} jets, {Ft.shape[1]} EFPs)')
    c2 = {}
    for gen in ('py', 'hw'):
        L0, y, info = load_sample(gen, a.per_class); E = make_edits(L0, gen, a.edits)
        out = dict(y=y)
        conds = [('base', L0, np.ones(len(L0), bool))] + [(k, v[0], v[1]) for k, v in E.items()]
        for cn, Lx, ok in conds:
            F = efps(Lx, pool); N = np.array([len(x) for x in Lx], float)
            for nm, m in clf.items(): out[f'{cn}|{nm}'] = m.decision_function(feat(F, N, nm.startswith('efpn'))).astype(np.float32)
            for tg in ('py', 'hw'):
                out[f'{cn}|lrn_{tg}'] = T[tg][np.minimum(N.astype(int), 299)].astype(np.float32)
                out[f'{cn}|lrnh_{tg}'] = T[tg + '_hist'][np.minimum(N.astype(int), 299)].astype(np.float32)
            if cn != 'base': out[f'ok|{cn}'] = ok
            log(gen, cn, 'baselines done', '; '.join(f'{k.split("|")[1]} {np.mean((out[k] - out["base|" + k.split("|")[1]])[ok]):+.4f}'
                                                     for k in out if k.startswith(cn + '|')) if cn != 'base' else '')
        tag = ('_x' if a.edits == 'extra' else '') + ('_smoke' if a.smoke else '')
        np.savez_compressed(RES / f'h14_logits_{gen}_base{tag}.npz', **out)
        rs = np.random.default_rng(5); sub = np.sort(rs.choice(len(L0), min(10000, len(L0)), replace=False))
        Ls = [L0[i] for i in sub]; c2[gen] = {'none': H.c2st(Ls, Ls)}
        for en in [e for e in ('add_soft', 'add_soft_swap', 'rmsoft', 'split_hard', 'ptd_up', 'ptd_dn', 'z1_up', 'z1_dn',
                               'pair_dr0', 'pair_dr0.02', 'pair_rot', 'ptd34_up', 'ptd34_dn') if e in E]:
            c2[gen][en] = H.c2st(Ls, [E[en][0][i] for i in sub]); log('c2st', gen, en, c2[gen][en])
    (RES / f'h14_baselines{"_x" if a.edits == "extra" else ""}{"_smoke" if a.smoke else ""}.json').write_text(json.dumps(dict(lrn=lrinfo, c2st=c2), indent=1))
    pool.close(); log('efp step done')


# -------------------------------------------------------------------------------------------------------------- statistics
class Boot:
    """shared bootstrap resamples (B x J) and sign-flip matrix (P x J) for all tests on one sample (common random numbers)"""
    def __init__(self, J, B=1000, P=2000, seed=SEED):
        r = np.random.default_rng(seed); self.idx = r.integers(0, J, (B, J), dtype=np.int32)
        self.sg = r.choice(np.array([-1.0, 1.0], np.float32), (P, J)); self.J = J

    def mean_ci(self, d, ok):
        dd = np.where(ok, d, 0.0).astype(np.float64); w = ok.astype(np.float64); s = []
        for c in range(0, len(self.idx), 250):
            ix = self.idx[c:c + 250]; s.append(dd[ix].sum(1) / np.maximum(w[ix].sum(1), 1))
        s = np.concatenate(s); return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))], s

    def signflip(self, d, ok):
        dd = np.where(ok, d, 0.0).astype(np.float32); obs = abs(dd.sum())
        cnt = sum(int((np.abs(self.sg[c:c + 500] @ dd) >= obs - 1e-9).sum()) for c in range(0, len(self.sg), 500))
        return (cnt + 1) / (len(self.sg) + 1)


def _tp(d):
    from scipy.stats import ttest_1samp
    return float(ttest_1samp(d, 0).pvalue) if d.std() > 0 else (1.0 if abs(d.mean()) < 1e-12 else 0.0)


def load_gen(gen, tag=''):
    """merge the logit files of one generator -> (y, base{model}, edit{(edit, model)}, ok{edit}, obs{(cond, key)}, model info)"""
    base, ed, ok, obs, info = {}, {}, {}, {}, {}
    y = None
    for st in ('pythia', 'herwig', 'keras', 'base', 'pythia_x', 'herwig_x', 'keras_x', 'base_x'):
        f = RES / f'h14_logits_{gen}_{st}{tag}.npz'
        if not f.exists(): log('missing', f); continue
        Z = np.load(f)
        if y is None: y = Z['y'].astype(int)
        assert (Z['y'] == y).all(), f
        fb = {k.split('|')[1]: Z[k].astype(np.float64) for k in Z.files if k.startswith('base|')}
        for m, v in fb.items():
            if m not in base: base[m] = v
        for k in Z.files:
            p = k.split('|')
            if p[0] == 'ok':
                if p[1] in ok: assert (ok[p[1]] == Z[k]).all(), f'edit mask mismatch {k} in {f}'
                ok[p[1]] = Z[k].astype(bool)
            elif p[0] == 'obs':
                if (p[1], p[2]) not in obs: obs[(p[1], p[2])] = Z[k].astype(np.float64)
            elif p[0] == 'base': continue
            elif len(p) == 2 and p[0] not in ('y', 'idx'):    # delta against this file's own unedited logits, re-anchored
                ed[(p[0], p[1])] = Z[k].astype(np.float64) - fb[p[1]] + base[p[1]]
        mj = RES / f'h14_models_{gen}_{st}{tag}.json'
        if mj.exists(): info.update(json.loads(mj.read_text()))
    return y, base, ed, ok, obs, info


TRAINED_ON = lambda m: 'herwig' if (m.startswith('hw_') or m.endswith('_hw')) else 'pythia'
NETS = ('part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_t', 'efn_k', 'pfn_k', 'hw_part_kin', 'hw_pnet_kin')


def step_stats(a):
    from interp import h5_qg_counting as H
    tag = '_smoke' if a.smoke else ''
    BL = json.loads((RES / f'h14_baselines{tag}.json').read_text()) if (RES / f'h14_baselines{tag}.json').exists() else {}
    R = dict(sample=dict(per_class=a.per_class, seed=SEED, pythia='data_qg test, random rows', herwig='Zenodo 3066475 files 18-19, random rows'),
             dptd=DPTD, dz1=DZ1, lrn=BL.get('lrn'), c2st=BL.get('c2st'), gens={})
    tests = []
    for gen in ('py', 'hw'):
        y, base, ed, ok, obs, info = load_gen(gen, tag)
        if y is None: continue
        J = len(y); Bt = Boot(J); G = dict(J=J, quark_frac=float(y.mean()), models_info=info, unedited={}, edits={}, compare={},
                                            ksplit={}, h10={}, split_decomp={}, obs={})
        models = [m for m in base]
        ALL = [x for x in EDITS + EDITS_X if x[0] in ok]
        dN = {e: d for e, d, _ in ALL}
        # observables: mean change per edit
        if ('base', 'N') in obs:
            for e, _, _ in ALL:
                o = ok[e]
                G['obs'][e] = {k: dict(mean_delta=float(np.mean((obs[(e, k)] - obs[('base', k)])[o])),
                                       mean_abs_delta=float(np.mean(np.abs(obs[(e, k)] - obs[('base', k)])[o])))
                               for k in ('N', 'ptd', 'z1', 'z12', 'dr12', 'girth', 'C1', 'mass')}
                G['obs'][e]['applied'] = int(o.sum())
            G['obs']['base'] = {k: dict(q=float(obs[('base', k)][y == 1].mean()), g=float(obs[('base', k)][y == 0].mean()))
                                for k in ('N', 'ptd', 'z1', 'z12', 'dr12', 'girth', 'C1', 'mass')}
        for m in models:
            G['unedited'][m] = dict(**H.roc_metrics(base[m], y), AUC_ci=H.boot_auc(base[m], y), trained_on=TRAINED_ON(m),
                                    mean_logit_q=float(base[m][y == 1].mean()), mean_logit_g=float(base[m][y == 0].mean()))
            log(gen, m, G['unedited'][m])
        for e, _, fam in ALL:
            G['edits'][e] = dict(dN=dN[e], family=fam, applied=int(ok[e].sum()), models={})
            for m in models:
                if (e, m) not in ed: continue
                d = ed[(e, m)] - base[m]; o = ok[e]; dv = d[o]
                ci, _ = Bt.mean_ci(d, o)
                r = dict(mean=float(dv.mean()), ci=ci, p_t=_tp(dv), p_perm=Bt.signflip(d, o),
                         by_class=dict(q=float(dv[y[o] == 1].mean()), g=float(dv[y[o] == 0].mean())))
                if dN[e]: r['per_dN'] = float(dv.mean() / dN[e])
                if e in ('add_soft', 'rmsoft', 'split_hard', 'merge_split', 'add_soft_swap', 'ptd_up', 'z1_up', 'pair_dr0'):
                    w = H.wp_effects(base[m][o], ed[(e, m)][o], y[o]); r['calls_changed_pct'] = w['calls_changed_pct']; r['mcnemar'] = w['mcnemar']
                G['edits'][e]['models'][m] = r; tests.append((gen, e, m, 'edit = 0', r['p_t']))
            log(gen, e, '; '.join(f'{m} {G["edits"][e]["models"][m]["mean"]:+.4f}' for m in G['edits'][e]['models']))
        # Delta N = 0 controls: paired difference edit - control
        for e1, e2 in (('add_soft', 'add_soft_swap'), ('split_hard', 'merge_split'), ('pair_dr0', 'pair_rot')):
            o = ok[e1] & ok[e2]; G['compare'][f'{e1} - {e2}'] = {}
            for m in models:
                if (e1, m) not in ed: continue
                d = ed[(e1, m)] - ed[(e2, m)]; ci, _ = Bt.mean_ci(d, o)
                c1, _ = Bt.mean_ci(ed[(e2, m)] - base[m], o)
                r = dict(n=int(o.sum()), edit=float((ed[(e1, m)] - base[m])[o].mean()), control=float((ed[(e2, m)] - base[m])[o].mean()),
                         control_ci=c1, diff=float(d[o].mean()), ci=ci, p_t=_tp(d[o]), p_perm=Bt.signflip(d, o))
                G['compare'][f'{e1} - {e2}'][m] = r; tests.append((gen, f'{e1} - {e2}', m, 'edit = control', r['p_t']))
        # H1: network response to a count edit vs the count-only LR(N) of the same generator (paired per jet)
        for e in ('add_soft', 'rmsoft'):
            ref = f'lrn_{gen}'; o = ok[e]; key = f'{e}: model - {ref}'; G['compare'][key] = {}
            if (e, ref) not in ed: continue
            dr = ed[(e, ref)] - base[ref]
            for m in models:
                if (e, m) not in ed or m.startswith('lrn'): continue
                d = (ed[(e, m)] - base[m]) - dr; ci, _ = Bt.mean_ci(d, o)
                r = dict(n=int(o.sum()), model=float((ed[(e, m)] - base[m])[o].mean()), lrn=float(dr[o].mean()),
                         diff=float(d[o].mean()), ci=ci, p_t=_tp(d[o]), p_perm=Bt.signflip(d, o))
                G['compare'][key][m] = r; tests.append((gen, key, m, 'edit response = LR(N) response', r['p_t']))
        # random-split slope k = 1..3 (same jets)
        o = ok['ksplit_1']; ks = np.array([1, 2, 3])
        for m in models:
            if ('ksplit_1', m) not in ed: continue
            D = np.stack([(ed[(f'ksplit_{k}', m)] - base[m]) for k in ks], 1)
            mu = D[o].mean(0); sl = float(np.polyfit(ks, mu, 1)[0]); bs = []
            for c in range(0, 1000, 250):
                ix = Bt.idx[c:c + 250]; w = o[ix].astype(float)
                M_ = np.stack([(np.where(o, D[:, k], 0)[ix]).sum(1) / w.sum(1) for k in range(3)], 1)
                bs.append(np.polyfit(ks, M_.T, 1)[0])
            bs = np.concatenate(bs)
            G['ksplit'][m] = dict(means=mu.tolist(), slope=sl, slope_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        # H10 coefficients: logit per unit pT^D at fixed z1 (arm i), per unit z1 at fixed pT^D (arm ii)
        if ('base', 'ptd') in obs:
            arms = dict(ptd=('ptd_up', 'ptd_dn', 'ptd'), z1=('z1_up', 'z1_dn', 'z1'), ptd34=('ptd34_up', 'ptd34_dn', 'ptd'))
            for arm, (eu, edn, key) in arms.items():
                if eu not in ok: continue
                o = ok[eu] & ok[edn]; xu = obs[(eu, key)] - obs[('base', key)]; xd = obs[(edn, key)] - obs[('base', key)]
                held = 'z1' if key == 'ptd' else 'ptd'
                hu = obs[(eu, held)] - obs[('base', held)]
                G['h10'][arm] = dict(n=int(o.sum()), dx_up=float(xu[o].mean()), dx_dn=float(xd[o].mean()),
                                     held=held, held_max_abs=float(np.abs(hu[o]).max()),
                                     dgirth_up=float((obs[(eu, 'girth')] - obs[('base', 'girth')])[o].mean()),
                                     dC1_up=float((obs[(eu, 'C1')] - obs[('base', 'C1')])[o].mean()), models={})
                for m in models:
                    if (eu, m) not in ed: continue
                    du, dd_ = ed[(eu, m)] - base[m], ed[(edn, m)] - base[m]
                    num = np.where(o, du - dd_, 0); den = np.where(o, xu - xd, 0); bs = []
                    for c in range(0, 1000, 250):
                        ix = Bt.idx[c:c + 250]; bs.append(num[ix].sum(1) / den[ix].sum(1))
                    bs = np.concatenate(bs); beta = float(num.sum() / den.sum())
                    ou, od = ok[eu], ok[edn]
                    G['h10'][arm]['models'][m] = dict(beta=beta, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                      up=float(du[o].mean()), dn=float(dd_[o].mean()),
                                                      beta_up_own=float(du[ou].mean() / xu[ou].mean()), n_up=int(ou.sum()),
                                                      beta_dn_own=float(dd_[od].mean() / xd[od].mean()), n_dn=int(od.sum()))
            # predicted split_hard response from the two coefficients (linear in the observed changes) vs measured
            o = ok['split_hard']
            dp_s = (obs[('split_hard', 'ptd')] - obs[('base', 'ptd')]); dz_s = (obs[('split_hard', 'z1')] - obs[('base', 'z1')])
            for m in models:
                if m not in G['h10']['ptd']['models']: continue
                bp, bz = G['h10']['ptd']['models'][m]['beta'], G['h10']['z1']['models'][m]['beta']
                pair0 = G['edits']['pair_dr0']['models'][m]['mean']
                meas = G['edits']['split_hard']['models'][m]['mean']
                b34 = G['h10'].get('ptd34', {}).get('models', {}).get(m, {}).get('beta')
                G['split_decomp'][m] = dict(measured=meas, ptd_term=float(bp * dp_s[o].mean()), z1_term=float(bz * dz_s[o].mean()),
                                            ptd34_term=float(b34 * dp_s[o].mean()) if b34 is not None else None,
                                            pair_dr0_term=pair0, dptd=float(dp_s[o].mean()), dz1=float(dz_s[o].mean()))
        R['gens'][gen] = G
    adj = H.holm([t[4] for t in tests])
    R['holm'] = dict(n_tests=len(tests), n_pass_005=int((adj < 0.05).sum()),
                     tests=[dict(gen=t[0], edit=t[1], model=t[2], h0=t[3], p=t[4], holm=float(h)) for t, h in zip(tests, adj)])
    (RES / f'h14_run{tag}.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote', RES / f'h14_run{tag}.json')


def step_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((RES / 'h14_run.json').read_text())
    lab = dict(part_kin='ParT kin', part_kinpid='ParT kinpid', pnet_kin='PN kin', pnet_kinpid='PN kinpid', lorentznet='LorentzNet',
               pfn_t='PFN (H5, 400k jets)', efn_t='EFN (H5, 400k jets)', efn_k='EFN (IRC-safe, H2)', pfn_k='PFN (H2)', hw_part_kin='ParT kin (Herwig-trained)', hw_pnet_kin='PN kin (Herwig-trained)',
               efpn_py='GBDT EFP+N (Pythia)', efpn_hw='GBDT EFP+N (Herwig)', efp_py='GBDT EFP (Pythia)', efp_hw='GBDT EFP (Herwig)',
               lrn_py='LR(N) neg. binomial (Pythia)', lrn_hw='LR(N) neg. binomial (Herwig)')
    rows = [m for m in lab if m in R['gens']['py']['edits']['add_soft']['models']]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 6), sharey=True)
    for k, (e, title) in enumerate((('add_soft', 'add one soft particle (ΔN = +1, realistic position)'),
                                    ('split_hard', 'split hardest particle, z = 0.5, ΔR = 0 (ΔN = +1)'))):
        for i, (gen, c, nm) in enumerate((('py', '#1f77b4', 'Pythia jets'), ('hw', '#d62728', 'Herwig jets'))):
            if gen not in R['gens']: continue
            M = R['gens'][gen]['edits'][e]['models']
            mu = np.array([M[m]['mean'] if m in M else np.nan for m in rows]); ci = np.array([M[m]['ci'] if m in M else [np.nan] * 2 for m in rows])
            ax[k].barh(np.arange(len(rows)) + (i - 0.5) * 0.38, mu, height=0.38, color=c, label=nm,
                       xerr=[mu - ci[:, 0], ci[:, 1] - mu], error_kw=dict(lw=0.7))
        ax[k].axvline(0, color='k', lw=0.6); ax[k].set_title(title, fontsize=9.5)
        ax[k].set_xlabel('mean Δ logit = Δ[log p(gluon) − log p(quark)], 95 % CI')
    ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([lab[m] for m in rows], fontsize=8.5); ax[0].invert_yaxis()
    ax[0].legend(fontsize=8.5, loc='lower right')
    fig.suptitle(f'Quark–gluon taggers and baselines on Pythia and Herwig jets ({R["sample"]["per_class"]} quark + {R["sample"]["per_class"]} gluon jets each): '
                 'mean logit change per edit', fontsize=10)
    fig.tight_layout(); fig.savefig(RES / 'h14_generator.png', dpi=130); log('wrote h14_generator.png')
    # hard-core coefficients, per +0.01 of the edited quantity (Pythia jets)
    G = R['gens']['py']
    if 'ptd' in G['h10']:
        rows2 = [m for m in ('part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_t', 'efn_k', 'efpn_py', 'efp_py',
                             'part_kin_perm') if m in G['h10']['ptd']['models']]
        lab['part_kin_perm'] = 'ParT kin, permuted weights'
        arms = (('ptd', 'o', '#1f77b4', 'p_T^D +0.01, momentum re-shared among particles 3..N (z1, z2 fixed; jet girth -0.0045)'),
                ('ptd34', 's', '#2ca02c', 'p_T^D +0.01, momentum moved between particles 3 and 4 only (girth change < 0.0002)'),
                ('z1', '^', '#d62728', 'z1 +0.01 at fixed p_T^D (particles 2-4 re-shared)'))
        fig, ax = plt.subplots(figsize=(9, 6))
        for j, (arm, mk, c, nm) in enumerate(arms):
            if arm not in G['h10']: continue
            M = G['h10'][arm]['models']
            b = 0.01 * np.array([M[m]['beta'] for m in rows2]); ci = 0.01 * np.array([M[m]['ci'] for m in rows2])
            ax.errorbar(b, np.arange(len(rows2)) + (j - 1) * 0.22, xerr=[b - ci[:, 0], ci[:, 1] - b], fmt=mk, color=c, ms=5, lw=1,
                        label=f'{nm} (n = {G["h10"][arm]["n"]})')
        ax.axvline(0, color='k', lw=0.6); ax.set_yticks(range(len(rows2))); ax.set_yticklabels([lab[m] for m in rows2], fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel('Δ logit = Δ[log p(gluon) − log p(quark)] per +0.01 change, 95 % CI (Pythia jets)')
        ax.legend(fontsize=7.5, loc='lower left')
        ax.set_title('Logit change per +0.01 of p_T^D or of the leading-particle fraction z1, particle count and positions fixed', fontsize=9.5)
        fig.tight_layout(); fig.savefig(RES / 'h14_hardcore.png', dpi=130); log('wrote h14_hardcore.png')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['hwdata', 'eval', 'efp', 'stats', 'fig'])
    ap.add_argument('--gen', default='py', choices=['py', 'hw']); ap.add_argument('--set', default='pythia', choices=['pythia', 'herwig', 'keras'])
    ap.add_argument('--per_class', type=int, default=20000); ap.add_argument('--cpu', type=int, default=4)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--edits', default='main', choices=['main', 'extra'])
    a = ap.parse_args()
    dict(hwdata=step_hwdata, eval=step_eval, efp=step_efp, stats=step_stats, fig=step_fig)[a.step](a)
