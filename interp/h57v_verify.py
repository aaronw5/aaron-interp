"""H57 verification (loop D): fresh-seed regeneration of V0, NA-, PA-, IR+ (new worker ids 5761-5764), an input-level split
of PA- into its string-knob part and its stable-pi0 part, and count-like competing observables.

Why:
  (1) Replicate the H57 shifts (ParT_kin, ParticleNet_kin, PFN, EFN, LR(N), LR(pT^D)) on new events.
  (2) PA- = softer string fragmentation + pi0 kept stable. The stable pi0 is a merged collinear photon pair. Here every stable
      pi0 of the fresh PA- jets is decayed by hand (pi0 -> gamma gamma, isotropic in the pi0 rest frame, four-momentum
      conserved) -> PAmdec. V0 -> PAmdec is the string-knob part (with pi0 decays on), PAmdec -> PAm (same jets, the photon
      pairs merged back) is the pi0-merge part. Each part is compared with LR(N) and LR(pT^D).
  (3) Competing readings of "count": the networks could count particles above a pT threshold or at finite angular
      resolution instead of the raw N. Histogram likelihood ratios trained on KMT train jets for N(pT > 1, 2, 4 GeV) and
      N_res(delta) (number of single-linkage clusters of constituents at Delta R < delta = 0.01, 0.03) are scored on the
      fresh settings and on all 38 H57 fit settings (H57 network scores reused), and the fit
      ds_net = gamma ds_EFN + beta ds_LR(X) is compared across X.

Steps
  --step gen    (CPU 8, env-gen)  -> data/qg/gen/h57v/<s>.npz, physics/h57_verify_gen.json
  --step score  (GPU, env-part)   PAmdec built and saved; nets on V0, NAm, PAm, IRp, PAmdec -> physics/h57_verify_scores.npz
  --step stats  (CPU)             -> physics/h57_verify.json
Score convention: s = log p(quark) - log p(gluon).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from interp import h57_qg_count_vs_ptd as H57

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
VG = V / 'data' / 'qg' / 'gen' / 'h57v'
W_V = 5761
SETS = ['V0', 'NAm', 'PAm', 'IRp']
M_PI0 = 0.1349768
MODELS = ['part_kin', 'pnet_kin', 'pfn_k', 'efn_k', 'hw_part_kin', 'perm0']
NETS = ['part_kin', 'pnet_kin', 'pfn_k']
ALT = ['N', 'ptd', 'N_pt1', 'N_pt2', 'N_pt4', 'Nres_0.01', 'Nres_0.03']


# ------------------------------------------------------------------------------------------------------------------- gen
def step_gen(a):
    from multiprocessing import Pool
    g = H57.G(); sp = H57.final_specs(); VG.mkdir(parents=True, exist_ok=True); tmp = VG / 'tmp'; tmp.mkdir(exist_ok=True)
    per = a.jets // 2; tasks = []
    for nm in SETS:
        if (VG / f'{nm}.npz').exists(): continue
        for cls in (0, 1):
            for w in range(2):
                tasks.append(dict(cls=cls, widx=W_V + 2 * cls + w, extra=H57.extra_of(sp[nm]), n_jets=per, max_events=40_000,
                                  out=str(tmp / f'{nm}_{cls}_{w}.npz'), hash_events=100, name=nm))
    log('settings', {k: H57.extra_of(sp[k]) for k in SETS})
    with Pool(a.cpu) as pool: summ = pool.map(g.gen_worker, tasks)
    info = {}
    for nm in SETS:
        tt = [t for t in tasks if t['name'] == nm]
        if not tt: continue
        Z = [np.load(t['out']) for t in tt]; Lmax = max(z['X'].shape[1] for z in Z)
        X = np.concatenate([np.pad(z['X'], ((0, 0), (0, Lmax - z['X'].shape[1]), (0, 0))) for z in Z])
        np.savez_compressed(VG / f'{nm}.npz', X=X, n=np.concatenate([z['n'] for z in Z]), y=np.concatenate([z['y'] for z in Z]),
                            event=np.concatenate([z['event'] for z in Z]), widx=np.concatenate([z['widx'] for z in Z]),
                            jet=np.concatenate([z['jet'] for z in Z]), hashes=np.concatenate([z['hashes'] for z in Z]),
                            hash_widx=np.concatenate([np.full(len(z['hashes']), t['widx']) for z, t in zip(Z, tt)]))
        info[nm] = dict(extra=H57.extra_of(sp[nm]), jets=int(len(X)), workers=[s for s, t in zip(summ, tasks) if t['name'] == nm])
        for t in tt: os.remove(t['out'])
        log('wrote', nm, len(X))
    Z0 = np.load(VG / 'V0.npz'); h0 = {(int(w), int(h[0])): (h[1], h[2]) for w, h in zip(Z0['hash_widx'], Z0['hashes'])}
    for nm in SETS:
        Z1 = np.load(VG / f'{nm}.npz'); h1 = {(int(w), int(h[0])): (h[1], h[2]) for w, h in zip(Z1['hash_widx'], Z1['hashes'])}
        com = sorted(set(h0) & set(h1)); info.setdefault(nm, {})['parton_identical_frac'] = float(np.mean([h0[k] == h1[k] for k in com]))
        info[nm]['parton_hash_n'] = len(com); log(nm, 'parton identical to V0 in', info[nm]['parton_identical_frac'], 'of', len(com))
    (RES / 'h57_verify_gen.json').write_text(json.dumps(info, indent=1, default=str))


# ------------------------------------------------------------------------------------------------------- pi0 decay by hand
def decay_pi0(a, rng):
    """(n,4) (pt, y, phi, pid) with stable pi0 (|pid| 111) -> each pi0 replaced by two photons (isotropic two-body decay in the
    pi0 rest frame, boosted to the lab; four-momentum conserved). Returns new array (pT-sorted), number decayed, max |dp4|."""
    m = np.abs(a[:, 3]) == 111
    if not m.any(): return a.copy(), 0, 0.0
    P = a[m]; k = len(P); pt, y, phi = P[:, 0], P[:, 1], P[:, 2]
    mt = np.sqrt(pt ** 2 + M_PI0 ** 2); E = mt * np.cosh(y); p = np.c_[pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y)]
    beta = p / E[:, None]; b2 = (beta ** 2).sum(1); gam = E / M_PI0
    ct = rng.uniform(-1, 1, k); ph = rng.uniform(0, 2 * np.pi, k); st = np.sqrt(1 - ct ** 2)
    n = np.c_[st * np.cos(ph), st * np.sin(ph), ct]; e = M_PI0 / 2
    out = []; s4 = np.zeros((k, 4))
    for sg in (1, -1):
        ps = sg * e * n; bp = (beta * ps).sum(1)
        pl = ps + ((gam - 1) * bp / np.maximum(b2, 1e-300) + gam * e)[:, None] * beta; El = gam * (e + bp)
        s4 += np.c_[El, pl]
        gpt = np.hypot(pl[:, 0], pl[:, 1]); gy = 0.5 * np.log((El + pl[:, 2]) / (El - pl[:, 2]))
        gphi = np.arctan2(pl[:, 1], pl[:, 0]); gphi = phi + (gphi - phi + np.pi) % (2 * np.pi) - np.pi
        out.append(np.c_[gpt, gy, gphi, np.full(k, 22.0)])
    dp = float(np.abs(s4 - np.c_[E, p]).max())
    b = np.vstack([a[~m]] + out); b = b[np.argsort(-b[:, 0], kind='stable')]
    return b, k, dp


def to_npz(L, Z, path):
    n = np.array([len(x) for x in L]); X = np.zeros((len(L), n.max(), 4), np.float32)
    for j, x in enumerate(L): X[j, :len(x)] = x
    np.savez_compressed(path, X=X, n=n, y=Z['y'], event=Z['event'], widx=Z['widx'], jet=Z['jet'])


def make_pamdec():
    L, y, key, Z, _ = H57.load_npz(VG / 'PAm.npz', rel=False); rng = np.random.default_rng(57057); out = []; nd = 0; dpm = 0.0
    for x in L:
        b, k, dp = decay_pi0(x, rng); out.append(b); nd += k; dpm = max(dpm, dp)
    to_npz(out, Z, VG / 'PAmdec.npz')
    return dict(pi0_decayed=nd, jets=len(L), pi0_per_jet=nd / len(L), max_abs_dp4_GeV=dpm)


# ----------------------------------------------------------------------------------------------------------------- score
def step_score(a):
    import torch
    import copy
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dec = make_pamdec(); log('PAmdec', dec)
    M = H.Models(dev)
    M.add_weaver('part_kin', 'part', False); M.add_weaver('pnet_kin', 'pn', False)
    for kind in ('efn', 'pfn'): H14.add_keras_ef(M, kind)
    H14.add_weaver_dir(M, 'hw_part_kin', 'part', False, V / 'models' / 'qg_herwig' / 'part_kin')
    kind, arch, pf, n0 = M.m['part_kin']; gg = torch.Generator().manual_seed(0); n2 = copy.deepcopy(n0).cpu()
    with torch.no_grad():
        for prm in n2.parameters():
            fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=gg)])
    M.m['perm0'] = (kind, arch, pf, n2.eval().to(dev))
    out = {}
    for s in SETS + ['PAmdec']:
        L, y, key, Z, nrel = H57.load_npz(VG / f'{s}.npz', rel=True)
        out[f'y|{s}'] = y.astype(np.int8); out[f'key|{s}'] = key
        for nm in MODELS:
            v = []
            for i in range(0, len(L), 10000):
                Ls = L[i:i + 10000]; v.append(-M.logit(nm, H.from_list(Ls, Lpad=max(len(x) for x in Ls))))
            out[f'{s}|{nm}'] = np.concatenate(v).astype(np.float32)
        log(s, len(L), f'relabelled {nrel}; acc', ' '.join(f'{m} {np.mean((out[f"{s}|{m}"] > 0) == (y == 1)):.4f}' for m in MODELS))
    np.savez_compressed(RES / 'h57_verify_scores.npz', **out)
    (RES / 'h57_verify_pamdec.json').write_text(json.dumps(dec, indent=1))
    log('wrote h57_verify_scores.npz')


# ----------------------------------------------------------------------------------------------------------------- stats
def alt_obs(L, angles='abs'):
    """per jet: N, pT^D, N(pT > 1, 2, 4 GeV), N_res(0.01, 0.03) = number of single-linkage clusters at Delta R < delta"""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    O = {k: np.zeros(len(L)) for k in ALT}
    for j, x in enumerate(L):
        pt = x[:, 0]; O['N'][j] = len(x); O['ptd'][j] = np.sqrt((pt ** 2).sum()) / pt.sum()
        for t in (1, 2, 4): O[f'N_pt{t}'][j] = (pt > t).sum()
        dy = x[:, 1][:, None] - x[:, 1][None]; dp = (x[:, 2][:, None] - x[:, 2][None] + np.pi) % (2 * np.pi) - np.pi
        d = np.hypot(dy, dp)
        for dl in (0.01, 0.03):
            O[f'Nres_{dl}'][j] = connected_components(csr_matrix(d < dl), directed=False)[0]
    return O


def lr_tables(Ot, yt):
    from interp.h54_qg_generator_knobs import _lr_hist
    T = {}
    for k in ALT:
        x = Ot[k]
        if k == 'ptd':
            bp = np.quantile(x, np.linspace(0, 1, 41)); bp[0], bp[-1] = 0, 1.01
        else:
            bp = np.arange(-0.5, 301.5, 1.0)
        T[k] = (bp, _lr_hist(x, yt, bp))
    return T


def lr_score(T, O):
    out = {}
    for k, (bp, h) in T.items():
        out[f'lr_{k}'] = h[np.clip(np.searchsorted(bp, O[k], side='right') - 1, 0, len(h) - 1)]
    return out


def step_stats(a):
    from interp import h14_qg_generator as H14
    from interp.h5_qg_counting import holm
    from scipy.stats import norm
    Lt, yt, _ = H14.efp_train_lists('py', n_per_class=30000, seed=5401 + 2)
    Ot = alt_obs(Lt); T = lr_tables(Ot, yt)
    lrn_nb = -H14.lrn_tables()[0]['py']
    acc_tr = {k: float(np.mean((lr_score(T, Ot)[f'lr_{k}'] > 0) == (yt == 1))) for k in ALT}
    log('LR(X) train accuracy', acc_tr)
    LRK = [f'lr_{k}' for k in ALT]

    # ---- per-sample table: scores (nets + LRs), y, key
    S = {}
    Zv = dict(np.load(RES / 'h57_verify_scores.npz'))
    for s in SETS + ['PAmdec']:
        L, y, key, Z, _ = H57.load_npz(VG / f'{s}.npz', rel=False); O = alt_obs(L)
        assert (y == Zv[f'y|{s}']).all() and (key == Zv[f'key|{s}']).all()
        d = {m: Zv[f'{s}|{m}'].astype(np.float64) for m in MODELS}; d.update(lr_score(T, O))
        d['lr_n_nb'] = lrn_nb[np.minimum(O['N'].astype(int), 299)]
        S[f'V|{s}'] = dict(sc=d, y=y, key=key, obs=O); log('fresh', s, len(L))
    Zs = np.load(RES / 'h57_scores.npz'); Ds = H57.samples()
    for sn, paths in Ds.items():
        if sn in ('P|nominal_b', 'F|V0b', 'F|NAp', 'F|PAp'): continue
        L, y, key = [], [], []
        for p in paths:
            l, yy, k, Z, _ = H57.load_npz(p, rel=False); L += l; y.append(yy); key.append(k)
        y = np.concatenate(y); key = np.concatenate(key); O = alt_obs(L)
        assert (key == Zs[f'key|{sn}']).all() and np.allclose(O['N'], Zs[f'obs|{sn}|N']), sn
        d = {m: Zs[f'{sn}|{m}'].astype(np.float64) for m in MODELS if f'{sn}|{m}' in Zs}; d.update(lr_score(T, O))
        d['lr_n_nb'] = Zs[f'{sn}|lr_n'].astype(np.float64)
        S[sn] = dict(sc=d, y=y, key=key, obs=O); log('h57', sn, len(L))

    rng = np.random.default_rng(57571)

    def pairs(s0, s1, cls):
        a0, a1 = S[s0], S[s1]
        pos = {int(k): j for j, k in enumerate(a0['key']) if a0['y'][j] == cls}
        pr = [(pos[int(k)], j) for j, k in enumerate(a1['key']) if a1['y'][j] == cls and int(k) in pos]
        return np.array([p[0] for p in pr], int), np.array([p[1] for p in pr], int)

    def block(s0, s1, cls, B):
        i0, i1 = pairs(s0, s1, cls); ms = [m for m in S[s0]['sc'] if m in S[s1]['sc']]
        Dm = np.stack([S[s1]['sc'][m][i1] - S[s0]['sc'][m][i0] for m in ms], 1); J = len(i0)
        bm = np.empty((B, len(ms)))
        for b in range(B):
            c = np.bincount(rng.integers(0, J, J), minlength=J).astype(np.float64); bm[b] = c @ Dm / J
        d = Dm.mean(0)
        r = dict(n_pairs=J, obs_rel={k: float(S[s1]['obs'][k][i1].mean() / S[s0]['obs'][k][i0].mean() - 1) for k in ALT})
        r['_d'] = {m: d[i] for i, m in enumerate(ms)}; r['_b'] = {m: bm[:, i] for i, m in enumerate(ms)}
        for i, m in enumerate(ms):
            r[m] = dict(d=float(d[i]), ci=np.percentile(bm[:, i], [2.5, 97.5]).tolist(), se=float(Dm[:, i].std() / np.sqrt(J)))
        return r

    def diff(r, m, ref):
        dd = r['_d'][m] - r['_d'][ref]; bd = r['_b'][m] - r['_b'][ref]; se = bd.std()
        return dict(d=float(dd), ci=np.percentile(bd, [2.5, 97.5]).tolist(), p=float(2 * norm.sf(abs(dd) / se)) if se > 0 else 1.0)

    def kappa(r, m, ref, e='efn_k'):
        den = r['_d'][ref] - r['_d'][e]; bk = (r['_b'][m] - r['_b'][e]) / (r['_b'][ref] - r['_b'][e])
        return dict(k=float((r['_d'][m] - r['_d'][e]) / den), ci=np.percentile(bk, [2.5, 97.5]).tolist(), den=float(den))

    R = dict(lr_train_acc=acc_tr, accuracy={}, fresh={}, decomposition={}, tests=[], fit={})
    for s in SETS + ['PAmdec']:
        y = S[f'V|{s}']['y']; R['accuracy'][s] = {m: float(np.mean((S[f'V|{s}']['sc'][m] > 0) == (y == 1))) for m in S[f'V|{s}']['sc']}
    comps = {'NAm': ('V|V0', 'V|NAm'), 'PAm': ('V|V0', 'V|PAm'), 'IRp': ('V|V0', 'V|IRp'),
             'PAm_string_part': ('V|V0', 'V|PAmdec'), 'PAm_pi0_merge_part': ('V|PAmdec', 'V|PAm')}
    for cn, cls in (('q', 1), ('g', 0)):
        for nm, (s0, s1) in comps.items():
            r = block(s0, s1, cls, a.boot)
            for m in NETS + ['hw_part_kin']:
                r[m]['minus'] = {ref: diff(r, m, ref) for ref in ['efn_k', 'lr_n_nb'] + LRK}
                r[m]['kappa_N'] = kappa(r, m, 'lr_n_nb'); r[m]['kappa_pTD'] = kappa(r, m, 'lr_ptd')
                if m in NETS:
                    for ref in ['efn_k', 'lr_n_nb'] + LRK:
                        R['tests'].append(dict(comp=nm, cls=cn, model=m, ref=ref, d=r[m]['minus'][ref]['d'], p=r[m]['minus'][ref]['p']))
            log(nm, cn, 'obs', {k: round(100 * v, 2) for k, v in r['obs_rel'].items()})
            log('   ', ' '.join(f"{m} {r[m]['d']:+.3f}" for m in MODELS + ['lr_n_nb'] + LRK))
            r.pop('_d'); r.pop('_b')
            (R['decomposition'] if 'part' in nm else R['fresh']).setdefault(nm, {})[cn] = r
    adj = holm([t['p'] for t in R['tests']])
    for t, h in zip(R['tests'], adj): t['holm'] = float(h)
    R['holm'] = dict(n_tests=len(R['tests']), n_pass_005=int((np.asarray(adj) < 0.05).sum()))

    # ---- fits over the 38 H57 settings: ds_net = gamma ds_EFN + beta ds_LR(X)  (one X), and + alpha ds_LR(N)
    rows = [s for s in S if s.startswith('P|') and s != 'P|nominal']
    rows = [(s, 'P|nominal') for s in rows] + [(f'F|{n}', 'F|V0') for n in ('NAm', 'PAm', 'IRp', 'Hzp')]
    R['fit_rows'] = [s for s, _ in rows]
    blk = {}
    for s1, s0 in rows:
        for cn, cls in (('q', 1), ('g', 0)):
            r = block(s0, s1, cls, 2); blk[(s1, cn)] = {m: (r[m]['d'], r[m]['se']) for m in r if isinstance(r[m], dict) and 'd' in r[m]}
    fresh = {(nm, cn): {m: R[('decomposition' if 'part' in nm else 'fresh')][nm][cn][m]['d'] for m in MODELS + ['lr_n_nb'] + LRK}
             for nm in comps for cn in 'qg'}
    for cn in ('q', 'g'):
        XE = np.array([blk[(s, cn)]['efn_k'][0] for s, _ in rows])
        Xk = {k: np.array([blk[(s, cn)][k][0] for s, _ in rows]) for k in ['lr_n_nb'] + LRK}
        R['fit'].setdefault(cn, {})['regressor_corr'] = {k: float(np.corrcoef(Xk['lr_ptd'], v)[0, 1]) for k, v in Xk.items()}
        for m in NETS + ['hw_part_kin', 'perm0']:
            yv = np.array([blk[(s, cn)][m][0] for s, _ in rows]); se = np.array([blk[(s, cn)][m][1] for s, _ in rows])
            w = 1 / np.maximum(se, 1e-3) ** 2; res = {}
            specs = {f'efn+{k}': [XE, Xk[k]] for k in ['lr_n_nb'] + LRK}
            specs.update({f'efn+lr_n_nb+{k}': [XE, Xk['lr_n_nb'], Xk[k]] for k in LRK if k != 'lr_N'})
            for fn, cols in specs.items():
                X = np.stack(cols, 1); sw = np.sqrt(w)
                c = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)[0]; chi2 = float((w * (yv - X @ c) ** 2).sum())
                r2 = np.random.default_rng(7); bs = []
                for _ in range(1000):
                    ix = r2.integers(0, len(yv), len(yv)); bs.append(np.linalg.lstsq(X[ix] * sw[ix, None], yv[ix] * sw[ix], rcond=None)[0])
                bs = np.array(bs)
                # out-of-sample prediction on the fresh comparisons
                regs = ['efn_k'] + fn.split('+')[1:]
                pred = {nm: float(sum(cc * fresh[(nm, cn)][rg] for cc, rg in zip(c, regs))) for nm in comps}
                obs = {nm: fresh[(nm, cn)][m] for nm in comps}
                res[fn] = dict(coef=c.tolist(), ci=[np.percentile(bs[:, j], [2.5, 97.5]).tolist() for j in range(X.shape[1])],
                               chi2=chi2, ndf=len(yv) - X.shape[1], fresh_pred=pred, fresh_obs=obs,
                               fresh_rms=float(np.sqrt(np.mean([(pred[k] - obs[k]) ** 2 for k in comps]))))
            R['fit'][cn][m] = res
            if m in NETS:
                log('fit', cn, m, {k: (round(v['chi2'], 1), [round(x, 2) for x in v['coef']], round(v['fresh_rms'], 3)) for k, v in res.items()})
    (RES / 'h57_verify.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h57_verify.json')


# ------------------------------------------------------------------------------------------------- decay-tool validation
def step_deccheck(a):
    """validation of decay_pi0: pilot D:pi0 (pi0 stable, Pythia) decayed by hand vs pilot nominal (same events, pi0 decayed
    by Pythia). If the hand decay is equivalent, the paired mean shift of every model is ~0. Also D:pi0 (stable) vs nominal."""
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = H.Models(dev)
    M.add_weaver('part_kin', 'part', False); M.add_weaver('pnet_kin', 'pn', False)
    for kind in ('efn', 'pfn'): H14.add_keras_ef(M, kind)
    H14.add_weaver_dir(M, 'hw_part_kin', 'part', False, V / 'models' / 'qg_herwig' / 'part_kin')
    rng = np.random.default_rng(57058); out = {}
    def score(L):
        r = {}
        for nm in ['part_kin', 'pnet_kin', 'pfn_k', 'efn_k', 'hw_part_kin']:
            r[nm] = np.concatenate([-M.logit(nm, H.from_list(L[i:i + 5000], Lpad=max(len(x) for x in L[i:i + 5000]))) for i in range(0, len(L), 5000)])
        r['N'] = np.array([len(x) for x in L], float); r['ptd'] = np.array([np.sqrt((x[:, 0] ** 2).sum()) / x[:, 0].sum() for x in L])
        return r
    for cls, cn in ((1, 'q'), (0, 'g')):
        Ln, yn, kn, _, _ = H57.load_npz(H57.PIL / f'nominal_{cls}.npz', rel=True)
        Lp, yp, kp, _, _ = H57.load_npz(H57.PIL / f'D:pi0_{cls}.npz', rel=False)
        Ld = [decay_pi0(x, rng)[0] for x in Lp]
        Ls = [x.copy() for x in Lp]
        for x in Ls: x[:, 3], _ = H57.relabel(x[:, 3])
        pos = {int(k): j for j, k in enumerate(kn)}; ii = np.array([pos[int(k)] for k in kp if int(k) in pos]); jj = np.array([j for j, k in enumerate(kp) if int(k) in pos])
        Sn, Sd, Ss = score(Ln), score(Ld), score(Ls)
        res = {}
        for nm in Sn:
            for lab, S1 in (('handdecay_minus_nominal', Sd), ('stable_minus_nominal', Ss)):
                d = S1[nm][jj] - Sn[nm][ii]
                if nm in ('N', 'ptd'): d = d / Sn[nm][ii].mean()
                res.setdefault(lab, {})[nm] = dict(d=float(d.mean()), ci=[float(d.mean() - 1.96 * d.std() / np.sqrt(len(d))), float(d.mean() + 1.96 * d.std() / np.sqrt(len(d)))])
        res['n_pairs'] = int(len(ii)); out[cn] = res; log(cn, json.dumps(res))
    (RES / 'h57_verify_deccheck.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['gen', 'score', 'stats', 'deccheck'])
    ap.add_argument('--jets', type=int, default=6000); ap.add_argument('--cpu', type=int, default=8)
    ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args()
    dict(gen=step_gen, score=step_score, stats=step_stats, deccheck=step_deccheck)[a.step](a)
