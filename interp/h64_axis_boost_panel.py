"""H64 (loop D, new D claim; links D1, D4m, A4, A14): under a Lorentz boost along the jet axis at fixed invariant mass, does each
network respond like a reader of the invariant jet mass m, or like a reader of the jet's angular size (opening angle ~ 2 m / pT)?

Edits (imported from interp.h12_boost_panel, applied to every constituent four-vector; lam = factor on the jet pT):
  B(lam)  exact Lorentz boost along the jet three-momentum with |P_jet| -> lam |P_jet|: m fixed exactly, jet pT x lam, every
          constituent's angle to the axis shrinks by ~ 1 / lam (log m: 0, log pT: +log lam, log angle: -log lam).
  P(lam)  pT-only control: every four-vector x lam at fixed (eta, phi): m x lam, pT x lam, angles fixed.
  A(lam)  angle-only control: every (deta, dphi) about the stored axis x 1 / lam at fixed constituent pT: m x ~1/lam, pT fixed.
  R(lam)  random control of equal size: rotation about the axis with the same pT-weighted mean displacement as B on that jet
          (m, pT, every IRC-safe shape fixed).
  To first order B = P + A in (log m, log pT): a reader of m gives Delta_B = 0; a reader of the angular size at fixed pT
  normalisation (or of m / pT, which equals the angle scale) gives Delta_B = Delta_A, Delta_P = 0; a reader of pT gives Delta_A = 0.
Response: per class k, l_k = log p_k - log(1 - p_k) (one-vs-rest log-odds from the softmax); Delta_X,k(lam) = mean over jets of true
  class k of l_k(edited by X) - l_k(unedited), 95 % bootstrap over jets (stratified by class).
  R_B,k = Delta_B,k / Delta_A,k (1: angular-size / m/pT reader; 0: invariant-mass reader); R_m,k = Delta_B,k / Delta_P,k;
  ratios reported only where the denominator's 95 % interval excludes 0. Model summary R_B^w(lam) = sum_k s_k Delta_B,k /
  sum_k |Delta_A,k| with s_k = sign(Delta_A,k) (classes weighted by their A response), likewise R_m^w with P.
Tests (trained models and references, per model x lam x class): H0_inv Delta_B = 0 (normal approximation on per-jet differences);
  H0_ang Delta_B = Delta_A (paired); H0_null Delta_B(trained) = Delta_B(null copy) for the null of the same dataset with the
  largest p (paired, same jets); McNemar of call changes at the 50 % working point (threshold on l_k giving 50 % efficiency for
  true class k on unedited jets) B vs A and B vs R. Holm over model x lam x class within each family (per part here, global in
  --part summary).
Nulls: untrained copies (every weight tensor redrawn from a Gaussian with the trained tensor's mean and spread, the project's
  convention) and weight-permuted copies (entries of every weight tensor permuted). References: sklearn MLPs on
  [log m, log pT], [log(m / pT)], [log dR_prongs, log pT], trained on disjoint real jets of the same split (dR_prongs = distance
  between two pT-weighted 2-means axes in the (deta, dphi) plane: start at the hardest particle and the particle maximising
  pT dR^2 from it, 10 iterations).
Training-range check: share of edited jets with jet pT outside the dataset's pT range; C2ST (HGB, 2-fold, weighted AUC) of B jets
  vs disjoint real jets reweighted to the B jets' pT distribution (features log m, N, pT-weighted mean dR, log pT; coverage =
  share of B jets in pT bins that hold >= 5 real jets); the same for the unedited sample (calibration).
Samples: hls4ml jedi_core.sample(2000, 6401) (+ 3,000 other rows per class for MLPs / C2ST, rng 6411); JetClass 5,000 random rows
  per class of the first test file (h1_irc_panel.jc_sample, seed 6402) split at random (rng 6412) into 2,000 evaluated + 3,000
  reference; Top Landscape test 2,000 per class (rng 6403) + 10,000 disjoint per class (rng 6413); QG test 2,000 per class
  (rng 6404) + 10,000 disjoint per class (rng 6414). lam in {0.8, 0.9, 1.1, 1.25}.
Run
  ./job.sh d-h64-jedi 1 2 'python -m interp.h64_axis_boost_panel --part hls_jedi'
  AI_PY=conda ./job.sh d-h64-small 1 2 'python -m interp.h64_axis_boost_panel --part hls_small'
  ./part_job.sh d-h64-gpu 'python -m interp.h64_axis_boost_panel --part jetclass && python -m interp.h64_axis_boost_panel --part top && python -m interp.h64_axis_boost_panel --part qg'
  ./job.sh d-h64-sum 1 2 'python -m interp.h64_axis_boost_panel --part summary'
  -> $AI_RESULTS/physics/h64_<part>.json, h64_<part>_perjet.npz, h64_summary.json, h64_wjet_boost.png, h64_RB.png
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h12_boost_panel as H  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
LAMS = (0.8, 0.9, 1.1, 1.25)
NB = 1000
T0 = time.time()
SEEDS = dict(hls=6401, jetclass=6402, top=6403, qg=6404)
NULL_MAX = 10.0    # summary: a null copy whose mean |Delta_B| exceeds 10 log-odds is treated as unbounded
DEN_MIN = 0.02     # summary: a class-weighted ratio is reported only when its denominator |sum| >= 0.02 log-odds


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ============================================================================================================== observables
def prong_dr(T, it=10):
    de, dp = H.rel(T); w = T['pt']; real = w > 0; J = len(w); r = np.arange(J)
    i0 = np.argmax(w, 1); ae, ap = de[r, i0].copy(), dp[r, i0].copy()
    d1 = (de - ae[:, None]) ** 2 + (dp - ap[:, None]) ** 2; i1 = np.argmax(np.where(real, w * d1, -1), 1)
    be, bp = de[r, i1].copy(), dp[r, i1].copy()
    for _ in range(it):
        d1 = (de - ae[:, None]) ** 2 + (dp - ap[:, None]) ** 2; d2 = (de - be[:, None]) ** 2 + (dp - bp[:, None]) ** 2
        c1 = real & (d1 <= d2); c2 = real & (d1 > d2)
        for c, (xe, xp) in ((c1, (ae, ap)), (c2, (be, bp))):
            W = (w * c).sum(1); ok = W > 0
            xe[ok] = ((w * c * de).sum(1)[ok] / W[ok]); xp[ok] = ((w * c * dp).sum(1)[ok] / W[ok])
    return np.hypot(ae - be, ap - bp)


def feats(T):
    O = H.jet_obs(T); O['dr'] = prong_dr(T); return O


def c2st_X(O):
    return np.stack([np.log(np.maximum(O['m'], 1e-3)), O['N'], O['mdr'], np.log(np.maximum(O['jpt'], 1e-3))], 1)


REF_FEATS = {'MLP(m,pT)': lambda O: np.stack([np.log(np.maximum(O['m'], 1e-3)), np.log(O['jpt'])], 1),
             'MLP(m/pT)': lambda O: np.log(np.maximum(O['m'], 1e-3) / O['jpt'])[:, None],
             'MLP(dR,pT)': lambda O: np.stack([np.log(np.maximum(O['dr'], 1e-4)), np.log(O['jpt'])], 1)}


def train_refs(Opool, ypool):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    refs = {}
    for k, fx in REF_FEATS.items():
        t = time.time()
        clf = make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=400, early_stopping=True, random_state=0)).fit(fx(Opool), ypool)
        refs[k] = clf; log(f'reference {k}: trained on {len(ypool)} jets, train acc {clf.score(fx(Opool), ypool):.4f} ({time.time() - t:.0f} s)')
    return refs


# ============================================================================================================== nulls
def null_torch(module, kind, seed):
    """copy of a torch module: kind 'perm' permutes the entries of every parameter tensor; 'untr' redraws every parameter tensor
    with > 1 entry from a Gaussian with its trained mean and spread (buffers such as BN running stats kept)"""
    import torch
    m2 = copy.deepcopy(module); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m2.parameters():
            v = p.data.detach().cpu().float()
            if kind == 'perm':
                f = v.reshape(-1); new = f[torch.randperm(f.numel(), generator=g)].reshape(v.shape)
            else:
                if v.numel() <= 1: continue
                new = torch.randn(v.shape, generator=g) * v.std() + v.mean()
            p.data.copy_(new.to(p.dtype).to(p.device))
    return m2.eval()


def null_flow(fl, kind, seed):
    rng = np.random.default_rng(seed); f2 = copy.copy(fl)
    nz = lambda a: rng.permutation(a.ravel()).reshape(a.shape) if kind == 'perm' else rng.normal(a.mean(), a.std(), a.shape).astype(a.dtype)
    f2.phi = [(nz(K), nz(b)) for K, b in fl.phi]; f2.F = [(nz(K), nz(b)) for K, b in fl.F]; f2.out = (nz(fl.out[0]), nz(fl.out[1]))
    return f2


# ============================================================================================================== statistics
def logodds_all(Lg):
    Lg = np.clip(np.nan_to_num(np.asarray(Lg, np.float64), neginf=-60.0, posinf=60.0), -1e6, 1e6)
    ls = H.logsoft(Lg); C = ls.shape[1]; out = np.empty_like(ls)
    for k in range(C):
        o = np.delete(ls, k, 1); mx = o.max(1, keepdims=True); out[:, k] = ls[:, k] - (mx[:, 0] + np.log(np.exp(o - mx).sum(1)))
    return out


def pnorm(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2: return 1.0
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0: return 1.0 if abs(d.mean()) == 0 else 0.0
    return float(math.erfc(abs(d.mean()) / se / math.sqrt(2)))


def pct(a):
    return np.percentile(a, [2.5, 97.5]).tolist()


def excl0(ci):
    return ci[0] > 0 or ci[1] < 0


def response_stats(DD, y, C, classes, rng, null_dd=None):
    """DD: {(kind, lam): per-jet own-class Delta l (J,)}; returns per lam / class stats and weighted summaries"""
    out = {}
    for lam in LAMS:
        cls = {}; bm = {}
        for k in range(C):
            sel = np.flatnonzero(y == k); n = len(sel)
            if n == 0: continue
            I = rng.integers(0, n, (NB, n)); x = {kd: DD[(kd, lam)][sel] for kd in 'BPAR'}
            b = {kd: x[kd][I].mean(1) for kd in 'BPAR'}; bm[k] = b
            st = {kd: dict(mean=float(x[kd].mean()), ci=pct(b[kd])) for kd in 'BPAR'}
            st['n'] = int(n)
            for nm, num, den in (('R_B', 'B', 'A'), ('R_m', 'B', 'P')):
                ok = excl0(st[den]['ci'])
                st[nm] = dict(value=float(x[num].mean() / x[den].mean()) if ok else None,
                              ci=pct(b[num] / b[den]) if ok else None, denominator_ci_excludes_0=bool(ok))
            st['B_minus_PplusA'] = dict(mean=float((x['B'] - x['P'] - x['A']).mean()), ci=pct(b['B'] - b['P'] - b['A']))
            st['p_H0_inv_B_eq_0'] = pnorm(x['B']); st['p_H0_ang_B_eq_A'] = pnorm(x['B'] - x['A']); st['p_B_eq_P'] = pnorm(x['B'] - x['P'])
            if null_dd:
                ps = {nn: pnorm(x['B'] - d[(('B', lam))][sel]) for nn, d in null_dd.items()}
                worst = max(ps, key=ps.get); st['p_H0_null'] = ps[worst]; st['p_H0_null_vs'] = worst; st['p_vs_each_null'] = ps
            cls[classes[k]] = st
        W = {}
        for nm, den in (('R_B_w', 'A'), ('R_m_w', 'P')):
            s = {k: np.sign(np.mean(bm[k][den])) for k in bm}
            num = sum(s[k] * bm[k]['B'] for k in bm); dd = sum(s[k] * bm[k][den] for k in bm)
            point_num = sum(s[k] * cls[classes[k]]['B']['mean'] for k in bm); point_den = sum(s[k] * cls[classes[k]][den]['mean'] for k in bm)
            W[nm] = dict(value=float(point_num / point_den) if point_den != 0 else None, ci=pct(num / dd) if np.all(dd != 0) else None,
                         denominator=float(point_den))
        out[str(lam)] = dict(classes=cls, **W)
    return out


def wp_stats(L0o, L1o, y, C, classes):
    """L0o/L1o: one-vs-rest log-odds (J, C) unedited / {(kind, lam): edited}; 50 % working point per class"""
    out = {}
    for k in range(C):
        thr = float(np.median(L0o[y == k, k])); c0 = L0o[:, k] > thr; r = {}
        ch = {key: (L1[:, k] > thr) != c0 for key, L1 in L1o.items()}
        for lam in LAMS:
            a = {kd: ch[(kd, lam)] for kd in 'BPAR'}
            r[str(lam)] = dict(changed={kd: float(v.mean()) for kd, v in a.items()},
                               mcnemar_B_vs_A=H.mcnemar(a['B'], a['A']), mcnemar_B_vs_R=H.mcnemar(a['B'], a['R']))
        out[classes[k]] = dict(threshold=thr, lams=r)
    return out


def weighted_c2st(Fb, jb, Fp, jp, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    edges = np.unique(np.quantile(jb, np.linspace(0, 1, 31))); edges[0] -= 1e-6; edges[-1] += 1e-6
    hb = np.histogram(jb, edges)[0]; hp = np.histogram(jp, edges)[0]
    ib = np.clip(np.digitize(jb, edges) - 1, 0, len(hb) - 1)
    ip = np.digitize(jp, edges) - 1; inr = (ip >= 0) & (ip < len(hb)); ipc = np.clip(ip, 0, len(hb) - 1)
    good = hp >= 5; cov = float(good[ib].mean())
    if cov < 0.05: return dict(auc=None, coverage=cov, note='real jets do not cover the edited jets pT range')
    keepb = good[ib]; wp = np.where(inr & good[ipc], hb[ipc] / np.maximum(hp[ipc], 1), 0.0)
    Xb, Xp = Fb[keepb], Fp[wp > 0]; wpp = wp[wp > 0]; wpp = wpp * len(Xb) / wpp.sum()
    X = np.concatenate([Xb, Xp]); yy = np.r_[np.ones(len(Xb)), np.zeros(len(Xp))]; w = np.r_[np.ones(len(Xb)), wpp]
    rng = np.random.default_rng(seed); f = rng.integers(0, 2, len(yy)); s = np.zeros(len(yy))
    for k in (0, 1):
        tr, te = f != k, f == k
        c = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[tr], yy[tr], sample_weight=w[tr]); s[te] = c.predict_proba(X[te])[:, 1]
    return dict(auc=float(roc_auc_score(yy, s, sample_weight=w)), coverage=cov, n_edited=int(len(Xb)),
                n_eff_real=float(wpp.sum() ** 2 / (wpp ** 2).sum()))


# ============================================================================================================== panel
def run_panel(part, T0, y, Tp, yp, classes, models, prep, window, info, seed):
    """models: {name: dict(f=callable(X) -> logits (J, C), src='prep' | 'table', role='trained' | 'null', known=...)}"""
    rng = np.random.default_rng(seed); C = len(classes); J = len(y)
    res = dict(part=part, N=J, labels=np.bincount(y, minlength=C).tolist(), classes=classes, lams=list(LAMS), window=window,
               models={}, edits={}, resp={}, wp={}, c2st={}, **info)
    O0 = feats(T0); Op = feats(Tp)
    res['unedited_obs'] = {k: dict(median=float(np.median(v)), q=np.percentile(v, [5, 95]).tolist()) for k, v in O0.items()}
    refs = train_refs(Op, yp)
    for k, clf in refs.items():
        models[k] = dict(f=(lambda O, clf=clf, k=k: clf.predict_log_proba(REF_FEATS[k](O))), src='table', role='ref', known=None)
    NPZ = dict(y=y.astype(np.int8), m0=O0['m'].astype(np.float32), jpt0=O0['jpt'].astype(np.float32), dr0=O0['dr'].astype(np.float32))
    X0 = prep(H.sortT(T0)); L0o = {}
    for mn, M in models.items():
        t = time.time(); lg = M['f'](X0 if M['src'] == 'prep' else O0); c = lg.argmax(1); L0o[mn] = logodds_all(lg)
        r = dict(role=M['role'], n=J, acc=float((c == y).mean()), acc_ci=H.boot_frac(c == y, rng), known_acc=M.get('known'),
                 per_class={classes[k]: float((c[y == k] == k).mean()) for k in range(C)}, frac_called={classes[k]: float((c == k).mean()) for k in range(C)})
        if C == 2:
            from sklearn.metrics import roc_auc_score
            r['auc'] = float(roc_auc_score(y == 1, L0o[mn][:, 1])) if np.ptp(L0o[mn][:, 1]) > 0 else 0.5
        res['models'][mn] = r
        log(f'{part} {mn} [{M["role"]}]: N {J} accuracy {r["acc"]:.4f} (known {M.get("known")}) auc {r.get("auc")} ({time.time() - t:.0f} s)')
    # calibration C2ST: unedited sample vs reference jets (pT reweighted)
    res['c2st']['unedited'] = weighted_c2st(c2st_X(O0), O0['jpt'], c2st_X(Op), Op['jpt'], seed)
    log(f'{part} C2ST unedited vs reference real jets: {res["c2st"]["unedited"]}')
    DD = {mn: {} for mn in models}; L1o = {mn: {} for mn in models}
    erng = np.random.default_rng(seed + 1000)
    for lam in LAMS:
        B, _ = H.boost(T0, lam); Dsp = H.displacement(T0, B)
        for kd, U in (('B', B), ('P', H.pscale(T0, lam)[0]), ('A', H.ascale(T0, lam)[0]), ('R', H.rotate(T0, Dsp, erng)[0])):
            t = time.time(); O1 = feats(U); out = (O1['jpt'] < window[0]) | (O1['jpt'] > window[1])
            e = dict(kind=kd, lam=lam, outside_window_frac=float(out.mean()),
                     rel_change={k: dict(median=float(np.median(O1[k] / np.maximum(O0[k], 1e-12) - 1)),
                                         absmax=float(np.max(np.abs(O1[k] / np.maximum(O0[k], 1e-12) - 1)))) for k in ('m', 'jpt', 'mA', 'mdr', 'dr')})
            if kd == 'B': e['displacement_median'] = float(np.median(Dsp))
            if kd in ('B', 'P', 'A'):
                res['c2st'][f'{kd}{lam}'] = weighted_c2st(c2st_X(O1), O1['jpt'], c2st_X(Op), Op['jpt'], seed)
            X1 = prep(H.sortT(U))
            for mn, M in models.items():
                lg = M['f'](X1 if M['src'] == 'prep' else O1); lo = logodds_all(lg); L1o[mn][(kd, lam)] = lo
                DD[mn][(kd, lam)] = (lo - L0o[mn])[np.arange(J), y]
                NPZ[f'{mn}__{kd}{lam}'] = DD[mn][(kd, lam)].astype(np.float32)
            NPZ[f'jpt__{kd}{lam}'] = O1['jpt'].astype(np.float32); NPZ[f'm__{kd}{lam}'] = O1['m'].astype(np.float32)
            res['edits'][f'{kd}{lam}'] = e
            log(f'{part} {kd}{lam}: dm med {e["rel_change"]["m"]["median"]:+.2e} max {e["rel_change"]["m"]["absmax"]:.1e} djpt {e["rel_change"]["jpt"]["median"]:+.3f} '
                f'ddR {e["rel_change"]["dr"]["median"]:+.3f} outside {e["outside_window_frac"]:.3f} c2st {res["c2st"].get(f"{kd}{lam}", {}).get("auc")} | '
                + ' '.join(f'{mn} {np.mean(DD[mn][(kd, lam)]):+.3f}' for mn in models) + f' ({time.time() - t:.0f} s)')
    nulls = [mn for mn, M in models.items() if M['role'] == 'null']
    for mn, M in models.items():
        nd = {nn: DD[nn] for nn in nulls} if M['role'] != 'null' and nulls else None
        res['resp'][mn] = response_stats(DD[mn], y, C, classes, rng, nd)
        res['wp'][mn] = wp_stats(L0o[mn], L1o[mn], y, C, classes)
        r = res['resp'][mn]
        log(f'{part} {mn}: R_B^w ' + ' '.join(f'{l}:{r[str(l)]["R_B_w"]["value"]}' for l in LAMS) + ' | R_m^w '
            + ' '.join(f'{l}:{r[str(l)]["R_m_w"]["value"]}' for l in LAMS))
    # Holm within the part (trained + refs)
    fam = {'H0_inv': 'p_H0_inv_B_eq_0', 'H0_ang': 'p_H0_ang_B_eq_A', 'H0_null': 'p_H0_null'}
    res['holm'] = {}
    for fn, key in fam.items():
        keys, ps = [], []
        for mn, M in models.items():
            if M['role'] != 'trained': continue
            for l in LAMS:
                for c, st in res['resp'][mn][str(l)]['classes'].items():
                    if key in st: keys.append(f'{mn}|{l}|{c}'); ps.append(st[key])
        res['holm'][fn] = dict(zip(keys, H.holm(ps))) if ps else {}
    for fn, key in (('McNemar_B_vs_A', 'mcnemar_B_vs_A'), ('McNemar_B_vs_R', 'mcnemar_B_vs_R')):
        keys, ps = [], []
        for mn, M in models.items():
            if M['role'] != 'trained': continue
            for c, st in res['wp'][mn].items():
                for l in LAMS: keys.append(f'{mn}|{l}|{c}'); ps.append(st['lams'][str(l)][key]['p'])
        res['holm'][fn] = dict(zip(keys, H.holm(ps)))
    res['roles'] = {mn: M['role'] for mn, M in models.items()}
    return res, NPZ, O0


def save(res, NPZ, part):
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / f'h64_{part}.json'
    p.write_text(json.dumps(res, indent=1, default=float)); np.savez_compressed(OUT / f'h64_{part}_perjet.npz', **NPZ); log(f'wrote {p}')


# ============================================================================================================== hls4ml
def hls_tables(seed, per=2000, per_ref=3000):
    from jetdistill.config import RESULTS
    from interp.h1_irc_panel import hls_table
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); yall = np.asarray(d['y'])
    rng = np.random.default_rng(seed)    # identical draw to interp.jedi_core.sample(per_class=per, seed=seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yall == c), per, replace=False) for c in range(5)]))
    r2 = np.random.default_rng(seed + 10); used = np.zeros(len(yall), bool); used[idx] = True
    ref = np.sort(np.concatenate([r2.choice(np.flatnonzero((yall == c) & ~used), per_ref, replace=False) for c in range(5)]))
    x = np.asarray(d['x'], np.float32)
    mk = lambda ii: {k: v for k, v in hls_table(x[ii]).items() if k != 'sgn'}
    T, Tp = mk(idx), mk(ref); log(f'hls4ml: {len(idx)} evaluated + {len(ref)} reference jets (seed {seed})')
    return T, yall[idx].astype(int), Tp, yall[ref].astype(int), idx


def hls_known():
    p = ROOT / 'inbox_d' / 'merged' / 'H12_report.md'
    return dict(jedi_n64='0.8100 (H12 seed 301) / 0.8068 (E26)', dsets='0.8075 (H12)', tf_d64='0.8147 (H12)')


def part_hls_jedi():
    from interp.jedi_core import Jedi
    from jetdistill import network as N
    seed = SEEDS['hls']; T0, y, Tp, yp, idx = hls_tables(seed)
    nets = dict(jedi_n64=Jedi(64, '3-feature'), jedi_n64_untr=Jedi(64, '3-feature', untrained=True, seed=0))
    orig = N.redraw_untrained; N.redraw_untrained = H.permute_weights
    try: nets['jedi_n64_perm'] = Jedi(64, '3-feature', untrained=True, seed=7)
    finally: N.redraw_untrained = orig
    kn = hls_known()
    models = {k: dict(f=(lambda X, n=n: n.run(X)['logits']), src='prep', role='trained' if k == 'jedi_n64' else 'null',
                      known=kn.get(k)) for k, n in nets.items()}
    res, NPZ, O0 = run_panel('hls_jedi', T0, y, Tp, yp, H.HLS_CL, models, H.hls_prep, H.hls_window(T0),
                             dict(sample=f'interp.jedi_core.sample(per_class=2000, seed={seed})'), seed)
    save(res, NPZ, 'hls_jedi'); fig_jet(T0, y, O0)


def part_hls_small():
    import torch
    from interp.h1_irc_panel import load_small, torch_logits
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    seed = SEEDS['hls']; T0, y, Tp, yp, idx = hls_tables(seed); models = {}; kn = hls_known()
    for tag, run in (('dsets', 'dsets-d64-n64'), ('tf_d64', 'tf-2b-d64-n64-e100')):
        d = V / 'models' / 'small' / run / 'seed0'; net, sh, sc = load_small(d)
        known = json.loads((d / 'test.json').read_text())['accuracy']
        mk = lambda nt, sh=sh, sc=sc: (lambda X: torch_logits(nt, X, sh, sc))
        models[tag] = dict(f=mk(net), src='prep', role='trained', known=f'{known} (test.json); {kn.get(tag)}')
        models[f'{tag}_untr'] = dict(f=mk(null_torch(net, 'untr', 0)), src='prep', role='null')
        models[f'{tag}_perm'] = dict(f=mk(null_torch(net, 'perm', 7)), src='prep', role='null')
    res, NPZ, _ = run_panel('hls_small', T0, y, Tp, yp, H.HLS_CL, models, H.hls_prep, H.hls_window(T0),
                            dict(sample=f'interp.jedi_core.sample(per_class=2000, seed={seed})'), seed)
    save(res, NPZ, 'hls_small')


# ============================================================================================================== JetClass
def part_jetclass():
    from interp.h1_irc_panel import jc_sample, jc_table, pn_logits
    from interp.part_core import ParTNetwork, CL, inputs, build, sub
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); seed = SEEDS['jetclass']
    Jall = jc_sample(5000, seed); yall = Jall['label'].astype(int); r = np.random.default_rng(seed + 10)
    ev, rf = [], []
    for c in range(len(CL)):
        ii = r.permutation(np.flatnonzero(yall == c)); ev.append(ii[:2000]); rf.append(ii[2000:])
    ev, rf = np.sort(np.concatenate(ev)), np.sort(np.concatenate(rf))
    J0 = sub(Jall, ev); y = yall[ev]
    tab = lambda Jx: (lambda Tj: dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext']))(jc_table(Jx))
    T0 = tab(J0); Jr = sub(Jall, rf); yp = yall[rf]; Tp = tab(Jr); Tp['ext'] = {}; del Jall, Jr
    log(f'JetClass: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        Jx = dict(J0); Jx['jet_eta'] = T['ax_eta'].astype(np.float32); Jx['jet_phi'] = T['ax_phi'].astype(np.float32)
        sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = H.rel(T)
        return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
    models = {}
    known = dict(ParT_full='0.8575 (H12) / 0.859 (D4)', ParT_kin='0.7388 (H12) / 0.745 (D4)', PN_full='0.8409 (H12)', PN_kin='0.7004 (H12)')
    pt_f = lambda net, n: (lambda J: fwd(net, *inputs(J, n))[1].astype(np.float64))
    parts = {n: ParTNetwork(n) for n in ('full', 'kin')}
    for n, net in parts.items(): models[f'ParT_{n}'] = dict(f=pt_f(net, n), src='prep', role='trained', known=known[f'ParT_{n}'])
    models['ParT_full_untr'] = dict(f=pt_f(ParTNetwork('full', untrained=True, seed=0), 'full'), src='prep', role='null')
    pk = copy.copy(parts['kin']); pk.model = null_torch(parts['kin'].model, 'perm', 7)
    models['ParT_kin_perm'] = dict(f=pt_f(pk, 'kin'), src='prep', role='null')
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        mk = lambda p, n=n: (lambda J: pn_logits(p, J, n))
        models[f'PN_{n}'] = dict(f=mk(pn), src='prep', role='trained', known=known[f'PN_{n}'])
        if n == 'kin': models['PN_kin_perm'] = dict(f=mk(null_torch(pn, 'perm', 7)), src='prep', role='null')
        else: models['PN_full_untr'] = dict(f=mk(null_torch(pn, 'untr', 0)), src='prep', role='null')
    res, NPZ, _ = run_panel('jetclass', T0, y, Tp, yp, list(CL), models, prep, [500.0, 1000.0],
                            dict(sample=f'h1_irc_panel.jc_sample(5000, seed {seed}) random rows per class, 2,000 per class evaluated (rng {seed + 10})'), seed)
    save(res, NPZ, 'jetclass')


# ============================================================================================================== Top / QG
def split_rows(ylab, per, per_ref, seed):
    ev = H._rows(ylab, per, seed); used = np.zeros(len(ylab), bool); used[ev] = True; r = np.random.default_rng(seed + 10)
    rf = np.sort(np.concatenate([r.choice(np.flatnonzero((ylab == c) & ~used), per_ref, replace=False) for c in (0, 1)]))
    return ev, rf


def part_top():
    import torch
    from interp import data_top as dt
    from interp.h4_top_wmass import WeaverNet, LNet, Flow, kin3_of
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; seed = SEEDS['top']
    d = dt.DST / 'test'; ylab = np.load(d / 'label.npy'); ev, rf = split_rows(ylab, 2000, 10000, seed)
    Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy')
    tab = lambda ii: H.table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1]))
    T0, Tp = tab(ev), tab(rf); y, yp = ylab[ev].astype(int), ylab[rf].astype(int); log(f'Top: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
        P4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
        return dict(P4=P4.astype(np.float32), nn=real.sum(1))
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    known = dict(ParT_kin='0.9398 (H12) / 0.940 (H2)', PN_kin='0.9402 (H12)', LorentzNet='0.9415 (H12)', PFN='0.9285 (H12)', EFN='0.9221 (H12)')
    models = {}; wf = lambda net: (lambda X: two(net.score(X['P4'], X['nn'])))
    for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
        net = WeaverNet(kind, V / 'models/top' / dname / 'net_best_epoch_state.pt', dev)
        models[f'{kind}_kin'] = dict(f=wf(net), src='prep', role='trained', known=known[f'{kind}_kin'])
        n2 = copy.copy(net); n2.model = null_torch(net.model, 'perm', 7); models[f'{kind}_kin_perm'] = dict(f=wf(n2), src='prep', role='null')
        if kind == 'ParT':
            n3 = copy.copy(net); n3.model = null_torch(net.model, 'untr', 0); models['ParT_kin_untr'] = dict(f=wf(n3), src='prep', role='null')
    ln = LNet(dev, mass64=True); order = np.random.default_rng(seed + 5).permutation(len(ev))
    lf = lambda net: (lambda X: two(net.score(X['P4'], X['nn'], order)))
    models['LorentzNet'] = dict(f=lf(ln), src='prep', role='trained', known=known['LorentzNet'])
    l2 = copy.copy(ln); l2.model = null_torch(ln.model, 'perm', 7); models['LorentzNet_perm'] = dict(f=lf(l2), src='prep', role='null')
    for a in ('pfn', 'efn'):
        fl = Flow(a); ff = lambda f_: (lambda X: two(f_.score(kin3_of(X['P4'], X['nn']))))
        models[a.upper()] = dict(f=ff(fl), src='prep', role='trained', known=known[a.upper()])
        if a == 'efn': models['EFN_untr'] = dict(f=ff(null_flow(fl, 'untr', 0)), src='prep', role='null')
    res, NPZ, _ = run_panel('top', T0, y, Tp, yp, ['QCD', 'top'], models, prep, [550.0, 650.0],
                            dict(sample=f'Top Landscape test, 2,000 top + 2,000 QCD random rows (rng {seed}); reference 10,000 + 10,000 disjoint rows (rng {seed + 10})'), seed)
    save(res, NPZ, 'top')


def part_qg():
    import torch
    from interp import data_qg as dq
    from interp.h5_qg_counting import Models
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; seed = SEEDS['qg']
    d = dq.DST / 'test'; ylab = np.load(d / 'label.npy'); ev, rf = split_rows(ylab, 2000, 10000, seed)
    Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy'); pidm = np.load(d / 'pid.npy', mmap_mode='r')

    def tab(ii, with_pid):
        T = H.table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1])); T['m'] = np.zeros_like(T['m'])
        if with_pid: T['ext'] = dict(pid=np.asarray(pidm[ii]).astype(np.int64) * (T['pt'] > 0))
        return T
    T0, Tp = tab(ev, True), tab(rf, False); y, yp = ylab[ev].astype(int), ylab[rf].astype(int); log(f'QG: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        real = T['pt'] > 0
        return dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], pid=T['ext']['pid'].astype(np.int64) * real, n=real.sum(1))
    M = Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    M.add_permuted('part_kin_perm', 'part_kin'); M.add_permuted('pnet_kin_perm', 'pnet_kin')
    kd, arch, pf, net = M.m['part_kin']; M.m['part_kin_untr'] = (kd, arch, pf, null_torch(net, 'untr', 0).to(dev))
    have_ln = False
    try: M.add_lnet(); have_ln = True
    except Exception as ex: log(f'QG LorentzNet not loaded: {ex!r}')
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    known = dict(part_kin='0.8263 (H12) / 0.827 (H2)', pnet_kin='0.8269 (H12)', lorentznet=M.info.get('lorentznet', {}).get('released_test_acc'))
    names = ['part_kin', 'pnet_kin'] + (['lorentznet'] if have_ln else []) + ['part_kin_perm', 'pnet_kin_perm', 'part_kin_untr']
    models = {k: dict(f=(lambda X, k=k: two(-M.logit(k, X))), src='prep', role='null' if k.endswith(('_perm', '_untr')) else 'trained',
                      known=known.get(k)) for k in names}
    res, NPZ, _ = run_panel('qg', T0, y, Tp, yp, ['gluon', 'quark'], models, prep, [500.0, 550.0],
                            dict(sample=f'QG test, 2,000 quark + 2,000 gluon random rows (rng {seed}); reference 10,000 + 10,000 disjoint rows (rng {seed + 10})',
                                 ckpt=M.info), seed)
    save(res, NPZ, 'qg')


# ============================================================================================================== figures / summary
def fig_jet(T0, y, O0):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    w = np.flatnonzero((y == 2) & (np.abs(O0['m'] - 80) < 8) & (O0['N'] >= 15))
    j = w[np.argmin(np.abs(O0['dr'][w] - np.median(O0['dr'][w])))]
    lam = 1.25; sub = lambda T: {k: (v[j:j + 1] if isinstance(v, np.ndarray) else v) for k, v in T.items()}
    Tj = sub(T0); B = H.boost(Tj, lam)[0]; A = H.ascale(Tj, lam)[0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.4), sharex=True, sharey=True)
    for a, (T, ttl) in zip(ax, ((Tj, 'unedited W jet'), (B, f'B: boost along the jet axis, pT x {lam}'), (A, f'A: angles x 1/{lam}, pT fixed'))):
        O = feats(T); de, dp = H.rel(T); real = T['pt'][0] > 0
        a.scatter(de[0, real], dp[0, real], s=6 + 400 * T['pt'][0, real] / T['pt'][0].max(), c='#2e5eaa', alpha=0.6, edgecolors='k', linewidths=0.3)
        a.set_title(f'{ttl}\nm = {O["m"][0]:.1f} GeV, jet pT = {O["jpt"][0]:.0f} GeV, prong dR = {O["dr"][0]:.3f}', fontsize=9.5)
        a.axhline(0, color='0.8', lw=0.5); a.axvline(0, color='0.8', lw=0.5); a.set_xlabel('Δη to jet axis'); a.set_aspect('equal')
    ax[0].set_ylabel('Δφ to jet axis'); lim = 0.2; ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
    fig.suptitle('Boost (B) and angle-only scaling (A) shrink the prong separation by the same factor; only A also lowers the jet mass\n'
                 '(one hls4ml W jet; marker area proportional to particle pT)', fontsize=10.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.9]); fig.savefig(OUT / 'h64_wjet_boost.png', dpi=130); log(f'wrote {OUT / "h64_wjet_boost.png"}')


LABEL = {('hls_jedi', 'jedi_n64'): 'hls4ml JEDI-linear', ('hls_small', 'dsets'): 'hls4ml Deep Sets', ('hls_small', 'tf_d64'): 'hls4ml transformer',
         ('jetclass', 'ParT_full'): 'JetClass ParT full', ('jetclass', 'ParT_kin'): 'JetClass ParT kin', ('jetclass', 'PN_full'): 'JetClass PN full',
         ('jetclass', 'PN_kin'): 'JetClass PN kin', ('top', 'ParT_kin'): 'Top ParT', ('top', 'PN_kin'): 'Top PN', ('top', 'LorentzNet'): 'Top LorentzNet',
         ('top', 'PFN'): 'Top PFN', ('top', 'EFN'): 'Top EFN', ('qg', 'part_kin'): 'QG ParT', ('qg', 'pnet_kin'): 'QG PN', ('qg', 'lorentznet'): 'QG LorentzNet'}


def part_summary():
    R = {p: json.loads((OUT / f'h64_{p}.json').read_text()) for p in ('hls_jedi', 'hls_small', 'jetclass', 'top', 'qg') if (OUT / f'h64_{p}.json').exists()}
    S = dict(parts=list(R), accuracy={}, R_w={}, holm_global={})
    for p, r in R.items():
        for mn, m in r['models'].items():
            S['accuracy'][f'{p}|{mn}'] = dict(role=m['role'], acc=m['acc'], acc_ci=m['acc_ci'], auc=m.get('auc'), known=m.get('known_acc'))
            S['R_w'][f'{p}|{mn}'] = {l: {nm: (v[nm] if abs(v[nm]['denominator']) >= DEN_MIN else dict(v[nm], value=None, ci=None, note=f'|denominator| < {DEN_MIN}'))
                                         for nm in ('R_B_w', 'R_m_w')} for l, v in r['resp'][mn].items()}
        # null band: per null copy and lam, the class with the largest |mean Delta_B| (with its 95 % interval)
        S.setdefault('null_band', {})[p] = {}
        for mn, role in r['roles'].items():
            if role != 'null': continue
            S['null_band'][p][mn] = {}
            for l in map(str, LAMS):
                c, st = max(r['resp'][mn][l]['classes'].items(), key=lambda kv: abs(kv[1]['B']['mean']))
                S['null_band'][p][mn][l] = dict(cls=c, B=st['B'])
    unbounded = {p: sorted(mn for mn, role in r['roles'].items() if role == 'null' and any(abs(st['B']['mean']) > NULL_MAX
                 for l in r['resp'][mn].values() for st in l['classes'].values())) for p, r in R.items()}
    S['nulls_left_out_of_H0_null'] = unbounded; log(f'nulls with unbounded outputs (left out of H0_null): {unbounded}')
    for fam in ('H0_inv', 'H0_ang', 'H0_null', 'McNemar_B_vs_A', 'McNemar_B_vs_R'):
        keys, ps = [], []
        for p, r in R.items():
            for k in r['holm'].get(fam, {}):
                mn, l, c = k.split('|')
                if r['roles'][mn] != 'trained': continue
                st = r['resp'][mn][l]['classes'][c]
                if fam == 'H0_null':      # nulls with unbounded outputs (mean |Delta_B| > NULL_MAX at any lam / class) are left out
                    ok = [nn for nn in st['p_vs_each_null'] if nn not in unbounded[p]]
                    ps.append(max(st['p_vs_each_null'][nn] for nn in ok))
                else:
                    ps.append(st[{'H0_inv': 'p_H0_inv_B_eq_0', 'H0_ang': 'p_H0_ang_B_eq_A'}[fam]] if fam.startswith('H0')
                              else r['wp'][mn][c]['lams'][l]['mcnemar_' + fam.split('_', 1)[1]]['p'])
                keys.append(f'{p}|{k}')
        hp = H.holm(ps) if ps else []
        S['holm_global'][fam] = dict(n_tests=len(ps), n_below_0p05=int(np.sum(np.array(hp) < 0.05)) if ps else 0, p=dict(zip(keys, hp)))
    (OUT / 'h64_summary.json').write_text(json.dumps(S, indent=1, default=float)); log('wrote h64_summary.json')
    fig_rb(R)
    try:
        T0, y, _, _, _ = hls_tables(SEEDS['hls']); fig_jet(T0, y, feats(T0))
    except Exception as ex:
        log(f'W-jet figure not redrawn: {ex!r}')


def fig_rb(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rows = []
    for p, r in R.items():
        for mn, role in r['roles'].items():
            lab = LABEL.get((p, mn), f'{ {"hls_jedi": "hls4ml", "hls_small": "hls4ml", "jetclass": "JetClass", "top": "Top", "qg": "QG"}[p]} {mn}')
            if p == 'hls_small' and role == 'ref': continue        # same reference MLPs as hls_jedi (same jets)
            rows.append((p, mn, role, lab))
    order = {'trained': 0, 'ref': 1, 'null': 2}; rows = [x for x in rows if x[2] != 'null']
    rows.sort(key=lambda x: (order[x[2]], ['hls_jedi', 'hls_small', 'jetclass', 'top', 'qg'].index(x[0])))
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(rows) + 1.6))
    cols = {0.9: '#2e5eaa', 1.1: '#d1495b'}
    for i, (p, mn, role, lab) in enumerate(rows):
        for dl, lam in ((-0.15, 0.9), (0.15, 1.1)):
            v = R[p]['resp'][mn][str(lam)]['R_B_w']
            if v['value'] is None or v['ci'] is None or abs(v['denominator']) < DEN_MIN: continue
            x = np.clip(v['value'], -1.5, 2.5); lo, hi = np.clip(v['ci'], -1.5, 2.5)
            mk = '>' if v['value'] > 2.5 else '<' if v['value'] < -1.5 else 'o'
            ax.errorbar(x, i + dl, xerr=[[x - lo], [hi - x]], fmt=mk, color=cols[lam], ms=4.5, lw=1, label=f'pT x {lam}' if i == 0 else None)
    ax.axvline(0, color='k', lw=0.8, ls='--'); ax.axvline(1, color='k', lw=0.8, ls=':')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([('ref: ' if r[2] == 'ref' else '') + r[3] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel('R_B = response to the boost / response to the angle-only scaling (Δ log-odds, class-weighted)\n'
                  'predictions: 0 = reads invariant mass only, 1 = reads angular size or m/pT; arrows = beyond the axis', fontsize=8.5)
    ax.set_title('Response to a boost at fixed mass, relative to an angle-only scaling of the same size\n(95 % bootstrap intervals; refs = MLPs on the named jet quantities)', fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc='lower right'); fig.tight_layout(); fig.savefig(OUT / 'h64_RB.png', dpi=130); log(f'wrote {OUT / "h64_RB.png"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['hls_jedi', 'hls_small', 'jetclass', 'top', 'qg', 'summary'])
    a = ap.parse_args(); log(f'H64 part {a.part}')
    if a.part == 'summary': part_summary()
    else: dict(hls_jedi=part_hls_jedi, hls_small=part_hls_small, jetclass=part_jetclass, top=part_top, qg=part_qg)[a.part]()
    log('done')
