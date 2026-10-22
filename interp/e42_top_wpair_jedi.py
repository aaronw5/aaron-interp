"""E42 (loop A, round 20, exploratory; new claim candidate A20t, relates to A3, A5n, A6): how JEDI-linear calls top jets.
Does the top call of JEDI-linear (official 3-feature n64, hls4ml 5-class, class t = label 4) use the mass of the W-candidate
subjet pair inside a three-prong jet beyond the total jet mass, and through which second-round objects S / z neurons?

PLAN (fixed before the first run)
Samples: jedi_core.sample(2000, 4201) = fit (reference MLP, unit scan, naming fits) and sample(2000, 4202) = test (all
  headline numbers; the unit scan is confirmed there).  Top arm: label-4 jets with 105 < mSD < 210 GeV and 65 < mW < 95 GeV.
  Background arm: q/g jets (labels 0, 1) in the same mSD window with mW 50-70 or 90-115 GeV.
Subjets and edits (interp.h4_top_wmass / h4_verify.edit_set, numba, conda env; the loop-D H4/H29 definition): three
  exclusive-kT subjets (E-scheme, rapidity-azimuth distances) of the 64 constituents taken as massless four-vectors
  (eta = deta, phi = dphi of the stored arrays).  W pair = the pair with mass closest to 80.4 GeV, c = the third subjet.
  Every edit moves whole subjets rigidly in (eta, phi) (pT of every particle and the subjet masses unchanged):
  W<t>  opening of the W pair rescaled about its pT-weighted centroid so that mW = t (t = 60, 100 GeV), c moved along the
        line from the W centroid so that the jet mass is restored, tau32 restored by a common subjet-size factor h.
  N<t>  control: the non-W pair (harder W daughter + c) gets the same relative mass change, W pair not moved, jet mass and
        tau32 restored.
  R<t>  control: all constituents rotated about the jet axis by the angle whose pT-weighted mean displacement equals the W edit's.
  Q80   q/g jets edited to mW = 80.4 GeV (W / N / R the same way).
  DIL   real-mass reference: all (deta, dphi) about the E-scheme axis scaled by lambda = 0.92 and 1.08 (total mass and all
        pair masses change by about the same factor, tau32 unchanged) on the eligible tops.
  Acceptance (reclustered): W |dm| < 0.5 GeV, |dtau32| < 0.02, |mW - t| < 5 GeV; N |dm| < 0.5, |dtau32| < 0.02,
  |mW - mW0| < 2 GeV; analysis set = jets where W and N are both accepted (R on the same jets).
  Network input: every variant (identity included) rebuilt about its own E-scheme axis ("recentred", the H23/H29 rc build),
  so no edit shifts the axis the network sees.  The unedited stored-axis input is used for the accuracy check.
C2ST (classifier two-sample test): HistGradientBoosting, 2-fold, AUC for edited test-sample jets vs natural jets of the fit
  sample in the same eligibility class; features A (axis-free, no pair masses) = m, mSD, tau1, tau21, tau32, n;
  B = A + mW, m12min, m123.  AUC 0.5 = indistinguishable.
Reference: MLP(m, tau32, mW) (sklearn, 32-32, early stopping) fitted on the fit sample's natural t vs q/g jets; its log-odds
  change Dref predicts how a classifier that reads the W-pair mass would respond.
Score: top log-odds s = L_t - logsumexp(L_g, L_q, L_W, L_Z); top call = argmax is t.  Per jet D_v = s_v - s_id.
Metrics (per set, test sample): W - N (the W-pair-specific effect) mean with 95 % jet bootstrap CI (2,000), paired sign-flip
  permutation p (10,000, two-sided), McNemar (exact) on the top calls W vs N; W - R; call-rate changes.  H0: W - N = 0.
  Calibration: k_DIL = sum(Dnet Dref) / sum(Dref^2) over DIL jets (network log-odds per unit of reference log-odds for a real
  total-mass change); R_WN = mean(Dnet_W - Dnet_N) / (k_DIL mean(Dref_W - Dref_N)): 1 = the network's W-pair response equals
  that of a classifier reading mW, scaled to the network's own response to a real mass change; 0 = no W-pair response.
Routing (fit sample T60, confirmed on the test sample): for each live S unit k, hold S_k at its unedited value (run_jetwise
  override) in the W60 and N60 jets; share removed = 1 - sum(D_hold) / sum(D) for the W60 response (D = s_W - s_id) and for
  W - N.  Null: 500 random 1-dim and 500 random 3-dim holds (random orthonormal directions in the standardised live-S space,
  the edited jet's S projected back to the unedited S along them); p = (1 + #null share >= share) / 501, Holm over live S
  units; the top-3 units held jointly vs the 3-dim null.  Same for the 16 z units (1-dim null in z space).
Naming: the top-3 S units fitted with the E33 rank-window hinge library (OMP, <= 8 terms, fit sample, test R^2), and probed by
  one-particle radial edits (one particle's (deta, dphi) x 1.1 on test tops; mean dS_k / dr per pT-rank window and r bin).
z-window read: D z for z0 (~75 GeV window), z7 / z14 (94-98), z10 (135-148) under W<t> and N<t>; W - N with CI and p,
  Holm over 4 units x 3 sets.
Controls: 10 untrained redraws (Jedi(untrained=True, seed 0..9)); the 7 other EDIT_NETS of E39 (official n32, perm-inv n64,
  HGQ-r150 s0-s2, float s0-s1) through e33 Model: W - N, R_WN.
Output: $AI_RESULTS/jedi/e42_inputs.npz, e42_edit.json (part edit, conda), e42.json, e42_wpair.png (part jedi, jax env).
Run:  AI_PY=conda ./job.sh e42-run 1 2 'python -m interp.e42_top_wpair_jedi --part edit &&
      /j-jepa-vol/phat-jet-aaron/env/bin/python -m interp.e42_top_wpair_jedi --part jedi'   (E42_SMOKE=<n>: n jets per class)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
T00 = time.time()
SMOKE = int(os.environ.get('E42_SMOKE', '0'))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
TAG = 'e42_smoke' if SMOKE else 'e42'
PER = SMOKE or 2000
SEEDS = (4201, 4202)
FIT, TEST = SEEDS
SETS = ('T60', 'T100', 'Q80')
TARGET = {'T60': 60.0, 'T100': 100.0, 'Q80': 80.4}
LAMS = (0.85, 0.92, 0.96, 1.04, 1.08, 1.15)
OBSK = ('m', 'mSD', 'tau1', 'tau21', 'tau32', 'mW', 'm12min', 'm123', 'n')
NB = 400 if SMOKE else 2000
NPERM = 1000 if SMOKE else 10000
NNULL = 100 if SMOKE else 500
NUNTR = 2 if SMOKE else 10
EDIT_NETS = ['official_n32', 'perminv_n64', 'hgq150_s0', 'hgq150_s1', 'hgq150_s2', 'float_s0', 'float_s1']
ZW = {'z0': 0, 'z7': 7, 'z14': 14, 'z10': 10}


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 6)
    return o


# ================================================================================================ part edit (conda)
def dilate(P4, nn, lam):
    from interp.h29_jedi_wpair_rebuild import p4_of, axis, wrap
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None] < nn[:, None]
    pt = np.hypot(P[..., 0], P[..., 1]); ptc = np.where(real, pt, 1.0)
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / ptc); phi = np.arctan2(P[..., 1], P[..., 0])
    ae, ap = axis(P4, nn); dx = eta - ae[:, None]; dy = wrap(phi - ap[:, None])
    return p4_of(pt * real, ae[:, None] + lam * dx, ap[:, None] + lam * dy)


def part_edit():
    from interp import h4_top_wmass as H
    from interp import h4_verify as HV
    from interp.h29_jedi_wpair_rebuild import p4_of, axis, model_x
    from interp.jedi_core import sample
    OUT.mkdir(parents=True, exist_ok=True)
    D, meta = {}, dict(node='E42', smoke=SMOKE, per_class=PER, seeds=dict(fit=FIT, test=TEST), sets={}, dil={})
    OBS = {}
    for seed in SEEDS:
        x, y, _ = sample(PER, seed); real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
        assert all(real[j, :nn[j]].all() for j in range(len(nn))), 'padding not contiguous'
        P4 = H._identity(p4_of(x[..., 0], x[..., 1], x[..., 2]), nn).astype(np.float32)
        log(f'[{seed}] sample N {len(y)}; identity rebuild vs stored max |d| {np.abs(model_x(P4, nn) - x).max():.2e}')
        D[f'y__{seed}'] = y; D[f'U__{seed}__st'] = x; D[f'U__{seed}__rc'] = model_x(P4, nn, axis(P4, nn))
        tq = np.flatnonzero(np.isin(y, [0, 1, 4])); Pt, nt = P4[tq], nn[tq]; yb = (y[tq] == 4).astype(int)
        O = H._obs(Pt, nt); log(f'[{seed}] observables on {len(tq)} t+q/g jets')
        D[f'tq__{seed}'] = tq
        for k in OBSK: D[f'O__{seed}__nat__{k}'] = O[k]
        OBS[(seed, 'nat')] = O
        win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nt >= 3)
        eT = win & (yb == 1) & (O['mW'] > 65) & (O['mW'] < 95)
        eQ = win & (yb == 0) & (((O['mW'] > 50) & (O['mW'] < 70)) | ((O['mW'] > 90) & (O['mW'] < 115)))
        D[f'eligT__{seed}'] = eT; D[f'eligQ__{seed}'] = eQ
        log(f'[{seed}] tops {int(yb.sum())}, eligible tops {int(eT.sum())}, eligible q/g {int(eQ.sum())}')
        for key in SETS:
            t = TARGET[key]; rows = np.flatnonzero(eQ if key == 'Q80' else eT); t1 = time.time()
            ed = HV.edit_set(Pt, nt, rows, t); acc = {}
            for k in 'WN':
                e = ed[k]; rr = e['rows']
                if len(rr) == 0: acc[k] = set(); continue
                Oe = H._obs(e['P'], nt[rr]); dm = np.abs(Oe['m'] - O['m'][rr]); dtau = np.abs(Oe['tau32'] - O['tau32'][rr])
                okW = np.abs(Oe['mW'] - t) < 5.0 if k == 'W' else np.abs(Oe['mW'] - O['mW'][rr]) < 2
                acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
            ar = np.array(sorted(acc['W'] & acc['N'] & set(ed['R']['rows'].tolist())), np.int64)
            cnt = dict(eligible=int(len(rows)), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(len(ar)), seconds=round(time.time() - t1))
            meta['sets'][f'{seed}|{key}'] = dict(counts=cnt)
            log(f'[{seed}] {key}: {cnt}')
            if len(ar) < 5: continue
            Vv = {'id': Pt[ar]}
            for k in 'WNR': Vv[k] = ed[k]['P'][np.searchsorted(ed[k]['rows'], ar)]
            n0 = nt[ar]; D[f'rows__{seed}__{key}'] = ar
            med = {}
            for v, P in Vv.items():
                D[f'X__{seed}__{key}__{v}'] = model_x(P, n0, axis(P, n0)); Ov = H._obs(P, n0); OBS[(seed, key, v)] = Ov
                for k in OBSK: D[f'O__{seed}__{key}__{v}__{k}'] = Ov[k]
                med[v] = {k: float(np.median(Ov[k])) for k in ('m', 'mSD', 'tau32', 'mW')}
            meta['sets'][f'{seed}|{key}']['median_obs'] = med
            meta['sets'][f'{seed}|{key}']['abs_dtau32_median'] = {v: float(np.median(np.abs(OBS[(seed, key, v)]['tau32'] - OBS[(seed, key, 'id')]['tau32']))) for v in 'WNR'}
            log(f'[{seed}] {key} medians mW id/W/N/R ' + '/'.join(f'{med[v]["mW"]:.1f}' for v in ('id', 'W', 'N', 'R'))
                + ' m ' + '/'.join(f'{med[v]["m"]:.1f}' for v in ('id', 'W', 'N', 'R')))
            if seed == TEST and key == 'T60':   # example jet for the figure: the analysis jet with mW0 closest to the median
                j = int(np.argmin(np.abs(O['mW'][ar] - np.median(O['mW'][ar])))); n_ = int(n0[j])
                F = np.zeros((1, Pt.shape[1], 4)); F[0] = Pt[ar[j]]; lab = H.kt3_batch(F, n0[j:j + 1])[0, :n_]
                ex = dict(row_tq=int(ar[j]), n=n_, lab=lab.tolist(), pairs={})
                for v in ('id', 'W', 'N'):
                    jt = H.Jet(Vv[v][j, :n_].astype(np.float64), lab); pm, mt = jt.pair_masses()
                    ex['pairs'][v] = dict(m=mt, **{f'{a}{b}': m_ for (a, b), m_ in pm.items()})
                    D[f'EX__{v}'] = D[f'X__{seed}__{key}__{v}'][j]
                D['EX__lab'] = lab; meta['example'] = ex
        # DIL on the eligible tops
        rows = np.flatnonzero(eT); D[f'rows__{seed}__DIL'] = rows; n0 = nt[rows]
        D[f'X__{seed}__DIL__id'] = model_x(Pt[rows], n0, axis(Pt[rows], n0)); OBS[(seed, 'DIL', 'id')] = {k: O[k][rows] for k in O}
        for k in OBSK: D[f'O__{seed}__DIL__id__{k}'] = O[k][rows]
        for lam in LAMS:
            v = f'L{lam:g}'; P = dilate(Pt[rows], n0, lam); D[f'X__{seed}__DIL__{v}'] = model_x(P, n0, axis(P, n0))
            Ov = H._obs(P, n0); OBS[(seed, 'DIL', v)] = Ov
            for k in OBSK: D[f'O__{seed}__DIL__{v}__{k}'] = Ov[k]
            meta['dil'][f'{seed}|{v}'] = dict(n=int(len(rows)), dlog_m_median=float(np.median(np.log(Ov['m'] / O['m'][rows]))),
                                              dlog_mW_median=float(np.median(np.log(Ov['mW'] / O['mW'][rows]))),
                                              abs_dtau32_median=float(np.median(np.abs(Ov['tau32'] - O['tau32'][rows]))))
            log(f'[{seed}] DIL {v}: {meta["dil"][f"{seed}|{v}"]}')
    # ---------------------------------------------------------------- reference MLP(m, tau32, mW), t vs q/g
    # REF (primary, changed after the smoke run: the 450-jet smoke fit was flat in mW): 25,000 t + 25,000 q/g hls4ml
    # train-archive jets (dev split excluded, rng 4205, first 64 particles, as H29); REFfit: the fit sample's natural t, q/g jets.
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from interp.h4_top_wmass import _auc
    import h5py
    feat = lambda O: np.column_stack([O['m'], O['tau32'], O['mW']])
    lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    V = Path('/j-jepa-vol/aaron-interp'); NREF = 2000 if SMOKE else 25000; t1 = time.time()
    Xa = np.load(V / 'data' / 'train_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(V / 'data' / 'train_meta.h5', 'r') as f: ya = f['label'][:].astype(np.int64)
    dev = np.asarray(np.load(V / 'data' / 'splits.npz')['dev'], np.int64); isdev = np.zeros(len(ya), bool); isdev[dev] = True
    r_ = np.random.default_rng(4205)
    it = np.sort(np.r_[r_.choice(np.flatnonzero(~isdev & (ya == 4)), NREF, replace=False), r_.choice(np.flatnonzero(~isdev & np.isin(ya, [0, 1])), NREF, replace=False)])
    xr = np.asarray(Xa[it, :64], np.float32); del Xa; yr = (ya[it] == 4).astype(int); nr = (xr[..., 0] > 0).sum(1).astype(np.int64)
    Or = H._obs(H._identity(p4_of(xr[..., 0], xr[..., 1], xr[..., 2]), nr).astype(np.float32), nr)
    log(f'reference rows: {len(it)} train-archive jets ({int(yr.sum())} t), read + observables in {time.time() - t1:.0f}s')
    Ot = OBS[(TEST, 'nat')]; ybt = (D[f'y__{TEST}'][D[f'tq__{TEST}']] == 4).astype(int); meta['ref'] = {}; REFS = {}
    for nm, (Ofit, yfit) in (('REF', (Or, yr)), ('REFfit', (OBS[(FIT, 'nat')], (D[f'y__{FIT}'][D[f'tq__{FIT}']] == 4).astype(int)))):
        Xf = feat(Ofit); sc = StandardScaler().fit(Xf)
        mlp = MLPClassifier(hidden_layer_sizes=(32, 32), early_stopping=True, random_state=0, max_iter=500).fit(sc.transform(Xf), yfit)
        REFS[nm] = (lambda O, mlp=mlp, sc=sc: lo(mlp.predict_proba(sc.transform(feat(O)))[:, 1]))
        meta['ref'][nm] = dict(features=['m', 'tau32', 'mW'], fit_jets=int(len(yfit)), iterations=int(mlp.n_iter_), val_acc=float(mlp.best_validation_score_),
                               test_auc_t_vs_qg=float(_auc(REFS[nm](Ot), ybt)))
        log(f'reference {nm} MLP(m, tau32, mW): {meta["ref"][nm]}')
    for key_, O in OBS.items():
        if key_[1] == 'nat': continue
        for nm, f in REFS.items(): D[f'{nm}__{key_[0]}__{key_[1]}__{key_[2]}'] = f(O)
    # ---------------------------------------------------------------- C2ST of edits vs natural fit-sample jets
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    FA = ('m', 'mSD', 'tau1', 'tau21', 'tau32', 'n'); FB = FA + ('mW', 'm12min', 'm123')

    def c2st(Oa, Ob, fs):
        Xa = np.column_stack([Oa[k] for k in fs]); Xb = np.column_stack([Ob[k] for k in fs])
        X = np.concatenate([Xa, Xb]); yy = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))].astype(int); p = np.zeros(len(yy))
        for tr, te in StratifiedKFold(2, shuffle=True, random_state=0).split(X, yy):
            p[te] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
        return float(_auc(p, yy))
    On = OBS[(FIT, 'nat')]
    natT = {k: On[k][D[f'eligT__{FIT}']] for k in OBSK}; natQ = {k: On[k][D[f'eligQ__{FIT}']] for k in OBSK}
    meta['c2st'] = {}
    for key in SETS + ('DIL',):
        vs = ('id', 'W', 'N', 'R') if key != 'DIL' else ('id',) + tuple(f'L{l:g}' for l in LAMS)
        for v in vs:
            if (TEST, key, v) not in OBS: continue
            Oe = OBS[(TEST, key, v)]; ref_ = natQ if key == 'Q80' else natT
            if len(Oe['m']) < 10: continue
            meta['c2st'][f'{key}|{v}'] = dict(n=int(len(Oe['m'])), auc_A=c2st(ref_, Oe, FA), auc_B=c2st(ref_, Oe, FB))
            log(f'C2ST {key} {v}: {meta["c2st"][f"{key}|{v}"]}')
    np.savez_compressed(OUT / f'{TAG}_inputs.npz', **D)
    (OUT / f'{TAG}_edit.json').write_text(json.dumps(clean(meta), indent=1))
    log(f'wrote {TAG}_inputs.npz ({len(D)} arrays), {TAG}_edit.json')


# ================================================================================================ part jedi (jax env)
def lse(a):
    m = a.max(1, keepdims=True); return (m + np.log(np.exp(a - m).sum(1, keepdims=True)))[:, 0]


def score(L):
    L = np.asarray(L, np.float64); return L[:, 4] - lse(L[:, :4])


def call(L): return np.asarray(L).argmax(1) == 4


class Boot:
    def __init__(self, J, seed):
        rng = np.random.default_rng(seed); self.J = J; self.W = np.zeros((NB, J), np.float64)
        for b in range(NB): self.W[b] = np.bincount(rng.integers(0, J, J), minlength=J)

    def means(self, v): return self.W @ np.asarray(v, np.float64) / self.J

    def ci(self, v): m = self.means(v); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]

    def ratio_ci(self, a, b):
        r = self.means(a) / self.means(b); return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def perm_p(x, rng, n=NPERM):
    x = np.asarray(x, float); obs = abs(x.mean()); cnt = 0; done = 0
    while done < n:
        k = min(1000, n - done); s = rng.choice([-1.0, 1.0], (k, len(x))); cnt += int(np.sum(np.abs((s * x).mean(1)) >= obs - 1e-15)); done += k
    return float((cnt + 1) / (n + 1))


def binom_p(k, n):
    from scipy.stats import binomtest
    return 1.0 if n == 0 else float(binomtest(k, n, 0.5).pvalue)


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(rate_a=float(a.mean()), rate_b=float(b.mean()), a_only=n10, b_only=n01, p=binom_p(n10, n10 + n01))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def kdil(Z, sc, seed, ref_key='REF'):
    """calibration of the network against the reference on the DIL jets (real total-mass changes).
    k = antisymmetric (linear-order) ratio: sum over lambda pairs (1/l, l) of [Dnet(l+) - Dnet(l-)] / same for the reference,
    pooled over jets (changed after the smoke run: the network's DIL response is even in lambda (both directions lower the top
    score), so a through-origin fit of Dnet on Dref mixes the even part into the slope; the through-origin k is kept as k_fit)."""
    s0 = sc[('DIL', 'id')]; r0 = Z[f'{ref_key}__{seed}__DIL__id']
    dn = {l: sc[('DIL', f'L{l:g}')] - s0 for l in LAMS}; dr = {l: Z[f'{ref_key}__{seed}__DIL__L{l:g}'] - r0 for l in LAMS}
    pairs = [(a, b) for a, b in zip(sorted(l for l in LAMS if l < 1), sorted((l for l in LAMS if l > 1), reverse=True))]
    num = sum(dn[b] - dn[a] for a, b in pairs); den = sum(dr[b] - dr[a] for a, b in pairs)
    bt = Boot(len(num), 4299); k = float(num.mean() / den.mean())
    an = np.concatenate([dn[l] for l in LAMS]); ar = np.concatenate([dr[l] for l in LAMS])
    per = {f'L{l:g}': dict(dnet=float(dn[l].mean()), dnet_ci=bt.ci(dn[l]), dref=float(dr[l].mean()), dref_ci=bt.ci(dr[l])) for l in LAMS}
    per['k_fit_through_origin'] = float((an * ar).sum() / (ar ** 2).sum()); per['pairs'] = pairs
    per['dnet_dlnm_linear'] = float(num.mean() / sum(np.log(b / a) for a, b in pairs)); per['dref_dlnm_linear'] = float(den.mean() / sum(np.log(b / a) for a, b in pairs))
    return k, bt.ratio_ci(num, den), per


def contrasts(Z, sc, lg, seed, key, k_dil, rng):
    """W vs N / R statistics for one set; sc[(key, v)] scores, lg[(key, v)] logits"""
    s0, sW, sN, sR = (sc[(key, v)] for v in ('id', 'W', 'N', 'R')); n = len(s0); bt = Boot(n, 4290 + n)
    dW, dN, dR = sW - s0, sN - s0, sR - s0; x = dW - dN; xr = dW - dR
    rW = Z[f'REF__{seed}__{key}__W'] - Z[f'REF__{seed}__{key}__id']; rN = Z[f'REF__{seed}__{key}__N'] - Z[f'REF__{seed}__{key}__id']
    rf = (Z[f'REFfit__{seed}__{key}__W'] - Z[f'REFfit__{seed}__{key}__N'])
    c0, cW, cN, cR = (call(lg[(key, v)]) for v in ('id', 'W', 'N', 'R'))
    M = dict(n=n, dW=float(dW.mean()), dW_ci=bt.ci(dW), dN=float(dN.mean()), dN_ci=bt.ci(dN), dR=float(dR.mean()), dR_ci=bt.ci(dR),
             W_minus_N=float(x.mean()), W_minus_N_ci=bt.ci(x), p_perm_WN=perm_p(x, rng), W_minus_R=float(xr.mean()), W_minus_R_ci=bt.ci(xr),
             p_perm_WR=perm_p(xr, rng), ref_W_minus_N=float((rW - rN).mean()), ref_W_minus_N_ci=bt.ci(rW - rN),
             reffit_W_minus_N=float(rf.mean()), reffit_W_minus_N_ci=bt.ci(rf), ref_W=float(rW.mean()), ref_N=float(rN.mean()),
             calls=dict(id=float(c0.mean()), W=float(cW.mean()), N=float(cN.mean()), R=float(cR.mean()),
                        changed_W=float((cW != c0).mean()), changed_N=float((cN != c0).mean()), mcnemar_W_vs_N=mcnemar(cW, cN),
                        mcnemar_W_vs_R=mcnemar(cW, cR)))
    if k_dil is not None:
        den = k_dil * (rW - rN)
        M['R_WN'] = float(x.mean() / den.mean()); M['R_WN_ci'] = bt.ratio_ci(x, den)
        M['R_W'] = float(dW.mean() / (k_dil * rW).mean()); M['R_W_ci'] = bt.ratio_ci(dW, k_dil * rW)
        M['pred_W_minus_N_if_mW_reader'] = float(den.mean())
    return M


def part_jedi():
    import jax  # noqa: F401
    from interp.jedi_core import Jedi
    rng = np.random.default_rng(4203)
    Z = np.load(OUT / f'{TAG}_inputs.npz'); meta = json.loads((OUT / f'{TAG}_edit.json').read_text())
    R = dict(node='E42', smoke=SMOKE, plan='module docstring (fixed before the first run)', edit=meta, nets={})
    keys = lambda seed: [(k.split('__')[2], k.split('__')[3]) for k in Z.files if k.startswith(f'X__{seed}__')]
    J = Jedi(64, '3-feature')

    def score_net(run, seed):
        sc, lg = {}, {}
        for kv in keys(seed):
            L = run(np.asarray(Z[f'X__{seed}__{kv[0]}__{kv[1]}'], np.float32)); lg[kv] = L; sc[kv] = score(L)
        return sc, lg
    # ---------------------------------------------------------------- unedited accuracy and main contrasts (official)
    off = {}
    for seed in SEEDS:
        y = Z[f'y__{seed}']; Lst = J.run(Z[f'U__{seed}__st'])['logits']; Lrc = J.run(Z[f'U__{seed}__rc'])['logits']
        acc = dict(acc_stored=float((Lst.argmax(1) == y).mean()), acc_recentred=float((Lrc.argmax(1) == y).mean()),
                   top_eff_stored=float((Lst.argmax(1)[y == 4] == 4).mean()), top_eff_recentred=float((Lrc.argmax(1)[y == 4] == 4).mean()))
        log(f'[{seed}] official unedited: {acc}')
        sc, lg = score_net(lambda x: J.run(x)['logits'], seed); off[seed] = (sc, lg)
        k, kci, per = kdil(Z, sc, seed); res = dict(unedited=acc, k_dil=k, k_dil_ci=kci, dil=per, sets={})
        log(f'[{seed}] k_DIL {k:.3f} {np.round(kci, 3)}; {per}')
        for key in SETS:
            if (key, 'W') not in sc: continue
            res['sets'][key] = M = contrasts(Z, sc, lg, seed, key, k, rng)
            log(f'[{seed}] {key} n {M["n"]}: dW {M["dW"]:+.3f} dN {M["dN"]:+.3f} dR {M["dR"]:+.3f}; W-N {M["W_minus_N"]:+.3f} {np.round(M["W_minus_N_ci"], 3)} '
                f'p {M["p_perm_WN"]:.3g}; W-R {M["W_minus_R"]:+.3f}; ref W-N {M["ref_W_minus_N"]:+.3f}; R_WN {M["R_WN"]:+.2f} {np.round(M["R_WN_ci"], 2)}; '
                f'calls id/W/N {M["calls"]["id"]:.3f}/{M["calls"]["W"]:.3f}/{M["calls"]["N"]:.3f} McNemar p {M["calls"]["mcnemar_W_vs_N"]["p"]:.3g}')
        R['nets'][f'official_n64|{seed}'] = res
    fam = [(f'{s}|{k}|{t}', R['nets'][f'official_n64|{s}']['sets'][k][f]) for s in SEEDS for k in R['nets'][f'official_n64|{s}']['sets']
           for t, f in (('W-N perm', 'p_perm_WN'), ('W-R perm', 'p_perm_WR'))]
    fam += [(f'{s}|{k}|McNemar W vs N', R['nets'][f'official_n64|{s}']['sets'][k]['calls']['mcnemar_W_vs_N']['p']) for s in SEEDS for k in R['nets'][f'official_n64|{s}']['sets']]
    hp = holm([p for _, p in fam]); R['holm_main'] = [dict(test=t, p=p, p_holm=float(h)) for (t, p), h in zip(fam, hp)]
    # ---------------------------------------------------------------- z-window read (test sample)
    zr = {}; zp = []
    sc, lg = off[TEST]
    for key in SETS:
        if (key, 'W') not in sc: continue
        Zs = {v: J.run(np.asarray(Z[f'X__{TEST}__{key}__{v}'], np.float32), keep=('z',))['z'] for v in ('id', 'W', 'N')}
        bt = Boot(len(Zs['id']), 4211)
        for nm, u in ZW.items():
            dW = Zs['W'][:, u] - Zs['id'][:, u]; dN = Zs['N'][:, u] - Zs['id'][:, u]; x = dW - dN
            zr[f'{key}|{nm}'] = dict(dW=float(dW.mean()), dW_ci=bt.ci(dW), dN=float(dN.mean()), dN_ci=bt.ci(dN), W_minus_N=float(x.mean()),
                                     W_minus_N_ci=bt.ci(x), p_perm=perm_p(x, rng), sd_natural=float(Zs['id'][:, u].std()))
            zp.append((f'{key}|{nm}', zr[f'{key}|{nm}']['p_perm']))
            log(f'z read {key} {nm}: dW {dW.mean():+.3f} dN {dN.mean():+.3f} W-N {x.mean():+.3f} {np.round(bt.ci(x), 3)} p {zr[f"{key}|{nm}"]["p_perm"]:.3g}')
    for (t, _), h in zip(zp, holm([p for _, p in zp])): zr[t]['p_holm'] = float(h)
    R['z_read'] = zr
    # ---------------------------------------------------------------- unit holds (fit sample T60 scan, test confirmation)
    R['holds'] = holds(J, Z, rng)
    # ---------------------------------------------------------------- naming of the top-3 S units
    R['naming'] = naming(J, Z, R['holds'])
    # ---------------------------------------------------------------- untrained redraws
    unt = {}
    for s in range(NUNTR):
        Ju = Jedi(64, '3-feature', untrained=True, seed=s); scu, lgu = score_net(lambda x: Ju.run(x)['logits'], TEST)
        y = Z[f'y__{TEST}']; acc = float((Ju.run(Z[f'U__{TEST}__st'])['logits'].argmax(1) == y).mean())
        unt[s] = dict(acc=acc, sets={k: contrasts(Z, scu, lgu, TEST, k, None, rng) for k in SETS if (k, 'W') in scu})
        log(f'untrained seed {s}: acc {acc:.4f}; ' + '; '.join(f'{k} W-N {v["W_minus_N"]:+.4f} dW {v["dW"]:+.4f}' for k, v in unt[s]['sets'].items()))
        del Ju
    R['untrained'] = unt
    # ---------------------------------------------------------------- other EDIT_NETS (E39 set)
    from interp.e33_program_recurrence_jedi import Model
    oth = {}
    for nm in EDIT_NETS:
        try:
            M = Model(nm)
        except Exception as e:  # noqa: BLE001
            log(f'{nm}: load failed {e!r}'); oth[nm] = dict(error=repr(e)); continue
        scm, lgm = score_net(lambda x: M.run(x)['logits'], TEST); y = Z[f'y__{TEST}']
        acc = float((M.run(Z[f'U__{TEST}__st'])['logits'].argmax(1) == y).mean()); k, kci, per = kdil(Z, scm, TEST)
        oth[nm] = dict(acc=acc, k_dil=k, k_dil_ci=kci, dil=per, sets={kk: contrasts(Z, scm, lgm, TEST, kk, k, rng) for kk in SETS if (kk, 'W') in scm})
        log(f'{nm}: acc {acc:.4f} k_DIL {k:.3f}; ' + '; '.join(f'{kk} W-N {v["W_minus_N"]:+.3f} {np.round(v["W_minus_N_ci"], 3)} R_WN {v["R_WN"]:+.2f}'
                                                             for kk, v in oth[nm]['sets'].items()))
        del M
    R['edit_nets'] = oth
    (OUT / f'{TAG}.json').write_text(json.dumps(clean(R), indent=1)); log(f'wrote {TAG}.json')
    try:
        figure(R, Z)
    except Exception as e:  # noqa: BLE001
        log(f'figure failed: {e!r}')


def rand_basis(d, k, rng):
    q, _ = np.linalg.qr(rng.standard_normal((d, k))); return q


def hold_scan(J, Z, seed, key, stage, live, sd, rng, units=None, nnull=NNULL, ks=(1, 3), joint=None):
    """share of the W response (s_W - s_id) and of W - N removed by holding units of `stage` at their unedited values"""
    xs = {v: np.asarray(Z[f'X__{seed}__{key}__{v}'], np.float32) for v in ('id', 'W', 'N')}
    st = {v: J.run(xs[v], keep=(stage, 'logits')) for v in xs}
    A = {v: st[v][stage] for v in xs}; s = {v: score(st[v]['logits']) for v in xs}
    n = len(s['id']); bt = Boot(n, 4220 + n); W = A['id'].shape[1]
    resp = s['W'] - s['id']; wn = s['W'] - s['N']

    def run_hold(vals_W, vals_N, mask):
        hW = score(J.run_jetwise(xs['W'], {stage: (mask, vals_W)})['logits'])
        hN = score(J.run_jetwise(xs['N'], {stage: (mask, vals_N)})['logits'])
        return hW, hN
    full = np.zeros(W, np.float32); full[live] = 1
    hW0, hN0 = run_hold(A['W'], A['N'], full)
    chk = float(np.abs(hW0 - s['W']).max())

    def shares(hW, hN):
        rW = resp - (hW - s['id']); rWN = wn - (hW - hN)
        return rW, rWN
    out = dict(n=n, resp_W=float(resp.mean()), resp_W_ci=bt.ci(resp), W_minus_N=float(wn.mean()), W_minus_N_ci=bt.ci(wn),
               identity_hold_max_dscore=chk, units={})
    units = live if units is None else units
    for u in units:
        m = np.zeros(W, np.float32); m[u] = 1
        hW, hN = run_hold(A['id'], A['id'], m); rW, rWN = shares(hW, hN)
        out['units'][int(u)] = dict(share_W=float(rW.sum() / resp.sum()), share_W_ci=bt.ratio_ci(rW, resp),
                                    share_WN=float(rWN.sum() / wn.sum()), removed_W=float(rW.mean()), removed_W_ci=bt.ci(rW))
    # null: random directions in the standardised live space
    nulls = {}
    sdl = np.maximum(sd[live], 1e-9)
    for k in ks:
        sh = []
        for _ in range(nnull):
            U = rand_basis(len(live), k, rng)

            def proj(Ae):
                d = (A['id'][:, live] - Ae[:, live]) / sdl; out_ = Ae.copy(); out_[:, live] = Ae[:, live] + (d @ U @ U.T) * sdl; return out_
            hW, hN = run_hold(proj(A['W']), proj(A['N']), full); rW, _ = shares(hW, hN); sh.append(float(rW.sum() / resp.sum()))
        nulls[k] = np.array(sh)
        out[f'null{k}'] = dict(mean=float(nulls[k].mean()), q95=float(np.quantile(nulls[k], 0.95)), q99=float(np.quantile(nulls[k], 0.99)), max=float(nulls[k].max()))
    if 1 in nulls:
        ps = []
        for u in units:
            sh_ = out['units'][int(u)]['share_W']; p = float((1 + np.sum(nulls[1] >= sh_)) / (len(nulls[1]) + 1)); out['units'][int(u)]['p_null1'] = p; ps.append(p)
        for u, h in zip(units, holm(ps)): out['units'][int(u)]['p_holm'] = float(h)
    top = sorted(units, key=lambda u: -out['units'][int(u)]['share_W'])[:3] if joint is None else list(joint)
    m = np.zeros(W, np.float32); m[top] = 1; hW, hN = run_hold(A['id'], A['id'], m); rW, rWN = shares(hW, hN)
    out['top3'] = dict(units=[int(u) for u in top], share_W=float(rW.sum() / resp.sum()), share_W_ci=bt.ratio_ci(rW, resp),
                       share_WN=float(rWN.sum() / wn.sum()))
    if 3 in nulls: out['top3']['p_null3'] = float((1 + np.sum(nulls[3] >= out['top3']['share_W'])) / (len(nulls[3]) + 1))
    return out


def holds(J, Z, rng):
    res = {}
    for stage in ('S', 'z'):
        A = J.run(np.asarray(Z[f'U__{FIT}__st'], np.float32), keep=(stage,))[stage]; sd = A.std(0); live = [int(u) for u in np.flatnonzero(sd > 1e-9)]
        log(f'holds {stage}: {len(live)} live units')
        f = hold_scan(J, Z, FIT, 'T60', stage, live, sd, rng)
        log(f'  fit T60: resp_W {f["resp_W"]:+.3f} W-N {f["W_minus_N"]:+.3f}; identity hold max d {f["identity_hold_max_dscore"]:.2e}; null1 q95 {f["null1"]["q95"]:.3f} '
            f'null3 q95 {f["null3"]["q95"]:.3f}; top3 {f["top3"]}')
        top = sorted(live, key=lambda u: -f['units'][u]['share_W'])[:8]
        for u in top:
            q = f['units'][u]; log(f'    {stage}{u}: share_W {q["share_W"]:+.3f} {np.round(q["share_W_ci"], 3)} share_WN {q["share_WN"]:+.3f} p {q["p_null1"]:.3g} Holm {q["p_holm"]:.3g}')
        t = hold_scan(J, Z, TEST, 'T60', stage, live, sd, rng, units=top, ks=(1, 3), joint=f['top3']['units'])
        log(f'  test T60 confirm: resp_W {t["resp_W"]:+.3f}; top3 {t["top3"]}; ' + '; '.join(f'{stage}{u} {t["units"][u]["share_W"]:+.3f}' for u in top))
        t100 = hold_scan(J, Z, TEST, 'T100', stage, live, sd, rng, units=top, ks=(1,), joint=f['top3']['units']) if f'X__{TEST}__T100__W' in Z.files else None
        if t100: log(f'  test T100: resp_W {t100["resp_W"]:+.3f}; top3 {t100["top3"]}')
        res[stage] = dict(live=live, fit_T60=f, test_T60=t, test_T100=t100, ranked_fit=top)
    return res


def naming(J, Z, H):
    from interp.e33_program_recurrence_jedi import library, omp
    units = H['S']['fit_T60']['top3']['units']
    xf = np.asarray(Z[f'U__{FIT}__st'], np.float32); xt = np.asarray(Z[f'U__{TEST}__st'], np.float32)
    Sf = J.run(xf, keep=('S',))['S']; St = J.run(xt, keep=('S',))['S']
    Lf, names = library(xf); Lt, _ = library(xt); out = {}
    for u in units:
        out[int(u)] = dict(omp=omp(Lf, Sf[:, u], Lt, St[:, u], names)); log(f'naming S{u}: {out[int(u)]["omp"]["terms"]} test R2 {out[int(u)]["omp"]["r2_test"]}')
    # one-particle radial edits on the test-sample eligible tops (recentred inputs)
    rows = np.flatnonzero(Z[f'eligT__{TEST}'])
    x0 = np.asarray(Z[f'X__{TEST}__DIL__id'], np.float32); S0 = J.run(x0, keep=('S',))['S']
    RW = [(0, 4), (4, 14), (14, 64)]; RB = [0, 0.1, 0.2, 0.4, 0.8]
    acc = {u: np.zeros((len(RW), len(RB) - 1)) for u in units}; cnt = np.zeros((len(RW), len(RB) - 1))
    r0 = np.hypot(x0[..., 1], x0[..., 2]); real = x0[..., 0] > 0
    for i in range(64):
        ok = real[:, i]
        if ok.sum() == 0: continue
        x1 = x0.copy(); x1[:, i, 1:] *= 1.1; S1 = J.run(x1, keep=('S',))['S']; dr = 0.1 * r0[:, i]
        wi = next(q for q, (a, b) in enumerate(RW) if a <= i < b)
        for rb in range(len(RB) - 1):
            m = ok & (r0[:, i] >= RB[rb]) & (r0[:, i] < RB[rb + 1]) & (dr > 1e-6)
            if m.sum() == 0: continue
            cnt[wi, rb] += m.sum()
            for u in units: acc[u][wi, rb] += ((S1[m, u] - S0[m, u]) / dr[m]).sum()
    out['one_particle'] = dict(rank_windows=RW, r_bins=RB, n=cnt.tolist(), n_jets=int(len(rows)),
                               dS_dr={int(u): (acc[u] / np.maximum(cnt, 1)).tolist() for u in units},
                               S_sd_natural={int(u): float(Sf[:, u].std()) for u in units})
    for u in units:
        log(f'one-particle radial dS{u}/dr (rows rank windows {RW}, cols r bins {RB}): {np.round(acc[u] / np.maximum(cnt, 1), 3).tolist()}')
    return out


def figure(R, Z):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    lab = Z['EX__lab']; ex = R['edit'].get('example', {}); cols = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (v, mk, ttl) in enumerate((('id', 'o', 'unedited'), ('W', '^', 'W pair to 60 GeV'), ('N', 's', 'non-W pair control'))):
        x = Z[f'EX__{v}']; n = len(lab)
        for k in range(3):
            m = lab == k
            ax[0].scatter(x[:n][m, 1] + 0.0, x[:n][m, 2], s=4 + 3 * x[:n][m, 0], marker=mk, facecolors='none' if i else cols[k],
                          edgecolors=cols[k], alpha=0.8 if i == 0 else 0.5, label=f'{ttl}' if k == 0 else None)
    pr = ex.get('pairs', {})
    if pr:
        q = min(('01', '02', '12'), key=lambda q: abs(pr['id'][q] - 80.4))
        ax[0].set_title(f"Example top jet: W-pair mass {pr['id'][q]:.0f} (unedited), {pr['W'][q]:.0f} (W edit), {pr['N'][q]:.0f} GeV (control);\n"
                        f"jet mass {pr['id']['m']:.0f} / {pr['W']['m']:.0f} / {pr['N']['m']:.0f} GeV; colours = 3 subjets", fontsize=8)
    ax[0].set_xlabel(r'$\Delta\eta$ from own jet axis'); ax[0].set_ylabel(r'$\Delta\phi$ from own jet axis'); ax[0].legend(fontsize=7, loc='best')
    # panel b: W - N per net
    rows = []
    for key in ('T60', 'T100', 'Q80'):
        o = R['nets'][f'official_n64|{TEST}']['sets'].get(key)
        if o is None: continue
        rows.append((f'official {key}', o['W_minus_N'], o['W_minus_N_ci'], 'k'))
        rows.append((f'  mW reader (scaled) {key}', o['pred_W_minus_N_if_mW_reader'], None, 'tab:red'))
        for nm, v in R['edit_nets'].items():
            if 'sets' in v and key in v['sets']: rows.append((f'  {nm}', v['sets'][key]['W_minus_N'], v['sets'][key]['W_minus_N_ci'], 'tab:gray'))
        u = [v['sets'][key]['W_minus_N'] for v in R['untrained'].values() if key in v['sets']]
        if u: rows.append((f'  untrained (max |.| of {len(u)})', float(max(u, key=abs)), None, 'tab:purple'))
    yy = np.arange(len(rows))[::-1]
    for y_, (nm, v, ci, c) in zip(yy, rows):
        ax[1].plot([v], [y_], 'o', color=c, ms=4)
        if ci: ax[1].plot(ci, [y_, y_], '-', color=c, lw=1)
    ax[1].axvline(0, color='0.5', lw=0.6); ax[1].set_yticks(yy); ax[1].set_yticklabels([r[0] for r in rows], fontsize=6)
    ax[1].set_xlabel('W edit minus non-W control: change of top log-odds'); ax[1].set_title('W-pair-specific response (test sample, 95% CI)', fontsize=9)
    # panel c: hold shares
    h = R['holds']['S']['fit_T60']; us = R['holds']['S']['ranked_fit']
    v = [h['units'][u]['share_W'] for u in us]; ci = [h['units'][u]['share_W_ci'] for u in us]
    ax[2].bar(range(len(us)), v, color='tab:blue', yerr=np.abs(np.array(ci).T - np.array(v)), capsize=2)
    ax[2].axhline(h['null1']['q95'], color='tab:red', ls='--', lw=0.8, label='random 1-dim hold, 95th pct')
    ax[2].set_xticks(range(len(us))); ax[2].set_xticklabels([f'S{u}' for u in us], fontsize=7)
    ax[2].set_ylabel('share of the W60 top-score change removed'); ax[2].set_title('Holding one second-round object at its unedited value (fit sample)', fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f'{TAG}_wpair.png', dpi=130); log(f'wrote {TAG}_wpair.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['edit', 'jedi'])
    a = ap.parse_args()
    rc = 0
    try:
        part_edit() if a.part == 'edit' else part_jedi()
    except BaseException:  # noqa: BLE001
        import traceback; traceback.print_exc(); rc = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)   # the edit part's worker pool / numba threads kept the interpreter alive after the work was done (smoke run)
