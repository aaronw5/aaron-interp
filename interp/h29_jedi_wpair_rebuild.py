"""H29 (loop D; claim D1, JEDI-linear part): W-pair-mass edit inside hls4ml top jets, each edited jet built two ways.

Question. D1 says JEDI-linear shows no W-mass constraint inside top. That null (H3/H3v) was measured with every edited jet
built about the fixed stored (0, 0) axis. For JetClass ParT the analogous null was overturned once edited jets were rebuilt
about their own E-scheme axis (H23/H23v). A4: JEDI-linear's mass variable is sqrt(M0 M2) about the given axis, so the build
convention can change the result. Here every edit is scored with both builds:
  st  inputs (pT, deta, dphi) measured about the stored (0, 0) axis (as H3/H3v);
  rc  inputs measured about each (edited) jet's own E-scheme axis: the sum of the four-momenta of the 64 massless constituents
      the model sees (eta = deta, phi = dphi of the stored arrays).

Sample. hls4ml test jets (data/n64/full_test.npz, the 260k val archive), 2,000 per class drawn with default_rng(2901) from
the rows not used by H3 (sample(4000, 501)), H3v (2,000/class rng 601 from the rows minus the 501 draw) or H23 (sample(1000,
9523)); random rows per class, never first-k. Top-vs-QCD pair: t (label 4) vs q + g (labels 1, 0); W/Z jets kept for the
5-class accuracy.
Arms.
  H4-type W-pair edit (h4_verify.edit_set, as H23/H23v; three exclusive-kT subjets of the 64 particles, W pair = the pair closest
  to 80.4 GeV; subjets moved rigidly in (eta, phi) at fixed pT; jet mass restored with the third subjet, tau32 with a common
  subjet-size factor): T60, T70, T90, T100 (tops with 105 < mSD < 210, 65 < mW < 95 -> 60 / 70 / 90 / 100 GeV) and Q80 (q/g jets in
  the mSD window with mW 50-70 or 90-115 -> 80.4). Acceptance: |dm| < 0.5 GeV, |dtau32| < 0.02, |mW - t| < 5 GeV (W); N: same
  m, tau32, |mW - mW0| < 2 GeV. Controls: N (non-W pair given the same relative mass change, W pair fixed), R (rotation about
  the axis with the W edit's pT-weighted displacement), TN / TW (unedited jet translated rigidly by the E-scheme axis shift that
  N / W induce; nothing else changed).
  H3v-type k-means edit (h3_verify.wshift_all; the D1 null's own edit): W pair -> 60 / 100 GeV at fixed jet mass, control B
  (W pair held at its mass, the non-W pair moved by the same dm), control A (H3's, W pair free); TB / TE translation controls.
Models. JEDI-linear official n64 3-feature (jedi_core), the same with redrawn weights (null); models/small 5-class
  dsets-d64-n64, jedis-d64-n64, tf-2b-d32-n64 (seeds 0, 1); H16 binary hls4ml t-vs-q/g PFN with pT-fraction (hlsA-frac-pfn) and
  absolute-pT (hlsA-abs-pfn) inputs (seeds 0, 1).
Score. top-vs-QCD log-odds: 5-class s = L_t - logsumexp(L_q, L_g); binary s = L_t - L_bkg. References retrained here on
  hls4ml train-archive jets (25,000 t + 25,000 q/g, dev split excluded, first 64 particles): MLPx (mSD, tau1-3, tau32, tau21, n,
  sorted subjet pT fractions and masses; no pair mass), MLPxW (MLPx + mW), LR2 (mSD, tau32) histogram LR (h19_wpair_cross.fit_refs).
  The reference features are axis-independent, so they are scored once.
Metrics (per arm x model x build): d_v = s_v - s_id (same build); W - N with 95 % bootstrap CI over jets (B = 2000); ratio to
  MLPx and MLPxW W - N with CI; sign-flip permutation p (10,000) and one-sample t p, one-sided in the mW direction, of
  (W - N)_model - (W - N)_MLPx; two-sided vs MLPxW; McNemar W vs N calls at the eps_S 0.5 threshold (unedited tops, same build)
  and of the top argmax call; st - rc paired difference (two-sided); axis term (dN_st - dN_rc) - dTN_st; dose ordering on the
  tops accepted at all four targets (paired, one-sided: W - N more negative at 60 than 70 and at 100 than 90); Holm over the
  family (t-test p for the mean tests, exact McNemar p for calls). Also m_A = sqrt(M0 M2) about each build's axis per variant.
Parts (results $AI_RESULTS/physics/h29_*):
  --part edit   (conda: numba, sklearn) sample, edits, inputs for both builds, references   -> h29_inputs.npz, h29_refs.npz, h29_edit.json
  --part small  (conda: torch) small models                                                   -> h29_scores_small.npz, h29_models_small.json
  --part jedi   (jax env) JEDI-linear official + redrawn                                      -> h29_scores_jedi.npz, h29_models_jedi.json
  --part stats  (conda) statistics, Holm, figure                                              -> h29.json, h29_wpair_rebuild.png
Run: AI_PY=conda ./job.sh d-h29-run 2 4 'python -m interp.h29_jedi_wpair_rebuild --part edit && python -m interp.h29_jedi_wpair_rebuild
     --part small && /j-jepa-vol/phat-jet-aaron/env/bin/python -m interp.h29_jedi_wpair_rebuild --part jedi && python -m
     interp.h29_jedi_wpair_rebuild --part stats'      (H29_SMOKE=<n>: n jets per class, small references, tag h29_smoke)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T00 = time.time()
SMOKE = int(os.environ.get('H29_SMOKE', '0'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
TAG = 'h29_smoke' if SMOKE else 'h29'
SEED = 2901
PER = SMOKE or 2000
NREF = 2000 if SMOKE else 25000
B = 400 if SMOKE else 2000
NPERM = 1000 if SMOKE else 10000
TSETS = ('T60', 'T70', 'T90', 'T100')
H4SETS = TSETS + ('Q80',)
H3SETS = ('H3v_60', 'H3v_100')
TARGET = {'T60': 60.0, 'T70': 70.0, 'T90': 90.0, 'T100': 100.0, 'Q80': 80.4}
DIRECTION = {'T60': -1, 'T70': -1, 'T90': -1, 'T100': -1, 'Q80': 1, 'H3v_60': -1, 'H3v_100': -1}
CONTRASTS = {**{k: [('W', 'N'), ('W', 'R')] for k in H4SETS}, **{k: [('E', 'B'), ('E', 'A')] for k in H3SETS}}
TRANS = {**{k: {'N': 'TN', 'W': 'TW'} for k in H4SETS}, **{k: {'B': 'TB', 'E': 'TE'} for k in H3SETS}}
REFS = ('MLPx', 'MLPxW', 'LR2')


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    return o


# ================================================================================================== geometry
def p4_of(pt, eta, phi):
    """massless four-vectors (J, L, 4) of the real particles (pt > 0); padding 0"""
    pt = np.asarray(pt, np.float64); real = pt > 0
    P = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1)
    return (P * real[..., None]).astype(np.float32)


def axis(P4, nn):
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    S = (P * real[..., None]).sum(1); pt = np.hypot(S[:, 0], S[:, 1])
    return np.arcsinh(S[:, 2] / pt), np.arctan2(S[:, 1], S[:, 0])


def model_x(P4, nn, ax=None):
    """(J, 64, 3) (pT, deta, dphi) about ax = (eta, phi) per jet (None: the stored (0, 0) axis); padding 0"""
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    pt = np.hypot(P[..., 0], P[..., 1]); ptc = np.where(real, pt, 1.0)
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / ptc); phi = np.arctan2(P[..., 1], P[..., 0])
    if ax is not None: eta = eta - ax[0][:, None]; phi = wrap(phi - ax[1][:, None])
    x = np.stack([pt, eta, wrap(phi)], -1) * real[..., None]
    return x[:, :64].astype(np.float32)


def mA_of(x):
    """sqrt(M0 M2) about the input axis, M0 = sum pT, M2 = sum pT r^2 (A4's JEDI-linear mass variable)"""
    pt = x[..., 0].astype(np.float64); r2 = x[..., 1].astype(np.float64) ** 2 + x[..., 2].astype(np.float64) ** 2
    return np.sqrt(pt.sum(1) * (pt * r2).sum(1))


def translate(P4, nn, de, dp):
    from interp.h23_axis_audit import translate as tr
    return tr(P4, nn, de, dp)


# ================================================================================================== sample
def draw_sample():
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y = np.asarray(d['y']).astype(int)
    r = np.random.default_rng(501); h3 = np.concatenate([r.choice(np.flatnonzero(y == c), 4000, replace=False) for c in range(5)])
    used = np.zeros(len(y), bool); used[h3] = True
    r = np.random.default_rng(601); h3v = np.concatenate([r.choice(np.flatnonzero((y == c) & ~used), 2000, replace=False) for c in range(5)])
    r = np.random.default_rng(9523); h23 = np.concatenate([r.choice(np.flatnonzero(y == c), 1000, replace=False) for c in range(5)])
    excl = np.zeros(len(y), bool); excl[h3] = excl[h3v] = excl[h23] = True
    rng = np.random.default_rng(SEED)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & ~excl), PER, replace=False) for c in range(5)]))
    x = np.asarray(d['x'][idx], np.float32); yy = y[idx]
    info = dict(file='data/n64/full_test.npz', per_class=PER, seed=SEED, N=int(len(idx)), excluded=int(excl.sum()),
                overlap_with_excluded=int(excl[idx].sum()), excluded_draws='H3 rng 501 (4000/class), H3v rng 601 (2000/class, minus 501), H23 rng 9523 (1000/class)')
    log(f'sample {info}')
    return x, yy, idx, info


# ================================================================================================== edit part
def part_edit():
    from interp import h4_top_wmass as H
    from interp import h4_verify as HV
    from interp import h19_wpair_cross as X
    rng = np.random.default_rng(29)
    x, y5, idx, info = draw_sample()
    real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
    assert all(real[j, :nn[j]].all() for j in range(len(nn))), 'padding is not contiguous at the end'
    P4 = H._identity(p4_of(x[..., 0], x[..., 1], x[..., 2]), nn).astype(np.float32)
    xid = model_x(P4, nn); log(f'identity rebuild vs stored inputs: max |d| {np.abs(xid - x).max():.2e}')
    a0 = axis(P4, nn); off = np.hypot(a0[0], wrap(a0[1]))
    info['stored_vs_escheme_axis'] = dict(median=float(np.median(off)), p90=float(np.quantile(off, 0.9)), p99=float(np.quantile(off, 0.99)))
    log(f'stored (0,0) vs E-scheme axis of the 64 particles: {info["stored_vs_escheme_axis"]}')
    D = {'y5': y5, 'idx': idx, 'U__st': x, 'U__rc': model_x(P4, nn, a0)}
    meta = dict(node='H29', smoke=SMOKE, sample=info, sets={})
    # ---------------------------------------------------------------- H4-type edits on the t vs q/g subset
    tq = np.flatnonzero(np.isin(y5, [0, 1, 4])); yb = (y5[tq] == 4).astype(int); Pt, nt = P4[tq], nn[tq]
    O = H._obs(Pt, nt); win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nt >= 3)
    meta['tops_in_window'] = int((win & (yb == 1)).sum()); meta['tops_eligible_T'] = int((win & (yb == 1) & (O['mW'] > 65) & (O['mW'] < 95)).sum())
    elig = {k: win & (yb == 1) & (O['mW'] > 65) & (O['mW'] < 95) for k in TSETS}
    elig['Q80'] = win & (yb == 0) & (((O['mW'] > 50) & (O['mW'] < 70)) | ((O['mW'] > 90) & (O['mW'] < 115)))
    log(f'tops {int(yb.sum())}, in mSD window {meta["tops_in_window"]}, eligible for T {meta["tops_eligible_T"]}; QCD eligible for Q80 {int(elig["Q80"].sum())}')
    REFP = {}
    for key in H4SETS:
        t = TARGET[key]; rows = np.flatnonzero(elig[key])
        ed = HV.edit_set(Pt, nt, rows, t); acc = {}
        for k in 'WN':
            e = ed[k]; rr = e['rows']
            if len(rr) == 0: acc[k] = set(); continue
            Oe = H._obs(e['P'], nt[rr]); dm = np.abs(Oe['m'] - O['m'][rr]); dtau = np.abs(Oe['tau32'] - O['tau32'][rr])
            okW = np.abs(Oe['mW'] - t) < 5.0 if k == 'W' else np.abs(Oe['mW'] - O['mW'][rr]) < 2
            acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
        ar = np.array(sorted(acc['W'] & acc['N'] & set(ed['R']['rows'].tolist())), np.int64)
        if len(ar) < 5:
            meta['sets'][key] = dict(counts=dict(eligible=int(len(rows)), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(len(ar))))
            log(f'{key}: {meta["sets"][key]["counts"]} (fewer than 5 jets; skipped)'); continue
        Vv = {'id': Pt[ar]}
        for k in 'WNR': Vv[k] = ed[k]['P'][np.searchsorted(ed[k]['rows'], ar)]
        n0 = nt[ar]; ax0 = axis(Vv['id'], n0); shift = {}
        for k in 'WNR':
            ak = axis(Vv[k], n0); shift[k] = (ak[0] - ax0[0], wrap(ak[1] - ax0[1]))
        for k in 'WN': Vv[f'T{k}'] = translate(Vv['id'], n0, *shift[k])
        Oa = {v: H._obs(P, n0) for v, P in Vv.items()}
        S = dict(counts=dict(eligible=int(len(rows)), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(len(ar))),
                 rows_tq=ar, rows_sample=tq[ar],
                 axis_shift={k: dict(median=float(np.median(np.hypot(*shift[k]))), p90=float(np.quantile(np.hypot(*shift[k]), 0.9))) for k in 'WNR'},
                 obs={v: {o: float(np.median(Oa[v][o])) for o in ('m', 'mSD', 'tau32', 'tau21', 'mW')} for v in Oa}, mA={})
        for v, P in Vv.items():
            for p in ('st', 'rc'):
                xx = model_x(P, n0, None if p == 'st' else axis(P, n0)); D[f'X__{key}__{v}__{p}'] = xx
                S['mA'][f'{v}|{p}'] = float(np.median(np.log(mA_of(xx)) - np.log(mA_of(D[f'X__{key}__id__{p}']))))
            S['mA'][f'{v}|inv_m'] = float(np.median(np.log(np.maximum(Oa[v]['m'], 1e-6)) - np.log(np.maximum(Oa['id']['m'], 1e-6))))
            REFP[(key, v)] = (P, n0)
        D[f'rows__{key}'] = tq[ar]; meta['sets'][key] = S
        log(f'{key}: {S["counts"]}; axis shift W {S["axis_shift"]["W"]["median"]:.4f} N {S["axis_shift"]["N"]["median"]:.4f}; '
            f'median mW id/W/N {S["obs"]["id"]["mW"]:.1f}/{S["obs"]["W"]["mW"]:.1f}/{S["obs"]["N"]["mW"]:.1f}; dlog mA st W/N/TN '
            f'{S["mA"]["W|st"]:+.4f}/{S["mA"]["N|st"]:+.4f}/{S["mA"]["TN|st"]:+.4f} rc W/N {S["mA"]["W|rc"]:+.4f}/{S["mA"]["N|rc"]:+.4f}')
    # ---------------------------------------------------------------- H3v k-means edit on the tops (the D1 null's own edit)
    from interp import h3_verify as H3V
    from interp.h3_d3_fullstats import top_split
    tops = np.flatnonzero(y5 == 4); xt = x[tops].astype(np.float64); pt, de, dp = xt[..., 0], xt[..., 1], xt[..., 2]
    TS = top_split(pt, de, dp); log(f'H3v: tops with a W pair {int(TS["ok"].sum())}/{len(tops)}')
    for t in (60.0, 100.0):
        key = f'H3v_{t:g}'; R = H3V.wshift_all(pt, de, dp, t, TS); m = R['ok'] & R['okE'] & R['okA'] & R['okB']; rr = np.flatnonzero(m)
        if len(rr) < 5: meta['sets'][key] = dict(counts=dict(analysis=int(len(rr)))); log(f'{key}: n {len(rr)} (skipped)'); continue
        n0 = nn[tops[rr]]; Vv = {}
        for v, (de_, dp_) in (('id', (de, dp)), ('E', R['edit']), ('A', R['ctrlA']), ('B', R['ctrlB'])):
            Vv[v] = H._identity(p4_of(pt[rr], de_[rr], dp_[rr]), n0).astype(np.float32)
        ax0 = axis(Vv['id'], n0); shift = {}
        for k in ('E', 'B'):
            ak = axis(Vv[k], n0); shift[k] = (ak[0] - ax0[0], wrap(ak[1] - ax0[1])); Vv[f'T{k}'] = translate(Vv['id'], n0, *shift[k])
        S = dict(counts=dict(analysis=int(len(rr)), with_W_pair=int(TS['ok'].sum())), rows_sample=tops[rr],
                 masses={k: R['med'](R[k], m) for k in ('P0', 'PE', 'PA', 'PB')},
                 axis_shift={k: dict(median=float(np.median(np.hypot(*shift[k]))), p90=float(np.quantile(np.hypot(*shift[k]), 0.9))) for k in shift}, mA={})
        for v, P in Vv.items():
            for p in ('st', 'rc'):
                xx = model_x(P, n0, None if p == 'st' else axis(P, n0)); D[f'X__{key}__{v}__{p}'] = xx
                S['mA'][f'{v}|{p}'] = float(np.median(np.log(mA_of(xx)) - np.log(mA_of(D[f'X__{key}__id__{p}']))))
            REFP[(key, v)] = (P, n0)
        D[f'rows__{key}'] = tops[rr]; meta['sets'][key] = S
        log(f'{key}: n {len(rr)}; pair masses {json.dumps(S["masses"])}; axis shift E {S["axis_shift"]["E"]["median"]:.4f} B {S["axis_shift"]["B"]["median"]:.4f}')
    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RES / f'{TAG}_inputs.npz', **D); log(f'wrote {TAG}_inputs.npz ({len(D)} arrays)')
    # ---------------------------------------------------------------- references on hls4ml train-archive jets
    import h5py
    t0 = time.time(); Xa = np.load(V / 'data' / 'train_ptetaphi_f16.npy')
    with h5py.File(V / 'data' / 'train_meta.h5', 'r') as f: ya = f['label'][:].astype(np.int64)
    dev = np.asarray(np.load(V / 'data' / 'splits.npz')['dev'], np.int64); isdev = np.zeros(len(ya), bool); isdev[dev] = True
    rr_ = np.random.default_rng(2902)
    it = np.sort(np.r_[rr_.choice(np.flatnonzero(~isdev & (ya == 4)), NREF, replace=False), rr_.choice(np.flatnonzero(~isdev & np.isin(ya, [0, 1])), NREF, replace=False)])
    xr = np.asarray(Xa[it, :64], np.float32); del Xa; yr = (ya[it] == 4).astype(int)
    log(f'reference rows: {len(it)} train-archive jets ({int(yr.sum())} t), read in {time.time() - t0:.0f}s')
    nr = (xr[..., 0] > 0).sum(1).astype(np.int64)
    Pr = H._identity(p4_of(xr[..., 0], xr[..., 1], xr[..., 2]), nr).astype(np.float32)
    Xf, Of = H._features(Pr, nr); X.log = log
    REF, rinfo = X.fit_refs(Xf, Of, yr, 'hls')
    meta['refs'] = dict(rows=int(len(it)), seed=2902, source='data/train_ptetaphi_f16.npy (dev split excluded), first 64 particles', fit=rinfo)
    RS = {}
    Xu, Ou = H._features(Pt, nt); su = X.eval_refs(REF, Xu, Ou)
    from interp.h4_top_wmass import _auc
    meta['refs']['unedited_tq'] = {k: dict(AUC=float(_auc(v, yb)), acc=float(((v > 0) == (yb == 1)).mean())) for k, v in su.items()}
    for k, v in su.items(): RS[f'R__U__{k}'] = v
    RS['R__U__rows'] = tq
    log(f'references on the unedited t vs q/g sample: {meta["refs"]["unedited_tq"]}')
    for (key, v), (P, n0) in REFP.items():
        Xe, Oe = H._features(P, n0)
        for k, s in X.eval_refs(REF, Xe, Oe).items(): RS[f'R__{key}__{v}__{k}'] = s
    np.savez_compressed(RES / f'{TAG}_refs.npz', **RS)
    (RES / f'{TAG}_edit.json').write_text(json.dumps(jsonable(meta), indent=1)); log(f'wrote {TAG}_refs.npz, {TAG}_edit.json')


# ================================================================================================== scoring parts
def input_keys(Z):
    return [k for k in Z.files if k.startswith('X__') or k.startswith('U__')]


def score_all(models, tag):
    Z = np.load(RES / f'{TAG}_inputs.npz'); keys = input_keys(Z); out = {}
    for name, (fn, info) in models.items():
        t = time.time()
        for k in keys:
            out[f'S__{name}__{k}'] = np.asarray(fn(np.asarray(Z[k], np.float32)), np.float32)
        y5 = Z['y5']; L = out[f'S__{name}__U__st']; Lr = out[f'S__{name}__U__rc']
        if L.shape[1] == 5:
            info['acc5_st'] = float((L.argmax(1) == y5).mean()); info['acc5_rc'] = float((Lr.argmax(1) == y5).mean())
            info['top_eff_st'] = float((L.argmax(1)[y5 == 4] == 4).mean()); info['top_eff_rc'] = float((Lr.argmax(1)[y5 == 4] == 4).mean())
        else:
            m = np.isin(y5, [0, 1, 4]); yb = (y5[m] == 4)
            info['acc2_st'] = float((L[m].argmax(1) == yb).mean()); info['acc2_rc'] = float((Lr[m].argmax(1) == yb).mean())
            info['top_eff_st'] = float((L[m].argmax(1)[yb] == 1).mean()); info['top_eff_rc'] = float((Lr[m].argmax(1)[yb] == 1).mean())
        log(f'{tag} {name}: scored {len(keys)} arrays in {time.time() - t:.0f}s; unedited ' + ', '.join(f'{k} {v:.4f}' for k, v in info.items() if isinstance(v, float)))
    np.savez_compressed(RES / f'{TAG}_scores_{tag}.npz', **out)
    (RES / f'{TAG}_models_{tag}.json').write_text(json.dumps(jsonable({n: i for n, (_, i) in models.items()}), indent=1))
    log(f'wrote {TAG}_scores_{tag}.npz')


def part_small():
    import torch
    from interp.h1_irc_panel import load_small, torch_logits
    from interp import h16_train as HT
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    models = {}
    for tag, run in (('dsets', 'dsets-d64-n64'), ('jedis', 'jedis-d64-n64'), ('tf', 'tf-2b-d32-n64')):
        for s in (0, 1):
            d = V / 'models' / 'small' / run / f'seed{s}'
            if not (d / 'best.pt').exists(): log(f'missing {d}'); continue
            net, sh, sc = load_small(d); tj = json.loads((d / 'test.json').read_text())
            models[f'{tag}_s{s}'] = ((lambda x, net=net, sh=sh, sc=sc: torch_logits(net, x.copy(), sh, sc)),
                                     dict(run=str(d), classes=5, known_acc5=tj.get('accuracy'), known_top_eff=tj.get('per_class_accuracy', {}).get('t')))
    for tag, run in (('pfnfrac', 'hlsA-frac-pfn'), ('pfnabs', 'hlsA-abs-pfn')):
        for s in (0, 1):
            d = V / 'models' / 'h16' / run / f'seed{s}'
            if not (d / 'best.pt').exists(): log(f'missing {d}'); continue
            net = HT.Net(d); tj = json.loads((d / 'test.json').read_text())
            models[f'{tag}_s{s}'] = ((lambda x, net=net: net(x)), dict(run=str(d), classes=2, known_acc2=tj.get('accuracy'), known_auc=tj.get('auc')))
    log(f'small models: {list(models)}')
    score_all(models, 'small')


def part_jedi():
    from interp.jedi_core import Jedi
    nets = dict(jedi=Jedi(64, '3-feature'), jedi_redrawn=Jedi(64, '3-feature', untrained=True, seed=0))
    known = json.loads((V / 'data' / 'official_acc.json').read_text())['3-feature']['64']
    models = {k: ((lambda x, n=n: n.run(x)['logits']), dict(classes=5, known_acc5=known if k == 'jedi' else None)) for k, n in nets.items()}
    score_all(models, 'jedi')


# ================================================================================================== statistics
def tp(x, sign):
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) < 3 or np.allclose(x, x[0]): return 1.0
    t, p2 = stats.ttest_1samp(x, 0.0)
    return float(p2) if sign == 0 else float(p2 / 2 if sign * t > 0 else 1 - p2 / 2)


def perm_p(x, sign, rng, n=NPERM):
    x = np.asarray(x, float); cnt = 0; done = 0
    while done < n:
        k = min(1000, n - done); s = 2.0 * rng.integers(0, 2, (k, len(x)), dtype=np.int8) - 1.0; null = (s * x).mean(1)
        cnt += int(np.sum(np.abs(null) >= abs(x.mean()) - 1e-15)) if sign == 0 else int(np.sum(sign * null >= sign * x.mean() - 1e-15)); done += k
    return float((cnt + 1) / (n + 1))


def bci(x, bi):
    return np.percentile(np.asarray(x, float)[bi].mean(1), [2.5, 97.5]).tolist()


def ratio_ci(x, r, bi):
    bx, br = x[bi].mean(1), r[bi].mean(1)
    return float(x.mean() / r.mean()), np.percentile(bx / br, [2.5, 97.5]).tolist()


def mcnemar(a, b):
    from interp.h1_irc_panel import binom_two_sided
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(a=float(a.mean()), b=float(b.mean()), a_only=n10, b_only=n01, p=binom_two_sided(n10, n10 + n01))


def holm(ps):
    from interp.h1_irc_panel import holm as h
    return h(ps) if len(ps) else []


def score_of(L):
    L = np.asarray(L, np.float64)
    if L.shape[1] == 5: return L[:, 4] - np.logaddexp(L[:, 0], L[:, 1])
    return L[:, 1] - L[:, 0]


def topcall(L):
    return np.asarray(L).argmax(1) == (4 if L.shape[1] == 5 else 1)


def part_stats():
    from interp.h4_top_wmass import _auc
    rng = np.random.default_rng(2903)
    Z = np.load(RES / f'{TAG}_inputs.npz'); RZ = np.load(RES / f'{TAG}_refs.npz'); meta = json.loads((RES / f'{TAG}_edit.json').read_text())
    SC, MI = {}, {}
    for part in ('jedi', 'small'):
        f = RES / f'{TAG}_scores_{part}.npz'
        if not f.exists(): log(f'missing {f}'); continue
        S = np.load(f); MI.update(json.loads((RES / f'{TAG}_models_{part}.json').read_text()))
        for k in S.files: SC[k] = S[k]
    models = list(MI); y5 = Z['y5']; tq = np.isin(y5, [0, 1, 4]); yb = (y5[tq] == 4).astype(int)
    L = lambda m, k: SC[f'S__{m}__{k}']
    s = lambda m, key, v, p: score_of(L(m, f'X__{key}__{v}__{p}'))
    ref = lambda key, v, r: RZ[f'R__{key}__{v}__{r}'].astype(np.float64)
    # ---------------------------------------------------------------- unedited performance and thresholds (t vs q/g jets, per build)
    perf = {}
    for m in models:
        for p in ('st', 'rc'):
            Lu = L(m, f'U__{p}'); su = score_of(Lu)[tq]
            d = dict(AUC_t_vs_qg=float(_auc(su, yb)), thr_eS50=float(np.quantile(su[yb == 1], 0.5)), thr_eS30=float(np.quantile(su[yb == 1], 0.7)),
                     top_eff=float(topcall(Lu)[y5 == 4].mean()))
            if Lu.shape[1] == 5: d['acc5'] = float((Lu.argmax(1) == y5).mean())
            else: d['acc2_t_vs_qg'] = float((Lu[tq].argmax(1) == yb).mean())
            perf[f'{m}|{p}'] = d
        log(f'unedited {m}: ' + ' | '.join(f"{p} " + ' '.join(f'{k} {v:.4f}' for k, v in perf[f'{m}|{p}'].items() if not k.startswith('thr')) for p in ('st', 'rc'))
            + f"; known {({k: v for k, v in MI[m].items() if k.startswith('known')})}")
    fam = []; R = dict(node='H29', smoke=SMOKE, sample=meta['sample'], refs=meta['refs'], unedited=perf, models=MI, sets={})
    WN = {}
    for key in H4SETS + H3SETS:
        if f'X__{key}__id__st' not in Z.files: continue
        n = len(Z[f'rows__{key}']); Sm = dict(meta=meta['sets'][key], n=n, contrasts={})
        if n < 5: R['sets'][key] = Sm; continue
        bi = rng.integers(0, n, (B, n)); sg = DIRECTION[key]
        for a, b in CONTRASTS[key]:
            lab = f'{a}_vs_{b}'; C = {}; primary = (a, b) in (('W', 'N'), ('E', 'B'))
            rx = {r: (ref(key, a, r) - ref(key, 'id', r)) - (ref(key, b, r) - ref(key, 'id', r)) for r in REFS}
            for r in REFS: C[r] = dict(diff=float(rx[r].mean()), ci95=bci(rx[r], bi))
            for m in models:
                for p in ('st', 'rc'):
                    s0, sa, sb = s(m, key, 'id', p), s(m, key, a, p), s(m, key, b, p); da, db = sa - s0, sb - s0; x = da - db
                    M = dict(d_edit=float(da.mean()), d_edit_ci95=bci(da, bi), d_ctrl=float(db.mean()), d_ctrl_ci95=bci(db, bi),
                             diff=float(x.mean()), ci95=bci(x, bi), p_t_zero=tp(x, sg))
                    for r in ('MLPx', 'MLPxW'):
                        M[f'ratio_{r}'], M[f'ratio_{r}_ci95'] = ratio_ci(x, rx[r], bi)
                        z = x - rx[r]; M[f'minus_{r}'] = float(z.mean()); M[f'minus_{r}_ci95'] = bci(z, bi)
                    z = x - rx['MLPx']; M['p_vs_MLPx_t'] = tp(z, sg)
                    if primary: M['p_perm_zero'] = perm_p(x, sg, rng); M['p_vs_MLPx_perm'] = perm_p(z, sg, rng)
                    M['p_vs_MLPxW_t_two_sided'] = tp(x - rx['MLPxW'], 0)
                    thr = perf[f'{m}|{p}']['thr_eS50']; c0, ca, cb = s0 >= thr, sa >= thr, sb >= thr
                    M['calls_eS50'] = dict(pass_id=float(c0.mean()), pass_edit=float(ca.mean()), pass_ctrl=float(cb.mean()),
                                           changed_edit=float((ca != c0).mean()), changed_ctrl=float((cb != c0).mean()), mcnemar_edit_vs_ctrl=mcnemar(cb, ca))
                    t0_, ta, tb = (topcall(L(m, f'X__{key}__{v}__{p}')) for v in ('id', a, b))
                    M['top_argmax'] = dict(id=float(t0_.mean()), edit=float(ta.mean()), ctrl=float(tb.mean()), changed_edit=float((ta != t0_).mean()),
                                           changed_ctrl=float((tb != t0_).mean()), mcnemar_edit_vs_ctrl=mcnemar(tb, ta))
                    if primary:
                        fam.append(dict(arm=key, model=m, path=p, test=f'{lab} minus MLPx (one-sided, mW direction; t-test)', effect=M['minus_MLPx'], p=M['p_vs_MLPx_t']))
                        fam.append(dict(arm=key, model=m, path=p, test=f'{lab} eS50 calls McNemar', effect=M['calls_eS50']['changed_edit'] - M['calls_eS50']['changed_ctrl'],
                                        p=M['calls_eS50']['mcnemar_edit_vs_ctrl']['p']))
                        WN[(key, m, p)] = x
                    C[f'{m}|{p}'] = M
                if True:   # st - rc and the translation term, every contrast
                    xs = s(m, key, a, 'st') - s(m, key, b, 'st'); xr = s(m, key, a, 'rc') - s(m, key, b, 'rc')
                    dd = xs - xr; C[f'{m}|st_minus_rc'] = dict(diff=float(dd.mean()), ci95=bci(dd, bi), p_t_two_sided=tp(dd, 0))
                    if primary: fam.append(dict(arm=key, model=m, path='st-rc', test=f'{lab} st minus rc (two-sided)', effect=float(dd.mean()), p=C[f'{m}|st_minus_rc']['p_t_two_sided']))
                    T = TRANS[key].get(b)
                    if T is not None:
                        dbs = s(m, key, b, 'st') - s(m, key, 'id', 'st'); dbr = s(m, key, b, 'rc') - s(m, key, 'id', 'rc'); dT = s(m, key, T, 'st') - s(m, key, 'id', 'st')
                        dTr = s(m, key, T, 'rc') - s(m, key, 'id', 'rc'); z = (dbs - dbr) - dT
                        AT = dict(ctrl_st_minus_rc=float((dbs - dbr).mean()), ctrl_st_minus_rc_ci95=bci(dbs - dbr, bi), dT_ctrl_st=float(dT.mean()),
                                  dT_ctrl_st_ci95=bci(dT, bi), dT_ctrl_rc=float(dTr.mean()), residual=float(z.mean()), residual_ci95=bci(z, bi), p_residual_t=tp(z, 0))
                        Ta = TRANS[key].get(a)
                        if Ta is not None:
                            dTa = s(m, key, Ta, 'st') - s(m, key, 'id', 'st'); AT['dT_edit_st'] = float(dTa.mean()); AT['dT_edit_st_ci95'] = bci(dTa, bi)
                            AT['dT_edit_minus_dT_ctrl_st'] = float((dTa - dT).mean()); AT['dT_edit_minus_dT_ctrl_st_ci95'] = bci(dTa - dT, bi)
                        C[f'{m}|axis_term'] = AT
            Sm['contrasts'][lab] = C
        R['sets'][key] = Sm
        c = Sm['contrasts'][f'{CONTRASTS[key][0][0]}_vs_{CONTRASTS[key][0][1]}']
        log(f'{key} n {n}: MLPx {c["MLPx"]["diff"]:+.3f} {np.round(c["MLPx"]["ci95"], 3)} MLPxW {c["MLPxW"]["diff"]:+.3f} LR2 {c["LR2"]["diff"]:+.3f}')
        for m in models:
            q = [f"{p} {c[f'{m}|{p}']['diff']:+.3f} {np.round(c[f'{m}|{p}']['ci95'], 3)} x{c[f'{m}|{p}']['ratio_MLPx']:.2f} p {c[f'{m}|{p}']['p_vs_MLPx_t']:.2g} "
                 f"calls {100 * c[f'{m}|{p}']['calls_eS50']['changed_edit']:.1f}/{100 * c[f'{m}|{p}']['calls_eS50']['changed_ctrl']:.1f}%" for p in ('st', 'rc')]
            at = c.get(f'{m}|axis_term', {})
            log(f'   {m:14s} ' + ' | '.join(q) + f" | st-rc {c[f'{m}|st_minus_rc']['diff']:+.3f} dT {at.get('dT_ctrl_st', np.nan):+.3f} resid {at.get('residual', np.nan):+.3f}")
    # ---------------------------------------------------------------- dose ordering on tops accepted at all four targets (paired)
    dose = {}
    if all(f'rows__{k}' in Z.files for k in TSETS):
        common = Z[f'rows__{TSETS[0]}']
        for k in TSETS[1:]: common = np.intersect1d(common, Z[f'rows__{k}'])
        dose['n_common'] = int(len(common)); pos = {k: np.searchsorted(Z[f'rows__{k}'], common) for k in TSETS}
        if len(common) >= 5:
            bi = rng.integers(0, len(common), (B, len(common)))
            rxs = {k: ((ref(k, 'W', 'MLPx') - ref(k, 'N', 'MLPx'))[pos[k]], (ref(k, 'W', 'MLPxW') - ref(k, 'N', 'MLPxW'))[pos[k]]) for k in TSETS}
            dose['MLPx'] = {k: float(rxs[k][0].mean()) for k in TSETS}; dose['MLPxW'] = {k: float(rxs[k][1].mean()) for k in TSETS}
            for m in models:
                for p in ('st', 'rc'):
                    w = {k: WN[(k, m, p)][pos[k]] for k in TSETS}
                    d = dict(mean={k: float(w[k].mean()) for k in TSETS})
                    for a_, b_ in (('T60', 'T70'), ('T100', 'T90')):
                        z = w[a_] - w[b_]; d[f'{a_}_minus_{b_}'] = float(z.mean()); d[f'{a_}_minus_{b_}_ci95'] = bci(z, bi); d[f'p_{a_}_{b_}'] = tp(z, -1)
                        fam.append(dict(arm='dose', model=m, path=p, test=f'W-N at {a_} more negative than at {b_} (paired, one-sided)', effect=d[f'{a_}_minus_{b_}'], p=d[f'p_{a_}_{b_}']))
                    dose[f'{m}|{p}'] = d
            log(f"dose (n common {len(common)}): MLPx {dose['MLPx']} MLPxW {dose['MLPxW']}")
            for m in models:
                log(f'   {m:14s} ' + ' | '.join(f"{p} " + ' '.join(f"{k} {dose[f'{m}|{p}']['mean'][k]:+.3f}" for k in TSETS)
                                                + f" 60-70 {dose[f'{m}|{p}']['T60_minus_T70']:+.3f} 100-90 {dose[f'{m}|{p}']['T100_minus_T90']:+.3f}" for p in ('st', 'rc')))
    R['dose'] = dose
    adj = holm([t['p'] for t in fam])
    for t, h in zip(fam, adj): t['holm'] = float(h)
    R['tests'] = fam; R['holm_family'] = f'{len(fam)} tests (models x sets x builds: vs MLPx, eS50 McNemar; st-rc per model x set; dose); Holm over all'
    R['n_holm_lt_05'] = int(sum(h < 0.05 for h in adj))
    (RES / f'{TAG}.json').write_text(json.dumps(jsonable(R), indent=1)); log(f'wrote {TAG}.json; {len(fam)} tests, {R["n_holm_lt_05"]} Holm < 0.05')
    for t in fam:
        if t['model'] in ('jedi', 'jedi_redrawn'): log(f"  {t['arm']} {t['model']} {t['path']} {t['test']}: {t['effect']:+.3f} p {t['p']:.2g} holm {t['holm']:.2g}")
    figure(R)


def figure(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('jedi', 'JEDI-linear (official)'), ('jedi_redrawn', 'JEDI-linear, redrawn weights'), ('dsets', 'Deep Sets'),
              ('jedis', 'JEDI-style (small)'), ('tf', 'transformer 2b-d32'), ('pfnfrac', 'PFN, pT-fraction input'), ('pfnabs', 'PFN, absolute-pT input')]
    xs = [TARGET[k] for k in TSETS]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.4), sharey=True); axs = axs.ravel()
    for i, (m0, title) in enumerate(panels):
        ax = axs[i]
        for k in ('MLPx', 'MLPxW'):
            ys = [R['sets'][s]['contrasts']['W_vs_N'][k]['diff'] if s in R['sets'] and 'W_vs_N' in R['sets'][s]['contrasts'] else np.nan for s in TSETS]
            ax.plot(xs, ys, color='0.6' if k == 'MLPx' else 'k', lw=1.2, ls=':' if k == 'MLPx' else '-', label='MLPx (no pair mass)' if k == 'MLPx' else 'MLPxW (with pair mass)')
        for m in [m for m in R['models'] if m == m0 or m.startswith(m0 + '_s')]:
            for p, col, mk in (('rc', '#2a6fdb', 'o'), ('st', '#d9822b', 's')):
                v, lo, hi = [], [], []
                for s in TSETS:
                    M = R['sets'].get(s, {}).get('contrasts', {}).get('W_vs_N', {}).get(f'{m}|{p}')
                    v.append(M['diff'] if M else np.nan); lo.append(M['ci95'][0] if M else np.nan); hi.append(M['ci95'][1] if M else np.nan)
                v, lo, hi = map(np.asarray, (v, lo, hi)); off = (-0.8 if p == 'rc' else 0.8) + (0.4 if m.endswith('_s1') else 0)
                ax.errorbar(np.asarray(xs) + off, v, yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], color=col, marker=mk, ms=4, lw=1, capsize=2,
                            alpha=1.0 if not m.endswith('_s1') else 0.55,
                            label=(f'{"own axis (rc)" if p == "rc" else "stored axis (st)"}' if (i == 0 and not m.endswith('_s1')) else None))
        ax.axhline(0, color='k', lw=0.5); ax.axvline(80.4, color='0.85', lw=0.8); ax.set_title(title, fontsize=9); ax.set_xticks(xs)
        if i % 4 == 0: ax.set_ylabel('W − N change in top log-odds')
        if i >= 3: ax.set_xlabel('target W-pair mass (GeV)')
    h, l = axs[0].get_legend_handles_labels(); axs[-1].axis('off'); axs[-1].legend(h, l, fontsize=9, loc='center', frameon=False)
    fig.suptitle('hls4ml tops, W-candidate pair mass moved to 60-100 GeV at fixed jet mass and τ32: rebuilt about its own axis (blue), JEDI-linear responds\n'
                 'like the reference without the pair mass (MLPx, dotted), below the reference with it (MLPxW, solid); the stored-axis build (orange) flips the sign.\n'
                 'y: W edit minus non-W control N, change in top-vs-QCD log-odds; seed 1 faded; bars 95 % CI over jets', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91)); f = RES / f'{TAG}_wpair_rebuild.png'; fig.savefig(f, dpi=140); log(f'wrote {f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['edit', 'small', 'jedi', 'stats', 'figure']); a = ap.parse_args()
    log(f'H29 part {a.part}; smoke {SMOKE}')
    if a.part == 'figure': figure(json.loads((RES / f'{TAG}.json').read_text()))
    else: dict(edit=part_edit, small=part_small, jedi=part_jedi, stats=part_stats)[a.part]()
    log('done')
