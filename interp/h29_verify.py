"""H29 verify (loop D): rerun of the H29 W-pair edit on a fresh hls4ml sample, plus a reference-spread control.

Parts:
  --part edit   (conda) fresh sample (default_rng(2911), 2,000/class, rows not used by H3 / H3v / H23 / H29), H29's own
                part_edit with TAG h29_verify (H4-type T60/T70/T90/T100/Q80 and H3v sets; st and rc builds; refs rng 2902)
  --part jedi   (jax env) JEDI-linear official + redrawn on those inputs
  --part stats  (conda) H29's statistics on the fresh sample (small models not rerun) -> h29_verify.json, figure
  --part refens (conda) reference spread: MLPx / MLPxW retrained 5x (sklearn MLP, random_state i, a different 50k
                train-archive subset each) and a gradient-boosted pair (HGB-x, HGB-xW). W - N of each reference on the H29
                sample and on the fresh sample; JEDI-linear rc W - N ratio to every member; JEDI-linear rc minus each
                MLPx member with bootstrap CI.                                                 -> h29_verify_refens.json
The H29 code is imported unchanged; only SEED, TAG and the sample exclusion are replaced.
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
from interp import h29_jedi_wpair_rebuild as M  # noqa: E402

M.SEED = 2911
M.TAG = 'h29_verify'
RES = M.RES
log = M.log


def draw_sample():
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y = np.asarray(d['y']).astype(int)
    r = np.random.default_rng(501); h3 = np.concatenate([r.choice(np.flatnonzero(y == c), 4000, replace=False) for c in range(5)])
    used = np.zeros(len(y), bool); used[h3] = True
    r = np.random.default_rng(601); h3v = np.concatenate([r.choice(np.flatnonzero((y == c) & ~used), 2000, replace=False) for c in range(5)])
    r = np.random.default_rng(9523); h23 = np.concatenate([r.choice(np.flatnonzero(y == c), 1000, replace=False) for c in range(5)])
    excl = np.zeros(len(y), bool); excl[h3] = excl[h3v] = excl[h23] = True
    r = np.random.default_rng(2901); h29 = np.concatenate([r.choice(np.flatnonzero((y == c) & ~excl), 2000, replace=False) for c in range(5)])
    excl[h29] = True
    rng = np.random.default_rng(M.SEED)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & ~excl), M.PER, replace=False) for c in range(5)]))
    x = np.asarray(d['x'][idx], np.float32); yy = y[idx]
    info = dict(file='data/n64/full_test.npz', per_class=M.PER, seed=M.SEED, N=int(len(idx)), excluded=int(excl.sum()),
                overlap_with_excluded=int(excl[idx].sum()), overlap_with_h29=int(np.isin(idx, h29).sum()),
                excluded_draws='H3 rng 501, H3v rng 601, H23 rng 9523, H29 rng 2901')
    log(f'sample {info}')
    return x, yy, idx, info


M.draw_sample = draw_sample


# ================================================================================================== reference spread
def refens():
    import h5py
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    from interp.h4_top_wmass import _auc
    V = M.V
    # pool of 75k t + 75k q/g train-archive jets (dev excluded); every member uses its own random 50k subset
    Xa = np.load(V / 'data' / 'train_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(V / 'data' / 'train_meta.h5', 'r') as f: ya = f['label'][:].astype(np.int64)
    dev = np.asarray(np.load(V / 'data' / 'splits.npz')['dev'], np.int64); isdev = np.zeros(len(ya), bool); isdev[dev] = True
    r = np.random.default_rng(2919)
    it = np.sort(np.r_[r.choice(np.flatnonzero(~isdev & (ya == 4)), 75000, replace=False), r.choice(np.flatnonzero(~isdev & np.isin(ya, [0, 1])), 75000, replace=False)])
    t0 = time.time(); xr = np.asarray(Xa[it, :64], np.float32); yr = (ya[it] == 4).astype(int); log(f'pool {len(it)} read in {time.time() - t0:.0f}s')
    nr = (xr[..., 0] > 0).sum(1).astype(np.int64)
    Pr = H._identity(M.p4_of(xr[..., 0], xr[..., 1], xr[..., 2]), nr).astype(np.float32); del xr
    t0 = time.time(); Xf, Of = H._features(Pr, nr); del Pr; log(f'pool features in {time.time() - t0:.0f}s')
    REFS = {}
    for i in range(5):
        sub = np.sort(np.random.default_rng(3000 + i).choice(len(it), 50000, replace=False))
        for nm, w in (('MLPx', False), ('MLPxW', True)):
            F = X.xsel(Xf[sub], {k: v[sub] for k, v in Of.items()}, w); sc = StandardScaler().fit(F)
            m = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=i, max_iter=200).fit(sc.transform(F), yr[sub])
            REFS[f'{nm}_{i}'] = ('mlp', m, sc, w); log(f'{nm}_{i}: {m.n_iter_} it, val {m.best_validation_score_:.4f}')
    sub = np.sort(np.random.default_rng(3100).choice(len(it), 50000, replace=False))
    for nm, w in (('HGBx', False), ('HGBxW', True)):
        F = X.xsel(Xf[sub], {k: v[sub] for k, v in Of.items()}, w)
        m = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=0).fit(F, yr[sub]); REFS[nm] = ('hgb', m, None, w)
        log(f'{nm}: {m.n_iter_} it')

    def ev(Xe, Oe):
        out = {}
        for k, (kind, m, sc, w) in REFS.items():
            F = X.xsel(Xe, Oe, w); F = sc.transform(F) if sc is not None else F
            p = np.clip(m.predict_proba(F)[:, 1], 1e-7, 1 - 1e-7); out[k] = np.log(p / (1 - p))
        return out

    rng = np.random.default_rng(2929); OUT = {}
    for tag in ('h29', 'h29_verify'):
        f = RES / f'{tag}_inputs.npz'; fs = RES / f'{tag}_scores_jedi.npz'
        if not f.exists() or not fs.exists(): log(f'missing {f} or {fs}'); continue
        Z = np.load(f); S = np.load(fs); y5 = Z['y5']; tq = np.isin(y5, [0, 1, 4]); yb = (y5[tq] == 4).astype(int)
        xu = Z['U__st'][tq]; nu = (xu[..., 0] > 0).sum(1).astype(np.int64)
        Pu = H._identity(M.p4_of(xu[..., 0], xu[..., 1], xu[..., 2]), nu).astype(np.float32); su = ev(*H._features(Pu, nu))
        R = dict(unedited_AUC={k: float(_auc(v, yb)) for k, v in su.items()}, sets={})
        log(f'{tag} unedited AUC: ' + ' '.join(f'{k} {v:.4f}' for k, v in R['unedited_AUC'].items()))
        for key in M.H4SETS + ('H3v_60',):
            if f'X__{key}__id__st' not in Z.files: continue
            a, b = ('W', 'N') if key in M.H4SETS else ('E', 'B'); sc_ = {}
            for v in ('id', a, b):
                xx = Z[f'X__{key}__{v}__st']; nn = (xx[..., 0] > 0).sum(1).astype(np.int64)
                P = H._identity(M.p4_of(xx[..., 0], xx[..., 1], xx[..., 2]), nn).astype(np.float32); sc_[v] = ev(*H._features(P, nn))
            rx = {k: sc_[a][k] - sc_[b][k] for k in REFS}
            js = {p: M.score_of(S[f'S__jedi__X__{key}__{a}__{p}']) - M.score_of(S[f'S__jedi__X__{key}__{b}__{p}']) for p in ('rc', 'st')}
            n = len(js['rc']); bi = rng.integers(0, n, (2000, n))
            C = dict(n=n, jedi_rc=float(js['rc'].mean()), jedi_rc_ci95=M.bci(js['rc'], bi), jedi_st=float(js['st'].mean()), refs={})
            for k in REFS:
                d = dict(diff=float(rx[k].mean()), ci95=M.bci(rx[k], bi))
                d['jedi_rc_ratio'], d['jedi_rc_ratio_ci95'] = M.ratio_ci(js['rc'], rx[k], bi)
                z = js['rc'] - rx[k]; d['jedi_rc_minus'] = float(z.mean()); d['jedi_rc_minus_ci95'] = M.bci(z, bi)
                d['p_two_sided_t'] = M.tp(z, 0)
                C['refs'][k] = d
            mx = np.array([C['refs'][f'MLPx_{i}']['diff'] for i in range(5)]); mw = np.array([C['refs'][f'MLPxW_{i}']['diff'] for i in range(5)])
            C['MLPx_members'] = dict(mean=float(mx.mean()), sd=float(mx.std(ddof=1)), min=float(mx.min()), max=float(mx.max()))
            C['MLPxW_members'] = dict(mean=float(mw.mean()), sd=float(mw.std(ddof=1)), min=float(mw.min()), max=float(mw.max()))
            C['jedi_rc_ratio_to_MLPx_members'] = [float(C['jedi_rc'] / v) for v in mx]
            C['jedi_rc_ratio_to_MLPxW_members'] = [float(C['jedi_rc'] / v) for v in mw]
            R['sets'][key] = C
            log(f'{tag} {key} n {n}: jedi rc {C["jedi_rc"]:+.3f} {np.round(C["jedi_rc_ci95"], 3)} | MLPx members {np.round(mx, 3)} | '
                f'MLPxW members {np.round(mw, 3)} | HGBx {C["refs"]["HGBx"]["diff"]:+.3f} HGBxW {C["refs"]["HGBxW"]["diff"]:+.3f} | '
                f'ratio to MLPx members {np.round(C["jedi_rc_ratio_to_MLPx_members"], 2)} to MLPxW {np.round(C["jedi_rc_ratio_to_MLPxW_members"], 2)}')
        OUT[tag] = R
    (RES / 'h29_verify_refens.json').write_text(json.dumps(M.jsonable(OUT), indent=1)); log('wrote h29_verify_refens.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['edit', 'jedi', 'stats', 'refens']); a = ap.parse_args()
    log(f'H29 verify part {a.part}; seed {M.SEED}; tag {M.TAG}')
    dict(edit=M.part_edit, jedi=M.part_jedi, stats=M.part_stats, refens=refens)[a.part]()
    log('done')
