"""E31 verification (fresh sample): re-measure the C2ST / look-real-subset numbers of interp.e31_onmanifold_jedi on a new jet sample,
and add three controls the E31 run did not have.

Network: official JEDI-linear 3-feature n64, exact forward pass (interp.jedi_core). Edit sample sample(1200, 4101) = 6,000 jets
(E31 used sample(1600, 3101)); edits, matched dilations and null edits built by interp.e31_onmanifold_jedi.build_edits with
rng 4104; the subset kept: shift, turn5, split13, soft, boost with their references and nulls.

1. Reproduction: C2ST AUC (same HistGBDT, two swapped half-assignments, 5-fold OOF) on obs / obs8 / plist; look-real band
   [q25, q75] of s0; R_h4, W->Z / Z->W, calls changed on the band subsets; McNemar edit vs null.
2. Axis-free feature set 'inv' = (m, jet pT, N, tau21, tau32, D2, pT^D): observables that do not depend on where the stored axis
   is (a rigid eta translation of all particles is a longitudinal boost, so every 'inv' feature of the shifted jet equals the
   unedited value up to tau-axis seeding).  'obs8' still contains girth = sum pT r / M0 measured about the stored axis.
   H0 for the shift: AUC_inv = 0.5.
3. Support (on-manifold) test that a pure distribution shift does not fail: features log|f| standardised on reference jets
   (unedited half A); for each query jet the mean distance to its 10 nearest reference jets; out-of-support = distance above the
   99th percentile of held-out unedited jets (half B).  Reported for each edit, its matched dilation and its null edit (half-B
   jets).  A real mass change (dilation) moves jets within the support; an off-manifold edit moves them outside.
4. Selection control for the band subsets: the headline number on the complement jets (ok, not in the band) reweighted to the
   band's composition in strata of (true class x quintile of unedited log M2 x tercile of unedited jet pT).  If the band value
   equals the reweighted-complement value, the change from all jets to the band comes from which jets the band selects
   (class, width, pT), not from the edited jets looking real.
5. Natural centroid offsets |M1|^2/(M0 M2) on a fresh pool sample(2000, 4102).
Writes $AI_RESULTS/jedi/e31_verify.json.  Run: ./job.sh e31-verify 'python -m interp.e31_verify_jedi'  (--smoke N)
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
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time(); G_, Q_, W_, Z_ = 0, 1, 2, 3
KEEP = ['shift', 'dil_M2', 'rot_shift', 'turn5', 'dil_m5', 'rot_turn5', 'split13', 'rot_split13', 'soft', 'soft_ctrl', 'dil5',
        'boost', 'dil_B', 'rot_boost']
PLAN = dict(shift=('dil_M2', 'rot_shift', 'wz'), turn5=('dil_m5', 'rot_turn5', 'zw'), split13=('dil5', 'rot_split13', 'ch'),
            soft=('dil5', 'soft_ctrl', 'ch'), boost=('dil_B', 'rot_boost', 'ch'))
INV = [0, 1, 2, 3, 4, 5, 7]   # indices into OBS: m, jpt, N, tau21, tau32, D2, ptD


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


def knn_support(Fref, Fq_list, Fhold, k=10):
    from sklearn.neighbors import NearestNeighbors
    tr = lambda F: np.log(np.abs(F) + 1e-9)
    R = tr(Fref); mu, sd = R.mean(0), R.std(0) + 1e-9; z = lambda F: (tr(F) - mu) / sd
    nn = NearestNeighbors(n_neighbors=k).fit(z(Fref))
    dist = lambda F: nn.kneighbors(z(F))[0].mean(1)
    dh = dist(Fhold); thr = float(np.quantile(dh, 0.99))
    return thr, dh, [dist(F) for F in Fq_list]


def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.x10_das_jedi import clean, RidgeR, mass
    from interp.e28w_split_route_jedi import mcnemar
    from interp.e31_onmanifold_jedi import build_edits, featsets, c2st, dauc, OBS
    sm = a.smoke; pc = sm or 1200; NB = 200 if sm else 300; NP = 200 if sm else 300; folds = 5
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / ('e31_verify' + ('_smoke' if sm else '') + '.json')
    R = dict(sample=f'sample(per_class={pc}, seed=4101)', plan=PLAN)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    net = Jedi(64)
    fwd = lambda xx: net.run(xx, keep=('logits', 'h4'), batch=1000)
    x, y, _ = sample(pc, 4101); J = len(y)
    B0 = fwd(x); c0 = B0['logits'].argmax(1); R['N'] = J; R['acc'] = float((c0 == y).mean())
    R['frac_N64'] = float(((x[:, :, 0] > 0).sum(1) == 64).mean())
    log(f'N {J} acc {R["acc"]:.4f}; share of jets with 64 particles {R["frac_N64"]:.3f}')
    logm0 = np.log(np.maximum(mass(x), 1e-3)); pr = np.random.default_rng(4103).permutation(J); hA = np.sort(pr[:J // 2])
    nf = 2 * len(hA) // 3; RO = RidgeR(B0['h4'], logm0, hA[:nf], hA[nf:]); ro0 = RO(B0['h4'])
    E, INFO = build_edits(x, np.random.default_rng(4104), log); E = {k: E[k] for k in KEEP}; R['edit_info'] = INFO
    RESP = {}
    for k, (xe, ok) in E.items():
        P = fwd(xe); ce = P['logits'].argmax(1)
        RESP[k] = dict(call=ce, ch=ce != c0, wz=(c0 == W_) & (ce == Z_), zw=(c0 == Z_) & (ce == W_), dro=RO(P['h4']) - ro0)
    log('forward passes done'); save()

    F0, Fu = featsets(x); F0['inv'] = F0['obs'][:, INV]; FE = {}
    for k, (xe, ok) in E.items():
        f = featsets(xe)[0]; f['inv'] = f['obs'][:, INV]; FE[k] = f
    log('features done')
    R['inv_features'] = [OBS[i] for i in INV]
    # max |change| of each inv feature under the shift (should be ~0 up to tau seeding)
    R['shift_inv_feature_max_abs_rel_change'] = {OBS[i]: float(np.quantile(np.abs(FE['shift']['obs'][:, i] - F0['obs'][:, i]) /
                                                                          np.maximum(np.abs(F0['obs'][:, i]), 1e-9), 0.99)) for i in INV + [6, 8]}
    R['c2st'] = {}; SC = {}
    for fs in ('obs', 'obs8', 'plist', 'inv'):
        R['c2st'][fs] = {'_retrain_null': c2st(F0[fs], F0[fs], np.ones(J, bool), 4200, folds, 100, 100)[0]['auc']}
        for i, (k, (xe, ok)) in enumerate(E.items()):
            res, sE, s0 = c2st(F0[fs], FE[k][fs], ok, 4110 + i, folds, NB, NP); R['c2st'][fs][k] = res; SC[(fs, k)] = (sE, s0)
            log(f'C2ST {fs} {k}: AUC {res["auc"]:.4f} {res["ci"]} p {res["p_swap"]:.4f}')
        R['c2st'][fs]['_delta'] = {k: dict(minus_ref=dauc(SC[(fs, k)], SC[(fs, ref)], NB, 1), minus_null=dauc(SC[(fs, k)], SC[(fs, nul)], NB, 2))
                                   for k, (ref, nul, _) in PLAN.items()}
        save()

    # ---- support test (kNN) on half-B jets
    R['support'] = {}
    isA = np.zeros(J, bool); isA[hA] = True
    for fs in ('obs', 'obs8', 'inv'):
        names = list(E.keys()); thr, dh, dq = knn_support(F0[fs][isA], [FE[k][fs][~isA & E[k][1]] for k in names], F0[fs][~isA])
        o = dict(threshold_q99_unedited=thr, unedited_out_frac=float((dh > thr).mean()))
        rb = np.random.default_rng(4120)
        for k, d in zip(names, dq):
            out = (d > thr).astype(float)
            o[k] = dict(n=int(len(d)), out_frac=float(out.mean()), out_ci=qci([out[rb.integers(0, len(out), len(out))].mean() for _ in range(300)]),
                        median_dist_over_thr=float(np.median(d) / thr))
        R['support'][fs] = o
        log(f'support {fs}: ' + ', '.join(f'{k} {o[k]["out_frac"]:.3f}' for k in names))
    save()

    # ---- bands, headlines, selection control
    Wb = np.random.default_rng(4105).poisson(1.0, (NB, J)).astype(np.float32)
    bm = lambda w, v: (w @ v) / np.maximum(w.sum(1), 1e-12)
    logM2 = np.log(np.maximum(Fu['M2'], 1e-30)); jpt = Fu['jpt']
    q5 = np.searchsorted(np.quantile(logM2, [.2, .4, .6, .8]), logM2); t3 = np.searchsorted(np.quantile(jpt, [1 / 3, 2 / 3]), jpt)
    stratum = y * 15 + q5 * 3 + t3

    def metric(k, K, w=None):
        ref, nul, hk = PLAN[k]; e, d, n = RESP[k], RESP[ref], RESP[nul]
        sel = {'wz': K & (c0 == W_), 'zw': K & (c0 == Z_), 'ch': K}[hk]
        ww = np.ones(J) if w is None else w
        o = dict(n=int(K.sum()), n_by_class=np.bincount(y[K], minlength=5).tolist())
        num, den = (ww[K] * e['dro'][K]).sum(), (ww[K] * d['dro'][K]).sum()
        o['R_h4'] = float(num / den) if K.sum() >= 5 and abs(den) > 1e-12 else None
        o['headline_edit'] = float((ww[sel] * e[hk][sel]).sum() / max(ww[sel].sum(), 1e-12)) if sel.any() else None
        o['headline_null'] = float((ww[sel] * n[hk][sel]).sum() / max(ww[sel].sum(), 1e-12)) if sel.any() else None
        o['headline_ref'] = float((ww[sel] * d[hk][sel]).sum() / max(ww[sel].sum(), 1e-12)) if sel.any() else None
        if w is None and K.sum() >= 5:
            wb = Wb[:, K]; o['R_h4_ci'] = qci(bm(wb, e['dro'][K]) / bm(wb, d['dro'][K]))
            if sel.any(): o['headline_edit_ci'] = qci(bm(Wb[:, sel], e[hk][sel].astype(np.float32)))
            o['mcnemar_vs_null'] = mcnemar(e[hk][sel], n[hk][sel])
        return o

    R['bands'] = {}
    for k in E:
        ok = E[k][1]; R['bands'][k] = {}
        for fs in ('obs', 'obs8', 'plist', 'inv'):
            sE, s0 = SC[(fs, k)]; sc = np.isfinite(sE) & np.isfinite(s0); lo, hi = np.quantile(s0[sc], [.25, .75])
            K = ok & sc & (sE >= lo) & (sE <= hi); R['bands'][k][f'kept_frac_{fs}'] = float(K.sum() / max(sc.sum(), 1))
            if k in PLAN:
                o = metric(k, K)
                C = ok & sc & ~K
                if K.sum() >= 20 and C.sum() >= 20:
                    nk = np.bincount(stratum[K], minlength=75).astype(float); nc = np.bincount(stratum[C], minlength=75).astype(float)
                    cov = float(nk[nc > 0].sum() / nk.sum()); w = np.where(nc[stratum] > 0, nk[stratum] / np.maximum(nc[stratum], 1), 0.0)
                    o['complement_reweighted'] = metric(k, C, w); o['complement_reweighted']['stratum_coverage'] = cov
                    o['complement_unweighted'] = metric(k, C, np.ones(J))
                R['bands'][k][fs] = o
        if k in PLAN:
            R['bands'][k]['all'] = metric(k, ok)
            log(f'{k}: kept ' + ', '.join(f'{fs} {R["bands"][k][f"kept_frac_{fs}"]:.3f}' for fs in ('obs', 'obs8', 'plist', 'inv')))
            for fs in ('all', 'obs', 'obs8', 'plist', 'inv'):
                o = R['bands'][k][fs]; cr = o.get('complement_reweighted', {})
                log(f'   {fs}: n {o["n"]} R {o["R_h4"]} {o.get("R_h4_ci")} {PLAN[k][2]} {o["headline_edit"]} null {o["headline_null"]} '
                    f'| complement reweighted R {cr.get("R_h4")} {PLAN[k][2]} {cr.get("headline_edit")} (cov {cr.get("stratum_coverage")})')
        else:
            log(f'{k}: kept ' + ', '.join(f'{fs} {R["bands"][k][f"kept_frac_{fs}"]:.3f}' for fs in ('obs', 'obs8', 'plist', 'inv')))
        save()

    # ---- natural offsets on a fresh pool
    xp, _, _ = sample(sm or 2000, 4102); xp = np.asarray(xp, np.float64); pt = np.where(xp[..., 0] > 0, xp[..., 0], 0)
    M0 = pt.sum(1); M2 = (pt * (xp[..., 1] ** 2 + xp[..., 2] ** 2)).sum(1); M1 = np.hypot((pt * xp[..., 1]).sum(1), (pt * xp[..., 2]).sum(1))
    off = M1 ** 2 / np.maximum(M0 * M2, 1e-30); dm = -0.5 * np.log(np.maximum(1 - off, 1e-12))
    R['natural'] = dict(N=int(len(xp)), off_q50_q99_max=np.quantile(off, [.5, .99, 1.]).tolist(), dlogm_q99_max=np.quantile(dm, [.99, 1.]).tolist(),
                        shift_off_median=float(np.median(FE['shift']['obs'][:, 8])), turn5_off_median=float(np.median(FE['turn5']['obs'][E['turn5'][1], 8])),
                        frac_shift_off_above_natural_max=float((FE['shift']['obs'][:, 8] > off.max()).mean()),
                        frac_turn5_off_above_natural_max=float((FE['turn5']['obs'][E['turn5'][1], 8] > off.max()).mean()))
    log(f'natural: {R["natural"]}')
    R['seconds'] = time.time() - T0; save(); log(f'done, wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0)
    main(ap.parse_args())
