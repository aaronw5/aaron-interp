"""E32 (claims A4, A4o, D4; on-manifold axis test, open direction from round 14): JEDI-linear on realistic jet-axis errors.

Question.  E31/E31v showed that the rigid 0.06 rad shift moves jets off the real-jet manifold (real jets never have their pT
centroid that far from the stored axis).  What does JEDI-linear do when the axis error is one that a trigger would actually make?

Network: official JEDI-linear 3-feature n64 (inputs pT, deta, dphi of up to 64 particles about the stored jet axis), exact forward
pass (interp.jedi_core).  Notation per jet (sums over real particles, r_i = sqrt(deta_i^2 + dphi_i^2) about the stored axis):
  M0 = sum pT, M1 = sum pT (deta, dphi), M2 = sum pT r^2 (un-recentred: about the stored axis, not about the centroid).
Samples: --part main / c2st34: interp.jedi_core.sample(5000, 3201), N = 25,000 (5 classes x 5,000).
         --part rep: sample(5000, 3202) minus jets also in sample 3201 (byte-identical particle arrays).

Edits (each re-expresses the same physical particles about a different axis; every particle keeps its pT and slot, the jet four-
momentum is unchanged; only (deta, dphi) change: deta -> deta - c_eta, dphi -> wrap(dphi - c_phi); c = per-jet axis offset):
  T0  E-scheme axis of the 64 stored particles (massless four-vectors summed; axis = rapidity and azimuth of the sum).  Reference:
      the gap between the stored axis (computed from all constituents) and the axis recomputed from the 64 kept particles.
  T1  E-scheme axis from particles quantised to calorimeter towers of 0.0873 x 0.0873 in (eta, phi) (L1 trigger tower size): each
      particle is moved to the centre of its tower and summed as a massless vector of the same pT (ET-scheme tower deposit).  The
      absolute jet position is not stored, so the tower grid phase is drawn uniformly per jet in [0, 0.0873) in eta and phi.
  T2  the same with towers of 0.0436 x 0.0436.
  T3  hardest-particle axis (L1 seed): the axis is placed on the highest-pT particle.
  T4  hardest-particle seed plus one E-scheme recentring step: E-scheme axis of the particles within R = 0.4 of the seed.
Predicted change of the network's mass variable (thesis: mass variable = un-recentred M2 about the stored axis, times M0):
  dlog = Delta log(M0 M2) = log(M2'/M2), M2' = M2 about the new axis (= M2 - 2 c.M1 + M0 |c|^2; M0 unchanged).
Controls (same jets):
  C1  matched dilation: every real particle's (deta, dphi) x a, a = sqrt(M2'/M2) per jet (a real-looking mass change of the same
      predicted Delta log(M0 M2)); the denominator of R.
  C2  rigid translation by the same |c| in a uniformly random azimuth (tests whether the direction of the seeded offset matters).
  C3  rotation of the whole jet about the stored axis by a uniform random angle (no change of M0, M1 magnitude, M2; null for calls).
  C4  untrained JEDI-linear (jedi_core untrained=True, redraw seeds 1, 2; --part c2st34): R of the untrained net.
Metrics (per T):
  calls changed %, W->Z = share of W-called jets called Z after the edit (Z->W likewise), with counts; McNemar (exact, paired)
  T vs C3 and T vs C1 on the per-jet indicator 'call changed'.  Share of W->Z flows with dlog > 0 and of Z->W flows with dlog < 0
  (sign-matched to Delta M2: a larger M2 should push W toward Z, Z (91 GeV) is heavier than W (80 GeV)).
  R_margin = sum_j e_j d_j / sum_j d_j^2 over jets called W or Z, e_j = change of the W - Z logit margin under T, d_j = change under
  C1 (projection of the edit response on the matched-dilation response; R = 1 when the network responds to T exactly as to a real
  mass change of the same predicted size).  R_h4 = the same with the ridge log-m read-out of h4 (fit on unedited jets, as E31) over
  all jets, plus its ratio-of-sums version.  95 % CIs: 500 Poisson jet-bootstrap weights; H0 R = 1: two-sided bootstrap p.
  R for C2 and C3 (against C1) gives the null of the estimator.  R in terciles (and quintiles for the figure) of |c|.
Freezes under T1 (where the T1 call change passes): unit values held at their unedited per-jet values (run_jetwise):
  g (all 64 context units), S0+S29 (S0 = sum pT of the 14 hardest via g; S29 ~ sum pT r^2 + sum r^2 + soft count), h45_top (the 4 h4
  and 4 h5 live units with the largest standardised response to a 5 % M2 dilation, ranked on half A of the sample), h45_E6 (E6's
  mass units h4 18, h5 0, h5 28).  Metric: share of T1 call changes removed, on the evaluation set (T1-changed jets + random jets).
  Null: 200 (main) / 100 (rep) random holds of the same number of live units (S0+S29: live S units; h4/h5 sets: live h4 u h5 units;
  g: live S u h4 u h5 units, no size-matched set exists inside g); rank p; Holm over the 4 holds.
On-manifold check (--part main for T0, T1, T2, C1(T1), C3; --part c2st34 for T3, T4, C1(T3), C1(T4)) on 10,000 random jets:
  HistGBDT C2ST of edited vs exact-axis jets (interp.e31_onmanifold_jedi.c2st: two swapped halves x 5-fold OOF, AUC, 300 bootstrap,
  300 swap permutations) on feature sets inv (axis-free: m, jet pT, N, tau21, tau32, D2, pT^D), obs8 (inv + girth about the axis),
  obs (obs8 + centroid term |M1|^2/(M0 M2)), plist (30 hardest x (pT, deta, dphi)); kNN support share (E31v: mean distance to the 10
  nearest unedited jets of half A in standardised log-feature space, outside = above the 99th percentile of unedited half-B jets).
Writes $AI_RESULTS/jedi/e32_<part>.json.
  ./job.sh e32-t12 'python -m interp.e32_seeded_axis_jedi --part main'
  ./job.sh e32-t34 'python -m interp.e32_seeded_axis_jedi --part c2st34'
  ./job.sh e32-rep 'python -m interp.e32_seeded_axis_jedi --part rep'          (--smoke N: N jets per class)
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
T0_ = time.time(); G_, Q_, W_, Z_, T_ = 0, 1, 2, 3, 4
TOWER = dict(T1=0.0873, T2=0.0436)
TS = ('T0', 'T1', 'T2', 'T3', 'T4')
INV = [0, 1, 2, 3, 4, 5, 7]          # OBS indices: m, jpt, N, tau21, tau32, D2, ptD
OBS8 = [0, 1, 2, 3, 4, 5, 6, 7]


def log(*a): print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


def reaxis(x, ae, ap):
    """real particles' (deta, dphi) about the axis (ae, ap) given in stored relative coordinates; padding and pT untouched"""
    real = x[..., 0] > 0; e = np.array(x, np.float32)
    de = x[..., 1].astype(np.float64) - ae[:, None]; dp = wrap(x[..., 2].astype(np.float64) - ap[:, None])
    e[..., 1] = np.where(real, de, x[..., 1]); e[..., 2] = np.where(real, dp, x[..., 2])
    return e


def dilate(x, a):
    e = np.array(x, np.float64); rl = e[..., 0] > 0; a = np.asarray(a, np.float64)[:, None]
    for j in (1, 2): e[..., j] = np.where(rl, e[..., j] * a, e[..., j])
    return e.astype(np.float32)


def rotate(x, th):
    e = np.array(x, np.float64); rl = e[..., 0] > 0; c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    de, dp = e[..., 1].copy(), e[..., 2].copy()
    e[..., 1] = np.where(rl, c * de - s * dp, de); e[..., 2] = np.where(rl, s * de + c * dp, dp)
    return e.astype(np.float32)


def m2(x):
    x = np.asarray(x, np.float64); pt = np.where(x[..., 0] > 0, x[..., 0], 0.0)
    return (pt * (x[..., 1] ** 2 + x[..., 2] ** 2)).sum(1)


def escheme(w, e, p):
    """rapidity and azimuth of the summed massless four-vectors of weight (pT) w at (e, p)"""
    E = (w * np.cosh(e)).sum(1); pz = (w * np.sinh(e)).sum(1); px = (w * np.cos(p)).sum(1); py = (w * np.sin(p)).sum(1)
    return 0.5 * np.log((E + pz) / np.maximum(E - pz, 1e-30)), np.arctan2(py, px)


def axes(x, rng):
    x64 = np.asarray(x, np.float64); real = x64[..., 0] > 0; pt = np.where(real, x64[..., 0], 0.0)
    de = np.where(real, x64[..., 1], 0.0); dp = np.where(real, x64[..., 2], 0.0); J = len(x); rr = np.arange(J)
    A = dict(T0=escheme(pt, de, dp)); info = {}
    for k, t in TOWER.items():
        oe, op = rng.uniform(0, t, J)[:, None], rng.uniform(0, t, J)[:, None]
        qe = (np.floor((de - oe) / t) + 0.5) * t + oe; qp = (np.floor((dp - op) / t) + 0.5) * t + op
        A[k] = escheme(pt, qe, qp)
        info[k] = dict(tower=t, max_particle_quantisation_shift=float(np.max(np.abs(np.where(real, qe - de, 0)))),
                       towers_per_jet_median=float(np.median([len(set(zip(np.floor((de[j, real[j]] - oe[j]) / t).astype(int),
                                                                          np.floor((dp[j, real[j]] - op[j]) / t).astype(int))))
                                                              for j in range(min(J, 2000))])))
    k = pt.argmax(1); se, sp = de[rr, k], dp[rr, k]; A['T3'] = (se, sp)
    near = real & (np.hypot(de - se[:, None], wrap(dp - sp[:, None])) < 0.4)
    A['T4'] = escheme(pt * near, de, dp); info['T4_frac_pt_within_0p4'] = float(np.median((pt * near).sum(1) / pt.sum(1)))
    return A, info


def build(x, rng):
    """per T: dict(x=edited, c=(ce, cp), dlog, C1, C2, C3)"""
    A, info = axes(x, rng); J = len(x); M2 = m2(x); E = {}
    for t in TS:
        ce, cp = A[t]; xt = reaxis(x, ce, cp); M2t = m2(xt); a = np.sqrt(M2t / M2)
        psi = rng.uniform(0, 2 * np.pi, J); c = np.hypot(ce, cp)
        E[t] = dict(x=xt, ce=ce, cp=cp, c=c, dlog=np.log(M2t / M2), C1=dilate(x, a), C2=reaxis(x, c * np.cos(psi), c * np.sin(psi)),
                    C3=rotate(x, rng.uniform(0, 2 * np.pi, J)))
        E[t]['C1_dlog_check'] = float(np.max(np.abs(np.log(m2(E[t]['C1']) / M2) - E[t]['dlog'])))
    return E, info


def mcn(a, b):
    from scipy.stats import binomtest
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(n_first_only=n10, n_second_only=n01, p=float(binomtest(n10, n10 + n01).pvalue) if n10 + n01 else 1.0)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def boot_p1(Rb):
    Rb = np.asarray(Rb); Rb = Rb[np.isfinite(Rb)]
    if not len(Rb): return None
    return float(min(1.0, 2 * max(min((Rb <= 1).mean(), (Rb >= 1).mean()), 1 / len(Rb))))


def proj(Wb, e, d, sel):
    """R = sum e d / sum d^2 over sel; returns value, CI, bootstrap p for H0 R = 1"""
    if sel.sum() < 5: return dict(R=None, n=int(sel.sum()))
    ed, dd = (e * d)[sel], (d * d)[sel]; den = dd.sum()
    if den <= 1e-15: return dict(R=None, n=int(sel.sum()), note='zero dilation response')
    wb = Wb[:, sel]; Rb = (wb @ ed) / np.maximum(wb @ dd, 1e-30)
    return dict(R=float(ed.sum() / den), ci=qci(Rb), p_R_eq_1=boot_p1(Rb), n=int(sel.sum()))


def ratio(Wb, e, d, sel):
    if sel.sum() < 5 or abs(d[sel].sum()) < 1e-15: return dict(R=None)
    wb = Wb[:, sel]; Rb = (wb @ e[sel]) / (wb @ d[sel])
    return dict(R=float(e[sel].sum() / d[sel].sum()), ci=qci(Rb), p_R_eq_1=boot_p1(Rb))


def metrics(t, Et, RS, c0, y, Wb, rng):
    """RS[name] = dict(call, margin, ro) for name in T, C1, C2, C3"""
    m0 = RS['base']; o = {}; wz = (c0 == W_) | (c0 == Z_); c = Et['c']; dlog = Et['dlog']
    o['offset_mrad'] = dict(q=dict(zip(['q10', 'q25', 'q50', 'q75', 'q90', 'q99'], (1e3 * np.quantile(c, [.1, .25, .5, .75, .9, .99])).tolist())),
                            mean=float(1e3 * c.mean()), frac_below_7mrad=float((c < 0.007).mean()),
                            median_by_true_class=[float(1e3 * np.median(c[y == k])) for k in range(5)])
    o['dlog_M0M2'] = dict(q=dict(zip(['q10', 'q50', 'q90'], np.quantile(dlog, [.1, .5, .9]).tolist())), mean=float(dlog.mean()),
                          frac_positive=float((dlog > 0).mean()), C1_match_max_abs=Et['C1_dlog_check'])
    o['by_edit'] = {}
    for k in ('T', 'C1', 'C2', 'C3'):
        r = RS[k]; ch = r['call'] != c0; wz_ = (c0 == W_) & (r['call'] == Z_); zw = (c0 == Z_) & (r['call'] == W_)
        nW, nZ = int((c0 == W_).sum()), int((c0 == Z_).sum())
        oo = dict(calls_changed=float(ch.mean()), calls_changed_ci=qci((Wb @ ch.astype(np.float32)) / Wb.sum(1)), n_changed=int(ch.sum()),
                  W_to_Z=float(wz_.sum() / nW), n_W_to_Z=int(wz_.sum()), Z_to_W=float(zw.sum() / nZ), n_Z_to_W=int(zw.sum()),
                  WZ_flow_dlog_pos=float((dlog[wz_] > 0).mean()) if wz_.any() else None,
                  ZW_flow_dlog_neg=float((dlog[zw] < 0).mean()) if zw.any() else None,
                  flow_table=np.array([[int(((c0 == i) & (r['call'] == j)).sum()) for j in range(5)] for i in range(5)]).tolist(),
                  mean_dmargin_W_called=float((r['margin'] - m0['margin'])[c0 == W_].mean()),
                  mean_dro_h4=float((r['ro'] - m0['ro']).mean()))
        if k != 'C1':
            e = r['margin'] - m0['margin']; d = RS['C1']['margin'] - m0['margin']
            eh = r['ro'] - m0['ro']; dh = RS['C1']['ro'] - m0['ro']
            oo['R_margin'] = proj(Wb, e, d, wz); oo['R_h4'] = proj(Wb, eh, dh, np.ones(len(c0), bool))
            oo['R_h4_ratio_of_sums'] = ratio(Wb, eh, dh, np.ones(len(c0), bool))
            oo['R_margin_true_W'] = proj(Wb, e, d, y == W_); oo['R_margin_true_Z'] = proj(Wb, e, d, y == Z_)
        if k != 'C3':
            oo['mcnemar_vs_C3'] = mcn(ch, RS['C3']['call'] != c0)
        if k == 'T':
            oo['mcnemar_vs_C1'] = mcn(ch, RS['C1']['call'] != c0); oo['mcnemar_vs_C2'] = mcn(ch, RS['C2']['call'] != c0)
            oo['mcnemar_WZ_vs_C1'] = mcn(wz_, (c0 == W_) & (RS['C1']['call'] == Z_))
            oo['mcnemar_ZW_vs_C1'] = mcn(zw, (c0 == Z_) & (RS['C1']['call'] == W_))
        o['by_edit'][k] = oo
    # R by |c| bins
    e = RS['T']['margin'] - m0['margin']; d = RS['C1']['margin'] - m0['margin']; eh = RS['T']['ro'] - m0['ro']; dh = RS['C1']['ro'] - m0['ro']
    en = RS['C3']['margin'] - m0['margin']
    for nb, key in ((3, 'by_c_tercile'), (5, 'by_c_quintile')):
        qs = np.quantile(c, np.linspace(0, 1, nb + 1)); b = np.clip(np.searchsorted(qs[1:-1], c, side='right'), 0, nb - 1); L = []
        for i in range(nb):
            s = b == i
            L.append(dict(c_range_mrad=[float(1e3 * qs[i]), float(1e3 * qs[i + 1])], c_median_mrad=float(1e3 * np.median(c[s])),
                          dlog_mean=float(dlog[s].mean()), R_margin=proj(Wb, e, d, s & wz), R_h4=proj(Wb, eh, dh, s),
                          R_margin_C3=proj(Wb, en, d, s & wz), calls_changed_T=float((RS['T']['call'] != c0)[s].mean()),
                          calls_changed_C1=float((RS['C1']['call'] != c0)[s].mean()),
                          calls_changed_C3=float((RS['C3']['call'] != c0)[s].mean())))
        o[key] = L
    return o


def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.x10_das_jedi import clean, RidgeR, mass
    sm = a.smoke; pc = sm or 5000; part = a.part; NB = 500; NNULL = 0 if part == 'c2st34' else (200 if part == 'main' else 100)
    if sm: NNULL = min(NNULL, 10)
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / (f'e32_{part}' + ('_smoke' if sm else '') + '.json')
    seed = 3202 if part == 'rep' else 3201
    R = dict(part=part, sample=f'sample(per_class={pc}, seed={seed})', network='official JEDI-linear 3-feature n64 (jedi_core.Jedi(64))')
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    net = Jedi(64)
    x, y, _ = sample(pc, seed); x = np.ascontiguousarray(x, np.float32)
    if part == 'rep':
        xo, _, _ = sample(pc, 3201); hs = {xo[j].tobytes() for j in range(len(xo))}
        keep = np.array([x[j].tobytes() not in hs for j in range(len(x))]); R['overlap_removed'] = int((~keep).sum())
        x, y = x[keep], y[keep]; del xo, hs
    J = len(y); R['N'] = J; R['N_by_class'] = np.bincount(y, minlength=5).tolist()
    KEEPS = ('logits', 'h4', 'g', 'S', 'h5')
    B0 = net.run(x, keep=KEEPS, batch=1000); c0 = B0['logits'].argmax(1); R['acc'] = float((c0 == y).mean())
    log(f'part {part}: N {J} acc {R["acc"]:.4f}')
    logm0 = np.log(np.maximum(mass(x), 1e-3)); pr = np.random.default_rng(seed + 10).permutation(J); hA = np.sort(pr[:J // 2])
    nf = 2 * len(hA) // 3; RO = RidgeR(B0['h4'], logm0, hA[:nf], hA[nf:])
    R['readout_heldout_R2'] = float(1 - ((RO(B0['h4'])[pr[J // 2:]] - logm0[pr[J // 2:]]) ** 2).mean() / logm0[pr[J // 2:]].var())
    mg = lambda L: L[:, W_] - L[:, Z_]
    base = dict(call=c0, margin=mg(B0['logits']), ro=RO(B0['h4']))
    rng = np.random.default_rng(seed + 1)
    E, R['axis_info'] = build(x, rng); log('edits built', R['axis_info'])
    Wb = np.random.default_rng(seed + 2).poisson(1.0, (NB, J)).astype(np.float32)

    def resp(xx, nn=net):
        P = nn.run(xx, keep=('logits', 'h4'), batch=1000); return dict(call=P['logits'].argmax(1), margin=mg(P['logits']), ro=RO(P['h4']))

    if part in ('main', 'rep'):
        R['edits'] = {}; RS_T1 = None
        for t in TS:
            RS = dict(base=base, T=resp(E[t]['x']), C1=resp(E[t]['C1']), C2=resp(E[t]['C2']), C3=resp(E[t]['C3']))
            R['edits'][t] = metrics(t, E[t], RS, c0, y, Wb, rng)
            b = R['edits'][t]['by_edit']
            log(f'{t}: |c| median {R["edits"][t]["offset_mrad"]["q"]["q50"]:.2f} mrad, dlog mean {R["edits"][t]["dlog_M0M2"]["mean"]:.4f}; '
                f'calls T {b["T"]["calls_changed"]:.4f} C1 {b["C1"]["calls_changed"]:.4f} C2 {b["C2"]["calls_changed"]:.4f} C3 {b["C3"]["calls_changed"]:.4f}; '
                f'R_margin T {b["T"]["R_margin"].get("R")} {b["T"]["R_margin"].get("ci")} C2 {b["C2"]["R_margin"].get("R")} C3 {b["C3"]["R_margin"].get("R")}; '
                f'R_h4 {b["T"]["R_h4"].get("R")} {b["T"]["R_h4"].get("ci")}; WZ {b["T"]["n_W_to_Z"]} ZW {b["T"]["n_Z_to_W"]}')
            if t == 'T1': RS_T1 = RS
            save()
        # example W jet for the figure (main): true W, called W, T1 offset nearest the T1 median
        if part == 'main':
            c = E['T1']['c']; cand = np.flatnonzero((y == W_) & (c0 == W_)); j = int(cand[np.argmin(np.abs(c[cand] - np.median(c)))])
            rl = x[j, :, 0] > 0
            R['example'] = dict(index=j, particles=x[j, rl].tolist(), T1_axis=[float(E['T1']['ce'][j]), float(E['T1']['cp'][j])],
                                T3_axis=[float(E['T3']['ce'][j]), float(E['T3']['cp'][j])], T4_axis=[float(E['T4']['ce'][j]), float(E['T4']['cp'][j])],
                                T0_axis=[float(E['T0']['ce'][j]), float(E['T0']['cp'][j])], mass=float(mass(x[j:j + 1])[0]),
                                dlog_T1=float(E['T1']['dlog'][j]), call=int(c0[j]), call_T1=int(RS_T1['T']['call'][j]),
                                margin=float(base['margin'][j]), margin_T1=float(RS_T1['T']['margin'][j]))
        # ---- freezes under T1
        live = {L: B0[L].std(0) > 1e-9 for L in ('g', 'S', 'h4', 'h5')}; R['live_units'] = {L: int(v.sum()) for L, v in live.items()}
        xd = dilate(x, np.full(J, np.sqrt(1.05))); Pd = net.run(xd[hA], keep=('h4', 'h5'), batch=1000); top = {}
        for L in ('h4', 'h5'):
            sd = B0[L][hA].std(0) + 1e-12; sc = np.where(live[L], np.abs((Pd[L] - B0[L][hA]).mean(0)) / sd, -1); top[L] = np.argsort(-sc)[:4].tolist()
            R[f'top_{L}_by_M2_dilation'] = dict(units=top[L], score=[float(sc[u]) for u in top[L]])
        del Pd, xd
        HOLDS = dict(g={'g': list(range(64))}, S0_S29={'S': [0, 29]}, h45_top={'h4': top['h4'], 'h5': top['h5']}, h45_E6={'h4': [18], 'h5': [0, 28]})
        xT1 = E['T1']['x']; chT1 = RS_T1['T']['call'] != c0
        ev = np.union1d(np.flatnonzero(chT1), np.random.default_rng(seed + 3).choice(J, min(J, 3000 if part == 'main' else 2000), replace=False))
        R['freeze_eval_set'] = dict(n=int(len(ev)), n_T1_changed=int(chT1.sum()))
        idS = np.random.default_rng(seed + 4)
        chk = net.run_jetwise(x[ev[:500]], {'S': (np.ones(64), B0['S'][ev[:500]])})['logits']
        R['run_jetwise_identity_max_abs'] = float(np.abs(chk - B0['logits'][ev[:500]]).max())

        def held(units):
            ov = {}
            for L, us in units.items():
                m = np.zeros(B0[L].shape[1]); m[us] = 1; ov[L] = (m, B0[L][ev])
            P = net.run_jetwise(xT1[ev], ov, keep=('logits',), batch=1000)['logits']; return P.argmax(1), mg(P)

        n_ch = int(chT1[ev].sum()); eT = RS_T1['T']['margin'][ev] - base['margin'][ev]; wzv = ((c0 == W_) | (c0 == Z_))[ev]
        def removed(call, marg):
            ch = call != c0[ev]; em = marg - base['margin'][ev]
            return float(1 - ch.sum() / max(n_ch, 1)), float(1 - (em * eT)[wzv].sum() / max((eT * eT)[wzv].sum(), 1e-30))
        pool = dict(S=[('S', u) for u in np.flatnonzero(live['S']) if u not in (0, 29)],
                    h45=[('h4', u) for u in np.flatnonzero(live['h4'])] + [('h5', u) for u in np.flatnonzero(live['h5'])],
                    down=[('S', u) for u in np.flatnonzero(live['S'])] + [('h4', u) for u in np.flatnonzero(live['h4'])] + [('h5', u) for u in np.flatnonzero(live['h5'])])
        R['freezes'] = {}; ps = []
        for name, units in HOLDS.items():
            cl, mgv = held(units); rc, rm = removed(cl, mgv)
            k = int(live['g'].sum()) if name == 'g' else sum(len(v) for v in units.values())
            src = 'down' if name == 'g' else ('S' if name == 'S0_S29' else 'h45')
            if name.startswith('h45'): pl = [p for p in pool['h45'] if p[1] not in units.get(p[0], [])]
            else: pl = pool[src]
            nul_c, nul_m = [], []
            for _ in range(NNULL):
                pick = [pl[i] for i in idS.choice(len(pl), min(k, len(pl)), replace=False)]; U = {}
                for L, u in pick: U.setdefault(L, []).append(int(u))
                a_, b_ = removed(*held(U)); nul_c.append(a_); nul_m.append(b_)
            nul_c = np.array(nul_c); p = float((1 + (nul_c >= rc).sum()) / (1 + len(nul_c))) if len(nul_c) else None; ps.append(p if p is not None else 1.0)
            R['freezes'][name] = dict(units={L: list(map(int, v)) for L, v in units.items()}, k_size_matched=k, null_pool=src,
                                      removed_calls=rc, removed_margin_proj=rm, null_removed_calls_q50=float(np.median(nul_c)) if len(nul_c) else None,
                                      null_removed_calls_q975=float(np.quantile(nul_c, .975)) if len(nul_c) else None,
                                      null_removed_calls_max=float(nul_c.max()) if len(nul_c) else None,
                                      null_removed_margin_q50=float(np.median(nul_m)) if len(nul_m) else None,
                                      null_removed_margin_q975=float(np.quantile(nul_m, .975)) if len(nul_m) else None, p_rank=p, n_null=len(nul_c))
            log(f'freeze {name}: removed calls {rc:.3f} margin {rm:.3f}; null q50 {R["freezes"][name]["null_removed_calls_q50"]} '
                f'q97.5 {R["freezes"][name]["null_removed_calls_q975"]} p {p}'); save()
        for (name, o), h in zip(R['freezes'].items(), holm(ps)): o['p_holm4'] = h

    if part in ('main', 'c2st34'):
        from interp.e31_onmanifold_jedi import featsets, c2st
        from interp.e31_verify_jedi import knn_support
        nsub = min(J, 2000 * 5 if not sm else J); sub = np.sort(np.random.default_rng(seed + 5).choice(J, nsub, replace=False))
        F0 = featsets(x[sub])[0]; F0['inv'] = F0['obs'][:, INV]
        if part == 'main':
            ED = dict(T0=E['T0']['x'], T1=E['T1']['x'], T2=E['T2']['x'], C1_T1=E['T1']['C1'], C3=E['T1']['C3'])
        else:
            ED = dict(T3=E['T3']['x'], T4=E['T4']['x'], C1_T3=E['T3']['C1'], C1_T4=E['T4']['C1'])
        FE = {}
        for k, xe in ED.items():
            f = featsets(xe[sub])[0]; f['inv'] = f['obs'][:, INV]; FE[k] = f
        log('features done')
        R['c2st'] = {}; ok = np.ones(nsub, bool); nb = 100 if sm else 300
        for fs in ('inv', 'obs8', 'obs', 'plist'):
            R['c2st'][fs] = {}
            for i, k in enumerate(ED):
                res = c2st(F0[fs], FE[k][fs], ok, 3210 + i, 5, nb, nb)[0]; R['c2st'][fs][k] = res
                log(f'C2ST {fs} {k}: AUC {res["auc"]:.4f} {res["ci"]} p {res["p_swap"]:.4f}')
            save()
        isA = np.zeros(nsub, bool); isA[np.random.default_rng(seed + 6).permutation(nsub)[:nsub // 2]] = True
        R['support'] = {}
        for fs in ('inv', 'obs8', 'obs'):
            names = list(ED); thr, dh, dq = knn_support(F0[fs][isA], [FE[k][fs][~isA] for k in names], F0[fs][~isA])
            o = dict(threshold_q99_unedited=thr, unedited_out_frac=float((dh > thr).mean())); rb = np.random.default_rng(3220)
            for k, d in zip(names, dq):
                out = (d > thr).astype(float)
                o[k] = dict(n=int(len(d)), out_frac=float(out.mean()), out_ci=qci([out[rb.integers(0, len(out), len(out))].mean() for _ in range(300)]))
            R['support'][fs] = o; log(f'support {fs}: ' + ', '.join(f'{k} {o[k]["out_frac"]:.4f}' for k in names)); save()
        # natural centroid term vs the edited one
        R['centroid_term'] = {k: dict(q50=float(np.median(FE[k]['obs'][:, 8])), q99=float(np.quantile(FE[k]['obs'][:, 8], .99)),
                                      frac_above_unedited_max=float((FE[k]['obs'][:, 8] > F0['obs'][:, 8].max()).mean())) for k in ED}
        R['centroid_term']['unedited'] = dict(q50=float(np.median(F0['obs'][:, 8])), q99=float(np.quantile(F0['obs'][:, 8], .99)),
                                              max=float(F0['obs'][:, 8].max()))
        save()

    if part == 'c2st34':   # C4: untrained redraws
        R['untrained'] = {}; wz = (c0 == W_) | (c0 == Z_)
        for s in (1, 2):
            un = Jedi(64, untrained=True, seed=s); P0 = un.run(x, keep=('logits',), batch=1000)['logits']; u0 = mg(P0); uc0 = P0.argmax(1)
            o = dict(n_distinct_calls=int(len(np.unique(uc0))), margin_sd=float(u0.std()))
            for t in TS:
                Pt = un.run(E[t]['x'], keep=('logits',), batch=1000)['logits']; Pd = un.run(E[t]['C1'], keep=('logits',), batch=1000)['logits']
                o[t] = dict(R_margin_all=proj(Wb, mg(Pt) - u0, mg(Pd) - u0, np.ones(J, bool)), R_margin_trainedWZ=proj(Wb, mg(Pt) - u0, mg(Pd) - u0, wz),
                            calls_changed=float((Pt.argmax(1) != uc0).mean()), calls_changed_C1=float((Pd.argmax(1) != uc0).mean()))
                log(f'untrained {s} {t}: R {o[t]["R_margin_all"].get("R")} {o[t]["R_margin_all"].get("ci")} calls {o[t]["calls_changed"]:.4f}')
            R['untrained'][f'seed{s}'] = o; save(); del un
    R['seconds'] = time.time() - T0_; save(); log(f'done, wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', default='main', choices=['main', 'c2st34', 'rep'])
    ap.add_argument('--smoke', type=int, default=0)
    main(ap.parse_args())
