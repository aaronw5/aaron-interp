"""H3 (claim D3) — full-statistics rerun of the D1 / D2 physics edits (JEDI-linear J27/J27b, J30; ParT G192, G194, G199, G200,
G202-G204) with the corrected JetClass build (eta = jet_eta + sign(jet_eta)·deta), random rows, a matched control for every edit,
paired tests and background rejection at fixed signal efficiency.

Samples
  JEDI-linear (64 particles, 3-feature, official checkpoint): jedi_core.sample(per_class=4000, seed=501) = 20,000 hls4ml test jets.
  ParT_full / ParT_kin: 4,000 random rows (numpy default_rng(502), drawn class by class in CLASSES order, without replacement) of each
  class's first JetClass test file (100,000 jets each) = 40,000 jets. Unedited calls are taken on the rebuilt jets
  (part_core.build of the unedited kinematics), so edited and unedited jets share one convention; the raw-file accuracy, the
  corrected-rebuild change and the old (mirrored) rebuild change are logged as checks.

Edits (each with its control; "rows" = the classes the edit is applied to)
  boost k (k = 0.8, 0.9, 1.1, 1.25)  pT × k, (Δη, Δφ) / k.  Control: rotation of (Δη, Δφ) about the jet axis by
                                     θ = 2 arcsin(|1/k − 1| / 2) (random sign), which moves each particle by the same distance
                                     |r|·|1/k − 1| and keeps ΣpT.  All classes.
  decay collapse                     primary Lund declustering (C/A, harder branch) of the 64 hardest particles; the softer branch
                                     of the largest-kt emission is moved onto the harder branch's direction.  Control: the same for
                                     the other primary emission whose softer-branch pT is closest (in log) to that of the largest-kt
                                     one.  All classes.
  W-mass shift in top (t = 60/100/120) top jets with a 65-95 GeV pair among 3 pT-weighted k-means subjets: the W pair's subjets are
                                     spread / squeezed about their centroid until the pair mass is t, then all angles rescaled so
                                     the jet mass is the original one (as _wintop.move_w).  Control: the same procedure on the
                                     non-W pair (b + harder W daughter) with target m_pair + (t − m_W).  Top only.
  remove b                           drop the b subjet, recentre on the rest, boost back to the original ΣpT (J27b / G192).
                                     Control: drop the W daughter whose pT is closest to the b's.  Top only; key flow = call W or Z.
  ParT_full only (track / particle-ID inputs):
  displaced made prompt              d0, dz × 0 for charged tracks with |d0|/σ > 2 (G194).
  lifetime × 0.35 / × 2.5            d0, dz × 0.35 (H→bb) / × 2.5 (H→cc) for the same tracks (count at S > 2 kept, G194).
  keep 2 displaced                   only the 2 most displaced stay displaced, the others' d0, dz × 0 (G194).
  drop displaced tracks              remove the S > 2 tracks.  Control: remove as many prompt (S < 2) charged tracks, pT-matched.
  lepton de-isolation                the 4 particles nearest the leading lepton (outside ΔR 0.1) moved to ΔR 0.03 from it (G203).
                                     Control: the same move around the charged hadron of pT closest to the lepton's; second control:
                                     the same particles moved the same distance in a random direction (G203's control).
  lepton charge flip                 lepton charge and d0 sign flipped (b side untouched).  Control: one charged hadron of nearest
                                     pT rank flipped (G203).
  π⁰ merge                           photon pairs with m_γγ in 85-185 MeV merged into one photon (pT summed, pT-weighted direction).
                                     Control: as many photon pairs with m_γγ in 250-450 MeV merged (closest first).
  dead-cone fill                     6 photons of 1 GeV within ΔR 0.015 of each prong's displaced-track centroid (G204).
                                     Control: the same photons at ΔR 0.05-0.08.
Metrics per (model, edit, class): % calls changed; key indicator (own-class call, or call ∈ key set) before → after, its change with
95 % bootstrap CI (2,000 resamples of jets), the control's change, edit − control; McNemar exact tests (edit vs unedited, edit vs
control); mean Δ log p(true class) with CI and a sign-flip permutation test edit vs control; Holm across the edit-vs-control family
of each model.  Background rejection 1/ε_B at ε_S = 50 % and at the classical tagger's ε_S, before / after (bootstrap CI of the
ratio), plus the classical tagger's own ε_S, ε_B on the edited jets.  Discriminant p_S / (p_S + p_B).  Classical taggers:
W/Z: 65 < m_SD < 105 GeV and τ21 < 0.5; top: 140 < m_SD < 200 GeV and τ32 < 0.6; H→bb (ParT): 100 < m_SD < 150 GeV and ≥ 2 charged
tracks with |d0|/σ > 3.  m_SD: soft drop β = 0, z_cut 0.1 on the 40 hardest particles; τN: pT-weighted k-means axes, R0 = 0.8.

  ./job.sh d-h3-run 1 2 'python -m interp.h3_d3_fullstats jedi'          -> $AI_RESULTS/physics/h3_jedi.json
  ./part_job.sh d-h3-part 'python -m interp.h3_d3_fullstats part'        -> $AI_RESULTS/physics/h3_part.json
  (argument 'smoke' after the model: 100 jets per class)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jetdistill.observables.compute import _kmeans, _p4, _m4, _softdrop
from interp._lund import lund

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
NB = 2000
ITERS, TOL, WONLY = 5, None, False    # 'wshift' mode: ITERS = 30, TOL = 2 GeV, only the W-shift edits
RNG = np.random.default_rng(503)


# ----------------------------------------------------------------------------------------------------------- statistics
def boot_ci(a, B=NB):
    a = np.asarray(a, np.float64); n = len(a)
    if n == 0: return [float('nan')] * 2
    ms = np.concatenate([a[RNG.integers(0, n, (200, n))].mean(1) for _ in range(B // 200)])
    return [float(np.quantile(ms, 0.025)), float(np.quantile(ms, 0.975))]


def mcnemar(x0, x1):
    """exact two-sided McNemar on paired booleans: b = x0 & ~x1, c = ~x0 & x1"""
    from scipy.stats import binom
    b, c = int((x0 & ~x1).sum()), int((~x0 & x1).sum()); n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))
    return dict(b=b, c=c, p=p)


def signflip(d, P=5000):
    d = np.asarray(d, np.float64); n = len(d)
    if n == 0: return float('nan')
    T = abs(d.mean()); cnt = 0
    for _ in range(P // 500):
        s = RNG.choice([-1.0, 1.0], (500, n)); cnt += int((np.abs((s * d).mean(1)) >= T - 1e-12).sum())
    return (cnt + 1) / (P + 1)


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def logsoft(L):
    L = np.asarray(L, np.float64); M = L.max(1, keepdims=True); return L - M - np.log(np.exp(L - M).sum(1, keepdims=True))


# ----------------------------------------------------------------------------------------------------------- geometry
def mass_of(pt, de, dp, sel=None):
    w = pt if sel is None else pt * sel; return _m4(*_p4(w, de, dp).sum(1).T)


def boost(pt, de, dp, k):
    return pt * k, de / k, dp / k


def rotate(pt, de, dp, k):
    th = 2 * np.arcsin(min(1.0, abs(1 / k - 1) / 2)) * RNG.choice([-1.0, 1.0], len(pt))[:, None]
    c, s = np.cos(th), np.sin(th); return pt, de * c - dp * s, de * s + dp * c


def collapse_pair(pt, de, dp, H=64):
    """(edit de, dp), (control de, dp), ok, soft-branch pT of edit and control emissions"""
    H = min(H, pt.shape[1]); EM = lund(np.stack([pt[:, :H], de[:, :H], dp[:, :H]], -1).astype(np.float64))
    deE, dpE, deC, dpC = de.copy(), dp.copy(), de.copy(), dp.copy(); n = len(pt)
    ok = np.zeros(n, bool); sE, sC = np.zeros(n), np.zeros(n)
    for r, E in enumerate(EM):
        if len(E) < 2: continue
        spt = np.array([pt[r, :H][e['soft']].sum() for e in E]); hk = int(np.argmax([e['lkt'] for e in E]))
        oth = [k for k in range(len(E)) if k != hk and spt[k] > 0]
        if not oth or spt[hk] <= 0: continue
        kc = min(oth, key=lambda k: abs(np.log(spt[k] / spt[hk])))
        for k, D, P in ((hk, deE, dpE), (kc, deC, dpC)):
            s = E[k]['soft']; v = D[r, :H]; v[s] = E[k]['hard_yphi'][0]; v = P[r, :H]; v[s] = E[k]['hard_yphi'][1]
        ok[r] = True; sE[r], sC[r] = spt[hk], spt[kc]
    return (deE, dpE), (deC, dpC), ok, sE, sC


def top_split(pt, de, dp):
    rows = np.arange(len(pt)); ax, d = _kmeans(pt, de, dp, 3); a = d.argmin(-1); a[pt <= 0] = -1
    v = np.stack([_p4(pt * (a == j), de, dp).sum(1) for j in range(3)], 1); P = np.array([(0, 1), (0, 2), (1, 2)])
    pm = np.stack([_m4(*(v[:, i] + v[:, j]).T) for i, j in P], 1); w = np.abs(pm - 80.4).argmin(1)
    ok = np.abs(pm[rows, w] - 80.4) < 15; i, j = P[w, 0], P[w, 1]; b = 3 - i - j; ptj = np.hypot(v[..., 0], v[..., 1])
    hardW = np.where(ptj[rows, i] >= ptj[rows, j], i, j)
    lightW = np.where(np.abs(ptj[rows, i] - ptj[rows, b]) <= np.abs(ptj[rows, j] - ptj[rows, b]), i, j)
    return dict(a=a, ok=ok, i=i, j=j, b=b, hardW=hardW, lightW=lightW, ptj=ptj, mW=pm[rows, w])


def move_set(pt, de, dp, inS, target):
    m_top = mass_of(pt, de, dp); de, dp = de.copy(), dp.copy()
    for _ in range(ITERS):
        wp = pt * inS; c = (wp[..., None] * np.stack([de, dp], -1)).sum(1) / np.maximum(wp.sum(1), 1e-9)[:, None]
        f = (target / np.maximum(mass_of(pt, de, dp, inS), 1e-3))[:, None]
        de = np.where(inS, c[:, None, 0] + f * (de - c[:, None, 0]), de); dp = np.where(inS, c[:, None, 1] + f * (dp - c[:, None, 1]), dp)
        s = (m_top / np.maximum(mass_of(pt, de, dp), 1e-3))[:, None]; de, dp = de * s, dp * s
    return de, dp, mass_of(pt, de, dp, inS), mass_of(pt, de, dp)


def wshift(pt, de, dp, t, TS):
    """edit and control (de, dp) for target W pair mass t; ok; achieved masses"""
    a = TS['a']; inW = (a == TS['i'][:, None]) | (a == TS['j'][:, None]); inC = (a == TS['b'][:, None]) | (a == TS['hardW'][:, None])
    inW &= pt > 0; inC &= pt > 0
    mC0 = mass_of(pt, de, dp, inC); tC = mC0 + (t - TS['mW']); ok = TS['ok'] & (tC > 5)
    deE, dpE, mWe, mTe = move_set(pt, de, dp, inW, np.full(len(pt), t))
    deC, dpC, mCc, mTc = move_set(pt, de, dp, inC, np.maximum(tC, 5.0))
    if TOL is not None: ok = ok & (np.abs(mWe - t) < TOL) & (np.abs(mCc - tC) < TOL)   # matched achieved shifts (wshift mode)
    return (deE, dpE), (deC, dpC), ok, dict(mW_before=TS['mW'], mW_after=mWe, mC_before=mC0, mC_target=tC, mC_after=mCc,
                                             mjet_before=mass_of(pt, de, dp), mjet_edit=mTe, mjet_ctrl=mTc)


def drop_prong(pt, de, dp, a, which):
    pt1 = np.where(a == which[:, None], 0.0, pt)
    c = (pt1[..., None] * np.stack([de, dp], -1)).sum(1) / np.maximum(pt1.sum(1), 1e-9)[:, None]
    k = (pt.sum(1) / np.maximum(pt1.sum(1), 1e-9))[:, None]
    return pt1 * k, np.where(pt1 > 0, (de - c[:, None, 0]) / k, 0), np.where(pt1 > 0, (dp - c[:, None, 1]) / k, 0)


def classical_obs(pt, de, dp, chunk=1000, HSD=40):
    out = dict(msd=[], tau21=[], tau32=[])
    for i0 in range(0, len(pt), chunk):
        p, e, f = (np.asarray(a[i0:i0 + chunk], np.float64) for a in (pt, de, dp))
        o = np.argsort(-p, 1, kind='stable'); p, e, f = (np.take_along_axis(a, o, 1) for a in (p, e, f))
        out['msd'].append(_softdrop(p[:, :HSD], e[:, :HSD], f[:, :HSD])[0]); z = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
        tau = {k: (z * _kmeans(p, e, f, k)[1].min(-1)).sum(1) / 0.8 for k in (1, 2, 3)}
        out['tau21'].append(tau[2] / np.maximum(tau[1], 1e-12)); out['tau32'].append(tau[3] / np.maximum(tau[2], 1e-12))
    return {k: np.concatenate(v) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------------------- evaluation
class Store:
    """unedited logits L0[model] (N, C), labels y; variants[name] = dict(rows, L={model: (n, C)}, obs, info)"""

    def __init__(self, y, L0, classes, obs0, eta_sign=None):
        self.y, self.L0, self.CL, self.obs0, self.V = y, L0, classes, obs0, {}; self.eta_sign = eta_sign

    def add(self, name, rows, L, obs=None, info=None):
        self.V[name] = dict(rows=np.asarray(rows), L=L, obs=obs, info=info or {})


def key_ind(c, y, key, CL):
    if key == 'own': return c == y
    return np.isin(c, [CL.index(k) for k in key])


def compare(S, model, edit, ctrl, cls, key='own', split_eta=False):
    """metrics on the jets of class cls valid in both edit and control"""
    CL = S.CL; ci = CL.index(cls); E = S.V[edit]; rows = E['rows'][S.y[E['rows']] == ci]
    if ctrl is not None:
        C = S.V[ctrl]; rows = np.intersect1d(rows, C['rows'])
    n = len(rows)
    if n < 20: return dict(n=n)
    pos = lambda V: np.searchsorted(V['rows'], rows)
    L0 = S.L0[model][rows]; LE = E['L'][model][pos(E)]; y = S.y[rows]; c0, cE = L0.argmax(1), LE.argmax(1)
    lp0, lpE = logsoft(L0)[np.arange(n), y], logsoft(LE)[np.arange(n), y]; I0, IE = key_ind(c0, y, key, CL), key_ind(cE, y, key, CL)
    dist = lambda c: [float((c == k).mean()) for k in range(len(CL))]
    r = dict(n=n, key=key, before=dist(c0), after=dist(cE), key_before=float(I0.mean()), key_after=float(IE.mean()),
             key_change=float(IE.mean() - I0.mean()), key_change_ci=boot_ci(IE.astype(float) - I0),
             changed=float((cE != c0).mean()), changed_ci=boot_ci((cE != c0).astype(float)),
             dlogp=float((lpE - lp0).mean()), dlogp_ci=boot_ci(lpE - lp0), mcnemar_vs_unedited=mcnemar(I0, IE))
    if ctrl is not None:
        LC = C['L'][model][pos(C)]; cC = LC.argmax(1); lpC = logsoft(LC)[np.arange(n), y]; IC = key_ind(cC, y, key, CL)
        r.update(ctrl_after=dist(cC), ctrl_key_after=float(IC.mean()), ctrl_key_change=float(IC.mean() - I0.mean()),
                 ctrl_key_change_ci=boot_ci(IC.astype(float) - I0), ctrl_changed=float((cC != c0).mean()),
                 ctrl_changed_ci=boot_ci((cC != c0).astype(float)), ctrl_dlogp=float((lpC - lp0).mean()), ctrl_dlogp_ci=boot_ci(lpC - lp0),
                 diff=float(IE.mean() - IC.mean()), diff_ci=boot_ci(IE.astype(float) - IC), mcnemar_vs_ctrl=mcnemar(IC, IE),
                 dlogp_diff=float((lpE - lpC).mean()), dlogp_diff_ci=boot_ci(lpE - lpC), dlogp_perm_p=signflip(lpE - lpC))
    if split_eta and S.eta_sign is not None:
        r['by_eta_sign'] = {}
        for lab, m in (('eta<0', S.eta_sign[rows] < 0), ('eta>=0', S.eta_sign[rows] >= 0)):
            sub = dict(n=int(m.sum()), key_before=float(I0[m].mean()), key_after=float(IE[m].mean()), key_change_ci=boot_ci(IE[m].astype(float) - I0[m]))
            if ctrl is not None: sub.update(ctrl_key_after=float(IC[m].mean()), diff_ci=boot_ci(IE[m].astype(float) - IC[m]), mcnemar_vs_ctrl=mcnemar(IC[m], IE[m]))
            r['by_eta_sign'][lab] = sub
    return r


def disc(L, s, bkg):
    p = np.exp(logsoft(L)); return p[:, s] / np.maximum(p[:, s] + p[:, bkg].sum(1), 1e-30)


def rej_at(sS, sB, eff):
    t = np.quantile(sS, 1 - eff); eB = (sB > t).mean(); return 1.0 / max(eB, 0.5 / len(sB)), float(eB)


def rejection(S, model, name, sig, bkgs, cut):
    """ROC rejection at 50 % and at the classical tagger's efficiency, before / after the edit (signal jets valid for the edit;
    background edited when the edit covers it, else unedited); classical tagger's ε_S, ε_B before / after"""
    CL = S.CL; s = CL.index(sig); b = [CL.index(x) for x in bkgs]; V = S.V[name]
    rs = V['rows'][S.y[V['rows']] == s]
    if len(rs) < 50: return None
    rbE = V['rows'][np.isin(S.y[V['rows']], b)]; edited_bkg = len(rbE) > 0
    rb = rbE if edited_bkg else np.flatnonzero(np.isin(S.y, b))
    pos = lambda r: np.searchsorted(V['rows'], r)
    sS0, sB0 = disc(S.L0[model][rs], s, b), disc(S.L0[model][rb], s, b)
    sS1 = disc(V['L'][model][pos(rs)], s, b); sB1 = disc(V['L'][model][pos(rb)], s, b) if edited_bkg else sB0
    O0 = S.obs0; O1 = V['obs']
    cS0 = cut(O0, rs); cB0 = cut(O0, rb)
    if O1 is not None:
        cS1 = cut(O1, pos(rs), local=True); cB1 = cut(O1, pos(rb), local=True) if edited_bkg else cB0
    else:
        cS1, cB1 = cS0, cB0
    eps_cl = float(cS0.mean()); out = dict(n_sig=len(rs), n_bkg=len(rb), bkg_edited=edited_bkg,
                                           classical=dict(eps_S=[eps_cl, float(cS1.mean())], eps_B=[float(cB0.mean()), float(cB1.mean())]))
    for lab, eff in (('eps50', 0.5), ('eps_classical', eps_cl)):
        if eff <= 0.01: continue
        R0, e0 = rej_at(sS0, sB0, eff); R1, e1 = rej_at(sS1, sB1, eff); bs = []
        for _ in range(500):
            i = RNG.integers(0, len(rs), len(rs)); j = RNG.integers(0, len(rb), len(rb))
            a0 = rej_at(sS0[i], sB0[j], eff)[0]; a1 = rej_at(sS1[i], sB1[j], eff)[0]; bs.append((a0, a1, a1 / a0))
        bs = np.array(bs); q = lambda k: [float(np.quantile(bs[:, k], 0.025)), float(np.quantile(bs[:, k], 0.975))]
        out[lab] = dict(eff=float(eff), R_before=R0, R_after=R1, R_before_ci=q(0), R_after_ci=q(1), ratio=R1 / R0, ratio_ci=q(2))
    return out


def cut_factory(kind):
    def cut(O, idx, local=False):
        m, t21, t32 = O['msd'][idx], O['tau21'][idx], O['tau32'][idx]
        if kind == 'V': return (m > 65) & (m < 105) & (t21 < 0.5)
        if kind == 't': return (m > 140) & (m < 200) & (t32 < 0.6)
        if kind == 'bb': return (m > 100) & (m < 150) & (O['nS3'][idx] >= 2)
    return cut


def summarise(S, model, specs, pairs, split_eta_edits=()):
    """specs: list of (edit, ctrl, classes, key). Holm over edit-vs-control (family 1) and edit-vs-unedited (family 2)."""
    rows = []
    for edit, ctrl, classes, key in specs:
        if edit not in S.V or (ctrl is not None and ctrl not in S.V): continue
        for c in classes:
            r = compare(S, model, edit, ctrl, c, key, split_eta=edit in split_eta_edits)
            r.update(edit=edit, ctrl=ctrl, cls=c); rows.append(r)
    f1 = [i for i, r in enumerate(rows) if 'mcnemar_vs_ctrl' in r]; f2 = [i for i, r in enumerate(rows) if 'mcnemar_vs_unedited' in r]
    for fam, key, out in ((f1, 'mcnemar_vs_ctrl', 'p_holm_vs_ctrl'), (f2, 'mcnemar_vs_unedited', 'p_holm_vs_unedited')):
        if fam:
            adj = holm([rows[i][key]['p'] for i in fam])
            for i, a in zip(fam, adj): rows[i][out] = float(a)
    for i in f1: rows[i]['p_holm_dlogp_vs_ctrl'] = None
    if f1:
        adj = holm([rows[i]['dlogp_perm_p'] for i in f1])
        for i, a in zip(f1, adj): rows[i]['p_holm_dlogp_vs_ctrl'] = float(a)
    rej = {}
    for name in S.V:
        for sig, bkgs, kind in pairs:
            r = rejection(S, model, name, sig, bkgs, cut_factory(kind))
            if r is not None: rej[f'{name} | {sig} vs {"+".join(bkgs)}'] = r
    return dict(comparisons=rows, rejection=rej)


def fmt(r, CL):
    if 'key_before' not in r: return f"{r['edit']} [{r['cls']}]: n={r['n']} (skipped)"
    s = (f"{r['edit']} [{r['cls']}, n={r['n']}, key {r['key']}]: {100 * r['key_before']:.1f} -> {100 * r['key_after']:.1f} % "
         f"(Δ {100 * r['key_change']:+.1f}, CI {100 * r['key_change_ci'][0]:+.1f}..{100 * r['key_change_ci'][1]:+.1f}); changed {100 * r['changed']:.1f} %")
    if 'ctrl_key_after' in r:
        s += (f" | ctrl {r['ctrl']}: {100 * r['ctrl_key_after']:.1f} % (Δ {100 * r['ctrl_key_change']:+.1f}); edit-ctrl {100 * r['diff']:+.1f} "
              f"(CI {100 * r['diff_ci'][0]:+.1f}..{100 * r['diff_ci'][1]:+.1f}), McNemar p {r['mcnemar_vs_ctrl']['p']:.2g}")
    s += ' | after: ' + ' '.join(f'{c} {100 * v:.0f}' for c, v in zip(CL, r['after']) if v >= 0.03)
    return s


# ----------------------------------------------------------------------------------------------------------- JEDI
def run_jedi(per):
    from interp.jedi_core import Jedi, sample
    CL = ['g', 'q', 'W', 'Z', 't']
    x, y, Q = sample(per_class=per, seed=501); x = np.asarray(x, np.float64); y = y.astype(int)
    log(f'JEDI sample: {len(y)} jets, labels {np.bincount(y).tolist()}')
    net = Jedi(64, '3-feature'); run = lambda xx: net.run(np.asarray(xx, np.float32))['logits']
    L0 = run(x); acc = float((L0.argmax(1) == y).mean()); log(f'JEDI-linear unedited accuracy on the sample: {100 * acc:.2f} %')
    pt, de, dp = x[..., 0], x[..., 1], x[..., 2]
    O0 = classical_obs(pt, de, dp); log('classical observables (unedited) done')
    S = Store(y, {'jedi': L0}, CL, O0)

    def resort(pt, de, dp):
        o = np.argsort(-pt, 1, kind='stable'); X = np.stack([np.take_along_axis(a, o, 1) for a in (pt, de, dp)], -1)
        X[X[..., 0] <= 0] = 0; return X

    def add(name, rows, X, info=None):
        S.add(name, rows, {'jedi': run(X)}, classical_obs(X[..., 0], X[..., 1], X[..., 2]), info); log(f'variant {name}: {len(rows)} jets')

    allr = np.arange(len(y))
    for k in (() if WONLY else (0.8, 0.9, 1.1, 1.25)):
        add(f'boost x{k}', allr, np.stack(boost(pt, de, dp, k), -1)); add(f'rotation (boost x{k} displacement)', allr, np.stack(rotate(pt, de, dp, k), -1))
    if not WONLY:
        (dE, pE), (dC, pC), ok, sE, sC = collapse_pair(pt, de, dp); r = allr[ok]
        info = dict(soft_pt_edit_median=float(np.median(sE[ok])), soft_pt_ctrl_median=float(np.median(sC[ok])),
                    log_ratio_median=float(np.median(np.log(sC[ok] / sE[ok]))))
        add('decay collapse', r, np.stack([pt, dE, pE], -1)[ok], info); add('control: matched-pT emission collapse', r, np.stack([pt, dC, pC], -1)[ok], info)
        log(f'collapse info {info}')
    top = allr[y == 4]; TS = top_split(pt[top], de[top], dp[top]); log(f'top jets with a W pair: {int(TS["ok"].sum())}/{len(top)}')
    for t in (60.0, 80.4, 100.0, 120.0):
        (dE, pE), (dC, pC), ok, inf = wshift(pt[top], de[top], dp[top], t, TS); r = top[ok]
        info = {k: float(np.median(v[ok])) for k, v in inf.items()}
        add(f'W pair -> {t:g} GeV', r, np.stack([pt[top], dE, pE], -1)[ok], info)
        add(f'control: non-W pair shifted by {t:g}-mW', r, np.stack([pt[top], dC, pC], -1)[ok], info); log(f'W shift {t}: {info}')
    ok = TS['ok']; r = top[ok]; ptt, det, dpt = pt[top][ok], de[top][ok], dp[top][ok]; a = TS['a'][ok]
    ratio = TS['ptj'][ok][np.arange(ok.sum()), TS['lightW'][ok]] / TS['ptj'][ok][np.arange(ok.sum()), TS['b'][ok]]
    info = dict(light_over_b_pt_median=float(np.median(ratio)))
    add('remove b (recentred, boosted back)', r, resort(*drop_prong(ptt, det, dpt, a, TS['b'][ok])), info)
    add('control: remove the pT-matched W daughter', r, resort(*drop_prong(ptt, det, dpt, a, TS['lightW'][ok])), info)
    specs = []
    for k in (0.8, 0.9, 1.1, 1.25): specs.append((f'boost x{k}', f'rotation (boost x{k} displacement)', CL, 'own'))
    specs.append(('decay collapse', 'control: matched-pT emission collapse', CL, 'own'))
    for t in (60.0, 80.4, 100.0, 120.0): specs.append((f'W pair -> {t:g} GeV', f'control: non-W pair shifted by {t:g}-mW', ['t'], 'own'))
    specs.append(('remove b (recentred, boosted back)', 'control: remove the pT-matched W daughter', ['t'], ('W', 'Z')))
    pairs = [('W', ['g', 'q'], 'V'), ('Z', ['g', 'q'], 'V'), ('t', ['g', 'q'], 't')]
    res = summarise(S, 'jedi', specs, pairs)
    for r in res['comparisons']: log(fmt(r, CL))
    for k, v in res['rejection'].items():
        log(f"rej {k}: " + '; '.join(f"{lab} R {v[lab]['R_before']:.1f} -> {v[lab]['R_after']:.1f} (ratio CI {v[lab]['ratio_ci'][0]:.2f}..{v[lab]['ratio_ci'][1]:.2f})" for lab in ('eps50', 'eps_classical') if lab in v)
            + f"; classical eps_S {v['classical']['eps_S']}, eps_B {v['classical']['eps_B']}")
    out = dict(model='JEDI-linear 64p 3-feature', sample=dict(per_class=per, seed=501, N=len(y)), accuracy=acc,
               own_class=[float((L0.argmax(1)[y == k] == k).mean()) for k in range(5)], classes=CL,
               variants={k: dict(n=len(v['rows']), info=v['info']) for k, v in S.V.items()}, **res)
    return out


# ----------------------------------------------------------------------------------------------------------- ParT
def run_part(per):
    from interp.part_core import ParTNetwork, kin, build, jmass, sub, cat, fof, read_root, CL, PK
    CL = list(CL); rng = np.random.default_rng(502); Js = []; NROW = 25000 if per <= 100 else 100000
    for c in CL:
        rows = np.sort(rng.choice(NROW, per, replace=False)); parts = []
        for s in range(0, NROW, 25000):
            sel = rows[(rows >= s) & (rows < s + 25000)] - s
            if len(sel) == 0: continue
            J = read_root(fof(c), start=s, stop=s + 25000); parts.append(sub(J, sel))
        Js.append(cat(parts)); log(f'read {c}: {per} random rows (first {rows[:3].tolist()}, max {rows.max()})')
    J0 = cat(Js); y = J0['label'].astype(int); N = len(y); del Js
    nets = {n: ParTNetwork(n) for n in ('full', 'kin')}
    run = lambda J, models=('full', 'kin'): {n: nets[n].run(J)['logits'].astype(np.float32) for n in models}
    pt, de, dp, ms = kin(J0); pt, de, dp, ms = (np.asarray(a, np.float64) for a in (pt, de, dp, ms))
    Jb = build(J0, pt, de, dp, ms); sgn = np.where(J0['jet_eta'] < 0, -1.0, 1.0)[:, None]; Jm = build(J0, pt, de * sgn, dp, ms)
    Lraw, L0, Lm = run(J0), run(Jb), run(Jm); checks = {}
    for n in ('full', 'kin'):
        checks[n] = dict(acc_raw=float((Lraw[n].argmax(1) == y).mean()), acc_rebuilt=float((L0[n].argmax(1) == y).mean()),
                         acc_mirrored_build=float((Lm[n].argmax(1) == y).mean()),
                         changed_by_corrected_rebuild=float((Lraw[n].argmax(1) != L0[n].argmax(1)).mean()),
                         changed_by_mirrored_build=float((Lraw[n].argmax(1) != Lm[n].argmax(1)).mean()),
                         own_class=[float((L0[n].argmax(1)[y == k] == k).mean()) for k in range(10)])
        log(f'ParT_{n}: {checks[n]}')
    checks['max_abs_dE_rebuild'] = float(np.abs(Jb['energy'] - J0['energy']).max()); del Jm, Lm
    log(f"rebuild max |ΔE| {checks['max_abs_dE_rebuild']:.3g} GeV")

    def sig(J):
        m = (J['mask'] > 0) & (J['charge'] != 0); return np.where(m, np.abs(J['d0val']) / np.maximum(J['d0err'], 1e-6), 0)

    def obs_of(J):
        p, e, f, _ = kin(J); O = classical_obs(p, e, f); O['nS3'] = (sig(J) > 3).sum(1); return O

    O0 = obs_of(Jb); log('classical observables (unedited) done')
    S = Store(y, L0, CL, O0, eta_sign=J0['jet_eta'].astype(np.float64))
    PAIRCL = ['QCD', 'Hbb', 'Zqq', 'Wqq', 'Tbqq']

    def add(name, rows, J1, models=('full', 'kin'), info=None):
        L = run(J1, models); O = None
        if np.isin(y[rows], [CL.index(c) for c in PAIRCL]).any(): O = obs_of(J1)
        S.add(name, rows, L, O, info)
        d = dict(n=len(rows), mass_median=float(np.median(jmass(J1)))); log(f'variant {name}: {d}')
        S.V[name]['info'].update(d)

    def subrows(rows): return sub(J0, rows), pt[rows], de[rows], dp[rows], ms[rows]
    allr = np.arange(N)
    for k in (() if WONLY else (0.8, 0.9, 1.1, 1.25)):
        add(f'boost x{k}', allr, build(J0, *boost(pt, de, dp, k), ms))
        add(f'rotation (boost x{k} displacement)', allr, build(J0, *rotate(pt, de, dp, k), ms))
    if not WONLY:
        (dE, pE), (dC, pC), ok, sE, sC = collapse_pair(pt, de, dp); r = allr[ok]
        info = dict(soft_pt_edit_median=float(np.median(sE[ok])), soft_pt_ctrl_median=float(np.median(sC[ok])),
                    log_ratio_median=float(np.median(np.log(sC[ok] / sE[ok]))))
        Jr, p_, d_, f_, m_ = subrows(r)
        add('decay collapse', r, build(Jr, p_, dE[ok], pE[ok], m_), info=info); add('control: matched-pT emission collapse', r, build(Jr, p_, dC[ok], pC[ok], m_), info=info)
        log(f'collapse info {info}')
    top = allr[y == CL.index('Tbqq')]; Jt, p_, d_, f_, m_ = subrows(top); TS = top_split(p_, d_, f_)
    log(f'top jets with a W pair: {int(TS["ok"].sum())}/{len(top)}')
    for t in (60.0, 80.4, 100.0, 120.0):
        (dE, pE), (dC, pC), ok, inf = wshift(p_, d_, f_, t, TS); info = {k: float(np.median(v[ok])) for k, v in inf.items()}
        Jo = sub(Jt, ok)
        add(f'W pair -> {t:g} GeV', top[ok], build(Jo, p_[ok], dE[ok], pE[ok], m_[ok]), info=info)
        add(f'control: non-W pair shifted by {t:g}-mW', top[ok], build(Jo, p_[ok], dC[ok], pC[ok], m_[ok]), info=info); log(f'W shift {t}: {info}')
    ok = TS['ok']; Jo = sub(Jt, ok); a = TS['a'][ok]
    ratio = TS['ptj'][ok][np.arange(ok.sum()), TS['lightW'][ok]] / TS['ptj'][ok][np.arange(ok.sum()), TS['b'][ok]]
    info = dict(light_over_b_pt_median=float(np.median(ratio)))
    add('remove b (recentred, boosted back)', top[ok], build(Jo, *drop_prong(p_[ok], d_[ok], f_[ok], a, TS['b'][ok]), m_[ok]), info=info)
    add('control: remove the pT-matched W daughter', top[ok], build(Jo, *drop_prong(p_[ok], d_[ok], f_[ok], a, TS['lightW'][ok]), m_[ok]), info=info)

    # ---- ParT_full: tracks and particle ID
    F = ('full',)
    if WONLY: return part_summary(S, y, CL, O0, L0, checks, per, N, [], [])
    hf = allr[np.isin(y, [CL.index(c) for c in ('QCD', 'Hbb', 'Hcc', 'Tbqq')])]; Jh = sub(J0, hf); Sh = sig(Jh)

    def scale_disp(J, s, keep_top=None):
        E = dict(J); Sg = sig(J); d = Sg > 2
        if keep_top is not None:
            rk = np.argsort(-Sg, 1).argsort(1); f = np.where(d & (rk < keep_top), 1.0, np.where(d, 0.0, 1.0))
        else: f = np.where(d, s, 1.0)
        E['d0val'] = (J['d0val'] * f).astype(np.float32); E['dzval'] = (J['dzval'] * f).astype(np.float32); return E
    nS3 = lambda J: (sig(J) > 3).sum(1)
    for name, J1 in (('displaced made prompt (d0, dz x0)', scale_disp(Jh, 0.0)), ('lifetime x0.35 (count at S>2 kept)', scale_disp(Jh, 0.35)),
                     ('lifetime x2.5 (count at S>2 kept)', scale_disp(Jh, 2.5)), ('keep 2 most displaced', scale_disp(Jh, None, keep_top=2))):
        add(name, hf, J1, F, info={f'nS3_mean_{c}': [float(nS3(Jh)[y[hf] == CL.index(c)].mean()), float(nS3(J1)[y[hf] == CL.index(c)].mean())] for c in ('QCD', 'Hbb', 'Hcc', 'Tbqq')})
    # drop displaced tracks vs drop pT-matched prompt tracks
    p_, d_, f_, m_ = (np.asarray(v, np.float64) for v in kin(Jh)); dropE = np.zeros(p_.shape, bool); dropC = np.zeros(p_.shape, bool); okd = np.zeros(len(hf), bool)
    for r_ in range(len(hf)):
        disp = np.flatnonzero((Sh[r_] > 2) & (p_[r_] > 0))
        if len(disp) == 0: continue
        cand = list(np.flatnonzero((Jh['charge'][r_] != 0) & (Sh[r_] < 2) & (p_[r_] > 0))); sel = []
        for q in disp:
            if not cand: break
            t_ = min(cand, key=lambda u: abs(p_[r_, u] - p_[r_, q])); sel.append(t_); cand.remove(t_)
        if len(sel) < len(disp): continue
        dropE[r_, disp] = True; dropC[r_, sel] = True; okd[r_] = True
    Jo = sub(Jh, okd); info = dict(n_dropped_mean=float(dropE[okd].sum(1).mean()), pt_dropped_edit_mean=float((p_ * dropE)[okd].sum(1).mean()),
                                   pt_dropped_ctrl_mean=float((p_ * dropC)[okd].sum(1).mean()))
    add('drop displaced tracks', hf[okd], build(Jo, np.where(dropE, 0, p_)[okd], d_[okd], f_[okd], m_[okd]), F, info=info)
    add('control: drop as many pT-matched prompt tracks', hf[okd], build(Jo, np.where(dropC, 0, p_)[okd], d_[okd], f_[okd], m_[okd]), F, info=info)
    log(f'drop-tracks info {info}')

    # ---- leptons
    lc = allr[np.isin(y, [CL.index('Hqql'), CL.index('Tbl')])]; Jl0 = sub(J0, lc)
    lep = ((Jl0['isElectron'] > 0) | (Jl0['isMuon'] > 0)) & (Jl0['mask'] > 0); has = lep.any(1); lc = lc[has]; Jl = sub(Jl0, has); del Jl0
    pl, dl, fl, ml = (np.asarray(v, np.float64) for v in kin(Jl)); lepm = ((Jl['isElectron'] > 0) | (Jl['isMuon'] > 0)) & (Jl['mask'] > 0)
    L_ = np.where(lepm, pl, -1).argmax(1); chh = (Jl['isChargedHadron'] > 0) & (Jl['mask'] > 0)

    def deiso(centre, random_dir=False):
        de1, dp1 = dl.copy(), fl.copy(); ok_ = np.zeros(len(lc), bool)
        for r_ in range(len(lc)):
            c_ = centre[r_]
            if c_ < 0: continue
            dr = np.hypot(dl[r_] - dl[r_, c_], fl[r_] - fl[r_, c_]); cand = np.flatnonzero((pl[r_] > 0) & (dr >= 0.1)); cand = cand[cand != L_[r_]]
            for q in cand[np.argsort(dr[cand])][:4]:
                ux, uy = (dl[r_, q] - dl[r_, c_]) / dr[q], (fl[r_, q] - fl[r_, c_]) / dr[q]; st = dr[q] - 0.03
                if random_dir: th = RNG.uniform(0, 2 * np.pi); ux, uy = np.cos(th), np.sin(th)
                de1[r_, q] -= ux * st; dp1[r_, q] -= uy * st
            ok_[r_] = len(cand) >= 4
        return de1, dp1, ok_
    hadc = np.array([(lambda h: h[np.argmin(np.abs(pl[r_, h] - pl[r_, L_[r_]]))] if len(h) else -1)(np.flatnonzero(chh[r_])) for r_ in range(len(lc))])
    e1, f1, okA = deiso(L_); e2, f2, okB = deiso(hadc); e3, f3, okC = deiso(L_, True); okI = okA & okB & okC & (hadc >= 0)
    Jo = sub(Jl, okI)
    add('lepton de-isolated (4 nearest to dR 0.03)', lc[okI], build(Jo, pl[okI], e1[okI], f1[okI], ml[okI]), F)
    add('control: same move around the pT-matched charged hadron', lc[okI], build(Jo, pl[okI], e2[okI], f2[okI], ml[okI]), F)
    add('control: same particles moved in random directions', lc[okI], build(Jo, pl[okI], e3[okI], f3[okI], ml[okI]), F)

    def flip(idx):
        E = dict(Jl); q = Jl['charge'].copy(); d0 = Jl['d0val'].copy(); r_ = np.flatnonzero(idx >= 0)
        q[r_, idx[r_]] *= -1; d0[r_, idx[r_]] *= -1; E['charge'] = q; E['d0val'] = d0; return E
    rank_h = np.array([(lambda h: h[np.argmin(np.abs(h - L_[r_]))] if len(h) else -1)(np.flatnonzero(chh[r_])) for r_ in range(len(lc))])
    okF = rank_h >= 0
    add('lepton charge flipped (and d0 sign)', lc[okF], sub(flip(L_), okF), F)
    add('control: charged hadron of nearest pT rank flipped', lc[okF], sub(flip(rank_h), okF), F)

    # ---- pi0 merge
    pc = allr[np.isin(y, [CL.index(c) for c in ('QCD', 'Hbb', 'Wqq', 'Tbqq')])]; Jp = sub(J0, pc)
    pp, dpp, fpp, mpp = (np.asarray(v, np.float64) for v in kin(Jp)); ph = (Jp['isPhoton'] > 0) & (Jp['mask'] > 0)
    PE = [pp.copy(), dpp.copy(), fpp.copy()]; PC = [pp.copy(), dpp.copy(), fpp.copy()]; okP = np.zeros(len(pc), bool); nm = []

    def merge(A, r_, i, j):
        P_, D_, F_ = A; s_ = P_[r_, i] + P_[r_, j]; h, l = (i, j) if P_[r_, i] >= P_[r_, j] else (j, i)
        D_[r_, h] = (P_[r_, i] * D_[r_, i] + P_[r_, j] * D_[r_, j]) / s_; F_[r_, h] = (P_[r_, i] * F_[r_, i] + P_[r_, j] * F_[r_, j]) / s_
        P_[r_, h] = s_; P_[r_, l] = 0.0
    for r_ in range(len(pc)):
        g = np.flatnonzero(ph[r_])
        if len(g) < 4: continue
        a_, b_ = np.triu_indices(len(g), 1); i_, j_ = g[a_], g[b_]
        mm = np.sqrt(np.maximum(2 * pp[r_, i_] * pp[r_, j_] * (np.cosh(dpp[r_, i_] - dpp[r_, j_]) - np.cos(fpp[r_, i_] - fpp[r_, j_])), 0))
        dr = np.hypot(dpp[r_, i_] - dpp[r_, j_], fpp[r_, i_] - fpp[r_, j_])

        def greedy(mask, order):
            used, out = set(), []
            for k in order[mask[order]]:
                if i_[k] not in used and j_[k] not in used: out.append(k); used |= {i_[k], j_[k]}
            return out
        e_ = greedy((mm > 0.085) & (mm < 0.185), np.argsort(np.abs(mm - 0.135))); c_ = greedy((mm > 0.25) & (mm < 0.45), np.argsort(dr))[:len(e_)]
        if len(e_) == 0 or len(c_) < len(e_): continue
        for k in e_: merge(PE, r_, i_[k], j_[k])
        for k in c_: merge(PC, r_, i_[k], j_[k])
        okP[r_] = True; nm.append(len(e_))
    Jo = sub(Jp, okP); info = dict(pairs_merged_mean=float(np.mean(nm)) if nm else 0.0)
    add('pi0 merge (photon pairs 85-185 MeV merged)', pc[okP], build(Jo, PE[0][okP], PE[1][okP], PE[2][okP], mpp[okP]), F, info=info)
    add('control: as many 250-450 MeV photon pairs merged', pc[okP], build(Jo, PC[0][okP], PC[1][okP], PC[2][okP], mpp[okP]), F, info=info)
    log(f'pi0 info {info}')

    # ---- dead cone
    dc = allr[np.isin(y, [CL.index('Hbb'), CL.index('Hcc')])]; Jd = sub(J0, dc); pd_, dd_, fd_, md_ = (np.asarray(v, np.float64) for v in kin(Jd))
    ax, dist_ = _kmeans(pd_, dd_, fd_, 2); asg = dist_.argmin(-1); asg[pd_ <= 0] = -1; Sd = sig(Jd) > 3
    th6 = RNG.uniform(0, 2 * np.pi, (len(dc), 2, 6)); rin = RNG.uniform(0, 0.015, (len(dc), 2, 6)); rout = RNG.uniform(0.05, 0.08, (len(dc), 2, 6))
    cen = np.full((len(dc), 2, 2), np.nan)
    for j in range(2):
        w = pd_ * (Sd & (asg == j)); n_ = (Sd & (asg == j)).sum(1); okj = n_ >= 2
        c2 = (w[..., None] * np.stack([dd_, fd_], -1)).sum(1) / np.maximum(w.sum(1), 1e-9)[:, None]; cen[okj, j] = c2[okj]
    okD = ~np.isnan(cen).all((1, 2))

    def fill(rr):
        K = 12; x = {k: np.concatenate([Jd[k], np.zeros((len(dc), K), np.float32)], 1) for k in PK}
        P2, D2, F2, M2 = (np.concatenate([v, np.zeros((len(dc), K))], 1) for v in (pd_, dd_, fd_, md_))
        for j in range(2):
            v_ = ~np.isnan(cen[:, j, 0]); sl = slice(128 + 6 * j, 128 + 6 * j + 6)
            P2[v_, sl] = 1.0; D2[v_, sl] = cen[v_, j, 0][:, None] + rr[v_, j] * np.cos(th6[v_, j]); F2[v_, sl] = cen[v_, j, 1][:, None] + rr[v_, j] * np.sin(th6[v_, j])
            x['isPhoton'][v_, sl] = 1; x['mask'][v_, sl] = 1
        return build(Jd, P2, D2, F2, M2, x)
    add('dead cone filled (6 x 1 GeV photons within dR 0.015 of each b/c)', dc[okD], sub(fill(rin), okD), F)
    add('control: the same photons at dR 0.05-0.08', dc[okD], sub(fill(rout), okD), F)

    # ---- comparisons
    common = []
    for k in (0.8, 0.9, 1.1, 1.25): common.append((f'boost x{k}', f'rotation (boost x{k} displacement)', CL, 'own'))
    common.append(('decay collapse', 'control: matched-pT emission collapse', CL, 'own'))
    for t in (60.0, 80.4, 100.0, 120.0): common.append((f'W pair -> {t:g} GeV', f'control: non-W pair shifted by {t:g}-mW', ['Tbqq'], 'own'))
    common.append(('remove b (recentred, boosted back)', 'control: remove the pT-matched W daughter', ['Tbqq'], ('Wqq', 'Zqq')))
    full_only = [('drop displaced tracks', 'control: drop as many pT-matched prompt tracks', ['QCD', 'Hbb', 'Hcc', 'Tbqq'], 'own'),
                 ('drop displaced tracks', 'lifetime x0.35 (count at S>2 kept)', ['Hbb'], 'own'),
                 ('keep 2 most displaced', 'lifetime x0.35 (count at S>2 kept)', ['Hbb'], 'own'),
                 ('displaced made prompt (d0, dz x0)', 'lifetime x0.35 (count at S>2 kept)', ['Hbb', 'Hcc', 'Tbqq'], 'own'),
                 ('lifetime x2.5 (count at S>2 kept)', None, ['Hcc'], 'own'),
                 ('lepton de-isolated (4 nearest to dR 0.03)', 'control: same move around the pT-matched charged hadron', ['Hqql', 'Tbl'], 'own'),
                 ('lepton de-isolated (4 nearest to dR 0.03)', 'control: same particles moved in random directions', ['Hqql', 'Tbl'], 'own'),
                 ('lepton charge flipped (and d0 sign)', 'control: charged hadron of nearest pT rank flipped', ['Hqql', 'Tbl'], 'own'),
                 ('pi0 merge (photon pairs 85-185 MeV merged)', 'control: as many 250-450 MeV photon pairs merged', ['QCD', 'Hbb', 'Wqq', 'Tbqq'], 'own'),
                 ('dead cone filled (6 x 1 GeV photons within dR 0.015 of each b/c)', 'control: the same photons at dR 0.05-0.08', ['Hbb', 'Hcc'], 'own')]
    return part_summary(S, y, CL, O0, L0, checks, per, N, common, full_only)


def part_summary(S, y, CL, O0, L0, checks, per, N, common, full_only):
    if WONLY:
        common = [(f'W pair -> {t:g} GeV', f'control: non-W pair shifted by {t:g}-mW', ['Tbqq'], 'own') for t in (60.0, 80.4, 100.0, 120.0)]
    pairs = [('Wqq', ['QCD'], 'V'), ('Zqq', ['QCD'], 'V'), ('Tbqq', ['QCD'], 't'), ('Hbb', ['QCD'], 'bb')]
    wnames = {f'W pair -> {t:g} GeV' for t in (60.0, 80.4, 100.0, 120.0)}
    out = dict(sample=dict(per_class=per, seed=502, N=N, rows='random rows of each class file _100 (100,000 jets)', build='eta = jet_eta + sign(jet_eta)*deta'),
               checks=checks, classes=CL, variants={k: dict(n=len(v['rows']), info=v['info']) for k, v in S.V.items()}, models={})
    for n in ('full', 'kin'):
        S_n = Store(y, {n: L0[n]}, CL, O0, S.eta_sign); S_n.V = {k: v for k, v in S.V.items() if n in v['L']}
        res = summarise(S_n, n, common + (full_only if n == 'full' else []), pairs, split_eta_edits=wnames)
        log(f'==== ParT_{n}')
        for r in res['comparisons']: log(fmt(r, CL))
        for k, v in res['rejection'].items():
            log(f"rej {k}: " + '; '.join(f"{lab} R {v[lab]['R_before']:.1f} -> {v[lab]['R_after']:.1f} (ratio CI {v[lab]['ratio_ci'][0]:.2f}..{v[lab]['ratio_ci'][1]:.2f})" for lab in ('eps50', 'eps_classical') if lab in v)
                + f"; classical eps_S {v['classical']['eps_S']}, eps_B {v['classical']['eps_B']}")
        out['models'][n] = res
    return out


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    return o


def main():
    global ITERS, TOL, WONLY
    which = sys.argv[1]; smoke = 'smoke' in sys.argv[2:]; per = 100 if smoke else 4000
    if 'wshift' in sys.argv[2:]: ITERS, TOL, WONLY = 30, 2.0, True
    RES.mkdir(parents=True, exist_ok=True)
    out = run_jedi(per) if which == 'jedi' else run_part(per)
    out['smoke'] = smoke; out['wshift_mode'] = dict(iters=ITERS, tol_gev=TOL, only_wshift=WONLY)
    f = RES / f"h3_{which}{'_wshift' if WONLY else ''}{'_smoke' if smoke else ''}.json"
    f.write_text(json.dumps(jsonable(out), indent=1)); log(f'wrote {f}; H3 DONE')


if __name__ == '__main__':
    main()
