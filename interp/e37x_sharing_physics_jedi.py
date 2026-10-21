"""E37x (claim A4s v2, D18; stage 3): what does JEDI-linear's h4 prong-sharing coordinate compute, and what carries the
Z-jet response to a more lopsided prong split?

Network: official JEDI-linear 3-feature n64 (fixed point), exact tail from h4 with the next layer's range clip (x10 JNet).
Samples: test T = W+Z jets of interp.jedi_core.sample(3000, 3737) (6,000 jets); fit F = W+Z jets of sample(3000, 3738) minus
  rows in T (refit of the sharing direction, natural slopes).  5-class accuracy on all 15,000 stored T jets.
Prongs: two exclusive-kT subjets (p = 1, E-scheme) of all massless constituents (E35 kt2_labels).  z = z_h = vector-pT
  fraction of the harder prong (z_h = 1 - z_min); Delta R12 = rapidity-azimuth distance of the two prong four-vectors.
The coordinate: u = h4 . v.  v = 1-dim DAS direction refitted on F exactly as E37 (D(dc) self + cross interchange pairs, MSE
  on the W-Z margin, 400 Adam steps; E37 code); alternative v_dm = difference of means h4(D+0.3) - h4(D-0.3) on F (E37v).
  Both oriented so that corr(u, z) > 0 on unedited F jets.  Cosine with the stored E37 h4 k=1 direction reported.
Edits on T (all massless constituents keep their prong label; network input rebuilt about each edited jet's own E-scheme axis,
  pT-sorted; H45/E37 model_x):
  Ez(dz)  z_h -> z_h + dz, dz in DZS (H50 arm C with dz_soft = -dz): constituent pT of each prong scaled so that the prong
          vector pTs become z_h W and (1 - z_h) W (W = sum of the prong vector pTs, unchanged); each prong's offsets from its own
          axis rescaled (bisection) so the prong mass equals its original value; each prong translated back onto its original
          axis (Delta R12 exact); then the whole jet translated in (y, phi) so the jet axis is at its original position.
          Kept exactly: Delta R12, both prong masses, W.  NOT kept: the jet mass.  At fixed prong masses, Delta R12 and pT the
          jet mass is fixed by m^2 ~ m_a^2/z + m_b^2/(1-z) + z(1-z) pT^2 Delta R^2, so z cannot change at fixed (pT, m, Delta R12,
          prong masses); the controls below are therefore matched to Ez's jet-mass change.
  Edr(dz) Delta R12 changed at fixed z and prong masses: both prongs translated rigidly along their joining line (H50 separate),
          scale lambda solved so the jet mass equals m(Ez(dz)).  Same Delta m as Ez, z unchanged.
  Em(dz)  all angles dilated about the jet axis by k (bisection) so the jet mass equals m(Ez(dz)): a real mass change of the same
          size (z unchanged; Delta R12 and prong masses x k).
  Emf(f)  fixed dilations f in EMF (calibration of the read-out per unit log m).
  Erot(dz) rigid rotation of the jet about its axis in (y, phi) by theta (random sign), 2 sin(theta/2) <r>_pT = Ez's pT-weighted
          mean constituent displacement: z, Delta R12, prong masses, m, pT all exact (matched no-information edit).
  D(+-0.3) E35's rest-frame decay-axis rotation at fixed jet four-momentum (z and Delta R12 co-move at fixed m).
  Acceptance: Ez and Edr: reclustering (kT-2) moves < 1 % of the jet scalar pT between prongs and both prong centres within
  Delta R 0.75 of the axis; Edr / Em bisections converge.  A jet enters dz only if Ez, Edr, Em are all accepted.
Read-outs per edit, by unedited jet-mass bin BINS and class: mean Delta u; W-Z logit margin change; W->Z / Z->W call flips.
  Calibrated slope R_z = mean[Delta u(Ez) - Delta u(Em)] / (beta_z * mean Delta z), beta_z = natural partial slope of u on z
  in unedited F jets after k-NN (K = 50) residualisation on (log m, log pT) (u and z each minus their neighbour means), per bin
  for the per-bin ratio; R_Edr = mean[Delta u(Edr) - Delta u(Em)] / (same denominator); R_rot = mean Delta u(Erot) / (same).
  Pooled over dz with the sign of dz.  95 % CI: 2,000 jet bootstrap resamples of T (beta resampled over F jointly).
Interaction model: rows = (jet, edit) for all edits; Delta margin ~ 1 + a Delta u + b Delta log m + c Delta u (m0 - 85)/10;
  per class and pooled; 2,000 jet bootstraps; H0 c = 0 (two-sided bootstrap p).  Per-bin slopes a_bin (with b).
Physics naming: for q in Z_QUANT, natural OLS on F: u ~ 1 + q + log m + (log m)^2 + log pT; edit rows Ez, Edr, Em, Emf, D:
  predicted Delta u = beta_q Delta q + mass part; calibrated slope s_q = sum(Delta u_net beta_q Delta q) / sum((beta_q Delta q)^2),
  Delta u_net = Delta u - mass part; edit R^2; held-out partial R^2 on T; per-edit-type s_q.
Carriers (holds at h4 on T): edits D(+0.3) and Ez(+0.15).  hold(Q): h' = h_e + (h_0 - h_e) Q Q^T (Q kept at its unedited
  value); removed share = 1 - mean dmargin(held) / mean dmargin(free), per class.  Sets: (i) v; (ii) M2 = E37's refitted M2
  DAS subspace at h4 (k = 2, X10/E25w-style do(M2) interchange); (iii) v + M2 (3 dims); (iv) copy: h' = h_0 + (h_e - h_0) Q Q^T
  for v + M2 (sufficiency; reproduced share = mean dmargin(copy) / mean dmargin(free)).  Nulls: random subspaces of the
  live h4 units (k = 1, 2, 3; E26v rule); rank p = (1 + #random >= observed) / (1 + n_random); Holm over 4 sets x 2 classes x 2 edits.
  Nulls: 2,000 random subspaces per dimension (debug 1: 200 floored the Holm-adjusted p at 16/201 = 0.08).
  McNemar on paired call changes (free vs hold).  v_dm holds reported outside the Holm family.
Floor: untrained redraw (Jedi(untrained=True, seed=0), calls argmax(logits - mean logits)): v_dm, natural partial R^2 of each q,
  R_z, R_Edr, margin responses in margin-SD units.
Run: ./job.sh e37x-run 1 2 'python -m interp.e37x_sharing_physics_jedi'  (--smoke K: K jets per class, fewer steps / nulls)
Writes $AI_RESULTS/jedi/e37x[_smoke].json.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import e37_sharing_das_jedi as E  # noqa: E402
from interp import h50_z_vs_dr as H50  # noqa: E402
from interp.h45_prong_sharing import rap, wrap, axis, prong_offsets_ok, label_sep  # noqa: E402
from interp.e35_prong_sharing_route_jedi import kt2_labels  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time()
TSEED, FSEED = 3737, 3738
DZS = (-0.15, -0.075, 0.075, 0.15)
EMF = (0.95, 0.98, 1.02, 1.05)
BINS = ((65, 75), (75, 85), (85, 95), (95, 105))
MIGMAX = 0.01
QN = ['z', 'zz', 'dR', 'pmr', 'D2', 'tau21', 'm', 'pT', 'm_hi', 'm_lo']
Z_QUANT = ['z', 'zz', 'dR', 'pmr', 'D2', 'tau21']
QDESC = dict(z='z_h (harder-prong pT fraction)', zz='z(1-z)', dR='Delta R12', pmr='m_lo/m_hi (prong-mass ratio)', D2='D2 (beta 1)', tau21='tau21 (WTA-approx)')
KNN = 50


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


jsonable = E.jsonable
qci = E.qci
margin = E.margin


# ------------------------------------------------------------------ per-jet edits
def obs1(P, lab):
    o = E.jet_obs(P, lab)   # z_min, D2, tau21, m_hi, m_lo, pTD, lead, m, pT
    zh = 1.0 - o[0]
    return np.array([zh, zh * (1 - zh), label_sep(P, lab), o[4] / max(o[3], 1e-9), o[1], o[2], o[7], o[8], o[3], o[4]])


def jet_variants(j, P, lab, seed):
    pt, y, phi, _ = H50.kin(P); m = np.zeros_like(pt)
    T = P.sum(0); ya, pa = float(rap(T)), float(np.arctan2(T[1], T[0]))
    M0 = [H50.pmass(pt, y, phi, m, lab == k) for k in (0, 1)]
    rng = np.random.default_rng([seed, int(j)]); out = {}; info = {}
    r_ = np.hypot(y - ya, wrap(phi - pa)); mr = float((pt * r_).sum() / pt.sum())

    def dil(f): return ya + f * (y - ya), pa + f * wrap(phi - pa)

    def rot(th):
        dy, dp = y - ya, wrap(phi - pa); c, s = math.cos(th), math.sin(th); return ya + c * dy - s * dp, pa + s * dy + c * dp
    for f in EMF: out[('Emf', f)] = H50.build(pt, *dil(f), m)
    for dz in DZS:
        r, inf = H50.arm_c(pt, y, phi, m, lab, -dz, M0)
        if r is None: info[('Ez', dz)] = inf['reason']; continue
        pt2, y2, p2 = r
        V = H50.build(pt2, y2, p2, m).sum(0); y2 = y2 + (ya - float(rap(V))); p2 = p2 + float(wrap(pa - np.arctan2(V[1], V[0])))
        QZ = H50.build(pt2, y2, p2, m); mZ = H50.mass4(QZ.sum(0))
        rs, lam = H50.mass_by_sep(pt, y, phi, m, lab, mZ)
        if rs is None: info[('Ez', dz)] = 'dr_bisect'; continue
        QR = H50.build(pt, *rs, m)
        k = H50.bisect(lambda q: H50.jm(pt, *dil(q), m), 0.2, 5.0, mZ)
        if k is None: info[('Ez', dz)] = 'dil_bisect'; continue
        QM = H50.build(pt, *dil(k), m)
        disp = float((pt * np.hypot(y2 - y, wrap(p2 - phi))).sum() / pt.sum())
        th = 2 * math.asin(min(1.0, disp / max(2 * mr, 1e-9))) * float(rng.choice([-1.0, 1.0]))
        QT = H50.build(pt, *rot(th), m)
        ok = True; mig = []
        for Q in (QZ, QR):
            L2 = kt2_labels(Q); mg = H50.migration(lab, L2, np.hypot(Q[:, 0], Q[:, 1])); mig.append(mg)
            ok &= (mg < MIGMAX) and prong_offsets_ok(Q, lab)
        info[('Ez', dz)] = 'ok' if ok else 'migration_or_offset'
        info[('par', dz)] = dict(zsoft0=inf['z0'], zsoft1=inf['z1'], lam=lam, k=k, th=th, migZ=mig[0], migR=mig[1])
        if ok:
            out[('Ez', dz)] = QZ; out[('Edr', dz)] = QR; out[('Em', dz)] = QM; out[('Erot', dz)] = QT
    return out, info


def build_variants(DT, seed, sm):
    """Ez / Edr / Em / Erot (per dz), Emf (per f), D(+-0.3) on the T jets: model inputs, observables, acceptance"""
    P4, nn, LAB = DT['P4'], DT['nn'], DT['LAB']; J = len(P4)
    keys = [(k, dz) for dz in DZS for k in ('Ez', 'Edr', 'Em', 'Erot')] + [('Emf', f) for f in EMF]
    PV = {k: P4.copy() for k in keys}; ACC = {k: np.zeros(J, bool) for k in keys}; OB = {k: np.full((J, len(QN)), np.nan) for k in keys}
    O0 = np.full((J, len(QN)), np.nan); PAR = {dz: {nm: np.full(J, np.nan) for nm in ('zsoft0', 'zsoft1', 'lam', 'k', 'th', 'migZ', 'migR')} for dz in DZS}
    reasons = {dz: {} for dz in DZS}; t = time.time(); chk = {}
    for j in range(J):
        n = int(nn[j])
        if n < 3: continue
        P = P4[j, :n]; lab = LAB[j, :n]; O0[j] = obs1(P, lab)
        out, info = jet_variants(j, P, lab, seed)
        for dz in DZS:
            rr = info.get(('Ez', dz), 'none'); reasons[dz][rr] = reasons[dz].get(rr, 0) + 1
            if ('par', dz) in info:
                for nm, v in info[('par', dz)].items(): PAR[dz][nm][j] = v
        for k, Q in out.items():
            PV[k][j, :n] = Q; ACC[k][j] = True; OB[k][j] = obs1(Q, lab)
        if ('Ez', 0.15) in out:   # invariants of the edit (float64)
            Ob = OB[('Ez', 0.15)][j]
            for nm_, v_ in (('Ez_dR12_abs', abs(Ob[2] - O0[j][2])), ('Ez_prong_mass_rel_to_m', max(abs(Ob[8] - O0[j][8]), abs(Ob[9] - O0[j][9])) / max(O0[j][6], 1)),
                            ('Edr_Em_mass_match_rel', max(abs(OB[('Edr', 0.15)][j][6] - Ob[6]), abs(OB[('Em', 0.15)][j][6] - Ob[6])) / Ob[6]),
                            ('Edr_z_abs', abs(OB[('Edr', 0.15)][j][0] - O0[j][0])), ('Erot_mass_rel', abs(OB[('Erot', 0.15)][j][6] - O0[j][6]) / O0[j][6]),
                            ('Erot_z_abs', abs(OB[('Erot', 0.15)][j][0] - O0[j][0])), ('Erot_dR12_abs', abs(OB[('Erot', 0.15)][j][2] - O0[j][2]))):
                chk[nm_] = max(chk.get(nm_, 0.0), float(v_))
        if j % 1000 == 0: log(f'  variants: jet {j}/{J} [{time.time() - t:.0f}s]')
    log(f'  variants built in {time.time() - t:.0f}s; reasons ' + json.dumps({str(dz): reasons[dz] for dz in DZS}) +
        f'; max invariant errors (Ez/Edr/Erot +0.15) {json.dumps(chk)}')
    X = {}
    for k in keys:
        X[k] = E.model_xn(PV[k], nn, axis(PV[k], nn))[0]
    ex_P = {k: PV[k] for k in (('Ez', 0.15), ('Edr', 0.15), ('Em', 0.15))}
    del PV
    for dc in (-0.3, 0.3):
        k = ('D', dc); Pe = DT['PE'][(dc, 'D')]; ACC[k] = DT['acc'][dc].copy(); X[k] = DT['X'][(dc, 'D')]
        OB[k] = np.full((J, len(QN)), np.nan)
        for j in np.flatnonzero(ACC[k]): OB[k][j] = obs1(Pe[j, :nn[j]], LAB[j, :nn[j]])
    log(f'  observables of D(+-0.3) done; accepted ' + ', '.join(f'{k[0]}{k[1]:+}: {ACC[k].mean():.3f}' for k in ACC))
    return X, ACC, OB, O0, PAR, reasons, chk, ex_P


# ------------------------------------------------------------------ statistics helpers
def bweights(J, B, seed):
    rng = np.random.default_rng(seed); return np.stack([np.bincount(rng.integers(0, J, J), minlength=J) for _ in range(B)]).astype(np.float32)


def bmean(W, v, s):
    Ws = W[:, s]; return (Ws @ v[s]) / np.maximum(Ws.sum(1), 1)


def mstat(W, v, sel):
    s = np.flatnonzero(sel & np.isfinite(v))
    if len(s) < 5: return dict(n=int(len(s)))
    return dict(n=int(len(s)), mean=float(v[s].mean()), ci=qci(bmean(W, v, s)))


def pboot(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))) if len(b) else None


def mcn(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum()); return dict(first_only=n10, second_only=n01, p=E.binom_p(n10, n10 + n01))


def knn_resid(F, vals, K=KNN):
    """k-NN (on standardised (log m, log pT)) residuals of each column of vals; F (J, 2)"""
    from sklearn.neighbors import NearestNeighbors
    Z = (F - F.mean(0)) / F.std(0); nb = NearestNeighbors(n_neighbors=K + 1).fit(Z); _, idx = nb.kneighbors(Z); idx = idx[:, 1:]
    return [v - v[idx].mean(1) for v in vals], nb, idx


def ols(X, y, w=None):
    if w is None: return np.linalg.lstsq(X, y, rcond=None)[0]
    sw = np.sqrt(w); return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def natX(O, qi=None):
    lm = np.log(O[:, QN.index('m')]); lp = np.log(O[:, QN.index('pT')])
    cols = [np.ones(len(O)), lm, lm ** 2, lp] + ([O[:, qi]] if qi is not None else [])
    return np.stack(cols, 1)


# ------------------------------------------------------------------ analyses
def readout_core(u0F, OF, u0T, OT, DU, ACC, OB, y, W, WF, mT, tag):
    """calibrated slopes R_z / R_Edr / R_rot (k-NN natural slope), per bin and pooled; naming of the quantity"""
    res = dict(tag=tag)
    okF = np.isfinite(OF).all(1) & (OF[:, QN.index('m')] > 1); okT = np.isfinite(OT).all(1) & (OT[:, QN.index('m')] > 1)
    Fi = np.flatnonzero(okF); feat = np.c_[np.log(OF[Fi, QN.index('m')]), np.log(OF[Fi, QN.index('pT')])]
    iz, idr = QN.index('z'), QN.index('dR')
    (ru, rz, rd), _, _ = knn_resid(feat, [u0F[Fi], OF[Fi, iz], OF[Fi, idr]])
    mF = OF[Fi, QN.index('m')]; WFi = WF[:, Fi]
    def beta(rq, sel):
        s = np.flatnonzero(sel); b = float((ru[s] * rq[s]).sum() / (rq[s] ** 2).sum())
        bb = (WFi[:, s] @ (ru[s] * rq[s])) / (WFi[:, s] @ (rq[s] ** 2)); return b, bb
    BZ = {'all': beta(rz, np.ones(len(Fi), bool))}; BD = {'all': beta(rd, np.ones(len(Fi), bool))}
    for lo, hi in BINS:
        BZ[f'{lo}-{hi}'] = beta(rz, (mF >= lo) & (mF < hi)); BD[f'{lo}-{hi}'] = beta(rd, (mF >= lo) & (mF < hi))
    res['beta_z_knn'] = {k: dict(beta=v[0], ci=qci(v[1])) for k, v in BZ.items()}
    res['beta_dR_knn'] = {k: dict(beta=v[0], ci=qci(v[1])) for k, v in BD.items()}
    res['natural_partial_corr_u_z_knn'] = float(np.corrcoef(ru, rz)[0, 1]); res['natural_partial_corr_u_dR_knn'] = float(np.corrcoef(ru, rd)[0, 1])
    res['natural_corr_z_dR_knn_resid'] = float(np.corrcoef(rz, rd)[0, 1])
    # calibrated slopes
    cal = {}
    for bn in ['all'] + [f'{lo}-{hi}' for lo, hi in BINS]:
        if bn == 'all': inb = okT.copy()
        else:
            lo, hi = map(float, bn.split('-')); inb = okT & (mT >= lo) & (mT < hi)
        for cls, cm in (('W', y == 2), ('Z', y == 3), ('WZ', np.ones(len(y), bool))):
            row = {}; numZ = numR = numT = den = 0.0; bnZ = bnR = bnT = bden = 0.0
            for dz in DZS:
                s = np.flatnonzero(inb & cm & ACC[('Ez', dz)])
                if len(s) < 10: continue
                dZ = DU[('Ez', dz)] - DU[('Em', dz)]; dR = DU[('Edr', dz)] - DU[('Em', dz)]; dT = DU[('Erot', dz)]
                dzr = OB[('Ez', dz)][:, iz] - OT[:, iz]; ddr = (OB[('Edr', dz)][:, idr] - OB[('Em', dz)][:, idr])
                b, bb = BZ[bn]; sg = np.sign(dz)
                mZ, mR, mT_, mz = dZ[s].mean(), dR[s].mean(), dT[s].mean(), dzr[s].mean()
                bZ_, bR_, bT_, bz_ = bmean(W, dZ, s), bmean(W, dR, s), bmean(W, dT, s), bmean(W, dzr, s)
                bdr, bbd = BD[bn]
                row[f'{dz:+}'] = dict(n=int(len(s)), dz_realised=float(mz), du_Ez=float(DU[('Ez', dz)][s].mean()), du_Em=float(DU[('Em', dz)][s].mean()),
                                     du_Edr=float(DU[('Edr', dz)][s].mean()), du_Erot=float(mT_),
                                     R_z=float(mZ / (b * mz)), R_z_ci=qci(bZ_ / (bb * bz_)), R_Edr=float(mR / (b * mz)), R_Edr_ci=qci(bR_ / (bb * bz_)),
                                     R_rot=float(mT_ / (b * mz)), R_rot_ci=qci(bT_ / (bb * bz_)),
                                     dR12_change_Edr_minus_Em=float(ddr[s].mean()),
                                     R_dR_Edr=float(mR / (bdr * ddr[s].mean())) if abs(ddr[s].mean()) > 1e-6 else None,
                                     R_dR_Edr_ci=qci(bR_ / (bbd * bmean(W, ddr, s))) if abs(ddr[s].mean()) > 1e-6 else None)
                numZ += sg * mZ; numR += sg * mR; numT += sg * mT_; den += sg * b * mz
                bnZ = bnZ + sg * bZ_; bnR = bnR + sg * bR_; bnT = bnT + sg * bT_; bden = bden + sg * bb * bz_
            if row:
                row['pooled'] = dict(R_z=float(numZ / den), R_z_ci=qci(bnZ / bden), R_Edr=float(numR / den), R_Edr_ci=qci(bnR / bden),
                                     R_rot=float(numT / den), R_rot_ci=qci(bnT / bden),
                                     R_Edr_over_R_z=float(numR / numZ), R_Edr_over_R_z_ci=qci(bnR / bnZ),
                                     p_R_Edr_eq_0=pboot(bnR / bden), p_R_z_minus_R_Edr=pboot(bnZ / bden - bnR / bden))
            cal[f'{bn}|{cls}'] = row
    res['calibrated'] = cal
    # naming: natural OLS model per quantity; edit rows
    kinds = [('Ez', dz) for dz in DZS] + [('Edr', dz) for dz in DZS] + [('Em', dz) for dz in DZS] + [('Emf', f) for f in EMF] + [('D', -0.3), ('D', 0.3)]
    nm_res = {}; J = len(y)
    for q in Z_QUANT:
        qi = QN.index(q); XF = natX(OF[okF], qi); bq = ols(XF, u0F[okF]); XT = natX(OT[okT], qi)
        b0 = ols(natX(OF[okF]), u0F[okF]); sse = ((u0T[okT] - XT @ bq) ** 2).sum(); sse0 = ((u0T[okT] - natX(OT[okT]) @ b0) ** 2).sum()
        sst = ((u0T[okT] - u0T[okT].mean()) ** 2).sum()
        r = dict(beta_q=float(bq[-1]), heldout_R2=float(1 - sse / sst), heldout_R2_mass_pt_only=float(1 - sse0 / sst), heldout_partial_R2=float(1 - sse / sse0))
        # per-jet sums for the bootstrap
        A = np.zeros(J); Bq = np.zeros(J); C = np.zeros(J); Dm = np.zeros(J); N_ = np.zeros(J); per_kind = {}
        for kd in kinds:
            s = np.flatnonzero(ACC[kd] & okT & np.isfinite(OB[kd]).all(1))
            if not len(s): continue
            Oe = OB[kd][s]; Oo = OT[s]
            dX = natX(Oe, qi) - natX(Oo, qi); mass_part = dX[:, 1:4] @ bq[1:4]; pq = dX[:, 4] * bq[4]
            dun = DU[kd][s] - mass_part
            np.add.at(A, s, dun * pq); np.add.at(Bq, s, pq * pq); np.add.at(C, s, (dun - pq) ** 2); np.add.at(Dm, s, DU[kd][s] ** 2); np.add.at(N_, s, 1)
            g = kd[0]; pk = per_kind.setdefault(g, [0.0, 0.0])
            pk[0] += float((dun * pq).sum()); pk[1] += float((pq * pq).sum())
        r['s_q'] = float(A.sum() / Bq.sum()) if Bq.sum() > 0 else None
        bs = (W @ A) / np.maximum(W @ Bq, 1e-30); r['s_q_ci'] = qci(bs)
        r['edit_R2_q_plus_mass'] = float(1 - C.sum() / Dm.sum())
        r['s_q_by_edit'] = {g: (v[0] / v[1] if v[1] > 0 else None) for g, v in per_kind.items()}
        r['abs_log_s'] = float(abs(math.log(r['s_q']))) if r['s_q'] and r['s_q'] > 0 else None
        nm_res[q] = r
    cand = [q for q in Z_QUANT if nm_res[q]['abs_log_s'] is not None and nm_res[q]['s_q_ci'][0] is not None and nm_res[q]['s_q_ci'][0] > 0]
    res['naming'] = dict(per_quantity=nm_res, closest_to_1=min(cand, key=lambda q: nm_res[q]['abs_log_s']) if cand else None,
                         rule='quantity with s_q CI > 0 and |log s_q| smallest; s_q = calibrated slope of the mass-corrected read-out change on beta_q Delta q over all edit rows')
    # dilation calibration of the mass slope (causal vs natural)
    lmd = []; dud = []
    for f in EMF:
        s = np.flatnonzero(ACC[('Emf', f)] & okT); lmd.append(np.log(OB[('Emf', f)][s, QN.index('m')] / OT[s, QN.index('m')])); dud.append(DU[('Emf', f)][s])
    lmd, dud = np.concatenate(lmd), np.concatenate(dud); res['dilation_gain_du_per_dlogm'] = float((lmd * dud).sum() / (lmd ** 2).sum())
    return res


def interaction(DU, DM, DLM, ACC, y, mT, W, kinds):
    """Delta margin ~ 1 + a du + b dlogm + c du (m0 - 85)/10; per-jet normal equations, bootstrap over jets"""
    J = len(y); mc = (mT - 85.0) / 10.0; out = {}
    for cls, cm in (('W', y == 2), ('Z', y == 3), ('WZ', (y == 2) | (y == 3))):
        XtX = np.zeros((J, 4, 4)); Xty = np.zeros((J, 4)); n = 0
        for kd in kinds:
            s = np.flatnonzero(ACC[kd] & cm & np.isfinite(mT) & np.isfinite(DLM[kd]))
            X = np.stack([np.ones(len(s)), DU[kd][s], DLM[kd][s], DU[kd][s] * mc[s]], 1); yy = DM[kd][s]; n += len(s)
            np.add.at(XtX, s, X[:, :, None] * X[:, None, :]); np.add.at(Xty, s, X * yy[:, None])
        b = np.linalg.solve(XtX.sum(0), Xty.sum(0))
        bb = np.stack([np.linalg.solve(np.tensordot(w, XtX, 1), w @ Xty) for w in W.astype(np.float64)])
        out[cls] = dict(n_rows=n, a_du=float(b[1]), a_ci=qci(bb[:, 1]), b_dlogm=float(b[2]), b_ci=qci(bb[:, 2]), c_du_x_m=float(b[3]), c_ci=qci(bb[:, 3]),
                        p_c_eq_0=pboot(bb[:, 3]), intercept=float(b[0]))
        # per-bin slopes
        pb = {}
        for lo, hi in BINS:
            inb = (mT >= lo) & (mT < hi); XtX2 = np.zeros((J, 3, 3)); Xty2 = np.zeros((J, 3)); nn_ = 0
            for kd in kinds:
                s = np.flatnonzero(ACC[kd] & cm & inb & np.isfinite(DLM[kd]))
                X = np.stack([np.ones(len(s)), DU[kd][s], DLM[kd][s]], 1); nn_ += len(s)
                np.add.at(XtX2, s, X[:, :, None] * X[:, None, :]); np.add.at(Xty2, s, X * DM[kd][s][:, None])
            if nn_ < 30: continue
            b2 = np.linalg.solve(XtX2.sum(0), Xty2.sum(0))
            bb2 = []
            for w in W[:300].astype(np.float64):
                try: bb2.append(np.linalg.solve(np.tensordot(w, XtX2, 1), w @ Xty2)[1])
                except np.linalg.LinAlgError: pass
            pb[f'{lo}-{hi}'] = dict(n_rows=nn_, a_du=float(b2[1]), a_ci=qci(bb2), b_dlogm=float(b2[2]))
        out[cls]['per_bin'] = pb
    return out


def holds(net, proj, hb, he, mb, me, y, sel, Qs, live, nrand, W, rng_seed):
    """removed / reproduced share per class for named subspaces and random nulls"""
    rg = net.rng_['h4']; out = {}
    def run_hold(Q):
        return margin(net.exact('h4', proj(rg, he, he + ((hb - he) @ Q) @ Q.T))[0])
    def run_copy(Q):
        return margin(net.exact('h4', proj(rg, hb, hb + ((he - hb) @ Q) @ Q.T))[0])
    free = me - mb
    cls_ = {'W': np.flatnonzero(sel & (y == 2)), 'Z': np.flatnonzero(sel & (y == 3))}
    def share(mh, copy=False):
        r = {}
        for c, s in cls_.items():
            d = mh[s] - mb[s]; f = free[s]
            r[c] = float(d.mean() / f.mean()) if copy else float(1 - d.mean() / f.mean())
        return r
    def share_boot(mh, copy=False):
        r = {}
        for c, s in cls_.items():
            bd = bmean(W, mh - mb, s); bf = bmean(W, free, s); r[c] = qci(bd / bf if copy else 1 - bd / bf)
        return r
    for nm, (Q, mode) in Qs.items():
        mh = run_copy(Q) if mode == 'copy' else run_hold(Q)
        o = dict(dim=int(Q.shape[1]), mode=mode, share=share(mh, mode == 'copy'), ci=share_boot(mh, mode == 'copy'))
        # McNemar on the W/Z call (margin sign) change: free vs hold
        o['mcnemar'] = {c: mcn((np.sign(me[s]) != np.sign(mb[s])), (np.sign(mh[s]) != np.sign(mb[s]))) for c, s in cls_.items()}
        o['free_flip'] = {c: float((np.sign(me[s]) != np.sign(mb[s])).mean()) for c, s in cls_.items()}
        o['held_flip'] = {c: float((np.sign(mh[s]) != np.sign(mb[s])).mean()) for c, s in cls_.items()}
        out[nm] = o
    rr = np.random.default_rng(rng_seed); R = {}
    for k in (1, 2, 3):
        R[f'hold_k{k}'] = [share(run_hold(E.rand_live(rr, live, 64, k))) for _ in range(nrand)]
    R['copy_k3'] = [share(run_copy(E.rand_live(rr, live, 64, 3)), copy=True) for _ in range(nrand)]
    out['random'] = {key: {c: dict(median=float(np.median([r[c] for r in v])), q=np.quantile([r[c] for r in v], [.025, .5, .95, .975]).tolist()) for c in ('W', 'Z')} for key, v in R.items()}
    for nm, (Q, mode) in Qs.items():
        key = 'copy_k3' if mode == 'copy' else f'hold_k{Q.shape[1]}'
        if key not in R: continue
        out[nm]['rank_p'] = {c: float((1 + sum(r[c] >= out[nm]['share'][c] for r in R[key])) / (1 + nrand)) for c in ('W', 'Z')}
        out[nm]['null'] = key
    out['free_dmargin'] = {c: dict(mean=float(free[s].mean()), ci=qci(bmean(W, free, s)), n=int(len(s))) for c, s in cls_.items()}
    return out


# ------------------------------------------------------------------ main
def main(a):
    from interp.jedi_core import sample
    from interp.x10_das_jedi import JNet, proj
    import jax
    sm = a.smoke; per = sm or 3000; tag = 'e37x' + ('_smoke' if sm else ''); OUT.mkdir(parents=True, exist_ok=True)
    steps = 60 if sm else 400; nrand = 20 if sm else 200; B = 200 if sm else 2000
    nrand_h = a.nrand_holds or (20 if sm else 2000)   # debug 1: 200 nulls floor rank p at 1/201, Holm x16 at 0.08
    res = dict(node='E37x', smoke=sm, per_class=per, seeds=dict(T=TSEED, F=FSEED), dzs=list(DZS), emf=list(EMF), bins=[list(b) for b in BINS], steps=steps, nrand=nrand, nrand_holds=nrand_h, B=B)
    save = lambda: (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1))
    xT, yT, _ = sample(per, TSEED); xF, yF, _ = sample(per, FSEED); rT = E.rows_of(per, TSEED); rF = E.rows_of(per, FSEED)
    xT = np.asarray(xT, np.float32); xF = np.asarray(xF, np.float32)
    log(f'samples: T {len(yT)} jets (seed {TSEED}), F {len(yF)} (seed {FSEED}); F rows also in T: {int(np.isin(rF, rT).sum())}')
    net = JNet(); net.off = np.zeros(5)
    R0 = net.reps(xT, keep=('S', 'h4', 'h5', 'logits')); res['acc_T_5class_stored'] = float((R0['logits'].argmax(1) == yT).mean())
    res['acc_F_5class_stored'] = float((net.net.run(xF)['logits'].argmax(1) == yF).mean())
    res['tail_check'] = net.check(xT[:2000], {k: v[:2000] for k, v in R0.items()})
    log(f'accuracy 5-class stored inputs: T {res["acc_T_5class_stored"]:.4f}, F {res["acc_F_5class_stored"]:.4f}; tail {res["tail_check"]}')
    del R0
    log('F edits (E37 prep: D, cn, dil, cross)'); DF = E.prep(xF, yF, rF, FSEED, exclude_rows=rT, sm=sm)
    log('T edits (E37 prep)'); DT = E.prep(xT, yT, rT, TSEED, sm=sm)
    J = DT['J']; y = DT['y']; res['N'] = dict(T_WZ=J, F_WZ=DF['J'])
    RF = E.reps_all(net, DF); RT = E.reps_all(net, DT)
    res['acc_T_WZ_rebuilt_5class'] = float((RT['id']['logits'].argmax(1) == y).mean())
    live = [int(u) for u in np.flatnonzero(RT['id']['h4'].std(0) > 1e-9)]; lm = np.isin(np.arange(64), live); res['n_live_h4'] = len(live)
    # ---- refit the sharing direction (E37 DAS, h4, k = 1) on F
    PF = E.build_pairs(DF, RF, 'h4'); fit = [PF['self'], PF['cross']]
    hb = np.concatenate([p.hb for p in fit]); hs = np.concatenate([p.hs for p in fit]); tg = np.concatenate([p.mt for p in fit]); grp = np.concatenate([p.jet for p in fit])
    t = time.time(); Q1, info = E.das_fit_margin(net, 'h4', hb, hs, tg, grp, 1, live, seed=3737, steps=steps); jax.clear_caches()
    del hb, hs, tg, grp, PF, fit
    v = Q1[:, 0].copy()
    dm_ = (RF[(0.3, 'D')]['h4'][DF['acc'][0.3]] - RF['id']['h4'][DF['acc'][0.3]]).mean(0) - (RF[(-0.3, 'D')]['h4'][DF['acc'][-0.3]] - RF['id']['h4'][DF['acc'][-0.3]]).mean(0)
    dm_[~lm] = 0; vdm = dm_ / np.linalg.norm(dm_)
    stored = {}
    try:
        QS = dict(np.load(OUT / 'e37_das_Q.npz')); MS = dict(np.load(OUT / 'e37_das_M.npz'))
        M2 = MS['M2_h4'].copy(); M2[~lm] = 0; M2 = np.linalg.qr(M2)[0]; stored = dict(h4_k1=QS['h4_k1'][:, 0])
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f'stored E37 subspaces missing: {e!r}')
    # orient by natural corr with z on F
    OF = np.full((DF['J'], len(QN)), np.nan)
    for j in range(DF['J']):
        if DF['nn'][j] >= 3: OF[j] = obs1(DF['P4'][j, :DF['nn'][j]], DF['LAB'][j, :DF['nn'][j]])
    okF = np.isfinite(OF).all(1)
    for nm in ('v', 'vdm'):
        vv = v if nm == 'v' else vdm; c = np.corrcoef(RF['id']['h4'][okF] @ vv, OF[okF, 0])[0, 1]
        if c < 0:
            if nm == 'v': v = -v
            else: vdm = -vdm
    res['direction'] = dict(das_train=info, cos_das_vs_diffmeans=float(abs(v @ vdm)), cos_das_vs_stored_E37=float(abs(v @ stored['h4_k1'])),
                            cos_diffmeans_vs_stored_E37=float(abs(vdm @ stored['h4_k1'])),
                            max_cos_das_vs_M2=float(np.linalg.norm(M2.T @ v)), max_cos_dm_vs_M2=float(np.linalg.norm(M2.T @ vdm)),
                            corr_u_z_F=float(np.corrcoef(RF['id']['h4'][okF] @ v, OF[okF, 0])[0, 1]), v_top_units=[int(i) for i in np.argsort(-np.abs(v))[:6]],
                            v_top_weights=np.round(v[np.argsort(-np.abs(v))[:6]], 3).tolist(), fit_seconds=time.time() - t)
    # interchange check on T (reproduces E37?)
    PT = E.build_pairs(DT, RT, 'h4')
    res['direction']['T_interchange'] = {nm: E.metrics(E.patch(net, 'h4', PT[nm].hb, PT[nm].hs, Q1), PT[nm]) for nm in ('self', 'dil')}
    del PT
    log('direction: ' + json.dumps(jsonable({k: v_ for k, v_ in res['direction'].items() if k != 'T_interchange'})) + '; T interchange self frac '
        f'{res["direction"]["T_interchange"]["self"]["frac"]:.3f} IIA {res["direction"]["T_interchange"]["self"]["iia_wz"]}; dil frac {res["direction"]["T_interchange"]["dil"]["frac"]:.3f}')
    save()
    # ---- T variants
    log('T variants (Ez, Edr, Em, Erot, Emf)'); X, ACC, OB, OT, PAR, reasons, chk, exP = build_variants(DT, TSEED, sm)
    res['edits'] = dict(reasons={str(dz): reasons[dz] for dz in DZS}, max_invariant_error=chk,
                        accepted={f'{k[0]}{k[1]:+}': float(ACC[k].mean()) for k in ACC},
                        params={str(dz): {nm: dict(mean=float(np.nanmean(v_)), median=float(np.nanmedian(v_))) for nm, v_ in PAR[dz].items()} for dz in DZS})
    mT = OT[:, QN.index('m')]
    REP = {'id': dict(h4=RT['id']['h4'], logits=RT['id']['logits'])}
    for k, xv in X.items(): REP[k] = net.reps(xv, keep=('h4', 'logits'))
    m0 = margin(REP['id']['logits']); c0 = REP['id']['logits'].argmax(1)
    W = bweights(J, B, 11); WF = bweights(DF['J'], B, 12)
    kinds = list(X.keys())
    # per-edit descriptive table by bin and class
    tab = {}
    for dirn, vv in (('das', v), ('dm', vdm)):
        u0 = REP['id']['h4'] @ vv
        DU = {k: REP[k]['h4'] @ vv - u0 for k in kinds}
        if dirn == 'das': DUd = DU; u0d = u0
        for k in kinds:
            dmg = margin(REP[k]['logits']) - m0; ce = REP[k]['logits'].argmax(1); dlm = np.log(OB[k][:, QN.index('m')] / mT)
            for bn in ['all'] + [f'{lo}-{hi}' for lo, hi in BINS]:
                inb = np.ones(J, bool) if bn == 'all' else (mT >= float(bn.split('-')[0])) & (mT < float(bn.split('-')[1]))
                for cls, ci_ in (('W', 2), ('Z', 3)):
                    sel = ACC[k] & inb & (y == ci_); s = np.flatnonzero(sel)
                    key = f'{k[0]}{k[1]:+}|{bn}|{cls}'; d = tab.setdefault(key, dict(n=int(len(s))))
                    d[f'du_{dirn}'] = mstat(W, DU[k], sel)
                    if dirn == 'das' and len(s) >= 5:
                        d['dmargin'] = mstat(W, dmg, sel); d['dlogm'] = float(dlm[s].mean()); d['dz'] = float((OB[k][s, 0] - OT[s, 0]).mean())
                        d['ddR12'] = float((OB[k][s, 2] - OT[s, 2]).mean())
                        d['W_to_Z'] = float(((c0[s] == 2) & (ce[s] == 3)).mean()); d['Z_to_W'] = float(((c0[s] == 3) & (ce[s] == 2)).mean())
    res['table'] = tab
    for dz in DZS:
        log(f'  dz {dz:+}: ' + '; '.join(f'{kk} W dmg {tab.get(f"{kk}{dz:+}|all|W", {}).get("dmargin", {}).get("mean", float("nan")):+.3f} Z dmg {tab.get(f"{kk}{dz:+}|all|Z", {}).get("dmargin", {}).get("mean", float("nan")):+.3f} '
                                        f'du {tab.get(f"{kk}{dz:+}|all|W", {}).get("du_das", {}).get("mean", float("nan")):+.3f}/{tab.get(f"{kk}{dz:+}|all|Z", {}).get("du_das", {}).get("mean", float("nan")):+.3f} '
                                        f'dlogm {tab.get(f"{kk}{dz:+}|all|W", {}).get("dlogm", float("nan")):+.4f}' for kk in ('Ez', 'Edr', 'Em', 'Erot')))
    # McNemar on call changes: Ez vs Edr, Ez vs Erot, Ez vs Em (same jets)
    mc = {}
    for dz in DZS:
        s0 = ACC[('Ez', dz)]; chg = {kk: REP[(kk, dz)]['logits'].argmax(1) != c0 for kk in ('Ez', 'Edr', 'Em', 'Erot')}
        for cls, ci_ in (('W', 2), ('Z', 3)):
            s = s0 & (y == ci_)
            mc[f'{dz:+}|{cls}'] = {f'Ez_vs_{kk}': mcn(chg['Ez'][s], chg[kk][s]) for kk in ('Edr', 'Em', 'Erot')}
    res['mcnemar_calls'] = mc
    save()
    # ---- read-outs, calibrated slopes, naming
    res['readout'] = {}
    for dirn, vv in (('das', v), ('dm', vdm)):
        u0F = RF['id']['h4'] @ vv; u0T = REP['id']['h4'] @ vv; DU = {k: REP[k]['h4'] @ vv - u0T for k in kinds}
        r = readout_core(u0F, OF, u0T, OT, DU, ACC, OB, y, W, WF, mT, dirn); res['readout'][dirn] = r
        pz = r['calibrated'].get('all|WZ', {}).get('pooled', {})
        log(f'readout {dirn}: beta_z {r["beta_z_knn"]["all"]}; pooled R_z {pz.get("R_z")} {pz.get("R_z_ci")}, R_Edr {pz.get("R_Edr")} {pz.get("R_Edr_ci")}, R_rot {pz.get("R_rot")}; '
            f'naming {r["naming"]["closest_to_1"]}: ' + '; '.join(f'{q}: s {o["s_q"]:.3f} {o["s_q_ci"]} partialR2 {o["heldout_partial_R2"]:.3f}' for q, o in r['naming']['per_quantity'].items()))
        save()
    # ---- interaction model
    DM = {k: margin(REP[k]['logits']) - m0 for k in kinds}; DLM = {k: np.log(OB[k][:, QN.index('m')] / mT) for k in kinds}
    res['interaction'] = interaction(DUd, DM, DLM, ACC, y, mT, W, kinds)
    log('interaction: ' + json.dumps(jsonable({c: {kk: o[kk] for kk in ('a_du', 'b_dlogm', 'c_du_x_m', 'c_ci', 'p_c_eq_0')} for c, o in res['interaction'].items()})))
    save()
    # ---- holds (carriers of the response)
    Qv = v[:, None]; Qb = np.linalg.qr(np.c_[v, M2])[0]; Qdm = vdm[:, None]; Qbd = np.linalg.qr(np.c_[vdm, M2])[0]
    sets = {'i_hold_v': (Qv, 'hold'), 'ii_hold_M2': (M2, 'hold'), 'iii_hold_v_M2': (Qb, 'hold'), 'iv_copy_v_M2': (Qb, 'copy'),
            'x_hold_vdm': (Qdm, 'hold'), 'x_hold_vdm_M2': (Qbd, 'hold'), 'x_copy_vdm_M2': (Qbd, 'copy'), 'x_copy_v': (Qv, 'copy'), 'x_copy_M2': (M2, 'copy')}
    res['holds'] = {}; ps = []; keys = []
    for ek in (('D', 0.3), ('Ez', 0.15)):
        sel = ACC[ek]; hb_, he_ = REP['id']['h4'], REP[ek]['h4']; mb_, me_ = m0, margin(REP[ek]['logits'])
        chkt = float(np.abs(margin(net.exact('h4', he_[:2000])[0]) - me_[:2000]).max())
        o = holds(net, proj, hb_, he_, mb_, me_, y, sel, sets, live, nrand_h, W, 41 if ek[0] == 'D' else 42); o['tail_check'] = chkt
        name = f'{ek[0]}{ek[1]:+}'; res['holds'][name] = o
        for nm in ('i_hold_v', 'ii_hold_M2', 'iii_hold_v_M2', 'iv_copy_v_M2'):
            for c in ('W', 'Z'): ps.append(o[nm]['rank_p'][c]); keys.append(f'{name}|{nm}|{c}')
        log(f'holds {name}: free {o["free_dmargin"]}; ' + '; '.join(f'{nm} {o[nm]["share"]} rank p {o[nm].get("rank_p")}' for nm in sets) +
            '; random ' + json.dumps(jsonable({k_: {c: o['random'][k_][c]['median'] for c in ('W', 'Z')} for k_ in o['random']})))
        save()
    res['holm_holds'] = dict(zip(keys, np.asarray(E.holm(ps), float).tolist())); res['holm_n'] = len(ps)
    # ---- example jet (W, 75-85 GeV, Ez +0.15 accepted)
    k15 = ('Ez', 0.15); ce = REP[k15]['logits'].argmax(1)
    cand = np.flatnonzero((y == 2) & ACC[k15] & (mT >= 75) & (mT < 85) & (DT['nn'] >= 15) & (DT['nn'] <= 40) & (c0 == 2))
    flip = cand[ce[cand] == 3]; pick = flip if len(flip) else cand
    if len(pick):
        j = int(pick[0]); n = int(DT['nn'][j]); ex = dict(index=j, N=n, prong=DT['LAB'][j, :n].tolist(), flips_to_Z=bool(ce[j] == 3))
        for nm, P in (('unedited', DT['P4'][j, :n]), ('Ez+0.15', exP[k15][j, :n]), ('Edr+0.15', exP[('Edr', 0.15)][j, :n]), ('Em+0.15', exP[('Em', 0.15)][j, :n])):
            pt = np.hypot(P[:, 0], P[:, 1]); T_ = P.sum(0); eT = np.arcsinh(T_[2] / np.hypot(T_[0], T_[1])); fT = np.arctan2(T_[1], T_[0])
            ex[nm] = dict(pt=pt.tolist(), deta=(np.arcsinh(P[:, 2] / pt) - eT).tolist(), dphi=wrap(np.arctan2(P[:, 1], P[:, 0]) - fT).tolist(),
                          obs=dict(zip(QN, obs1(P, DT['LAB'][j, :n]).tolist())))
        ex['margin'] = {nm: float(margin(REP[k]['logits'][j:j + 1])[0]) for nm, k in (('unedited', 'id'), ('Ez+0.15', k15), ('Edr+0.15', ('Edr', 0.15)), ('Em+0.15', ('Em', 0.15)))}
        ex['u'] = {nm: float(REP[k]['h4'][j] @ v) for nm, k in (('unedited', 'id'), ('Ez+0.15', k15), ('Edr+0.15', ('Edr', 0.15)), ('Em+0.15', ('Em', 0.15)))}
        res['example'] = ex
    res['u_sd_T'] = float(u0d.std()); res['margin_sd_T'] = float(m0.std())
    save(); log('trained part written')
    # ---- untrained floor
    del net; jax.clear_caches()
    un = JNet(untrained=True, seed=0); un.off = np.zeros(5)
    RFu = {k: un.reps(DF['X'][k], keep=('h4', 'logits')) for k in ('id', (0.3, 'D'), (-0.3, 'D'))}
    off = RFu['id']['logits'].mean(0)
    REPu = {'id': un.reps(DT['X']['id'], keep=('h4', 'logits'))}
    for k, xv in X.items(): REPu[k] = un.reps(xv, keep=('h4', 'logits'))
    liveu = np.flatnonzero(REPu['id']['h4'].std(0) > 1e-9)
    d_ = (RFu[(0.3, 'D')]['h4'][DF['acc'][0.3]] - RFu['id']['h4'][DF['acc'][0.3]]).mean(0) - (RFu[(-0.3, 'D')]['h4'][DF['acc'][-0.3]] - RFu['id']['h4'][DF['acc'][-0.3]]).mean(0)
    mk = np.zeros(64, bool); mk[liveu] = True; d_[~mk] = 0
    fl = dict(n_live_h4=int(len(liveu)), acc_T_WZ_5class=float(((REPu['id']['logits'] - off).argmax(1) == y).mean()))
    if np.linalg.norm(d_) > 0:
        vu = d_ / np.linalg.norm(d_)
        if np.corrcoef(RFu['id']['h4'][okF] @ vu, OF[okF, 0])[0, 1] < 0: vu = -vu
        u0F = RFu['id']['h4'] @ vu; u0T = REPu['id']['h4'] @ vu; DU = {k: REPu[k]['h4'] @ vu - u0T for k in kinds}
        try:
            r = readout_core(u0F, OF, u0T, OT, DU, ACC, OB, y, W[:300], WF[:300], mT, 'untrained_dm')
        except Exception as e:  # noqa: BLE001
            r = None; fl['readout_error'] = repr(e)
        if r is not None: fl['readout'] = dict(beta_z_knn=r['beta_z_knn']['all'], pooled=r['calibrated'].get('all|WZ', {}).get('pooled'),
                             naming={q: {kk: o[kk] for kk in ('s_q', 's_q_ci', 'heldout_partial_R2', 'heldout_R2')} for q, o in r['naming']['per_quantity'].items()})
    mu = margin(REPu['id']['logits'] - off); sdu = float(mu.std()); fl['margin_sd'] = sdu
    fl['dmargin_sd_units'] = {f'{k[0]}{k[1]:+}': {c: float((margin(REPu[k]['logits'] - off) - mu)[ACC[k] & (y == ci_)].mean() / max(sdu, 1e-12)) for c, ci_ in (('W', 2), ('Z', 3))}
                              for k in kinds if k[0] in ('Ez', 'Edr', 'Em', 'Erot')}
    fl['trained_dmargin_sd_units'] = {f'{k[0]}{k[1]:+}': {c: float(DM[k][ACC[k] & (y == ci_)].mean() / m0.std()) for c, ci_ in (('W', 2), ('Z', 3))}
                                      for k in kinds if k[0] in ('Ez', 'Edr', 'Em', 'Erot')}
    res['untrained_floor'] = fl
    log('untrained floor: ' + json.dumps(jsonable(fl))[:1500])
    res['wall_s'] = time.time() - T0; save(); log(f'wrote {OUT / tag}.json; E37x DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--nrand-holds', type=int, default=0)
    main(ap.parse_args())
