"""E25 step 0: how the 16 jetConstituentList columns relate to (pT, eta, phi) and the stored jet axis (needed to recompute all
16 features after an edit).  python -m interp.e25_diag  -> $AI_RESULTS/jedi/e25_diag.json and the log.

Uses the rows of interp.jedi_core.sample(3000, 2501) (first 2,000 jets), read from val_c16_f16.npy.  Checks (real particles,
pT > 0): pT = hypot(px, py); phi = atan2(py, px); eta = asinh(pz / pT); E^2 - p^2; jet axis eta_J = eta - etarel and
phi_J = phi - phirel (spread within a jet); pT_J = pT / ptrel, E_J = E / erel (spread within a jet); deltaR vs hypot(etarel, phirel);
costheta vs pz / |p|; costhetarel vs candidate definitions; etarot, phirot as a per-jet affine map of (etarel, phirel)
(residual, orthogonality, angle vs the pT-weighted principal axis).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
C = {k: i for i, k in enumerate(['px', 'py', 'pz', 'e', 'erel', 'pt', 'ptrel', 'eta', 'etarel', 'etarot', 'phi', 'phirel',
                                 'phirot', 'deltaR', 'costheta', 'costhetarel'])}


def q(v): v = np.asarray(v, np.float64); v = v[np.isfinite(v)]; return [float(np.quantile(v, p)) for p in (0, .5, .99, 1)] if v.size else None


def main():
    from jetdistill.config import RESULTS, DATA
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    rng = np.random.default_rng(2501)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), 3000, replace=False) for c in range(5)]))[::7][:2000]
    X = np.asarray(np.load(DATA / 'val_c16_f16.npy', mmap_mode='r')[idx, :64]).astype(np.float64)
    R = {}
    rl = X[..., C['pt']] > 0; v = lambda k: X[..., C[k]][rl]
    px, py, pz, e, pt, eta, phi = (v(k) for k in ('px', 'py', 'pz', 'e', 'pt', 'eta', 'phi'))
    R['pt_vs_hypot'] = q(np.abs(np.hypot(px, py) / pt - 1))
    R['phi_vs_atan2'] = q(np.abs(np.angle(np.exp(1j * (np.arctan2(py, px) - phi)))))
    R['eta_vs_asinh'] = q(np.abs(np.arcsinh(pz / pt) - eta))
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2); R['m2_over_e2'] = q((e ** 2 - p ** 2) / e ** 2)
    R['e_vs_ptcosh'] = q(np.abs(pt * np.cosh(eta) / e - 1))
    J = np.broadcast_to(np.arange(len(X))[:, None], rl.shape)[rl]
    def per_jet(val):
        out = np.full(len(X), np.nan); sp = np.zeros(len(X))
        for j in range(len(X)):
            w = val[J == j]
            if w.size: out[j] = np.median(w); sp[j] = np.abs(w - out[j]).max()
        return out, sp
    etaJ, s1 = per_jet(eta - v('etarel')); R['etaJ_spread'] = q(s1)
    dphi = np.angle(np.exp(1j * (phi - v('phirel')))); phiJ, s2 = per_jet(dphi); R['phiJ_spread'] = q(s2)
    ptJ, s3 = per_jet(pt / v('ptrel')); R['ptJ_rel_spread'] = q(s3 / ptJ)
    eJ, s4 = per_jet(e / v('erel')); R['eJ_rel_spread'] = q(s4 / eJ)
    R['phirel_wrapped_vs_raw'] = q(np.abs(v('phirel') - (phi - phiJ[J])))
    er, pr = v('etarel'), v('phirel')
    R['deltaR_vs_hypot'] = q(np.abs(v('deltaR') - np.hypot(er, pr)))
    R['costheta_vs_pz_p'] = q(np.abs(v('costheta') - pz / p))
    R['costheta_vs_tanh_eta'] = q(np.abs(v('costheta') - np.tanh(eta)))
    # costhetarel candidates: angle between particle 3-momentum and the jet axis direction (eta_J, phi_J); jet 3-momentum
    nJ = np.stack([np.cos(phiJ), np.sin(phiJ), np.sinh(etaJ)], 1) / np.cosh(etaJ)[:, None]
    cand = dict(axis=(px * nJ[J, 0] + py * nJ[J, 1] + pz * nJ[J, 2]) / p, tanh_etarel=np.tanh(er), cos_deltaR=np.cos(np.hypot(er, pr)))
    P3 = np.zeros((len(X), 3)); np.add.at(P3, J, np.stack([px, py, pz], 1)); nP = P3 / np.linalg.norm(P3, axis=1, keepdims=True)
    cand['jet_sum64'] = (px * nP[J, 0] + py * nP[J, 1] + pz * nP[J, 2]) / p
    R['costhetarel'] = {k: q(np.abs(v('costhetarel') - c)) for k, c in cand.items()}
    R['costhetarel_range'] = q(v('costhetarel'))
    # etarot / phirot: per-jet affine fit on (etarel, phirel, 1)
    eo, po = v('etarot'), v('phirot'); res, orth, ang, ang_pa, det, off = [], [], [], [], [], []
    for j in range(len(X)):
        m = J == j
        if m.sum() < 4: continue
        A = np.stack([er[m], pr[m], np.ones(m.sum())], 1); Y = np.stack([eo[m], po[m]], 1)
        W, *_ = np.linalg.lstsq(A, Y, rcond=None); res.append(np.abs(A @ W - Y).max())
        M = W[:2].T; orth.append(np.abs(M @ M.T - np.eye(2)).max()); det.append(np.linalg.det(M)); off.append(np.abs(W[2]).max())
        ang.append(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        w = pt[m]; ce, cp = (w * er[m]).sum() / w.sum(), (w * pr[m]).sum() / w.sum()
        cxx = (w * (er[m] - ce) ** 2).sum(); cyy = (w * (pr[m] - cp) ** 2).sum(); cxy = (w * (er[m] - ce) * (pr[m] - cp)).sum()
        ang_pa.append(np.degrees(0.5 * np.arctan2(2 * cxy, cxx - cyy)))
    R['rot'] = dict(fit_resid=q(res), orth_err=q(orth), det=q(det), offset=q(off), angle_deg=q(ang),
                    angle_minus_pa_mod90=q(np.abs(((np.array(ang) + np.array(ang_pa)) + 45) % 90 - 45)),
                    angle_plus_pa_mod90=q(np.abs(((np.array(ang) - np.array(ang_pa)) + 45) % 90 - 45)),
                    examples=[[float(a), float(b)] for a, b in zip(ang[:10], ang_pa[:10])])
    R['etarot_vs_etarel'] = q(np.abs(eo - er)); R['phirot_vs_phirel'] = q(np.abs(po - pr))
    # diag 2: etarot/phirot relative to the leading particle (slot 0) with the second particle on the -phirot axis; costhetarel
    d_e = X[..., C['etarel']] - X[:, :1, C['etarel']]; d_p = X[..., C['phirel']] - X[:, :1, C['phirel']]
    d_p = np.angle(np.exp(1j * d_p)); dd = np.hypot(d_e, d_p)[rl]
    R['rot2'] = dict(lead_etarot_abs=q(np.abs(X[:, 0, C['etarot']])), lead_phirot_abs=q(np.abs(X[:, 0, C['phirot']])),
                     second_etarot_abs=q(np.abs(X[:, 1, C['etarot']][rl[:, 1]])), second_phirot=q(X[:, 1, C['phirot']][rl[:, 1]]),
                     dist_vs_hypot_rot=q(np.abs(dd - np.hypot(eo, po))), dist_vs_hypot_rot_rel=q(np.abs(dd - np.hypot(eo, po)) / np.maximum(dd, 1e-3)),
                     third_etarot_sign=q(np.sign(X[:, 2, C['etarot']][rl[:, 2]])))
    # exact linear map per jet on lead-relative offsets (no intercept)
    res2, orth2 = [], []
    for j in range(len(X)):
        m = rl[j]
        if m.sum() < 4: continue
        A = np.stack([d_e[j][m], d_p[j][m]], 1); Y = np.stack([X[j, m, C['etarot']], X[j, m, C['phirot']]], 1)
        W, *_ = np.linalg.lstsq(A, Y, rcond=None); res2.append(np.abs(A @ W - Y).max()); orth2.append(np.abs(W.T @ W - np.eye(2)).max())
    R['rot2']['lin_fit_resid'] = q(res2); R['rot2']['lin_orth_err'] = q(orth2)
    ctr = v('costhetarel')
    R['costhetarel_eq'] = {k: float(np.mean(np.abs(ctr - v(k)) < 1e-6)) for k in C}
    R['costhetarel_absdiff'] = {k: q(np.abs(ctr - v(k))) for k in C}
    R['costhetarel_nonzero_frac'] = float(np.mean(ctr != 0)); R['etarot_nonzero_frac'] = float(np.mean(eo != 0))
    R['first_jet_rows'] = X[0, :6].tolist(); R['N_jets'] = len(X)
    for k, val in R.items(): log(k, val)
    (OUT / 'e25_diag.json').write_text(json.dumps(R, indent=1)); log('done')


main()
