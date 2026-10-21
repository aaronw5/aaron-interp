"""H45 (loop D, new claim D18, exploratory): two-prong momentum sharing. Do two-prong taggers respond to how the momentum is
shared between the two prongs (z = softer-prong pT fraction, equivalently |cos theta*| of the decay axis in the prong-pair rest
frame) beyond what the side effects of changing it on tau21, D2 and the prong separation Delta R predict?

Physics. For a two-body decay X -> a b boosted to energy E_X, the prong energies are E_a,b = (E_X / 2)(1 +- beta cos theta*)
(equal daughter masses), so z ~ (1 - |cos theta*|) / 2 at beta -> 1. A W / Z decaying to quarks gives dN/dcos theta* ~ 1 +- cos^2
or (1 - cos^2) shapes (balanced or moderately asymmetric splits); a QCD 1 -> 2 splitting follows P(z), which grows as z -> 0
(asymmetric). Mass-drop / soft-drop taggers cut on z for this reason. At fixed pair mass m, m^2 ~ z (1 - z) pT^2 Delta R^2, so
changing z also changes Delta R, tau21 and D2: those are the side effects controlled here.

Samples (random rows, never first-k)
  jc   JetClass test files WToQQ_100 (Wqq), ZToQQ_100 (Zqq), ZJetsToNuNu_100 (QCD): stored 500 < jet_pt < 1000 GeV, |jet_eta| < 2,
       >= 3 particles; PER random rows per class (default_rng 4501 / 4502 / 4503 among the eligible rows; h41.read_window).
  hls  hls4ml test jets of jedi_core.sample(per_class=PER, seed=4504) (same rows; 5 classes drawn, tops used only for the
       5-class accuracy check); analysis on W, Z, q, g. Massless constituents, 64 slots, stored axis (0, 0).
Prongs: two exclusive-kT subjets (p = 1, E-scheme recombination, rapidity-azimuth distance) of all constituents; every
  constituent belongs to one of them, so the pair four-momentum is the jet four-momentum.
Edit D(Dc), Dc in {-0.3, -0.15, +0.15, +0.3}: boost every constituent to the jet (pair) rest frame (velocity beta = p_jet / E_jet);
  n_+ = rest-frame direction of the prong that moves forward (n_+ . beta_hat >= 0), c0 = |cos theta*| = n_+ . beta_hat;
  target c1 = clip(c0 + Dc, 0, 0.95); rotate all constituents by R = rotation about k = n_+ x beta_hat by
  alpha = arccos c0 - arccos c1; boost back. Every constituent undergoes the same Lorentz transformation
  Lambda = B(-beta) R B(beta), which leaves the jet four-momentum fixed: jet mass, jet pT, jet direction, every pairwise invariant
  mass p_i . p_j (so each subjet mass) and the multiplicity are exactly unchanged; the lab-frame energies (z), angles (Delta R,
  tau21, D2) change. Dc < 0: more balanced prongs; Dc > 0: more asymmetric.
Controls on the same jets
  cphi   rotation by the same |alpha| about beta_hat in the rest frame (random sign): |cos theta*| and the prong energies exactly
         fixed; the prongs move azimuthally around the jet axis (z kept, prongs moved).
  cside  per-jet edit in (eta, phi) at fixed constituent pT (scalar prong pT fractions, hence z, fixed): the two prongs are moved
         rigidly along their joining line about their pT-weighted centre so that the prong separation Delta R equals D's, and the
         constituent offsets from each prong centre are scaled by s_a, s_b (s_b / s_a in 14 log-spaced ratios 0.12-8, s_a solved per
         ratio so that the jet invariant mass equals the unedited one); among the 14 the one whose reclustered tau21 is closest to
         D's is kept. Matched (H32 lesson): |dm| < 1 GeV, |tau21 - tau21_D| <= 5 % of tau21_D, |Delta R - Delta R_D| <= 5 % of Delta R_D.
  TN     translation-only: the unedited jet moved rigidly in (eta, phi) by D's pT-weighted mean constituent displacement in a
         random direction.
  Acceptance (D, cphi): both prong centres within Delta R 0.75 of the jet axis after the edit.
  Every jet (edited and unedited) is rebuilt about its own E-scheme axis (rc build, round-D5 rule) for every network.
Models
  jc   ParT_full, ParT_kin, PN_full, PN_kin (released JetClass weights); nulls: NPERMNET ParT_full copies with every weight tensor
       permuted (torch seeds 5, 6, ...).
  hls  JEDI-linear official n64 3-feature; small hls4ml nets dsets-d64-n64, jedis-d64-n64, tf-2b-d32-n64 (seed0); nulls: JEDI with
       weights redrawn (seeds 0-2), dsets with every weight tensor permuted (seeds 7-9).
Scores (log-odds)
  jc   WQ = L_Wqq - L_QCD, ZQ = L_Zqq - L_QCD, WZ = L_Wqq - L_Zqq (10-class logits).
  hls  WQG = L_W - logsumexp(L_q, L_g), ZQG = L_Z - logsumexp(L_q, L_g), WZ = L_W - L_Z.
References (per dataset, per score pair, trained on train rows, NSEED retrains: MLP random_state / LR bootstrap resample)
  MLPx  MLP(64, 64) on (m, pT, tau21, D2, N, Delta R) (the side effects of the edit, no momentum-sharing variable)
  MLPz  MLPx + z_g (soft-drop momentum fraction, z_cut 0.1, beta 0, R0 0.8, C/A)
  LRzg  logistic regression on (z_g, z_g^2)
  jc train rows: JetClass train_100M WToQQ_000 / ZToQQ_000 / ZJetsToNuNu_000 in the same window (NREF per class, rng 4506-4508);
  hls train rows: hls4ml train archive (dev split excluded), NREF per class W, Z, q, g (rng 4509); first 64 particles.
Metrics per dataset x (sample class, score) cell x Dc x model: per-jet Delta s = s(edit) - s(unedited); D - cphi (rows accepted
  for D and cphi) and D - cside (rows accepted for D, cphi, cside); mean, 95 % bootstrap CI over jets (B), two-sided sign-flip
  permutation p (NPERM) and one-sample t-test p; Holm (on the t-test p: the permutation p is floored at 1 / (NPERM + 1)) over
  networks x cells x Dc x {D-cphi, D-cside}, per dataset and combined; per-jet net - MLPx (seed 0), permutation and t-test p, Holm;
  hls4ml: change of JEDI-linear's input moments M0 = sum pT and m_A = sqrt(M0 M2) under each edit; permuted/redrawn null band (min, max of the null models' means); MLPx / MLPz / LRzg retrain ranges;
  McNemar on calls (threshold = median unedited score of the score's signal class, eps_S = 0.5) D vs cphi and D vs cside;
  dose = Spearman of the D - cside means against Dc; C2ST (2-fold HGB AUC on jet observables) of edited vs natural train jets
  of the same class matched (nearest neighbour) in standardised (m, |cos theta*|), for id / D / cphi / cside at Dc = +-0.3.
Parts (results $AI_RESULTS/physics/h45_*):
  --part jcedit  (conda: numba, sklearn)  JetClass sample, edits, observables, references -> h45_jc_inputs.npz, h45_jc_obs.npz
  --part jcnets  (GPU, part_job)          JetClass networks                                -> h45_jc_logits.npz
  --part hedit   (conda)                  hls4ml sample, edits, observables, references   -> h45_hls_inputs.npz, h45_hls_obs.npz
  --part hsmall  (conda: torch)           small hls4ml nets                                -> h45_hls_logits_small.npz
  --part hjedi   (jax env)                JEDI-linear official + redrawn                   -> h45_hls_logits_jedi.npz
  --part stats   (conda)                  statistics, Holm, C2ST, figures                  -> h45_prong_sharing.json, h45_edit.png, h45_response.png
  --selftest                              edit invariants on a synthetic two-prong jet
Run: AI_PY=conda ./job.sh d-h45-edit 4 8 'python -m interp.h45_prong_sharing --part jcedit && python -m interp.h45_prong_sharing
     --part hedit && python -m interp.h45_prong_sharing --part hsmall && /j-jepa-vol/phat-jet-aaron/env/bin/python -m
     interp.h45_prong_sharing --part hjedi'; ./part_job.sh d-h45-nets 'python -m interp.h45_prong_sharing --part jcnets';
     AI_PY=conda ./job.sh d-h45-stats 2 6 'python -m interp.h45_prong_sharing --part stats'      (H45_SMOKE=<n>: n jets per class)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ.setdefault('NUMBA_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '4'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from numba import njit, prange
except ImportError:  # jax env / local import check
    njit = lambda *a, **k: (lambda f: f); prange = range

T00 = time.time()
SMOKE = int(os.environ.get('H45_SMOKE', '0'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
TAG = 'h45_smoke' if SMOKE else 'h45'
PER = SMOKE or 2000
NREF = 1500 if SMOKE else 10000
NSEED = 2 if SMOKE else 5
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
NPERMNET = 2 if SMOKE else 6
DCS = (-0.3, -0.15, 0.15, 0.3)
CMAX = 0.95
DRMAX = 0.75
RATIOS = np.exp(np.linspace(np.log(0.12), np.log(8.0), 14))
MATCH = 0.05
TRAIN = Path('/j-jepa-vol/JetClass/Pythia/train_100M')
JC_CLASSES = {'Wqq': 7, 'Zqq': 6, 'QCD': 0}
JC_PRE = {'Wqq': 'WToQQ', 'Zqq': 'ZToQQ', 'QCD': 'ZJetsToNuNu'}
HLS_CLASSES = {'g': 0, 'q': 1, 'W': 2, 'Z': 3, 't': 4}
JC_NETS = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin']
HLS_NETS = ['jedi', 'dsets', 'jedis', 'tf']
REFS = ('MLPx', 'MLPz', 'LRzg')
VARS = ['D', 'cphi', 'cside', 'TN']
# (sample class, score) cells; score -> (signal class, background classes) for references and working points
JC_SCORES = {'WQ': ('Wqq', ['QCD']), 'ZQ': ('Zqq', ['QCD']), 'WZ': ('Wqq', ['Zqq'])}
JC_CELLS = [('Wqq', 'WQ'), ('Wqq', 'WZ'), ('Zqq', 'ZQ'), ('Zqq', 'WZ'), ('QCD', 'WQ'), ('QCD', 'ZQ')]
HLS_SCORES = {'WQG': ('W', ['q', 'g']), 'ZQG': ('Z', ['q', 'g']), 'WZ': ('W', ['Z'])}
HLS_CELLS = [('W', 'WQG'), ('W', 'WZ'), ('Z', 'ZQG'), ('Z', 'WZ'), ('q', 'WQG'), ('q', 'ZQG'), ('g', 'WQG'), ('g', 'ZQG')]
OBSK = ('m', 'pt', 'mSD', 'tau21', 'tau32', 'N', 'D2', 'dR', 'z', 'cos', 'zg', 'Rg', 'msub_hi', 'msub_lo')


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


# ================================================================================================================ kinematics
def boost(P, b):
    """four-vectors P (..., 4) = (px, py, pz, E) seen in a frame moving with velocity b (3,) (h31_top_dalitz.boost)"""
    b2 = float(b @ b)
    if b2 < 1e-16: return P.copy()
    g = 1.0 / np.sqrt(1.0 - b2); bp = P[..., :3] @ b
    out = np.empty_like(P)
    out[..., :3] = P[..., :3] + (((g - 1.0) * bp / b2) - g * P[..., 3])[..., None] * b
    out[..., 3] = g * (P[..., 3] - bp)
    return out


def boostv(P, b):
    """per-row boost: P (J, 4), b (J, 3)"""
    b2 = np.maximum((b * b).sum(1), 1e-16); g = 1.0 / np.sqrt(1.0 - np.minimum(b2, 1 - 1e-12)); bp = (P[:, :3] * b).sum(1)
    out = np.empty_like(P); out[:, :3] = P[:, :3] + (((g - 1.0) * bp / b2) - g * P[:, 3])[:, None] * b; out[:, 3] = g * (P[:, 3] - bp)
    return out


def rodrigues(k, a):
    k = k / np.linalg.norm(k); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def m2(v):
    return v[..., 3] ** 2 - (v[..., :3] ** 2).sum(-1)


def ptetaphim(P):
    pt = np.hypot(P[..., 0], P[..., 1]); eta = np.arcsinh(P[..., 2] / np.maximum(pt, 1e-12)); phi = np.arctan2(P[..., 1], P[..., 0])
    m = np.sqrt(np.maximum(P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1), 0)); return pt, eta, phi, m


def p4(pt, eta, phi, m):
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)


def rap(P):
    e = np.maximum(P[..., 3], np.abs(P[..., 2]) + 1e-9); return 0.5 * np.log((e + P[..., 2]) / (e - P[..., 2]))


def sort_pt(P4, nn):
    pt = np.hypot(P4[..., 0], P4[..., 1]); real = np.arange(P4.shape[1])[None, :] < nn[:, None]
    o = np.argsort(-np.where(real, pt, -1), 1, kind='stable'); return np.take_along_axis(P4, o[..., None], 1), o


def axis(P4, nn):
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    S = (P * real[..., None]).sum(1); pt = np.hypot(S[:, 0], S[:, 1])
    return np.arcsinh(S[:, 2] / pt), np.arctan2(S[:, 1], S[:, 0])


# ================================================================================================================ numba observables
try:
    from interp.h2_onesignal_setup import _cluster as _h2c
except Exception:  # jax env
    _h2c = None


@njit(parallel=True, cache=False)
def zg_batch(F, nn, zcut):
    """soft drop (beta = 0) on the C/A tree of all constituents: z_g = min(pT_a, pT_b) / (pT_a + pT_b) at the first declustering
    passing z > zcut, R_g = rapidity-azimuth distance of that pair; 0 if no declustering passes"""
    J = F.shape[0]; out = np.zeros((J, 2))
    for j in prange(J):
        n = nn[j]
        if n < 2: continue
        px = F[j, :n, 0].copy(); py = F[j, :n, 1].copy(); pz = F[j, :n, 2].copy(); E = F[j, :n, 3].copy()
        _, vx, vy, vz, vE, pt, c1, c2, nxt = _h2c(px, py, pz, E, n, 0, False, 1, True)
        node = nxt - 1
        while True:
            a = c1[node]; b = c2[node]
            if a < 0: break
            z = min(pt[a], pt[b]) / (pt[a] + pt[b])
            if z > zcut:
                ea = max(vE[a], abs(vz[a]) + 1e-9); eb = max(vE[b], abs(vz[b]) + 1e-9)
                ya = 0.5 * np.log((ea + vz[a]) / (ea - vz[a])); yb = 0.5 * np.log((eb + vz[b]) / (eb - vz[b]))
                dphi = abs(np.arctan2(vy[a], vx[a]) - np.arctan2(vy[b], vx[b]))
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                out[j, 0] = z; out[j, 1] = np.sqrt((ya - yb) ** 2 + dphi ** 2); break
            node = a if pt[a] >= pt[b] else b
    return out


@njit(parallel=True, cache=False)
def tau21_batch(F, nn):
    """tau2 / tau1 (beta 1, exclusive-kT WTA axes; the h2/h4 definition) only"""
    J = F.shape[0]; out = np.zeros(J)
    for j in prange(J):
        n = nn[j]
        if n < 2: continue
        px = F[j, :n, 0].copy(); py = F[j, :n, 1].copy(); pz = F[j, :n, 2].copy(); E = F[j, :n, 3].copy()
        rw, _, _, _, _, _, _, _, _ = _h2c(px, py, pz, E, n, 1, True, 1, False)
        t1 = 0.0; t2 = 0.0
        for i in range(n):
            ptp = np.hypot(px[i], py[i]); php = np.arctan2(py[i], px[i]); e = max(E[i], abs(pz[i]) + 1e-9)
            yp = 0.5 * np.log((e + pz[i]) / (e - pz[i]))
            dphi = abs(php - rw[5, 6])
            if dphi > np.pi: dphi = 2 * np.pi - dphi
            t1 += ptp * np.sqrt((yp - rw[5, 5]) ** 2 + dphi ** 2)
            dm = 1e9
            for k in range(2):
                dphi = abs(php - rw[3 + k, 6])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                d = np.sqrt((yp - rw[3 + k, 5]) ** 2 + dphi ** 2)
                if d < dm: dm = d
            t2 += ptp * dm
        out[j] = t2 / max(t1, 1e-12)
    return out


@njit(parallel=True, cache=False)
def d2_batch(F, nn):
    """D2 (beta 1) = e3 / e2^3 on all constituents, rapidity-azimuth distances, z_i = pT_i / sum pT (h13_pileup_panel.d2_numba)"""
    J = F.shape[0]; out = np.zeros(J)
    for j in prange(J):
        n = nn[j]
        if n < 3: continue
        pt = np.zeros(n); yy = np.zeros(n); ph = np.zeros(n)
        for i in range(n):
            px = F[j, i, 0]; py = F[j, i, 1]; pz = F[j, i, 2]; E = F[j, i, 3]
            pt[i] = np.hypot(px, py); ph[i] = np.arctan2(py, px); e = max(E, abs(pz) + 1e-9); yy[i] = 0.5 * np.log((e + pz) / (e - pz))
        z = pt / pt.sum(); D = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                dphi = abs(ph[a] - ph[b])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                D[a, b] = np.sqrt((yy[a] - yy[b]) ** 2 + dphi ** 2); D[b, a] = D[a, b]
        e2 = 0.0; e3 = 0.0
        for a in range(n):
            for b in range(a + 1, n):
                w = z[a] * z[b] * D[a, b]; e2 += w
                for c in range(b + 1, n): e3 += w * z[c] * D[a, c] * D[b, c]
        out[j] = e3 / max(e2 ** 3, 1e-30)
    return out


def kt2_labels(P4, nn):
    from interp.h41_colour_flow import ktn_batch
    return ktn_batch(np.ascontiguousarray(P4.astype(np.float64)), nn.astype(np.int64), 2)


def prong_vars(P4, nn, LAB):
    """from kT-2 labels: subjet four-vectors, z (softer scalar-vector pT fraction), Delta R (rapidity-azimuth), |cos theta*|"""
    L = LAB.shape[1]; real = np.arange(L)[None, :] < nn[:, None]
    oh = np.stack([(LAB == k) & real for k in range(2)], -1).astype(np.float64)
    S = np.einsum('jlk,jld->jkd', oh, P4[:, :L].astype(np.float64)); spt = np.hypot(S[..., 0], S[..., 1])
    z = np.min(spt, 1) / np.maximum(spt.sum(1), 1e-9)
    yy = rap(S); ph = np.arctan2(S[..., 1], S[..., 0]); dR = np.hypot(yy[:, 0] - yy[:, 1], wrap(ph[:, 0] - ph[:, 1]))
    T = S.sum(1); be = T[:, :3] / np.maximum(T[:, 3:], 1e-9); a = boostv(S[:, 0], be)
    na = a[:, :3] / np.maximum(np.linalg.norm(a[:, :3], axis=1, keepdims=True), 1e-12); nh = be / np.maximum(np.linalg.norm(be, axis=1, keepdims=True), 1e-12)
    cos = np.abs((na * nh).sum(1)); ms = np.sqrt(np.maximum(m2(S), 0))
    return dict(z=z, dR=dR, cos=cos, msub_hi=ms.max(1), msub_lo=ms.min(1), S=S)


def observables(P4, nn):
    from interp import h4_top_wmass as H
    P4 = np.ascontiguousarray(P4, np.float32); nn = nn.astype(np.int64); F = np.ascontiguousarray(P4.astype(np.float64))
    O = H._obs(P4, nn); LAB = kt2_labels(P4, nn); pv = prong_vars(P4, nn, LAB)
    zg = zg_batch(F, nn, 0.1); d2 = d2_batch(F, nn); S = F.sum(1)
    return dict(m=O['m'], pt=np.hypot(S[:, 0], S[:, 1]), mSD=O['mSD'], tau21=O['tau21'], tau32=O['tau32'], N=O['n'].astype(float), D2=d2,
                dR=pv['dR'], z=pv['z'], cos=pv['cos'], zg=zg[:, 0], Rg=zg[:, 1], msub_hi=pv['msub_hi'], msub_lo=pv['msub_lo'])


# ================================================================================================================ edits
def label_sep(P, lab):
    """rapidity-azimuth separation of the two label-defined prongs of one jet"""
    Va, Vb = P[lab == 0].sum(0), P[lab == 1].sum(0)
    return float(np.hypot(rap(Va) - rap(Vb), wrap(np.arctan2(Va[1], Va[0]) - np.arctan2(Vb[1], Vb[0]))))


def edit_one(P, lab, dc, rng):
    """D, cphi, TN for one jet (P (n, 4) float64, labels 0/1); returns dict kind -> P_new or None, and info"""
    s0, s1 = lab == 0, lab == 1
    if s0.sum() < 1 or s1.sum() < 1: return None, dict(reason='empty_prong')
    Va = P[s0].sum(0); T = P.sum(0); be = T[:3] / T[3]; bn = np.linalg.norm(be)
    if bn < 1e-6: return None, dict(reason='at_rest')
    nh = be / bn; na = boost(Va, be)[:3]; na = na / np.linalg.norm(na); cs = float(na @ nh); npl = na if cs >= 0 else -na
    c0 = abs(cs); c1 = float(np.clip(c0 + dc, 0.0, CMAX)); al = float(np.arccos(min(c0, 1.0)) - np.arccos(c1))
    k = np.cross(npl, nh)
    if np.linalg.norm(k) < 1e-9: return None, dict(reason='collinear')
    Pr = boost(P, be)

    def apply(R):
        Q = Pr.copy(); Q[:, :3] = Q[:, :3] @ R.T; return boost(Q, -be)
    out = {'D': apply(rodrigues(k, al)), 'cphi': apply(rodrigues(nh, rng.choice([-1.0, 1.0]) * abs(al)))}
    pt0, e0, p0, m0 = ptetaphim(P); _, e1, p1, _ = ptetaphim(out['D'])
    DS = float((pt0 * np.hypot(e1 - e0, wrap(p1 - p0))).sum() / pt0.sum()); th = rng.uniform(0, 2 * np.pi)
    out['TN'] = p4(pt0, e0 + DS * np.cos(th), p0 + DS * np.sin(th), m0)
    return out, dict(reason='ok', c0=c0, c1=c1, alpha=al, disp=DS)


def prong_offsets_ok(P, lab):
    """both label-defined prong centres within DRMAX of the jet axis"""
    T = P.sum(0); ej, pj = rap(T), np.arctan2(T[1], T[0]); ok = True
    for k in (0, 1):
        V = P[lab == k].sum(0); ok &= bool(np.hypot(rap(V) - ej, wrap(np.arctan2(V[1], V[0]) - pj)) <= DRMAX)
    return ok


def cside_candidates(P, lab, dR_target, m_target):
    """(len(RATIOS), n, 4) candidates at fixed constituent pT: prongs moved rigidly along their joining line to separation
    dR_target (in pseudorapidity-azimuth of the prong centres), offsets inside prong k scaled by s_k (s_b / s_a = ratio), s_a
    solved by bisection so that the jet invariant mass is m_target; also the mass error per candidate"""
    pt, eta, phi, m = ptetaphim(P); sel = [lab == 0, lab == 1]
    V = [P[s].sum(0) for s in sel]; ce = [ptetaphim(v)[1] for v in V]; cp = [ptetaphim(v)[2] for v in V]
    w = np.array([pt[s].sum() for s in sel]); dvec = np.array([ce[1] - ce[0], wrap(cp[1] - cp[0])]); d0 = label_sep(P, lab)
    if d0 < 1e-6 or np.linalg.norm(dvec) < 1e-6: return None, None
    lam = dR_target / d0                                                               # same separation measure as the target
    C = np.array([ce[0], cp[0]]) + dvec * w[1] / w.sum()                               # pT-weighted centre (frame of prong a)
    cen_old = [np.array([ce[0], cp[0]]), np.array([ce[0], cp[0]]) + dvec]
    cen_new = [C + lam * (c - C) for c in cen_old]
    off = np.stack([eta - np.where(sel[0], ce[0], ce[1]), wrap(phi - np.where(sel[0], cp[0], cp[1]))], -1)   # (n, 2)
    cn = np.where(sel[0][:, None], cen_new[0][None], cen_new[1][None])                                     # (n, 2)

    def build(sa):   # sa (R,) -> (R, n, 4)
        sk = np.where(sel[0][None, :], sa[:, None], (sa * RATIOS)[:, None])                                 # (R, n)
        e = cn[None, :, 0] + sk * off[None, :, 0]; f = cn[None, :, 1] + sk * off[None, :, 1]
        return p4(pt[None], e, f, m[None])

    def mass(sa):
        S = build(sa).sum(1); return np.sqrt(np.maximum(m2(S), 0))
    lo = np.full(len(RATIOS), 0.02); hi = np.full(len(RATIOS), 6.0)
    flo, fhi = mass(lo) - m_target, mass(hi) - m_target; ok = (flo <= 0) & (fhi >= 0)
    for _ in range(45):
        mid = 0.5 * (lo + hi); fm = mass(mid) - m_target; lo = np.where(fm < 0, mid, lo); hi = np.where(fm < 0, hi, mid)
    sa = 0.5 * (lo + hi); Pc = build(sa); err = np.where(ok, np.abs(np.sqrt(np.maximum(m2(Pc.sum(1)), 0)) - m_target), np.inf)
    return Pc, err


def run_edits(P4, nn, LAB, O0, rows, seed):
    """D / cphi / TN for every dc on rows (jets with >= 3 particles); then c-side matched to D (needs D observables)"""
    J, L = P4.shape[:2]
    E = {(dc, k): dict(P=P4.copy(), conv=np.zeros(J, bool)) for dc in DCS for k in VARS}
    INFO = {dc: dict(c0=np.full(J, np.nan), c1=np.full(J, np.nan), alpha=np.full(J, np.nan), disp=np.full(J, np.nan), reasons={}) for dc in DCS}
    t = time.time()
    for j in rows:
        n = nn[j]; P = P4[j, :n].astype(np.float64); lab = LAB[j, :n]
        for i, dc in enumerate(DCS):
            rng = np.random.default_rng([seed, int(j), i]); r, inf = edit_one(P, lab, dc, rng); I = INFO[dc]
            I['reasons'][inf['reason']] = I['reasons'].get(inf['reason'], 0) + 1
            if r is None: continue
            for k in ('c0', 'c1', 'alpha', 'disp'): I[k][j] = inf[k]
            okD = prong_offsets_ok(r['D'], lab); okP = prong_offsets_ok(r['cphi'], lab)
            if not okD: I['reasons']['D_prong_outside_0.75'] = I['reasons'].get('D_prong_outside_0.75', 0) + 1
            for kind, okk in (('D', okD), ('cphi', okP), ('TN', okD)):
                E[(dc, kind)]['P'][j, :n] = r[kind].astype(np.float32); E[(dc, kind)]['conv'][j] = okk
    log(f'  D / cphi / TN edits on {len(rows)} jets in {time.time() - t:.0f}s: ' + json.dumps({dc: INFO[dc]['reasons'] for dc in DCS}))
    OBS = {}
    for dc in DCS:
        for kind in ('D', 'cphi'):
            OBS[(dc, kind)] = observables(E[(dc, kind)]['P'], nn); log(f'  observables dc={dc:+.2f} {kind}')
    # ---- c-side
    from interp.h4_top_wmass import _obs
    for dc in DCS:
        t = time.time(); conv = E[(dc, 'D')]['conv']; cand = np.zeros((len(RATIOS), J, L, 4), np.float32); cerr = np.full((len(RATIOS), J), np.inf)
        why = {}
        for j in np.flatnonzero(conv):
            n = nn[j]; P = P4[j, :n].astype(np.float64); lab = LAB[j, :n]
            dRt = label_sep(E[(dc, 'D')]['P'][j, :n].astype(np.float64), lab)
            Pc, err = cside_candidates(P, lab, dRt, O0['m'][j])
            if Pc is None: why['degenerate'] = why.get('degenerate', 0) + 1; continue
            cand[:, j, :n] = Pc.astype(np.float32); cerr[:, j] = err
        cerr = np.where(cerr < 1.0, cerr, np.inf)
        tau = np.full((len(RATIOS), J), np.nan)
        for r in range(len(RATIOS)):
            okr = np.isfinite(cerr[r])
            if okr.any():
                Fc = np.ascontiguousarray(cand[r][okr].astype(np.float64)); tau[r, okr] = tau21_batch(Fc, nn[okr].astype(np.int64))
        tD = OBS[(dc, 'D')]['tau21']; cost = np.where(np.isfinite(cerr), np.abs(tau - tD[None]), np.inf); best = np.argmin(cost, 0)
        have = np.isfinite(cost[best, np.arange(J)])
        Pc = P4.copy(); Pc[have] = cand[best[have], np.flatnonzero(have)]
        why['no_mass_solution'] = int((conv & ~have).sum()) - why.get('degenerate', 0); why['built'] = int(have.sum())
        E[(dc, 'cside')]['P'] = Pc; E[(dc, 'cside')]['conv'] = have
        OBS[(dc, 'cside')] = observables(Pc, nn); Oc = OBS[(dc, 'cside')]; OD = OBS[(dc, 'D')]
        match = have & (np.abs(Oc['m'] - O0['m']) < 1.0) & (np.abs(Oc['tau21'] - OD['tau21']) <= MATCH * np.abs(OD['tau21'])) & \
            (np.abs(Oc['dR'] - OD['dR']) <= MATCH * np.abs(OD['dR'])) & prong_ok_batch(Pc, nn, LAB)
        E[(dc, 'cside')]['conv'] = match; why['matched'] = int(match.sum())
        INFO[dc]['cside'] = why; INFO[dc]['cside_ratio_pick'] = np.bincount(best[have], minlength=len(RATIOS)).tolist()
        log(f'  c-side dc={dc:+.2f}: {why} in {time.time() - t:.0f}s; ratio picks {INFO[dc]["cside_ratio_pick"]}')
    return E, OBS, INFO


def prong_ok_batch(P4, nn, LAB):
    out = np.zeros(len(nn), bool)
    for j in range(len(nn)):
        n = nn[j]
        if n >= 3 and (LAB[j, :n] == 0).any() and (LAB[j, :n] == 1).any(): out[j] = prong_offsets_ok(P4[j, :n].astype(np.float64), LAB[j, :n])
    return out


def edit_all(P4, nn, rows, seed, tag):
    """observables, labels, edits; returns (P variants dict, OBS dict incl. 'id' and 'TN' (= id observables shifted), acceptance)"""
    LAB = kt2_labels(P4, nn); O0 = observables(P4, nn); log(f'[{tag}] unedited observables on {len(nn)} jets')
    E, OBS, INFO = run_edits(P4, nn, LAB, O0, rows, seed)
    OBS['id'] = O0
    for dc in DCS: OBS[(dc, 'TN')] = observables(E[(dc, 'TN')]['P'], nn)
    log(f'[{tag}] TN observables done')
    return E, OBS, INFO, LAB


# ================================================================================================================ references
def ref_X(O, kind):
    base = [O['m'], O['pt'], O['tau21'], O['D2'], O['N'], O['dR']]
    if kind == 'MLPx': return np.column_stack(base)
    if kind == 'MLPz': return np.column_stack(base + [O['zg']])
    return np.column_stack([O['zg'], O['zg'] ** 2])


def fit_refs(Opool, ypool, scores, cls_of):
    """per score (signal, background classes): MLPx / MLPz (random_state s) and LRzg (bootstrap resample s), s < NSEED"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    R, info = {}, {}
    for sc, (sig, bkg) in scores.items():
        sel = np.isin(ypool, [cls_of[sig]] + [cls_of[b] for b in bkg]); Os = {k: v[sel] for k, v in Opool.items()}; y = (ypool[sel] == cls_of[sig]).astype(int)
        w = np.where(y == 1, 1.0, (y == 1).sum() / max((y == 0).sum(), 1))   # background classes weighted to the signal count
        tr = np.random.default_rng(4520).random(len(y)) < 0.8
        for kind in REFS:
            Xf = ref_X(Os, kind)
            for s in range(NSEED):
                scl = StandardScaler().fit(Xf[tr])
                if kind == 'LRzg':
                    bi = np.random.default_rng(4530 + s).choice(np.flatnonzero(tr), tr.sum(), replace=True)
                    mdl = LogisticRegression(max_iter=2000).fit(scl.transform(Xf[bi]), y[bi], sample_weight=w[bi])
                else:
                    # equal class weight through resampling the background to the signal count (MLPClassifier has no weights)
                    rs = np.random.default_rng(4540 + s); it = np.flatnonzero(tr)
                    pos, neg = it[y[it] == 1], it[y[it] == 0]; neg = rs.choice(neg, min(len(neg), len(pos)), replace=False)
                    ii = np.concatenate([pos, neg])
                    mdl = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=s, max_iter=300).fit(scl.transform(Xf[ii]), y[ii])
                pr = mdl.predict_proba(scl.transform(Xf[~tr]))[:, 1]
                from interp.h4_top_wmass import _auc
                R[(sc, kind, s)] = (mdl, scl); info[f'{sc}|{kind}|{s}'] = dict(heldout_AUC=float(_auc(pr, y[~tr])))
            log(f'  ref {sc} {kind}: held-out AUC ' + ', '.join(f"{info[f'{sc}|{kind}|{s}']['heldout_AUC']:.4f}" for s in range(NSEED)))
    return R, info


def ref_scores(R, O):
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return {f'{kind}@{s}|{sc}': lo(mdl.predict_proba(scl.transform(ref_X(O, kind)))[:, 1]).astype(np.float32) for (sc, kind, s), (mdl, scl) in R.items()}


def save_obs(path, OBS, E, INFO, y, extra):
    D = {'y': y}
    for key, O in OBS.items():
        nm = 'id' if key == 'id' else f'{key[0]}|{key[1]}'
        for q in OBSK: D[f'O|{nm}|{q}'] = np.asarray(O[q], np.float32)
    for (dc, kind), e in E.items(): D[f'acc|{dc}|{kind}'] = e['conv']
    for dc, I in INFO.items():
        for k in ('c0', 'c1', 'alpha', 'disp'): D[f'info|{dc}|{k}'] = I[k].astype(np.float32)
    D.update(extra); np.savez_compressed(path, **D)


# ================================================================================================================ JetClass parts
def part_jcedit():
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    from interp.h41_colour_flow import read_window
    import interp.h41_colour_flow as H41
    from interp.part_core import cat, kin, PK
    H41.log = log
    meta = dict(node='H45', part='jcedit', smoke=SMOKE, dcs=DCS, sample={})
    Js = []
    for k, c in enumerate(JC_CLASSES):
        J, inf = read_window(X.jc_file(c, 100), PER, 4501 + k); Js.append(J); meta['sample'][c] = inf
    J = cat(Js); y = J['label'].astype(int); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32)
    log(f'jc sample {len(nn)} jets; labels {np.bincount(y, minlength=10).tolist()}; median jet pT {np.median(J["jet_pt"]):.0f} GeV; median n {np.median(nn):.0f}')
    rows = np.flatnonzero(nn >= 3)
    E, OBS, INFO, LAB = edit_all(P4, nn, rows, 4510, 'jc')
    D = {'nn': nn, 'y': y, 'LAB': LAB.astype(np.int16), 'id': P4}
    for (dc, kind), e in E.items(): D[f'P|{dc}|{kind}'] = e['P']
    for k in PK + ['pmass']: D[f'base|{k}'] = J[k]
    for k in ('jet_eta', 'jet_phi', 'jet_pt', 'label'): D[f'base|{k}'] = J[k]
    RES.mkdir(parents=True, exist_ok=True); np.savez_compressed(RES / f'{TAG}_jc_inputs.npz', **D); log(f'wrote {TAG}_jc_inputs.npz')
    # ---- references on train rows
    Os, ys = [], []
    for k, c in enumerate(JC_CLASSES):
        Jr, inf = read_window(TRAIN / f'{JC_PRE[c]}_000.root', NREF, 4506 + k); meta.setdefault('ref_rows', {})[c] = inf
        Pr, nr = X.jc_P4(Jr); Pr = H._identity(Pr, nr).astype(np.float32); Os.append(observables(Pr, nr)); ys.append(Jr['label'].astype(int))
        log(f'ref observables {c}: {len(nr)} jets')
    Op = {q: np.concatenate([o[q] for o in Os]) for q in OBSK}; yp = np.concatenate(ys)
    R, meta['ref_fit'] = fit_refs(Op, yp, JC_SCORES, JC_CLASSES)
    extra = {f'pool|{q}': Op[q].astype(np.float32) for q in OBSK}; extra['pool|y'] = yp
    for key, O in OBS.items():
        nm = 'id' if key == 'id' else f'{key[0]}|{key[1]}'
        for k, v in ref_scores(R, O).items(): extra[f'R|{nm}|{k}'] = v
    save_obs(RES / f'{TAG}_jc_obs.npz', OBS, E, INFO, y, extra)
    meta['edit_info'] = {str(dc): dict(reasons=I['reasons'], cside=I.get('cside'), cside_ratio_pick=I.get('cside_ratio_pick')) for dc, I in INFO.items()}
    (RES / f'{TAG}_jc_edit.json').write_text(json.dumps(jsonable(meta), indent=1)); log(f'wrote {TAG}_jc_obs.npz, {TAG}_jc_edit.json')


def part_jcnets():
    import torch
    from interp.h23_axis_audit import Nets
    from interp.part_core import ParTNetwork
    import interp.h23_axis_audit as H23
    H23.log = log
    Z = np.load(RES / f'{TAG}_jc_inputs.npz'); nn = Z['nn']; base = {k.split('|', 1)[1]: Z[k] for k in Z.files if k.startswith('base|')}
    N = Nets(top=False); names = list(JC_NETS)
    for s in range(NPERMNET):
        netp = ParTNetwork('full'); g = torch.Generator().manual_seed(5 + s)
        with torch.no_grad():
            for p in netp.model.parameters():
                if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
        N.jc[f'ParT_perm{s}'] = ('part', 'full', netp); names.append(f'ParT_perm{s}')
    log(f'models {names}')
    out = {}
    keys = ['id'] + [k for k in Z.files if k.startswith('P|')]
    for k in keys:
        P = Z[k]; lg = {}; t = time.time()
        N.score('jc', P, nn, base, axis(P, nn), None, jc_names=names, top=False, logits=lg)
        nm = 'id' if k == 'id' else k[2:]
        for m, L in lg.items(): out[f'L|{nm}|{m}'] = L.astype(np.float32)
        log(f'scored {nm} in {time.time() - t:.0f}s')
    lg = {}; N.score('jc', Z['id'], nn, base, (base['jet_eta'].astype(np.float64), base['jet_phi'].astype(np.float64)), None, jc_names=names, top=False, logits=lg)
    for m, L in lg.items(): out[f'L|id_st|{m}'] = L.astype(np.float32)
    y = Z['y']
    for m in names:
        a_rc = float((out[f'L|id|{m}'].argmax(1) == y).mean()); a_st = float((out[f'L|id_st|{m}'].argmax(1) == y).mean())
        log(f'{m}: 10-class accuracy on the sample rc {a_rc:.4f} st {a_st:.4f}')
    np.savez_compressed(RES / f'{TAG}_jc_logits.npz', **out); log(f'wrote {TAG}_jc_logits.npz')


# ================================================================================================================ hls4ml parts
def hls_draw():
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y = np.asarray(d['y']).astype(int)
    rng = np.random.default_rng(4504)   # = interp.jedi_core.sample(per_class=PER, seed=4504) rows
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), PER, replace=False) for c in range(5)]))
    return np.asarray(d['x'][idx], np.float32), y[idx], idx


def model_x(P4, nn, ax):
    """(J, 64, 3) (pT, deta, dphi) about ax, constituents sorted by pT (descending); padding 0"""
    P, _ = sort_pt(P4.astype(np.float64), nn); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    pt = np.hypot(P[..., 0], P[..., 1]); ptc = np.where(real, pt, 1.0)
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / ptc) - ax[0][:, None]; phi = wrap(np.arctan2(P[..., 1], P[..., 0]) - ax[1][:, None])
    return (np.stack([pt, eta, phi], -1) * real[..., None])[:, :64].astype(np.float32)


def part_hedit():
    import h5py
    from interp import h4_top_wmass as H
    x, y, idx = hls_draw(); real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
    P4 = H._identity(p4(x[..., 0].astype(np.float64), x[..., 1], x[..., 2], 0.0) * real[..., None], nn).astype(np.float32)
    log(f'hls sample {len(y)} jets (rng 4504), per class {np.bincount(y).tolist()}; identity rebuild max |d pT| '
        f'{np.abs(np.hypot(P4[..., 0], P4[..., 1]) - x[..., 0]).max():.2e}')
    rows = np.flatnonzero((nn >= 3) & (y != 4))
    E, OBS, INFO, LAB = edit_all(P4, nn, rows, 4511, 'hls')
    D = {'y': y, 'idx': idx, 'nn': nn, 'X|id_st': x, 'X|id': model_x(P4, nn, axis(P4, nn)), 'P|id': P4}
    for (dc, kind), e in E.items(): D[f'X|{dc}|{kind}'] = model_x(e['P'], nn, axis(e['P'], nn)); D[f'P|{dc}|{kind}'] = e['P']
    RES.mkdir(parents=True, exist_ok=True); np.savez_compressed(RES / f'{TAG}_hls_inputs.npz', **D); log(f'wrote {TAG}_hls_inputs.npz')
    meta = dict(node='H45', part='hedit', smoke=SMOKE, sample=dict(file='data/_jets/n64/full_test.npz', per_class=PER, seed=4504, N=int(len(y))))
    # ---- references on the hls4ml train archive
    t0 = time.time()
    with h5py.File(V / 'data' / 'train_meta.h5', 'r') as f: ya = f['label'][:].astype(np.int64)
    dev = np.asarray(np.load(V / 'data' / 'splits.npz')['dev'], np.int64); isdev = np.zeros(len(ya), bool); isdev[dev] = True
    rr = np.random.default_rng(4509); it = np.sort(np.concatenate([rr.choice(np.flatnonzero(~isdev & (ya == c)), NREF, replace=False) for c in range(4)]))
    xr = np.asarray(np.load(V / 'data' / 'train_ptetaphi_f16.npy')[it, :64], np.float32); yr = ya[it]
    log(f'reference rows {len(it)} (W, Z, q, g; rng 4509) read in {time.time() - t0:.0f}s')
    rl = xr[..., 0] > 0; nr = rl.sum(1).astype(np.int64)
    Pr = H._identity((p4(xr[..., 0].astype(np.float64), xr[..., 1], xr[..., 2], 0.0) * rl[..., None]).astype(np.float32), nr).astype(np.float32)
    Op = observables(Pr, nr); log('reference observables done')
    R, meta['ref_fit'] = fit_refs(Op, yr, HLS_SCORES, HLS_CLASSES)
    extra = {f'pool|{q}': Op[q].astype(np.float32) for q in OBSK}; extra['pool|y'] = yr
    for key, O in OBS.items():
        nm = 'id' if key == 'id' else f'{key[0]}|{key[1]}'
        for k, v in ref_scores(R, O).items(): extra[f'R|{nm}|{k}'] = v
    save_obs(RES / f'{TAG}_hls_obs.npz', OBS, E, INFO, y, extra)
    meta['edit_info'] = {str(dc): dict(reasons=I['reasons'], cside=I.get('cside'), cside_ratio_pick=I.get('cside_ratio_pick')) for dc, I in INFO.items()}
    (RES / f'{TAG}_hls_edit.json').write_text(json.dumps(jsonable(meta), indent=1)); log(f'wrote {TAG}_hls_obs.npz, {TAG}_hls_edit.json')


def hls_score_all(models, tag, known):
    Z = np.load(RES / f'{TAG}_hls_inputs.npz'); y = Z['y']; out = {}; info = {}
    keys = [k for k in Z.files if k.startswith('X|')]
    for name, fn in models.items():
        t = time.time()
        for k in keys: out[f'L|{k[2:]}|{name}'] = np.asarray(fn(np.asarray(Z[k], np.float32)), np.float32)
        info[name] = dict(acc5_rc=float((out[f'L|id|{name}'].argmax(1) == y).mean()), acc5_st=float((out[f'L|id_st|{name}'].argmax(1) == y).mean()),
                          known=known.get(name))
        log(f'{tag} {name}: {len(keys)} arrays in {time.time() - t:.0f}s; 5-class accuracy rc {info[name]["acc5_rc"]:.4f} st {info[name]["acc5_st"]:.4f} (known {known.get(name)})')
    np.savez_compressed(RES / f'{TAG}_hls_logits_{tag}.npz', **out)
    (RES / f'{TAG}_hls_models_{tag}.json').write_text(json.dumps(jsonable(info), indent=1)); log(f'wrote {TAG}_hls_logits_{tag}.npz')


def part_hsmall():
    import torch
    from interp.h1_irc_panel import load_small, torch_logits
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    models, known = {}, {}
    for tag, run in (('dsets', 'dsets-d64-n64'), ('jedis', 'jedis-d64-n64'), ('tf', 'tf-2b-d32-n64')):
        d = V / 'models' / 'small' / run / 'seed0'; net, sh, sc = load_small(d)
        models[tag] = (lambda x, net=net, sh=sh, sc=sc: torch_logits(net, x.copy(), sh, sc))
        tj = json.loads((d / 'test.json').read_text()); known[tag] = tj.get('accuracy', tj.get('acc'))
    for s in range(3):
        net, sh, sc = load_small(V / 'models' / 'small' / 'dsets-d64-n64' / 'seed0'); g = torch.Generator().manual_seed(7 + s)
        with torch.no_grad():
            for p in net.parameters():
                f = p.data.flatten(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p))
        models[f'dsets_perm{s}'] = (lambda x, net=net, sh=sh, sc=sc: torch_logits(net, x.copy(), sh, sc))
    hls_score_all(models, 'small', known)


def part_hjedi():
    from interp.jedi_core import Jedi
    known = json.loads((V / 'data' / 'official_acc.json').read_text())['3-feature']['64']
    nets = {'jedi': Jedi(64, '3-feature')}
    for s in range(3): nets[f'jedi_redrawn{s}'] = Jedi(64, '3-feature', untrained=True, seed=s)
    hls_score_all({k: (lambda x, n=n: n.run(x)['logits']) for k, n in nets.items()}, 'jedi', {'jedi': known})


# ================================================================================================================ statistics
def boot(x, bi):
    return float(x.mean()), np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist()


def perm2(x, rng, n=NPERM):
    x = np.asarray(x, float)
    if len(x) == 0 or np.all(np.abs(x) < 1e-12): return 1.0
    obs = abs(x.mean()); cnt = 0
    for s0 in range(0, n, 1000):
        k = min(1000, n - s0); sg = rng.choice([-1.0, 1.0], (k, len(x))); cnt += int(np.sum(np.abs((sg * x).mean(1)) >= obs - 1e-15))
    return float((cnt + 1) / (n + 1))


def tp(x):
    """two-sided one-sample t-test p for mean(x) = 0 (used for the Holm families: the sign-flip permutation p is floored at
    1 / (NPERM + 1), which bounds Holm-adjusted p from below at n_tests / (NPERM + 1))"""
    from scipy.stats import ttest_1samp
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x) < 1e-12: return 1.0
    return float(max(ttest_1samp(x, 0.0).pvalue, 1e-300))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(rate_control=float(a.mean()), rate_edit=float(b.mean()), control_only=n10, edit_only=n01,
                p=float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 > 0 else 1.0)


def c2st(Xa, Xb, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from interp.h4_top_wmass import _auc
    Xc = np.concatenate([Xa, Xb]); yy = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))].astype(int); p = np.zeros(len(yy))
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(Xc, yy):
        p[te] = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xc[tr], yy[tr]).predict_proba(Xc[te])[:, 1]
    return float(_auc(p, yy))


C2F = ('pt', 'm', 'mSD', 'tau21', 'tau32', 'D2', 'N', 'dR', 'z', 'zg')


def matched_c2st(Oe, Op):
    from scipy.spatial import cKDTree
    k2 = lambda O: np.stack([O['m'], O['cos']], 1)
    Zp = k2(Op); mu, sd = Zp.mean(0), Zp.std(0) + 1e-9; _, nb = cKDTree((Zp - mu) / sd).query((k2(Oe) - mu) / sd, k=1)
    Xe = np.column_stack([Oe[q] for q in C2F]); Xp = np.column_stack([Op[q][nb] for q in C2F])
    return c2st(Xe, Xp)


def lse(a, b):
    m = np.maximum(a, b); return m + np.log(np.exp(a - m) + np.exp(b - m))


def score_of(ds, L, sc):
    if ds == 'jc': return {'WQ': L[:, 7] - L[:, 0], 'ZQ': L[:, 6] - L[:, 0], 'WZ': L[:, 7] - L[:, 6]}[sc]
    return {'WQG': L[:, 2] - lse(L[:, 0], L[:, 1]), 'ZQG': L[:, 3] - lse(L[:, 0], L[:, 1]), 'WZ': L[:, 2] - L[:, 3]}[sc]


def stats_dataset(ds):
    from interp.h4_top_wmass import _auc
    z_ = np.load(RES / f'{TAG}_{ds}_obs.npz'); Zo = {k: z_[k] for k in z_.files}; y = Zo['y']
    if ds == 'jc':
        z_ = np.load(RES / f'{TAG}_jc_logits.npz'); Zl = {k: z_[k] for k in z_.files}; cls_of, scores, cells, nets = JC_CLASSES, JC_SCORES, JC_CELLS, JC_NETS
        nulls = sorted({k.split('|')[-1] for k in Zl if '|ParT_perm' in k})
    else:
        Zl = {}
        for t in ('small', 'jedi'):
            z = np.load(RES / f'{TAG}_hls_logits_{t}.npz'); Zl.update({k: z[k] for k in z.files})
        cls_of, scores, cells, nets = HLS_CLASSES, HLS_SCORES, HLS_CELLS, HLS_NETS
        nulls = sorted({k.split('|')[-1] for k in Zl if ('_perm' in k or '_redrawn' in k)})
    names = {v: k for k, v in cls_of.items()}; cname = np.array([names.get(int(v), str(v)) for v in y])
    O = lambda nm, q: Zo[f'O|{nm}|{q}'].astype(np.float64)
    refk = sorted({k.split('|')[2] for k in Zo if k.startswith('R|id|')})          # e.g. MLPx@0
    out = dict(dataset=ds, n_per_class={c: int((cname == c).sum()) for c in cls_of}, nets=nets, nulls=nulls, refs=refk)

    def S(model, v, sc):
        if model in refk or '@' in model: return Zo[f'R|{v}|{model}|{sc}'].astype(np.float64)
        return score_of(ds, Zl[f'L|{v}|{model}'].astype(np.float64), sc)
    # ---- unedited accuracy
    out['unedited'] = {}
    for m in nets + nulls:
        L = Zl[f'L|id|{m}']; Lst = Zl[f'L|id_st|{m}']; d = dict(acc_rc=float((L.argmax(1) == y).mean()), acc_st=float((Lst.argmax(1) == y).mean()))
        for c in cls_of:
            if (cname == c).any(): d[f'eff_argmax_{c}'] = float((L[cname == c].argmax(1) == cls_of[c]).mean())
        for sc, (sig, bkg) in scores.items():
            sel = np.isin(cname, [sig] + bkg); d[f'AUC_{sc}'] = float(_auc(score_of(ds, L, sc)[sel], (cname[sel] == sig).astype(int)))
        if ds == 'hls':
            sel = y != 4; d['acc_rc_WZqg_only'] = float((L[sel].argmax(1) == y[sel]).mean())
        out['unedited'][m] = d
    out['ref_unedited_AUC'] = {}
    for r in refk:
        for sc, (sig, bkg) in scores.items():
            sel = np.isin(cname, [sig] + bkg); out['ref_unedited_AUC'][f'{r}|{sc}'] = float(_auc(S(r, 'id', sc)[sel], (cname[sel] == sig).astype(int)))
    log(f'[{ds}] unedited: ' + json.dumps({m: {k: round(v, 4) for k, v in d.items()} for m, d in out['unedited'].items() if m in nets}))
    # ---- thresholds (eps_S = 0.5 of the score's signal class)
    thr = {(m, sc): float(np.median(S(m, 'id', sc)[cname == scores[sc][0]])) for m in nets + nulls for sc in scores}
    rng = np.random.default_rng(4550); res = {}; tests = []; tests_x = []
    out['observables'] = {}; out['edit'] = {}
    for dc in DCS:
        acc = {k: Zo[f'acc|{dc}|{k}'] for k in VARS}
        base_p = acc['D'] & acc['cphi']; base_s = base_p & acc['cside']
        for c in cls_of:
            if c == 't' or not (cname == c).any(): continue
            rp = np.flatnonzero(base_p & (cname == c)); rs = np.flatnonzero(base_s & (cname == c))
            ed = dict(n_class=int((cname == c).sum()), n_D=int((acc['D'] & (cname == c)).sum()), n_D_cphi=int(len(rp)), n_D_cphi_cside=int(len(rs)),
                      c0_mean=float(np.nanmean(Zo[f'info|{dc}|c0'][rp])) if len(rp) else None,
                      c1_target_mean=float(np.nanmean(Zo[f'info|{dc}|c1'][rp])) if len(rp) else None,
                      disp_median=float(np.nanmedian(Zo[f'info|{dc}|disp'][rp])) if len(rp) else None)
            out['edit'][f'{dc}|{c}'] = ed
            ob = {}
            for q in ('z', 'cos', 'zg', 'dR', 'tau21', 'D2', 'm', 'mSD', 'pt', 'Rg', 'msub_hi', 'msub_lo'):
                ob[q] = dict(id_mean=float(O('id', q)[rs].mean()) if len(rs) else None)
                for v in ('D', 'cphi', 'cside', 'TN'):
                    rr_ = rs if len(rs) >= 5 else rp
                    if len(rr_) < 5: continue
                    dq = O(f'{dc}|{v}', q)[rr_] - O('id', q)[rr_]; bi = rng.integers(0, len(rr_), (500, len(rr_)))
                    ob[q][v] = dict(mean=float(dq.mean()), ci95=np.percentile(dq[bi].mean(1), [2.5, 97.5]).tolist(), median_abs=float(np.median(np.abs(dq))))
            out['observables'][f'{dc}|{c}'] = ob
            if 'D' in ob['z'] and 'cside' in ob['z']:
                log(f'[{ds}] dc={dc:+.2f} {c}: n D&cphi {len(rp)} +cside {len(rs)}; dz D {ob["z"]["D"]["mean"]:+.4f} cside {ob["z"]["cside"]["mean"]:+.4f} cphi {ob["z"]["cphi"]["mean"]:+.4f}; '
                    f'dcos D {ob["cos"]["D"]["mean"]:+.3f}; dzg D {ob["zg"]["D"]["mean"]:+.4f} cside {ob["zg"]["cside"]["mean"]:+.4f}; dDR D {ob["dR"]["D"]["mean"]:+.4f} cside {ob["dR"]["cside"]["mean"]:+.4f}; '
                    f'dtau21 D {ob["tau21"]["D"]["mean"]:+.4f} cside {ob["tau21"]["cside"]["mean"]:+.4f} cphi {ob["tau21"]["cphi"]["mean"]:+.4f}; dD2 D {ob["D2"]["D"]["mean"]:+.3f} cside {ob["D2"]["cside"]["mean"]:+.3f}; '
                    f'dm D {ob["m"]["D"]["mean"]:+.3f} cside {ob["m"]["cside"]["mean"]:+.3f}')
    for (c, sc) in cells:
        for dc in DCS:
            acc = {k: Zo[f'acc|{dc}|{k}'] for k in VARS}
            rp = np.flatnonzero(acc['D'] & acc['cphi'] & (cname == c)); rs = np.flatnonzero(acc['D'] & acc['cphi'] & acc['cside'] & (cname == c))
            key = f'{c}|{sc}|{dc}'; R = dict(n_D_cphi=int(len(rp)), n_D_cphi_cside=int(len(rs)), models={})
            if len(rp) < 10: res[key] = R; continue
            bip = rng.integers(0, len(rp), (B, len(rp))); bis = rng.integers(0, len(rs), (B, len(rs))) if len(rs) >= 10 else None
            dlt = {}
            for m in nets + nulls + refk:
                s0 = S(m, 'id', sc); d = {v: S(m, f'{dc}|{v}', sc) - s0 for v in VARS}; dlt[m] = d
                Mm = {}
                for v in VARS: Mm[f'd{v}'], Mm[f'd{v}_ci95'] = boot(d[v][rp], bip)
                Mm['D_minus_cphi'], Mm['D_minus_cphi_ci95'] = boot((d['D'] - d['cphi'])[rp], bip)
                if bis is not None:
                    Mm['D_minus_cside'], Mm['D_minus_cside_ci95'] = boot((d['D'] - d['cside'])[rs], bis)
                    Mm['cside_minus_cphi'], Mm['cside_minus_cphi_ci95'] = boot((d['cside'] - d['cphi'])[rs], bis)
                    Mm['dD_on_cside_rows'] = float(d['D'][rs].mean())
                if m in nets:
                    Mm['D_minus_cphi_p'] = perm2((d['D'] - d['cphi'])[rp], rng); Mm['D_minus_cphi_tp'] = tp((d['D'] - d['cphi'])[rp])
                    if bis is not None:
                        Mm['D_minus_cside_p'] = perm2((d['D'] - d['cside'])[rs], rng); Mm['D_minus_cside_tp'] = tp((d['D'] - d['cside'])[rs])
                if m in nets or m in nulls:
                    t_ = thr[(m, sc)]; call = lambda v, rows: S(m, 'id' if v == 'id' else f'{dc}|{v}', sc)[rows] >= t_
                    Mm['calls'] = dict(threshold=t_, pass_id=float(call('id', rp).mean()), pass_D=float(call('D', rp).mean()), pass_cphi=float(call('cphi', rp).mean()),
                                       mcnemar_D_vs_cphi=mcnemar(call('cphi', rp), call('D', rp)))
                    if bis is not None:
                        Mm['calls'].update(pass_cside=float(call('cside', rs).mean()), mcnemar_D_vs_cside=mcnemar(call('cside', rs), call('D', rs)))
                R['models'][m] = Mm
            # net vs references (per jet, seed 0) and bands
            for m in nets:
                Mm = R['models'][m]
                for r0 in ('MLPx@0', 'MLPz@0'):
                    if bis is not None:
                        xx = (dlt[m]['D'] - dlt[m]['cside'] - (dlt[r0]['D'] - dlt[r0]['cside']))[rs]
                        Mm[f'D_minus_cside_minus_{r0}'], Mm[f'D_minus_cside_minus_{r0}_ci95'] = boot(xx, bis); Mm[f'D_minus_cside_minus_{r0}_p'] = perm2(xx, rng)
                        Mm[f'D_minus_cside_minus_{r0}_tp'] = tp(xx)
                    xx = (dlt[m]['D'] - dlt[m]['cphi'] - (dlt[r0]['D'] - dlt[r0]['cphi']))[rp]
                    Mm[f'D_minus_cphi_minus_{r0}'], Mm[f'D_minus_cphi_minus_{r0}_ci95'] = boot(xx, bip); Mm[f'D_minus_cphi_minus_{r0}_p'] = perm2(xx, rng)
                    Mm[f'D_minus_cphi_minus_{r0}_tp'] = tp(xx)
                if bis is not None: tests.append((key, m, 'D_minus_cside_tp')); tests_x.append((key, m, 'D_minus_cside_minus_MLPx@0_tp'))
                tests.append((key, m, 'D_minus_cphi_tp')); tests_x.append((key, m, 'D_minus_cphi_minus_MLPx@0_tp'))
            for con in ('D_minus_cphi', 'D_minus_cside', 'dD'):
                vals = [R['models'][m][con] for m in nulls if con in R['models'][m]]
                if vals: R[f'null_band_{con}'] = [float(min(vals)), float(max(vals))]
                for base_ in REFS:
                    vals = [R['models'][r][con] for r in refk if r.startswith(base_ + '@') and con in R['models'][r]]
                    if vals: R[f'ref_range_{base_}_{con}'] = [float(min(vals)), float(max(vals))]
            res[key] = R
            msg = ' | '.join(f"{m} D-cside {R['models'][m].get('D_minus_cside', np.nan):+.3f} D-cphi {R['models'][m]['D_minus_cphi']:+.3f}" for m in nets)
            log(f'[{ds}] {key} n {len(rp)}/{len(rs)}: {msg} | null {R.get("null_band_D_minus_cside")} MLPx {R.get("ref_range_MLPx_D_minus_cside")} '
                f'MLPz {R.get("ref_range_MLPz_D_minus_cside")} LRzg {R.get("ref_range_LRzg_D_minus_cside")}')
    out['cells'] = res; out['tests'] = tests; out['tests_vs_MLPx'] = tests_x
    # ---- JEDI-linear's input moments (hls only): M0 = sum pT and m_A = sqrt(M0 M2) of the 64 inputs about each jet's own axis
    if ds == 'hls':
        Zi = np.load(RES / f'{TAG}_hls_inputs.npz'); out['input_moments'] = {}

        def mom(x):
            x = x.astype(np.float64); pt = x[..., 0]; M0 = pt.sum(1); M2 = (pt * (x[..., 1] ** 2 + x[..., 2] ** 2)).sum(1); return M0, np.sqrt(M0 * M2)
        M0i, mAi = mom(Zi['X|id'])
        for dc in DCS:
            acc = {k: Zo[f'acc|{dc}|{k}'] for k in VARS}
            for v in ('D', 'cphi', 'cside'):
                M0v, mAv = mom(Zi[f'X|{dc}|{v}'])
                for c in ('W', 'Z', 'q', 'g'):
                    rr_ = np.flatnonzero(acc['D'] & acc['cphi'] & (acc['cside'] if v == 'cside' else True) & (cname == c))
                    if len(rr_) < 5: continue
                    dl = np.log(mAv[rr_] / mAi[rr_]); d0 = M0v[rr_] / M0i[rr_] - 1; dmi = Zo[f'O|{dc}|{v}|m'][rr_] - Zo['O|id|m'][rr_]
                    out['input_moments'][f'{dc}|{v}|{c}'] = dict(n=int(len(rr_)), dlog_mA_mean=float(dl.mean()), dlog_mA_p90abs=float(np.quantile(np.abs(dl), 0.9)),
                                                                 dM0_rel_mean=float(d0.mean()), dM0_rel_p90abs=float(np.quantile(np.abs(d0), 0.9)),
                                                                 mA_id_mean=float(mAi[rr_].mean()), dm_inv_max_abs=float(np.abs(dmi).max()))
        log(f'[{ds}] input moments: ' + json.dumps({k: (round(v['dlog_mA_mean'], 4), round(v['dM0_rel_mean'], 5)) for k, v in out['input_moments'].items() if '|D|' in k}))
    # ---- dose
    out['dose'] = {}
    from scipy.stats import spearmanr
    for (c, sc) in cells:
        for m in nets + nulls + refk:
            for con in ('D_minus_cside', 'D_minus_cphi'):
                xs, ys = [], []
                for dc in DCS:
                    Mm = res.get(f'{c}|{sc}|{dc}', {}).get('models', {}).get(m, {})
                    if con in Mm: xs.append(dc); ys.append(Mm[con])
                if len(xs) >= 3: out['dose'][f'{c}|{sc}|{m}|{con}'] = dict(spearman=float(spearmanr(xs, ys)[0]), means=ys, dcs=xs)
    # ---- C2ST at dc = -0.3, +0.3
    out['c2st'] = {}
    pool = {q: Zo[f'pool|{q}'].astype(np.float64) for q in OBSK}; yp = Zo['pool|y']
    for dc in (-0.3, 0.3):
        for c in cls_of:
            if c == 't' or not (cname == c).any(): continue
            acc = {k: Zo[f'acc|{dc}|{k}'] for k in VARS}; rs = np.flatnonzero(acc['D'] & acc['cphi'] & acc['cside'] & (cname == c))
            if len(rs) < 40: continue
            sub = rs if len(rs) <= 1500 else np.sort(rng.permutation(rs)[:1500]); pm = yp == cls_of[c]; Op = {q: v[pm] for q, v in pool.items()}
            r_ = {}
            for v in ('id', 'D', 'cphi', 'cside'):
                nm = 'id' if v == 'id' else f'{dc}|{v}'; Oe = {q: O(nm, q)[sub] for q in OBSK}; r_[v] = matched_c2st(Oe, Op)
            out['c2st'][f'{dc}|{c}'] = r_; log(f'[{ds}] C2ST dc={dc:+.2f} {c} (n {len(sub)}): {r_}')
    return out


def part_stats():
    R = dict(node='H45', smoke=SMOKE, dcs=DCS, per_class=PER, n_ref_per_class=NREF, n_retrains=NSEED, bootstrap=B, permutations=NPERM, sets={})
    for ds in ('jc', 'hls'):
        if (RES / f'{TAG}_{ds}_obs.npz').exists() and ((RES / f'{TAG}_jc_logits.npz').exists() if ds == 'jc' else (RES / f'{TAG}_hls_logits_jedi.npz').exists()):
            R['sets'][ds] = stats_dataset(ds)
            for f in (f'{TAG}_{ds}_edit.json',):
                if (RES / f).exists(): R['sets'][ds]['edit_meta'] = json.loads((RES / f).read_text())
    for fam, dss, tk in (('per_dataset_jc', ['jc'], 'tests'), ('per_dataset_hls', ['hls'], 'tests'), ('combined', list(R['sets']), 'tests'),
                         ('vs_MLPx_combined', list(R['sets']), 'tests_vs_MLPx')):
        lst = [(ds, k, m, p) for ds in dss if ds in R['sets'] for (k, m, p) in R['sets'][ds][tk]]
        if not lst: continue
        adj = holm([R['sets'][ds]['cells'][k]['models'][m][p] for ds, k, m, p in lst])
        for (ds, k, m, p), a in zip(lst, adj): R['sets'][ds]['cells'][k]['models'][m][f'{p}_holm_{fam}'] = float(a)
        R[f'holm_{fam}'] = dict(n_tests=len(lst), n_lt_0p05=int((adj < 0.05).sum()))
        log(f'Holm {fam}: {R[f"holm_{fam}"]}')
    (RES / f'{TAG}_prong_sharing.json').write_text(json.dumps(jsonable(R), indent=1)); log(f'wrote {TAG}_prong_sharing.json')
    try:
        figures(R)
    except Exception:
        import traceback; traceback.print_exc()


# ================================================================================================================ figures
def figures(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # ---- figure 1: one real W jet, before / after D(+0.3) / cphi
    Z = np.load(RES / f'{TAG}_jc_inputs.npz'); Zo = np.load(RES / f'{TAG}_jc_obs.npz'); y = Z['y']; nn = Z['nn']; LAB = Z['LAB']
    dc = 0.3; ok = Zo[f'acc|{dc}|D'] & Zo[f'acc|{dc}|cphi'] & (y == 7) & (nn >= 20)
    c0 = Zo[f'info|{dc}|c0']; mm = Zo['O|id|m']; cand = np.flatnonzero(ok & (c0 > 0.2) & (c0 < 0.45) & (mm > 70) & (mm < 90))
    j = int(cand[0]) if len(cand) else int(np.flatnonzero(ok)[0])
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.2), sharex=True, sharey=True)
    P0 = Z['id'][j, :nn[j]].astype(np.float64); T = P0.sum(0); eJ, pJ = np.arcsinh(T[2] / np.hypot(T[0], T[1])), np.arctan2(T[1], T[0])
    for ax, (title, key) in zip(axs, (('unedited W jet', 'id'), (f'D: |cos θ*| + {dc}', f'{dc}|D'), ('control cφ: same angle, azimuthal', f'{dc}|cphi'))):
        P = (Z['id'] if key == 'id' else Z[f'P|{key}'])[j, :nn[j]].astype(np.float64); pt, eta, phi, _ = ptetaphim(P)
        for k, col in ((0, '#1f77b4'), (1, '#d62728')):
            s = LAB[j, :nn[j]] == k; ax.scatter(eta[s] - eJ, wrap(phi[s] - pJ), s=4 + 300 * pt[s] / pt.max(), c=col, alpha=0.65, edgecolors='none',
                                                label=f'prong {k + 1}: pT {pt[s].sum():.0f} GeV')
        nm = 'id' if key == 'id' else key
        ax.set_title(f'{title}\nz = {Zo[f"O|{nm}|z"][j]:.2f}, |cos θ*| = {Zo[f"O|{nm}|cos"][j]:.2f}, ΔR = {Zo[f"O|{nm}|dR"][j]:.2f}, '
                     f'm = {Zo[f"O|{nm}|m"][j]:.1f} GeV', fontsize=9.5)
        ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, ls=':', color='grey')); ax.set_aspect('equal'); ax.legend(fontsize=8, loc='lower left')
        ax.set_xlabel('Δη from jet axis'); ax.grid(alpha=0.2)
    axs[0].set_ylabel('Δφ from jet axis [rad]'); axs[0].set_xlim(-0.85, 0.85); axs[0].set_ylim(-0.85, 0.85)
    fig.suptitle(f'H45: decay-axis rotation in the prong-pair rest frame (JetClass W→qq jet, pT {np.hypot(T[0], T[1]):.0f} GeV); '
                 'jet four-momentum unchanged, marker area ∝ particle pT', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(RES / f'{TAG}_edit.png', dpi=130); plt.close(fig); log(f'wrote {TAG}_edit.png (jet row {j})')
    # ---- figure 2: D - cphi vs Dc (nets, z-blind MLPx and MLPz reference ranges, permuted / redrawn null band)
    panels = [('jc', 'QCD', 'WQ', 'JetClass QCD jets, score log p(W)/p(QCD)'), ('jc', 'Wqq', 'WQ', 'JetClass W jets, score log p(W)/p(QCD)'),
              ('hls', 'q', 'WQG', 'hls4ml q jets, score log p(W)/p(q or g)'), ('hls', 'W', 'WQG', 'hls4ml W jets, score log p(W)/p(q or g)')]
    panels = [p for p in panels if p[0] in R['sets']]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), squeeze=False); axs = axs[0]
    for ax, (ds, c, sc, title) in zip(axs, panels):
        S_ = R['sets'][ds]; cells = [S_['cells'].get(f'{c}|{sc}|{dc}', {}) for dc in DCS]
        for nm_, col in (('MLPz', '#2ca02c'), ('MLPx', '#7f7f7f')):
            lo_ = [cl.get(f'ref_range_{nm_}_D_minus_cphi', [np.nan, np.nan])[0] for cl in cells]; hi_ = [cl.get(f'ref_range_{nm_}_D_minus_cphi', [np.nan, np.nan])[1] for cl in cells]
            ax.fill_between(DCS, lo_, hi_, color=col, alpha=0.3, label={'MLPz': 'MLPz: MLPx + z_g', 'MLPx': 'MLPx: m, pT, τ21, D2, N, ΔR'}[nm_] + ' (5 retrains)')
        lo_ = [cl.get('null_band_D_minus_cphi', [np.nan, np.nan])[0] for cl in cells]; hi_ = [cl.get('null_band_D_minus_cphi', [np.nan, np.nan])[1] for cl in cells]
        ax.fill_between(DCS, lo_, hi_, color='k', alpha=0.08, hatch='//', label='permuted / redrawn nets')
        for i, m in enumerate(S_['nets']):
            mu = [cl.get('models', {}).get(m, {}).get('D_minus_cphi', np.nan) for cl in cells]
            ci = np.array([cl.get('models', {}).get(m, {}).get('D_minus_cphi_ci95', [np.nan, np.nan]) for cl in cells])
            xs = np.array(DCS) + (i - 1.5) * 0.008
            ax.errorbar(xs, mu, yerr=[np.array(mu) - ci[:, 0], ci[:, 1] - np.array(mu)], marker='o', ms=4, capsize=2, label=m)
        ax.axhline(0, color='k', lw=0.6); ax.set_title(title, fontsize=9.5); ax.set_xlabel('Δ|cos θ*| requested (− = more balanced prongs)')
        ax.grid(alpha=0.2)
    axs[0].set_ylabel('mean Δ log-odds: edit D − control cφ')
    for i, (ds, *_r) in enumerate(panels):
        if i == len(panels) - 1 or panels[i + 1][0] != ds: axs[i].legend(fontsize=7, loc='best')
    fig.suptitle('H45: tagger response to the prong momentum sharing at fixed jet four-momentum, next to a reference that never sees z (95 % CI over jets)', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(RES / f'{TAG}_response.png', dpi=130); plt.close(fig); log(f'wrote {TAG}_response.png')


# ================================================================================================================ self-test
def selftest():
    rng = np.random.default_rng(0); P = []
    for (e, p, pt, nk) in ((0.0, 0.0, 450.0, 10), (0.18, 0.08, 150.0, 8)):
        for _ in range(nk):
            ee, pp, q = e + rng.normal(0, 0.02), p + rng.normal(0, 0.02), pt / nk * rng.uniform(0.5, 1.5)
            P.append([q * np.cos(pp), q * np.sin(pp), q * np.sinh(ee), np.sqrt((q * np.cosh(ee)) ** 2 + 0.14 ** 2)])
    P = np.array(P); lab = np.repeat([0, 1], [10, 8]); nn = np.array([len(P)])
    pv0 = prong_vars(P[None], nn, lab[None]); minv = lambda Q: float(np.sqrt(max(m2(Q.sum(0)), 0)))
    pair = lambda Q: Q[:, 3][:, None] * Q[:, 3][None] - Q[:, :3] @ Q[:, :3].T
    for dc in DCS:
        r, inf = edit_one(P, lab, dc, np.random.default_rng(1))
        for kind in ('D', 'cphi'):
            Q = r[kind]; pv = prong_vars(Q[None], nn, lab[None])
            log(f'selftest dc={dc:+.2f} {kind}: d(jet four-momentum) {np.abs(Q.sum(0) - P.sum(0)).max():.1e}; dm {minv(Q) - minv(P):+.1e}; '
                f'max d(p_i.p_j) {np.abs(pair(Q) - pair(P)).max():.1e}; d(subjet masses) {pv["msub_hi"][0] - pv0["msub_hi"][0]:+.1e}/{pv["msub_lo"][0] - pv0["msub_lo"][0]:+.1e}; '
                f'|cos| {pv0["cos"][0]:.3f} -> {pv["cos"][0]:.3f} (target {inf["c1"]:.3f}); z {pv0["z"][0]:.3f} -> {pv["z"][0]:.3f}; dR {pv0["dR"][0]:.3f} -> {pv["dR"][0]:.3f}')
            assert np.abs(Q.sum(0) - P.sum(0)).max() < 1e-6 * P[:, 3].sum()
            if kind == 'D': assert abs(pv['cos'][0] - inf['c1']) < 1e-6
            if kind == 'cphi': assert abs(pv['cos'][0] - pv0['cos'][0]) < 1e-6
        Pc, err = cside_candidates(P, lab, label_sep(r['D'], lab), minv(P))
        b = int(np.argmin(err)); pc = prong_vars(Pc[b][None], nn, lab[None])
        log(f'selftest dc={dc:+.2f} cside: mass err {err[b]:.1e} GeV; dR -> {pc["dR"][0]:.3f} (D {prong_vars(r["D"][None], nn, lab[None])["dR"][0]:.3f}); '
            f'z {pv0["z"][0]:.4f} -> {pc["z"][0]:.4f}; subjet masses {pv0["msub_hi"][0]:.2f}/{pv0["msub_lo"][0]:.2f} -> {pc["msub_hi"][0]:.2f}/{pc["msub_lo"][0]:.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', default=None); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest: selftest()
    else:
        log(f'H45 part {a.part}; smoke={SMOKE}; per class {PER}; ref per class {NREF}; dc {DCS}')
        if a.part in ('jcedit', 'hedit'): selftest()
        dict(jcedit=part_jcedit, jcnets=part_jcnets, hedit=part_hedit, hsmall=part_hsmall, hjedi=part_hjedi, stats=part_stats,
             fig=lambda: figures(json.loads((RES / f'{TAG}_prong_sharing.json').read_text())))[a.part]()
        log('done')
