"""H71 (loop D, new D claim): do jet taggers read dimensionful quantities (absolute jet mass m, jet pT, constituent energies)
or dimensionless shapes (m / pT, tau21, tau32, D2, C2)? Detector-like energy-scale edits on four datasets.

Edits (every edit acts on constituent four-momenta p_i = (E, px, py, pz); directions (eta, phi) about the stored axis are kept
unless stated; lam = scale factor):
  U(lam)   uniform energy scale: every constituent p_i -> lam p_i. Jet m -> lam m, jet pT -> lam pT; every angle, m / pT, tau21,
           tau32, D2, C2 and every momentum fraction z_i = pT_i / sum pT are unchanged exactly.
  UB(lam)  U(lam) followed by a boost along the jet momentum that returns the jet pT (and |P|) to its original value at fixed
           invariant mass (interp.h12_boost_panel.boost with factor 1 / lam). Jet m -> lam m at the original pT: the same mass
           change as U, but m / pT -> lam m / pT and the opening angles grow by ~lam. Separates the absolute scale (U: m and pT
           move together, m / pT fixed) from a mass change at fixed pT.
  Ch(lam)  charged-only scale ("tracker scale"): p_i -> lam p_i for charged constituents only, neutral constituents fixed.
  Nh(lam)  neutral-only scale ("calorimeter scale"): p_i -> lam p_i for neutral constituents (neutral hadrons and photons) only.
           Charged = JetClass part_charge != 0; QG |PDG id| in {211, 321, 2212, 11, 13}. hls4ml and Top Landscape have no
           particle IDs: there Ch / Nh act on a random half of the constituents (each particle independently with prob. 0.5,
           rng seed + 7) and its complement, so any Ch - Nh difference there is the kinematic baseline.
  Rh / Rc  random-subset control for Ch / Nh: per jet, a random subset of constituents whose summed pT share is matched to the
           charged pT share (prefix of a random permutation with cumulative pT share closest to f_ch) and its complement; scaled
           by lam. Rh has (to within one particle) the same energy shift as Ch, Rc as Nh, but the particles are chosen at random.
  S(a)     per-particle energy smearing at fixed direction: p_i -> g_i p_i with g_i log-normal, mean 1, relative spread
           r_i = a / sqrt(E_i / GeV) + b (b = 0.03), sigma_log = sqrt(log(1 + r_i^2)); a in {0.5, 1.0} (calorimeter-like
           stochastic term). hls4ml tables store eta relative to the jet axis, so E_i there is pT_i cosh(deta_i) (~ pT_i).
  lam in LAMS = (0.9, 0.95, 0.98, 1.02, 1.05, 1.1) for U, UB, Ch, Nh, Rh, Rc.
Response: per jet of true class k, l_k = log p_k - log(1 - p_k) (one-vs-rest log-odds from the softmax / two-class output);
  Delta s_X = l_k(edited by X) - l_k(unedited). Slope g_X = (Delta s_X(1.1) - Delta s_X(0.9)) / log(1.1 / 0.9) per jet
  (log-odds per unit log lam; symmetric finite difference).
References (sklearn, trained on disjoint real jets of the same dataset): MLP(m, pT) on [log m, log pT]; MLP(m/pT) on
  [log(m / pT)]; GBDT(shape) = HistGradientBoosting on [tau21, tau32, log D2, log C2] (tau_N with WTA kT axes, beta 1, R0 0.8;
  D2 = e3 / e2^3, C2 = e3 / e2^2 with beta 1 on all constituents, z_i = pT_i / sum pT); LR(N) logistic regression on the
  constituent count N. Shapes and N are unchanged by U exactly (checked numerically on 300 jets), so GBDT(shape) and LR(N) are
  scored on U with the unedited shapes; their shapes are recomputed on UB / Ch / Nh / Rh / Rc at lam 0.9 and 1.1 and on S.
Outputs per part: $AI_RESULTS/physics/h71_<part>.npz (per-jet own-class log-odds per model x edit, m and pT per edit) and
  h71_<part>.json (accuracies, edit diagnostics). --part summary: statistics (bootstrap, sign-flip permutation, McNemar, Holm)
  -> h71_summary.json, figure h71_wjet_scale.png.
Samples: hls4ml jedi_core.sample(1000, 7101) + 3,000 per class reference rows (rng 7111); JetClass jc_sample(3000, 7102) random rows
  of each class's first test file, 1,000 per class evaluated + 2,000 reference (rng 7112); Top Landscape test 2,000 + 2,000 rows
  (rng 7103) + 5,000 + 5,000 disjoint reference (rng 7113); QG test 2,000 + 2,000 (rng 7104) + 5,000 + 5,000 (rng 7114).
Run
  ./job.sh d-h71-jedi 1 2 'python -m interp.h71_energy_scale_panel --part hls_jedi'
  AI_PY=conda ./job.sh d-h71-hobs 1 2 'python -m interp.h71_energy_scale_panel --part hls_obs'
  ./part_job.sh d-h71-gpu 'python -m interp.h71_energy_scale_panel --part jetclass && ... --part top && ... --part qg'
  AI_PY=conda ./job.sh d-h71-sum 1 2 'python -m interp.h71_energy_scale_panel --part summary'
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
LAMS = (0.9, 0.95, 0.98, 1.02, 1.05, 1.1)
SHAPE_LAMS = (0.9, 1.1)
SMEAR_A = (0.5, 1.0)
SMEAR_B = 0.03
NB = 1000
NPERM = 2000
T0 = time.time()
SEEDS = dict(hls=7101, jetclass=7102, top=7103, qg=7104)
CH_PID = (211, 321, 2212, 11, 13)


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ============================================================================================================== edits
def scale_mask(T, lam, mask):
    U = H.copyT(T); U['pt'] = np.where(mask, T['pt'] * lam, T['pt']); U['m'] = np.where(mask, T['m'] * lam, T['m']); return U


def matched_subset(T, ch, rng):
    """per jet: prefix of a random permutation of the real particles whose cumulative pT share is closest to the charged share"""
    real = T['pt'] > 0; pt = T['pt']; S = pt.sum(1); f = (pt * ch).sum(1) / np.maximum(S, 1e-12)
    key = np.where(real, rng.random(pt.shape), np.inf); o = np.argsort(key, 1)
    cs = np.cumsum(np.take_along_axis(pt, o, 1), 1) / np.maximum(S, 1e-12)[:, None]
    cs0 = np.concatenate([np.zeros((len(pt), 1)), cs], 1)                     # share after taking 0, 1, ... particles
    nr = real.sum(1); bad = np.arange(cs0.shape[1])[None, :] > nr[:, None]
    k = np.argmin(np.where(bad, np.inf, np.abs(cs0 - f[:, None])), 1)
    take_sorted = np.arange(pt.shape[1])[None, :] < k[:, None]
    M = np.zeros_like(real); np.put_along_axis(M, o, take_sorted, 1)
    return M & real, f


def smear(T, a, rng):
    real = T['pt'] > 0; E = np.sqrt((T['pt'] * np.cosh(T['eta'])) ** 2 + T['m'] ** 2)
    r = a / np.sqrt(np.maximum(E, 1e-3)) + SMEAR_B; sl = np.sqrt(np.log1p(r ** 2))
    g = np.exp(sl * rng.standard_normal(E.shape) - 0.5 * sl ** 2)
    U = H.copyT(T); U['pt'] = np.where(real, T['pt'] * g, 0); U['m'] = np.where(real, T['m'] * g, 0); return U


def make_edits(T, ch, seed):
    """ordered {name: table}; ch: (J, L) bool charged (or random-half) mask; deterministic in seed"""
    real = T['pt'] > 0; ch = ch & real; nh = real & ~ch
    rh, fch = matched_subset(T, ch, np.random.default_rng(seed + 1)); rc = real & ~rh
    E = {'id': T}
    for lam in LAMS:
        E[f'U{lam}'] = scale_mask(T, lam, real)
        E[f'UB{lam}'] = H.boost(scale_mask(T, lam, real), 1.0 / lam)[0]
        E[f'Ch{lam}'] = scale_mask(T, lam, ch); E[f'Nh{lam}'] = scale_mask(T, lam, nh)
        E[f'Rh{lam}'] = scale_mask(T, lam, rh); E[f'Rc{lam}'] = scale_mask(T, lam, rc)
    for a in SMEAR_A: E[f'S{a}'] = smear(T, a, np.random.default_rng(seed + 2 + int(10 * a)))
    S = np.maximum(T['pt'].sum(1), 1e-12)
    diag = dict(f_ch=dict(mean=float(fch.mean()), q=np.percentile(fch, [5, 50, 95]).tolist()),
                f_rh=dict(mean=float(((T['pt'] * rh).sum(1) / S).mean())),
                share_diff_rh_minus_ch=dict(median_abs=float(np.median(np.abs((T['pt'] * rh).sum(1) / S - fch))),
                                            p95_abs=float(np.percentile(np.abs((T['pt'] * rh).sum(1) / S - fch), 95))),
                n_ch_mean=float(ch.sum(1).mean()), n_real_mean=float(real.sum(1).mean()))
    return E, diag


def random_half(T, seed):
    return (np.random.default_rng(seed + 7).random(T['pt'].shape) < 0.5) & (T['pt'] > 0)


# ============================================================================================================== observables
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # local import check only
    njit = lambda *a, **k: (lambda f: f); prange = range; HAVE_NUMBA = False


@njit(parallel=True, cache=False)
def ecf_batch(F, nn):
    """normalised e2, e3 (beta 1) on all particles, rapidity-azimuth distances, z_i = pT_i / sum pT"""
    J = F.shape[0]; out = np.zeros((J, 2))
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
        out[j, 0] = e2; out[j, 1] = e3
    return out


def basic_obs(T):
    O = H.jet_obs(T); return dict(m=O['m'], jpt=O['jpt'], N=O['N'])


def shape_obs(T):
    from interp.h2_onesignal_setup import top_obs
    S = H.sortT(T); real = S['pt'] > 0; nn = real.sum(1).astype(np.int64); w = max(int(nn.max()), 3)
    P = H.four(S); F = np.zeros(S['pt'].shape[:1] + (w, 6))
    for c in range(4): F[..., c] = P[c][:, :w]
    F = np.ascontiguousarray(F); o = top_obs(F, nn, 0.1, 0.8); e = ecf_batch(F, nn)
    e2, e3 = np.maximum(e[:, 0], 1e-12), np.maximum(e[:, 1], 1e-30)
    return dict(tau21=o[:, 3] / np.maximum(o[:, 2], 1e-12), tau32=o[:, 4] / np.maximum(o[:, 3], 1e-12), D2=e3 / e2 ** 3, C2=e3 / e2 ** 2)


REF_X = {'MLP(m,pT)': lambda O: np.stack([np.log(np.maximum(O['m'], 1e-3)), np.log(O['jpt'])], 1),
         'MLP(m/pT)': lambda O: np.log(np.maximum(O['m'], 1e-3) / O['jpt'])[:, None],
         'GBDT(shape)': lambda O: np.stack([O['tau21'], O['tau32'], np.log(O['D2']), np.log(O['C2'])], 1),
         'LR(N)': lambda O: O['N'][:, None]}


def train_refs(Op, yp):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    refs = {}
    for k, fx in REF_X.items():
        t = time.time()
        if k.startswith('MLP'): clf = make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=400, early_stopping=True, random_state=0))
        elif k.startswith('GBDT'): clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=0)
        else: clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        X = fx(Op); clf.fit(X, yp); refs[k] = clf
        log(f'reference {k}: trained on {len(yp)} jets, train acc {clf.score(X, yp):.4f} ({time.time() - t:.0f} s)')
    return refs


def ref_logp(clf, X):
    return np.log(np.clip(clf.predict_proba(X), 1e-12, 1.0))


# ============================================================================================================== nulls / log-odds
def null_torch(module, seed=7):
    """copy of a torch module with the entries of every parameter tensor permuted"""
    import torch
    m2 = copy.deepcopy(module); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m2.parameters():
            v = p.data.detach().cpu().float(); f = v.reshape(-1); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].reshape(v.shape).to(p.dtype).to(p.device))
    return m2.eval()


def logodds_all(Lg):
    Lg = np.clip(np.nan_to_num(np.asarray(Lg, np.float64), neginf=-60.0, posinf=60.0), -1e6, 1e6)
    ls = H.logsoft(Lg); C = ls.shape[1]; out = np.empty_like(ls)
    for k in range(C):
        o = np.delete(ls, k, 1); mx = o.max(1, keepdims=True); out[:, k] = ls[:, k] - (mx[:, 0] + np.log(np.exp(o - mx).sum(1)))
    return out


# ============================================================================================================== panel
def run_part(part, T0, y, Tp, yp, classes, models, prep, ch, has_ids, seed, with_refs, info):
    """models: {name: dict(f=callable(X) -> logits or log-probs (J, C), role, known)}; ch: charged (or random-half) mask"""
    C = len(classes); J = len(y); r = np.arange(J)
    E, ediag = make_edits(T0, ch, seed)
    res = dict(part=part, N=J, labels=np.bincount(y, minlength=C).tolist(), classes=classes, lams=list(LAMS), smear_a=list(SMEAR_A),
               smear_b=SMEAR_B, has_particle_ids=has_ids, edit_diag=ediag, models={}, edits={}, **info)
    NPZ = dict(y=y.astype(np.int8))
    O0 = basic_obs(T0)
    if with_refs:
        t = time.time(); Op = basic_obs(Tp); Op.update(shape_obs(Tp)); S0 = shape_obs(T0); O0.update(S0)
        log(f'{part}: shapes of {len(yp)} reference + {J} evaluated jets ({time.time() - t:.0f} s); numba {HAVE_NUMBA}')
        sub = {k: (v[:300] if isinstance(v, np.ndarray) else v) for k, v in E['U1.1'].items()}
        sub['ext'] = {k: v[:300] for k, v in E['U1.1']['ext'].items()}
        chk = shape_obs(sub); res['U_shape_invariance_maxrel'] = {k: float(np.max(np.abs(chk[k] / np.maximum(np.abs(S0[k][:300]), 1e-12) - 1))) for k in chk}
        log(f'{part}: U1.1 shape invariance check (max relative change, 300 jets): {res["U_shape_invariance_maxrel"]}')
        refs = train_refs(Op, yp)
        for k, clf in refs.items():
            models[k] = dict(f=(lambda O, clf=clf, k=k: ref_logp(clf, REF_X[k](O))), role='ref', src='obs', known=None)
        res['ref_train_n'] = int(len(yp))
    L0 = {}
    for mn, M in models.items():
        t = time.time(); lg = M['f'](O0 if M.get('src') == 'obs' else prep(H.sortT(T0))); c = lg.argmax(1); lo = logodds_all(lg); L0[mn] = lo
        rr = dict(role=M['role'], n=J, acc=float((c == y).mean()), known_acc=M.get('known'),
                  per_class={classes[k]: float((c[y == k] == k).mean()) for k in range(C)})
        if C == 2:
            from sklearn.metrics import roc_auc_score
            rr['auc'] = float(roc_auc_score(y == 1, lo[:, 1])) if np.ptp(lo[:, 1]) > 0 else 0.5
        res['models'][mn] = rr; NPZ[f'lo__{mn}__id'] = lo[r, y].astype(np.float32)
        log(f'{part} {mn} [{M["role"]}]: N {J} accuracy {rr["acc"]:.4f} (known {M.get("known")}) auc {rr.get("auc")} ({time.time() - t:.0f} s)')
    NPZ['m__id'] = O0['m'].astype(np.float32); NPZ['jpt__id'] = O0['jpt'].astype(np.float32)
    for name, U in E.items():
        if name == 'id': continue
        t = time.time(); O1 = basic_obs(U); lamv = float(name.lstrip('UBChNRcS'))
        if with_refs:
            if name.startswith('U') and not name.startswith('UB'): O1.update({k: O0[k] for k in ('tau21', 'tau32', 'D2', 'C2')})
            elif name.startswith('S') or lamv in SHAPE_LAMS: O1.update(shape_obs(U))
        X1 = prep(H.sortT(U))
        for mn, M in models.items():
            if M.get('src') == 'obs':
                if M is models.get('GBDT(shape)') and 'tau21' not in O1: continue
                inp = O1
            else: inp = X1
            NPZ[f'lo__{mn}__{name}'] = logodds_all(M['f'](inp))[r, y].astype(np.float32)
        NPZ[f'm__{name}'] = O1['m'].astype(np.float32); NPZ[f'jpt__{name}'] = O1['jpt'].astype(np.float32)
        e = dict(dlogm_median=float(np.median(np.log(np.maximum(O1['m'], 1e-6) / np.maximum(O0['m'], 1e-6)))),
                 dlogpt_median=float(np.median(np.log(O1['jpt'] / O0['jpt']))))
        if with_refs and 'tau21' in O1 and not (name.startswith('U') and not name.startswith('UB')):
            e.update({f'dlog{k}_median': float(np.median(np.log(np.maximum(O1[k], 1e-12) / np.maximum(O0[k], 1e-12)))) for k in ('tau21', 'tau32', 'D2', 'C2')})
        res['edits'][name] = e
        log(f'{part} {name}: dlog m {e["dlogm_median"]:+.4f} dlog pT {e["dlogpt_median"]:+.4f} | '
            + ' '.join(f'{mn} {np.mean(NPZ[f"lo__{mn}__{name}"] - NPZ[f"lo__{mn}__id"]):+.3f}' for mn in models if f'lo__{mn}__{name}' in NPZ)
            + f' ({time.time() - t:.0f} s)')
    res['roles'] = {mn: M['role'] for mn, M in models.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'h71_{part}.json').write_text(json.dumps(res, indent=1, default=float)); np.savez_compressed(OUT / f'h71_{part}.npz', **NPZ)
    log(f'wrote {OUT / f"h71_{part}.json"} and .npz')


# ============================================================================================================== datasets
def hls_data():
    from interp.h64_axis_boost_panel import hls_tables
    seed = SEEDS['hls']; T0, y, Tp, yp, idx = hls_tables(seed, per=1000, per_ref=3000)
    return seed, T0, y, Tp, yp


def part_hls_jedi():
    from interp.jedi_core import Jedi
    from jetdistill import network as N
    seed, T0, y, Tp, yp = hls_data()
    nets = dict(jedi_n64=Jedi(64, '3-feature'))
    orig = N.redraw_untrained; N.redraw_untrained = H.permute_weights
    try: nets['jedi_n64_perm'] = Jedi(64, '3-feature', untrained=True, seed=7)
    finally: N.redraw_untrained = orig
    models = {k: dict(f=(lambda X, n=n: n.run(X)['logits']), role='trained' if k == 'jedi_n64' else 'null',
                      known='0.8066 (H64, 10,000 jets) / 0.8100 (H12)' if k == 'jedi_n64' else None) for k, n in nets.items()}
    run_part('hls_jedi', T0, y, Tp, yp, H.HLS_CL, models, H.hls_prep, random_half(T0, seed), False, seed, False,
             dict(sample=f'interp.jedi_core.sample(per_class=1000, seed={seed})'))


def part_hls_obs():
    seed, T0, y, Tp, yp = hls_data()
    run_part('hls_obs', T0, y, Tp, yp, H.HLS_CL, {}, H.hls_prep, random_half(T0, seed), False, seed, True,
             dict(sample=f'interp.jedi_core.sample(per_class=1000, seed={seed}); references on 3,000 other rows per class (rng {seed + 10})'))


def part_jetclass():
    from interp.h1_irc_panel import jc_sample, jc_table, pn_logits
    from interp.part_core import ParTNetwork, CL, inputs, build, sub
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); seed = SEEDS['jetclass']
    Jall = jc_sample(3000, seed); yall = Jall['label'].astype(int); rr = np.random.default_rng(seed + 10); ev, rf = [], []
    for c in range(len(CL)):
        ii = rr.permutation(np.flatnonzero(yall == c)); ev.append(ii[:1000]); rf.append(ii[1000:])
    ev, rf = np.sort(np.concatenate(ev)), np.sort(np.concatenate(rf))
    J0 = sub(Jall, ev); y = yall[ev]
    tab = lambda Jx: (lambda Tj: dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext']))(jc_table(Jx))
    T0 = tab(J0); Jr = sub(Jall, rf); yp = yall[rf]; Tp = tab(Jr); Tp['ext'] = {}; del Jall, Jr
    log(f'JetClass: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        Jx = dict(J0); Jx['jet_eta'] = T['ax_eta'].astype(np.float32); Jx['jet_phi'] = T['ax_phi'].astype(np.float32)
        sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = H.rel(T)
        return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
    known = dict(ParT_full='0.8583 (H64) / 0.859 (D4)', ParT_kin='0.7416 (H64) / 0.745 (D4)', PN_full='0.8384 (H64)', PN_kin='0.7040 (H64)')
    pt_f = lambda net, n: (lambda J: fwd(net, *inputs(J, n))[1].astype(np.float64))
    models = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); models[f'ParT_{n}'] = dict(f=pt_f(net, n), role='trained', known=known[f'ParT_{n}'])
        nn_ = copy.copy(net); nn_.model = null_torch(net.model, 7); models[f'ParT_{n}_perm'] = dict(f=pt_f(nn_, n), role='null')
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        mk = lambda p, n=n: (lambda J: pn_logits(p, J, n))
        models[f'PN_{n}'] = dict(f=mk(pn), role='trained', known=known[f'PN_{n}'])
        models[f'PN_{n}_perm'] = dict(f=mk(null_torch(pn, 7)), role='null')
    ch = (T0['ext']['charge'] != 0) & (T0['pt'] > 0)
    run_part('jetclass', T0, y, Tp, yp, list(CL), models, prep, ch, True, seed, True,
             dict(sample=f'h1_irc_panel.jc_sample(3000, seed {seed}) random rows per class; 1,000 per class evaluated, 2,000 reference (rng {seed + 10})'))


def split_rows(ylab, per, per_ref, seed):
    ev = H._rows(ylab, per, seed); used = np.zeros(len(ylab), bool); used[ev] = True; r = np.random.default_rng(seed + 10)
    rf = np.sort(np.concatenate([r.choice(np.flatnonzero((ylab == c) & ~used), per_ref, replace=False) for c in (0, 1)]))
    return ev, rf


def part_top():
    import torch
    from interp import data_top as dt
    from interp.h4_top_wmass import WeaverNet, LNet, Flow, kin3_of
    from interp.h64_axis_boost_panel import null_flow
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; seed = SEEDS['top']
    d = dt.DST / 'test'; ylab = np.load(d / 'label.npy'); ev, rf = split_rows(ylab, 2000, 5000, seed)
    Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy')
    tab = lambda ii: H.table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1]))
    T0, Tp = tab(ev), tab(rf); y, yp = ylab[ev].astype(int), ylab[rf].astype(int); log(f'Top: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
        P4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
        return dict(P4=P4.astype(np.float32), nn=real.sum(1))
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    known = dict(ParT_kin='0.937 (H64) / 0.940 (H2)', PN_kin='0.940 (H64)', LorentzNet='0.9433 (H64)', PFN='0.9293 (H64)', EFN='0.9227 (H64)')
    models = {}; wf = lambda net: (lambda X: two(net.score(X['P4'], X['nn'])))
    for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
        net = WeaverNet(kind, V / 'models/top' / dname / 'net_best_epoch_state.pt', dev)
        models[f'{kind}_kin'] = dict(f=wf(net), role='trained', known=known[f'{kind}_kin'])
        n2 = copy.copy(net); n2.model = null_torch(net.model, 7); models[f'{kind}_kin_perm'] = dict(f=wf(n2), role='null')
    ln = LNet(dev, mass64=True); order = np.random.default_rng(seed + 5).permutation(len(ev))
    lf = lambda net: (lambda X: two(net.score(X['P4'], X['nn'], order)))
    models['LorentzNet'] = dict(f=lf(ln), role='trained', known=known['LorentzNet'])
    for a in ('pfn', 'efn'):
        fl = Flow(a); ff = lambda f_: (lambda X: two(f_.score(kin3_of(X['P4'], X['nn']))))
        models[a.upper()] = dict(f=ff(fl), role='trained', known=known[a.upper()])
        if a == 'pfn': models['PFN_perm'] = dict(f=ff(null_flow(fl, 'perm', 7)), role='null')
    run_part('top', T0, y, Tp, yp, ['QCD', 'top'], models, prep, random_half(T0, seed), False, seed, True,
             dict(sample=f'Top Landscape test, 2,000 top + 2,000 QCD random rows (rng {seed}); reference 5,000 + 5,000 disjoint rows (rng {seed + 10})'))


def part_qg():
    import torch
    from interp import data_qg as dq
    from interp import h5_qg_counting as Q
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; seed = SEEDS['qg']
    d = dq.DST / 'test'; ylab = np.load(d / 'label.npy'); ev, rf = split_rows(ylab, 2000, 5000, seed)
    Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy'); pidm = np.load(d / 'pid.npy', mmap_mode='r')

    def tab(ii):
        T = H.table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1])); T['m'] = np.zeros_like(T['m'])
        T['ext'] = dict(pid=np.asarray(pidm[ii]).astype(np.int64) * (T['pt'] > 0)); return T
    T0, Tp = tab(ev), tab(rf); Tp['ext'] = {}; y, yp = ylab[ev].astype(int), ylab[rf].astype(int); log(f'QG: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        real = T['pt'] > 0
        return dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], pid=T['ext']['pid'].astype(np.int64) * real, n=real.sum(1))
    M = Q.Models(dev)
    for name, arch, pf in (('part_kin', 'part', False), ('pnet_kin', 'pn', False), ('part_kinpid', 'part', True), ('pnet_kinpid', 'pn', True)):
        try: M.add_weaver(name, arch, pf)
        except Exception as ex: log(f'QG {name} not loaded: {ex!r}')
    for src in ('part_kinpid', 'part_kin'):
        if src in M.m: M.add_permuted(f'{src}_perm', src)
    try: M.add_lnet()
    except Exception as ex: log(f'QG LorentzNet not loaded: {ex!r}')
    for kind in ('pfn', 'efn'):
        try:
            net = Q.make_ef(kind); net.load_state_dict(torch.load(V / 'models' / 'qg' / f'h5_{kind}_torch.pt', map_location='cpu'))
            M.add_torch(f'{kind}_t', net, json.loads((V / 'models' / 'qg' / f'h5_{kind}_torch.json').read_text()))
        except Exception as ex: log(f'QG {kind}_t not loaded: {ex!r}')
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    known = dict(part_kin='0.841 (H64) / 0.827 (H2)', pnet_kin='0.8355 (H64)', lorentznet='0.8542 (H64)')
    models = {k: dict(f=(lambda X, k=k: two(-M.logit(k, X))), role='null' if k.endswith('_perm') else 'trained', known=known.get(k))
              for k in M.m}
    ch = np.isin(np.abs(T0['ext']['pid']), CH_PID) & (T0['pt'] > 0)
    run_part('qg', T0, y, Tp, yp, ['gluon', 'quark'], models, prep, ch, True, seed, True,
             dict(sample=f'QG test, 2,000 quark + 2,000 gluon random rows (rng {seed}); reference 5,000 + 5,000 disjoint rows (rng {seed + 10})',
                  ckpt=M.info))


# ============================================================================================================== statistics
def boot_ci(x, rng, B=NB):
    n = len(x); I = rng.integers(0, n, (B, n)); b = x[I].mean(1); return np.percentile(b, [2.5, 97.5]).tolist(), b


def signflip_p(d, rng, P=NPERM):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d)
    if n < 2 or np.all(d == 0): return 1.0
    obs = abs(d.mean()); s = rng.choice([-1.0, 1.0], (P, n)); null = np.abs((s * d).mean(1)); k = int((null >= obs - 1e-15).sum())
    if k > 0: return float((1 + k) / (P + 1))
    # no exceedance in P random sign flips: the sign-flip null of the mean is normal with variance sum d^2 / n^2 (n >= 500 here);
    # its tail probability replaces the resolution floor 1 / (P + 1), which would otherwise block Holm over > 100 tests
    return float(math.erfc(obs / (math.sqrt((d ** 2).sum()) / n) / math.sqrt(2)))


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans (call changed under arm a vs under arm b)"""
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum()); n = n01 + n10
    if n == 0: return dict(n01=0, n10=0, p=1.0)
    from math import comb
    k = min(n01, n10); p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n < 1000 else \
        float(math.erfc(abs(n01 - n10) / math.sqrt(n) / math.sqrt(2)))
    return dict(n01=n01, n10=n10, p=float(p))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o): run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj.tolist()


LOG_STEP = math.log(1.1 / 0.9)


ARMS = ('U', 'UB', 'Ch', 'Nh', 'Rh', 'Rc')


def mstat(x, rng):
    x = np.asarray(x, np.float64); return dict(mean=float(x.mean()), ci=boot_ci(x, rng)[0], p_signflip_eq0=signflip_p(x, rng))


def model_stats(D, mn, y, classes, rng):
    """D: npz dict. Per class k (jets of true class k), Delta s_X(lam) = own-class log-odds change.
    curves: mean Delta s per arm and lam; pm: Delta s at lam 0.9 / 1.1 with CI and sign-flip p; odd part g = (Ds(1.1) - Ds(0.9)) /
    log(1.1/0.9) (slope d Delta s / d log lam); even part h = (Ds(1.1) + Ds(0.9)) / 2 (a drop on both sides = a window around the
    nominal scale). U_minus_UB: paired Delta s_U - Delta s_UB. asym: Ch/Nh vs Rh/Rc at each lam. Model summary rho = sum_k size_U,k /
    sum_k size_UB,k with size = (|mean Ds(0.9)| + |mean Ds(1.1)|) / 2."""
    lo = lambda e: D.get(f'lo__{mn}__{e}')
    base = lo('id'); out = dict(classes={}); size = {}; per = {}
    for k, cn in enumerate(classes):
        sel = y == k; n = int(sel.sum()); st = dict(n=n, curves={}, pm={}, odd_g={}, even_h={}, wp={})
        d = {e: (lo(e) - base)[sel].astype(np.float64) for e in [f'{a}{l}' for a in ARMS for l in LAMS] + [f'S{a}' for a in SMEAR_A]
             if lo(e) is not None}
        for a in ARMS:
            st['curves'][a] = {str(l): float(d[f'{a}{l}'].mean()) for l in LAMS if f'{a}{l}' in d}
            if f'{a}1.1' in d and f'{a}0.9' in d:
                st['pm'][a] = {l: mstat(d[f'{a}{l}'], rng) for l in ('0.9', '1.1')}
                st['odd_g'][a] = mstat((d[f'{a}1.1'] - d[f'{a}0.9']) / LOG_STEP, rng)
                st['even_h'][a] = mstat((d[f'{a}1.1'] + d[f'{a}0.9']) / 2, rng)
                size[(k, a)] = (abs(d[f'{a}0.9'].mean()) + abs(d[f'{a}1.1'].mean())) / 2; per[(k, a)] = (d[f'{a}0.9'], d[f'{a}1.1'])
        if 'U' in st['pm'] and 'UB' in st['pm']:
            st['U_minus_UB'] = {l: mstat(d[f'U{l}'] - d[f'UB{l}'], rng) for l in ('0.9', '1.1')}
        if all(a in st['pm'] for a in ('Ch', 'Nh', 'Rh', 'Rc')):
            st['asym'] = {}
            for l in ('0.9', '1.1'):
                ch, nh, rh, rc = (d[f'{a}{l}'] for a in ('Ch', 'Nh', 'Rh', 'Rc'))
                st['asym'][l] = dict(ch_minus_nh=mstat(ch - nh, rng), rh_minus_rc=mstat(rh - rc, rng), excess=mstat((ch - nh) - (rh - rc), rng),
                                     ch_minus_rh=mstat(ch - rh, rng), nh_minus_rc=mstat(nh - rc, rng),
                                     additivity_ch_plus_nh_minus_u=mstat(ch + nh - d[f'U{l}'], rng) if f'U{l}' in d else None)
        for a in SMEAR_A:
            if f'S{a}' in d: st[f'S{a}'] = dict(mstat(d[f'S{a}'], rng), mean_abs=float(np.abs(d[f'S{a}']).mean()))
        # working point: 50 % efficiency for true class k on unedited jets (own-class log-odds threshold, frozen)
        b0 = base[sel].astype(np.float64); thr = float(np.median(b0)); c0 = b0 > thr; st['wp']['threshold'] = thr
        for e in [f'{a}{l}' for a in ARMS for l in (0.9, 1.1)] + [f'S{a}' for a in SMEAR_A]:
            if lo(e) is None: continue
            c1 = lo(e)[sel] > thr; st['wp'][e] = dict(eff=float(c1.mean()), changed=float((c1 != c0).mean()))
        for e1, e2 in (('U1.1', 'UB1.1'), ('U0.9', 'UB0.9'), ('Ch1.1', 'Rh1.1'), ('Ch0.9', 'Rh0.9'), ('Nh1.1', 'Rc1.1'), ('Nh0.9', 'Rc0.9')):
            if lo(e1) is None or lo(e2) is None: continue
            a1 = (lo(e1)[sel] > thr) != c0; a2 = (lo(e2)[sel] > thr) != c0; st['wp'][f'mcnemar_{e1}_vs_{e2}'] = mcnemar(a1, a2)
        out['classes'][cn] = st
    C = len(classes)
    if all((k, 'U') in size and (k, 'UB') in size for k in range(C)):
        num = sum(size[(k, 'U')] for k in range(C)); den = sum(size[(k, 'UB')] for k in range(C)); bn, bd = np.zeros(NB), np.zeros(NB)
        for k in range(C):
            n = len(per[(k, 'U')][0]); I = rng.integers(0, n, (NB, n))
            bn += (np.abs(per[(k, 'U')][0][I].mean(1)) + np.abs(per[(k, 'U')][1][I].mean(1))) / 2
            bd += (np.abs(per[(k, 'UB')][0][I].mean(1)) + np.abs(per[(k, 'UB')][1][I].mean(1))) / 2
        out['rho_U_over_UB'] = dict(value=float(num / den) if den > 0 else None, ci=np.percentile(bn / np.maximum(bd, 1e-12), [2.5, 97.5]).tolist(),
                                    size_U=float(num), size_U_ci=np.percentile(bn, [2.5, 97.5]).tolist(), size_UB=float(den),
                                    size_UB_ci=np.percentile(bd, [2.5, 97.5]).tolist())
    return out


def window_stats(D, y, classes):
    """mass-window tagger per class: window = central 50 % of the class's unedited jet mass -> efficiency after U / UB / Ch / Nh"""
    out = {}
    for k, cn in enumerate(classes):
        sel = y == k; m0 = D['m__id'][sel]; lo, hi = np.percentile(m0, [25, 75]); r = dict(window=[float(lo), float(hi)])
        for e in [f'{a}{l}' for a in ('U', 'UB', 'Ch', 'Nh', 'Rh', 'Rc') for l in LAMS] + [f'S{a}' for a in SMEAR_A]:
            if f'm__{e}' in D:
                m1 = D[f'm__{e}'][sel]; r[e] = float(((m1 > lo) & (m1 < hi)).mean())
        out[cn] = r
    return out


PARTS = ('hls_jedi', 'hls_obs', 'jetclass', 'top', 'qg')


def part_summary():
    rng = np.random.default_rng(7199); S = dict(parts={}, holm={})
    tests = {'H0_U_eq0': [], 'H0_U_eq_UB': [], 'H0_asym_excess': [], 'H0_Ch_eq_Rh': [], 'H0_Nh_eq_Rc': [], 'H0_S1.0_eq0': [],
             'McNemar_U_vs_UB': [], 'McNemar_Ch_vs_Rh': [], 'McNemar_Nh_vs_Rc': []}
    for p in PARTS:
        fj, fn = OUT / f'h71_{p}.json', OUT / f'h71_{p}.npz'
        if not fj.exists(): log(f'summary: {p} missing'); continue
        R = json.loads(fj.read_text()); D = dict(np.load(fn)); y = D['y'].astype(int); classes = R['classes']
        P = dict(N=R['N'], classes=classes, has_particle_ids=R['has_particle_ids'], edit_diag=R['edit_diag'], edits=R['edits'],
                 U_shape_invariance=R.get('U_shape_invariance_maxrel'), accuracy={}, models={})
        if p != 'hls_jedi': P['mass_window'] = window_stats(D, y, classes)
        for mn, m in R['models'].items():
            P['accuracy'][mn] = dict(role=m['role'], acc=m['acc'], auc=m.get('auc'), known=m.get('known_acc'), per_class=m['per_class'])
            st = model_stats(D, mn, y, classes, rng); P['models'][mn] = st
            if m['role'] != 'trained': continue
            for cn, c in st['classes'].items():
                key = f'{p}|{mn}|{cn}'
                for l in ('0.9', '1.1'):
                    if 'U' in c['pm']: tests['H0_U_eq0'].append((f'{key}|{l}', c['pm']['U'][l]['p_signflip_eq0']))
                    if 'U_minus_UB' in c: tests['H0_U_eq_UB'].append((f'{key}|{l}', c['U_minus_UB'][l]['p_signflip_eq0']))
                    if 'asym' in c:
                        tests['H0_asym_excess'].append((f'{key}|{l}', c['asym'][l]['excess']['p_signflip_eq0']))
                        tests['H0_Ch_eq_Rh'].append((f'{key}|{l}', c['asym'][l]['ch_minus_rh']['p_signflip_eq0']))
                        tests['H0_Nh_eq_Rc'].append((f'{key}|{l}', c['asym'][l]['nh_minus_rc']['p_signflip_eq0']))
                if 'S1.0' in c: tests['H0_S1.0_eq0'].append((key, c['S1.0']['p_signflip_eq0']))
                for fam, e1, e2 in (('McNemar_U_vs_UB', 'U1.1', 'UB1.1'), ('McNemar_Ch_vs_Rh', 'Ch1.1', 'Rh1.1'), ('McNemar_Nh_vs_Rc', 'Nh1.1', 'Rc1.1')):
                    mc = c['wp'].get(f'mcnemar_{e1}_vs_{e2}')
                    if mc: tests[fam].append((key, mc['p']))
            log(f'summary {p} {mn}: rho {st.get("rho_U_over_UB", {}).get("value")} ' +
                ' '.join(f'{cn}: U {c["curves"].get("U", {}).get("0.9", float("nan")):+.3f}/{c["curves"].get("U", {}).get("1.1", float("nan")):+.3f} '
                         f'UB {c["curves"].get("UB", {}).get("0.9", float("nan")):+.3f}/{c["curves"].get("UB", {}).get("1.1", float("nan")):+.3f} '
                         f'asym1.1 {c.get("asym", {}).get("1.1", {}).get("excess", {}).get("mean", float("nan")):+.3f}' for cn, c in st['classes'].items()))
        S['parts'][p] = P
    for fam, lst in tests.items():
        if not lst: continue
        hp = holm([x[1] for x in lst])
        S['holm'][fam] = dict(n_tests=len(lst), n_below_0p05=int(sum(h < 0.05 for h in hp)), p=dict(zip([x[0] for x in lst], hp)))
        log(f'Holm {fam}: {S["holm"][fam]["n_below_0p05"]} / {len(lst)} below 0.05')
    (OUT / 'h71_summary.json').write_text(json.dumps(S, indent=1, default=float)); log('wrote h71_summary.json')
    try: fig_wjet()
    except Exception as ex: log(f'figure failed: {ex!r}')


def fig_wjet():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    D = dict(np.load(OUT / 'h71_jetclass.npz')); R = json.loads((OUT / 'h71_jetclass.json').read_text()); y = D['y'].astype(int)
    kW = R['classes'].index('Wqq'); m0 = D['m__id']; s0 = D['lo__ParT_full__id']
    cand = np.flatnonzero((y == kW) & (np.abs(m0 - 80) < 5)); j = cand[np.argmin(np.abs(s0[cand] - np.median(s0[cand])))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    cols = dict(ParT_full='#2e5eaa', ParT_kin='#7aa6d6', PN_full='#d1495b', PN_kin='#eaa0a8')
    lams = [0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1]
    for mn, c in cols.items():
        for arm, ls, lab in (('U', '-', 'U: all momenta x λ (m, pT x λ)'), ('UB', '--', 'UB: m x λ at fixed pT')):
            ms = [m0[j] if l == 1.0 else D[f'm__{arm}{l}'][j] for l in lams]; ss = [s0[j] if l == 1.0 else D[f'lo__{mn}__{arm}{l}'][j] for l in lams]
            if mn == 'ParT_full' or arm == 'U':
                ax[0].plot(ms, np.array(ss) - (D[f'lo__{mn}__id'][j]), ls, color=c, marker='o', ms=3, label=f'{mn}, {arm}')
    ax[0].set_xlabel('jet mass of the edited jet [GeV]'); ax[0].set_ylabel('change of W log-odds  Δs')
    ax[0].set_title(f'one JetClass W jet (m = {m0[j]:.1f} GeV, pT = {D["jpt__id"][j]:.0f} GeV), λ = 0.9 … 1.1', fontsize=10)
    ax[0].axhline(0, color='0.7', lw=0.6); ax[0].legend(fontsize=7.5, frameon=False)
    sel = y == kW
    for mn, c in list(cols.items()) + [('MLP(m,pT)', 'k'), ('MLP(m/pT)', '0.5'), ('GBDT(shape)', '#5a9e6f')]:
        if f'lo__{mn}__U1.1' not in D: continue
        mu = [0.0 if l == 1.0 else float((D[f'lo__{mn}__U{l}'] - D[f'lo__{mn}__id'])[sel].mean()) for l in lams]
        ax[1].plot(lams, mu, '-o', color=c, ms=3, label=mn)
    ax[1].set_xlabel('uniform energy scale λ (every constituent four-momentum x λ)'); ax[1].set_ylabel('mean Δs over 1,000 W jets')
    ax[1].set_title('W jets: mean response to a uniform energy scale (m/pT, τ21, D2 fixed)', fontsize=10)
    ax[1].axhline(0, color='0.7', lw=0.6); ax[1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / 'h71_wjet_scale.png', dpi=130); log(f'wrote {OUT / "h71_wjet_scale.png"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=list(PARTS) + ['summary'])
    a = ap.parse_args(); log(f'H71 part {a.part}')
    dict(hls_jedi=part_hls_jedi, hls_obs=part_hls_obs, jetclass=part_jetclass, top=part_top, qg=part_qg, summary=part_summary)[a.part]()
    log('done')
