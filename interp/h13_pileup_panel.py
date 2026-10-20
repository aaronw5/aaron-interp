"""H13 (loop D; claims D4m, D5, D8; new sub-claim D10): soft contamination / pileup panel. How do jet taggers respond to diffuse
soft particles spread over the jet area (pileup-like), compared with the ungroomed invariant mass, the soft-drop mass, the
axis-moment mass m_A and classical taggers at their published working points, and how large is the E-scheme axis offset the
contamination induces (review weakness 10)?

Samples (seed 302, rows not used by H1/H4/H5):
  hls4ml   jets of jedi_core.sample(per_class=2000, seed=302) (10,000 jets, 64 slots; relative coordinates, stored axis (0, 0))
  JetClass 1,000 random rows per class of each class's first test file (seed 302; 10,000 jets, 128 slots; eta = jet_eta +
           sign(jet_eta) deta)
  Top      data_top.sample(5000, 302) (5,000 top + 5,000 QCD test jets, random rows; up to 200 constituents) and, for the
           background rejection, 100,000 random QCD test jets (seed 303, rows of the main sample excluded)
Models
  --part jedi     (jax env)   JEDI-linear official n64 3-feature; weight-redrawn JEDI (C4)
  --part small    (conda env) Deep Sets dsets-d64-n64 seed0; the H1 EFN (IRC safe, hls4ml; C4); Deep Sets with the entries of
                              every weight tensor permuted (seed 7; C4)
  --part jetclass (GPU)       ParT_full, ParT_kin, ParticleNet_full, ParticleNet_kin; weight-redrawn ParT_full (C4)
  --part top      (GPU)       ParT_kin, ParticleNet_kin (H2 weaver trainings, best epoch), LorentzNet (released weights), PFN,
                              EFN (H2 energyflow; EFN = C4), PFN with every weight tensor's entries permuted (seed 7; C4)
Edit (pu_rho, rho in {5, 10, 20, 40} GeV per unit eta-phi area): a Poisson number of particles with mean rho * A / <q>
  (A = pi R^2, R = 0.8) placed uniformly in the disc of radius R about the stored jet axis; per-particle pT q drawn from the
  sample's softest-quartile constituent spectrum (all real constituents of the sample with pT < their 25th percentile q25;
  <q> its mean); massless; JetClass: neutral hadron, charge 0, d0 = dz = 0. The jet is then re-formed as the model's
  preprocessing does: particles pT-sorted and cut to the slot limit (64 / 128 / 200), the E-scheme axis recomputed (hls4ml
  and JetClass: from all particles, as the stored axis is the clustered jet's; Top: from the stored constituents, as the Top
  Landscape conversion computes it), and every input (deta, dphi, deltaR, log pT/pT_jet, log E/E_jet) taken about the new axis.
Conditions (same injected particles within a rho)
  id        stored jet, stored axis (the unedited reference; all comparisons are to it)
  reax0     stored particles about their own E-scheme axis (no particle added; the axis-recomputation floor)
  wta       stored particles about the winner-take-all axis (D4 edit; calibration point of the moment fit)
  sk (c1)   SoftKiller-style cut: every particle below q25 removed (hardest always kept), axis recomputed. Because every injected
            particle has q < q25, "inject at any rho, then cut at q25" gives this same jet exactly: the original jet minus its
            own particles below the cut ("returns to the original up to the cut").
  c1m_rho   inject, then cut at the median of the injected spectrum (removes about half of the injected particles and the
            original particles below that median; an imperfect SoftKiller)
  pufix_rho inject, inputs about the stored axis (no axis offset; isolates the added particles from the axis offset)
  c2_rho    the injected pT sum placed as one massless particle on the stored axis (same added pT, mass shift ~ q m^2 / E,
            axis unchanged to O(1e-6))
  c3_rho    the original particles about the pu_rho axis (pure translation: the same axis offset, nothing added)
Quantities per jet and condition (on the particles the model sees): A = 1/2 log(M0 M2) about the input axis (M0 = sum pT,
  M2 = sum pT dR^2), m_A = sqrt(M0 M2), invariant mass m of the kept particles, ungroomed mass of all particles, axis offsets
  |new axis - stored|, |new axis - E-scheme axis of the stored particles|, |new axis - WTA axis| (dR in eta-phi).
Metrics: % calls changed (argmax; Top: score > threshold at eps_S = 0.5 frozen on unedited tops) with 95 % interval;
  mean delta log-softmax per class / delta log-odds per class with 95 % bootstrap interval; delta(Z - W logit) on W/Z jets;
  paired McNemar pu_rho vs sk (c1) and vs c2_rho, Holm over models x rho; Top: 1/eps_B at eps_S 0.3 / 0.5 (thresholds re-set
  on the edited signal, and frozen) over the 100k QCD jets, 200 bootstrap resamples.
Moment-formula test (D4m): per model, score s (hls4ml Z - W logit on W/Z jets; JetClass Zqq - Wqq logit on Wqq/Zqq jets; Top
  log-odds on all jets); (i) slope a from the pure-axis conditions (reax0, wta, c3_rho; delta log m = 0 exactly), prediction
  for pu_rho = a * delta A, ratio observed / predicted with bootstrap CI; (ii) two-variable fit delta s = a dA + b dlog m + c
  pooled over reax0, wta, pufix, c2, c3, pu (bootstrap over jets); (iii) axis part of the edit, s(pu) - s(pufix), against c3.
Classical references (--part obs, conda CPU, rows [::5] = 2,000 jets, all particles, no slot cut): m, m_SD (C/A, zcut 0.1,
  beta 0), tau21, tau32 (WTA kT axes, beta 1, R0 0.8), D2 (beta 1), m_A about the E-scheme axis; tagger calls W(m) 65 < m < 95,
  W(mSD) 65 < mSD < 95, top(mSD) 105 < mSD < 210 and tau32 < 0.65, top(m) 105 < m < 210 and tau32 < 0.65.
Run
  ./job.sh d-h13-jedi 1 2 'python -m interp.h13_pileup_panel --part jedi'
  AI_PY=conda ./job.sh d-h13-small 1 2 'python -m interp.h13_pileup_panel --part small'
  ./part_job.sh d-h13-jc 'python -m interp.h13_pileup_panel --part jetclass'
  ./part_job.sh d-h13-top 'python -m interp.h13_pileup_panel --part top'
  AI_PY=conda ./job.sh d-h13-obs 4 6 'JETDISTILL_TAGGER=part JETDISTILL_DATA=/j-jepa-vol/aaron-interp/code/cluster/data_part
      python -m interp.h13_pileup_panel --part obs --ds jc' (hls4ml: --ds hls with the default env; top: --ds top)
  ./job.sh d-h13-sum 'python -m interp.h13_pileup_panel --part summary'
  -> $AI_RESULTS/physics/h13_{jedi,small,jetclass,top,obs_hls,obs_jc,obs_top,summary}.json, h13_*_perjet.npz, h13_panel.png
  --smoke N: N jets per class (writes *_smoke files)
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
from interp.h1_irc_panel import (p3, from3, wrap, copyT, finalize, jet4, boot_binary, binom_two_sided, holm, logsoftmax,  # noqa: E402
                                 cond_stats, paired, hls_sample, EXT_KEYS)

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
T0 = time.time()
RHOS = (5, 10, 20, 40)
R = 0.8
AREA = math.pi * R * R
SEED = 302
NB = 500
BKG_KEEP = ('id', 'sk', 'pu_5', 'pu_10', 'pu_20', 'pu_40', 'c1m_40', 'c2_40')   # conditions scored on the 100k QCD jets


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ================================================================================================================ geometry
def eaxis(T):
    """E-scheme axis (pseudorapidity, azimuth of the summed 3-momentum of the real particles)"""
    real = T['pt'] > 0
    px, py, pz = p3(T['pt'], T['eta'], T['phi'])
    _, e, p = from3((px * real).sum(1), (py * real).sum(1), (pz * real).sum(1))
    return e, p


def dR(e1, p1, e2, p2):
    return np.hypot(e1 - e2, wrap(p1 - p2))


def wta_axis_one(pt, y, p):
    """verbatim from interp/p1v_verify_part.py: C/A clustering with winner-take-all recombination; (y, p) of the final branch"""
    n = len(pt)
    if n == 1: return float(y[0]), float(p[0])
    pt = pt.astype(np.float64).copy(); y = y.astype(np.float64).copy(); p = p.astype(np.float64).copy(); alive = np.ones(n, bool)
    D = (y[:, None] - y[None, :]) ** 2 + (p[:, None] - p[None, :]) ** 2; np.fill_diagonal(D, np.inf); i = 0
    for _ in range(n - 1):
        i, j = divmod(int(np.argmin(D)), n)
        if pt[j] > pt[i]: y[i], p[i] = y[j], p[j]
        pt[i] += pt[j]; alive[j] = False; D[j, :] = np.inf; D[:, j] = np.inf
        d = (y - y[i]) ** 2 + (p - p[i]) ** 2; d[~alive] = np.inf; d[i] = np.inf; D[i, :] = d; D[:, i] = d
    return float(y[i]), float(p[i])


def wta_axes(T):
    """WTA axis of the stored particles, absolute (eta, phi); computed in coordinates relative to the stored axis"""
    J = len(T['pt']); we, wp = np.zeros(J), np.zeros(J)
    for j in range(J):
        r = T['pt'][j] > 0
        a, b = wta_axis_one(T['pt'][j, r], T['eta'][j, r] - T['ax_eta'][j], wrap(T['phi'][j, r] - T['ax_phi'][j]))
        we[j], wp[j] = T['ax_eta'][j] + a, T['ax_phi'][j] + b
    return we, wp


def spectrum(T):
    pts = T['pt'][T['pt'] > 0]; q25 = float(np.percentile(pts, 25)); pool = pts[pts < q25]
    return dict(q25=q25, pool=pool, mean=float(pool.mean()), qmed=float(np.median(pool)), n_pool=int(len(pool)),
                all_median=float(np.median(pts)))


def inject(T, rho, spec, rng):
    J, W = T['pt'].shape
    n = rng.poisson(rho * AREA / spec['mean'], J); K = max(int(n.max()), 1)
    U = copyT(T, K); on = np.arange(K)[None, :] < n[:, None]
    q = rng.choice(spec['pool'], (J, K)); r = R * np.sqrt(rng.random((J, K))); th = 2 * np.pi * rng.random((J, K))
    U['pt'][:, W:] = np.where(on, q, 0.0)
    U['eta'][:, W:] = np.where(on, T['ax_eta'][:, None] + r * np.cos(th), 0.0)
    U['phi'][:, W:] = np.where(on, wrap(T['ax_phi'][:, None] + r * np.sin(th)), 0.0)
    if U['ext']:
        U['ext']['isNeutralHadron'][:, W:] = on; U['ext']['mask'][:, W:] = on
    return U, dict(n=n, sumpt=(q * on).sum(1))


def cut(T, thr):
    U = copyT(T); rows = np.arange(len(T['pt'])); keep = T['pt'] >= thr; keep[rows, T['pt'].argmax(1)] = True
    U['pt'] = np.where(keep, T['pt'], 0.0)
    return U


def onaxis(T, sumpt):
    U = copyT(T, 1); c = U['pt'].shape[1] - 1; ok = sumpt > 0
    U['pt'][:, c] = np.where(ok, sumpt, 0.0); U['eta'][:, c] = T['ax_eta']; U['phi'][:, c] = T['ax_phi']
    if U['ext']:
        U['ext']['isNeutralHadron'][:, c] = ok; U['ext']['mask'][:, c] = ok
    return U


def mass_all(T):
    p4 = jet4(T); return np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0))


def moments(T, ae, ap):
    real = T['pt'] > 0; pt = T['pt']
    de = np.where(real, T['eta'] - ae[:, None], 0); dp = np.where(real, wrap(T['phi'] - ap[:, None]), 0)
    M0 = pt.sum(1); M2 = (pt * (de ** 2 + dp ** 2)).sum(1); M1 = np.hypot((pt * de).sum(1), (pt * dp).sum(1))
    m = mass_all(T)
    return dict(A=0.5 * np.log(np.maximum(M0 * M2, 1e-30)), mA=np.sqrt(M0 * M2), logm=np.log(np.maximum(m, 1e-9)), m=m,
                d1=M1 / np.maximum(M0, 1e-30), sumpt=M0)


def conditions(T0, spec, seed, cap, pre_cap, wta=None, names=None, chunk=0):
    """yields (name, rho, Uf finalized, ae, ap, overflow, info); names: the condition-name prefixes to produce"""
    st = (T0['ax_eta'], T0['ax_phi'])
    want = lambda n: names is None or n.split('_')[0] in names

    def fin(U, axis=None):
        if axis is None and pre_cap: axis = eaxis(U)
        Uf, over = finalize(U, cap)
        if axis is None: axis = eaxis(Uf)
        return Uf, axis[0], axis[1], over, mass_all(U)
    if want('id'): yield ('id', 0) + fin(T0, st)
    if want('reax0'): yield ('reax0', 0) + fin(T0)
    if wta is not None and want('wta'): yield ('wta', 0) + fin(T0, wta)
    if want('sk'): yield ('sk', 0) + fin(cut(T0, spec['q25']))
    for rho in RHOS:
        rng = np.random.default_rng([seed, rho, chunk])
        U, inf = inject(T0, rho, spec, rng)
        pu = fin(U)
        if want('pu'): yield (f'pu_{rho}', rho) + pu
        if want('pufix'): yield (f'pufix_{rho}', rho) + fin(U, st)
        if want('c1m'): yield (f'c1m_{rho}', rho) + fin(cut(U, spec['qmed']))
        if want('c2'): yield (f'c2_{rho}', rho) + fin(onaxis(T0, inf['sumpt']), None)
        if want('c3'): yield (f'c3_{rho}', rho) + fin(T0, (pu[1], pu[2]))
        if want('inj'): yield (f'inj_{rho}', rho, inf)


# ================================================================================================================ statistics
def bmean(x, rng, B=NB):
    x = np.asarray(x, float); n = len(x)
    if n == 0: return dict(mean=None, ci=[None, None])
    m = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return dict(mean=float(x.mean()), ci=np.percentile(m, [2.5, 97.5]).tolist())


def q(x):
    x = np.asarray(x, float); d = {k: float(np.percentile(x, v)) for k, v in (('p10', 10), ('p50', 50), ('p90', 90), ('p99', 99))}
    d['mean'] = float(x.mean()); return d


def ols(X, y):
    X1 = np.column_stack([X, np.ones(len(y))]); c, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ c; r2 = 1 - (r ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return c, r2


def moment_fit(S, MOM, rows, conds, rng, B=300):
    """S: {cond: (J,) score}, MOM: {cond: moments}; rows: jets used. (i) axis-only slope; (ii) two-variable fit; (iii) axis part"""
    d = lambda c, k: (MOM[c][k] - MOM['id'][k])[rows]
    ds = lambda c: (S[c] - S['id'])[rows]
    ax_only = [c for c in conds if c in ('reax0', 'wta') or c.startswith('c3_')]
    two = [c for c in conds if c in ('reax0', 'wta') or c.split('_')[0] in ('c3', 'c2', 'pufix', 'pu')]
    n = len(rows); out = dict(n_jets=int(n), axis_only_conditions=ax_only, two_var_conditions=two)

    def fit_axis(ix):
        x = np.concatenate([d(c, 'A')[ix] for c in ax_only]); yv = np.concatenate([ds(c)[ix] for c in ax_only])
        return float((x * yv).sum() / max((x * x).sum(), 1e-30))       # through the origin (no edit = no response)

    def fit_two(ix):
        X = np.concatenate([np.column_stack([d(c, 'A')[ix], d(c, 'logm')[ix]]) for c in two]); yv = np.concatenate([ds(c)[ix] for c in two])
        return ols(X, yv)
    a0 = fit_axis(np.arange(n)); c0, r20 = fit_two(np.arange(n))
    boots_a, boots_c, ratios = [], [], {rho: [] for rho in RHOS}
    pu_obs = {rho: ds(f'pu_{rho}') for rho in RHOS if f'pu_{rho}' in S}; pu_dA = {rho: d(f'pu_{rho}', 'A') for rho in pu_obs}
    for _ in range(B):
        ix = rng.integers(0, n, n); a = fit_axis(ix); boots_a.append(a); boots_c.append(fit_two(ix)[0])
        for rho in pu_obs:
            den = a * pu_dA[rho][ix].mean(); ratios[rho].append(pu_obs[rho][ix].mean() / den if abs(den) > 1e-12 else np.nan)
    boots_c = np.array(boots_c)
    out['axis_slope'] = dict(a=a0, ci=np.percentile(boots_a, [2.5, 97.5]).tolist())
    out['two_var'] = dict(a_dA=float(c0[0]), b_dlogm=float(c0[1]), intercept=float(c0[2]), r2=float(r20),
                          a_ci=np.percentile(boots_c[:, 0], [2.5, 97.5]).tolist(), b_ci=np.percentile(boots_c[:, 1], [2.5, 97.5]).tolist())
    out['pu_prediction'] = {}
    for rho in pu_obs:
        o, pr = float(pu_obs[rho].mean()), float(a0 * pu_dA[rho].mean())
        out['pu_prediction'][str(rho)] = dict(observed=o, predicted_axis_slope=pr, dA=float(pu_dA[rho].mean()),
                                               dlogm=float(d(f'pu_{rho}', 'logm').mean()),
                                               ratio=o / pr if abs(pr) > 1e-12 else None,
                                               ratio_ci=np.nanpercentile(ratios[rho], [2.5, 97.5]).tolist())
        if f'pufix_{rho}' in S and f'c3_{rho}' in S:
            axp = ds(f'pu_{rho}') - ds(f'pufix_{rho}'); c3 = ds(f'c3_{rho}'); diff = axp - c3
            bd = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
            out['pu_prediction'][str(rho)]['axis_part'] = dict(pu_minus_pufix=bmean(axp, rng, B), c3=bmean(c3, rng, B),
                                                                diff=float(diff.mean()), diff_ci=np.percentile(bd, [2.5, 97.5]).tolist(),
                                                                p_boot=float(min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))))
    out['per_condition'] = {c: dict(ds=float(ds(c).mean()), dA=float(d(c, 'A').mean()), dlogm=float(d(c, 'logm').mean()))
                            for c in S if c != 'id'}
    return out


def offsets(AX, wta):
    """axis offsets per rho (dR in eta-phi, rad): pu axis vs stored, vs E-scheme of the stored particles, vs WTA"""
    st = AX['id']; e0 = AX['reax0']; out = dict(reax0_vs_stored=q(dR(*e0, *st)))
    if wta is not None: out['wta_vs_stored'] = q(dR(*wta, *st)); out['wta_vs_escheme'] = q(dR(*wta, *e0))
    out['sk_vs_stored'] = q(dR(*AX['sk'], *st))
    for rho in RHOS:
        a = AX.get(f'pu_{rho}')
        if a is None: continue
        o = dict(vs_stored=q(dR(*a, *st)), vs_escheme=q(dR(*a, *e0)))
        if wta is not None: o['vs_wta'] = q(dR(*a, *wta))
        if f'c1m_{rho}' in AX: o['c1m_vs_stored'] = q(dR(*AX[f'c1m_{rho}'], *st))
        if f'c2_{rho}' in AX: o['c2_vs_stored'] = q(dR(*AX[f'c2_{rho}'], *st))
        out[f'pu_{rho}'] = o
    return out


def cond_info(name, MOM, M0, over, mall, mall0, inf, rng):
    d = dict(overflow=float(over.mean()), dA=bmean(MOM['A'] - M0['A'], rng, 200), dlogm_kept=bmean(MOM['logm'] - M0['logm'], rng, 200),
             dm_all=bmean(mall - mall0, rng, 200), dsumpt_kept=bmean(MOM['sumpt'] - M0['sumpt'], rng, 200),
             d1_median=float(np.median(MOM['d1'])))
    if inf is not None:
        d['n_added'] = q(inf['n']); d['sumpt_added'] = q(inf['sumpt'])
    return d


# ================================================================================================================ generic panel
def run_panel(tag, T0, y, models, make_in, cap, pre_cap, classes, sig_bkg, score_fn, reg_rows, smoke, wta=None):
    """multi-class panel (hls4ml, JetClass). models: {name: f(inputs) -> logits (J, C)}; score_fn(L) -> scalar score per jet"""
    rng = np.random.default_rng(SEED); spec = spectrum(T0)
    res = dict(part=tag, N=int(len(y)), seed=SEED, cap=cap, axis_from='all particles' if pre_cap else 'kept particles',
               classes=classes, labels=np.bincount(y, minlength=len(classes)).tolist(),
               spectrum={k: v for k, v in spec.items() if k != 'pool'}, n_real=q((T0['pt'] > 0).sum(1)),
               pT_jet=q(np.hypot(*[(a * (T0['pt'] > 0)).sum(1) for a in p3(T0['pt'], T0['eta'], T0['phi'])[:2]])),
               models={}, conditions={})
    log(f'{tag}: soft spectrum q25 {spec["q25"]:.3f} GeV, mean {spec["mean"]:.3f}, median {spec["qmed"]:.3f} '
        f'(pool {spec["n_pool"]}); expected particles at rho 40: {40 * AREA / spec["mean"]:.1f}')
    L = {m: {} for m in models}; MOM = {}; AX = {}; NPZ = dict(y=y); INF = {}
    for name, rho, *rest in conditions(T0, spec, SEED, cap, pre_cap, wta, names=None):
        if name.startswith('inj'):
            INF[rho] = rest[0]; continue
        t = time.time(); Uf, ae, ap, over, mall = rest
        MOM[name] = moments(Uf, ae, ap); MOM[name]['mall'] = mall; MOM[name]['over'] = over; AX[name] = (ae, ap)
        X = make_in(Uf, ae, ap)
        for mn, f in models.items():
            L[mn][name] = f(X)
        msg = ' '.join(f'{mn} {(L[mn][name].argmax(1) != L[mn]["id"].argmax(1)).mean():.4f}' for mn in models)
        log(f'{tag} {name}: overflow {over.mean():.3f} dA {np.mean(MOM[name]["A"] - MOM["id"]["A"]):+.4f} '
            f'dm_all {np.mean(mall - MOM["id"]["mall"]):+.2f} GeV | changed {msg} ({time.time() - t:.0f} s)')
    conds = list(MOM)
    for name in conds:
        rho = int(name.split('_')[1]) if '_' in name else 0
        res['conditions'][name] = cond_info(name, MOM[name], MOM['id'], MOM[name]['over'], MOM[name]['mall'], MOM['id']['mall'],
                                            INF.get(rho) if name.startswith('pu_') else None, rng)
    res['axis_offsets'] = offsets(AX, wta)
    # classical mass-window calls from the cheap masses (all jets): W(m) on the ungroomed mass of all particles, W(mA)
    res['mass_window_calls'] = {}
    for name in conds:
        for k, v in (('W_m_all', MOM[name]['mall']), ('W_mA', MOM[name]['mA'])):
            v0 = MOM['id']['mall'] if k == 'W_m_all' else MOM['id']['mA']
            c0, c1 = (v0 > 65) & (v0 < 95), (v > 65) & (v < 95); ch = c0 != c1
            res['mass_window_calls'].setdefault(k, {})[name] = dict(changed=float(ch.mean()), ci=boot_binary(ch, rng),
                                                                     pass_before=float(c0.mean()), pass_after=float(c1.mean()))
    tests = []
    for mn in models:
        L0 = L[mn]['id']; y0 = L0.argmax(1); r = dict(acc=float((y0 == y).mean()), acc_ci=boot_binary(y0 == y, rng),
                                                       per_class={classes[k]: float((y0[y == k] == k).mean()) for k in range(len(classes))}, results={})
        NPZ[f'{mn}__id'] = L0.astype(np.float32)
        CH = {}
        for name in conds:
            if name == 'id': continue
            L1 = L[mn][name]; st = cond_stats(L0, L1, y, np.ones(len(y), bool), MOM[name]['over'], rng, classes, sig_bkg)
            s0, s1 = score_fn(L0), score_fn(L1); st['dscore_reg_rows'] = bmean((s1 - s0)[reg_rows], rng)
            r['results'][name] = st; CH[name] = L1.argmax(1) != y0
            NPZ[f'{mn}__{name}'] = L1.astype(np.float16)
        for rho in RHOS:
            pn = f'pu_{rho}'
            if pn not in CH: continue
            allj = np.ones(len(y), bool)
            r['results'][pn]['vs_sk'] = paired(CH[pn], CH['sk'], allj, rng)
            r['results'][pn]['vs_c2'] = paired(CH[pn], CH[f'c2_{rho}'], allj, rng)
            r['results'][pn]['vs_c3'] = paired(CH[pn], CH[f'c3_{rho}'], allj, rng)
            tests.append((mn, pn, r['results'][pn]['vs_sk']['p_mcnemar']))
        S = {c: score_fn(L[mn][c]) for c in conds}
        r['moment_fit'] = moment_fit(S, MOM, reg_rows, conds, rng)
        res['models'][mn] = r
        log(f'{tag} {mn}: accuracy {r["acc"]:.4f}; moment fit a {r["moment_fit"]["axis_slope"]["a"]:.3f} '
            f'two-var a {r["moment_fit"]["two_var"]["a_dA"]:.3f} b {r["moment_fit"]["two_var"]["b_dlogm"]:.3f}')
    hp = holm([t[2] for t in tests])
    for (mn, pn, _), h in zip(tests, hp): res['models'][mn]['results'][pn]['vs_sk']['holm_p_part'] = h
    res['n_tests_part'] = len(tests)
    for name in conds:
        for k in ('A', 'logm', 'mA', 'mall'): NPZ[f'mom__{name}__{k}'] = MOM[name][k].astype(np.float32)
        NPZ[f'ax__{name}'] = np.stack(AX[name], 1).astype(np.float32)
    if wta is not None: NPZ['ax__wta_abs'] = np.stack(wta, 1).astype(np.float32)
    return res, NPZ


def save(res, NPZ, part, smoke):
    OUT.mkdir(parents=True, exist_ok=True); sfx = '_smoke' if smoke else ''
    p = OUT / f'h13_{part}{sfx}.json'; p.write_text(json.dumps(res, indent=1, default=float))
    if NPZ is not None: np.savez_compressed(OUT / f'h13_{part}{sfx}_perjet.npz', **NPZ)
    log(f'wrote {p}')


# ------------------------------------------------------------------------------------------------ hls4ml
HLS_CL = ['g', 'q', 'W', 'Z', 't']; HLS_SB = [(2, [0, 1]), (3, [0, 1]), (4, [0, 1])]


def hls_in(Uf, ae, ap):
    real = Uf['pt'] > 0
    return np.stack([Uf['pt'], np.where(real, Uf['eta'] - ae[:, None], 0), np.where(real, wrap(Uf['phi'] - ap[:, None]), 0)], -1).astype(np.float32)


def load_hls(smoke):
    T, y = hls_sample(smoke or 2000, SEED)
    return T, y


def hls_score(L):
    return L[:, 3] - L[:, 2]


def part_jedi(smoke):
    from interp.jedi_core import Jedi
    T, y = load_hls(smoke); wta = wta_axes(T); log('WTA axes done')
    nets = dict(jedi=Jedi(64, '3-feature'), jedi_scrambled=Jedi(64, '3-feature', untrained=True, seed=0))
    models = {k: (lambda x, n=n: np.asarray(n.run(x)['logits'], np.float64)) for k, n in nets.items()}
    reg = np.flatnonzero(np.isin(y, [2, 3]))
    res, NPZ = run_panel('jedi', T, y, models, hls_in, 64, True, HLS_CL, HLS_SB, hls_score, reg, smoke, wta)
    res['sample'] = f'jets of jedi_core.sample(per_class={smoke or 2000}, seed={SEED})'
    res['known_acc'] = dict(jedi='0.8092 (260k test set); 0.8051 / 0.8116 on the H1 / H1v samples')
    save(res, NPZ, 'jedi', smoke)


def part_small(smoke):
    import torch
    from interp.h1_irc_panel import load_small, torch_logits, efn_class
    torch.set_num_threads(1)
    T, y = load_hls(smoke); wta = wta_axes(T); log('WTA axes done')
    models = {}
    net, sh, sc = load_small(SMALL / 'dsets-d64-n64' / 'seed0'); models['dsets'] = (lambda x, net=net, sh=sh, sc=sc: torch_logits(net, x, sh, sc))
    net2, sh2, sc2 = load_small(SMALL / 'dsets-d64-n64' / 'seed0'); g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in net2.parameters():
            f = p.data.flatten(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p))
    models['dsets_permuted'] = (lambda x, net=net2, sh=sh2, sc=sc2: torch_logits(net, x, sh, sc))
    efn = efn_class()(); efn.load_state_dict(torch.load(OUT / 'h1_efn' / 'best.pt', map_location='cpu')); efn.eval()
    models['efn'] = (lambda x: torch_logits(efn, x, np.zeros(3, np.float32), np.ones(3, np.float32)))
    reg = np.flatnonzero(np.isin(y, [2, 3]))
    res, NPZ = run_panel('small', T, y, models, hls_in, 64, True, HLS_CL, HLS_SB, hls_score, reg, smoke, wta)
    res['sample'] = f'jets of jedi_core.sample(per_class={smoke or 2000}, seed={SEED})'
    res['known_acc'] = dict(dsets=json.loads((SMALL / 'dsets-d64-n64' / 'seed0' / 'test.json').read_text())['accuracy'],
                            efn=json.loads((OUT / 'h1_efn' / 'info.json').read_text())['test_acc'])
    save(res, NPZ, 'small', smoke)


# ------------------------------------------------------------------------------------------------ JetClass
def load_jc(smoke):
    from interp.h1_irc_panel import jc_sample, jc_table
    J0 = jc_sample(smoke or 1000, SEED); y = J0['label'].astype(int); T = jc_table(J0)
    return J0, T, y


def jc_maker(J0):
    from interp.h1_irc_panel import jc_jets

    def make(Uf, ae, ap):
        Tin = dict(Uf); Tin['ax_eta'] = ae; Tin['ax_phi'] = ap; Tin['sgn'] = np.where(ae < 0, -1.0, 1.0)
        Jf = dict(J0); Jf['jet_eta'] = ae.astype(J0['jet_eta'].dtype); Jf['jet_phi'] = ap.astype(J0['jet_phi'].dtype)
        return jc_jets(Jf, Tin)
    return make


def part_jetclass(smoke):
    from interp.part_core import ParTNetwork, CL, inputs, build, kin
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from interp.h1_irc_panel import pn_logits
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu')
    J0, T, y = load_jc(smoke); make = jc_maker(J0)
    Jb = make(*finalize(T, 128)[:1], T['ax_eta'], T['ax_phi']); Jref = build(J0, *kin(J0))
    rt = max(float(np.abs(Jb[k] - Jref[k]).max()) for k in ('px', 'py', 'pz', 'energy', 'deta', 'dphi'))
    log(f'round trip table -> inputs vs build(kin(J0)): max abs diff {rt:.2e}')
    wta = wta_axes(T); log('WTA axes done')
    models = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); models[f'ParT_{n}'] = (lambda J, net=net, n=n: np.asarray(fwd(net, *inputs(J, n))[1], np.float64))
    net = ParTNetwork('full', untrained=True, seed=0); models['ParT_full_scrambled'] = (lambda J, net=net: np.asarray(fwd(net, *inputs(J, 'full'))[1], np.float64))
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        models[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
    sb = [(CL.index('Wqq'), [0]), (CL.index('Zqq'), [0]), (CL.index('Tbqq'), [0]), (CL.index('Hbb'), [0])]
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    reg = np.flatnonzero(np.isin(y, [iW, iZ]))
    res, NPZ = run_panel('jetclass', T, y, models, make, 128, True, list(CL), sb, lambda L: L[:, iZ] - L[:, iW], reg, smoke, wta)
    res['sample'] = f'random rows (seed {SEED}) of each class first test file, {smoke or 1000} per class'; res['roundtrip_maxdiff'] = rt
    res['known_acc'] = dict(ParT_full='0.859 (D4), 0.856 / 0.863 (H1 / H1v)', ParT_kin='0.745 (D4), 0.736 / 0.742', PN_full='0.839 / 0.846',
                            PN_kin='0.695 / 0.704')
    save(res, NPZ, 'jetclass', smoke)


# ------------------------------------------------------------------------------------------------ Top Landscape
def top_table(full):
    P = np.asarray(full[..., :4], np.float64); pt = np.hypot(P[..., 0], P[..., 1]); real = pt > 0
    _, eta, phi = from3(P[..., 0], P[..., 1], np.where(real, P[..., 2], 0))
    m = np.sqrt(np.maximum(P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1), 0))
    T = dict(pt=np.where(real, pt, 0), eta=np.where(real, eta, 0), phi=np.where(real, phi, 0), m=np.where(real, m, 0), ext={})
    T['ax_eta'], T['ax_phi'] = eaxis(T); T['sgn'] = np.where(T['ax_eta'] < 0, -1.0, 1.0)
    return T


def top_J(Uf, ae, ap):
    """the dict interp.h4_top_wmass.jet_inputs builds, with deta, dphi about the given axis (ae, ap)"""
    real = Uf['pt'] > 0; px, py, pz = p3(Uf['pt'], Uf['eta'], Uf['phi'])
    E = np.sqrt((Uf['pt'] * np.cosh(Uf['eta'])) ** 2 + Uf['m'] ** 2)
    px, py, pz, E = (np.where(real, a, 0) for a in (px, py, pz, E))
    sg = np.where(ae < 0, -1.0, 1.0)[:, None]
    return dict(px=px, py=py, pz=pz, energy=E, pt=np.where(real, Uf['pt'], 0), mask=real,
                deta=np.where(real, (Uf['eta'] - ae[:, None]) * sg, 0), dphi=np.where(real, wrap(Uf['phi'] - ap[:, None]), 0),
                jet_pt=np.hypot(px.sum(1), py.sum(1)), jet_energy=E.sum(1))


class TopModels:
    def __init__(self, dev):
        import torch
        from interp.h4_top_wmass import WeaverNet, LNet, Flow
        self.torch, self.dev = torch, dev
        self.weaver = {k: WeaverNet(k, V / 'models/top' / d / 'net_best_epoch_state.pt', dev) for k, d in (('ParT', 'part_kin'), ('PN', 'pnet_kin'))}
        self.lnet = LNet(dev, mass64=True)
        self.flow = {'PFN': Flow('pfn'), 'EFN': Flow('efn')}
        perm = Flow('pfn'); rng = np.random.default_rng(7)
        pr = lambda a: rng.permutation(a.ravel()).reshape(a.shape)
        perm.phi = [(pr(K), pr(b)) for K, b in perm.phi]; perm.F = [(pr(K), pr(b)) for K, b in perm.F]; perm.out = (pr(perm.out[0]), pr(perm.out[1]))
        self.flow['PFN_permuted'] = perm
        self.names = ['ParT', 'PN', 'LNet', 'PFN', 'EFN', 'PFN_permuted']

    def score(self, J, order, batch=512):
        from interp.h4_top_wmass import weaver_inputs
        torch = self.torch; N = len(J['jet_pt']); out = {}
        for k, w in self.weaver.items():
            s = np.zeros(N)
            with torch.no_grad():
                for a in range(0, N, batch):
                    Jb = {kk: v[a:a + batch] for kk, v in J.items()}; pts, x, v, m = weaver_inputs(Jb)
                    d = lambda t: torch.from_numpy(t).to(self.dev)
                    o = (w.model(d(x), v=d(v), mask=d(m)) if k == 'ParT' else w.model(d(pts), d(x), d(m))).float().cpu().numpy()
                    s[a:a + len(o)] = o[:, 0] - o[:, 1]
            out[k] = s
        nn = J['mask'].sum(1).astype(int); L = int(nn.max())
        P4 = np.stack([J['px'][:, :L], J['py'][:, :L], J['pz'][:, :L], J['energy'][:, :L]], -1).astype(np.float32)
        out['LNet'] = self.lnet.score(P4, nn, order)
        k3 = np.stack([J['pt'][:, :128], J['deta'][:, :128], J['dphi'][:, :128]], -1).astype(np.float16).astype(np.float32)
        for k, f in self.flow.items(): out[k] = f.score(k3)
        return out


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rej(ss, sb, eps, thr=None):
    t = np.quantile(ss, 1 - eps) if thr is None else thr; e = (sb > t).mean()
    return (1 / e if e > 0 else float('inf')), t


def rej_boot(ss0, ss1, sb0, sb1, rng, B=200):
    """1/eps_B at eps_S 0.3, 0.5 for unedited (0) and edited (1) signal / background scores: thresholds re-set on the edited
    signal ('reset') and frozen on the unedited signal ('frozen'); bootstrap over signal and background jets (paired)"""
    out = {}
    for eps in (0.3, 0.5):
        r0, t0 = rej(ss0, sb0, eps); r1, _ = rej(ss1, sb1, eps); rf, _ = rej(ss1, sb1, eps, t0)
        bs = dict(before=[], reset=[], frozen=[], ratio=[])
        for _ in range(B):
            i = rng.integers(0, len(ss0), len(ss0)); j = rng.integers(0, len(sb0), len(sb0))
            a, ta = rej(ss0[i], sb0[j], eps); b, _ = rej(ss1[i], sb1[j], eps); c, _ = rej(ss1[i], sb1[j], eps, ta)
            bs['before'].append(a); bs['reset'].append(b); bs['frozen'].append(c); bs['ratio'].append(b / a)
        ci = lambda v: np.percentile(np.asarray(v, float), [2.5, 97.5]).tolist()
        out[str(eps)] = dict(before=r0, before_ci=ci(bs['before']), reset=r1, reset_ci=ci(bs['reset']), frozen=rf, frozen_ci=ci(bs['frozen']),
                             eps_s_frozen=float((ss1 > t0).mean()), ratio_reset=r1 / r0, ratio_ci=ci(bs['ratio']))
    return out


def load_top(smoke):
    from interp import data_top as dt
    S = dt.sample(smoke or 5000, SEED)
    return S, top_table(S['full']), S['label'].astype(int)


def part_top(smoke):
    import torch
    from interp import data_top as dt
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    S, T, y = load_top(smoke); cap = 200
    log(f'top sample: {len(y)} jets, constituents per jet {q((T["pt"] > 0).sum(1))}')
    wta = wta_axes(T); log('WTA axes done')
    M = TopModels(dev); order = np.random.default_rng(305).permutation(len(y))
    rng = np.random.default_rng(SEED); spec = spectrum(T)
    log(f'top: soft spectrum q25 {spec["q25"]:.3f} GeV, mean {spec["mean"]:.3f}, median {spec["qmed"]:.3f}; expected particles at rho 40: {40 * AREA / spec["mean"]:.1f}')
    SC = {m: {} for m in M.names}; MOM = {}; AX = {}; INF = {}
    for name, rho, *rest in conditions(T, spec, SEED, cap, False, wta):
        if name.startswith('inj'):
            INF[rho] = rest[0]; continue
        t = time.time(); Uf, ae, ap, over, mall = rest
        MOM[name] = moments(Uf, ae, ap); MOM[name]['mall'] = mall; MOM[name]['over'] = over; AX[name] = (ae, ap)
        sc = M.score(top_J(Uf, ae, ap), order)
        for k, v in sc.items(): SC[k][name] = v
        log(f'top {name}: overflow {over.mean():.3f} dA {np.mean(MOM[name]["A"] - MOM["id"]["A"]):+.4f} dm {np.mean(mall - MOM["id"]["mall"]):+.2f} | '
            + ' '.join(f'{k} {np.mean(SC[k][name] - SC[k]["id"]):+.3f}' for k in M.names) + f' ({time.time() - t:.0f} s)')
    conds = list(MOM)
    res = dict(part='top', N=int(len(y)), seed=SEED, cap=cap, labels=np.bincount(y).tolist(), spectrum={k: v for k, v in spec.items() if k != 'pool'},
               n_real=q((T['pt'] > 0).sum(1)), sample=f'data_top.sample({smoke or 5000}, {SEED})', models={}, conditions={})
    for name in conds:
        rho = int(name.split('_')[1]) if '_' in name else 0
        res['conditions'][name] = cond_info(name, MOM[name], MOM['id'], MOM[name]['over'], MOM[name]['mall'], MOM['id']['mall'],
                                            INF.get(rho) if name.startswith('pu_') else None, rng)
    res['axis_offsets'] = offsets(AX, wta)
    tests = []; NPZ = dict(y=y)
    for k in M.names:
        s0 = SC[k]['id']; t50 = float(np.median(s0[y == 1])); t30 = float(np.quantile(s0[y == 1], 0.7))
        r = dict(auc=auc(s0, y), acc_logodds0=float(((s0 > 0) == (y == 1)).mean()), t50=t50, results={})
        CH = {}
        for name in conds:
            if name == 'id': continue
            s1 = SC[k][name]; c0, c1 = s0 > t50, s1 > t50; ch = c0 != c1; CH[name] = ch
            r['results'][name] = dict(changed=float(ch.mean()), ci=boot_binary(ch, rng), auc_after=auc(s1, y),
                                      dlogodds_top=bmean((s1 - s0)[y == 1], rng), dlogodds_qcd=bmean((s1 - s0)[y == 0], rng),
                                      tag_rate_top=[float(c0[y == 1].mean()), float(c1[y == 1].mean())],
                                      tag_rate_qcd=[float(c0[y == 0].mean()), float(c1[y == 0].mean())],
                                      changed_30=float(((s0 > t30) != (s1 > t30)).mean()))
            NPZ[f'{k}__{name}'] = s1.astype(np.float32)
        NPZ[f'{k}__id'] = s0.astype(np.float32)
        allj = np.ones(len(y), bool)
        for rho in RHOS:
            pn = f'pu_{rho}'
            r['results'][pn]['vs_sk'] = paired(CH[pn], CH['sk'], allj, rng)
            r['results'][pn]['vs_c2'] = paired(CH[pn], CH[f'c2_{rho}'], allj, rng)
            r['results'][pn]['vs_c3'] = paired(CH[pn], CH[f'c3_{rho}'], allj, rng)
            tests.append((k, pn, r['results'][pn]['vs_sk']['p_mcnemar']))
        r['moment_fit'] = moment_fit(SC[k], MOM, np.arange(len(y)), conds, rng)
        res['models'][k] = r
        log(f'top {k}: AUC {r["auc"]:.4f}; pu_40 changed {r["results"]["pu_40"]["changed"]:.4f}; axis slope {r["moment_fit"]["axis_slope"]["a"]:.3f}')
    hp = holm([t[2] for t in tests])
    for (k, pn, _), h in zip(tests, hp): res['models'][k]['results'][pn]['vs_sk']['holm_p_part'] = h
    res['n_tests_part'] = len(tests)
    res['known'] = dict(ParT='AUC 0.9858 (H2, full test)', PN='AUC 0.9858', LNet='AUC 0.9870 (released scores)', PFN='AUC 0.9819', EFN='AUC ~0.98')
    for name in conds:
        for kk in ('A', 'logm', 'mA', 'mall'): NPZ[f'mom__{name}__{kk}'] = MOM[name][kk].astype(np.float32)
    save(res, NPZ, 'top', smoke)
    # background rejection: 100k QCD jets (seed 303, main-sample rows excluded), edited in chunks of 10k with the same spectrum
    nb = 3000 if smoke else 100_000
    yt = np.load(dt.DST / 'test' / 'label.npy'); cand = np.setdiff1d(np.flatnonzero(yt == 0), S['idx'])
    bidx = np.sort(np.random.default_rng(303).choice(cand, nb, replace=False))
    names = ['id', 'sk', 'pu', 'c2', 'c1m']; SB = {k: {} for k in M.names}
    fullmm = np.load(dt.DST / 'test' / 'full.npy', mmap_mode='r')
    for ci, a in enumerate(range(0, nb, 10_000)):
        t = time.time(); Tb = top_table(np.asarray(fullmm[bidx[a:a + 10_000]])); ob = np.random.default_rng([306, ci]).permutation(len(Tb['pt']))
        for name, rho, *rest in conditions(Tb, spec, SEED + 1, cap, False, None, names=names, chunk=ci):
            if name not in BKG_KEEP: continue
            Uf, ae, ap, over, mall = rest
            for k, v in M.score(top_J(Uf, ae, ap), ob).items(): SB[k].setdefault(name, []).append(v)
        log(f'background chunk {ci}: {len(Tb["pt"])} jets ({time.time() - t:.0f} s)')
    res['rejection'] = dict(n_bkg=int(nb), n_sig=int((y == 1).sum()), bkg='QCD test rows, seed 303, main sample excluded', models={})
    for k in M.names:
        sb = {n: np.concatenate(v) for n, v in SB[k].items()}; ss = {n: SC[k][n][y == 1] for n in sb}
        res['rejection']['models'][k] = {n: rej_boot(ss['id'], ss[n], sb['id'], sb[n], rng) for n in sb}
        rr = res['rejection']['models'][k]
        log(f'top {k}: R30 id {rr["id"]["0.3"]["before"]:.0f}, pu_40 reset {rr["pu_40"]["0.3"]["reset"]:.0f}, sk {rr["sk"]["0.3"]["reset"]:.0f}')
        for n in sb: NPZ[f'bkg__{k}__{n}'] = sb[n].astype(np.float32)
    save(res, NPZ, 'top', smoke)


# ------------------------------------------------------------------------------------------------ classical observables
try:
    from numba import njit, prange
except ImportError:  # local import check only
    njit = lambda *a, **k: (lambda f: f); prange = range


@njit(parallel=True, cache=False)
def d2_numba(F, nn):
    """D2 (beta 1) = e3 / e2^3 on all particles, rapidity-azimuth distances, z_i = pT_i / sum pT"""
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


def obs_of(Uf, ae, ap):
    from interp.h2_onesignal_setup import top_obs
    real = Uf['pt'] > 0; px, py, pz = p3(Uf['pt'], Uf['eta'], Uf['phi']); E = np.sqrt((Uf['pt'] * np.cosh(Uf['eta'])) ** 2 + Uf['m'] ** 2)
    F = np.zeros(Uf['pt'].shape + (6,)); F[..., 0], F[..., 1], F[..., 2], F[..., 3] = (np.where(real, a, 0) for a in (px, py, pz, E))
    nn = real.sum(1).astype(np.int64); F = np.ascontiguousarray(F)
    o = top_obs(F, nn, 0.1, 0.8); d2 = d2_numba(F, nn); mo = moments(Uf, ae, ap)
    return dict(m=o[:, 0], mSD=o[:, 1], tau21=o[:, 3] / np.maximum(o[:, 2], 1e-12), tau32=o[:, 4] / np.maximum(o[:, 3], 1e-12), D2=d2, mA=mo['mA'], N=nn.astype(float))


TAGGERS = dict(W_m=lambda O: (O['m'] > 65) & (O['m'] < 95), W_mSD=lambda O: (O['mSD'] > 65) & (O['mSD'] < 95),
               W_mA=lambda O: (O['mA'] > 65) & (O['mA'] < 95),
               top_mSD_tau32=lambda O: (O['mSD'] > 105) & (O['mSD'] < 210) & (O['tau32'] < 0.65),
               top_m_tau32=lambda O: (O['m'] > 105) & (O['m'] < 210) & (O['tau32'] < 0.65))


def part_obs(ds, smoke):
    if ds == 'hls':
        T, y = load_hls(smoke); pre = True; classes = HLS_CL; sig = dict(W=[2], Z=[3], t=[4], qcd=[0, 1])
    elif ds == 'jc':
        from interp.part_core import CL
        _, T, y = load_jc(smoke); pre = True; classes = list(CL); sig = dict(W=[CL.index('Wqq')], Z=[CL.index('Zqq')], t=[CL.index('Tbqq')], qcd=[0])
    else:
        _, T, y = load_top(smoke); pre = False; classes = ['QCD', 'top']; sig = dict(t=[1], qcd=[0])
    rows = np.arange(len(y))[::5]; rng = np.random.default_rng(SEED); spec = spectrum(T)
    Ts = {k: T[k][rows] for k in ('pt', 'eta', 'phi', 'm', 'ax_eta', 'ax_phi', 'sgn')}; Ts['ext'] = {}
    ys = y[rows]; O = {}
    # same injected particles as the model parts: the conditions are generated on the full sample, then the rows are taken
    for name, rho, *rest in conditions(T, spec, SEED, 10 ** 6, pre, None, names=['id', 'sk', 'pu', 'c1m', 'c2']):
        if name.startswith('inj'): continue
        t = time.time(); Uf, ae, ap, over, mall = rest
        Us = {k: Uf[k][rows] for k in ('pt', 'eta', 'phi', 'm')}; w = int((Us['pt'] > 0).sum(1).max())
        Us = {k: np.ascontiguousarray(v[:, :w]) for k, v in Us.items()}
        O[name] = obs_of(Us, ae[rows], ap[rows])
        log(f'obs {ds} {name}: ' + ' '.join(f'{k} {np.mean(O[name][k] - O["id"][k]):+.3g}' for k in O[name]) + f' ({time.time() - t:.0f} s)')
    res = dict(part=f'obs_{ds}', rows='[::5] of the sample', n=int(len(rows)), labels=np.bincount(ys, minlength=len(classes)).tolist(),
               classes=classes, base={k: q(v) for k, v in O['id'].items()}, conditions={}, taggers={})
    pT_jet = np.hypot(*[(a * (Ts['pt'] > 0)).sum(1) for a in p3(Ts['pt'], Ts['eta'], Ts['phi'])[:2]])
    for name in O:
        if name == 'id': continue
        rho = int(name.split('_')[1]) if '_' in name else 0
        d = {k: bmean(O[name][k] - O['id'][k], rng, 300) for k in O[name]}
        for cl, ks in sig.items():
            sel = np.isin(ys, ks)
            if sel.any(): d[f'by_class_{cl}'] = {k: float(np.mean((O[name][k] - O['id'][k])[sel])) for k in ('m', 'mSD', 'mA', 'tau32', 'D2')}
        if name.startswith('pu_'):
            m0 = np.maximum(O['id']['m'], 1.0)
            pred_dm2 = pT_jet * rho * AREA * R ** 2 / 2                     # small-angle: dm^2 = pT_J sum_i q_i r_i^2, <r^2> = R^2 / 2
            d['pred_dm_small_angle'] = float(np.mean(np.sqrt(m0 ** 2 + pred_dm2) - m0))
        res['conditions'][name] = d
    for tn, f in TAGGERS.items():
        c0 = f(O['id']); r = dict(pass_before={cl: float(c0[np.isin(ys, ks)].mean()) for cl, ks in sig.items()}, conditions={})
        CH = {}
        for name in O:
            if name == 'id': continue
            c1 = f(O[name]); ch = c0 != c1; CH[name] = ch
            r['conditions'][name] = dict(changed=float(ch.mean()), ci=boot_binary(ch, rng),
                                         pass_after={cl: float(c1[np.isin(ys, ks)].mean()) for cl, ks in sig.items()})
        for rho in RHOS:
            r['conditions'][f'pu_{rho}']['vs_sk'] = paired(CH[f'pu_{rho}'], CH['sk'], np.ones(len(ys), bool), rng)
        res['taggers'][tn] = r
    NPZ = dict(y=ys, rows=rows, **{f'{n}__{k}': v.astype(np.float32) for n, o in O.items() for k, v in o.items()})
    save(res, NPZ, f'obs_{ds}', smoke)


# ------------------------------------------------------------------------------------------------ summary + figure
LS_CONDS = ('wta', 'pu_5', 'pufix_5', 'c1m_5', 'c2_40', 'c3_40')


def local_slopes(B=500):
    """matched-size comparison of the moment response: per model, the per-jet slope (through the origin) of the score change on
    dA = delta 1/2 log(M0 M2) in each condition; the WTA axis edit (dA ~ 0.1-0.2, delta log m = 0) against the smallest pileup
    edit pu_5 (dA ~ 0.1-0.2, delta log m ~ dA). H1'' (axis-relative moment only): slope(pu_5) / slope(wta) = 1; H1 (invariant
    mass only): slope(wta) = 0."""
    out = {}
    spec = dict(jedi=(lambda L: L[:, 3] - L[:, 2], [2, 3]), small=(lambda L: L[:, 3] - L[:, 2], [2, 3]),
                jetclass=(lambda L: L[:, 6] - L[:, 7], [6, 7]), top=(None, None))
    for p, (sf, cls) in spec.items():
        f = OUT / f'h13_{p}_perjet.npz'
        if not f.exists(): continue
        Z = np.load(f); y = Z['y']; rows = np.flatnonzero(np.isin(y, cls)) if cls else np.arange(len(y))
        models = sorted({k.split('__')[0] for k in Z.files if k.endswith('__id') and not k.startswith(('mom', 'ax', 'bkg'))})
        rng = np.random.default_rng(13); n = len(rows)
        for mn in models:
            s0 = Z[f'{mn}__id'].astype(np.float64); s0 = sf(s0) if sf else s0
            D = {}
            for c in LS_CONDS:
                if f'{mn}__{c}' not in Z.files: continue
                s1 = Z[f'{mn}__{c}'].astype(np.float64); s1 = sf(s1) if sf else s1
                D[c] = ((s1 - s0)[rows], (Z[f'mom__{c}__A'] - Z['mom__id__A'])[rows].astype(np.float64),
                        (Z[f'mom__{c}__logm'] - Z['mom__id__logm'])[rows].astype(np.float64))
            sl = lambda c, ix: float((D[c][1][ix] * D[c][0][ix]).sum() / max((D[c][1][ix] ** 2).sum(), 1e-30))
            r = {c: dict(slope=sl(c, np.arange(n)), ds=float(D[c][0].mean()), dA=float(D[c][1].mean()), dlogm=float(D[c][2].mean())) for c in D}
            bs = {c: [] for c in D}; ratio = []
            for _ in range(B):
                ix = rng.integers(0, n, n)
                for c in D: bs[c].append(sl(c, ix))
                ratio.append(bs['pu_5'][-1] / bs['wta'][-1] if abs(bs['wta'][-1]) > 1e-12 else np.nan)
            for c in D: r[c]['slope_ci'] = np.percentile(bs[c], [2.5, 97.5]).tolist()
            r['ratio_pu5_wta'] = dict(value=r['pu_5']['slope'] / r['wta']['slope'] if abs(r['wta']['slope']) > 1e-12 else None,
                                      ci=np.nanpercentile(ratio, [2.5, 97.5]).tolist())
            out[f'{p}|{mn}'] = r
            log(f'local slopes {p} {mn}: ' + ' '.join(f'{c} {r[c]["slope"]:+.2f}' for c in D) + f' ratio {r["ratio_pu5_wta"]["value"]}')
    return out


EFF_CONDS = ('id', 'sk', 'pu_5', 'pu_10', 'pu_20', 'pu_40', 'c1m_40', 'c2_40')


def eff_table():
    """signal efficiency and QCD mistag of each network's call (argmax == class) against classical taggers on the same jets:
    W(m) = 65 < m < 95 on the ungroomed mass of all particles (all jets of the sample), W(mSD) = 65 < mSD < 95 and
    top(mSD, tau32) (rows [::5], obs part). hls4ml: W = 2, Z = 3, t = 4, QCD = g + q; JetClass: Wqq = 7, Zqq = 6, Tbqq = 8,
    QCD = 0."""
    out = {}
    for p, sig, qcd, ods in (('jedi', dict(W=2, Z=3, t=4), [0, 1], 'hls'), ('small', dict(W=2, Z=3, t=4), [0, 1], 'hls'),
                             ('jetclass', dict(W=7, Z=6, t=8), [0], 'jc')):
        f = OUT / f'h13_{p}_perjet.npz'
        if not f.exists(): continue
        Z = np.load(f); y = Z['y']; qm = np.isin(y, qcd)
        models = sorted({k.split('__')[0] for k in Z.files if k.endswith('__id') and not k.startswith(('mom', 'ax'))})
        for mn in models:
            r = {}
            for c in EFF_CONDS:
                c1 = Z[f'{mn}__{c}'].astype(np.float32).argmax(1)
                r[c] = {k: dict(eff=float((c1[y == v] == v).mean()), mistag_qcd=float((c1[qm] == v).mean())) for k, v in sig.items()}
            out[f'{p}|{mn}'] = r
        if p == 'small': continue
        m0 = {c: Z[f'mom__{c}__mall'] for c in EFF_CONDS}
        out[f'{p}|W(m) ungroomed'] = {c: dict(W=dict(eff=float(((m0[c] > 65) & (m0[c] < 95))[y == sig['W']].mean()),
                                                       mistag_qcd=float(((m0[c] > 65) & (m0[c] < 95))[qm].mean()))) for c in EFF_CONDS}
        fo = OUT / f'h13_obs_{ods}_perjet.npz'
        if fo.exists():
            O = np.load(fo); yo = O['y']; qo = np.isin(yo, qcd); r1, r2, r3 = {}, {}, {}
            for c in EFF_CONDS:
                ms, mm, t32 = O[f'{c}__mSD'], O[f'{c}__m'], O[f'{c}__tau32']
                w = (ms > 65) & (ms < 95); tt = (ms > 105) & (ms < 210) & (t32 < 0.65); tm = (mm > 105) & (mm < 210) & (t32 < 0.65)
                r1[c] = dict(W=dict(eff=float(w[yo == sig['W']].mean()), mistag_qcd=float(w[qo].mean())))
                r2[c] = dict(t=dict(eff=float(tt[yo == sig['t']].mean()), mistag_qcd=float(tt[qo].mean())))
                r3[c] = dict(t=dict(eff=float(tm[yo == sig['t']].mean()), mistag_qcd=float(tm[qo].mean())))
            out[f'{p}|W(mSD) [rows ::5]'] = r1; out[f'{p}|top(mSD, tau32) [rows ::5]'] = r2; out[f'{p}|top(m, tau32) [rows ::5]'] = r3
    for k, r in out.items():
        log(f'eff {k}: ' + ' '.join(f'{c} ' + '/'.join(f'{v["eff"]:.2f}' for v in r[c].values()) for c in EFF_CONDS))
    return out


def part_summary():
    P = {p: json.loads((OUT / f'h13_{p}.json').read_text()) for p in ('jedi', 'small', 'jetclass', 'top', 'obs_hls', 'obs_jc', 'obs_top')
         if (OUT / f'h13_{p}.json').exists()}
    tests = []
    for p in ('jedi', 'small', 'jetclass', 'top'):
        if p not in P: continue
        for mn, m in P[p]['models'].items():
            for cn, st in m['results'].items():
                if 'vs_sk' in st: tests.append((p, mn, cn, st['vs_sk']['p_mcnemar'], st['vs_sk']['edit'], st['vs_sk']['control']))
    hp = holm([t[3] for t in tests])
    S = dict(n_tests=len(tests), tests={f'{p}|{mn}|{cn}': dict(p=pv, holm=h, edit=e, control=c) for (p, mn, cn, pv, e, c), h in zip(tests, hp)})
    S['accuracy'] = {f'{p}|{mn}': (m.get('acc') if 'acc' in m else m.get('auc')) for p in ('jedi', 'small', 'jetclass', 'top') if p in P for mn, m in P[p]['models'].items()}
    S['local_slopes'] = local_slopes()
    S['efficiency'] = eff_table()
    (OUT / 'h13_summary.json').write_text(json.dumps(S, indent=1)); log(f'summary: {len(tests)} tests')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    E = S['efficiency']; xs = [0] + list(RHOS); conds = ['id'] + [f'pu_{x}' for x in RHOS]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    def line(a, vals, lab, col, ls='-', mk='o'):
        a.plot(xs, [100 * v for v in vals], ls=ls, marker=mk, color=col, label=lab, ms=4, lw=1.4)
    for a, p, title in ((ax[0], 'jedi', 'hls4ml: true W jets'), (ax[1], 'jetclass', 'JetClass: true W→qq jets')):
        nets = [('jedi|jedi', 'JEDI-linear', '#1f77b4'), ('small|dsets', 'Deep Sets', '#6baed6'), ('small|efn', 'EFN (IRC safe)', '#555555')] if p == 'jedi' else \
               [('jetclass|ParT_full', 'ParT full', '#d62728'), ('jetclass|ParT_kin', 'ParT kin', '#ff9896'), ('jetclass|PN_full', 'ParticleNet full', '#2ca02c'),
                ('jetclass|PN_kin', 'ParticleNet kin', '#98df8a')]
        for k, lab, col in nets:
            if k in E: line(a, [E[k][c]['W']['eff'] for c in conds], lab + ' (call = W)', col)
        if f'{p}|W(m) ungroomed' in E: line(a, [E[f'{p}|W(m) ungroomed'][c]['W']['eff'] for c in conds], '65 < m < 95 GeV (ungroomed)', 'k', '--', 's')
        if f'{p}|W(mSD) [rows ::5]' in E: line(a, [E[f'{p}|W(mSD) [rows ::5]'][c]['W']['eff'] for c in conds], '65 < m$_{SD}$ < 95 GeV (soft drop)', 'k', ':', 'D')
        a.set_title(title, fontsize=10); a.legend(fontsize=7, frameon=False, loc='lower left')
    if 'top' in P and 'obs_top' in P:
        cols = dict(ParT='#9467bd', PN='#2ca02c', LNet='#8c564b', PFN='#e377c2', EFN='#555555')
        for k, col in cols.items():
            r = P['top']['models'][k]['results']; line(ax[2], [0.5] + [r[c]['tag_rate_top'][1] for c in conds[1:]], k + (' (IRC safe)' if k == 'EFN' else ''), col)
        for tn, lab, ls, mk in (('top_m_tau32', '105 < m < 210 GeV, τ32 < 0.65', '--', 's'), ('top_mSD_tau32', '105 < m$_{SD}$ < 210 GeV, τ32 < 0.65', ':', 'D')):
            t = P['obs_top']['taggers'][tn]; line(ax[2], [t['pass_before']['t']] + [t['conditions'][c]['pass_after']['t'] for c in conds[1:]], lab, 'k', ls, mk)
        ax[2].set_title('Top Landscape: true top jets (networks at the 50 % working point)', fontsize=10); ax[2].legend(fontsize=7, frameon=False, loc='lower left')
    for a in ax:
        a.set_xticks(xs); a.set_xlabel('added soft-particle density ρ [GeV per unit η–φ area]'); a.set_ylim(0, 100)
    ax[0].set_ylabel('signal jets still tagged (%)')
    fig.suptitle('Diffuse soft particles lower every network\'s signal efficiency faster than they lower the soft-drop mass tagger\'s', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'h13_panel.png', dpi=130); log(f'wrote {OUT / "h13_panel.png"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['jedi', 'small', 'jetclass', 'top', 'obs', 'summary'])
    ap.add_argument('--ds', default='hls', choices=['hls', 'jc', 'top']); ap.add_argument('--smoke', type=int, default=0); a = ap.parse_args()
    log(f'H13 part {a.part} ds {a.ds} smoke {a.smoke}')
    if a.part == 'obs': part_obs(a.ds, a.smoke)
    elif a.part == 'summary': part_summary()
    else: dict(jedi=part_jedi, small=part_small, jetclass=part_jetclass, top=part_top)[a.part](a.smoke)
    log('done')
