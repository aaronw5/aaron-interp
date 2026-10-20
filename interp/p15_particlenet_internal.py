"""P15 — ParticleNet internals (official JetClass weights): EdgeConv-layer mass direction, dynamic kNN graphs, channel atlas,
literal channel circuits, and LRP / EFP / Lund-plane baselines on the same jets (claims C5, C7 analogue; review weaknesses 6, 7).

Models: ParticleNet_full (17 per-particle features) and ParticleNet_kin (7), released weights, interp/pn_net.py (verbatim weaver-core
0.4.17 ParticleNet + official settings: 3 EdgeConv blocks k = 16, channels 64 / 128 / 256, no fusion, mean pooling over real particles,
FC 256 (ReLU) -> 10). Forward re-implemented in fw() (checked against weaver's forward at start). Sites:
  ec1, ec2, ec3 = outputs of EdgeConv blocks 1-3 (per particle, 64 / 128 / 256 channels);
  the pooled 256-d feature = jet mean of ec3 (so a jet-level interchange at ec3 IS an interchange of the pooled feature);
  fc = the 256 hidden units after the FC layer's ReLU (input of the last Linear).
Sample: JetClass val_5M, file <sample>_120.root of each of the 10 classes, 2,000 random rows per class (numpy default_rng(15), never
first-k) = 20,000 jets; rebuilt with part_core.build (eta = jet_eta + sign(jet_eta) deta). Fit/test halves: one random permutation of
the 20,000 (seed 16); the W/Z set = the 4,000 Wqq + Zqq jets, split by the same halves. P1 rows (test files, rows 3,000-4,000 per
class) re-run for the unedited-accuracy check against P4 (84.57 % full, 70.35 % kin).
Edits on the W/Z jets (pT, particle masses fixed unless stated; 4-vectors rebuilt): phi d (every real particle dphi + d, d = 0.03,
0.06, 0.09; invariant mass m fixed, m_A up), eta0.06 (deta + 0.06), angk (deta, dphi x k, k = exp(mean dlog m_A of phi0.06): real
mass change matched to phi0.06's m_A change), ang1.1, pt1.25 (pT and particle mass x 1.25: m and m_A x 1.25), rot0.3 (rotation of
(deta, dphi) about the axis by 0.3 rad: every particle's dR and all pairwise distances kept). m_A = sqrt(sum pT * sum pT dR^2) (axis-
moment mass about the stored axis, E6 / P8v 'mom').
(1) DAS per site: one unit vector q (Adam, STEPS steps, batch BS, lr 0.02) trained on the W/Z fit half so that the jet-level
    interchange h_p <- h_p + ((mean h'_cf - mean h) . q) q (fc: h <- h + ((h'_cf - h) . q) q) reproduces the counterfactual's 10
    logits (MSE); counterfactuals phi 0.03 / 0.06 / 0.09 of the same jet. Evaluation on the W/Z test half for phi0.06 (trained),
    eta0.06 (held-out), angk (real mass), pt1.25: logit recovery IIA = 1 - sum|L_int - L_cf|^2 / sum|L - L_cf|^2; d = logit_W -
    logit_Z, share of the edit's d change recovered = sum (d_int - d) / sum (d_cf - d). Controls: NRAND random unit q per site; DAS
    trained the same way on a weight-permuted ParticleNet (every conv / linear weight tensor's entries shuffled, seed 15).
    Slope ratios (direction's jet projection s = mean_p h_p . q): (a) edit regression over phi 0.03/0.06/0.09, eta0.06, angk,
    ang1.1, pt1.25, rot0.3: ds = b_A dlog m_A + b_inv dlog m (no intercept), ratio b_A / b_inv and share b_A / (b_A + b_inv)
    (m_A-only variable -> share 1, invariant mass -> 0); (b) P8v ratios (ds / dlog m_A)[e] / (ds / dlog m_A)[phi0.06] for angk,
    pt1.25 (m_A variable -> 1); (c) natural jets: s = b_A log m_A + b_inv log m + class term; (d) natural-donor substitution (P8v set
    B): recipient W/Z test jets get a donor's (same class, |dlog m_A| >= 0.10) jet-level value of s; slope of dd on ds relative to
    the self-interchange slope from the jet's own phi 0.06 counterfactual; joint regression dd = c_A dlog m_A + c_inv dlog m.
(2) Dynamic graph (W/Z test half; phi0.06, ang1.1, rot0.3): fraction of kNN edges changed per layer (real particles); variants
    with layer l's graph frozen to the unedited graph (l = 1, 2, 3, all) and "graphs only" (unedited features, edited-run graphs);
    share of the full edit's dd and dW->Z removed by each freeze.
(3) Channel atlas (all 20,000 jets): per channel of the pooled 256-d feature, Spearman rho with 20 physics observables (top 3,
    negative pole), class composition of the top / bottom 1 %, R^2 from class alone, conditional R^2 (within-class linear fit on the
    20 observables, fit half -> test half); same metrics for 256 random orthonormal directions (random-rotation basis control),
    Mann-Whitney p; Fisher-z p per (channel, observable), Holm over all of them.
(4) Literal channel circuits at levels ec2 (128 channels, per particle; donor features slot-aligned with wrap as weaver's padding;
    the ec3 kNN graph is rebuilt from the mixed features), pool (256 pooled channels = jet means of the ec3 output channels) and fc
    (256 hidden units): channels ranked by |mean s IG attribution| to d (IG from the donor value, alphas 0.1..0.9; s = +1 W, -1 Z;
    donors B = pT-matched opposite class; W/Z fit half). Top K kept, all other channels at that level resampled from the donor
    (W/Z test half: B donors; 10-class read: test-half jets [::2] with uniform random donors U). W/Z accuracy (10-class call ==
    label on W/Z jets), binary W/Z accuracy (sign d), faithfulness F = sum s (d_c - d_corr) / sum s (d - d_corr), 10-class accuracy
    and confusion; NRSET random live-channel sets of equal K (permutation p), McNemar vs the full model, Holm over levels x K x models.
(5) Baselines (W/Z test half): LRP-epsilon (eps = EPS at every ReLU pre-activation and at d; relevance at the normalised input
    features, summed per particle) vs dR and pT; relevance-weighted dR, <dR>_R = sum |R| dR / sum |R|, correlated with log m_A and
    log m; linear (ridge) probes of d from 35 EFPs (all connected multigraphs with <= 4 edges, beta = 1, on the NTOP hardest
    particles, plus their products up to degree 4), from primary Lund-plane bins (C/A, 6 x 8 bins in (ln 1/Delta, ln kT) + count),
    and from (log m, log m_A, log pT) (known-observable reference). Prediction of the phi0.06 change dd per jet: DAS interchange,
    LRP-eps linearisation sum g_LRP . dx, plain gradient . dx, probe changes; per-jet Pearson r with the observed dd (bootstrap),
    predicted vs observed W->Z flips; probes calibrated with the angk real-mass edit.
Statistics: NB bootstrap resamples over jets (percentile 95 % intervals), two-sided bootstrap p (floor 1/(NB+1); normal p from the
bootstrap s.e. also given), Holm where many tests are scanned.

  ./sync.sh && ./part_job.sh c-p15-smoke 'python -m interp.p15_particlenet_internal --smoke'
  ./sync.sh && ./part_job.sh c-p15-run 'python -m interp.p15_particlenet_internal'
      -> $AI_RESULTS/particlenet/p15_particlenet_internal.json, p15_das_layers.png, p15_circuits.png
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMOKE = '--smoke' in sys.argv
SEED = 15
VAL = Path(os.environ.get('P15_VAL', '/j-jepa-vol/JetClass/Pythia/val_5M'))
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'particlenet'
FN = OUT / ('p15_particlenet_internal_smoke.json' if SMOKE else 'p15_particlenet_internal.json')
PER = 60 if SMOKE else 2000
MAXROW = 25000 if SMOKE else None
NB = 50 if SMOKE else 1000
STEPS = 10 if SMOKE else 300
BS = 64 if SMOKE else 128
NRAND = 2 if SMOKE else 5
NRSET = 3 if SMOKE else 100
KS = (8, 16, 32, 64)
EPS = 0.1
NTOP = 60
SITES = ['ec1', 'ec2', 'ec3', 'fc']
DIM = dict(ec1=64, ec2=128, ec3=256, fc=256)
DAS_CF = ['phi0.03', 'phi0.06', 'phi0.09']
EVAL_CF = ['phi0.06', 'eta0.06', 'angk', 'pt1.25']
EDITS = ['phi0.03', 'phi0.06', 'phi0.09', 'eta0.06', 'angk', 'ang1.1', 'pt1.25', 'rot0.3']
GRAPH_EDITS = ['phi0.06', 'ang1.1', 'rot0.3']
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
LEVELS = ['ec2', 'pool', 'fc']
OBS = ['log_m', 'log_mA', 'log_pt', 'log_S2_over_S0', 'girth', 'n_part', 'n_charged', 'tau1', 'tau21', 'tau32', 'D2', 'pTD', 'z_lead',
       'dR12', 'f_charged', 'f_photon', 'f_nhad', 'n_lepton', 'axis_offset', 'log1p_d0sig_max']
iZ, iW = 6, 7
T0 = time.time()
DEV = 'cpu'


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def _imports():
    global torch, pc, pn_net, p4, FEATURES, inputs, lund_mod
    import torch
    from interp import part_core as pc, pn_net, p4_particlenet as p4, _lund as lund_mod
    p4._imports()
    from jetdistill.part.network import FEATURES, inputs
    assert (pc.CL.index('Zqq'), pc.CL.index('Wqq')) == (iZ, iW)


# ------------------------------------------------------------------------------------------------------------ statistics
def bootW(n, seed):
    rng = np.random.default_rng(seed); W = np.zeros((NB, n), np.float32)
    for b in range(NB): W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def p_two(bs, h=0.0):
    bs = np.asarray(bs, float); bs = bs[np.isfinite(bs)]
    if not len(bs): return dict(p=None, p_norm=None)
    p = 2 * min((bs <= h).mean(), (bs >= h).mean()); p = max(min(p, 1.0), 1.0 / (len(bs) + 1))
    from scipy.stats import norm
    se = bs.std(); pn = float(2 * norm.sf(abs(bs.mean() - h) / se)) if se > 0 else None
    return dict(p=float(p), p_norm=pn)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def mcnemar(ok0, ok1):
    from scipy.stats import binomtest
    b = int((ok0 & ~ok1).sum()); c = int((~ok0 & ok1).sum())
    return dict(right_to_wrong=b, wrong_to_right=c, p=(float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0))


def ratio_sums(W, a, b):
    """sum a / sum b with bootstrap"""
    v = float(a.sum() / b.sum()) if abs(b.sum()) > 0 else float('nan'); bs = (W @ a) / (W @ b)
    return dict(est=v, ci=ci(bs), **{'p_eq0': p_two(bs, 0.0)['p'], 'p_eq1': p_two(bs, 1.0)['p']}), bs


def mean_b(W, a):
    v = float(a.mean()); bs = (W @ a) / W.sum(1); return dict(est=v, ci=ci(bs)), bs


def wcorr(W, x, y):
    """Pearson r with bootstrap weights"""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.std() < 1e-7 or y.std() < 1e-7: return dict(est=None, ci=[None, None], note='constant input'), None
    r = float(np.corrcoef(x, y)[0, 1]); Wd = W.astype(np.float64); n = Wd.sum(1)
    mx = (Wd @ x) / n; my = (Wd @ y) / n
    cxy = (Wd @ (x * y)) / n - mx * my; cxx = (Wd @ (x * x)) / n - mx ** 2; cyy = (Wd @ (y * y)) / n - my ** 2
    bs = cxy / np.sqrt(np.maximum(cxx * cyy, 1e-300)); return dict(est=r, ci=ci(bs)), bs


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.floating): return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def ridge_fit(X, y, fit, val):
    """standardised ridge; alpha on val from a grid; returns predict(X)"""
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-12; Z = (X - mu) / sd; ym = y[fit].mean(); best = None
    for a in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000):
        A = Z[fit].T @ Z[fit] + a * len(fit) * 1e-3 * np.eye(Z.shape[1]); w = np.linalg.solve(A, Z[fit].T @ (y[fit] - ym))
        e = ((Z[val] @ w + ym - y[val]) ** 2).mean()
        if best is None or e < best[0]: best = (e, a)
    tr = np.concatenate([fit, val]); ym = y[tr].mean(); a = best[1]
    A = Z[tr].T @ Z[tr] + a * len(tr) * 1e-3 * np.eye(Z.shape[1]); w = np.linalg.solve(A, Z[tr].T @ (y[tr] - ym))
    return (lambda X2: ((X2 - mu) / sd) @ w + ym), a


def r2(p, y):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ------------------------------------------------------------------------------------------------------------ data
def load_class(c, rng):
    import uproot
    path = VAL / f'{pc.PRE[c]}_120.root'
    N = uproot.open(path)['tree'].num_entries
    if MAXROW: N = min(N, MAXROW)
    rows = np.sort(rng.choice(N, PER, replace=False)); parts = []
    for s in range(0, N, 25000):
        r = rows[(rows >= s) & (rows < s + 25000)]
        if len(r): parts.append(pc.sub(pc.read_root(path, s, min(s + 25000, N)), r - s))
    J = pc.cat(parts); assert (J['label'].astype(int) == pc.CL.index(c)).all(), c
    return J, rows, N


def basic_obs(J):
    pt, de, dp, _ = pc.kin(J); pt = pt.astype(np.float64); r2_ = de.astype(np.float64) ** 2 + dp.astype(np.float64) ** 2
    S0 = pt.sum(1); S2 = (pt * r2_).sum(1); m = np.maximum(pc.jmass(J).astype(np.float64), 1.0)
    return dict(log_m=np.log(m), log_mA=0.5 * np.log(np.maximum(S0 * S2, 1e-12)), log_pt=np.log(np.maximum(J['jet_pt'].astype(np.float64), 1.0)))


def efps(J, B=2000):
    """35 EFPs (beta = 1, theta = dR in (deta, dphi)), connected multigraphs with <= 4 edges and their products, NTOP hardest particles,
    z = pT / sum pT (all particles). Returns (N, 35) float64 and the names."""
    pt, de, dp, _ = pc.kin(J); S0 = pt.astype(np.float64).sum(1); out = []
    for i in range(0, len(pt), B):
        s = slice(i, i + B)
        u = torch.from_numpy(pt[s, :NTOP].astype(np.float64) / S0[s, None]).to(DEV)
        e = torch.from_numpy(de[s, :NTOP].astype(np.float64)).to(DEV); f = torch.from_numpy(dp[s, :NTOP].astype(np.float64)).to(DEV)
        M1 = torch.sqrt((e[:, :, None] - e[:, None]) ** 2 + (f[:, :, None] - f[:, None]) ** 2); M2, M3, M4 = M1 ** 2, M1 ** 3, M1 ** 4
        mv = lambda M, x: (M @ x[..., None])[..., 0]; dot = lambda a, b: (a * b).sum(-1)
        tr = lambda A: torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
        Mu1, Mu2, Mu3, Mu4 = mv(M1, u), mv(M2, u), mv(M3, u), mv(M4, u); v = u * Mu1; v2 = u * Mu2
        A = u[:, :, None] * M1; Bm = u[:, :, None] * M2; A2 = A @ A; A3 = A2 @ A; M1v = mv(M1, v)
        C = dict(e1=dot(u, Mu1),
                 e1x2=dot(u, Mu2), p2=dot(u, Mu1 ** 2),
                 e1x3=dot(u, Mu3), p2_21=dot(u, Mu2 * Mu1), p3=dot(v, M1v), star3=dot(u, Mu1 ** 3), tri=tr(A3),
                 e1x4=dot(u, Mu4), p2_31=dot(u, Mu3 * Mu1), p2_22=dot(u, Mu2 ** 2), p3_211=dot(v2, M1v), p3_121=dot(v, mv(M2, v)),
                 star3_211=dot(u, Mu2 * Mu1 ** 2), tri_dbl=tr(Bm @ A @ A), p4=dot(u, M1v ** 2), star4=dot(u, Mu1 ** 4),
                 chair=dot(u, Mu1 ** 2 * M1v), sq=tr(A2 @ A2), tri_tail=dot(torch.diagonal(A3, dim1=-2, dim2=-1), Mu1))
        deg = dict(e1=1, e1x2=2, p2=2, e1x3=3, p2_21=3, p3=3, star3=3, tri=3)
        d2 = ['e1x2', 'p2']; d3 = ['e1x3', 'p2_21', 'p3', 'star3', 'tri']
        P = dict(C)
        P['e1^2'] = C['e1'] ** 2; P['e1^3'] = C['e1'] ** 3; P['e1^4'] = C['e1'] ** 4
        for a in d2: P[f'e1*{a}'] = C['e1'] * C[a]; P[f'e1^2*{a}'] = C['e1'] ** 2 * C[a]
        for a in d3: P[f'e1*{a}'] = C['e1'] * C[a]
        for j, a in enumerate(d2):
            for b in d2[j:]: P[f'{a}*{b}'] = C[a] * C[b]
        names = list(P); out.append(torch.stack([P[k] for k in names], 1).cpu().numpy())
    return np.concatenate(out), names


def lund_feats(J):
    pt, de, dp, _ = pc.kin(J)
    x = np.stack([pt[:, :NTOP], de[:, :NTOP], dp[:, :NTOP]], -1).astype(np.float64)
    em = lund_mod.lund(x); F = np.zeros((len(pt), 6 * 8 + 1))
    for j, E in enumerate(em):
        for e in E:
            ix = int(np.clip(np.floor(e['lx']), 0, 5)); iy = int(np.clip(np.floor((e['lkt'] + 4) / 1.25), 0, 7)); F[j, ix * 8 + iy] += 1
        F[j, -1] = len(E)
    return F


def atlas_obs(J, E):
    """the 20 physics observables (OBS) per jet; E = efps(J) output (e1, tri for D2)"""
    pt, de, dp, _ = pc.kin(J); pt = pt.astype(np.float64); de = de.astype(np.float64); dp = dp.astype(np.float64); real = pt > 0
    S0 = pt.sum(1); r = np.hypot(de, dp); S2 = (pt * r * r).sum(1); m = np.maximum(pc.jmass(J).astype(np.float64), 1.0)
    ch = (J['charge'] != 0) & real
    o = dict(log_m=np.log(m), log_mA=0.5 * np.log(np.maximum(S0 * S2, 1e-12)), log_pt=np.log(np.maximum(J['jet_pt'].astype(np.float64), 1)),
             log_S2_over_S0=np.log(np.maximum(S2 / S0, 1e-12)), girth=(pt * r).sum(1) / S0, n_part=real.sum(1).astype(float),
             n_charged=ch.sum(1).astype(float))
    # N-subjettiness (beta = 1, R0 = 0.8): axes = the N hardest particles, one pT-weighted k-means update (one-pass approximation)
    p, e_, f_ = pt[:, :NTOP], de[:, :NTOP], dp[:, :NTOP]; tau = {}
    for Nax in (1, 2, 3):
        ax = np.stack([e_[:, :Nax], f_[:, :Nax]], -1)
        for it in range(2):
            d = np.hypot(e_[:, :, None] - ax[:, None, :, 0], f_[:, :, None] - ax[:, None, :, 1]); a = d.argmin(-1)
            if it == 1: break
            for k in range(Nax):
                w = p * (a == k); sw = w.sum(1)
                ok = sw > 0; ax[ok, k, 0] = (w * e_).sum(1)[ok] / sw[ok]; ax[ok, k, 1] = (w * f_).sum(1)[ok] / sw[ok]
        tau[Nax] = (p * d.min(-1)).sum(1) / (S0 * 0.8)
    names = E[1]; X = E[0]; e2 = 0.5 * X[:, names.index('e1')]; e3 = X[:, names.index('tri')] / 6.0
    o.update(tau1=tau[1], tau21=tau[2] / np.maximum(tau[1], 1e-12), tau32=tau[3] / np.maximum(tau[2], 1e-12), D2=e3 / np.maximum(e2, 1e-12) ** 3,
             pTD=np.sqrt((pt ** 2).sum(1)) / S0, z_lead=pt.max(1) / S0, dR12=np.hypot(de[:, 0] - de[:, 1], dp[:, 0] - dp[:, 1]) * real[:, 1],
             f_charged=(pt * ch).sum(1) / S0, f_photon=(pt * (J['isPhoton'] > 0)).sum(1) / S0, f_nhad=(pt * (J['isNeutralHadron'] > 0)).sum(1) / S0,
             n_lepton=(((J['isElectron'] > 0) | (J['isMuon'] > 0)) & real).sum(1).astype(float),
             axis_offset=np.hypot((pt * de).sum(1), (pt * dp).sum(1)) / S0,
             log1p_d0sig_max=np.log1p(np.where(ch, np.abs(J['d0val']) / np.maximum(J['d0err'], 1e-6), 0).max(1)))
    return np.stack([o[k] for k in OBS], 1)


# ------------------------------------------------------------------------------------------------------------ network
def fw(model, p, f, m, hooks=None, kfix=None, rec=(), start=None, sfts=None):
    """ParticleNet forward (use_fusion False) with hooks at ec1..ec3 (fts, mask) -> fts and fc (h, mask) -> h; kfix: per-layer kNN
    indices (None entries recompute); start='ec2' with sfts = ec2 features runs only block 3 and the head."""
    knn, ggf = pn_net.knn, pn_net.get_graph_feature_v1
    mask = m; cnt = torch.clamp(mask.sum(-1), min=1.0); shift = (mask == 0) * 1e9; out = dict(knn=[None] * 3)
    if start is None: fts = model.bn_fts(f * mask) * mask; l0 = 0
    else: fts = sfts * mask; l0 = int(start[2:])
    for li in range(l0, 3):
        conv = model.edge_convs[li]
        pts = (p * mask if li == 0 else fts) + shift
        idx = knn(pts, conv.k) if (kfix is None or kfix[li] is None) else kfix[li]
        out['knn'][li] = idx
        x = ggf(fts, conv.k, idx)
        for c2, b2, act in zip(conv.convs, conv.bns, conv.acts): x = act(b2(c2(x)))
        x = x.mean(-1)
        sc = conv.sc_bn(conv.sc(fts)) if conv.sc is not None else fts
        fts = conv.sc_act(sc + x) * mask
        s = f'ec{li + 1}'
        if hooks and s in hooks: fts = hooks[s](fts, mask) * mask
        if s in rec: out[s] = fts
        out[s + '_mean'] = fts.sum(-1) / cnt
    pooled = fts.sum(-1) / cnt; out['pool'] = pooled
    h = model.fc[0](pooled)
    if hooks and 'fc' in hooks: h = hooks['fc'](h, mask)
    out['fc'] = h; out['L'] = model.fc[1](h)
    return out


def head(model, pooled):
    return model.fc[1](model.fc[0](pooled))


def to_dev(X):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(DEV) for k, v in X.items()}


def fset(model, Xt, idx, B=500, hooks_fn=None, kfix_fn=None, knn=False, grad=False):
    """batched forward over rows idx of the device tensors Xt; returns L, pool, fc, ec*_mean (numpy) and knn (list of torch)"""
    R = dict(L=[], pool=[], fc=[], ec1_mean=[], ec2_mean=[], ec3_mean=[]); K = [[], [], []]
    with torch.no_grad():
        for i in range(0, len(idx), B):
            b = idx[i:i + B]; bt = torch.as_tensor(b, device=DEV)
            p, f, m = Xt['p'][bt], Xt['x'][bt], Xt['m'][bt]
            o = fw(model, p, f, m, hooks=hooks_fn(b, m) if hooks_fn else None, kfix=kfix_fn(b) if kfix_fn else None)
            for k in R: R[k].append(o[k].double().cpu().numpy())
            if knn:
                for li in range(3): K[li].append(o['knn'][li])
    R = {k: np.concatenate(v) for k, v in R.items()}; R['ec3_mean'] = R['pool']
    if knn: R['knn'] = [torch.cat(k) for k in K]
    return R


def site_val(R, site):
    return R['fc'] if site == 'fc' else R[f'{site}_mean']


def jm_hooks(site, q, tgt):
    """write the jet-level value tgt . q of the 1-dim variable q at site"""
    if site == 'fc': return {'fc': lambda h, m: h + ((tgt - h) @ q)[:, None] * q[None]}
    def hk(fts, m):
        mb = fts.sum(-1) / torch.clamp(m.sum(-1), min=1.0)
        return fts + ((tgt - mb) @ q)[:, None, None] * q[None, :, None]
    return {site: hk}


def permuted_model(path, nfeat):
    model = pn_net.load_particlenet(path, nfeat, DEV); g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for _, prm in model.named_parameters():
            if prm.ndim >= 2:
                flat = prm.detach().flatten().cpu(); prm.copy_(flat[torch.randperm(flat.numel(), generator=g)].view_as(prm).to(prm.device))
    return model


def freeze(model):
    for q in model.parameters(): q.requires_grad_(False)
    model.eval(); return model


# ------------------------------------------------------------------------------------------------------------ (1) DAS
def das_train(model, Xt, site, MW, LW, fit, rng):
    C = DIM[site]; g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
    v = torch.randn(C, generator=g, dtype=torch.float32).to(DEV).requires_grad_(True); opt = torch.optim.Adam([v], lr=0.02)
    cfs = np.array(DAS_CF); losses = []
    for step in range(STEPS):
        b = rng.choice(fit, BS); e = rng.integers(0, len(cfs), BS)
        tgt = np.stack([MW[cfs[k]][site][j] for k, j in zip(e, b)]).astype(np.float32)
        Lc = np.stack([LW[cfs[k]][j] for k, j in zip(e, b)]).astype(np.float32)
        q = v / v.norm(); bt = torch.as_tensor(b, device=DEV)
        o = fw(model, Xt['p'][bt], Xt['x'][bt], Xt['m'][bt], hooks=jm_hooks(site, q, torch.from_numpy(tgt).to(DEV)))
        loss = ((o['L'] - torch.from_numpy(Lc).to(DEV)) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    return (v / v.norm()).detach(), losses


def das_eval(model, Xt, site, q, MW, LW, test, cfs, tgt_fn=None):
    """per-jet arrays for each counterfactual: num |L_int - L_cf|^2, den |L - L_cf|^2, dd_int, dd_cf, calls"""
    out = {}
    for e in cfs:
        tg = (lambda b, m: jm_hooks(site, q, torch.from_numpy(MW[e][site][b].astype(np.float32)).to(DEV))) if tgt_fn is None else tgt_fn
        Li = fset(model, Xt, test, hooks_fn=tg)['L']; Lb = LW['base'][test]; Lc = LW[e][test]
        d = lambda L: L[:, iW] - L[:, iZ]
        out[e] = dict(num=((Li - Lc) ** 2).sum(1), den=((Lb - Lc) ** 2).sum(1), dd_int=d(Li) - d(Lb), dd_cf=d(Lc) - d(Lb),
                      c_int=Li.argmax(1), c_cf=Lc.argmax(1), c_b=Lb.argmax(1))
        if tgt_fn is not None: break
    return out


def das_stats(ev, W):
    o = {}; B = {}
    for e, v in ev.items():
        a = 1 - v['num'].sum() / v['den'].sum(); bs = 1 - (W @ v['num']) / (W @ v['den'])
        r, rb = ratio_sums(W, v['dd_int'], v['dd_cf'])
        ch = v['c_cf'] != v['c_b']
        o[e] = dict(iia=dict(est=float(a), ci=ci(bs)), share_dd=r, call_iia=float((v['c_int'][ch] == v['c_cf'][ch]).mean()) if ch.any() else None,
                    n_changed=int(ch.sum()), mean_dd_cf=float(v['dd_cf'].mean()), mean_dd_int=float(v['dd_int'].mean()))
        B[e] = dict(iia=bs, share=rb)
    return o, B


def slope_ratios(q, site, MW, test, OBW, W, y):
    """(a) edit regression, (b) P8v ratios, (c) natural regression for the 1-dim variable q at site"""
    qn = q.double().cpu().numpy(); s = {e: MW[e][site][test] @ qn for e in ['base'] + EDITS}
    dA = {e: OBW[e]['log_mA'][test] - OBW['base']['log_mA'][test] for e in EDITS}
    dm = {e: OBW[e]['log_m'][test] - OBW['base']['log_m'][test] for e in EDITS}
    ds = {e: s[e] - s['base'] for e in EDITS}
    n = len(test); P = np.zeros((n, 4)); Q = np.zeros((n, 2))
    for e in EDITS:
        x1, x2, yv = dA[e], dm[e], ds[e]; P += np.stack([x1 * x1, x1 * x2, x1 * x2, x2 * x2], 1); Q += np.stack([x1 * yv, x2 * yv], 1)
    def solve(Pm, Qm):
        return np.linalg.solve(np.asarray(Pm).reshape(-1, 2, 2), np.asarray(Qm).reshape(-1, 2, 1))[..., 0]
    beta = solve(P.sum(0), Q.sum(0))[0]; bb = solve(W.astype(np.float64) @ P, W.astype(np.float64) @ Q)
    ratio = beta[0] / beta[1]; rb = bb[:, 0] / bb[:, 1]; share = beta[0] / (beta[0] + beta[1]); sb = bb[:, 0] / (bb[:, 0] + bb[:, 1])
    o = dict(edit_regression=dict(b_A=float(beta[0]), b_A_ci=ci(bb[:, 0]), b_inv=float(beta[1]), b_inv_ci=ci(bb[:, 1]),
                                  ratio_A_over_inv=float(ratio), ratio_ci=ci(rb), frac_boot_ratio_gt2=float((rb > 2).mean()), frac_boot_share_gt_2over3=float((sb > 2 / 3).mean()),
                                  share_A=float(share), share_A_ci=ci(sb), p_share_eq1=p_two(sb, 1.0)['p'], p_share_eq0=p_two(sb, 0.0)['p']),
             per_edit={e: dict(mean_ds=float(ds[e].mean()), mean_dlogmA=float(dA[e].mean()), mean_dlogm=float(dm[e].mean())) for e in EDITS})
    ref = ratio_sums(W, ds['phi0.06'], dA['phi0.06'])[1]
    for e in ('angk', 'pt1.25', 'eta0.06'):
        a = ratio_sums(W, ds[e], dA[e])[1]; v = (ds[e].sum() / dA[e].sum()) / (ds['phi0.06'].sum() / dA['phi0.06'].sum())
        o[f'P8v_ratio_{e}_over_phi'] = dict(est=float(v), ci=ci(a / ref), p_eq1=p_two(a / ref, 1.0)['p'])
    # (c) natural regression on the test jets: s = b_A log m_A + b_inv log m + c_W isW + c
    X = np.stack([OBW['base']['log_mA'][test], OBW['base']['log_m'][test], (y[test] == iW).astype(float), np.ones(n)], 1); sv = s['base']
    b = np.linalg.lstsq(X, sv, rcond=None)[0]; Wd = W.astype(np.float64)
    bbs = np.array([np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ sv) for w in Wd[:200]])
    o['natural'] = dict(b_A=float(b[0]), b_A_ci=ci(bbs[:, 0]), b_inv=float(b[1]), b_inv_ci=ci(bbs[:, 1]), ratio=float(b[0] / b[1]), ratio_ci=ci(bbs[:, 0] / bbs[:, 1]),
                        corr_s_logmA=float(np.corrcoef(sv, X[:, 0])[0, 1]), corr_s_logm=float(np.corrcoef(sv, X[:, 1])[0, 1]),
                        corr_logmA_logm=float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]), n_boot=200)
    return o


def substitution(model, Xt, site, q, qr, MW, LW, test, OBW, y, rng, self_ev, W):
    """natural donors (same class, |dlog m_A| >= 0.10) write their jet-level value of q into the recipient"""
    lA = OBW['base']['log_mA']; lm = OBW['base']['log_m']; don = np.full(len(test), -1)
    for j, i in enumerate(test):
        c = test[(y[test] == y[i]) & (np.abs(lA[test] - lA[i]) >= 0.10)]
        if len(c): don[j] = rng.choice(c)
    ok = don >= 0; rec = test[ok]; dn = don[ok]; Wk = W[:, ok]; out = dict(N=int(ok.sum()))
    d = lambda L: L[:, iW] - L[:, iZ]; Lb = LW['base'][rec]
    self_dd = self_ev['dd_int'][ok]; qn = q.double().cpu().numpy(); base_site = MW['base'][site]
    ds_self = (MW['phi0.06'][site][rec] - base_site[rec]) @ qn
    s_self, b_self = float((self_dd * ds_self).sum() / (ds_self ** 2).sum()), (Wk @ (self_dd * ds_self)) / (Wk @ (ds_self ** 2))
    out['self_phi'] = dict(slope_dd_per_ds=s_self, ci=ci(b_self))
    pos = {int(r): k for k, r in enumerate(rec)}
    for name, qq in (('das', q), ('rand0', qr)):
        tgt_all = base_site[dn].astype(np.float32)
        hf = lambda b, m, qq=qq, tgt_all=tgt_all: jm_hooks(site, qq, torch.from_numpy(tgt_all[[pos[int(x)] for x in b]]).to(DEV))
        Ls = fset(model, Xt, rec, hooks_fn=hf)['L']; dd = d(Ls) - d(Lb); qv = qq.double().cpu().numpy()
        ds = (base_site[dn] - base_site[rec]) @ qv; dA = lA[dn] - lA[rec]; dm = lm[dn] - lm[rec]
        sl = float((dd * ds).sum() / (ds ** 2).sum()); bsl = (Wk @ (dd * ds)) / (Wk @ (ds ** 2))
        X = np.stack([dA, dm], 1); c = np.linalg.lstsq(X, dd, rcond=None)[0]
        cb = np.array([np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ dd) for w in Wk[:200].astype(np.float64)])
        e = dict(slope_dd_per_ds=sl, ci=ci(bsl), corr_ds_dlogmA=float(np.corrcoef(ds, dA)[0, 1]), corr_ds_dlogm=float(np.corrcoef(ds, dm)[0, 1]),
                 joint_c_A=float(c[0]), joint_c_A_ci=ci(cb[:, 0]), joint_c_inv=float(c[1]), joint_c_inv_ci=ci(cb[:, 1]),
                 joint_ratio=float(c[0] / c[1]), joint_ratio_ci=ci(cb[:, 0] / cb[:, 1]), mean_abs_dd=float(np.abs(dd).mean()))
        if name == 'das':
            rb = bsl / b_self; e['ratio_to_self'] = dict(est=float(sl / s_self), ci=ci(rb), p_eq1=p_two(rb, 1.0)['p'], p_eq0=p_two(rb, 0.0)['p'])
        out[name] = e
    out['donor'] = dict(mean_abs_dlogmA=float(np.abs(lA[dn] - lA[rec]).mean()), mean_abs_dlogm=float(np.abs(lm[dn] - lm[rec]).mean()),
                        corr_dlogmA_dlogm=float(np.corrcoef(lA[dn] - lA[rec], lm[dn] - lm[rec])[0, 1]))
    return out


# ------------------------------------------------------------------------------------------------------------ (2) graphs
def graph_block(model, Xt, XEt, test, y, LW0, W):
    K0 = fset(model, Xt, test, knn=True)['knn']; pos = torch.as_tensor(np.searchsorted(test, test), device=DEV)
    loc = {int(r): k for k, r in enumerate(test)}
    def kf(Ks, which):
        return lambda b: [Ks[li][torch.as_tensor([loc[int(x)] for x in b], device=DEV)] if which[li] else None for li in range(3)]
    mask_real = Xt['m'][torch.as_tensor(test, device=DEV)][:, 0] > 0
    def overlap(Ks):
        o = []
        for li in range(3):
            ov = (Ks[li][..., :, None] == K0[li][..., None, :]).any(-1).float().mean(-1)
            o.append(float((ov * mask_real).sum() / mask_real.sum()))
        return o
    d = lambda L: L[:, iW] - L[:, iZ]; Lb = LW0[test]; isW = y[test] == iW; isZ = y[test] == iZ; sgn = np.where(isW, 1.0, -1.0)
    Wt = W; out = dict(noise_overlap=overlap(fset(model, Xt, test, B=333, knn=True)['knn']))
    for e in GRAPH_EDITS:
        Rf = fset(model, XEt[e], test, knn=True); Ke = Rf['knn']
        V = {'full': Rf['L'], 'freeze_l1': fset(model, XEt[e], test, kfix_fn=kf(K0, (1, 0, 0)))['L'],
             'freeze_l2': fset(model, XEt[e], test, kfix_fn=kf(K0, (0, 1, 0)))['L'], 'freeze_l3': fset(model, XEt[e], test, kfix_fn=kf(K0, (0, 0, 1)))['L'],
             'freeze_all': fset(model, XEt[e], test, kfix_fn=kf(K0, (1, 1, 1)))['L'], 'graphs_only': fset(model, Xt, test, kfix_fn=kf(Ke, (1, 1, 1)))['L']}
        ov = overlap(Ke); res = dict(edges_changed_per_layer=[1 - x for x in ov], variants={})
        full_dd = d(V['full']) - d(Lb); full_wz = ((V['full'].argmax(1) == iZ).astype(float) - (Lb.argmax(1) == iZ)) * isW
        for k, L in V.items():
            dd = d(L) - d(Lb); wz = ((L.argmax(1) == iZ).astype(float) - (Lb.argmax(1) == iZ)) * isW
            sh, bsh = ratio_sums(Wt, dd, full_dd); shw, bshw = ratio_sums(Wt, wz, full_wz)
            res['variants'][k] = dict(mean_dd=float(dd.mean()), mean_dd_W=float(dd[isW].mean()), W_to_Z=float((L.argmax(1)[isW] == iZ).mean()),
                                      share_dd=sh, share_WtoZ=shw, removed_dd=dict(est=1 - sh['est'], ci=[1 - sh['ci'][1], 1 - sh['ci'][0]]),
                                      removed_WtoZ=dict(est=1 - shw['est'], ci=[1 - shw['ci'][1], 1 - shw['ci'][0]]),
                                      changed_vs_base=float((L.argmax(1) != Lb.argmax(1)).mean()))
        res['W_to_Z_base'] = float((Lb.argmax(1)[isW] == iZ).mean()); out[e] = res
    return out


# ------------------------------------------------------------------------------------------------------------ (3) atlas
def atlas(Y, X, y, fit, test, rng, extra=None):
    from scipy.stats import rankdata, norm, mannwhitneyu
    def metrics(Yc):
        live = Yc.std(0) > 1e-6
        RY = rankdata(Yc, axis=0); RX = rankdata(X, axis=0)
        RY = (RY - RY.mean(0)) / (RY.std(0) + 1e-12); RX = (RX - RX.mean(0)) / (RX.std(0) + 1e-12); rho = RY.T @ RX / len(X)
        # within-class R^2 from the 20 observables (class means from the fit half), and R^2 from class alone
        cm = lambda A: np.stack([A[fit][y[fit] == c].mean(0) for c in range(10)])
        Yw = Yc - cm(Yc)[y]; Xw = X - cm(X)[y]; mu, sd = Xw[fit].mean(0), Xw[fit].std(0) + 1e-12; Z = (Xw - mu) / sd
        A = Z[fit].T @ Z[fit] + 1e-3 * len(fit) * np.eye(Z.shape[1]); Bc = np.linalg.solve(A, Z[fit].T @ (Yw[fit] - Yw[fit].mean(0)))
        P = Z[test] @ Bc + Yw[fit].mean(0); vt = ((Yw[test] - Yw[test].mean(0)) ** 2).sum(0) + 1e-12
        cond = 1 - ((P - Yw[test]) ** 2).sum(0) / vt
        vy = ((Yc[test] - Yc[test].mean(0)) ** 2).sum(0) + 1e-12; rcls = 1 - ((Yc[test] - cm(Yc)[y[test]]) ** 2).sum(0) / vy
        return live, rho, cond, rcls
    Yc = Y - Y.mean(0); live, rho, cond, rcls = metrics(Yc)
    Q, _ = np.linalg.qr(rng.standard_normal((Y.shape[1], Y.shape[1]))); Yr = Yc @ Q; _, rho_r, cond_r, rcls_r = metrics(Yr)
    n = len(X); z = np.arctanh(np.clip(rho, -0.999999, 0.999999)) * math.sqrt(n - 3); p = 2 * norm.sf(np.abs(z))
    ph = np.array(holm(p[live].ravel())).reshape(p[live].shape) if live.any() else np.zeros((0, len(OBS)))
    chans = []; li = np.where(live)[0]; hrow = {c: ph[k] for k, c in enumerate(li)}
    for c in range(Y.shape[1]):
        if not live[c]: chans.append(dict(ch=c, live=False)); continue
        o = np.argsort(-np.abs(rho[c])); neg = int(np.argmin(rho[c]))
        top = np.argsort(-Y[:, c])[:max(1, n // 100)]; bot = np.argsort(Y[:, c])[:max(1, n // 100)]
        chans.append(dict(ch=c, live=True, sd=float(Y[:, c].std()), top=[(OBS[k], float(rho[c, k]), float(hrow[c][k])) for k in o[:3]],
                          neg_pole=(OBS[neg], float(rho[c, neg])), top1pct_classes=np.bincount(y[top], minlength=10).tolist(),
                          bottom1pct_classes=np.bincount(y[bot], minlength=10).tolist(), frac_zero=float((Y[:, c] <= 1e-9).mean()),
                          cond_r2=float(cond[c]), class_r2=float(rcls[c])))
    mx = np.abs(rho[live]).max(1); mxr = np.abs(rho_r).max(1)
    comp = {}
    for k, a, b in (('max_abs_rho', mx, mxr), ('top1_rho2', mx ** 2, mxr ** 2), ('cond_r2', cond[live], cond_r), ('class_r2', rcls[live], rcls_r)):
        rb = np.random.default_rng(SEED).integers
        da = np.array([np.median(a[rb(0, len(a), len(a))]) - np.median(b[rb(0, len(b), len(b))]) for _ in range(500)])
        comp[k] = dict(channels_median=float(np.median(a)), rotated_median=float(np.median(b)), diff=float(np.median(a) - np.median(b)), diff_ci=ci(da),
                       mannwhitney_p=float(mannwhitneyu(a, b).pvalue), frac_ge_0p5_channels=float((a >= 0.5).mean()), frac_ge_0p5_rotated=float((b >= 0.5).mean()))
    best = [OBS[int(np.argmax(np.abs(rho[c])))] for c in li]
    out = dict(n_channels=int(Y.shape[1]), n_live=int(live.sum()), n_tests=int(live.sum() * len(OBS)), n_sig_holm=int((ph < 0.05).sum()),
               top_observable_counts={k: best.count(k) for k in OBS if best.count(k)},
               rotated_top_observable_counts={k: [OBS[int(np.argmax(np.abs(rho_r[c])))] for c in range(Y.shape[1])].count(k) for k in OBS},
               comparison=comp, channels=chans)
    if extra:
        out['directions'] = {}
        for name, v in extra.items():
            from scipy.stats import spearmanr
            rr = np.array([spearmanr(v, X[:, k])[0] for k in range(len(OBS))]); o = np.argsort(-np.abs(rr))
            out['directions'][name] = dict(top=[(OBS[k], float(rr[k])) for k in o[:5]], rho_log_mA=float(rr[OBS.index('log_mA')]), rho_log_m=float(rr[OBS.index('log_m')]))
    return out


# ------------------------------------------------------------------------------------------------------------ (4) circuits
def donors_B(pt, y, rng):
    don = np.zeros(len(y), int)
    for a, b in ((iW, iZ), (iZ, iW)):
        ia, ib = np.where(y == a)[0], np.where(y == b)[0]; o = ib[np.argsort(pt[ib])]; ps = pt[o]
        for i in ia:
            j = np.searchsorted(ps, pt[i]); lo, hi = max(0, j - 5), min(len(o), j + 5)
            cand = o[lo:hi]; cand = cand[np.argsort(np.abs(pt[cand] - pt[i]))[:5]]; don[i] = rng.choice(cand)
    return don


def donors_U(n, rng):
    d = rng.integers(0, n - 1, n); return d + (d >= np.arange(n))


class LevelSet:
    """one jet set (device tensors Xs) with donors: the level's own and donor values, and logits with a kept-channel mask"""

    def __init__(self, model, Xs, don):
        self.model, self.Xs, self.don = model, Xs, torch.as_tensor(don, device=DEV); n = Xs['m'].shape[0]; self.n = n
        H2, P, F = [], [], []
        with torch.no_grad():
            for i in range(0, n, 500):
                s = slice(i, min(n, i + 500)); o = fw(model, Xs['p'][s], Xs['x'][s], Xs['m'][s], rec=('ec2',))
                H2.append(o['ec2'].cpu()); P.append(o['pool']); F.append(o['fc'])
        self.H2 = torch.cat(H2); self.P = torch.cat(P); self.F = torch.cat(F)
        self.cnt = Xs['m'][:, 0].sum(-1).long()

    def donor_ec2(self, sl):
        dn = self.don[sl]; Hd = self.H2[dn.cpu()].to(DEV); P = Hd.shape[-1]
        wrap = torch.arange(P, device=DEV)[None, :] % torch.clamp(self.cnt[dn], min=1)[:, None]
        return torch.gather(Hd, 2, wrap[:, None, :].expand(-1, Hd.shape[1], -1))

    def logits(self, level, keep):
        with torch.no_grad():
            if level == 'fc':
                k = keep[None].to(DEV); return self.model.fc[1](torch.where(k, self.F, self.F[self.don])).double().cpu().numpy()
            if level == 'pool':
                k = keep[None].to(DEV); return head(self.model, torch.where(k, self.P, self.P[self.don])).double().cpu().numpy()
            out = []; k = keep[None, :, None].to(DEV)
            for i in range(0, self.n, 500):
                sl = slice(i, min(self.n, i + 500)); h = self.H2[sl].to(DEV); hd = self.donor_ec2(sl)
                o = fw(self.model, None, None, self.Xs['m'][sl], start='ec2', sfts=torch.where(k, h, hd)); out.append(o['L'].double().cpu().numpy())
            return np.concatenate(out)

    def ig(self, level, s):
        """IG attribution per channel of D = d (per jet), from the donor value; returns mean_j s_j attr_jc and completeness"""
        C = dict(fc=256, pool=256, ec2=128)[level]; att = torch.zeros(C, dtype=torch.float64, device=DEV); tot = 0.0; dref = 0.0
        st = torch.as_tensor(s, device=DEV, dtype=torch.float32)
        for i in range(0, self.n, 250):
            sl = slice(i, min(self.n, i + 250)); ss = st[sl]
            if level == 'ec2': h = self.H2[sl].to(DEV); hd = self.donor_ec2(sl); m = self.Xs['m'][sl]
            else: base = self.F if level == 'fc' else self.P; h = base[sl]; hd = base[self.don[sl]]
            g = torch.zeros_like(h)
            for a in ALPHAS:
                x = (hd + a * (h - hd)).detach().requires_grad_(True)
                if level == 'ec2': L = fw(self.model, None, None, m, start='ec2', sfts=x)['L']
                elif level == 'pool': L = head(self.model, x)
                else: L = self.model.fc[1](x)
                D = ((L[:, iW] - L[:, iZ]) * ss).sum(); g += torch.autograd.grad(D, x)[0] / len(ALPHAS)
            a_ = (h - hd) * g; att += (a_.sum(-1) if level == 'ec2' else a_).double().sum(0); tot += float(a_.sum())
            with torch.no_grad():
                if level == 'ec2':
                    L1 = fw(self.model, None, None, m, start='ec2', sfts=h)['L']; L0 = fw(self.model, None, None, m, start='ec2', sfts=hd)['L']
                elif level == 'pool': L1, L0 = head(self.model, h), head(self.model, hd)
                else: L1, L0 = self.model.fc[1](h), self.model.fc[1](hd)
                dref += float((((L1[:, iW] - L1[:, iZ]) - (L0[:, iW] - L0[:, iZ])) * ss).sum())
        return (att / self.n).cpu().numpy(), tot / dref if dref else None

    def live(self, level):
        H = dict(fc=self.F, pool=self.P)[level] if level != 'ec2' else None
        if H is not None: return (H.std(0) > 1e-6).cpu().numpy()
        mm = self.Xs['m'][:, 0].cpu()[:, None, :] > 0
        sd = torch.stack([self.H2[:, c][mm[:, 0]].std() for c in range(self.H2.shape[1])]); return (sd > 1e-6).numpy()


def circuit_block(model, Xt, idx_fit, idx_wz, idx_10, y, ptj, rng, Lfull):
    d = lambda L: L[:, iW] - L[:, iZ]; res = {}
    sub_t = lambda ids: {k: v[torch.as_tensor(ids, device=DEV)] for k, v in Xt.items()}
    S_fit = LevelSet(model, sub_t(idx_fit), donors_B(ptj[idx_fit], y[idx_fit], rng))
    S_wz = LevelSet(model, sub_t(idx_wz), donors_B(ptj[idx_wz], y[idx_wz], rng))
    S_10 = LevelSet(model, sub_t(idx_10), donors_U(len(idx_10), rng))
    yw, y10 = y[idx_wz], y[idx_10]; s_fit = np.where(y[idx_fit] == iW, 1.0, -1.0); s_wz = np.where(yw == iW, 1.0, -1.0)
    L_wz, L_10 = Lfull[idx_wz], Lfull[idx_10]; ok_full = L_wz.argmax(1) == yw
    Wb = bootW(len(idx_wz), SEED + 7)
    tests = []
    for level in LEVELS:
        C = dict(fc=256, pool=256, ec2=128)[level]; t1 = time.time()
        score, compl = S_fit.ig(level, s_fit); order = np.argsort(-np.abs(score)); live = S_fit.live(level); live_ix = np.where(live)[0]
        allk = torch.ones(C, dtype=torch.bool); nok = torch.zeros(C, dtype=torch.bool)
        Lc_wz = S_wz.logits(level, nok); Lc_10 = S_10.logits(level, nok)
        Lchk = S_wz.logits(level, allk); chk = float(np.abs(Lchk - L_wz).max())
        def evals(keep):
            Lw = S_wz.logits(level, keep); L1 = S_10.logits(level, keep)
            F = float((s_wz * (d(Lw) - d(Lc_wz))).sum() / (s_wz * (d(L_wz) - d(Lc_wz))).sum())
            return dict(wz_acc=float((Lw.argmax(1) == yw).mean()), wz_binary_acc=float((np.sign(d(Lw)) == s_wz).mean()), F=F,
                        acc10=float((L1.argmax(1) == y10).mean())), Lw, L1
        R = dict(attribution_completeness=compl, mask_all_kept_max_dlogit=chk, n_live=int(live.sum()),
                 top_channels=[(int(c), float(score[c])) for c in order[:64]],
                 full=dict(wz_acc=float(ok_full.mean()), wz_binary_acc=float((np.sign(d(L_wz)) == s_wz).mean()), acc10=float((L_10.argmax(1) == y10).mean())),
                 all_resampled=dict(wz_acc=float((Lc_wz.argmax(1) == yw).mean()), acc10=float((Lc_10.argmax(1) == y10).mean())), K={})
        for K in KS:
            if K >= C: continue
            keep = torch.zeros(C, dtype=torch.bool); keep[torch.as_tensor(order[:K])] = True
            e, Lw, L1 = evals(keep); okc = Lw.argmax(1) == yw
            e['wz_acc_ci'] = ci((Wb @ okc.astype(np.float32)) / Wb.sum(1)); e['mcnemar_vs_full'] = mcnemar(ok_full, okc)
            cm = np.zeros((10, 10), int); np.add.at(cm, (y10, L1.argmax(1)), 1); e['confusion10'] = cm.tolist()
            rnd = []
            for r in range(NRSET):
                kk = torch.zeros(C, dtype=torch.bool); kk[torch.as_tensor(rng.choice(live_ix, min(K, len(live_ix)), replace=False))] = True
                rnd.append(evals(kk)[0])
            for m_ in ('wz_acc', 'acc10', 'F', 'wz_binary_acc'):
                from scipy.stats import norm
                a = np.array([x[m_] for x in rnd]); sd_ = a.std(ddof=1) if len(a) > 1 else 0.0
                e[f'random_{m_}'] = dict(mean=float(a.mean()), sd=float(sd_), q05=float(np.percentile(a, 5)), q95=float(np.percentile(a, 95)),
                                         p_perm=float((1 + (a >= e[m_]).sum()) / (len(a) + 1)),
                                         p_norm=float(norm.sf((e[m_] - a.mean()) / sd_)) if sd_ > 0 else None)
            tests.append(((level, K), e['random_wz_acc']['p_perm'], e['random_wz_acc']['p_norm'] if e['random_wz_acc']['p_norm'] is not None else 1.0)); R['K'][str(K)] = e
            log(f'      {level} K={K}: W/Z acc {100 * e["wz_acc"]:.1f} % (random {100 * e["random_wz_acc"]["mean"]:.1f}, 95th pct {100 * e["random_wz_acc"]["q95"]:.1f}; '
                f'perm p {e["random_wz_acc"]["p_perm"]:.3f}) binary {100 * e["wz_binary_acc"]:.1f} % F {e["F"]:.3f} (random {e["random_F"]["mean"]:.3f}) | '
                f'10-class {100 * e["acc10"]:.1f} % (random {100 * e["random_acc10"]["mean"]:.1f}) | McNemar vs full {e["mcnemar_vs_full"]}')
        log(f'   level {level}: live {R["n_live"]}/{C}, IG completeness {compl}, all-kept check max |dL| {chk:.2e}; full W/Z {100 * R["full"]["wz_acc"]:.1f} %, '
            f'all resampled {100 * R["all_resampled"]["wz_acc"]:.1f} %, 10-class {100 * R["full"]["acc10"]:.1f} -> {100 * R["all_resampled"]["acc10"]:.1f} % ({time.time() - t1:.0f} s)')
        res[level] = R
    return res, tests


# ------------------------------------------------------------------------------------------------------------ (5) LRP
class EpsScale:
    @staticmethod
    def make(eps):
        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, z):
                ctx.save_for_backward(z); return z.clone()

            @staticmethod
            def backward(ctx, g):
                z, = ctx.saved_tensors; return g * z / (z + eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z)))
        return F.apply


def lrp_grad(model, p, f, m, eps):
    """gradient of d = l_W - l_Z wrt the normalised input features xb, with LRP-eps rescaling at every ReLU pre-activation and at d
    (eps = 0: plain gradient). Returns xb, g, d."""
    knn, ggf = pn_net.knn, pn_net.get_graph_feature_v1; E = EpsScale.make(eps) if eps > 0 else (lambda z: z)
    mask = m; cnt = torch.clamp(mask.sum(-1), min=1.0); shift = (mask == 0) * 1e9
    xb = (model.bn_fts(f * mask) * mask).detach().requires_grad_(True); fts = xb
    for li, conv in enumerate(model.edge_convs):
        pts = (p * mask if li == 0 else fts.detach()) + shift; idx = knn(pts, conv.k)
        x = ggf(fts, conv.k, idx)
        for c2, b2, act in zip(conv.convs, conv.bns, conv.acts): x = act(E(b2(c2(x))))
        x = x.mean(-1); sc = conv.sc_bn(conv.sc(fts)) if conv.sc is not None else fts
        fts = conv.sc_act(E(sc + x)) * mask
    pooled = fts.sum(-1) / cnt; h = model.fc[0][1](E(model.fc[0][0](pooled))); L = model.fc[1](h)
    d = E(L[:, iW] - L[:, iZ]); g = torch.autograd.grad(d.sum(), xb)[0]
    return xb.detach(), g, d.detach()


def lrp_block(model, Xt, XEt, test, Jt, y, W, B=125):
    """relevance per particle; linear predictions of dd for phi0.06 and angk (LRP-eps and plain gradient)"""
    n = len(test); Rp = np.zeros((n, 128)); pred = {k: np.zeros(n) for k in ('lrp_phi0.06', 'grad_phi0.06', 'lrp_angk', 'grad_angk')}; dd0 = np.zeros(n)
    for i in range(0, n, B):
        b = test[i:i + B]; bt = torch.as_tensor(b, device=DEV); s = slice(i, i + len(b)); p, f, m = Xt['p'][bt], Xt['x'][bt], Xt['m'][bt]
        for eps, tag in ((EPS, 'lrp'), (0.0, 'grad')):
            xb, g, dv = lrp_grad(model, p, f, m, eps)
            if tag == 'lrp': Rp[s] = (xb * g).sum(1).double().cpu().numpy(); dd0[s] = dv.double().cpu().numpy()
            for e in ('phi0.06', 'angk'):
                xe = (model.bn_fts(XEt[e]['x'][bt] * m) * m).detach()
                pred[f'{tag}_{e}'][s] = ((xe - xb) * g).sum((1, 2)).double().cpu().numpy()
    pt, de, dp, _ = pc.kin(Jt); dR = np.hypot(de, dp)[:, :128]; ptp = pt[:, :128]; real = ptp > 0
    cons = Rp.sum(1) / np.where(np.abs(dd0) > 1e-6, dd0, np.nan)
    bins_dR = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]); bins_pt = np.array([0, 1, 2, 5, 10, 20, 50, 100, 1e5])
    rel = {}
    for cls, nm in ((iW, 'W'), (iZ, 'Z')):
        sel = (y[test] == cls)[:, None] & real; tot = np.abs(Rp[sel]).sum()
        for key, v, bins in (('dR', dR, bins_dR), ('pt', ptp, bins_pt)):
            ix = np.digitize(v[sel], bins) - 1; Rv = Rp[sel]
            rel[f'{nm}_{key}'] = [dict(lo=float(bins[k]), hi=float(bins[k + 1]), n=int((ix == k).sum()), mean_R=float(Rv[ix == k].mean()) if (ix == k).any() else None,
                                       frac_absR=float(np.abs(Rv[ix == k]).sum() / tot)) for k in range(len(bins) - 1)]
    aR = np.abs(Rp) * real; wdR = (aR * dR).sum(1) / np.maximum(aR.sum(1), 1e-12)
    ob = basic_obs(Jt); ca, ba = wcorr(W, wdR, ob['log_mA']); cm, bm = wcorr(W, wdR, ob['log_m'])
    return dict(conservation_median=float(np.nanmedian(cons)), conservation_iqr=[float(np.nanpercentile(cons, 25)), float(np.nanpercentile(cons, 75))],
                relevance_bins=rel, wdR_corr_log_mA=ca, wdR_corr_log_m=cm, wdR_corr_diff=dict(est=ca['est'] - cm['est'], ci=ci(ba - bm), p=p_two(ba - bm)['p']),
                wdR_corr_log_mA_withinW=float(np.corrcoef(wdR[y[test] == iW], ob['log_mA'][y[test] == iW])[0, 1]),
                wdR_corr_log_m_withinW=float(np.corrcoef(wdR[y[test] == iW], ob['log_m'][y[test] == iW])[0, 1])), pred


# ------------------------------------------------------------------------------------------------------------ main
def main():
    global DEV
    _imports(); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; OUT.mkdir(parents=True, exist_ok=True)
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    res = dict(smoke=SMOKE, seed=SEED, per_class=PER, n_boot=NB, steps=STEPS, nrand=NRAND, nrset=NRSET, eps=EPS, ntop=NTOP, device=DEV,
               config=p4.check_config(), models={})
    log(f'P15 start, device {DEV}; data configs {res["config"]}')
    # ---------------- sample
    rng = np.random.default_rng(SEED); parts = []; rows = {}
    for c in pc.CL:
        J, r, N = load_class(c, rng); parts.append(J); rows[c] = dict(file=f'{pc.PRE[c]}_120.root', n_file=int(N), first_rows=r[:5].tolist())
        log(f'   loaded {c}: {len(r)} random rows of {N}')
    J0 = pc.cat(parts); del parts; y = J0['label'].astype(int); N = len(y)
    pt, de, dp, ms = pc.kin(J0); real = (pt > 0).astype(np.float32); Jb = pc.build(J0, pt, de, dp, ms)
    perm = np.random.default_rng(SEED + 1).permutation(N); is_test = np.zeros(N, bool); is_test[perm[N // 2:]] = True
    WZ = np.where((y == iW) | (y == iZ))[0]; fitW = np.where(~is_test[WZ])[0]; testW = np.where(is_test[WZ])[0]   # positions within WZ
    fit_all = np.where(~is_test)[0]; test_all = np.where(is_test)[0]
    res['sample'] = dict(N=N, rows=rows, n_wz=len(WZ), n_wz_fit=len(fitW), n_wz_test=len(testW))
    # edits on W/Z
    JW0 = pc.sub(J0, WZ); yW = y[WZ]; ptW, deW, dpW, msW = pt[WZ], de[WZ], dp[WZ], ms[WZ]; rW = real[WZ]
    JEW = {'base': pc.sub(Jb, WZ)}; OBW = {'base': basic_obs(JEW['base'])}
    for dd_ in (0.03, 0.06, 0.09): JEW[f'phi{dd_}'] = pc.build(JW0, ptW, deW, dpW + dd_ * rW, msW)
    JEW['eta0.06'] = pc.build(JW0, ptW, deW + 0.06 * rW, dpW, msW)
    OBW['phi0.06'] = basic_obs(JEW['phi0.06']); kmatch = float(np.exp((OBW['phi0.06']['log_mA'] - OBW['base']['log_mA']).mean()))
    JEW['angk'] = pc.build(JW0, ptW, deW * kmatch, dpW * kmatch, msW); JEW['ang1.1'] = pc.build(JW0, ptW, deW * 1.1, dpW * 1.1, msW)
    JEW['pt1.25'] = pc.build(JW0, ptW * 1.25, deW, dpW, msW * 1.25)
    a = 0.3; JEW['rot0.3'] = pc.build(JW0, ptW, (deW * np.cos(a) - dpW * np.sin(a)) * rW, (deW * np.sin(a) + dpW * np.cos(a)) * rW, msW)
    for e in EDITS: OBW[e] = basic_obs(JEW[e]); assert np.array_equal(JEW[e]['mask'], JEW['base']['mask']), e
    res['edits'] = dict(k_match=kmatch, checks={e: dict(dlogm=float((OBW[e]['log_m'] - OBW['base']['log_m']).mean()), dlogmA=float((OBW[e]['log_mA'] - OBW['base']['log_mA']).mean()),
                                                        dlogm_absmax=float(np.abs(OBW[e]['log_m'] - OBW['base']['log_m']).max())) for e in EDITS})
    log(f'sample: {N} jets ({len(WZ)} W/Z; fit {len(fitW)} / test {len(testW)}); k_match {kmatch:.4f}; edits ' +
        '; '.join(f'{e}: dlog m {v["dlogm"]:+.4f} (max {v["dlogm_absmax"]:.4f}), dlog m_A {v["dlogmA"]:+.4f}' for e, v in res['edits']['checks'].items()))
    # baseline observables / features (model independent)
    Eall = efps(Jb); OBS_ALL = atlas_obs(Jb, Eall); log(f'EFPs {Eall[0].shape} and 20 atlas observables computed')
    EFP = {'base': Eall[0][WZ]}
    for e in ('phi0.06', 'angk'): EFP[e] = efps(JEW[e])[0]
    res['efp_phi_invariance_max_rel'] = float(np.max(np.abs(EFP['phi0.06'] - EFP['base']) / (np.abs(EFP['base']) + 1e-12)))
    t1 = time.time(); LUND = {'base': lund_feats(JEW['base'])}; tW = WZ[testW]
    for e in ('phi0.06', 'angk'): LUND[e] = np.zeros_like(LUND['base']); LUND[e][testW] = lund_feats(pc.sub(JEW[e], testW))
    res['lund_phi_invariance_max'] = float(np.abs(LUND['phi0.06'][testW] - LUND['base'][testW]).max())
    log(f'Lund features ({time.time() - t1:.0f} s); EFP phi-invariance max rel {res["efp_phi_invariance_max_rel"]:.1e}, Lund phi-invariance max {res["lund_phi_invariance_max"]}')
    MOBS = {e: np.stack([OBW[e]['log_m'], OBW[e]['log_mA'], OBW[e]['log_pt']], 1) for e in ('base', 'phi0.06', 'angk')}
    Wt = bootW(len(testW), SEED + 3); P1 = None
    tests_das = []
    for n in ('full', 'kin'):
        t1 = time.time(); key = f'ParticleNet_{n}'; nf = len(FEATURES[n])
        model = freeze(pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', nf, DEV)); R = {}
        Xall = to_dev(p4.pn_inputs(Jb, n)); idx_all = np.arange(N)
        RA = fset(model, Xall, idx_all); Lall = RA['L']
        with torch.no_grad():
            Lnat = model(Xall['p'][:500].clone(), Xall['x'][:500].clone(), Xall['m'][:500]).double().cpu().numpy()
        R['fw_vs_weaver_max_dlogit'] = float(np.abs(Lnat - Lall[:500]).max())
        R['acc'] = float((Lall.argmax(1) == y).mean()); R['acc_test_half'] = float((Lall[test_all].argmax(1) == y[test_all]).mean())
        R['own_class'] = [float((Lall.argmax(1)[y == k] == k).mean()) for k in range(10)]
        if P1 is None: P1 = pc.sample(1000 if not SMOKE else 30, 3000); P1b = pc.build(P1, *pc.kin(P1))
        X1 = to_dev(p4.pn_inputs(P1b, n)); R['acc_P1_rows'] = float((fset(model, X1, np.arange(len(P1b['label'])))['L'].argmax(1) == P1b['label'].astype(int)).mean()); del X1
        log(f'{key}: accuracy {100 * R["acc"]:.2f} % on the 20,000 val jets (test half {100 * R["acc_test_half"]:.2f} %); P1 rows {100 * R["acc_P1_rows"]:.2f} % '
            f'(P4: {"84.57" if n == "full" else "70.35"} %); fw vs weaver max |dL| {R["fw_vs_weaver_max_dlogit"]:.2e}')
        # W/Z edit runs
        XW = {e: to_dev(p4.pn_inputs(JEW[e], n)) for e in ['base'] + EDITS}
        RW = {e: fset(model, XW[e], np.arange(len(WZ))) for e in ['base'] + EDITS}
        LW = {e: RW[e]['L'] for e in RW}; MW = {e: {s: site_val(RW[e], s) for s in SITES} for e in RW}
        dfun = lambda L: L[:, iW] - L[:, iZ]
        R['edit_response'] = {e: dict(mean_dd_W=float((dfun(LW[e]) - dfun(LW['base']))[yW == iW].mean()), mean_dd_Z=float((dfun(LW[e]) - dfun(LW['base']))[yW == iZ].mean()),
                                      W_to_Z=float((LW[e].argmax(1)[yW == iW] == iZ).mean())) for e in EDITS}
        R['W_to_Z_base'] = float((LW['base'].argmax(1)[yW == iW] == iZ).mean())
        log(f'   W->Z base {R["W_to_Z_base"]:.3f}; ' + '; '.join(f'{e}: W->Z {v["W_to_Z"]:.3f}, dd|W {v["mean_dd_W"]:+.3f}' for e, v in R['edit_response'].items()))
        # ---------------- (1) DAS
        rng_d = np.random.default_rng(SEED + 10); R['das'] = {}; QD = {}
        pmodel = freeze(permuted_model(PNDIR / f'ParticleNet_{n}.pt', nf))
        RWp = {e: fset(pmodel, XW[e], np.arange(len(WZ))) for e in ['base'] + DAS_CF}
        LWp = {e: RWp[e]['L'] for e in RWp}; MWp = {e: {s: site_val(RWp[e], s) for s in SITES} for e in RWp}
        R['permuted_acc_wz'] = float((LWp['base'].argmax(1) == yW).mean())
        for site in SITES:
            t2 = time.time(); q, losses = das_train(model, XW['base'], site, MW, LW, fitW, rng_d); QD[site] = q
            ev = das_eval(model, XW['base'], site, q, MW, LW, testW, EVAL_CF); st, Bst = das_stats(ev, Wt)
            rnd = []; rq = []
            for r in range(NRAND):
                qr = torch.randn(DIM[site], generator=torch.Generator().manual_seed(SEED * 100 + r)).to(DEV); qr = qr / qr.norm(); rq.append(qr)
                rnd.append(das_stats(das_eval(model, XW['base'], site, qr, MW, LW, testW, ['phi0.06']), Wt))
            rnd_share = np.array([x[0]['phi0.06']['share_dd']['est'] for x in rnd]); rnd_iia = np.array([x[0]['phi0.06']['iia']['est'] for x in rnd])
            rb = np.mean([x[1]['phi0.06']['share'] for x in rnd], 0); diff = Bst['phi0.06']['share'] - rb; pd_ = p_two(diff)
            # permuted-model DAS
            qp, _ = das_train(pmodel, XW['base'], site, MWp, LWp, fitW, rng_d)
            stp, _ = das_stats(das_eval(pmodel, XW['base'], site, qp, MWp, LWp, testW, ['phi0.06']), Wt)
            sr = slope_ratios(q, site, MW, testW, OBW, Wt, yW)
            sub = substitution(model, XW['base'], site, q, rq[0], MW, LW, testW, OBW, yW, np.random.default_rng(SEED + 20), ev['phi0.06'], Wt)
            R['das'][site] = dict(train_loss_first_last=[losses[0], losses[-1]], eval=st, random=dict(share_dd=rnd_share.tolist(), iia=rnd_iia.tolist(),
                                  share_mean=float(rnd_share.mean()), diff_vs_random=dict(est=float(st['phi0.06']['share_dd']['est'] - rnd_share.mean()), ci=ci(diff), **pd_)),
                                  permuted=stp['phi0.06'], slopes=sr, substitution=sub,
                                  per_jet_pred_phi=dict(corr_dd_int_dd_cf=wcorr(Wt, ev['phi0.06']['dd_int'], ev['phi0.06']['dd_cf'])[0]))
            R['das'][site]['_pred'] = ev['phi0.06']['dd_int']
            tests_das.append(((key, site), pd_['p'] if pd_['p'] > 1.5 / (NB + 1) or pd_['p_norm'] is None else pd_['p_norm']))
            e6 = st['phi0.06']; er = sr['edit_regression']
            log(f'   DAS {site}: phi0.06 share of dd {e6["share_dd"]["est"]:.3f} [{e6["share_dd"]["ci"][0]:.3f}, {e6["share_dd"]["ci"][1]:.3f}], IIA {e6["iia"]["est"]:.3f}, '
                f'call IIA {e6["call_iia"]} (n changed {e6["n_changed"]}) | held-out ' + ', '.join(f'{e} share {st[e]["share_dd"]["est"]:.3f} IIA {st[e]["iia"]["est"]:.3f}' for e in EVAL_CF[1:]) +
                f' | random share {rnd_share.round(3).tolist()} | permuted-model DAS share {stp["phi0.06"]["share_dd"]["est"]:.3f} (dd_cf {stp["phi0.06"]["mean_dd_cf"]:+.3f}) | '
                f'edit regression b_A {er["b_A"]:.3f} b_inv {er["b_inv"]:.3f} share_A {er["share_A"]:.3f} {np.round(er["share_A_ci"], 3).tolist()} ratio {er["ratio_A_over_inv"]:.2f} | '
                f'P8v pt/phi {sr["P8v_ratio_pt1.25_over_phi"]["est"]:.3f}, angk/phi {sr["P8v_ratio_angk_over_phi"]["est"]:.3f} | natural ratio {sr["natural"]["ratio"]:.2f} '
                f'(corr s~log mA {sr["natural"]["corr_s_logmA"]:.3f}, s~log m {sr["natural"]["corr_s_logm"]:.3f}) | substitution ratio to self {sub["das"]["ratio_to_self"]["est"]:.3f} '
                f'{np.round(sub["das"]["ratio_to_self"]["ci"], 3).tolist()}, joint c_A/c_inv {sub["das"]["joint_ratio"]:.2f} ({time.time() - t2:.0f} s)')
        del pmodel, RWp
        # ---------------- (2) graphs
        t2 = time.time(); R['graphs'] = graph_block(model, XW['base'], XW, testW, yW, LW['base'], Wt)
        for e in GRAPH_EDITS:
            g = R['graphs'][e]
            log(f'   graphs {e}: edges changed per layer {np.round(g["edges_changed_per_layer"], 4).tolist()} | ' + '; '.join(
                f'{k} removed dd {v["removed_dd"]["est"]:.3f} [{v["removed_dd"]["ci"][0]:.3f}, {v["removed_dd"]["ci"][1]:.3f}] W->Z {v["W_to_Z"]:.3f}' for k, v in g['variants'].items() if k != 'full'))
        log(f'   graph noise overlap {R["graphs"]["noise_overlap"]} ({time.time() - t2:.0f} s)')
        # ---------------- (3) atlas
        t2 = time.time()
        extra = {'DAS_ec3': RA['pool'] @ QD['ec3'].double().cpu().numpy(), 'DAS_fc': RA['fc'] @ QD['fc'].double().cpu().numpy()}
        R['atlas_pool'] = atlas(RA['pool'], OBS_ALL, y, fit_all, test_all, np.random.default_rng(SEED + 30), extra)
        R['atlas_fc'] = {k: v for k, v in atlas(RA['fc'], OBS_ALL, y, fit_all, test_all, np.random.default_rng(SEED + 31)).items() if k != 'channels'}
        A_ = R['atlas_pool']
        log(f'   atlas pooled: live {A_["n_live"]}/256, Holm-significant {A_["n_sig_holm"]}/{A_["n_tests"]}; top-observable counts {A_["top_observable_counts"]}; ' +
            '; '.join(f'{k}: channels {v["channels_median"]:.3f} vs rotated {v["rotated_median"]:.3f} (MW p {v["mannwhitney_p"]:.1e})' for k, v in A_['comparison'].items()) +
            f' | DAS directions {A_["directions"]} ({time.time() - t2:.0f} s)')
        # ---------------- (4) circuits
        t2 = time.time(); ptj = J0['jet_pt'].astype(np.float64)
        idx10 = test_all[::2]
        R['circuits'], tc = circuit_block(model, Xall, WZ[fitW], WZ[testW], idx10, y, ptj, np.random.default_rng(SEED + 40), Lall)
        R['_circ_tests'] = tc; log(f'   circuits done ({time.time() - t2:.0f} s)')
        # ---------------- (5) baselines
        t2 = time.time(); Jt = pc.sub(JEW['base'], testW)
        lr, pred = lrp_block(model, XW['base'], XW, testW, Jt, yW, Wt); R['lrp'] = lr
        dB = dfun(LW['base']); dd_obs = {e: (dfun(LW[e]) - dB)[testW] for e in ('phi0.06', 'angk')}
        fitp, valp = fitW[: int(0.75 * len(fitW))], fitW[int(0.75 * len(fitW)):]
        probes = {}
        for pname, FE in (('EFP_linear', EFP), ('Lund_primary', LUND), ('mass_mA_pt', MOBS)):
            f_, alpha = ridge_fit(FE['base'], dB, fitp, valp); pb = f_(FE['base'][testW])
            pr = dict(alpha=alpha, r2_test=r2(pb, dB[testW]), call_agreement=float((np.sign(pb) == np.sign(dB[testW])).mean()),
                      wz_acc_of_sign=float((np.sign(pb) == np.where(yW[testW] == iW, 1, -1)).mean()))
            for e in ('phi0.06', 'angk'): pred[f'{pname}_{e}'] = f_(FE[e][testW]) - pb
            probes[pname] = pr
        R['probes'] = probes
        cmp = {}
        cand = {'DAS_ec1': R['das']['ec1']['_pred'], 'DAS_ec2': R['das']['ec2']['_pred'], 'DAS_ec3': R['das']['ec3']['_pred'], 'DAS_fc': R['das']['fc']['_pred'],
                'LRP_eps': pred['lrp_phi0.06'], 'gradient': pred['grad_phi0.06'], 'EFP_linear': pred['EFP_linear_phi0.06'],
                'Lund_primary': pred['Lund_primary_phi0.06'], 'mass_mA_pt': pred['mass_mA_pt_phi0.06']}
        obs = dd_obs['phi0.06']; flip_obs = (np.sign(dB[testW]) != np.sign(dB[testW] + obs)) & (yW[testW] == iW)
        for k, v in cand.items():
            c_, _ = wcorr(Wt, v, obs)
            fl = (np.sign(dB[testW]) != np.sign(dB[testW] + v)) & (yW[testW] == iW)
            cmp[k] = dict(r=c_, mean_pred=float(v.mean()), mean_obs=float(obs.mean()), ratio_mean=float(v.mean() / obs.mean()),
                          flips_pred=int(fl.sum()), flips_obs=int(flip_obs.sum()), flips_both=int((fl & flip_obs).sum()))
            if f'{k}_angk' in pred or k in ('LRP_eps', 'gradient'):
                pk = pred['lrp_angk'] if k == 'LRP_eps' else pred['grad_angk'] if k == 'gradient' else pred[f'{k}_angk']
                gain = float(pk.mean() / dd_obs['angk'].mean()); cmp[k]['angk_gain'] = gain; cmp[k]['angk_r'] = wcorr(Wt, pk, dd_obs['angk'])[0]
                cmp[k]['phi_calibrated_ratio'] = float(v.mean() / (gain * obs.mean())) if abs(gain) > 1e-9 else None
        R['phi_prediction'] = cmp
        log(f'   LRP: conservation median {lr["conservation_median"]:.3f}; <dR>_R corr log m_A {lr["wdR_corr_log_mA"]["est"]:.3f} vs log m {lr["wdR_corr_log_m"]["est"]:.3f} | probes ' +
            '; '.join(f'{k}: R2 {v["r2_test"]:.3f}, call agreement {v["call_agreement"]:.3f}' for k, v in probes.items()) + ' | phi0.06 per-jet r: ' +
            '; '.join(f'{k} {v["r"]["est"] if v["r"]["est"] is None else round(v["r"]["est"], 3)} (mean pred {v["mean_pred"]:+.3f} vs obs {v["mean_obs"]:+.3f}; angk gain {v.get("angk_gain")})' for k, v in cmp.items()) +
            f' ({time.time() - t2:.0f} s)')
        for s in SITES: R['das'][s].pop('_pred')
        res['models'][key] = R
        json.dump(clean(res), open(FN, 'w'), indent=1); log(f'{key} done ({time.time() - t1:.0f} s); wrote {FN}')
        del model, Xall, XW
        if DEV == 'cuda': torch.cuda.empty_cache()
    # Holm
    adj = holm([p for _, p in tests_das])
    res['holm_das_vs_random'] = [dict(test='/'.join(k), p=p, p_holm=a) for (k, p), a in zip(tests_das, adj)]
    ct = [((m, lv, K), p, pn) for m, R in res['models'].items() for (lv, K), p, pn in R.pop('_circ_tests')]
    adj = holm([p for _, p, _ in ct]); adjn = holm([pn for _, _, pn in ct])
    res['holm_circuits_vs_random'] = [dict(test=f'{m}/{lv}/K{K}', p=p, p_holm=a, p_norm=pn, p_norm_holm=an) for ((m, lv, K), p, pn), a, an in zip(ct, adj, adjn)]
    log('Holm DAS vs random: ' + '; '.join(f'{d_["test"]} {d_["p"]:.1e} -> {d_["p_holm"]:.1e}' for d_ in res['holm_das_vs_random']))
    log('Holm circuits vs random: ' + '; '.join(f'{d_["test"]} perm {d_["p"]:.3f} -> {d_["p_holm"]:.3f}, normal {d_["p_norm"]:.1e} -> {d_["p_norm_holm"]:.1e}' for d_ in res['holm_circuits_vs_random']))
    json.dump(clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}')
    try: plot(res)
    except Exception as ex: log(f'plot failed: {ex!r}')
    log('P15 DONE')


# ------------------------------------------------------------------------------------------------------------ figures
def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = res or json.load(open(FN)); M = res['models']; INK, INK2 = '#222', '#666'
    labels = {'ec1': 'EdgeConv 1\n(64 ch.)', 'ec2': 'EdgeConv 2\n(128 ch.)', 'ec3': 'EdgeConv 3 =\npooled (256 ch.)', 'fc': 'FC hidden\n(256 units)'}
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140); w = 0.2
    series = [('ParticleNet full: learned direction', 'ParticleNet_full', 'das', '#c0392b'), ('ParticleNet kin: learned direction', 'ParticleNet_kin', 'das', '#e67e22'),
              ('full: random directions (mean)', 'ParticleNet_full', 'rand', '#999999'), ('full: same search on weight-shuffled network', 'ParticleNet_full', 'perm', '#2c7fb8')]
    for j, (lab, m, kind, col) in enumerate(series):
        if m not in M: continue
        xs, ys, lo, hi = [], [], [], []
        for i, s in enumerate(SITES):
            D = M[m]['das'][s]
            if kind == 'das': v = D['eval']['phi0.06']['share_dd']; e, c_ = v['est'], v['ci']
            elif kind == 'rand': e = D['random']['share_mean']; c_ = [min(D['random']['share_dd']), max(D['random']['share_dd'])]
            else: v = D['permuted']['share_dd']; e, c_ = v['est'], v['ci']
            xs.append(i + (j - 1.5) * w); ys.append(e); lo.append(max(0, e - c_[0])); hi.append(max(0, c_[1] - e))
        ax.bar(xs, ys, w * 0.92, color=col, label=lab, yerr=[lo, hi], capsize=2, ecolor=INK2)
        for x, v, h_ in zip(xs, ys, hi): ax.text(x, max(v, 0) + h_ + 0.03, f'{v:.2f}', ha='center', fontsize=7.5, color=INK)
    ax.axhline(1, color=INK2, lw=0.8, ls='--'); ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(SITES))); ax.set_xticklabels([labels[s] for s in SITES], fontsize=9.5)
    ax.set_ylabel('share of the $\\phi$ + 0.06 rad change in\nlogit(W) $-$ logit(Z) reproduced by one direction', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=8, frameon=False, ncol=2, loc='upper right'); ax.set_ylim(-0.15, 1.95)
    f6 = M.get('ParticleNet_full', {}).get('das', {})
    ttl = 'One direction in EdgeConv 1 or 2 reproduces the response to a mass-preserving $\\phi$ shift;\nthe same search finds one in a weight-shuffled ParticleNet'
    ax.set_title(ttl, fontsize=11, color=INK, loc='left')
    fig.text(0.01, 0.005, f'W/Z test half of {res["sample"]["N"]:,} JetClass val jets ({res["sample"]["n_wz_test"]:,} W + Z); bars: 95 % bootstrap intervals (random: min-max of {res["nrand"]} directions).',
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / 'p15_das_layers.png', facecolor='white'); plt.close(fig); log('wrote', OUT / 'p15_das_layers.png')
    # circuits: W/Z accuracy vs K, pooled level, both models
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=140, sharey=True)
    for ax, m in zip(axs, ('ParticleNet_full', 'ParticleNet_kin')):
        if m not in M: continue
        for lv, col in (('pool', '#c0392b'), ('fc', '#2c7fb8'), ('ec2', '#27ae60')):
            C = M[m]['circuits'][lv]; Ks = [int(k) for k in C['K']]
            ax.plot(Ks, [100 * C['K'][str(k)]['wz_acc'] for k in Ks], 'o-', color=col, label={'pool': 'pooled channels', 'fc': 'FC hidden units', 'ec2': 'EdgeConv-2 channels'}[lv] + ' (top-K)')
            ax.fill_between(Ks, [100 * C['K'][str(k)]['random_wz_acc']['q05'] for k in Ks], [100 * C['K'][str(k)]['random_wz_acc']['q95'] for k in Ks], color=col, alpha=0.12)
        ax.axhline(100 * M[m]['circuits']['pool']['full']['wz_acc'], color=INK, lw=1, ls='--', label='full model')
        ax.axhline(100 * M[m]['circuits']['pool']['all_resampled']['wz_acc'], color=INK2, lw=1, ls=':', label='all channels resampled')
        ax.set_xscale('log', base=2); ax.set_xticks(list(KS)); ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel('channels kept (all others from a $p_T$-matched jet of the other class)', fontsize=9); ax.set_title(m.replace('_', ' '), fontsize=10.5, loc='left', color=INK)
        ax.spines[['top', 'right']].set_visible(False)
    axs[0].set_ylabel('W and Z jets called correctly [%]', fontsize=10); axs[0].legend(fontsize=7.5, frameon=False, loc='lower right')
    fig.suptitle('Literal channel circuits: W/Z accuracy when only K channels are kept (bands: 5-95 % of 100 random live-channel sets)', fontsize=10.5, x=0.01, ha='left', color=INK)
    fig.tight_layout(); fig.savefig(OUT / 'p15_circuits.png', facecolor='white'); plt.close(fig); log('wrote', OUT / 'p15_circuits.png')


if __name__ == '__main__':
    if '--plot' in sys.argv: plot()
    else: main()
