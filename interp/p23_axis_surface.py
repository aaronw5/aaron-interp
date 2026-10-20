"""P23 — shape of the axis response (claims C3, D4): ParT's (and ParticleNet's) response to moving the jet axis, measured on a
2-D grid of rigid translations of the constituents, split into the part odd under c -> -c (direction-dependent) and the part
even under c -> -c.

Question. A rigid translation of all constituents by c = (c_eta, c_phi) about the stored jet axis leaves the invariant mass
unchanged (exactly for c_phi; for c_eta at fixed pT up to the particle masses) but changes the axis moment
M2 = sum pT dR^2 about the stored axis: M2(c) = M2 + 2 c.M1 + |c|^2 M0 (M0 = sum pT, M1 = sum pT (deta, dphi)).
The axis-moment-mass reading (thesis) predicts a response proportional to P(c) = 1/2 log(M2(c) / M2) with one gain for its odd
part (c.M1 / M2 to first order) and its even part (|c|^2 M0 / 2 M2). A centroid/direction-dependent reading (as JEDI-linear in
E23v: odd gain 1.3-1.8, even 0.6-0.7 of a matched real change) predicts a larger odd gain. An input-convention reading (deta and
dphi are separate inputs) predicts different eta and phi gains.

Models: ParT_full, ParT_kin (released), ParT_full seeds s1, s2 retrained in P10 (2M-jet subset; models/part_seeds/full_s<k>/
net_best_epoch_state.pt), ParticleNet_full / ParticleNet_kin (released), two untrained ParT_full (weaver initialisation, torch
seeds 1 and 2).
Sample: JetClass test_20M WToQQ_105, ZToQQ_105, ZJetsToNuNu_105 (QCD), PER random rows each (numpy default_rng(2305), never
first-k); jets with < 2 particles dropped. Unedited accuracy reported on this sample (10-class argmax) and on the P1 sample
(rows 3,000-4,000 of each class file _100, N = 10,000).
Build: part_core.build (eta = jet_eta + sign(jet_eta) deta, phi = jet_phi + dphi); every edit is applied to the stored
(deta, dphi) of the real particles and the inputs (per-particle features and four-vectors) are rebuilt; 'base' = unedited
particles rebuilt.
Edits:
  grid    c = (c_eta, c_phi), each in {-0.08, -0.06, ..., 0.08} rad (9 x 9, 80 non-zero points; 40 mirror pairs +c / -c).
  rd0-3   per-jet random direction, |c| = 0.06 (held-out check of the fitted surface; equal-|c| direction spread).
  rot90/180/270  (deta, dphi) rotated about the stored axis (M0, M2 and all pair dR unchanged; M1 rotated).
  dil_p+, dil_p-, dil_e+, dil_e-  matched real change: per-jet dilation of the angles by k_j = sqrt(M2(c)/M2), the same
          1/2 dlog M2 as the translation c = (0, +0.06), (0, -0.06), (+0.06, 0), (-0.06, 0) (E23 / P11 'match' method).
Read-outs: d = per-jet change of logit_Z - logit_W (W and Z jets; also QCD jets) and of logit_QCD - (logit_W + logit_Z)/2 on
QCD jets.
Analysis per model and jet subset:
  mirror split: odd(c) = (d(c) - d(-c))/2, even(c) = (d(c) + d(-c))/2 over the 40 pairs; odd variance share
     = sum odd^2 / sum (odd^2 + even^2).
  predictors (exact): Po(c) = (P(c) - P(-c))/2, Pe(c) = (P(c) + P(-c))/2, P(c) = 1/2 log(M2(c)/M2); fixed-direction odd terms
     c_eta, c_phi (jet-independent). Reduced fits: odd = b_o Po + a_eta c_eta + a_phi c_phi; even = b_e Pe.
     Calibration kappa = slope through 0 of d(dil) on log k_j over the 4 matched dilations; gains g_o = b_o / kappa,
     g_e = b_e / kappa (1 = same response as a real change of equal 1/2 dlog M2).
  eta/phi split (first order): odd = b_oe c_eta M1e/M2 + b_op c_phi M1p/M2 + a_eta c_eta + a_phi c_phi;
     even = b_ee c_eta^2 M0/(2 M2) + b_ep c_phi^2 M0/(2 M2) + b_x c_eta c_phi M0/M2.
  per-jet surface: per-jet least-squares gradient (odd part, linear in c) and quadratic form (even part); cos between the
     per-jet gradient and the jet's centroid M1; per-jet anisotropy (H_ee - H_pp)/(H_ee + H_pp).
  held-out: R^2 of the fitted reduced surface (and of the even-only surface) on rd0-3.
  ratios R(c) = mean d(translation c) / mean d(matched dilation) for the four reference c.
  phi + 0.06 decomposition: mean d at c = (0, 0.06) = even + centroid-odd (b_o Po) + fixed-direction odd (a_phi 0.06) + rest.
  calls: W -> Z (10-class argmax) on true W, before -> after, exact McNemar on W correctness.
Statistics: 95 % bootstrap over jets (NB resamples, regression coefficients and kappa resampled together); H0 odd part = 0:
  sign-flip permutation of each jet's odd part (NPERM); H0 one gain g_o = g_e, H0 b_ee = b_ep and a_eta = a_phi: bootstrap
  two-sided p; Holm over the 8 models per test family.
Locate (ParT_full, ParT_kin, untrained ParT_full init1): per-block 1-d DAS on the jet-mean residual (sites emb, b1..b8; 'jm'
  interchange of P8: h_p <- h_p + ((mean h_cf - mean h) . q) q on real particles); counterfactuals = per-jet random direction
  translation +c and -c, |c| = 0.06, per-particle route only (four-vectors, hence pair features, from the unedited jet, as P8).
  q_odd trained so that (I(+c) - I(-c)) reproduces (C(+c) - C(-c)) (10 logits, MSE); q_even so that I(+c) + I(-c) reproduces
  C(+c) + C(-c). Held-out recovery = 1 - sum ||I_part - C_part||^2 / sum ||C_part||^2 for the odd and even parts, against 3
  random q at the same site; bootstrap CI; Holm over sites.

  ./sync.sh && ./part_job.sh c-p23-smoke 'python -m interp.p23_axis_surface --smoke'
  ./sync.sh && ./part_job.sh c-p23-run 'python -m interp.p23_axis_surface'
  -> $AI_RESULTS/part/p23_axis_surface[_smoke].json, p23_axis_surface[_smoke].npz (per-jet d), p23_axis_surface.png
"""
import argparse, contextlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--plot', action='store_true')
ap.add_argument('--no-das', action='store_true')
ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED = 2305
PER = 60 if SMOKE else 3000
NB = 200 if SMOKE else 1000
NPERM = 200 if SMOKE else 2000
STEPS = 10 if SMOKE else 150
BS = 64 if SMOKE else 128
N_DAS_TR = 100 if SMOKE else 1500
N_DAS_EV = 80 if SMOKE else 1000
NRAND = 3
CH = 256
C = 128
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
FILES = dict(Wqq='WToQQ_105.root', Zqq='ZToQQ_105.root', QCD='ZJetsToNuNu_105.root')
SEEDS_DIR = '/j-jepa-vol/aaron-interp/models/part_seeds'
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = '_smoke' if SMOKE else ''
FN = OUTD / f'p23_axis_surface{TAG}.json'
GRID = np.round(np.arange(-4, 5) * 0.02, 4)
REF = {'dil_p+': (0.0, 0.06), 'dil_p-': (0.0, -0.06), 'dil_e+': (0.06, 0.0), 'dil_e-': (-0.06, 0.0)}
SITES = ['emb'] + [f'b{l}' for l in range(1, 9)]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ statistics
def boot_W(n, nb, rng):
    W = np.zeros((nb, n), np.float64)
    for b in range(nb): W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def p_two(bs, v0=0.0):
    bs = np.asarray(bs, float); bs = bs[np.isfinite(bs)]; nb = len(bs)
    if nb == 0: return None
    return float(max(1.0 / nb, min(1.0, 2 * min((bs <= v0).mean(), (bs >= v0).mean()))))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def mcnemar(ok0, ok1):
    from scipy.stats import binomtest
    b = int((ok0 & ~ok1).sum()); c = int((~ok0 & ok1).sum())
    return dict(right_to_wrong=b, wrong_to_right=c, p=(float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0))


def strip(o):
    if isinstance(o, dict): return {str(k): strip(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def st(v, bs, v0=0.0):
    return dict(v=float(v), ci=ci(bs), p0=p_two(bs, v0))


# ------------------------------------------------------------------------------------------------ data and edits
def load_rows(path, n, rng, chunk=25000):
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries
    if SMOKE: N = min(N, 25000)
    rows = np.sort(rng.choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, s, min(s + chunk, N)), r - s))
    return cat(parts), rows


def K(ce, cp):
    return f'tr_{float(ce) + 0.0:+.2f}_{float(cp) + 0.0:+.2f}'


def edit_names():
    g = [K(ce, cp) for ce in GRID for cp in GRID if not (ce == 0 and cp == 0)]
    return ['base'] + g + [f'rd{k}' for k in range(4)] + ['rot90', 'rot180', 'rot270'] + list(REF)


def parse_tr(name):
    _, a, b = name.split('_'); return float(a), float(b)


def moments(pt, de, dp):
    pt, de, dp = (np.asarray(a, np.float64) for a in (pt, de, dp))
    return dict(M0=pt.sum(1), M1e=(pt * de).sum(1), M1p=(pt * dp).sum(1), M2=(pt * (de ** 2 + dp ** 2)).sum(1))


def M2_after(Mo, ce, cp):
    return Mo['M2'] + 2 * (ce * Mo['M1e'] + cp * Mo['M1p']) + (ce ** 2 + cp ** 2) * Mo['M0']


def make_edit(S, name):
    from interp.part_core import build
    pt, de, dp, ms, r = S['pt'], S['de'], S['dp'], S['ms'], S['real']
    if name == 'base': return build(S['J'], pt, de, dp, ms)
    if name.startswith('tr_'):
        ce, cp = parse_tr(name); return build(S['J'], pt, de + ce * r, dp + cp * r, ms)
    if name.startswith('rd'):
        c = S['rd'][:, int(name[2:])]; return build(S['J'], pt, de + c[:, :1] * r, dp + c[:, 1:] * r, ms)
    if name.startswith('rot'):
        th = math.radians(float(name[3:])); co, si = math.cos(th), math.sin(th)
        return build(S['J'], pt, (co * de - si * dp) * r, (si * de + co * dp) * r, ms)
    if name in REF:
        k = S['k'][name][:, None]; return build(S['J'], pt, de * k, dp * k, ms)
    raise ValueError(name)


def prep_chunk(J, rd):
    from interp.part_core import kin
    pt, de, dp, ms = kin(J); S = dict(J=J, pt=pt, de=de, dp=dp, ms=ms, real=(pt > 0).astype(np.float32), rd=rd)
    Mo = moments(pt, de, dp); S['k'] = {nm: np.sqrt(np.maximum(M2_after(Mo, *c), 1e-30) / Mo['M2']).astype(np.float32) for nm, c in REF.items()}
    return S


# ------------------------------------------------------------------------------------------------ models
def load_models():
    import torch
    from interp.part_core import ParTNetwork
    from interp import pn_net
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import CFG, FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    strip_mod = lambda sd: {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
    M = {}

    def part(name, feat, path):
        net = ParTNetwork(feat)
        if path is not None and path.startswith('init'):
            torch.manual_seed(int(path[4:])); fresh = ParticleTransformer(input_dim=len(FEATURES[feat]), **CFG)
            net.model.load_state_dict(fresh.state_dict(), strict=True)
        elif path is not None:
            net.model.load_state_dict(strip_mod(torch.load(path, map_location='cpu')), strict=True)
        net.model.to(net.device).eval()
        for p in net.model.parameters(): p.requires_grad_(False)
        M[name] = dict(kind='part', feat=feat, model=net.model, path=path, dev=net.device)

    part('ParT_full', 'full', None); part('ParT_kin', 'kin', None)
    for s in (1, 2):
        p = f'{SEEDS_DIR}/full_s{s}/net_best_epoch_state.pt'
        if os.path.exists(p): part(f'ParT_full_s{s}', 'full', p)
        else: log(f'missing {p}')
    PND = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    for feat in ('full', 'kin'):
        m = pn_net.load_particlenet(PND / f'ParticleNet_{feat}.pt', len(FEATURES[feat]), dev)
        for p in m.parameters(): p.requires_grad_(False)
        M[f'PN_{feat}'] = dict(kind='pn', feat=feat, model=m, path=str(PND / f'ParticleNet_{feat}.pt'), dev=dev)
    part('ParT_full_untr1', 'full', 'init1'); part('ParT_full_untr2', 'full', 'init2')
    return M, torch


def run_model(mm, X, torch):
    x, v, m = X[mm['feat']]; dev = mm['dev']; T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    with torch.no_grad():
        if mm['kind'] == 'part':
            L = mm['model'](T(x), T(v), T(m))
        else:
            from jetdistill.part.network import FEATURES
            P = x.shape[-1]; pad = lambda a: np.concatenate([a, np.zeros(a.shape[:2] + (128 - P,), np.float32)], -1) if P < 128 else a
            xp, mp = pad(x), pad(m); names = [f[0] for f in FEATURES[mm['feat']]]
            pts = xp[:, [names.index('deta'), names.index('dphi')]]
            L = mm['model'](T(pts), T(xp), T(mp))
    return L.float().cpu().numpy()


def all_inputs(J):
    from interp.part_core import inputs
    return {f: inputs(J, f) for f in ('full', 'kin')}


# ------------------------------------------------------------------------------------------------ analysis
def wls(G, b):
    return np.linalg.solve(G, b)


def fit_boot(Xs, ys, Wb):
    """Xs: (n, K, p) per-jet design over K points, ys: (n, K). Returns beta, bootstrap betas (NB, p), per-jet G, b"""
    G = np.einsum('nkp,nkq->npq', Xs, Xs); b = np.einsum('nkp,nk->np', Xs, ys); p = Xs.shape[2]
    beta = wls(G.sum(0) + 1e-12 * np.eye(p), b.sum(0))
    Gb = (Wb @ G.reshape(len(G), -1)).reshape(-1, p, p) + 1e-12 * np.eye(p); bb = Wb @ b
    return beta, np.linalg.solve(Gb, bb[..., None])[..., 0], G, b


def r2(y, yh):
    y = y.ravel(); yh = yh.ravel(); return float(1 - ((y - yh) ** 2).sum() / max((y ** 2).sum(), 1e-30))


def analyse(d, Mo, names, sel, rd, rng, y_lab, iW, iZ, Lbase, Ledit):
    """d: (N, E) read-out change; Mo: base moments; sel: bool jets. Returns summary dict (and private bootstrap arrays)."""
    E = {nm: i for i, nm in enumerate(names)}; idx = np.where(sel)[0]; n = len(idx); d = d[idx].astype(np.float64)
    M = {k: v[idx] for k, v in Mo.items()}
    Wb = boot_W(n, NB, rng)
    # mirror pairs (canonical: first non-zero coordinate > 0)
    pairs = []
    for ce in GRID:
        for cp in GRID:
            if (ce, cp) == (0, 0) or not (ce > 0 or (ce == 0 and cp > 0)): continue
            pairs.append((ce, cp))
    a = np.array(pairs); nK = len(a)
    ip = [E[K(ce, cp)] for ce, cp in pairs]; im = [E[K(-ce, -cp)] for ce, cp in pairs]
    dp_, dm_ = d[:, ip], d[:, im]; odd = (dp_ - dm_) / 2; even = (dp_ + dm_) / 2
    P = lambda ce, cp: 0.5 * np.log(np.maximum(M2_after(M, ce, cp), 1e-30) / M['M2'])
    Pp = np.stack([P(ce, cp) for ce, cp in pairs], 1); Pm = np.stack([P(-ce, -cp) for ce, cp in pairs], 1)
    Po, Pe = (Pp - Pm) / 2, (Pp + Pm) / 2
    ce_, cp_ = np.broadcast_to(a[:, 0], (n, nK)), np.broadcast_to(a[:, 1], (n, nK))
    # calibration
    yd = np.stack([d[:, E[k]] for k in REF], 1); kk = np.stack([0.5 * np.log(np.maximum(M2_after(M, *REF[k]), 1e-30) / M['M2']) for k in REF], 1)
    num, den = (yd * kk).sum(1), (kk * kk).sum(1); kap = num.sum() / den.sum(); kap_b = (Wb @ num) / (Wb @ den)
    out = dict(n=int(n), kappa=st(kap, kap_b))
    # odd variance share
    so, se = (odd ** 2).sum(1), (even ** 2).sum(1); fo = so.sum() / (so.sum() + se.sum()); fo_b = (Wb @ so) / (Wb @ (so + se))
    out['odd_var_share'] = st(fo, fo_b)
    # reduced odd fit: Po, c_eta, c_phi
    Xo = np.stack([Po, ce_, cp_], 2); bo, bo_b, Go, bvec = fit_boot(Xo, odd, Wb)
    Xe = Pe[..., None]; be, be_b, _, _ = fit_boot(Xe, even, Wb)
    go, go_b = bo[0] / kap, bo_b[:, 0] / kap_b; ge, ge_b = be[0] / kap, be_b[:, 0] / kap_b
    out['g_odd'] = st(go, go_b); out['g_even'] = st(ge, ge_b)
    out['g_odd_minus_even'] = st(go - ge, go_b - ge_b); out['g_odd_over_even'] = st(go / ge if ge else np.nan, go_b / ge_b)
    out['b_odd_raw'] = st(bo[0], bo_b[:, 0]); out['b_even_raw'] = st(be[0], be_b[:, 0])
    out['a_eta'] = st(bo[1], bo_b[:, 1]); out['a_phi'] = st(bo[2], bo_b[:, 2]); out['a_eta_minus_phi'] = st(bo[1] - bo[2], bo_b[:, 1] - bo_b[:, 2])
    # sign-flip permutation (H0: odd part = 0, response symmetric under c -> -c)
    Gs = Go.sum(0); S_ = rng.choice([-1.0, 1.0], size=(NPERM, n)); bp = np.linalg.solve(Gs, (S_ @ bvec).T).T
    out['perm_p_b_odd'] = float((1 + (np.abs(bp[:, 0]) >= abs(bo[0])).sum()) / (1 + NPERM))
    out['perm_p_a_eta'] = float((1 + (np.abs(bp[:, 1]) >= abs(bo[1])).sum()) / (1 + NPERM))
    out['perm_p_a_phi'] = float((1 + (np.abs(bp[:, 2]) >= abs(bo[2])).sum()) / (1 + NPERM))
    # variance explained
    out['R2_odd_full'] = r2(odd, np.einsum('nkp,p->nk', Xo, bo)); out['R2_odd_centroid_only'] = r2(odd, Po * fit_boot(Po[..., None], odd, Wb[:2])[0][0])
    out['R2_odd_fixed_only'] = r2(odd, np.einsum('nkp,p->nk', Xo[..., 1:], fit_boot(Xo[..., 1:], odd, Wb[:2])[0]))
    out['R2_even'] = r2(even, Pe * be[0])
    yh_all = np.einsum('nkp,p->nk', Xo, bo) + Pe * be[0]
    full_p = np.concatenate([odd + even, even - odd], 1); full_h = np.concatenate([yh_all, Pe * be[0] - np.einsum('nkp,p->nk', Xo, bo)], 1)
    out['R2_grid_total'] = r2(full_p, full_h)
    out['R2_grid_even_only'] = r2(full_p, np.concatenate([Pe * be[0]] * 2, 1))
    PP = np.concatenate([Pp, Pm], 1); b1 = fit_boot(PP[..., None], full_p, Wb[:2])[0][0]
    out['R2_grid_one_gain'] = r2(full_p, PP * b1); out['g_one'] = float(b1 / kap)
    # eta / phi split (first order)
    m1e, m1p, m0, m2 = M['M1e'][:, None], M['M1p'][:, None], M['M0'][:, None], M['M2'][:, None]
    Xo2 = np.stack([ce_ * m1e / m2, cp_ * m1p / m2, ce_, cp_], 2); bo2, bo2_b, _, _ = fit_boot(Xo2, odd, Wb)
    Xe2 = np.stack([ce_ ** 2 * m0 / (2 * m2), cp_ ** 2 * m0 / (2 * m2), ce_ * cp_ * m0 / m2], 2); be2, be2_b, _, _ = fit_boot(Xe2, even, Wb)
    out['split'] = dict(g_odd_eta=st(bo2[0] / kap, bo2_b[:, 0] / kap_b), g_odd_phi=st(bo2[1] / kap, bo2_b[:, 1] / kap_b),
                        g_odd_eta_minus_phi=st((bo2[0] - bo2[1]) / kap, (bo2_b[:, 0] - bo2_b[:, 1]) / kap_b),
                        g_even_eta=st(be2[0] / kap, be2_b[:, 0] / kap_b), g_even_phi=st(be2[1] / kap, be2_b[:, 1] / kap_b),
                        g_even_eta_minus_phi=st((be2[0] - be2[1]) / kap, (be2_b[:, 0] - be2_b[:, 1]) / kap_b),
                        g_even_cross=st(be2[2] / kap, be2_b[:, 2] / kap_b), a_eta=st(bo2[2], bo2_b[:, 2]), a_phi=st(bo2[3], bo2_b[:, 3]))
    # model-free axis comparison of the even part: sum even / sum Pe on the eta axis vs the phi axis
    ax_e = [i for i, (ce, cp) in enumerate(pairs) if cp == 0]; ax_p = [i for i, (ce, cp) in enumerate(pairs) if ce == 0]
    ne, de_ = (even[:, ax_e]).sum(1), Pe[:, ax_e].sum(1); npp, dpp = (even[:, ax_p]).sum(1), Pe[:, ax_p].sum(1)
    re_ = ne.sum() / de_.sum() / kap; rp_ = npp.sum() / dpp.sum() / kap
    re_b = (Wb @ ne) / (Wb @ de_) / kap_b; rp_b = (Wb @ npp) / (Wb @ dpp) / kap_b
    out['axis_even_gain'] = dict(eta=st(re_, re_b), phi=st(rp_, rp_b), eta_minus_phi=st(re_ - rp_, re_b - rp_b))
    # per-jet surface: gradient (odd) and quadratic form (even)
    A = a; gj = np.linalg.solve(A.T @ A, A.T @ odd.T).T                                    # (n, 2)
    Q = np.stack([a[:, 0] ** 2, a[:, 1] ** 2, 2 * a[:, 0] * a[:, 1]], 1); hj = np.linalg.solve(Q.T @ Q, Q.T @ even.T).T   # (n, 3)
    m1 = np.stack([M['M1e'], M['M1p']], 1); cosv = (gj * m1).sum(1) / np.maximum(np.linalg.norm(gj, axis=1) * np.linalg.norm(m1, axis=1), 1e-30)
    out['perjet'] = dict(R2_odd_linear=r2(odd, gj @ A.T), R2_even_quadratic=r2(even, hj @ Q.T), cos_grad_M1_mean=st(cosv.mean(), (Wb @ cosv) / n),
                         cos_grad_M1_median=float(np.median(cosv)), grad_eta_mean=st(gj[:, 0].mean(), (Wb @ gj[:, 0]) / n),
                         grad_phi_mean=st(gj[:, 1].mean(), (Wb @ gj[:, 1]) / n),
                         H_eta_mean=st(hj[:, 0].mean(), (Wb @ hj[:, 0]) / n), H_phi_mean=st(hj[:, 1].mean(), (Wb @ hj[:, 1]) / n),
                         aniso=st((hj[:, 0].sum() - hj[:, 1].sum()) / (hj[:, 0].sum() + hj[:, 1].sum()),
                                  ((Wb @ hj[:, 0]) - (Wb @ hj[:, 1])) / ((Wb @ hj[:, 0]) + (Wb @ hj[:, 1]))))
    # held-out random directions
    yr = np.stack([d[:, E[f'rd{k}']] for k in range(4)], 1); c = rd[idx]
    Pr = np.stack([P(c[:, k, 0], c[:, k, 1]) for k in range(4)], 1); Prm = np.stack([P(-c[:, k, 0], -c[:, k, 1]) for k in range(4)], 1)
    Por, Per = (Pr - Prm) / 2, (Pr + Prm) / 2
    yh = bo[0] * Por + bo[1] * c[:, :, 0] + bo[2] * c[:, :, 1] + be[0] * Per; yh_e = be[0] * Per
    out['heldout'] = dict(R2_fit=r2(yr, yh), R2_even_only=r2(yr, yh_e),
                          R2_one_gain=r2(yr, b1 * Pr), within_jet_sd=float(yr.std(1).mean()),
                          within_jet_sd_pred=float(yh.std(1).mean()), within_jet_sd_even_pred=float(yh_e.std(1).mean()),
                          mean=float(yr.mean()))
    # ratios to matched dilations
    out['R_matched'] = {}
    for k, (ce, cp) in REF.items():
        t_, dl = d[:, E[K(ce, cp)]], d[:, E[k]]; v = t_.mean() / dl.mean(); b_ = (Wb @ t_) / (Wb @ dl)
        out['R_matched'][k] = dict(trans=f'({ce:+.2f},{cp:+.2f})', **st(v, b_, 1.0), mean_trans=float(t_.mean()), mean_dil=float(dl.mean()))
    for ax, (kp, km) in dict(phi=('dil_p+', 'dil_p-'), eta=('dil_e+', 'dil_e-')).items():
        cp_, cm_ = REF[kp], REF[km]
        tp, tm = d[:, E[K(*cp_)]], d[:, E[K(*cm_)]]; dp2, dm2 = d[:, E[kp]], d[:, E[km]]
        v = tp.mean() / dp2.mean() - tm.mean() / dm2.mean(); b_ = (Wb @ tp) / (Wb @ dp2) - (Wb @ tm) / (Wb @ dm2)
        out['R_matched'][f'{ax}_plus_minus_minus'] = st(v, b_)
    # phi + 0.06 decomposition
    i06 = pairs.index((0.0, 0.06)); tot = d[:, E['tr_+0.00_+0.06']]
    parts = dict(total=tot, even=even[:, i06], odd=odd[:, i06], odd_centroid=bo[0] * Po[:, i06], odd_fixed=np.full(n, bo[2] * 0.06),
                 odd_rest=odd[:, i06] - bo[0] * Po[:, i06] - bo[2] * 0.06)
    out['phi006'] = {k: st(v.mean(), (Wb @ v) / n) for k, v in parts.items()}
    # rotations and reference shifts
    out['edits_mean'] = {}
    for nm in ['tr_+0.00_+0.06', 'tr_+0.00_-0.06', 'tr_+0.06_+0.00', 'tr_-0.06_+0.00', 'tr_+0.00_+0.08', 'tr_+0.08_+0.00', 'rot90', 'rot180', 'rot270'] + list(REF):
        v = d[:, E[nm]]; out['edits_mean'][nm] = dict(**st(v.mean(), (Wb @ v) / n), mean_abs=float(np.abs(v).mean()))
    # mean surface (9 x 9) for the figure
    S9 = np.zeros((9, 9))
    for i, ce in enumerate(GRID):
        for j, cp in enumerate(GRID):
            S9[i, j] = 0.0 if (ce == 0 and cp == 0) else d[:, E[K(ce, cp)]].mean()
    out['surface'] = S9
    out['_go_b'], out['_ge_b'] = go_b, ge_b
    return out


def calls(Lb, Le, y, iW, iZ):
    w = y == iW; out = {}
    for nm, L in Le.items():
        c0, c1 = Lb[w].argmax(1), L[w].argmax(1)
        out[nm] = dict(WtoZ_before=float((c0 == iZ).mean()), WtoZ_after=float((c1 == iZ).mean()), mcnemar_W=mcnemar(c0 == iW, c1 == iW))
    return out


# ------------------------------------------------------------------------------------------------ DAS
class Hooks:
    def __init__(self, model, torch):
        self.t = torch; self.m = model; self.mode = None; self.rec = {}; self.mp = None
        mods = {'emb': model.embed, **{f'b{l + 1}': b for l, b in enumerate(model.blocks)}}
        self.h = [mod.register_forward_hook(self._f(k)) for k, mod in mods.items()]

    def _f(self, key):
        def f(mod, inp, out):
            if self.mode is None: return None
            mp = self.mp; cnt = mp.sum(0)
            if self.mode == 'rec':
                self.rec[key] = ((out * mp).sum(0) / cnt).detach(); return None
            site, q, md = self.mode
            if key != site: return None
            qn = q / q.norm(); cur = (out * mp).sum(0) / cnt
            return out + (((md - cur) @ qn)[:, None] * qn[None, :])[None] * mp
        return f

    def remove(self):
        for h in self.h: h.remove()


def das(name, mm, J, sel_tr, sel_ev, rng, torch):
    """per-block 1-d DAS for the odd and even parts of the +-c response (c random direction, |c| = 0.06)"""
    from interp.part_core import inputs, sub
    model, dev = mm['model'], mm['dev']; H = Hooks(model, torch); T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    ids = np.concatenate([sel_tr, sel_ev]); Jd = sub(J, ids); n = len(ids)
    th = rng.uniform(0, 2 * np.pi, n); c = 0.06 * np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32)
    from interp.part_core import kin, build
    pt, de, dp, ms = kin(Jd); r = (pt > 0).astype(np.float32)
    Jb = build(Jd, pt, de, dp, ms); Jp = build(Jd, pt, de + c[:, :1] * r, dp + c[:, 1:] * r, ms); Jm = build(Jd, pt, de - c[:, :1] * r, dp - c[:, 1:] * r, ms)
    xb, vb, mb = inputs(Jb, mm['feat']); xp, _, mp_ = inputs(Jp, mm['feat']); xm, _, mm_ = inputs(Jm, mm['feat'])
    assert np.array_equal(mb, mp_) and np.array_equal(mb, mm_)
    cnt = mb[:, 0].sum(1).astype(int); ntr = len(sel_tr)

    def run(rows, x, mode=None, grad=False):
        P = int(cnt[rows].max()); mt = T(mb[rows][:, :, :P]); H.mp = mt[:, 0, :].T[..., None]; H.mode = mode; H.rec = {}
        with (contextlib.nullcontext() if grad else torch.no_grad()):
            L = model(T(x[rows][:, :, :P]), T(vb[rows][:, :, :P]), mt)
        H.mode = None; return L, dict(H.rec)

    res = dict(sites={})
    ev_rows = np.arange(ntr, n); ev_b = [ev_rows[np.argsort(cnt[ev_rows], kind='stable')][i:i + CH] for i in range(0, len(ev_rows), CH)]
    # eval references
    REFS = []
    for rows in ev_b:
        Lb, _ = run(rows, xb); Lp, Rp = run(rows, xp, 'rec'); Lm, Rm = run(rows, xm, 'rec')
        REFS.append((rows, Lb, Lp, Lm, Rp, Rm))
    Wb = boot_W(len(ev_rows), NB, rng)
    for site in SITES:
        Qs = {}
        for target in ('odd', 'even'):
            g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
            q = torch.randn(C, generator=g).to(dev).requires_grad_(True); opt = torch.optim.Adam([q], lr=0.02); losses = []
            for s in range(STEPS):
                rows = np.sort(rng.choice(ntr, min(BS, ntr), replace=False))
                with torch.no_grad():
                    Lb, _ = run(rows, xb); Lp, Rp = run(rows, xp, 'rec'); Lm, Rm = run(rows, xm, 'rec')
                Ip, _ = run(rows, xb, (site, q, Rp[site]), grad=True); Im, _ = run(rows, xb, (site, q, Rm[site]), grad=True)
                loss = (((Ip - Im) - (Lp - Lm)) ** 2).sum(1).mean() if target == 'odd' else (((Ip + Im) - (Lp + Lm)) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
            Qs[target] = (q.detach() / q.detach().norm(), float(np.mean(losses[:5])), float(np.mean(losses[-20:])))
        for k in range(NRAND):
            Qs[f'rand{k}'] = (torch.from_numpy(rng.standard_normal(C).astype(np.float32)).to(dev), None, None)
        acc = {qn: dict(eo=np.zeros(len(ev_rows)), ee=np.zeros(len(ev_rows)), so=np.zeros(len(ev_rows)), se=np.zeros(len(ev_rows))) for qn in Qs}
        ref_o = np.zeros(len(ev_rows)); ref_e = np.zeros(len(ev_rows)); pos = np.full(n, -1); pos[ev_rows] = np.arange(len(ev_rows))
        refz_o = np.zeros(len(ev_rows)); refz_e = np.zeros(len(ev_rows))
        iW, iZ = 7, 6
        for rows, Lb, Lp, Lm, Rp, Rm in REFS:
            pr = pos[rows]; Co = (Lp - Lm) / 2; Ce = (Lp + Lm) / 2 - Lb
            ref_o[pr] = (Co ** 2).sum(1).double().cpu().numpy(); ref_e[pr] = (Ce ** 2).sum(1).double().cpu().numpy()
            zo = (Co[:, iZ] - Co[:, iW]); ze = (Ce[:, iZ] - Ce[:, iW])
            refz_o[pr] = (zo ** 2).double().cpu().numpy(); refz_e[pr] = (ze ** 2).double().cpu().numpy()
            for qn, (qv, _, _) in Qs.items():
                Ip, _ = run(rows, xb, (site, qv, Rp[site])); Im, _ = run(rows, xb, (site, qv, Rm[site]))
                Io = (Ip - Im) / 2; Ie = (Ip + Im) / 2 - Lb
                acc[qn]['eo'][pr] = ((Io - Co) ** 2).sum(1).double().cpu().numpy(); acc[qn]['ee'][pr] = ((Ie - Ce) ** 2).sum(1).double().cpu().numpy()
                acc[qn]['so'][pr] = ((Io[:, iZ] - Io[:, iW]) * zo).double().cpu().numpy(); acc[qn]['se'][pr] = ((Ie[:, iZ] - Ie[:, iW]) * ze).double().cpu().numpy()
        o = dict(loss={t: Qs[t][1:] for t in ('odd', 'even')}, cos_odd_even=float(abs((Qs['odd'][0] @ Qs['even'][0]).item())))
        B = {}
        for qn in Qs:
            a_ = acc[qn]
            B[qn] = dict(rec_odd=1 - (Wb @ a_['eo']) / (Wb @ ref_o), rec_even=1 - (Wb @ a_['ee']) / (Wb @ ref_e),
                         zw_odd=(Wb @ a_['so']) / (Wb @ refz_o), zw_even=(Wb @ a_['se']) / (Wb @ refz_e))
            o[qn] = dict(rec_odd=1 - a_['eo'].sum() / ref_o.sum(), rec_even=1 - a_['ee'].sum() / ref_e.sum(),
                         zw_odd=a_['so'].sum() / refz_o.sum(), zw_even=a_['se'].sum() / refz_e.sum(),
                         **{f'{k}_ci': ci(v) for k, v in B[qn].items()})
        for t in ('odd', 'even'):
            rm = np.mean([B[f'rand{k}'][f'rec_{t}'] for k in range(NRAND)], 0); dlt = B[t][f'rec_{t}'] - rm
            o[f'{t}_minus_rand'] = dict(v=float(o[t][f'rec_{t}'] - np.mean([o[f'rand{k}'][f'rec_{t}'] for k in range(NRAND)])), ci=ci(dlt), p=p_two(dlt))
        res['sites'][site] = o
        log(f'  DAS {name} {site}: odd rec {o["odd"]["rec_odd"]:.3f} (rand {np.mean([o[f"rand{k}"]["rec_odd"] for k in range(NRAND)]):.3f}), '
            f'even rec {o["even"]["rec_even"]:.3f} (rand {np.mean([o[f"rand{k}"]["rec_even"] for k in range(NRAND)]):.3f}); '
            f'q_odd on even {o["odd"]["rec_even"]:.3f}, q_even on odd {o["even"]["rec_odd"]:.3f}, |cos| {o["cos_odd_even"]:.2f}')
    for t in ('odd', 'even'):
        hp = holm([res['sites'][s][f'{t}_minus_rand']['p'] for s in SITES])
        for s, h in zip(SITES, hp): res['sites'][s][f'{t}_minus_rand']['p_holm'] = h
    res['ref_energy'] = dict(odd=float(np.mean(ref_o)), even=float(np.mean(ref_e)))
    H.remove()
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    from interp.part_core import CL, sub, cat, sample, kin, jmass
    iW, iZ, iQ = CL.index('Wqq'), CL.index('Zqq'), CL.index('QCD')
    OUTD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    Js = []; rowinfo = {}
    for c, f in FILES.items():
        J, rows = load_rows(TEST / f, PER, rng); Js.append(J); rowinfo[c] = dict(file=f, first_rows=rows[:5].tolist(), n=len(rows))
        log(f'loaded {PER} random rows of {f}')
    J = cat(Js); del Js
    cnt = J['mask'].sum(1); keep = cnt >= 2; J = sub(J, np.where(keep)[0]); y = J['label'].astype(int); N = len(y)
    log(f'sample N = {N} (dropped {int((~keep).sum())} jets with < 2 particles); classes {np.bincount(y, minlength=10).tolist()}')
    M, torch = load_models(); log(f'models: {list(M)}')
    names = edit_names(); E = len(names); log(f'{E} edits')
    rd = np.zeros((N, 4, 2), np.float32); th = rng.uniform(0, 2 * np.pi, (N, 4)); rd[..., 0] = 0.06 * np.cos(th); rd[..., 1] = 0.06 * np.sin(th)
    pt, de, dp, ms = kin(J); Mo = moments(pt, de, dp); m0 = jmass(J)
    L = {k: np.zeros((N, E, 10), np.float32) for k in M}
    dmass = {nm: [] for nm in ('tr_+0.08_+0.00', 'tr_+0.00_+0.08', 'tr_+0.08_+0.08', 'rot90', 'rot180')}
    order = np.argsort(J['mask'].sum(1), kind='stable')
    t1 = time.time()
    for ci_, i in enumerate(range(0, N, CH)):
        idx = np.sort(order[i:i + CH]); Jc = sub(J, idx); S = prep_chunk(Jc, rd[idx])
        for e, nm in enumerate(names):
            Je = make_edit(S, nm); X = all_inputs(Je)
            if nm in dmass: dmass[nm].append(jmass(Je) / np.maximum(jmass(make_edit(S, 'base')), 1e-6) - 1)
            for k, mm in M.items(): L[k][idx, e] = run_model(mm, X, torch)
        if ci_ % 5 == 0: log(f'  chunk {ci_ + 1} / {math.ceil(N / CH)} ({time.time() - t1:.0f} s)')
    log(f'all edits done ({time.time() - t1:.0f} s)')
    res = dict(sample=dict(files=rowinfo, seed=SEED, N=N, per_class=PER, classes=np.bincount(y, minlength=10).tolist()), edits=names,
               grid=GRID.tolist(), mass_change={k: dict(median_abs=float(np.median(np.abs(np.concatenate(v)))), max_abs=float(np.max(np.abs(np.concatenate(v)))))
                                                for k, v in dmass.items()}, models={})
    # accuracy on the P1 sample
    try:
        J1 = sample(per_class=100 if SMOKE else 1000, start=3000); X1 = all_inputs(J1); y1 = J1['label'].astype(int)
        acc_p1 = {}
        for k, mm in M.items():
            Lp = np.concatenate([run_model(mm, {f: tuple(a[s] for a in X1[f]) for f in X1}, torch) for s in [slice(j, j + CH) for j in range(0, len(y1), CH)]])
            acc_p1[k] = float((Lp.argmax(1) == y1).mean())
        res['acc_P1_sample'] = acc_p1; log(f'P1-sample accuracy: {acc_p1}')
    except Exception as ex:                                       # noqa: BLE001
        log(f'P1 sample failed: {ex!r}'); res['acc_P1_sample'] = None
    np.savez_compressed(OUTD / f'p23_axis_surface{TAG}.npz', y=y, M0=Mo['M0'], M1e=Mo['M1e'], M1p=Mo['M1p'], M2=Mo['M2'], m=m0, rd=rd,
                        **{f'dzw_{k}': (L[k][..., iZ] - L[k][..., iW] - (L[k][:, :1, iZ] - L[k][:, :1, iW])).astype(np.float32) for k in M})
    selW, selZ, selQ = y == iW, y == iZ, y == iQ
    for k in M:
        Lk = L[k]; B = Lk[:, 0]
        acc = dict(sample_10class=float((B.argmax(1) == y).mean()), W=float((B[selW].argmax(1) == iW).mean()), Z=float((B[selZ].argmax(1) == iZ).mean()),
                   QCD=float((B[selQ].argmax(1) == iQ).mean()))
        dzw = (Lk[..., iZ] - Lk[..., iW]) - (B[:, None, iZ] - B[:, None, iW])
        dq = (Lk[..., iQ] - (Lk[..., iW] + Lk[..., iZ]) / 2) - (B[:, None, iQ] - (B[:, None, iW] + B[:, None, iZ]) / 2)
        r = dict(path=M[k]['path'], acc=acc)
        for sub_, (dd, sel) in dict(WZ=(dzw, selW | selZ), W=(dzw, selW), Z=(dzw, selZ), QCD=(dzw, selQ), QCD_qread=(dq, selQ)).items():
            r[sub_] = analyse(dd, Mo, names, sel, rd, np.random.default_rng(SEED + 1), y, iW, iZ, None, None)
        sel_e = ['tr_+0.00_+0.06', 'tr_+0.00_-0.06', 'tr_+0.06_+0.00', 'tr_-0.06_+0.00', 'tr_+0.00_+0.08', 'rot90', 'rot180'] + list(REF)
        r['calls'] = calls(B, {nm: Lk[:, names.index(nm)] for nm in sel_e}, y, iW, iZ)
        res['models'][k] = r
        a = r['WZ']
        log(f'{k}: acc {acc}; WZ kappa {a["kappa"]["v"]:.3f}; g_odd {a["g_odd"]["v"]:.3f} {a["g_odd"]["ci"]}, g_even {a["g_even"]["v"]:.3f} '
            f'{a["g_even"]["ci"]}; odd share {a["odd_var_share"]["v"]:.3f}; perm p {a["perm_p_b_odd"]:.4f}; a_eta {a["a_eta"]["v"]:.3f} a_phi {a["a_phi"]["v"]:.3f}; '
            f'even eta-phi {a["split"]["g_even_eta_minus_phi"]["v"]:.3f} {a["split"]["g_even_eta_minus_phi"]["ci"]}')
        with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    # Holm over models
    ks = list(res['models'])
    fam = dict(odd_perm=[res['models'][k]['WZ']['perm_p_b_odd'] for k in ks],
               one_gain=[res['models'][k]['WZ']['g_odd_minus_even']['p0'] for k in ks],
               even_eta_phi=[res['models'][k]['WZ']['split']['g_even_eta_minus_phi']['p0'] for k in ks],
               fixed_eta_phi=[res['models'][k]['WZ']['a_eta_minus_phi']['p0'] for k in ks],
               a_phi_perm=[res['models'][k]['WZ']['perm_p_a_phi'] for k in ks])
    res['holm'] = {f: dict(zip(ks, holm(p))) for f, p in fam.items()}
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    log(f'wrote {FN}')
    # DAS
    if not ARGS.no_das:
        res['das'] = {}
        wz = np.where(selW | selZ)[0]; perm = np.random.default_rng(SEED + 2).permutation(wz)
        tr, ev = perm[:N_DAS_TR], perm[N_DAS_TR:N_DAS_TR + N_DAS_EV]
        for k in ('ParT_full', 'ParT_kin', 'ParT_full_untr1'):
            if k not in M: continue
            t1 = time.time(); res['das'][k] = das(k, M[k], J, tr, ev, np.random.default_rng(SEED + 3), torch)
            log(f'DAS {k} done ({time.time() - t1:.0f} s)')
            with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    log(f'wrote {FN}')
    try:
        plot(res)
    except Exception as ex:                                       # noqa: BLE001
        log(f'plot failed: {ex!r}')
    log('P23 DONE')


def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = res or json.load(open(FN)); res = strip(res)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1, 1.5]))
    S = np.array(res['models']['ParT_full']['W']['surface']); g = np.array(res['grid']) * 1000
    v = np.abs(S).max(); im = ax[0].imshow(S.T, origin='lower', extent=[g[0] - 10, g[-1] + 10, g[0] - 10, g[-1] + 10], cmap='RdBu_r', vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax[0], label='mean change of logit(Z) - logit(W)')
    ax[0].set_xlabel(r'axis offset $c_\eta$ [mrad]'); ax[0].set_ylabel(r'axis offset $c_\phi$ [mrad]')
    ax[0].set_title('ParT_full, true W jets: mean response', fontsize=10)
    ks = [k for k in res['models'] if 'untr' not in k]; yy = np.arange(len(ks))   # untrained: real-change gain ~ 0, ratio undefined
    for off, key, col, lab in ((-0.15, 'g_odd', '#c0392b', 'odd part (direction-dependent)'), (0.15, 'g_even', '#2c3e8f', 'even part (|c|$^2$ term)')):
        vals = [res['models'][k]['WZ'][key]['v'] for k in ks]; cis = [res['models'][k]['WZ'][key]['ci'] for k in ks]
        lo = [v_ - c_[0] for v_, c_ in zip(vals, cis)]; hi = [c_[1] - v_ for v_, c_ in zip(vals, cis)]
        ax[1].errorbar(vals, yy + off, xerr=[lo, hi], fmt='o', color=col, label=lab, capsize=3)
    ax[1].axvline(1, color='k', lw=0.8, ls='--'); ax[1].axvline(0, color='gray', lw=0.6)
    ax[1].set_yticks(yy); ax[1].set_yticklabels(ks); ax[1].invert_yaxis()
    ax[1].set_xlabel('gain relative to a real mass change of equal $\\frac{1}{2}\\Delta\\log M_2$ (1 = same response)')
    ax[1].set_xlim(-0.3, 1.3); ax[1].legend(fontsize=8, loc='center right', framealpha=0.95)
    ax[1].set_title('W/Z jets: odd (direction) and even ($|c|^2$) gains, 95 % CI\n(1 = same as a real mass change; untrained nets not shown)', fontsize=10)
    fig.tight_layout(); out = OUTD / f'p23_axis_surface{TAG}.png'; fig.savefig(out, dpi=130); log(f'wrote {out}')


if __name__ == '__main__':
    if ARGS.plot: plot()
    else: main()
