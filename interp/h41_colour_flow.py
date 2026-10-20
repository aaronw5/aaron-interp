"""H41 (loop D, new claim D17, exploratory): colour flow (jet pull) in boosted W / Z / H -> qq / bb versus QCD.

Question. A colour-singlet boson decaying to two quarks puts its soft radiation between the two prongs; a QCD splitting colour-
connects the prongs to the rest of the event and puts more soft radiation outside. Do JetClass / Top Landscape taggers respond to
where the soft radiation sits relative to the prong-prong line, at fixed jet mass, tau21 and multiplicity?

Sample (random rows, never first-k)
  jc   JetClass test files WToQQ_100, ZToQQ_100, HToBB_100, ZJetsToNuNu_100; rows with stored 500 < jet_pt < 1000 GeV,
       |jet_eta| < 2, >= 3 particles; PER random rows per class (default_rng(SEED + k), k = 0..3, SEED 4101).
  top  Top Landscape test split (secondary arm), PER_TOP top + PER_TOP QCD (h4_top_wmass._sample, seed 4105).
Geometry
  jc   two exclusive-kT subjets (p = 1, E-scheme). Prong centres c_k = (eta, phi) of the subjet four-momentum sums.
  top  three exclusive-kT subjets; prongs = the pair with mass closest to 80.4 GeV (the W candidate); particles nearest the third
       subjet are never edited.
  Frame of prong k: u_k = unit vector from c_k toward the other prong, v_k perpendicular. Soft particles: pT < 2 % of the jet pT,
  Delta R > 0.05 from every subjet centre, nearest centre is one of the two prongs.
Edits (only soft particles move; pT, constituent mass, number of particles, distance to the nearest prong unchanged)
  M    mirror along u about the nearest prong: u_rel -> -u_rel (radiation between the prongs moves outside and vice versa).
  C1   mirror along v: v_rel -> -v_rel (same distance to the prong; f_btw and the u-component of the pull kept).
  C2   rotation about the nearest prong by an angle whose displacement 2 rho sin(alpha/2) equals the M displacement of a randomly
       chosen soft particle of the same jet (clipped at 2 rho), random sign.
  Mx, C1x, C2x   the same with the soft-particle pT scaled by one common factor per jet so that the jet invariant mass equals the
       unedited one (bisection).
  TN   translation-only control: the unedited jet moved rigidly by the axis shift that M induces.
  Every variant is rebuilt about its own E-scheme axis (rc, the H23 lesson); M is also scored about the unedited axis (M_st).
Matched set: |Delta m| < 1 GeV and |Delta tau21| < 0.01 for M, C1 and C2 (m, tau21 as h4_top_wmass._obs: invariant mass,
  tau21 with beta 1, R0 0.8, exclusive-kT WTA axes); top arm also |Delta tau32| < 0.01.
Models
  jc   ParT_full, ParT_kin, PN_full, PN_kin (released JetClass weights), ParT_perm (ParT_full with every weight tensor permuted,
       torch seed 5, as H35). Scores s_c = L_c - L_QCD for c = Wqq, Zqq, Hbb (log p(c)/p(QCD)).
  top  Top Landscape ParT, PN, LorentzNet, PFN, EFN and the weight-permuted PFN (h13_pileup_panel.TopModels).
References (jc, positive reference / H0 predictor), per signal class c vs QCD, trained on train_100M <c>_000 / ZJetsToNuNu_000
  in-window rows (NREF per class, rng 4111 + k): LR3 = logistic on (m, m^2, tau21, N); GB3 = HistGradientBoosting on (m, tau21, N);
  LR5 / GB5 = the same plus (theta_p1, theta_p2, f_btw). theta_pk = angle between the pull vector of prong k,
  t_k = sum_i pT_i |r_i| r_i / pT_k over the subjet's particles (r_i relative to c_k), and the vector to the other prong.
  f_btw = soft pT with 0 < u < d and |v| < d / 2 in the frame of the harder prong (d = prong separation) / total soft pT.
Metrics per sample x score x model x set: Delta s of M, C1, C2, TN, M_st (vs identity); M - C1 and M - C2 per jet; mean, 95 %
  bootstrap CI over jets (B 2000); two-sided sign-flip permutation p; Holm over (sample, score) x 4 nets x {M-C1, M-C2} on the
  matched set (48 tests); net - GB3 per jet (H0: response = the (m, tau21, N) prediction), Holm over the same 48 cells; McNemar
  M vs C1 calls at eps_S = 0.5 (threshold = median s_c of unedited class-c jets). C2ST: HGB on jet observables, edited jets of one
  half vs unedited jets of the other half, 5-fold AUC, with and without the colour-flow observables.
Run: ./sync.sh && ./part_job.sh d-h41-run 'python -m interp.h41_colour_flow'   (H41_SMOKE=1: 200 jets per class, h41_smoke*)
Outputs: $AI_RESULTS/physics/h41_colour_flow.json, h41_perjet.npz, h41_colour_flow.png
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ.setdefault('NUMBA_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h4_top_wmass as H          # noqa: E402
from interp import h19_wpair_cross as X       # noqa: E402

try:
    from numba import njit, prange
except ImportError:  # local import check only
    njit = lambda *a, **k: (lambda f: f); prange = range

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H41_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h41_smoke' if SMOKE else 'h41'
PER = 200 if SMOKE else 2000
PER_TOP = 200 if SMOKE else 2000
NREF = 1000 if SMOKE else 10000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
SEED = 4101
TRAIN = Path('/j-jepa-vol/JetClass/Pythia/train_100M')
QCD, HBB, ZQQ, WQQ = 0, 1, 6, 7
SIG = {'Wqq': WQQ, 'Zqq': ZQQ, 'Hbb': HBB}
SAMPLES = ['Wqq', 'Zqq', 'Hbb', 'QCD']
NETS = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin']
NULLS = ['ParT_perm']
REFN = ['LR3', 'GB3', 'LR5', 'GB5']
VARS = ['M', 'C1', 'C2', 'Mx', 'C1x', 'C2x', 'TN']
SOFT_FRAC, CORE = 0.02, 0.05
CF = ['theta1', 'theta2', 'f_btw']
OBS_ALL = ['m', 'mSD', 'tau21', 'N', 'theta1', 'theta2', 'f_btw', 'dR12', 'zsub', 'soft_frac']


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ================================================================================================================ clustering
@njit(cache=False)
def _ktn_labels(px, py, pz, E, n, nobj):
    """exclusive kT (p = 1, E-scheme, rapidity-azimuth dR) down to nobj objects (h4_top_wmass._kt3_labels with nobj free)"""
    vx = px.copy(); vy = py.copy(); vz = pz.copy(); vE = E.copy()
    pt = np.hypot(vx, vy); ph = np.arctan2(vy, vx); yy = np.zeros(n)
    for i in range(n):
        e = max(vE[i], abs(vz[i]) + 1e-9); yy[i] = 0.5 * np.log((e + vz[i]) / (e - vz[i]))
    alive = np.ones(n, np.bool_); lab = np.arange(n); cnt = n
    while cnt > nobj:
        best = 1e300; bi = -1; bj = -1
        for a in range(n):
            if not alive[a]: continue
            for b in range(a + 1, n):
                if not alive[b]: continue
                dphi = abs(ph[a] - ph[b])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                d = min(pt[a] ** 2, pt[b] ** 2) * ((yy[a] - yy[b]) ** 2 + dphi ** 2)
                if d < best: best = d; bi = a; bj = b
        vx[bi] += vx[bj]; vy[bi] += vy[bj]; vz[bi] += vz[bj]; vE[bi] += vE[bj]; alive[bj] = False
        pt[bi] = np.hypot(vx[bi], vy[bi]); ph[bi] = np.arctan2(vy[bi], vx[bi])
        e = max(vE[bi], abs(vz[bi]) + 1e-9); yy[bi] = 0.5 * np.log((e + vz[bi]) / (e - vz[bi]))
        for i in range(n):
            if lab[i] == bj: lab[i] = bi
        cnt -= 1
    out = -np.ones(n, np.int64); k = 0
    for a in range(n):
        if alive[a]:
            for i in range(n):
                if lab[i] == a: out[i] = k
            k += 1
    return out


@njit(parallel=True, cache=False)
def ktn_batch(F, nn, nobj):
    J = F.shape[0]; L = F.shape[1]; out = -np.ones((J, L), np.int64)
    for j in prange(J):
        n = nn[j]
        if n < nobj: continue
        out[j, :n] = _ktn_labels(F[j, :n, 0].astype(np.float64), F[j, :n, 1].astype(np.float64), F[j, :n, 2].astype(np.float64),
                                 F[j, :n, 3].astype(np.float64), n, nobj)
    return out


# ================================================================================================================ geometry
def pkin(P4, nn):
    P = P4.astype(np.float64); real = (np.arange(P.shape[1])[None, :] < nn[:, None])
    pt = np.hypot(P[..., 0], P[..., 1]); real &= pt > 0; ptc = np.where(real, pt, 1.0)
    eta = np.where(real, np.arcsinh(P[..., 2] / ptc), 0); phi = np.where(real, np.arctan2(P[..., 1], P[..., 0]), 0)
    m = np.sqrt(np.maximum(P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1), 0)) * real
    return dict(pt=np.where(real, pt, 0), eta=eta, phi=phi, m=m, real=real)


def p4_of(pt, eta, phi, m, real):
    out = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)
    return (out * real[..., None]).astype(np.float32)


def minv(P4):
    S = P4.astype(np.float64).sum(1); return np.sqrt(np.maximum(S[:, 3] ** 2 - (S[:, :3] ** 2).sum(1), 0))


def subjets(P4, nn, LAB, K):
    """subjet four-vectors (J, K, 4), centres (eta, phi) (J, K), pT (J, K)"""
    L = LAB.shape[1]; real = np.arange(L)[None, :] < nn[:, None]
    oh = np.stack([(LAB == k) & real for k in range(K)], -1).astype(np.float64)
    S = np.einsum('jlk,jld->jkd', oh, P4[:, :L].astype(np.float64)); spt = np.hypot(S[..., 0], S[..., 1])
    ce = np.arcsinh(S[..., 2] / np.maximum(spt, 1e-9)); cp = np.arctan2(S[..., 1], S[..., 0])
    return S, ce, cp, spt


def prong_frame(P4, nn, kind):
    """kind 'jc': kT-2, prongs (0 = harder, 1); 'top': kT-3, prongs = W candidate pair (harder first), third subjet c.
    returns dict with labels, centres (J, K), prong indices pa, pb, subjet four-vectors"""
    F = np.ascontiguousarray(P4.astype(np.float64)); K = 2 if kind == 'jc' else 3
    LAB = ktn_batch(F, nn.astype(np.int64), K); S, ce, cp, spt = subjets(P4, nn, LAB, K); J = len(nn); r = np.arange(J)
    if K == 2:
        pa = np.where(spt[:, 0] >= spt[:, 1], 0, 1); pb = 1 - pa; c = -np.ones(J, int); mW = np.full(J, np.nan)
    else:
        pairs = [(0, 1), (0, 2), (1, 2)]; m2 = lambda v: v[..., 3] ** 2 - (v[..., :3] ** 2).sum(-1)
        pm = np.stack([np.sqrt(np.maximum(m2(S[:, i] + S[:, j]), 0)) for i, j in pairs], 1)
        w = np.argmin(np.abs(pm - 80.4), 1); ab = np.array(pairs)[w]; a, b = ab[:, 0], ab[:, 1]; c = 3 - a - b
        harder = spt[r, a] >= spt[r, b]; pa = np.where(harder, a, b); pb = np.where(harder, b, a); mW = pm[r, w]
    return dict(LAB=LAB, S=S, ce=ce, cp=cp, spt=spt, pa=pa, pb=pb, c=c, K=K, mW=mW)


def soft_geometry(P4, nn, fr):
    """per particle: soft-eligible mask, nearest prong (0 = pa, 1 = pb), offsets e from that prong, u/v unit vectors"""
    k = pkin(P4, nn); J, L = k['pt'].shape; r = np.arange(J)
    jpt = np.hypot(P4[..., 0].astype(np.float64).sum(1), P4[..., 1].astype(np.float64).sum(1))
    # distance to every subjet centre
    de = k['eta'][:, :, None] - fr['ce'][:, None, :]; dp = wrap(k['phi'][:, :, None] - fr['cp'][:, None, :])
    dR = np.hypot(de, dp); near = np.argmin(dR, 2)
    pa, pb = fr['pa'][:, None], fr['pb'][:, None]
    near_prong = (near == pa) | (near == pb)
    soft = k['real'] & (k['pt'] < SOFT_FRAC * jpt[:, None]) & (dR.min(2) > CORE) & near_prong
    which = np.where(near == pa, 0, 1)                                   # 0: harder prong, 1: other prong
    ca = np.stack([fr['ce'][r, fr['pa']], fr['cp'][r, fr['pa']]], 1); cb = np.stack([fr['ce'][r, fr['pb']], fr['cp'][r, fr['pb']]], 1)
    dab = np.stack([cb[:, 0] - ca[:, 0], wrap(cb[:, 1] - ca[:, 1])], 1); d = np.hypot(dab[:, 0], dab[:, 1]); uh = dab / np.maximum(d, 1e-9)[:, None]
    # unit u per particle: from its own prong toward the other one
    u = np.where(which[..., None] == 0, uh[:, None, :], -uh[:, None, :]); v = np.stack([-u[..., 1], u[..., 0]], -1)
    cen = np.where(which[..., None] == 0, ca[:, None, :], cb[:, None, :])
    e = np.stack([k['eta'] - cen[..., 0], wrap(k['phi'] - cen[..., 1])], -1)
    ur = (e * u).sum(-1); vr = (e * v).sum(-1)
    return dict(k=k, soft=soft, which=which, u=u, v=v, cen=cen, e=e, ur=ur, vr=vr, d=d, ca=ca, cb=cb, uh=uh, jpt=jpt)


def make_variants(P4, nn, G, seed):
    """the edited four-vectors of M, C1, C2 (+ x variants and TN filled later); per-jet bookkeeping"""
    k, soft, e, u, v, ur, vr = G['k'], G['soft'], G['e'], G['u'], G['v'], G['ur'], G['vr']
    rng = np.random.default_rng(seed); J, L = soft.shape
    rho = np.hypot(e[..., 0], e[..., 1])
    eM = e - 2 * ur[..., None] * u; eC1 = e - 2 * vr[..., None] * v
    dispM = 2 * np.abs(ur)
    # C2: rotation by alpha with 2 rho sin(alpha / 2) = M displacement of a random soft particle of the same jet
    Dt = np.zeros((J, L))
    for j in range(J):
        s = np.flatnonzero(soft[j])
        if len(s): Dt[j, s] = dispM[j, rng.permutation(s)]
    al = 2 * np.arcsin(np.clip(Dt / np.maximum(2 * rho, 1e-12), 0, 1)) * rng.choice([-1.0, 1.0], (J, L))
    ca_, sa_ = np.cos(al), np.sin(al)
    eC2 = np.stack([ca_ * e[..., 0] - sa_ * e[..., 1], sa_ * e[..., 0] + ca_ * e[..., 1]], -1)
    out = {}; info = {}
    for name, ee in (('M', eM), ('C1', eC1), ('C2', eC2)):
        eta = np.where(soft, G['cen'][..., 0] + ee[..., 0], k['eta']); phi = np.where(soft, wrap(G['cen'][..., 1] + ee[..., 1]), k['phi'])
        out[name] = (eta, phi)
        disp = np.hypot(*(ee - e).transpose(2, 0, 1))
        info[name] = dict(disp_mean=float(disp[soft].mean()) if soft.any() else 0.0, disp_median=float(np.median(disp[soft])) if soft.any() else 0.0)
    info['n_soft_mean'] = float(soft.sum(1).mean()); info['soft_pt_frac_mean'] = float(((k['pt'] * soft).sum(1) / G['jpt']).mean())
    info['frac_jets_with_soft'] = float((soft.sum(1) > 0).mean())
    return out, info


def to_p4(k, eta, phi, scale=None, soft=None):
    """four-vectors; with scale, every soft particle's four-vector is multiplied by scale (pT and mass, so the jet four-vector
    is linear in scale and m(scale = 0) = mass of the other particles <= m0)"""
    if scale is None: return p4_of(k['pt'], eta, phi, k['m'], k['real'])
    return p4_of(np.where(soft, k['pt'] * scale[:, None], k['pt']), eta, phi, np.where(soft, k['m'] * scale[:, None], k['m']), k['real'])


def mass_exact(k, eta, phi, soft, m0, it=50):
    """common soft-pT factor s per jet with m_inv(edited, s) = m0 (bisection in [0, 4]); returns s and success mask"""
    lo = np.zeros(len(m0)); hi = np.full(len(m0), 4.0)
    f = lambda s: minv(to_p4(k, eta, phi, s, soft)) - m0
    flo, fhi = f(lo), f(hi); ok = (flo <= 0) & (fhi >= 0) & (soft.sum(1) > 0)
    for _ in range(it):
        mid = 0.5 * (lo + hi); fm = f(mid); lo = np.where(fm < 0, mid, lo); hi = np.where(fm < 0, hi, mid)
    s = np.where(ok, 0.5 * (lo + hi), 1.0); return s, ok


# ================================================================================================================ observables
def observables(P4, nn, kind):
    """m, mSD, tau21 (tau32), N from h4_top_wmass._obs; pull angles and f_btw from the prong frame of this jet"""
    O = H._obs(P4, nn); fr = prong_frame(P4, nn, kind); G = soft_geometry(P4, nn, fr); k = G['k']; J = len(nn); r = np.arange(J)
    th = []
    for p, q in (('pa', 'pb'), ('pb', 'pa')):
        ip, iq = fr[p], fr[q]; ce, cp = fr['ce'][r, ip][:, None], fr['cp'][r, ip][:, None]
        inS = (fr['LAB'] == ip[:, None]) & k['real'][:, :fr['LAB'].shape[1]]
        re, rp = k['eta'] - ce, wrap(k['phi'] - cp); rr = np.hypot(re, rp)
        w = np.where(inS, k['pt'] * rr, 0) / np.maximum(fr['spt'][r, ip], 1e-9)[:, None]
        t = np.stack([(w * re).sum(1), (w * rp).sum(1)], 1)
        dv = np.stack([fr['ce'][r, iq] - fr['ce'][r, ip], wrap(fr['cp'][r, iq] - fr['cp'][r, ip])], 1)
        th.append(np.abs(np.arctan2(t[:, 0] * dv[:, 1] - t[:, 1] * dv[:, 0], (t * dv).sum(1))))
    # f_btw in the frame of the harder prong
    ea = np.stack([k['eta'] - G['ca'][:, None, 0], wrap(k['phi'] - G['ca'][:, None, 1])], -1)
    uu = (ea * G['uh'][:, None, :]).sum(-1); vv = ea[..., 0] * (-G['uh'][:, None, 1]) + ea[..., 1] * G['uh'][:, None, 0]
    btw = G['soft'] & (uu > 0) & (uu < G['d'][:, None]) & (np.abs(vv) < G['d'][:, None] / 2)
    sp = (k['pt'] * G['soft']).sum(1)
    out = dict(m=O['m'], mSD=O['mSD'], tau21=O['tau21'], tau32=O['tau32'], N=O['n'], theta1=th[0], theta2=th[1],
               f_btw=np.where(sp > 0, (k['pt'] * btw).sum(1) / np.maximum(sp, 1e-12), 0.0), dR12=G['d'],
               zsub=np.minimum(fr['spt'][r, fr['pa']], fr['spt'][r, fr['pb']]) / np.maximum(fr['spt'][r, fr['pa']] + fr['spt'][r, fr['pb']], 1e-9),
               soft_frac=sp / G['jpt'], n_soft=G['soft'].sum(1).astype(float), minv=minv(P4), mW=fr['mW'])
    return out, fr, G


# ================================================================================================================ data
def read_window(path, per, seed, lo=500., hi=1000.):
    import uproot
    from interp.part_core import read_root, cat, sub
    t = uproot.open(path)['tree']; a = t.arrays(['jet_pt', 'jet_eta', 'jet_nparticles'], library='np')
    ok = (a['jet_pt'] > lo) & (a['jet_pt'] < hi) & (np.abs(a['jet_eta']) < 2) & (a['jet_nparticles'] >= 3)
    el = np.flatnonzero(ok); n = min(per, len(el)); idx = np.sort(np.random.default_rng(seed).choice(el, n, replace=False))
    parts = []; N = t.num_entries
    for s in range(0, N, 20000):
        loc = idx[(idx >= s) & (idx < s + 20000)] - s
        if len(loc): parts.append(sub(read_root(path, start=s, stop=min(s + 20000, N)), loc))
    log(f'{Path(path).name}: {len(el)} eligible of {N}; drew {n} (rng {seed})')
    return cat(parts), dict(file=Path(path).name, eligible=int(len(el)), rows=int(N), drawn=int(n), seed=seed)


# ================================================================================================================ models
class JCNets:
    def __init__(self):
        import torch
        from interp.h23_axis_audit import Nets
        from interp.part_core import ParTNetwork
        self.N = Nets(top=False)
        netp = ParTNetwork('full'); g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for p in netp.model.parameters():
                if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
        self.N.jc['ParT_perm'] = ('part', 'full', netp); log(f'jc models: {list(self.N.jc)}')

    def logits(self, P4, nn, base, ax):
        lg = {}; self.N.score('jc', P4, nn, base, ax, None, jc_names=NETS + NULLS, top=False, logits=lg); return lg


# ================================================================================================================ stats
def boot(x, rng, b=B):
    x = np.asarray(x, float)
    if len(x) == 0: return float('nan'), [float('nan')] * 2
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(1); return float(x.mean()), np.percentile(m, [2.5, 97.5]).tolist()


def perm2(x, rng, n=NPERM):
    x = np.asarray(x, float)
    if len(x) == 0 or np.all(np.abs(x) < 1e-12): return 1.0
    obs = abs(x.mean()); s = rng.choice([-1.0, 1.0], (n, len(x))); return float((np.sum(np.abs((s * x).mean(1)) >= obs - 1e-15) + 1) / (n + 1))


def intercept(x, Z, rng, b=B):
    """OLS x = a + Z beta; a = the contrast at zero mismatch in (Delta m, Delta tau21) between the two edits; bootstrap CI"""
    A = np.column_stack([np.ones(len(x)), Z]); co = np.linalg.lstsq(A, x, rcond=None)[0]; bs = []
    for _ in range(b):
        i = rng.integers(0, len(x), len(x)); bs.append(np.linalg.lstsq(A[i], x[i], rcond=None)[0][0])
    bs = np.asarray(bs); p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / b))
    return dict(intercept=float(co[0]), ci=np.percentile(bs, [2.5, 97.5]).tolist(), slope_m=float(co[1]), slope_tau21=float(co[2]), p_boot=p,
                Z_mean=Z.mean(0).tolist(), raw_mean=float(x.mean()))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def mcnemar(a, b):
    from interp.h1_irc_panel import binom_two_sided
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(rate_ref=float(a.mean()), rate_test=float(b.mean()), ref_only=n10, test_only=n01, p=binom_two_sided(n10, n10 + n01))


def c2st(Xa, Xb, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    Xc = np.concatenate([Xa, Xb]); yc = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    if len(Xa) < 20 or len(Xb) < 20: return float('nan')
    return float(cross_val_score(HistGradientBoostingClassifier(max_iter=200, random_state=seed), Xc, yc, cv=5, scoring='roc_auc').mean())


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


# ================================================================================================================ references
def ref_X(O, which):
    base = [O['m'], O['m'] ** 2 / 100.0, O['tau21'], O['N']] if which.startswith('LR') else [O['m'], O['tau21'], O['N']]
    if which.endswith('5'): base += [O['theta1'], O['theta2'], O['f_btw']]
    return np.column_stack(base)


def fit_refs(out):
    from interp.part_core import PRE, kin
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    OBS = {}
    for k, c in enumerate(SAMPLES):
        J, inf = read_window(TRAIN / f'{PRE[c]}_000.root', NREF, 4111 + k); out.setdefault('ref_rows', {})[c] = inf
        P4, nn = X.jc_P4(J); P4 = H._identity(P4, nn).astype(np.float32); OBS[c] = observables(P4, nn, 'jc')[0]
        log(f'ref observables {c}: {len(nn)} jets')
    R, info = {}, {}
    for c in SIG:
        Os = {kk: np.concatenate([OBS[c][kk], OBS['QCD'][kk]]) for kk in OBS[c]}; y = np.r_[np.ones(len(OBS[c]['m'])), np.zeros(len(OBS['QCD']['m']))]
        tr = np.random.default_rng(4120).random(len(y)) < 0.8
        for nm in REFN:
            Xf = ref_X(Os, nm)
            if nm.startswith('LR'):
                sc = StandardScaler().fit(Xf[tr]); mdl = LogisticRegression(max_iter=3000).fit(sc.transform(Xf[tr]), y[tr])
            else:
                sc = None; mdl = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xf[tr], y[tr])
            pr = mdl.predict_proba(Xf[~tr] if sc is None else sc.transform(Xf[~tr]))[:, 1]
            R[(c, nm)] = (mdl, sc); info[f'{c}|{nm}'] = dict(heldout_AUC=float(H._auc(pr, y[~tr])), heldout_acc=float(((pr > 0.5) == (y[~tr] == 1)).mean()))
            log(f'ref {c} {nm}: held-out AUC {info[f"{c}|{nm}"]["heldout_AUC"]:.4f}')
    return R, info


def ref_scores(R, O):
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    out = {}
    for (c, nm), (mdl, sc) in R.items():
        Xf = ref_X(O, nm); out[(c, nm)] = lo(mdl.predict_proba(Xf if sc is None else sc.transform(Xf))[:, 1])
    return out


# ================================================================================================================ JetClass arm
def run_jc(out):
    from interp.part_core import PRE, cat, kin
    from interp.h23_axis_audit import axis, translate
    Js = []
    for k, c in enumerate(SAMPLES):
        J, inf = read_window(X.jc_file(c, 100), PER, SEED + k); Js.append(J); out.setdefault('sample', {})[c] = inf
    J = cat(Js); lab = J['label'].astype(int); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32); cls = np.array([{QCD: 'QCD', HBB: 'Hbb', ZQQ: 'Zqq', WQQ: 'Wqq'}[l] for l in lab])
    log(f'jc sample {len(nn)} jets; median jet pT {np.median(J["jet_pt"]):.0f} GeV')
    # ---- observables, geometry and edits
    O0, fr, G = observables(P4, nn, 'jc'); log('unedited observables done')
    V, out['edit_info'] = make_variants(P4, nn, G, SEED + 10); k0 = G['k']; soft = G['soft']
    P = {'id': P4}
    for v in ('M', 'C1', 'C2'): P[v] = to_p4(k0, *V[v])
    ex = {}; BASE = {}
    for v in ('M', 'C1', 'C2'):
        s, ok = mass_exact(k0, *V[v], soft, O0['minv']); P[v + 'x'] = to_p4(k0, *V[v], s, soft); ex[v + 'x'] = ok
        BASE[v + 'x'] = dict(J); BASE[v + 'x']['pmass'] = np.where(soft, J['pmass'] * s[:, None], J['pmass']).astype(np.float32)
        out['edit_info'][v + 'x'] = dict(success=float(ok.mean()), scale_median=float(np.median(s[ok])) if ok.any() else None,
                                         scale_p05_p95=np.quantile(s[ok], [0.05, 0.95]).tolist() if ok.any() else None)
    a0 = axis(P4, nn); aM = axis(P['M'], nn); shM = (aM[0] - a0[0], wrap(aM[1] - a0[1])); P['TN'] = translate(P4, nn, *shM)
    out['edit_info']['axis_shift_M_median'] = float(np.median(np.hypot(*shM))); out['edit_info']['axis_shift_M_p90'] = float(np.quantile(np.hypot(*shM), 0.9))
    for v in ('M', 'C1', 'C2'):
        kk = pkin(P[v], nn); dr = np.hypot(kk['eta'] - a0[0][:, None], wrap(kk['phi'] - a0[1][:, None]))
        out['edit_info'][v]['frac_soft_moved_beyond_R08'] = float((dr[soft] > 0.8).mean()) if soft.any() else 0.0
    log('edit info', json.dumps(clean(out['edit_info'])))
    OBS = {'id': O0}
    for v in VARS:
        if v == 'TN': continue
        OBS[v] = observables(P[v], nn, 'jc')[0]
    log('edited observables done')
    out['obs_change'] = {}
    for c in SAMPLES:
        sm = cls == c
        out['obs_change'][c] = {v: {o: dict(mean=float((OBS[v][o] - O0[o])[sm].mean()), mean_abs=float(np.abs(OBS[v][o] - O0[o])[sm].mean()))
                                    for o in ('m', 'tau21', 'N', 'theta1', 'theta2', 'f_btw', 'soft_frac')} for v in OBS if v != 'id'}
        out['obs_change'][c]['id_means'] = {o: float(O0[o][sm].mean()) for o in ('m', 'tau21', 'N', 'theta1', 'theta2', 'f_btw', 'soft_frac')}
    log('observable changes M (Wqq):', {o: round(x['mean'], 4) for o, x in out['obs_change']['Wqq']['M'].items()})
    log('observable changes M (QCD):', {o: round(x['mean'], 4) for o, x in out['obs_change']['QCD']['M'].items()})
    # ---- sets
    dm = lambda v: np.abs(OBS[v]['m'] - O0['m']) < 1.0
    dt = lambda v: np.abs(OBS[v]['tau21'] - O0['tau21']) < 0.01
    has = soft.sum(1) > 0
    dm3 = lambda v: np.abs(OBS[v]['m'] - O0['m']) < 3.0
    dt3 = lambda v: np.abs(OBS[v]['tau21'] - O0['tau21']) < 0.03
    SETS = {'matched': has & dm('M') & dt('M') & dm('C1') & dt('C1') & dm('C2') & dt('C2'), 'all': has,
            'matched_loose': has & dm3('M') & dt3('M') & dm3('C1') & dt3('C1') & dm3('C2') & dt3('C2'),
            'mass_exact': has & ex['Mx'] & ex['C1x'] & ex['C2x']}
    out['sets'] = {s: {c: int((m & (cls == c)).sum()) for c in SAMPLES} for s, m in SETS.items()}
    out['mass_exact_dtau21'] = {v: float(np.abs(OBS[v]['tau21'] - O0['tau21'])[SETS['mass_exact']].mean()) for v in ('Mx', 'C1x', 'C2x')}
    out['mass_exact_dm_max'] = {v: float(np.abs(OBS[v]['minv'] - O0['minv'])[SETS['mass_exact']].max()) for v in ('Mx', 'C1x', 'C2x')}
    log('sets', out['sets'])
    # ---- networks
    nets = JCNets(); base = J; LG = {}
    for v, Pv in P.items():
        LG[v] = nets.logits(Pv, nn, BASE.get(v, base), axis(Pv, nn))
    LG['M_st'] = nets.logits(P['M'], nn, base, (J['jet_eta'].astype(np.float64), J['jet_phi'].astype(np.float64)))
    LG['id_st'] = nets.logits(P4, nn, base, (J['jet_eta'].astype(np.float64), J['jet_phi'].astype(np.float64)))
    Lraw = {}
    from interp.part_core import inputs
    from interp.p1v_verify_part import fwd
    Lraw = fwd(nets.N.jc['ParT_full'][2], *inputs(J, 'full'))[1]
    out['input_path_check'] = dict(ParT_full_max_abs_logit_raw_vs_id_st=float(np.abs(Lraw - LG['id_st']['ParT_full']).max()),
                                   ParT_full_acc_raw=float((Lraw.argmax(1) == lab).mean()))
    out['unedited'] = {}
    for m in NETS + NULLS:
        L = LG['id'][m]; d = dict(acc_10class=float((L.argmax(1) == lab).mean()), acc_10class_st=float((LG['id_st'][m].argmax(1) == lab).mean()))
        for c, ci in SIG.items():
            sel = (cls == c) | (cls == 'QCD'); s = L[sel, ci] - L[sel, QCD]
            d[f'AUC_{c}_vs_QCD'] = float(H._auc(s, (cls[sel] == c).astype(int))); d[f'eff_argmax_{c}'] = float((L[cls == c].argmax(1) == ci).mean())
        out['unedited'][m] = d; log(f'{m}: ' + json.dumps({k: round(v, 4) for k, v in d.items()}))
    # ---- references
    REF, out['ref_info'] = fit_refs(out)
    RS = {v: ref_scores(REF, OBS[v]) for v in OBS}
    out['ref_unedited'] = {}
    for c in SIG:
        sel = (cls == c) | (cls == 'QCD')
        for nm in REFN: out['ref_unedited'][f'{c}|{nm}'] = float(H._auc(RS['id'][(c, nm)][sel], (cls[sel] == c).astype(int)))
    log('ref unedited AUC on sample', out['ref_unedited'])
    # ---- scores and contrasts
    def score(model, v, c):
        ci = SIG[c]
        if model in NETS + NULLS: L = LG[v][model]; return L[:, ci] - L[:, QCD]
        return RS[v][(c, model)]

    rng = np.random.default_rng(SEED + 20); res = {}; tests = []; tests_ref = []; tests_int = []
    thr = {}
    for m in NETS + NULLS:
        for c in SIG: thr[(m, c)] = float(np.median(score(m, 'id', c)[cls == c]))
    CON = [('M', 'M', None), ('C1', 'C1', None), ('C2', 'C2', None), ('TN', 'TN', None), ('M-C1', 'M', 'C1'), ('M-C2', 'M', 'C2'),
           ('C1-C2', 'C1', 'C2'), ('Mst-M', 'M_st', 'M')]
    CONX = [('Mx', 'Mx', None), ('Mx-C1x', 'Mx', 'C1x'), ('Mx-C2x', 'Mx', 'C2x')]
    PJ = {}
    for sname, smask in SETS.items():
        for samp in SAMPLES:
            scs = [samp] if samp != 'QCD' else list(SIG)
            sel = smask & (cls == samp)
            if sel.sum() < 5: continue
            for c in scs:
                for model in NETS + NULLS + REFN:
                    cons = CONX if sname == 'mass_exact' else CON
                    for cn, a, b in cons:
                        if model in REFN and cn in ('TN', 'Mst-M'): continue
                        if a in ('M_st',) and model in REFN: continue
                        x = score(model, a, c) - (score(model, b, c) if b else score(model, 'id', c)); x = x[sel]
                        mu, ci_ = boot(x, rng); p = perm2(x, rng)
                        key = f'{sname}|{samp}|{c}|{model}|{cn}'; res[key] = dict(mean=mu, ci=ci_, p_perm=p, n=int(sel.sum()))
                        if sname == 'matched': PJ[key] = x.astype(np.float32)
                        if sname == 'all' and b in ('C1', 'C2') and a == 'M':
                            Z = np.column_stack([(OBS[a][o] - OBS[b][o])[sel] for o in ('m', 'tau21')])
                            res[key + '|intercept'] = intercept(x, Z, rng)
                            if model in NETS: tests_int.append(key + '|intercept')
                        if sname == 'matched' and model in NETS and cn in ('M-C1', 'M-C2'):
                            tests.append(key)
                            xr = x - (score('GB3', 'M', c) - score('GB3', cn.split('-')[1], c))[sel]
                            mr, cr = boot(xr, rng); pr = perm2(xr, rng)
                            res[key + '|minus_GB3'] = dict(mean=mr, ci=cr, p_perm=pr); tests_ref.append(key + '|minus_GB3')
                    if model in NETS + NULLS and sname != 'mass_exact':
                        call = lambda v: score(model, v, c)[sel] > thr[(model, c)]
                        res[f'{sname}|{samp}|{c}|{model}|calls'] = dict(id=float(call('id').mean()), M=float(call('M').mean()), C1=float(call('C1').mean()),
                                                                         C2=float(call('C2').mean()), mcnemar_M_vs_C1=mcnemar(call('C1'), call('M')),
                                                                         mcnemar_M_vs_C2=mcnemar(call('C2'), call('M')))
    for fam, lst in (('main', tests), ('vs_GB3', tests_ref), ('intercept', tests_int)):
        adj = holm([res[t]['p_perm' if fam != 'intercept' else 'p_boot'] for t in lst])
        for t, a in zip(lst, adj): res[t]['p_holm'] = float(a)
        out.setdefault('holm', {})[fam] = dict(n_tests=len(lst), n_holm_lt_0p05=int((np.asarray(adj) < 0.05).sum()))
    out['results'] = res; out['thresholds_eS50'] = {f'{m}|{c}': v for (m, c), v in thr.items()}
    for samp in SAMPLES:
        for c in ([samp] if samp != 'QCD' else list(SIG)):
            for model in NETS + NULLS + REFN:
                kk = f'matched|{samp}|{c}|{model}'
                if f'{kk}|M-C1' not in res: continue
                s_ = ' '.join(f"{cn} {res[f'{kk}|{cn}']['mean']:+.3f} [{res[f'{kk}|{cn}']['ci'][0]:+.3f},{res[f'{kk}|{cn}']['ci'][1]:+.3f}]"
                              for cn in ('M', 'C1', 'C2', 'M-C1', 'M-C2') if f'{kk}|{cn}' in res)
                hp = res[f'{kk}|M-C1'].get('p_holm'); log(f'{samp:4s} s_{c:3s} {model:10s} {s_}' + (f' holm(M-C1) {hp:.2g}' if hp is not None else ''))
                ia = res.get(f'all|{samp}|{c}|{model}|M-C1|intercept'); ib = res.get(f'all|{samp}|{c}|{model}|M-C2|intercept')
                if ia: log(f'      all-jets intercept M-C1 {ia["intercept"]:+.3f} [{ia["ci"][0]:+.3f},{ia["ci"][1]:+.3f}] (raw {ia["raw_mean"]:+.3f}); '
                           f'M-C2 {ib["intercept"]:+.3f} [{ib["ci"][0]:+.3f},{ib["ci"][1]:+.3f}] (raw {ib["raw_mean"]:+.3f})')
    # ---- C2ST
    out['c2st'] = {}
    half = np.random.default_rng(SEED + 30).random(len(nn)) < 0.5
    for samp in SAMPLES:
        for v in ('M', 'C1', 'C2', 'Mx'):
            sa = (cls == samp) & has & ~half; sb = (cls == samp) & has & half
            for fs, cols in (('all', OBS_ALL), ('no_colour_flow', [o for o in OBS_ALL if o not in CF])):
                Xa = np.column_stack([O0[o][sa] for o in cols]); Xb = np.column_stack([OBS[v][o][sb] for o in cols])
                out['c2st'][f'{samp}|{v}|{fs}'] = c2st(Xa, Xb)
        Xa = np.column_stack([O0[o][(cls == samp) & has & ~half] for o in OBS_ALL]); Xb = np.column_stack([O0[o][(cls == samp) & has & half] for o in OBS_ALL])
        out['c2st'][f'{samp}|id_vs_id|all'] = c2st(Xa, Xb)
    log('C2ST', {k: round(v, 3) for k, v in out['c2st'].items() if v is not None and np.isfinite(v)})
    np.savez_compressed(RES / f'{TAG}_perjet.npz', cls=cls, **{f'obs|{v}|{o}': np.asarray(a, np.float32) for v, Ov in OBS.items() for o, a in Ov.items()},
                        **{f'lg|{v}|{m}': a.astype(np.float32) for v, d in LG.items() for m, a in d.items()}, **PJ)


# ================================================================================================================ Top arm
def run_top(out):
    from interp.h23_axis_audit import axis, translate
    from interp.h13_pileup_panel import TopModels, top_table, top_J
    import torch
    S = H._sample(PER_TOP, 4105); full = S['full']; nn = np.minimum(S['jet'][:, 5].astype(np.int64), full.shape[1])
    P4 = H._identity(full[..., :4].astype(np.float32), nn).astype(np.float32); y = S['label'].astype(int)
    out['top_sample'] = dict(n=int(len(y)), n_top=int(y.sum()), seed=4105); log(f'top sample {len(y)} jets ({y.sum()} top)')
    O0, fr, G = observables(P4, nn, 'top'); V, out['top_edit_info'] = make_variants(P4, nn, G, 4106); k0 = G['k']; soft = G['soft']
    P = {'id': P4}
    for v in ('M', 'C1', 'C2'): P[v] = to_p4(k0, *V[v])
    a0 = axis(P4, nn); aM = axis(P['M'], nn); P['TN'] = translate(P4, nn, aM[0] - a0[0], wrap(aM[1] - a0[1]))
    OBS = {'id': O0}
    for v in ('M', 'C1', 'C2'): OBS[v] = observables(P[v], nn, 'top')[0]
    ok = lambda v: (np.abs(OBS[v]['m'] - O0['m']) < 1) & (np.abs(OBS[v]['tau21'] - O0['tau21']) < 0.01) & (np.abs(OBS[v]['tau32'] - O0['tau32']) < 0.01)
    ok3 = lambda v: (np.abs(OBS[v]['m'] - O0['m']) < 3) & (np.abs(OBS[v]['tau21'] - O0['tau21']) < 0.03) & (np.abs(OBS[v]['tau32'] - O0['tau32']) < 0.03)
    has = soft.sum(1) > 0; matched = has & ok('M') & ok('C1') & ok('C2'); loose = has & ok3('M') & ok3('C1') & ok3('C2')
    out['top_sets'] = dict(matched_top=int((matched & (y == 1)).sum()), matched_qcd=int((matched & (y == 0)).sum()),
                           loose_top=int((loose & (y == 1)).sum()), loose_qcd=int((loose & (y == 0)).sum()),
                           all_top=int((has & (y == 1)).sum()), all_qcd=int((has & (y == 0)).sum()))
    out['top_obs_change'] = {lab: {v: {o: float((OBS[v][o] - O0[o])[(y == yy) & has].mean()) for o in ('m', 'tau21', 'tau32', 'theta1', 'theta2', 'f_btw', 'mW')}
                                   for v in ('M', 'C1', 'C2')} for lab, yy in (('top', 1), ('qcd', 0))}
    out['top_obs_change']['id_means'] = {lab: {o: float(np.nanmean(O0[o][(y == yy) & has])) for o in ('theta1', 'theta2', 'f_btw', 'mW')} for lab, yy in (('top', 1), ('qcd', 0))}
    log('top sets', out['top_sets'], 'obs change', json.dumps(clean(out['top_obs_change'])))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; TM = TopModels(dev); order = np.random.default_rng(4107).permutation(len(nn))
    SC = {}
    for v, Pv in P.items():
        ax = axis(Pv, nn); SC[v] = TM.score(top_J(top_table(Pv), ax[0], ax[1]), order)
    out['top_unedited'] = {m: dict(AUC=float(H._auc(s, y)), acc=float(((s > 0) == (y == 1)).mean())) for m, s in SC['id'].items()}
    log('top unedited', out['top_unedited'])
    rng = np.random.default_rng(4108); res = {}; tests = []; tests_int = []
    for sname, sm in (('matched', matched), ('matched_loose', loose), ('all', has)):
        for lab, yy in (('top', 1), ('qcd', 0)):
            sel = sm & (y == yy)
            for m in TM.names:
                for cn, a, b in (('M', 'M', None), ('C1', 'C1', None), ('C2', 'C2', None), ('TN', 'TN', None), ('M-C1', 'M', 'C1'), ('M-C2', 'M', 'C2')):
                    x = (SC[a][m] - (SC[b][m] if b else SC['id'][m]))[sel]; mu, ci = boot(x, rng); p = perm2(x, rng)
                    key = f'{sname}|{lab}|{m}|{cn}'; res[key] = dict(mean=mu, ci=ci, p_perm=p, n=int(sel.sum()))
                    if sname == 'matched' and m != 'PFN_permuted' and cn in ('M-C1', 'M-C2'): tests.append(key)
                    if sname == 'all' and cn in ('M-C1', 'M-C2'):
                        Z = np.column_stack([(OBS[a][o] - OBS[b][o])[sel] for o in ('m', 'tau21', 'tau32')])
                        res[key + '|intercept'] = intercept(x, Z, rng)
                        if m != 'PFN_permuted': tests_int.append(key + '|intercept')
                if sname == 'matched':
                    thr = float(np.median(SC['id'][m][y == 1])); call = lambda v: SC[v][m][sel] > thr
                    res[f'{sname}|{lab}|{m}|calls'] = dict(id=float(call('id').mean()), M=float(call('M').mean()), C1=float(call('C1').mean()),
                                                           mcnemar_M_vs_C1=mcnemar(call('C1'), call('M')))
                ii = res.get(f'{sname}|{lab}|{m}|M-C1|intercept')
                log(f'top {sname} {lab} {m:13s} ' + ' '.join(f"{cn} {res[f'{sname}|{lab}|{m}|{cn}']['mean']:+.3f}" for cn in ('M', 'C1', 'C2', 'TN', 'M-C1', 'M-C2'))
                    + (f" | intercept M-C1 {ii['intercept']:+.3f} [{ii['ci'][0]:+.3f},{ii['ci'][1]:+.3f}]" if ii else ''))
    out['top_holm'] = {}
    for fam, lst, pk in (('matched', tests, 'p_perm'), ('intercept', tests_int, 'p_boot')):
        adj = holm([res[t][pk] for t in lst])
        for t, a in zip(lst, adj): res[t]['p_holm'] = float(a)
        out['top_holm'][fam] = dict(n_tests=len(lst), n_holm_lt_0p05=int((np.asarray(adj) < 0.05).sum()))
    out['top_results'] = res


# ================================================================================================================ main
def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = dict(node='H41', smoke=SMOKE, per_class=PER, per_top=PER_TOP, n_ref=NREF, B=B, n_perm=NPERM, seed=SEED,
               soft_def=f'pT < {SOFT_FRAC} pT_jet, dR > {CORE} from every subjet centre')
    arms = os.environ.get('H41_ARMS', 'jc,top').split(',')
    if 'jc' in arms:
        run_jc(out); (RES / f'{TAG}_colour_flow.json').write_text(json.dumps(clean(out), indent=1)); log('saved jc arm')
    if 'top' in arms:
        try:
            run_top(out)
        except Exception as e:  # the secondary arm must not lose the primary results
            import traceback; traceback.print_exc(); out['top_error'] = repr(e)
    (RES / f'{TAG}_colour_flow.json').write_text(json.dumps(clean(out), indent=1)); log('saved')
    try:
        fig(out)
    except Exception as e:
        import traceback; traceback.print_exc(); log('figure failed', repr(e))


def fig(out):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = out['results']
    cells = [('Wqq', 'Wqq', 'W jets\n(W score)'), ('Zqq', 'Zqq', 'Z jets\n(Z score)'), ('Hbb', 'Hbb', 'H→bb jets\n(H score)'),
             ('QCD', 'Wqq', 'QCD jets\n(W score)'), ('QCD', 'Zqq', 'QCD jets\n(Z score)'), ('QCD', 'Hbb', 'QCD jets\n(H score)')]
    models = [('ParT_full', 'ParT full'), ('ParT_kin', 'ParT kin'), ('PN_full', 'ParticleNet full'), ('PN_kin', 'ParticleNet kin'),
              ('ParT_perm', 'ParT, weights permuted'), ('GB5', 'BDT (m, τ21, N, pull, f_btw)'), ('GB3', 'BDT (m, τ21, N)')]
    cols = ['#2f5d8a', '#7aa6d6', '#c0603a', '#e3a07f', '#9a9a9a', '#3c8a3c', '#b5a33a']
    f_, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True); w = 0.11
    for ax, cn, tt in ((axs[0], 'M-C1', 'mirror along prong line minus mirror across it (M − C1)'), (axs[1], 'M-C2', 'mirror minus random rotation (M − C2)')):
        for i, (m, lab) in enumerate(models):
            mu, lo_, hi_ = [], [], []
            for s, c, _ in cells:
                r = R.get(f'matched|{s}|{c}|{m}|{cn}', dict(mean=np.nan, ci=[np.nan, np.nan])); mu.append(r['mean']); lo_.append(r['ci'][0]); hi_.append(r['ci'][1])
            mu, lo_, hi_ = map(lambda a: np.asarray(a, float), (mu, lo_, hi_)); xs = np.arange(len(cells)) + (i - 3) * w
            ax.bar(xs, mu, w, color=cols[i], label=lab, yerr=[mu - lo_, hi_ - mu], capsize=1.5)
        ax.axhline(0, color='k', lw=0.6); ax.set_xticks(range(len(cells))); ax.set_xticklabels([c[2] for c in cells], fontsize=8.5); ax.set_title(tt, fontsize=9.5)
    axs[0].set_ylabel('change in log p(signal)/p(QCD)'); axs[1].legend(fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    n = out['sets']['matched']
    f_.suptitle(f'JetClass, 500–1000 GeV, jets with m and τ21 kept (W {n["Wqq"]}, Z {n["Zqq"]}, H {n["Hbb"]}, QCD {n["QCD"]}): '
                'moving soft radiation from between the prongs to outside', fontsize=10)
    f_.tight_layout(); f_.savefig(RES / f'{TAG}_colour_flow.png', dpi=130); log('figure saved')


if __name__ == '__main__':
    if os.environ.get('H41_FIG'): fig(json.loads((RES / f'{TAG}_colour_flow.json').read_text()))
    else: main()
