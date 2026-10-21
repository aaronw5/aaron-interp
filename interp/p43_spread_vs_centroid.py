"""P43 -- spread about the stored axis vs centroid offset: compensated two-particle edits in ParT and ParticleNet (claim C18, C3).

Question. P39/P39v (C18) found that ParT's W/Z read-out follows M2 = sum_i pT_i r_i^2 about the STORED jet axis (r_i measured from
(jet_eta, jet_phi)), including the part M0 |c|^2 that comes from the pT-weighted centroid c sitting off the axis. Write
  M0 = sum pT,  c = sum pT (deta, dphi) / M0  (centroid about the stored axis),  M2 = sum pT r^2,  M2c = sum pT |r - c|^2,
so M2 = M2c + M0 |c|^2. Does the network read M2 as one quantity (H0), or the centroid-relative spread M2c with the offset only
partly (H1a), or the offset more than the spread (H1b)? Two-particle edits change one term at a time.

Sample. JetClass val_5M WToQQ_120.root and ZToQQ_120.root (the test_20M files end at _119), 10,000 random rows each,
numpy default_rng(4343) (never first-k); jets with < 2 particles dropped; jets with no admissible particle pair (below) dropped
and counted. Inputs rebuilt with part_core.build (eta = jet_eta + sign(jet_eta) deta; px, py, pz, E, jet_pt, jet_energy recomputed).
Models: released ParT_full, ParT_kin, ParticleNet_full (PN_full), ParticleNet_kin (PN_kin); nulls: untrained ParT_full (weaver
initialisation, torch seed 0) and weight-permuted ParT_full (every tensor's entries permuted, torch Generator seed 4343).

Pair. Per jet, among the 8 hardest particles, pairs (a, b) with pT ratio min/max >= 0.6 ("similar pT"); the admissible pair with
the largest pT_a + pT_b is used for every edit of that jet. A pair is admissible when all edits below exist with the two particles
kept at >= 1/2 of their separation (Delta M2 >= -0.75 w, w = d^2 / (1/pT_a + 1/pT_b), d = |x_a - x_b|), the E2 translation is
<= 0.2 rad, and the E5 direction exists.

E2 direction: e = minus the unit vector from the centroid to the pair's pT-weighted barycentre (the pair is moved toward / through
the centroid), chosen because this keeps the M2c change of step 1 small enough to be undone by the step-2 move; for delta >= 0.01
rad >> |c| (median 0.006) M2 rises in either direction. E2rec uses e = -c/|c|.
Edits (only the (deta, dphi) of particles a and b change; pT, mass, PID and track inputs fixed; u = (x_a - x_b)/d):
 E1 spread-only, f in {-10, -5, -2, +2, +5, +10} %: x_a += t u / pT_a, x_b -= t u / pT_b. sum pT dx = 0, so c is unchanged and
    Delta M2 = Delta M2c = 2 t d + t^2 (1/pT_a + 1/pT_b) = f M2 (t solved exactly). Raising f moves the two apart along their line.
 E2 centroid-only, delta in {0.01, 0.02, 0.04} rad along e (above), and 'E2rec' (inward by |c|, c -> 0): step 1 both particles
    translated by s = M0 delta e / (pT_a + pT_b) (c moves by delta e); step 2 an E1-type move on the same pair restores M2c exactly.
    M2 changes by M0 (|c + delta e|^2 - |c|^2).
 E3 = E2 with the stored axis (jet_eta, jet_phi) moved by the centroid shift delta e and deta/dphi of every particle re-expressed
    about the new axis: the lab-frame particles (and the invariant mass) are those of E2, c and M2 about the stored axis are those
    of the unedited jet. Displacement-matched control for E2.
 E4 real angular dilation about the stored axis: every particle's (deta, dphi) times k, k^2 = M2_target / M2 per jet, matched to
    each E1 / E2 edit's predicted Delta log M2 (E4 has no restriction to two particles).
 E5 null, magnitudes of E1 at f = +-10 %: same pair, same |t| (same per-particle displacement), direction v at angle theta to u with
    cos theta = -t (1/pT_a + 1/pT_b) / (2 d) (random sign of sin theta): c and M2 both unchanged.
Read-outs: d = logit_Z - logit_W (P39v's read-out; primary) and q = log(p_W + p_Z) - log(p_QCD) (W/Z-vs-QCD score; secondary).
R(edit) = sum_jets [read-out(edit) - read-out(base)] / sum_jets [read-out(E4 matched) - read-out(base)]; for E3 the E4 matched to
the paired E2; for E5 (pooled) sum(E5+ + E5-) / sum(E4(+10 %) - E4(-10 %)), i.e. relative to the mean size of the +-10 % dilation
responses; abs_ratio = mean |per-jet change| / mean |per-jet change under the matched E4|.
Statistics: 2,000 bootstrap resamples over jets (shared across edits, so differences are paired); sign-flip permutation; exact
McNemar on paired call indicators; Holm within families.
  ./sync.sh && ./part_job.sh c-p43-run 'python -m interp.p43_spread_vs_centroid'
  -> $AI_RESULTS/part/p43_spread_vs_centroid.json, p43_spread_vs_centroid.png
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
SMOKE = '--smoke' in sys.argv
SEED, FIDX = 4343, 120
PER = 400 if SMOKE else 10000
NB = 200 if SMOKE else 2000
NPERM = 300 if SMOKE else 5000
VAL = Path('/j-jepa-vol/JetClass/Pythia/val_5M')
TOPK, RATIO, SMAX, KEEP = 8, 0.6, 0.2, 0.75
FS = [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10]
DELTAS = [0.01, 0.02, 0.04]
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
OUT = RES / f'p43_spread_vs_centroid{"_smoke" if SMOKE else ""}.json'
FIG = RES / f'p43_spread_vs_centroid{"_smoke" if SMOKE else ""}.png'
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ geometry
def moments(pt, de, dp):
    p = pt.astype(np.float64); M0 = p.sum(1); ce, cp = (p * de).sum(1) / M0, (p * dp).sum(1) / M0
    M2 = (p * (de ** 2 + dp ** 2)).sum(1); return M0, ce, cp, M2, M2 - M0 * (ce ** 2 + cp ** 2)


def solve_t(d, A, D):
    """smallest-|t| root of A t^2 + 2 d t - D = 0 (D = target Delta M2); nan where none"""
    disc = d ** 2 + A * D
    return np.where(disc >= 0, (-d + np.sqrt(np.maximum(disc, 0))) / A, np.nan)


def plan_edits(pt, de, dp, rng):
    """choose the pair per jet and the edit parameters (all float64). Returns dict of arrays (n,) and keep mask."""
    n = len(pt); M0, ce, cp, M2, M2c = moments(pt, de, dp); cab = np.hypot(ce, cp)
    eh = np.stack([ce, cp], 1) / np.maximum(cab, 1e-12)[:, None]
    prs = [(a, b) for a in range(TOPK) for b in range(a + 1, TOPK)]; ia = np.array([p[0] for p in prs]); ib = np.array([p[1] for p in prs])
    pa, pb = pt[:, ia].astype(np.float64), pt[:, ib].astype(np.float64)
    xa = np.stack([de[:, ia], dp[:, ia]], -1).astype(np.float64); xb = np.stack([de[:, ib], dp[:, ib]], -1).astype(np.float64)
    pa_s, pb_s = np.maximum(pa, 1e-12), np.maximum(pb, 1e-12)
    dv = xa - xb; d = np.linalg.norm(dv, axis=-1); A = 1 / pa_s + 1 / pb_s; w = d ** 2 / A
    ok = (pa > 0) & (pb > 0) & (np.minimum(pa, pb) / np.maximum(np.maximum(pa, pb), 1e-12) >= RATIO) & (d > 1e-4)
    D = {}; diag = {'pair_similar_pt': ok.any(1).mean()}
    for f in FS:
        Dm = f * M2[:, None]; ok &= Dm >= -KEEP * w; D[('E1', f)] = Dm
    diag['E1'] = ok.any(1).mean()
    for f in (-0.10, 0.10):
        t = solve_t(d, A, f * M2[:, None]); ok &= np.abs(t) * A / (2 * np.maximum(d, 1e-12)) <= 0.95
    diag['E5'] = ok.any(1).mean()
    c2 = np.stack([ce, cp], 1)[:, None, :]                                                                 # (n, 1, 2)
    yp = (pa[..., None] * (xa - c2) + pb[..., None] * (xb - c2)) / (pa_s + pb_s)[..., None]                # pair barycentre - c
    ep = -yp / np.maximum(np.linalg.norm(yp, axis=-1, keepdims=True), 1e-12)                               # toward the centroid
    e2 = {}
    for nm, dl, e in [(f'E2_{x:.2f}', np.full(n, x), ep) for x in DELTAS] + [('E2rec', cab, np.broadcast_to(-eh[:, None, :], ep.shape))]:
        s = (M0 * dl)[:, None, None] / (pa_s + pb_s)[..., None] * e                                          # (n, P, 2)
        ok &= np.linalg.norm(s, axis=-1) <= SMAX; diag[nm + '_translation'] = ok.any(1).mean()
        M2p = M2[:, None] + pa * (((xa + s) ** 2).sum(-1) - (xa ** 2).sum(-1)) + pb * (((xb + s) ** 2).sum(-1) - (xb ** 2).sum(-1))
        cn = c2 + dl[:, None, None] * e                                                                     # (n, P, 2)
        M2cp = M2p - M0[:, None] * (cn ** 2).sum(-1); Dm = M2c[:, None] - M2cp
        ok &= Dm >= -KEEP * w; diag[nm + '_compensation'] = ok.any(1).mean(); e2[nm] = (s, Dm, cn, dl[:, None, None] * e)
    score = np.where(ok, pa + pb, -np.inf); k = score.argmax(1); keep = np.isfinite(score[np.arange(n), k])
    g = lambda a: a[np.arange(n), k]
    P = dict(a=ia[k], b=ib[k], pa=g(pa), pb=g(pb), xa=g(xa), xb=g(xb), d=g(d), A=g(A), u=g(dv) / np.maximum(g(d), 1e-12)[:, None],
             M0=M0, ce=ce, cp=cp, M2=M2, M2c=M2c, cab=cab, keep=keep, sgn5=rng.choice([-1.0, 1.0], n),
             diag={kk: float(v) for kk, v in diag.items()})
    for f in FS: P[('E1', f)] = solve_t(P['d'], P['A'], f * M2)
    for nm, (s, Dm, cn, dc) in e2.items():
        P[(nm, 's')] = g(s); P[(nm, 't')] = solve_t(P['d'], P['A'], g(Dm)); P[(nm, 'dc')] = g(dc)
        P[(nm, 'M2new')] = M2c + M0 * (g(cn) ** 2).sum(1)
    return P


def displaced(K, P, name):
    """per-particle (deta, dphi) arrays for edit `name`, plus the displacement vectors of a and b"""
    pt, de, dp, ms = K; de2, dp2 = de.astype(np.float64).copy(), dp.astype(np.float64).copy(); n = len(pt); r = np.arange(n)
    u = P['u']; pa, pb = P['pa'][:, None], P['pb'][:, None]
    if name.startswith('E1'):
        f = float(name.split('_')[1]); t = P[('E1', f)][:, None]; va, vb = t * u / pa, -t * u / pb
    elif name.startswith('E5'):
        f = 0.10 if name.endswith('+') else -0.10; t = P[('E1', f)]; ct = np.clip(-t * P['A'] / (2 * P['d']), -1, 1)
        st = P['sgn5'] * np.sqrt(1 - ct ** 2); v = np.stack([ct * u[:, 0] - st * u[:, 1], st * u[:, 0] + ct * u[:, 1]], 1)
        va, vb = t[:, None] * v / pa, -t[:, None] * v / pb
    else:
        nm = name.replace('E3', 'E2'); s = P[(nm, 's')]; t = P[(nm, 't')][:, None]; va, vb = s + t * u / pa, s - t * u / pb
    de2[r, P['a']] += va[:, 0]; dp2[r, P['a']] += va[:, 1]; de2[r, P['b']] += vb[:, 0]; dp2[r, P['b']] += vb[:, 1]
    return de2, dp2, va, vb


def make(J, K, P, name, k=None):
    from interp.part_core import build
    pt, de, dp, ms = K; r = (pt > 0)
    if name == 'base': return build(J, pt, de, dp, ms)
    if name == 'E4':
        k = k[:, None]; return build(J, pt, (de * k * r).astype(np.float32), (dp * k * r).astype(np.float32), ms)
    de2, dp2, _, _ = displaced(K, P, name)
    if name.startswith('E3'):
        dc = P[(name.replace('E3', 'E2'), 'dc')]; sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)
        J = dict(J); J['jet_eta'] = (J['jet_eta'] + sg * dc[:, 0]).astype(np.float32); J['jet_phi'] = (J['jet_phi'] + dc[:, 1]).astype(np.float32)
        de2 = de2 - dc[:, :1]; dp2 = dp2 - dc[:, 1:]
    return build(J, pt, np.where(r, de2, 0).astype(np.float32), np.where(r, dp2, 0).astype(np.float32), ms)


# ------------------------------------------------------------------------------------------------ models
def null_models(dev):
    import torch
    from interp.part_core import ParTNetwork
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import CFG, FEATURES
    torch.manual_seed(0); un = ParticleTransformer(input_dim=len(FEATURES['full']), **CFG).to(dev).eval()
    pm = ParTNetwork('full').model.to(dev).eval(); g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for p in pm.parameters():
            if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
    for m in (un, pm):
        for p in m.parameters(): p.requires_grad_(False)
    return {'ParT_full_untrained': dict(kind='part', feat='full', model=un), 'ParT_full_permuted': dict(kind='part', feat='full', model=pm)}


# ------------------------------------------------------------------------------------------------ statistics
def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def boot_p(diff_boot, est):
    """two-sided bootstrap p for H0: value = 0 (percentile method, centred)"""
    b = np.asarray(diff_boot, float); b = b[np.isfinite(b)]
    if not len(b): return None
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / len(b)))


def signflip_p(x, rng):
    x = np.asarray(x, float); t = abs(x.mean()); n = len(x); cnt = 0; done = 0
    while done < NPERM:
        b = min(100, NPERM - done); sg = rng.choice([-1.0, 1.0], size=(b, n)); cnt += int((np.abs(sg @ x / n) >= t).sum()); done += b
    return float((1 + cnt) / (1 + NPERM))


def mcnemar(a, b):
    from scipy.stats import binomtest
    x = int((a & ~b).sum()); z = int((~a & b).sum())
    return dict(only_first=x, only_second=z, p=(float(binomtest(x, x + z, 0.5).pvalue) if x + z else 1.0))


# ------------------------------------------------------------------------------------------------ main
def main():
    import torch
    from interp.p39v_verify import read_rand, load_nets, runner
    from interp.part_core import sample, kin, build, jmass, cat, sub
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev)
    M, FEATURES = load_nets(dev); M.update(null_models(dev)); names = list(M); run = runner(M, FEATURES, dev)
    CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']; iQ, iW, iZ = 0, 7, 6
    J1 = sample(100 if SMOKE else 1000, 3000); y1 = J1['label'].astype(int); L1 = run(build(J1, *kin(J1)), names)
    R = dict(node='P43', seed=SEED, files=f'val_5M {{WToQQ,ZToQQ}}_{FIDX}.root', per_class=PER, nb=NB, nperm=NPERM,
             readouts=dict(d='logit_Z - logit_W (primary)', q='log(p_W + p_Z) - log(p_QCD)'),
             pair_rule=dict(topk=TOPK, pt_ratio_min=RATIO, max_translation_rad=SMAX, min_sep_frac='Delta M2 >= -0.75 w'),
             acc_P1={k: float((L1[k].argmax(1) == y1).mean()) for k in names}, n_P1=int(len(y1)))
    log('P1 accuracy (10 classes, 10,000 jets)', R['acc_P1']); del J1, L1
    rng = np.random.default_rng(SEED); Js = []
    for c, pre in (('Wqq', 'WToQQ'), ('Zqq', 'ZToQQ')):
        J, rows = read_rand(VAL / f'{pre}_{FIDX}.root', PER, rng); assert np.all(J['label'] == CL.index(c)); Js.append(J); log(c, len(rows), 'jets read')
    J = cat(Js); del Js; n0 = len(J['label']); K = kin(J)
    Lall = run(build(J, *K), names); yall = J['label'].astype(int)
    R['acc_all_read_WZ'] = {k: float((Lall[k].argmax(1) == yall).mean()) for k in names}; log('unedited accuracy, all W/Z jets read', R['acc_all_read_WZ'])
    P = plan_edits(*K[:3], rng); keep = P['keep']
    R['acc_dropped_WZ'] = {k: float((Lall[k].argmax(1) == yall)[~keep].mean()) for k in names}; del Lall
    R['n_read'] = int(n0); R['n_kept'] = int(keep.sum()); R['kept_by_class'] = {c: int(keep[J['label'] == CL.index(c)].sum()) for c in ('Wqq', 'Zqq')}
    R['pair_constraint_pass'] = P['diag']; log(f'pairs: kept {keep.sum()} / {n0}; cumulative pass fractions', P['diag'])
    J = sub(J, np.where(keep)[0]); K = kin(J); P = plan_edits(*K[:3], np.random.default_rng(SEED + 1)); assert P['keep'].all()
    y = J['label'].astype(int); n = len(y); isW = y == iW
    # ---- edit list with matched dilations
    E1n = [f'E1_{f:+.2f}' for f in FS]; E2n = [f'E2_{x:.2f}' for x in DELTAS] + ['E2rec']; E3n = [e.replace('E2', 'E3') for e in E2n]
    dlogM2 = {e: np.full(n, np.log1p(f)) for e, f in zip(E1n, FS)}
    for e in E2n: dlogM2[e] = np.log(P[(e, 'M2new')] / P['M2']); dlogM2[e.replace('E2', 'E3')] = np.zeros(n)
    dlogM2['E5+'] = np.zeros(n); dlogM2['E5-'] = np.zeros(n)
    match = {e: e for e in E1n + E2n}; match.update({e3: e2 for e3, e2 in zip(E3n, E2n)}); match['E5+'] = 'E1_+0.10'; match['E5-'] = 'E1_-0.10'
    EDITS = ['base'] + E1n + E2n + E3n + ['E5+', 'E5-']
    L, geo = {}, {}
    Jb = make(J, K, P, 'base'); mb = jmass(Jb).astype(np.float64); Kb = kin(Jb); M0b, ceb, cpb, M2b, M2cb = moments(*Kb[:3])
    acc = {k: None for k in names}
    for e in EDITS + [f'E4|{e}' for e in E1n + E2n]:
        if e.startswith('E4|'):
            src = e[3:]; Je = make(J, K, P, 'E4', k=np.sqrt(np.exp(dlogM2[src])))
        else:
            Je = make(J, K, P, e)
        Ke = kin(Je); M0e, cee, cpe, M2e, M2ce = moments(*Ke[:3]); me = jmass(Je).astype(np.float64)
        _, _, va, vb = displaced(K, P, e) if e[:2] in ('E1', 'E2', 'E3', 'E5') else (None, None, np.zeros((n, 2)), np.zeros((n, 2)))
        geo[e] = dict(dlogM2_realised=float(np.mean(np.log(M2e / M2b))), dlogM2_planned=float(np.mean(dlogM2.get(e, np.log(M2e / M2b)))),
                      dlogM2c=float(np.mean(np.log(M2ce / M2cb))), dc_median=float(np.median(np.hypot(cee - ceb, cpe - cpb))),
                      max_abs_dc=float(np.max(np.hypot(cee - ceb, cpe - cpb))),
                      dlog_minv_mean=float(np.mean(np.log(np.maximum(me, 1e-3) / np.maximum(mb, 1e-3)))),
                      dlog_minv_median=float(np.median(np.log(np.maximum(me, 1e-3) / np.maximum(mb, 1e-3)))),
                      disp_a_median=float(np.median(np.linalg.norm(va, axis=1))), disp_b_median=float(np.median(np.linalg.norm(vb, axis=1))),
                      max_err_dlogM2=float(np.max(np.abs(np.log(M2e / M2b) - dlogM2[e]))) if e in dlogM2 else None)
        if e.startswith('E3'): geo[e]['max_abs_dlog_minv_vs_E2'] = None
        L[e] = run(Je, names); del Je
        if e == 'base':
            acc = {k: float((L[e][k].argmax(1) == y).mean()) for k in names}; R['acc_sample_WZ'] = acc
            R['acc_sample_by_class'] = {k: {c: float((L[e][k].argmax(1) == y)[y == CL.index(c)].mean()) for c in ('Wqq', 'Zqq')} for k in names}
            log('unedited accuracy on the P43 W/Z sample (10-class argmax)', acc)
        log(f'{e}: dlogM2 {geo[e]["dlogM2_realised"]:+.4f} dlogM2c {geo[e]["dlogM2c"]:+.4f} |dc| med {geo[e]["dc_median"]:.4f} '
            f'dlog m_inv {geo[e]["dlog_minv_mean"]:+.4f} disp a/b {geo[e]["disp_a_median"]:.4f}/{geo[e]["disp_b_median"]:.4f}')
    for e2, e3 in zip(E2n, E3n):
        Ja, Jc_ = make(J, K, P, e2), make(J, K, P, e3)
        geo[e3]['max_abs_dlog_minv_vs_E2'] = float(np.max(np.abs(np.log(np.maximum(jmass(Ja), 1e-3) / np.maximum(jmass(Jc_), 1e-3)))))
        geo[e3]['median_axis_shift'] = float(np.median(np.linalg.norm(P[(e2, 'dc')], axis=1))); del Ja, Jc_
    R['geometry'] = geo; R['sample_geometry'] = dict(median_cabs=float(np.median(P['cab'])), median_M2_over_M0=float(np.median(P['M2'] / P['M0'])),
                                                      median_pair_pt_frac=float(np.median((P['pa'] + P['pb']) / P['M0'])),
                                                      median_pair_sep=float(np.median(P['d'])), pair_rank_a=np.bincount(P['a'], minlength=TOPK).tolist(),
                                                      pair_rank_b=np.bincount(P['b'], minlength=TOPK).tolist(),
                                                      median_minv=float(np.median(mb)))
    # ---- statistics
    rb = np.random.default_rng(SEED + 2); C = np.zeros((NB, n), np.float32)
    for b in range(NB): C[b] = np.bincount(rb.integers(0, n, n), minlength=n)
    top20 = P['cab'] >= np.quantile(P['cab'], 0.8)
    rd = dict(d=lambda A: (A[:, iZ] - A[:, iW]).astype(np.float64),
              q=lambda A: (np.logaddexp(A[:, iW], A[:, iZ]) - A[:, iQ]).astype(np.float64) - 0.0)
    # q via log-softmax: log p_W+p_Z - log p_QCD = logsumexp(l_W, l_Z) - l_QCD (normaliser cancels)
    R['models'] = {}; fam_h0, fam_h0c, fam_rl, fam_e3 = [], [], [], []
    for k in names:
        Rk = {}
        for ro, fn in rd.items():
            base = fn(L['base'][k]); dlt = {e: fn(L[e][k]) - base for e in EDITS[1:]}
            dlt.update({f'E4|{e}': fn(L[f'E4|{e}'][k]) - base for e in E1n + E2n})
            sums = {e: (v.sum(), C @ v) for e, v in dlt.items()}
            out = {}
            for e in EDITS[1:]:
                den_e = f'E4|{match[e]}'; num, den = sums[e], sums[den_e]
                out[e] = dict(dmean=float(dlt[e].mean()), dmean_ci=ci(sums[e][1] / n), e4_dmean=float(dlt[den_e].mean()),
                              e4_dmean_ci=ci(sums[den_e][1] / n), R=float(num[0] / den[0]), R_ci=ci(num[1] / den[1]))
            # pooled R
            def pooled(es):
                nu = sum(sums[e][0] for e in es); de_ = sum(sums[f'E4|{match[e]}'][0] for e in es)
                nb_ = sum(sums[e][1] for e in es); db_ = sum(sums[f'E4|{match[e]}'][1] for e in es); return nu / de_, nb_ / db_
            pools = {'E1_raise': [e for e in E1n if '+' in e], 'E1_lower': [e for e in E1n if '-' in e], 'E2_raise': E2n[:3], 'E2rec': ['E2rec'],
                     'E3_all': E3n[:3], 'E5': ['E5+', 'E5-']}
            pv = {nm: pooled(es) for nm, es in pools.items() if nm != 'E5'}
            nu = sums['E5+'][0] + sums['E5-'][0]; de_ = sums['E4|E1_+0.10'][0] - sums['E4|E1_-0.10'][0]
            pv['E5'] = (nu / de_, (sums['E5+'][1] + sums['E5-'][1]) / (sums['E4|E1_+0.10'][1] - sums['E4|E1_-0.10'][1]))
            # per-jet response size: mean |delta| relative to the mean |delta| of the matched dilation
            out['abs_ratio'] = {e: dict(v=float(np.abs(dlt[e]).mean() / np.abs(dlt[f'E4|{match[e]}']).mean()), mean_abs=float(np.abs(dlt[e]).mean()))
                                for e in EDITS[1:]}
            out['pooled'] = {nm: dict(R=float(v[0]), R_ci=ci(v[1])) for nm, v in pv.items()}
            # H0: R(E1 raise) = R(E2 raise)
            # displacement-corrected E2: (E2 - E3) / E4, i.e. the part of the E2 response that needs the stored-axis offset
            e23n = sum(sums[e][0] - sums[e.replace('E2', 'E3')][0] for e in E2n[:3]); e23b = sum(sums[e][1] - sums[e.replace('E2', 'E3')][1] for e in E2n[:3])
            e4n = sum(sums[f'E4|{e}'][0] for e in E2n[:3]); e4b = sum(sums[f'E4|{e}'][1] for e in E2n[:3]); pv['E2_minus_E3'] = (e23n / e4n, e23b / e4b)
            d2 = pv['E1_raise'][1] - pv['E2_minus_E3'][1]
            out['pooled']['E2_minus_E3'] = dict(R=float(pv['E2_minus_E3'][0]), R_ci=ci(pv['E2_minus_E3'][1]))
            out['H0c_E1raise_minus_E2E3'] = dict(v=float(pv['E1_raise'][0] - pv['E2_minus_E3'][0]), ci=ci(d2), p_boot=boot_p(d2, None))
            dif = pv['E1_raise'][1] - pv['E2_raise'][1]
            out['H0_E1raise_minus_E2raise'] = dict(v=float(pv['E1_raise'][0] - pv['E2_raise'][0]), ci=ci(dif), p_boot=boot_p(dif, None))
            # raising vs lowering per magnitude and pooled
            rl = {}
            for f in (0.02, 0.05, 0.10):
                ep, em = f'E1_{f:+.2f}', f'E1_{-f:+.2f}'
                bp_ = sums[ep][1] / sums[f'E4|{ep}'][1] - sums[em][1] / sums[f'E4|{em}'][1]
                rl[f'{f:.2f}'] = dict(R_raise=out[ep]['R'], R_lower=out[em]['R'], diff=out[ep]['R'] - out[em]['R'], ci=ci(bp_), p_boot=boot_p(bp_, None))
            dpool = pv['E1_raise'][1] - pv['E1_lower'][1]
            rl['pooled'] = dict(diff=float(pv['E1_raise'][0] - pv['E1_lower'][0]), ci=ci(dpool), p_boot=boot_p(dpool, None))
            # same for the E4 dilation itself (raising vs lowering asymmetry of the reference): slope ratio d(+f)/(-d(-f))
            rl['E4_asym'] = {f'{f:.2f}': dict(v=float(-dlt[f'E4|E1_{f:+.2f}'].mean() / dlt[f'E4|E1_{-f:+.2f}'].mean()),
                                              ci=ci(-sums[f'E4|E1_{f:+.2f}'][1] / sums[f'E4|E1_{-f:+.2f}'][1])) for f in (0.02, 0.05, 0.10)}
            out['raise_vs_lower'] = rl
            # dose linearity: mean response vs Delta log M2 for E1 and E4, quadratic fit; and R vs dose slope
            x = np.array([np.log1p(f) for f in FS]); yE1 = np.array([dlt[e].mean() for e in E1n]); yE4 = np.array([dlt[f'E4|{e}'].mean() for e in E1n])
            bE1 = np.stack([sums[e][1] / n for e in E1n], 1); bE4 = np.stack([sums[f'E4|{e}'][1] / n for e in E1n], 1)
            V = np.stack([x, x ** 2], 1); fit = lambda Y: np.linalg.lstsq(V, Y.T, rcond=None)[0]
            f1, f4 = fit(yE1[None]), fit(yE4[None]); bf1, bf4 = fit(bE1), fit(bE4)
            Rd = np.array([out[e]['R'] for e in E1n]); bR = bE1 / bE4; Vr = np.stack([np.ones_like(x), x], 1)
            fr = np.linalg.lstsq(Vr, Rd, rcond=None)[0]; bfr = np.linalg.lstsq(Vr, bR.T, rcond=None)[0]
            out['dose'] = dict(x_dlogM2=x.tolist(), E1_mean=yE1.tolist(), E4_mean=yE4.tolist(), E1_lin=float(f1[0, 0]), E1_quad=float(f1[1, 0]),
                               E1_quad_ci=ci(bf1[1]), E4_lin=float(f4[0, 0]), E4_quad=float(f4[1, 0]), E4_quad_ci=ci(bf4[1]),
                               R_vs_dlogM2_intercept=float(fr[0]), R_vs_dlogM2_slope=float(fr[1]), R_slope_ci=ci(bfr[1]))
            # E3 vs 0 and E3/E2, E5 vs 0 (sign-flip permutation)
            prs = np.random.default_rng(SEED + 3)
            e3d = sum(dlt[e] for e in E3n[:3]); e2d = sum(dlt[e] for e in E2n[:3])
            out['E3_over_E2'] = dict(v=float(e3d.sum() / e2d.sum()), ci=ci(sum(sums[e][1] for e in E3n[:3]) / sum(sums[e][1] for e in E2n[:3])))
            if ro == 'd':
                out['p_E3_eq_0'] = signflip_p(e3d, prs); out['p_E5_eq_0'] = signflip_p(dlt['E5+'] + dlt['E5-'], prs)
                out['p_E2rec_eq_0'] = signflip_p(dlt['E2rec'], prs)
            # top-20 % |c| subset for E2rec
            ix = np.where(top20)[0]; Ct = C[:, ix]
            nu, de_ = dlt['E2rec'][ix], dlt['E4|E2rec'][ix]
            out['E2rec_top20'] = dict(n=int(len(ix)), R=float(nu.sum() / de_.sum()), R_ci=ci((Ct @ nu) / (Ct @ de_)), dmean=float(nu.mean()),
                                      pred_dlogM2=float(dlogM2['E2rec'][ix].mean()))
            Rk[ro] = out
            if ro == 'd' and not k.endswith(('untrained', 'permuted')):
                fam_h0.append((k, out['H0_E1raise_minus_E2raise']['p_boot'])); fam_h0c.append((k, out['H0c_E1raise_minus_E2E3']['p_boot'])); fam_rl.append((k, rl['pooled']['p_boot'])); fam_e3.append((k, out['p_E3_eq_0']))
        # calls
        c0 = L['base'][k].argmax(1); calls = {}
        for e in EDITS[1:] + [f'E4|{e}' for e in E1n + E2n]:
            ce_ = L[e][k].argmax(1); ch = ce_ != c0; zW0, zW1 = (c0 == iZ) & isW, (ce_ == iZ) & isW
            calls[e] = dict(changed=float(ch.mean()), changed_ci=ci((C @ ch.astype(np.float32)) / n),
                            WtoZ_base=float(zW0[isW].mean()), WtoZ_edit=float(zW1[isW].mean()), mcnemar_WtoZ=mcnemar(zW1[isW], zW0[isW]))
        for e in EDITS[1:]:
            m_ = f'E4|{match[e]}'; calls[e]['mcnemar_changed_vs_E4'] = mcnemar(L[e][k].argmax(1) != c0, L[m_][k].argmax(1) != c0)
        Rk['calls'] = calls; R['models'][k] = Rk
        o = Rk['d']
        log(f'{k}: acc {acc[k]:.4f} | R E1+ {o["pooled"]["E1_raise"]["R"]:.3f} {np.round(o["pooled"]["E1_raise"]["R_ci"], 3)} '
            f'E1- {o["pooled"]["E1_lower"]["R"]:.3f} {np.round(o["pooled"]["E1_lower"]["R_ci"], 3)} E2+ {o["pooled"]["E2_raise"]["R"]:.3f} '
            f'{np.round(o["pooled"]["E2_raise"]["R_ci"], 3)} E2rec {o["pooled"]["E2rec"]["R"]:.3f} E3 {o["pooled"]["E3_all"]["R"]:.3f} '
            f'{np.round(o["pooled"]["E3_all"]["R_ci"], 3)} E5 {o["pooled"]["E5"]["R"]:.3f} {np.round(o["pooled"]["E5"]["R_ci"], 3)} | '
            f'H0 diff {o["H0_E1raise_minus_E2raise"]["v"]:+.3f} p {o["H0_E1raise_minus_E2raise"]["p_boot"]}')
    for fam, key in ((fam_h0, 'H0_holm'), (fam_h0c, 'H0c_holm'), (fam_rl, 'raise_lower_holm'), (fam_e3, 'E3_holm')):
        for (k, _), ph in zip(fam, holm([f[1] for f in fam])): R['models'][k]['d'][key] = ph
    # ---- figure data: one real W jet
    wi = np.where(isW & (mb > 70) & (mb < 90))[0]; j = int(wi[np.argmax(P['d'][wi] * (P['pa'][wi] + P['pb'][wi]) / P['M0'][wi])]) if len(wi) else int(np.where(isW)[0][0])
    fj = dict(index=j, jet_pt=float(J['jet_pt'][j]), minv=float(mb[j]), pt=K[0][j].tolist(), deta=K[1][j].tolist(), dphi=K[2][j].tolist(), a=int(P['a'][j]), b=int(P['b'][j]),
              c=[float(P['ce'][j]), float(P['cp'][j])])
    for e in ('E1_+0.10', 'E1_-0.10', 'E2_0.04', 'E5+'):
        de2, dp2, _, _ = displaced(K, P, e); Je = make(J, K, P, e); fj[e] = dict(deta=de2[j].tolist(), dphi=dp2[j].tolist(), minv=float(jmass(sub(Je, [j]))[0]))
        Mx = moments(K[0][j:j + 1], de2[j:j + 1], dp2[j:j + 1]); fj[e].update(c=[float(Mx[1][0]), float(Mx[2][0])], dlogM2=float(np.log(Mx[3][0] / P['M2'][j])),
                                                                             dlogM2c=float(np.log(Mx[4][0] / P['M2c'][j])))
    R['fig_jet'] = fj
    np.savez_compressed(RES / f'p43_logits{"_smoke" if SMOKE else ""}.npz', y=y, **{f'{e}__{k}': L[e][k] for e in L for k in names},
                        **{f'dlogM2__{e}': v for e, v in dlogM2.items()}, cabs=P['cab'], M2=P['M2'], M0=P['M0'], minv=mb)
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    log('wrote', OUT)
    try:
        plot(R)
    except Exception as ex:  # noqa: BLE001
        log('plot failed', repr(ex))


def plot(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fj = R['fig_jet']; pt = np.array(fj['pt']); m = pt > 0; de, dp = np.array(fj['deta'])[m], np.array(fj['dphi'])[m]; pt = pt[m]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
    for a_, e, ttl in ((ax[0], 'E1_+0.10', 'E1: spread +10 %, centroid fixed'), (ax[1], 'E2_0.04', 'E2: centroid +0.04 rad, spread about centroid fixed')):
        a_.scatter(de, dp, s=4 + 300 * pt / pt.max(), c='0.6', alpha=0.6, lw=0)
        ed = fj[e]; a_.plot(0, 0, '+', c='k', ms=14, mew=2, label='stored jet axis')
        a_.plot(*fj['c'], 'x', c='C0', ms=11, mew=2, label='pT centroid before'); a_.plot(*ed['c'], 'x', c='C3', ms=11, mew=2, label='pT centroid after')
        for i, col in ((fj['a'], 'C1'), (fj['b'], 'C2')):
            x0, y0, x1, y1 = fj['deta'][i], fj['dphi'][i], ed['deta'][i], ed['dphi'][i]
            a_.scatter([x0], [y0], s=4 + 300 * fj['pt'][i] / pt.max(), facecolors='none', edgecolors=col, lw=1.5)
            a_.scatter([x1], [y1], s=4 + 300 * fj['pt'][i] / pt.max(), c=col, lw=0)
            a_.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
        a_.set_title(f'{ttl}\nm_inv {fj["minv"]:.1f} -> {ed["minv"]:.1f} GeV; dlog M2 {ed["dlogM2"]:+.3f}, dlog M2c {ed["dlogM2c"]:+.3f}', fontsize=9.5)
        a_.set_xlabel('Δη from stored axis [rad]'); a_.set_ylabel('Δφ from stored axis [rad]'); a_.set_aspect('equal', 'datalim')
    ax[0].legend(fontsize=8, loc='lower left')
    a_ = ax[2]; ks = [k for k in R['models']]; groups = [('E1_raise', 'E1 spread up'), ('E1_lower', 'E1 spread down'), ('E2_raise', 'E2 centroid out'),
                                                        ('E3_all', 'E3 = E2, axis moved'), ('E5', 'E5 random, M2 fixed')]
    w = 0.8 / len(ks)
    for i, k in enumerate(ks):
        po = R['models'][k]['d']['pooled']; v = [po[g]['R'] for g, _ in groups]; lo = [po[g]['R'] - po[g]['R_ci'][0] for g, _ in groups]
        hi = [po[g]['R_ci'][1] - po[g]['R'] for g, _ in groups]
        a_.bar(np.arange(len(groups)) + (i - len(ks) / 2 + 0.5) * w, np.clip(v, -0.5, 1.6), w, yerr=[lo, hi], label=k, capsize=2, error_kw=dict(lw=0.8))
    a_.axhline(1, c='k', lw=0.6, ls='--'); a_.axhline(0, c='k', lw=0.6); a_.set_ylim(-0.5, 1.6)
    a_.set_xticks(range(len(groups))); a_.set_xticklabels([g[1] for g in groups], fontsize=8.5, rotation=12)
    a_.set_ylabel('R = Δ(logit_Z − logit_W) under edit / under matched dilation'); a_.legend(fontsize=7.5, ncol=2)
    a_.set_title('Response relative to a real angular dilation with the same Δlog M2\n(nulls clipped to [−0.5, 1.6]; E3, E5 to the mean |response| of the ±10 % dilations)', fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG, dpi=130); log('wrote', FIG)


if __name__ == '__main__':
    main()
