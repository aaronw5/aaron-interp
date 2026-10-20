"""P27 — per-particle kernel of the axis response (claims A8, C12, P12; exploratory).

Question. C12: at the jet level ParT reads the axis moment M2 = sum pT r^2 (r = distance of a particle from the stored jet axis in
(deta, dphi)). Per particle, the M2 reading predicts that moving particle i outward by dr changes the W-vs-Z read-out in
proportion to dM2/M2 = pT_i ((r_i + dr)^2 - r_i^2) / M2 (about 2 pT_i r_i dr / M2), with one gain per jet. JEDI-linear (loop A/B)
builds its mass from pT-rank windows and hinges on pT share / radius; that reading predicts per-rank-window or threshold weights.

Models: ParT_full, ParT_kin (released), ParticleNet_full, ParticleNet_kin (released), controls: ParT_full architecture untrained
(weaver initialisation, torch seed 2701) and ParT_kin with the output rows of every weight tensor permuted (one permutation per
layer, seed 2701; P18 recipe).
Sample: JetClass test_20M WToQQ_106 and ZToQQ_106, PER random rows each (numpy default_rng(2701), never first-k); jets with < 2
eligible particles dropped. Unedited accuracy (10-class argmax) reported on these rows and on the P1 sample (rows 3,000-4,000 of
each class file _100, N = 10,000).
Build: part_core.build (eta = jet_eta + sign(jet_eta) deta, phi = jet_phi + dphi; four-vectors, deltaR, deta, dphi, energies,
jet_pt/jet_energy and hence pair features recomputed); pT of every particle kept, so the pT ordering does not change.
Edits (dr = 0.01 rad), on 'eligible' particles (pT share z_i = pT_i / sum pT > 0.5 %):
  rad_i    particle i moved radially away from the stored axis by dr.
  az_i     control (a): particle i rotated about the axis at fixed r_i, chord length dr, random sign s_i = +-1.
  pair_ij  particle i and j both moved radially by dr (NPAIR random pairs per jet): additivity.
  spread   control (b): the same total displacement dr spread over KSP random eligible particles (dr / KSP each), NSP draws.
  all      every eligible particle moved radially by dr together (calibration of sum_i d_i).
  base2    the unedited jet again in another batch (numerical floor).
  radin_i  particle i moved radially toward the axis by dr (through it when r_i < dr): per-particle odd part
           (d_out - d_in) / 2 against x_odd = (x_out - x_in) / 2 (first order 2 pT_i r_i dr / M2) and even part (d_out + d_in) / 2.
Read-out: d = change of logit_Z - logit_W (Zqq, Wqq of the 10 outputs) vs the rebuilt unedited jet.
Fits of d_i (rad edits) against per-particle forms (global coefficients, no intercept):
  M2        x = dM2/M2 (1 parameter)
  M2_ctx    x * (1, s, s^2, s^3), s = base (logit_Z - logit_W) / 5 clipped to +-2 (context-dependent gain, 4)
  M2_rankwin x * 1[rank in window], windows 1, 2, 3, 4-6, 7-10, 11-20, 21+ (7)
  rankwin_zr per window (z, r, z r) (21)
  hinge_z   (z - t)_+ and (z - t)_+ r for t in 0.005, 0.01, 0.02, 0.05, 0.1, 0.2 (12)
  hinge_r   z 1[r > t] and 1[r > t] for t in 0.05, 0.1, 0.2, 0.3 (8; derivative of JEDI's sum (r - t)_+ objects)
  pt_only   z, z^2, sqrt z (3);  r_only  r, r^2, sqrt r (3);  const  1 (1)
  gbdt      gradient-boosted trees on (z, r, rank, x, M0, M2, m_jet, n, s): flexible benchmark, not a low-parameter form.
  Held-out R^2 by 5-fold split over jets (SST about the pooled mean of d); 95 % bootstrap over jets; BIC on the full-data fit;
  H0 R^2(form) = R^2(M2): bootstrap two-sided p, Holm over forms x models. Shape metric: R^2 when each held-out jet gets its own
  gain on the form's prediction, fitted on even-rank particles and scored on odd-rank particles.
Additivity: A = 1 - sum (d_ij - d_i - d_j)^2 / sum (d_ij - mean)^2; interaction share sum (d_ij - d_i - d_j)^2 / sum d_ij^2.
Calibration: sum_jets sum_i d_i / sum_jets d_all. Azimuthal control: mean |d_az| / mean |d_rad|, and mean s_i d_az (odd part).

  ./sync.sh && ./part_job.sh c-p27-run 'python -m interp.p27_particle_kernel'
  -> $AI_RESULTS/part/p27_particle_kernel[_smoke].json, .npz (per-edit d), p27_particle_kernel.png
  python -m interp.p27_particle_kernel --analyse   (re-run the analysis from the npz; CPU)
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--analyse', action='store_true')
ap.add_argument('--plot', action='store_true')
ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED = 2701
PER = 40 if SMOKE else 2000
NB = 100 if SMOKE else 1000
DR = 0.01
ZMIN = 0.005
NPAIR = 6
KSP = 5
NSP = 2
BS = 256
JCH = 64
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
FILES = dict(Wqq='WToQQ_106.root', Zqq='ZToQQ_106.root')
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = '_smoke' if SMOKE else ''
FN = OUTD / f'p27_particle_kernel{TAG}.json'
NPZ = OUTD / f'p27_particle_kernel{TAG}.npz'
ETYPES = ['base', 'rad', 'az', 'pair', 'spread', 'all', 'base2', 'radin']
WINDOWS = [(0, 1), (1, 2), (2, 3), (3, 6), (6, 10), (10, 20), (20, 999)]
TZ = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
TR = [0.05, 0.1, 0.2, 0.3]
MODELS = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin', 'ParT_untr', 'ParT_kin_perm']
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


def st(v, bs, v0=0.0):
    from scipy.stats import norm
    bs_ = np.asarray(bs, float); sd = float(np.nanstd(bs_))
    return dict(v=float(v), ci=ci(bs), p0=p_two(bs, v0), p_norm=(float(2 * norm.sf(abs(float(v) - v0) / sd)) if sd > 0 else None))


def strip(o):
    if isinstance(o, dict): return {str(k): strip(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return strip(o.tolist())
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


# ------------------------------------------------------------------------------------------------ data
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


# ------------------------------------------------------------------------------------------------ models
def load_models():
    import torch
    from interp.part_core import ParTNetwork
    from interp import pn_net
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import CFG, FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = {}

    def put(name, kind, feat, model):
        model.to(dev).eval()
        for p in model.parameters(): p.requires_grad_(False)
        M[name] = dict(kind=kind, feat=feat, model=model, dev=dev)

    put('ParT_full', 'part', 'full', ParTNetwork('full').model); put('ParT_kin', 'part', 'kin', ParTNetwork('kin').model)
    PND = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    for feat in ('full', 'kin'):
        put(f'PN_{feat}', 'pn', feat, pn_net.load_particlenet(PND / f'ParticleNet_{feat}.pt', len(FEATURES[feat]), 'cpu'))
    torch.manual_seed(SEED); fresh = ParticleTransformer(input_dim=len(FEATURES['full']), **CFG)
    put('ParT_untr', 'part', 'full', fresh)
    pk = ParTNetwork('kin'); g = torch.Generator().manual_seed(SEED); sd = pk.model.state_dict(); new = {}; perms = {}
    for k, v in sd.items():
        if k.endswith('weight') and v.ndim >= 2 and v.is_floating_point():
            p = torch.randperm(v.shape[0], generator=g); perms[k[:-len('weight')]] = p; new[k] = v[p].clone()
    for k, v in sd.items():
        if k in new: continue
        pre = k[:-len('bias')] if k.endswith('bias') else None
        if pre is not None and pre in perms and v.ndim == 1 and v.shape[0] == len(perms[pre]): new[k] = v[perms[pre]].clone()
        elif k.endswith('in_proj_bias') and (k[:-len('in_proj_bias')] + 'in_proj_') in perms:
            new[k] = v[perms[k[:-len('in_proj_bias')] + 'in_proj_']].clone()
        else: new[k] = v
    pk.model.load_state_dict(new); put('ParT_kin_perm', 'part', 'kin', pk.model)
    log(f'models {list(M)} on {dev}; perm: {len(perms)} weight tensors row-permuted')
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


# ------------------------------------------------------------------------------------------------ edits
def radial(de, dp, d):
    r = np.hypot(de, dp); ok = r > 1e-6
    ue = np.where(ok, de / np.where(ok, r, 1), 1.0); up = np.where(ok, dp / np.where(ok, r, 1), 0.0)
    return de + d * ue, dp + d * up


def azim(de, dp, s):
    r = np.hypot(de, dp); th = s * 2 * np.arcsin(np.minimum(1.0, DR / np.maximum(2 * r, 1e-12)))
    c, si = np.cos(th), np.sin(th); return c * de - si * dp, si * de + c * dp


def plan_chunk(pt, de, dp, rng):
    """edit list for a chunk of jets: rows of (jet, type, i, j, sign), and the edited (de, dp) arrays"""
    n = len(pt); z = pt / pt.sum(1, keepdims=True); meta, DE, DP, SP = [], [], [], []
    for j in range(n):
        el = np.where(z[j] > ZMIN)[0]
        def add(t, i=-1, k=-1, s=0.0, mod=None, spread_set=None):
            e, p = de[j].copy(), dp[j].copy()
            if mod is not None: mod(e, p)
            meta.append((j, ETYPES.index(t), i, k, s)); DE.append(e); DP.append(p); SP.append(spread_set)
        add('base')
        for i in el:
            def f(e, p, i=i): e[i], p[i] = radial(e[i], p[i], DR)
            add('rad', i, mod=f)
        for i in el:
            def f(e, p, i=i): e[i], p[i] = radial(e[i], p[i], -DR)
            add('radin', i, mod=f)
        sg = rng.choice([-1.0, 1.0], len(el))
        for i, s in zip(el, sg):
            def f(e, p, i=i, s=s): e[i], p[i] = azim(e[i], p[i], s)
            add('az', i, s=s, mod=f)
        if len(el) >= 2:
            seen = set()
            for _ in range(NPAIR):
                a, b = np.sort(rng.choice(el, 2, replace=False))
                if (a, b) in seen: continue
                seen.add((a, b))
                def f(e, p, a=a, b=b):
                    e[a], p[a] = radial(e[a], p[a], DR); e[b], p[b] = radial(e[b], p[b], DR)
                add('pair', a, b, mod=f)
        for _ in range(NSP):
            k = min(KSP, len(el)); S = np.sort(rng.choice(el, k, replace=False))
            def f(e, p, S=S, k=k): e[S], p[S] = radial(e[S], p[S], DR / k)
            add('spread', mod=f, spread_set=S)
        def f(e, p, el=el): e[el], p[el] = radial(e[el], p[el], DR)
        add('all', mod=f)
        add('base2')
    return np.array(meta, dtype=np.float64), np.stack(DE), np.stack(DP), SP


# ------------------------------------------------------------------------------------------------ collection (GPU)
def collect():
    import torch
    from interp.part_core import kin, build, sub, sample, jmass
    from jetdistill.config import CLASSES as CL
    OUTD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED); Js, rowinfo = [], {}
    from interp.part_core import cat
    for c, fn in FILES.items():
        J, rows = load_rows(TEST / fn, PER, rng); Js.append(J); rowinfo[c] = dict(file=fn, n=len(rows), first_rows=rows[:5].tolist())
        log(f'loaded {fn}: {len(rows)} random rows')
    J = cat(Js); pt, de, dp, ms = kin(J); z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-9)
    keep = np.where((z > ZMIN).sum(1) >= 2)[0]; J = sub(J, keep); pt, de, dp, ms = kin(J)
    y = J['label'].astype(int); N = len(y); iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    log(f'N = {N} jets (dropped {sum(r["n"] for r in rowinfo.values()) - N} with < 2 eligible); classes {np.bincount(y, minlength=10).tolist()}')
    M, torch = load_models()
    ed_rng = np.random.default_rng(SEED + 1)
    META, LOG, SPS = [], {k: [] for k in M}, []
    t1 = time.time()
    for c0 in range(0, N, JCH):
        jj = np.arange(c0, min(c0 + JCH, N)); meta, DE, DP, SP = plan_chunk(pt[jj], de[jj], dp[jj], ed_rng)
        jl = meta[:, 0].astype(int); Jr = sub(J, jj[jl])
        for b0 in range(0, len(jl), BS):
            sl = slice(b0, b0 + BS); Jb = sub(Jr, np.arange(len(jl))[sl])
            p_, _, _, m_ = kin(Jb); X = all_inputs(build(Jb, p_, DE[sl], DP[sl], m_))
            for k, mm in M.items(): LOG[k].append(run_model(mm, X, torch)[:, [iW, iZ]])
        meta[:, 0] = jj[jl]; META.append(meta); SPS += [s if s is not None else np.array([], int) for s in SP]
        if (c0 // JCH) % 10 == 0: log(f'  jets {c0 + len(jj)} / {N}: {len(jl)} edits in chunk ({time.time() - t1:.0f} s)')
    META = np.concatenate(META); L = {k: np.concatenate(v) for k, v in LOG.items()}
    log(f'all edits done: {len(META)} forwards per model ({time.time() - t1:.0f} s)')
    # base logits for every jet (10 classes) and accuracy
    acc, accWZ = {}, {}
    Lb10 = {}
    for k, mm in M.items():
        out = []
        for b0 in range(0, N, BS):
            Jb = sub(J, np.arange(b0, min(b0 + BS, N))); p_, e_, f_, m_ = kin(Jb); out.append(run_model(mm, all_inputs(build(Jb, p_, e_, f_, m_)), torch))
        Lb10[k] = np.concatenate(out); acc[k] = float((Lb10[k].argmax(1) == y).mean())
        accWZ[k] = {c: float((Lb10[k].argmax(1)[y == CL.index(c)] == CL.index(c)).mean()) for c in FILES}
    log(f'accuracy (10-class argmax) on this sample: {acc}'); log(f'per class: {accWZ}')
    try:
        J1 = sample(per_class=100 if SMOKE else 1000, start=3000); y1 = J1['label'].astype(int); acc_p1 = {}
        X1 = all_inputs(J1)
        for k, mm in M.items():
            Lp = np.concatenate([run_model(mm, {f: tuple(a[s] for a in X1[f]) for f in X1}, torch) for s in [slice(j, j + BS) for j in range(0, len(y1), BS)]])
            acc_p1[k] = float((Lp.argmax(1) == y1).mean())
        log(f'P1-sample accuracy: {acc_p1}')
    except Exception as ex:  # noqa: BLE001
        log(f'P1 sample failed: {ex!r}'); acc_p1 = None
    Mo = dict(M0=pt.sum(1), M2=(pt * (de ** 2 + dp ** 2)).sum(1))
    sp_flat = np.concatenate([s for s in SPS]) if SPS else np.array([], int); sp_len = np.array([len(s) for s in SPS])
    np.savez_compressed(NPZ, meta=META, y=y, pt=pt, de=de, dp=dp, mjet=jmass(J), M0=Mo['M0'], M2=Mo['M2'], sp_flat=sp_flat, sp_len=sp_len,
                        **{f'L_{k}': L[k].astype(np.float32) for k in M}, **{f'Lb10_{k}': Lb10[k].astype(np.float32) for k in M})
    info = dict(sample=dict(files=rowinfo, seed=SEED, N=N, per_class=PER, classes=np.bincount(y, minlength=10).tolist()),
                acc_sample=acc, acc_sample_by_class=accWZ, acc_P1_sample=acc_p1, forwards_per_model=int(len(META)))
    (OUTD / f'p27_particle_kernel{TAG}_info.json').write_text(json.dumps(strip(info), indent=1))
    log(f'wrote {NPZ}')


# ------------------------------------------------------------------------------------------------ analysis (CPU)
def design(F):
    z, r, rk, x, s = F['z'], F['r'], F['rank'], F['x'], F['s']
    D = {'M2': x[:, None], 'M2_ctx': x[:, None] * np.stack([np.ones_like(s), s, s ** 2, s ** 3], 1),
         'M2_rankwin': np.stack([x * ((rk >= a) & (rk < b)) for a, b in WINDOWS], 1),
         'rankwin_zr': np.concatenate([np.stack([z * w, r * w, z * r * w], 1) for w in [((rk >= a) & (rk < b)).astype(float) for a, b in WINDOWS]], 1),
         'hinge_z': np.concatenate([np.stack([np.maximum(z - t, 0), np.maximum(z - t, 0) * r], 1) for t in TZ], 1),
         'hinge_r': np.concatenate([np.stack([z * (r > t), (r > t).astype(float)], 1) for t in TR], 1),
         'pt_only': np.stack([z, z ** 2, np.sqrt(z)], 1), 'r_only': np.stack([r, r ** 2, np.sqrt(r)], 1),
         'const': np.ones((len(z), 1))}
    return D


def lsq(X, y):
    G = X.T @ X; return np.linalg.lstsq(G + 1e-10 * np.trace(G) / len(G) * np.eye(len(G)), X.T @ y, rcond=None)[0]


def fit_forms(F, yv, jr, N, y, folds, Wb, M0, M2, mj, nreal):
    """held-out (5-fold over jets) R^2, per-jet-gain shape R^2, BIC and coefficients of every form for the target yv"""
    from jetdistill.config import CLASSES as CL
    D = design(F); ybar = yv.mean(); n = len(yv); sst_j = np.bincount(jr, (yv - ybar) ** 2, minlength=N); preds = {}
    for fn_, X in D.items():
        ph = np.zeros(n)
        for k in range(5):
            tr = folds[jr] != k; te = ~tr; b = lsq(X[tr], yv[tr]); ph[te] = X[te] @ b
        preds[fn_] = ph
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xg = np.stack([F['z'], F['r'], F['rank'], F['x'], M0[jr], M2[jr], mj[jr], nreal[jr], F['s']], 1); ph = np.zeros(n)
        for k in range(5):
            tr = folds[jr] != k; te = ~tr
            g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED).fit(Xg[tr], yv[tr]); ph[te] = g.predict(Xg[te])
        preds['gbdt'] = ph
    except Exception as ex:  # noqa: BLE001
        log(f'gbdt failed: {ex!r}')
    odd_rank = (F['rank'] % 2 == 1); ev = ~odd_rank; forms = {}
    for fn_, ph in preds.items():
        sse_j = np.bincount(jr, (yv - ph) ** 2, minlength=N); R2 = 1 - sse_j.sum() / sst_j.sum(); R2b = 1 - (Wb @ sse_j) / (Wb @ sst_j)
        num = np.bincount(jr[ev], ph[ev] * yv[ev], minlength=N); den = np.bincount(jr[ev], ph[ev] ** 2, minlength=N)
        gj = np.where(den > 1e-30, num / np.maximum(den, 1e-30), 0.0); ph2 = gj[jr] * ph; o = odd_rank
        sse2 = np.bincount(jr[o], (yv[o] - ph2[o]) ** 2, minlength=N); sst2 = np.bincount(jr[o], (yv[o] - yv[o].mean()) ** 2, minlength=N)
        R2s = 1 - sse2.sum() / sst2.sum(); R2sb = 1 - (Wb @ sse2) / (Wb @ sst2)
        k_ = D[fn_].shape[1] if fn_ in D else None; bic, bfull = None, None
        if fn_ in D:
            bfull = lsq(D[fn_], yv); sse_full = ((yv - D[fn_] @ bfull) ** 2).sum(); bic = float(n * np.log(sse_full / n) + k_ * np.log(n))
        qb = np.quantile(ph, np.linspace(0, 1, 21)); bb = np.clip(np.searchsorted(qb, ph, side='right') - 1, 0, 19)
        nb_ = np.bincount(bb, minlength=20).astype(float); yb = np.bincount(bb, yv, minlength=20) / np.maximum(nb_, 1); pb = np.bincount(bb, ph, minlength=20) / np.maximum(nb_, 1)
        R2bin = float(1 - (nb_ * (yb - pb) ** 2).sum() / max((nb_ * (yb - ybar) ** 2).sum(), 1e-30))
        forms[fn_] = dict(k=k_, R2_heldout=st(R2, R2b), R2_binned_means=R2bin, R2_shape_perjet_gain=st(R2s, R2sb), BIC=bic,
                          coef=(bfull.tolist() if bfull is not None and len(bfull) <= 21 else None), _R2b=R2b, _R2sb=R2sb)
        for c in ('Wqq', 'Zqq'):
            sj = (y == CL.index(c))[jr]; forms[fn_][f'R2_heldout_{c}'] = float(1 - ((yv - ph)[sj] ** 2).sum() / ((yv[sj] - yv[sj].mean()) ** 2).sum())
    best = max((f for f in forms if f != 'gbdt'), key=lambda f: forms[f]['R2_heldout']['v'])
    for fn_ in forms:
        if fn_ == 'M2': continue
        forms[fn_]['dR2_vs_M2'] = st(forms[fn_]['R2_heldout']['v'] - forms['M2']['R2_heldout']['v'], forms[fn_]['_R2b'] - forms['M2']['_R2b'])
        forms[fn_]['dR2_shape_vs_M2'] = st(forms[fn_]['R2_shape_perjet_gain']['v'] - forms['M2']['R2_shape_perjet_gain']['v'], forms[fn_]['_R2sb'] - forms['M2']['_R2sb'])
    return forms, best


def analyse():
    from jetdistill.config import CLASSES as CL
    Z = np.load(NPZ, allow_pickle=False); meta = Z['meta']; y = Z['y']; pt, de, dp = Z['pt'], Z['de'], Z['dp']
    N = len(y); jet, et, pi, pj, sg = meta[:, 0].astype(int), meta[:, 1].astype(int), meta[:, 2].astype(int), meta[:, 3].astype(int), meta[:, 4]
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    info = json.loads((OUTD / f'p27_particle_kernel{TAG}_info.json').read_text())
    res = dict(info, settings=dict(dr=DR, zmin=ZMIN, npair=NPAIR, ksp=KSP, nsp=NSP, NB=NB, windows=WINDOWS, tz=TZ, tr=TR, readout='d = change of logit_Z - logit_W'),
               models={})
    sp_len = Z['sp_len']; sp_off = np.concatenate([[0], np.cumsum(sp_len)]); sp_flat = Z['sp_flat']
    rng = np.random.default_rng(SEED + 5); Wb = boot_W(N, NB, rng)
    folds = np.random.default_rng(SEED + 6).permutation(N) % 5
    M0, M2, mj = Z['M0'], Z['M2'], Z['mjet']; nreal = (pt > 0).sum(1)
    ib = np.where(et == 0)[0]; base_row = np.full(N, -1); base_row[jet[ib]] = ib
    rad = np.where(et == 1)[0]; az = np.where(et == 2)[0]; pr = np.where(et == 3)[0]; spd = np.where(et == 4)[0]; al = np.where(et == 5)[0]; b2 = np.where(et == 6)[0]; rin = np.where(et == 7)[0]
    # per-particle features of the rad edits
    jr, ir = jet[rad], pi[rad]; ptr = pt[jr, ir]; rr = np.hypot(de[jr, ir], dp[jr, ir])
    F = dict(z=ptr / M0[jr], r=rr, rank=ir.astype(float), x=ptr * ((rr + DR) ** 2 - rr ** 2) / M2[jr])
    F['xodd'] = (F['x'] - ptr * ((rr - DR) ** 2 - rr ** 2) / M2[jr]) / 2
    res['n_particles_edited'] = int(len(rad)); res['n_elig_per_jet'] = dict(mean=float(len(rad) / N), median=float(np.median(np.bincount(jr, minlength=N))))
    allres, ptests = {}, []
    for mn in MODELS:
        if f'L_{mn}' not in Z: continue
        L = Z[f'L_{mn}'].astype(np.float64); sc = L[:, 1] - L[:, 0]; base = sc[base_row]; d = sc - base[jet]
        s_j = np.clip(base / 5.0, -2, 2); F['s'] = s_j[jr]
        di = d[rad]; out = dict()
        out['noise_floor_abs'] = float(np.abs(d[b2]).mean()); out['mean_abs_rad'] = float(np.abs(di).mean())
        # look-up d_i by (jet, particle)
        key = lambda j, i: j * 128 + i
        dmap = dict(zip(key(jr, ir).tolist(), di.tolist()))
        # ---- per-particle odd / even split from the outward (rad) and inward (radin) moves
        jin_map = dict(zip(key(jet[rin], pi[rin]).tolist(), d[rin].tolist()))
        din = np.array([jin_map[k] for k in key(jr, ir).tolist()]); odd_i, even_i = (di - din) / 2, (di + din) / 2
        so_, se_ = np.bincount(jr, odd_i ** 2, minlength=N), np.bincount(jr, even_i ** 2, minlength=N); cj = np.bincount(jr, minlength=N).astype(float)
        out['perparticle_even_share'] = st(se_.sum() / (so_.sum() + se_.sum()), (Wb @ se_) / (Wb @ (so_ + se_)))
        out['odd_mean'] = st(odd_i.mean(), (Wb @ np.bincount(jr, odd_i, minlength=N)) / (Wb @ cj))
        out['even_mean'] = st(even_i.mean(), (Wb @ np.bincount(jr, even_i, minlength=N)) / (Wb @ cj))
        # ---- forms: 5-fold held-out over jets, for the outward response d_i and for its odd part
        # ---- shared component with the azimuthal move of the same particle
        az_map = dict(zip(key(jet[az], pi[az]).tolist(), d[az].tolist())); daz_i = np.array([az_map[k] for k in key(jr, ir).tolist()])
        out['corr_rad_az'] = float(np.corrcoef(di, daz_i)[0, 1]); out['corr_absrad_absaz'] = float(np.corrcoef(np.abs(di), np.abs(daz_i))[0, 1])
        out['corr_even_az'] = float(np.corrcoef(even_i, daz_i)[0, 1])
        bz = (di * daz_i).sum() / (daz_i ** 2).sum(); out['R2_rad_from_az'] = float(1 - ((di - bz * daz_i) ** 2).sum() / ((di - di.mean()) ** 2).sum()); out['slope_rad_on_az'] = float(bz)
        out['forms_by_target'] = {}
        for tname, yv, xv in (('rad', di, F['x']), ('odd', odd_i, F['xodd']), ('rad_minus_az', di - daz_i, F['x'])):
            forms, best = fit_forms(dict(F, x=xv), yv, jr, N, y, folds, Wb, M0, M2, mj, nreal)
            for fn_ in forms:
                if fn_ != 'M2' and mn in ('ParT_full', 'ParT_kin', 'PN_full', 'PN_kin'): ptests.append((mn, tname, fn_, forms[fn_]['dR2_vs_M2']['p0']))
            out['forms_by_target'][tname] = dict(forms=forms, best_low_param_form=best)
        forms = out['forms_by_target']['rad']['forms']; out['best_low_param_form'] = out['forms_by_target']['rad']['best_low_param_form']
        # ---- binned mean response vs x (for the figure)
        qs = np.quantile(F['x'], np.linspace(0, 1, 13)); bi = np.clip(np.searchsorted(qs, F['x']) - 1, 0, 11)
        out['binned_rad'] = dict(x=[float(F['x'][bi == b].mean()) for b in range(12)], d=[float(di[bi == b].mean()) for b in range(12)])
        out['binned_odd'] = dict(x=[float(F['xodd'][bi == b].mean()) for b in range(12)], d=[float(odd_i[bi == b].mean()) for b in range(12)])
        out['binned_even'] = dict(x=out['binned_rad']['x'], d=[float(even_i[bi == b].mean()) for b in range(12)])
        out['binned_rad'] = dict(x=[float(F['x'][bi == b].mean()) for b in range(12)], d=[float(di[bi == b].mean()) for b in range(12)])
        # ---- azimuthal control (a)
        ja, ia = jet[az], pi[az]; da = d[az]
        num_a = np.bincount(ja, np.abs(da), minlength=N); num_r = np.bincount(jr, np.abs(di), minlength=N)
        out['az_over_rad_meanabs'] = st(num_a.sum() / num_r.sum(), (Wb @ num_a) / (Wb @ num_r))
        so = np.bincount(ja, sg[az] * da, minlength=N); cnt = np.bincount(ja, minlength=N).astype(float)
        out['az_signed_mean'] = st(so.sum() / cnt.sum(), (Wb @ so) / (Wb @ cnt)); out['az_mean'] = st(da.mean(), (Wb @ np.bincount(ja, da, minlength=N)) / (Wb @ cnt))
        out['rad_mean'] = st(di.mean(), (Wb @ np.bincount(jr, di, minlength=N)) / (Wb @ np.bincount(jr, minlength=N).astype(float)))
        # odd share of the azimuthal response: (signed part)^2 / total^2 over particles
        out['az_odd_share'] = float((sg[az] * da).mean() ** 2 / max((da ** 2).mean(), 1e-30))
        # sign-flip permutation for the azimuthal odd part, H0 mean s_i d_az = 0
        prng = np.random.default_rng(SEED + 7); obs = abs(so.sum()); per = np.array([abs((prng.choice([-1, 1], N) * so).sum()) for _ in range(2000)])
        out['az_signed_perm_p'] = float((1 + (per >= obs).sum()) / 2001)
        out['binned_az'] = dict(x=out['binned_rad']['x'], d=[float(np.nanmean(da[np.clip(np.searchsorted(qs, (pt[ja, ia] * ((np.hypot(de[ja, ia], dp[ja, ia]) + DR) ** 2 - np.hypot(de[ja, ia], dp[ja, ia]) ** 2) / M2[ja])) - 1, 0, 11) == b])) for b in range(12)])
        # ---- additivity (pairs)
        jp, a1, a2 = jet[pr], pi[pr], pj[pr]; dij = d[pr]
        di1 = np.array([dmap[k] for k in key(jp, a1).tolist()]); di2 = np.array([dmap[k] for k in key(jp, a2).tolist()]); res_ = dij - di1 - di2
        e1 = np.bincount(jp, res_ ** 2, minlength=N); e2 = np.bincount(jp, (dij - dij.mean()) ** 2, minlength=N); e3 = np.bincount(jp, dij ** 2, minlength=N)
        out['additivity'] = st(1 - e1.sum() / e2.sum(), 1 - (Wb @ e1) / (Wb @ e2)); out['interaction_share'] = st(e1.sum() / e3.sum(), (Wb @ e1) / (Wb @ e3))
        s1, s2 = np.bincount(jp, dij, minlength=N), np.bincount(jp, di1 + di2, minlength=N)
        out['pair_ratio_sum'] = st(s1.sum() / s2.sum(), (Wb @ s1) / (Wb @ s2), 1.0)
        # ---- calibration: sum_i d_i vs whole-jet radial move
        sum_i = np.bincount(jr, di, minlength=N); dall = np.zeros(N); dall[jet[al]] = d[al]
        out['calib_sum_over_all'] = st(sum_i.sum() / dall.sum(), (Wb @ sum_i) / (Wb @ dall), 1.0)
        out['calib_R2_perjet'] = float(1 - ((dall - sum_i) ** 2).sum() / ((dall - dall.mean()) ** 2).sum())
        out['all_mean'] = st(dall.mean(), (Wb @ dall) / N)
        # ---- spread control (b): d_spread vs mean of member d_i (linear prediction)
        sp_rows = spd; pred = np.zeros(len(sp_rows))
        for q, rrow in enumerate(sp_rows):
            S = sp_flat[sp_off[rrow]:sp_off[rrow + 1]]; jj_ = jet[rrow]; pred[q] = np.mean([dmap[key(jj_, int(i_))] for i_ in S])
        js = jet[sp_rows]; dsp = d[sp_rows]; n1, n2 = np.bincount(js, dsp, minlength=N), np.bincount(js, pred, minlength=N)
        out['spread_ratio_to_linear'] = st(n1.sum() / n2.sum(), (Wb @ n1) / (Wb @ n2), 1.0)
        out['spread_R2_linear'] = float(1 - ((dsp - pred) ** 2).sum() / ((dsp - dsp.mean()) ** 2).sum())
        out['spread_mean'] = st(dsp.mean(), (Wb @ n1) / (Wb @ np.bincount(js, minlength=N).astype(float)))
        allres[mn] = out
        log(f'{mn}: R2 held-out ' + ', '.join(f"{k} {v['R2_heldout']['v']:.3f}" for k, v in forms.items()) +
            f"; shape M2 {forms['M2']['R2_shape_perjet_gain']['v']:.3f}; add {out['additivity']['v']:.3f}; calib {out['calib_sum_over_all']['v']:.3f}; az/rad {out['az_over_rad_meanabs']['v']:.3f}")
    adj = holm([p for *_, p in ptests])
    for (mn, tn, fn_, p), pa in zip(ptests, adj): allres[mn]['forms_by_target'][tn]['forms'][fn_]['dR2_vs_M2']['p_holm'] = pa
    pn_ = [allres[mn]['forms_by_target'][tn]['forms'][fn_]['dR2_vs_M2']['p_norm'] for mn, tn, fn_, _ in ptests]
    adj2 = holm([1.0 if v is None else v for v in pn_])
    for (mn, tn, fn_, p), pa in zip(ptests, adj2): allres[mn]['forms_by_target'][tn]['forms'][fn_]['dR2_vs_M2']['p_norm_holm'] = pa
    res['holm_family'] = f'{len(ptests)} tests (targets x forms x models, H0 R2(form) = R2(M2))'
    res['models'] = allres
    FN.write_text(json.dumps(strip(res), indent=1)); log(f'wrote {FN}')
    return res


def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = res or json.loads(FN.read_text()); R = res['models']
    forms = ['M2', 'M2_ctx', 'M2_rankwin', 'rankwin_zr', 'hinge_z', 'hinge_r', 'pt_only', 'r_only', 'gbdt']
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    a = ax[0]; ms = [m for m in MODELS if m in R]; w = 0.8 / len(ms)
    for k, mn in enumerate(ms):
        FF = R[mn]['forms_by_target']['odd']['forms']
        v = [FF.get(f, {}).get('R2_heldout', {}).get('v', np.nan) for f in forms]
        lo = [FF.get(f, {}).get('R2_heldout', {}).get('ci', [np.nan, np.nan])[0] for f in forms]
        hi = [FF.get(f, {}).get('R2_heldout', {}).get('ci', [np.nan, np.nan])[1] for f in forms]
        xs = np.arange(len(forms)) + (k - len(ms) / 2 + 0.5) * w; v = np.array(v, float)
        a.bar(xs, v, w, label=mn, yerr=[v - np.array(lo, float), np.array(hi, float) - v], capsize=1.5)
    a.set_xticks(np.arange(len(forms))); a.set_xticklabels(forms, rotation=30, fontsize=8); a.axhline(0, color='k', lw=0.5)
    a.set_ylabel('held-out R$^2$ (5-fold over jets), odd part (out - in)/2'); a.set_ylim(-0.3, 1); a.legend(fontsize=7, ncol=2)
    a.set_title('(a) held-out R$^2$ of per-particle forms for the response to moving one particle by 0.01 rad', fontsize=9)
    a = ax[1]
    for mn, c in (('ParT_full', 'C0'), ('ParT_kin', 'C1'), ('PN_full', 'C2'), ('PN_kin', 'C3'), ('ParT_untr', 'C4')):
        if mn not in R: continue
        b = R[mn]['binned_rad']; a.plot(b['x'], b['d'], '-o', color=c, ms=3, label=f'{mn} radial')
        b2 = R[mn]['binned_az']; a.plot(b2['x'], b2['d'], ':', color=c, lw=1)
        b3 = R[mn]['binned_even']; a.plot(b3['x'], b3['d'], '--', color=c, lw=0.8)
    a.set_xscale('log'); a.axhline(0, color='k', lw=0.5)
    a.set_xlabel(r'$\Delta M_2 / M_2$ of the edit ($M_2=\sum p_T r^2$ about the stored axis)')
    a.set_ylabel(r'mean change of logit$_Z$ - logit$_W$'); a.legend(fontsize=7)
    a.set_title('(b) binned mean response; solid: outward, dashed: (out+in)/2, dotted: around the axis', fontsize=9)
    fig.tight_layout(); fn = OUTD / f'p27_particle_kernel{TAG}.png'; fig.savefig(fn, dpi=130); log(f'wrote {fn}')


if __name__ == '__main__':
    if ARGS.plot: plot()
    elif ARGS.analyse: plot(analyse())
    else:
        collect(); r = analyse(); plot(r)
