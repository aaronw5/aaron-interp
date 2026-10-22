"""B10 (Paper B claims B2n, B6t, B; stage 3, exploratory: method generality): the B9 if-statement (MARS) program extraction
recipe applied to non-JEDI hls4ml taggers (Deep Sets, Linformer, transformer, Particle Flow Network), with an azimuthal
feature pool and three repeated fits per network: fidelity vs description length, baselines at matched bits, and recall of the
networks' call changes under a rigid shift, a mass-lowering azimuthal turn, its mass-preserving control and a soft emission.

Question: do short if-statement programs (sums of hinge functions of jet observables) describe small set taggers other than
JEDI-linear as well as they describe JEDI-linear, and do they reproduce those networks' responses to input edits better than
programs fitted to other networks or to the truth labels?

PLAN (fixed before the first run)
Networks (all existing checkpoints; no training), every one sees the 64-slot (pT, deta, dphi) jet:
  jedi_off  official JEDI-linear 3-feature n64 (interp.jedi_core.Jedi; reference row); jedi_u0 = Jedi(untrained=True, seed 0).
  ds_s0/s1  Deep Sets dsets-d64-n64 seed 0 / 1 (train/models.py, torch; small/<run>/seed<k>/best.pt).
  linf_s0/s1  Linformer linf-2b-d32-k8-n64 seed 0 / 1.
  tf_s0/s1  transformer tf-2b-d32-n64 seed 0 and tf-2b-d32-n64-p30 seed 1 (the patience-30 rerun of the early-stopped seed 1).
  pfn_s0/s1  X9 PFN pfn_64-64-32 seed 0 / 1 (interp.x9_efn_pfn.Flow, exact numpy forward).
  untrained: ds_u0 / linf_u0 / tf_u0 = default PyTorch init under torch.manual_seed(5000) (X13), pfn_u0 = he_uniform kernels,
  zero biases, rng 317000 (X13 recipe).  Torch networks are run in the conda env (AI_PY=conda), the rest in the jax env.
Sample: interp.jedi_core.sample(per_class=4000, seed=3170) = 20,000 jets; random permutation rng(3170): fit0 / fit1 / fit2 =
  4,000 jets each (disjoint; three repeated program fits), dev 2,000 (model selection), test 6,000 (all reported numbers).
Feature pool (140 inputs, identical for every network):
  B9 pool (98): the rank-window sums of e11_mars_all.handbuilt via b6_pareto.features (sum pT, N, sum pT r^2, sum r^2, ... over
    pT-rank windows) + M0 = sum pT.
  azimuthal pool (42): complex moments C_n = sum_i pT_i z_i^n / sum_i pT_i with z_i = deta_i + i dphi_i (n = 1..4) over the 14
    hardest particles and over all particles: |C_n|, Re C_n, Im C_n (24); for the nine JEDI-linear l1 half-plane cuts (E14 units
    a29 a6 a40 a57 a3 a43 a30 a36 a46; line alpha_eta deta + alpha_phi dphi + beta = 0 taken from the official l1 kernel in
    physical units) the pT sum and the particle count on the positive side (18).
  standard observables (10, B8 STD): m, m_SD, tau21, tau32, D2, N, sum pT, sum pT r^n (n = 1, 2, 3).
Program (per network, per fit f = 0, 1, 2): B9 recipe (b9.mars_path / mars_prefix: MARS per logit on the network's 5 logits,
  forward path to 32 terms, nested K = 4, 8, 16, 32 terms per logit refitted by least squares; call = argmax).  Description
  length DL (bits) = per logit 32 + per term 32 + per factor [log2(140) + 2 + 32 (knot, unless linear)].
  B9-pool program (fit0 only, 98 inputs, K = 4..32): the B9 recipe without the azimuthal pool.
Baselines (fit0, matched by bits): MARS on the standard observables (K = 2..32); HistGradientBoosting classifier on the standard
  observables and on the full pool (quantised to 2^q levels; B9 grid; DL = b9.hgb_bits_q); CART on the pool (depth 2-12); Lasso
  per logit on the standardised pool (10 alphas).  Controls (K16): truth program (MARS fitted to log(0.95 onehot + 0.01) of the
  truth labels, never to the network), shuffled-logit program (fit rows' logits permuted, rng 3176), random-feature program
  (6 pool inputs drawn at random, rng 3177).
Edits on the test jets (padding never moved; E39 / B9 / E41v definitions):
  shift  every real particle's deta + 0.06 rad (rigid translation; invariant mass unchanged; M2 about the stored axis rises).
  turn10 the 14 hardest particles turned in azimuth about the stored axis toward the pT-weighted centroid direction by a common
         fraction t of their angle gap, t solved per jet so that the jet mass m falls 10 %; each particle keeps pT and r, so M0,
         M2 and every rank-window sum of the B9 pool are unchanged (E27w arm B, B9 'turn').
  ctrl10 control for turn10 (E41v): the same 14 particles turned by the same |dtheta_i|, each sign random (rng 3171), then a
         minimum-norm azimuth correction restoring the vector M1 and a common rescale holding the displacement equal to the
         turn's (e27w.solve_ctrl); pT, r, M0, M1, M2 and m held; only the azimuthal arrangement changes.  turn10 and ctrl10 are
         evaluated on the same jets (rows where both converge).
  soft   one 0.5 GeV particle added at r = 0.3, random azimuth (rng 3175); the other particles get one common scale and boost so
         the jet four-momentum is exact (E40 C2); jets with N <= 63.
  Per jet: network flip nf = call(edited) != call(unedited); program flip sf likewise; hit = nf & sf & same new call; recall =
  hits / network flips; precision = hits / program flips; kappa = Cohen kappa of (nf, sf).
Transfer: network A's fit0 K16 program scored against network B's calls (natural same call; edit recall) vs B's own programs
  (mean of fits 0-2) and the truth program.
Statistics: 1,000-draw jet bootstrap over test rows (edited sets use the draws of their base rows); fit-to-fit spread = range
  over the three fits; exact McNemar for paired per-jet indicators (program vs matched-DL baseline; own vs transfer hits);
  Holm across networks x baselines (per K and metric) and across networks (transfer).
Predictions.  Thesis (a short program describes small set taggers generally): same call >= 85 % at DL <= ~7 kbit for Deep Sets,
  PFN and Linformer; own > transfer > truth on shift recall beyond the fit-to-fit spread; turn recall > 0 with the azimuthal
  pool (the B9 pool cannot respond to the turn: every input is unchanged).  Alternative (the recipe works because of JEDI's
  rank windows and pure cuts): Linformer / transformer same call well below 85 % at matched bits, no gain over GBDT on the
  standard observables, own = transfer within the fit spread.
Run (from the repo root; ./sync.sh first):
  ./job.sh b10-prep 'python -m interp.b10_programs_small_models prep'                        jets, edits, features, jax nets
  AI_PY=conda ./job.sh b10-torch 'python -m interp.b10_programs_small_models torch'          torch nets' logits
  ./job.sh b10-ds 'python -m interp.b10_programs_small_models fit --nets ds_s0,ds_s1,ds_u0'  (linf, tf, pfn, jedi likewise)
  ./job.sh b10-agg 'python -m interp.b10_programs_small_models agg'                          -> b10.json, b10.png
  --smoke: 200 jets per class, K <= 8, 100 bootstrap draws (files b10smoke_*).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
PER, SEED = 4000, 3170
FITS = ['fit0', 'fit1', 'fit2']
NAT = FITS + ['dev', 'test']
EDITS = ['shift', 'turn10', 'ctrl10', 'soft']
SETS = NAT + EDITS
EDIT_LABEL = dict(shift='all particles deta + 0.06 rad', turn10='hardest 14 turned in azimuth, m - 10 % (pT, r kept)',
                  ctrl10='control: same 14 turned by the same amounts, M1 and m held', soft='one 0.5 GeV particle added at r = 0.3')
HP = [29, 6, 40, 57, 3, 43, 30, 36, 46]          # E14: the nine l1 half-plane statements of the official JEDI-linear
TORCH = dict(ds_s0=('dsets-d64-n64', 0), ds_s1=('dsets-d64-n64', 1), ds_u0=('dsets-d64-n64', 'u0'),
             linf_s0=('linf-2b-d32-k8-n64', 0), linf_s1=('linf-2b-d32-k8-n64', 1), linf_u0=('linf-2b-d32-k8-n64', 'u0'),
             tf_s0=('tf-2b-d32-n64', 0), tf_s1=('tf-2b-d32-n64-p30', 1), tf_u0=('tf-2b-d32-n64', 'u0'))
JAXNETS = ['jedi_off', 'jedi_u0', 'pfn_s0', 'pfn_s1', 'pfn_u0']
FAMILY = dict(jedi='JEDI-linear', ds='Deep Sets', linf='Linformer', tf='transformer', pfn='PFN')
TRAINED = ['jedi_off', 'ds_s0', 'ds_s1', 'linf_s0', 'linf_s1', 'tf_s0', 'tf_s1', 'pfn_s0', 'pfn_s1']
UNTR = ['jedi_u0', 'ds_u0', 'linf_u0', 'tf_u0', 'pfn_u0']
KS = [4, 8, 16, 32]
KSTD = [2, 4, 8, 16, 32]
DL_TARGET = 7000.0


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def fam(n): return n.split('_')[0]


def tag(smoke): return 'b10smoke' if smoke else 'b10'


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 6)
    return o


# ================================================================================================ features
def azimuthal(x, hp_lines):
    """42 azimuthal features of 64-slot jets: complex moments C_n (n = 1..4; hard14 / all; |C|, Re, Im) + half-plane pT / counts"""
    x = np.asarray(x, np.float64); pt = x[..., 0]; real = pt > 0; z = (x[..., 1] + 1j * x[..., 2]) * real
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); hard = real & (rank < 14)
    cols, keys = [], []
    for nm, msk in (('h14', hard), ('all', real)):
        w = pt * msk; W = np.maximum(w.sum(1), 1e-12)
        for n in range(1, 5):
            C = (w * z ** n).sum(1) / W
            cols += [np.abs(C), C.real, C.imag]; keys += [f'A:|C{n}|[{nm}]', f'A:ReC{n}[{nm}]', f'A:ImC{n}[{nm}]']
    for c, (ae, ap, b) in zip(HP, hp_lines):
        on = real & (ae * x[..., 1] + ap * x[..., 2] + b > 0)
        cols += [(pt * on).sum(1), on.sum(1).astype(np.float64)]; keys += [f'A:ΣpT[hp{c}]', f'A:N[hp{c}]']
    return np.stack(cols, 1), keys


def hp_lines_from_jedi():
    from interp.jedi_core import Jedi
    j0 = Jedi(64); K, b = j0.KB['l1']; b = b.reshape(-1); K = K.reshape(3, 64); sc, sh = j0.scale, j0.shift
    alpha = K / sc[:, None]; beta = b - (K * (sh / sc)[:, None]).sum(0)
    lines = [(float(alpha[1, c]), float(alpha[2, c]), float(beta[c])) for c in HP]
    info = {f'a{c}': dict(w_pt=float(alpha[0, c]), w_eta=l[0], w_phi=l[1], beta=l[2],
                          normal_deg=float(np.degrees(np.mod(np.arctan2(l[1], l[0]), 2 * np.pi))),
                          offset_rad=float(abs(l[2]) / max(np.hypot(l[0], l[1]), 1e-12))) for c, l in zip(HP, lines)}
    return lines, info


def make_edits(xT):
    from interp.e27w_mass_vs_m2_jedi import solve_B, solve_ctrl, M1v, wrap, to_x, mass_p, disp
    from interp.e40_qg_count_jedi import emit
    J = len(xT); real = xT[..., 0] > 0; N = real.sum(1); E, info = {}, {}
    s = xT.copy(); s[..., 1] += np.float32(0.06) * real; E['shift'] = (s, np.arange(J)); info['shift'] = dict(N=J)
    x64 = np.asarray(xT, np.float64); pt = x64[..., 0]; r = np.hypot(x64[..., 1], x64[..., 2]); th = np.arctan2(x64[..., 2], x64[..., 1])
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); sub = real & (rank < 14)
    M10 = M1v(pt, r, th); phu = np.arctan2(M10[:, 1], M10[:, 0]); d = np.where(sub, wrap(phu[:, None] - th), 0.0)
    tB, okB = solve_B(pt, r, th, d, 0.10); thB = th + tB[:, None] * d
    m0, m1 = mass_p(pt, r, th), mass_p(pt, r, thB); okB = okB & (np.abs(m1 / np.maximum(m0, 1e-9) - 0.9) < 0.005)
    DB = disp(r, th, thB); thC, okC = solve_ctrl(pt, r, th, sub, tB[:, None] * d, DB, np.random.default_rng(3171))
    mC = mass_p(pt, r, thC); okC = okC & (np.abs(mC / np.maximum(m0, 1e-9) - 1) < 1e-3)
    rows = np.flatnonzero(okB & okC)
    xB = to_x(xT, real, r, thB); xC = to_x(xT, real, r, thC)
    E['turn10'] = (xB[rows], rows); E['ctrl10'] = (xC[rows], rows)
    M2 = lambda p_, rr: (p_ * rr ** 2).sum(1)
    info['turn10'] = dict(N=int(len(rows)), n_turn_ok=int(okB.sum()), n_ctrl_ok=int((okB & okC).sum()),
                          median_dlogm=float(np.median(np.log(m1[rows] / m0[rows]))) if len(rows) else None,
                          max_rel_dM2=float(np.max(np.abs(M2(pt[rows], np.hypot(xB[rows][..., 1], xB[rows][..., 2])) / M2(pt[rows], r[rows]) - 1))) if len(rows) else None)
    info['ctrl10'] = dict(N=int(len(rows)), max_abs_dlogm=float(np.max(np.abs(np.log(mC[rows] / m0[rows])))) if len(rows) else None,
                          max_rel_dM1=float(np.max(np.hypot(*(M1v(pt[rows], r[rows], thC[rows]) - M10[rows]).T) / pt[rows].sum(1))) if len(rows) else None)
    Yo, ok, inf = emit(xT, 1, 0.5, 0.3, np.random.default_rng(3175)); rows = np.flatnonzero(ok)
    E['soft'] = (np.ascontiguousarray(Yo[rows], np.float32), rows); info['soft'] = dict(N=int(len(rows)), **inf)
    log('edits', json.dumps(clean(info)))
    return E, info


# ================================================================================================ prep (jax env)
def splits(J, smoke):
    p = np.random.default_rng(SEED).permutation(J); f = [0.2, 0.2, 0.2, 0.1, 0.3]
    cut = np.cumsum([0] + [int(round(J * v)) for v in f]); cut[-1] = J
    return {s: np.sort(p[cut[i]:cut[i + 1]]) for i, s in enumerate(NAT)}


def prep(smoke):
    from interp.jedi_core import sample
    from interp.b6_pareto import features
    from interp.b8_fidelity_dl_iia import std_obs, STD_NAMES
    per = 200 if smoke else PER; T = tag(smoke)
    x, y, _ = sample(per_class=per, seed=SEED); x = np.asarray(x, np.float32); J = len(y)
    idx = splits(J, smoke); log(f'sample(per_class={per}, seed={SEED}): N {J}; ' + ', '.join(f'{s} {len(v)}' for s, v in idx.items()))
    X = {s: x[idx[s]] for s in NAT}; Y = {s: y[idx[s]] for s in NAT}
    E, einfo = make_edits(X['test'])
    for e in EDITS: X[e] = E[e][0]
    lines, hpinfo = hp_lines_from_jedi(); log('half-planes', json.dumps(clean(hpinfo)))
    F, keys = {}, None
    for s in SETS:
        t = time.time(); P, k1 = features(X[s]); A, k2 = azimuthal(X[s], lines); St = std_obs(X[s])
        F[s] = np.concatenate([P, A], 1).astype(np.float64); F[s] = np.nan_to_num(F[s], nan=0.0, posinf=0.0, neginf=0.0)
        keys = list(k1) + k2; F['std|' + s] = St
        log(f'features {s}: {F[s].shape} + std {St.shape} in {time.time() - t:.0f} s')
    out = {f'x|{s}': X[s] for s in SETS}; out.update({f'y|{s}': Y[s] for s in NAT}); out.update({f'idx|{s}': idx[s] for s in NAT})
    out.update({f'base|{e}': E[e][1] for e in EDITS}); out.update({f'F|{s}': F[s] for s in SETS}); out.update({f'S|{s}': F['std|' + s] for s in SETS})
    np.savez_compressed(OUT / f'{T}_prep.npz', **out)
    meta = dict(node='B10', per_class=per, seed=SEED, N=J, sizes={s: int(len(v)) for s, v in idx.items()}, edits=einfo, keys=keys,
                n_b9_pool=len(k1), std_keys=STD_NAMES, halfplanes=hpinfo)
    (OUT / f'{T}_prep.json').write_text(json.dumps(clean(meta), indent=1))
    log(f'wrote {T}_prep.npz / .json ({len(keys)} pool inputs, {len(k1)} B9 + {len(k2)} azimuthal)')
    # jax-env networks
    L = {}
    for nm in JAXNETS:
        t = time.time(); run = jax_runner(nm)
        for s in SETS: L[f'{nm}|{s}'] = run(X[s]).astype(np.float32)
        acc = {s: float((L[f'{nm}|{s}'].argmax(1) == Y[s]).mean()) for s in NAT}
        accall = float(np.mean(np.concatenate([L[f'{nm}|{s}'].argmax(1) == Y[s] for s in NAT])))
        log(f'{nm}: accuracy full sample {accall * 100:.2f} %; ' + ', '.join(f'{s} {v * 100:.2f}' for s, v in acc.items()) + f' ({time.time() - t:.0f} s)')
    np.savez_compressed(OUT / f'{T}_logits_jax.npz', **L); log(f'wrote {T}_logits_jax.npz')


def jax_runner(nm):
    if nm.startswith('jedi'):
        from interp.jedi_core import Jedi
        j = Jedi(64, '3-feature', untrained=nm.endswith('u0'), seed=0); return lambda x: j.run(x)['logits']
    from interp.x9_efn_pfn import Flow
    import copy
    F = Flow('pfn_64-64-32_s1' if nm == 'pfn_s1' else 'pfn_64-64-32_s0')
    if nm == 'pfn_u0':
        r = np.random.default_rng(317000); U = copy.copy(F)
        hu = lambda W: r.uniform(-1, 1, W.shape) * np.sqrt(6.0 / W.shape[0])
        U.phi = [(hu(W), np.zeros_like(b)) for W, b in F.phi]; U.F = [(hu(W), np.zeros_like(b)) for W, b in F.F]
        U.out = (hu(F.out[0]), np.zeros_like(F.out[1])); F = U
    return lambda x: F.run(x, keep=('logits',))['logits']


# ================================================================================================ torch (conda env)
def torch_nets(smoke):
    import importlib.util, torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    T = tag(smoke); Z = np.load(OUT / f'{T}_prep.npz'); X = {s: Z[f'x|{s}'] for s in SETS}; Y = {s: Z[f'y|{s}'] for s in NAT}
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    L, info = {}, {}
    for nm, (run, s) in TORCH.items():
        t = time.time(); d = SMALL / run / f'seed{s if isinstance(s, int) else 0}'; arch = json.loads((d / 'arch.json').read_text())
        assert not arch['cls_token'] and not arch['inputs'].get('recentre') and not arch['inputs'].get('aug') and arch['n_particles'] == 64
        if not isinstance(s, int): torch.manual_seed(5000)
        net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
        if isinstance(s, int): net.load_state_dict(torch.load(d / 'best.pt', map_location='cpu'))
        net.eval(); sh = np.asarray(arch['inputs']['shift'], np.float32); sc = np.asarray(arch['inputs']['scale'], np.float32)

        def f(x, batch=1000):
            out = []
            with torch.no_grad():
                for i in range(0, len(x), batch):
                    xb = np.asarray(x[i:i + batch], np.float32); m = xb[..., 0] > 0; xn = ((xb - sh) / sc) * m[..., None]
                    out.append(net(torch.from_numpy(xn.astype(np.float32)), torch.from_numpy(m)).double().numpy())
            return np.concatenate(out)
        for st in SETS: L[f'{nm}|{st}'] = f(X[st]).astype(np.float32)
        acc = {st: float((L[f'{nm}|{st}'].argmax(1) == Y[st]).mean()) for st in NAT}
        ref = json.loads((d / 'test.json').read_text()).get('accuracy') if isinstance(s, int) else None
        info[nm] = dict(run=run, seed=s, dir=str(d), ref_test_acc=ref, acc=acc)
        log(f'{nm} ({run} seed {s}): accuracy ' + ', '.join(f'{k} {v * 100:.2f}' for k, v in acc.items()) + f'; reference test acc {ref}; {time.time() - t:.0f} s')
    np.savez_compressed(OUT / f'{T}_logits_torch.npz', **L); (OUT / f'{T}_torch.json').write_text(json.dumps(clean(info), indent=1))
    log(f'wrote {T}_logits_torch.npz')


# ================================================================================================ fit (jax env; sklearn)
def load_logits(T, nm):
    p = OUT / f'{T}_logits_{"torch" if nm in TORCH else "jax"}.npz'; Z = np.load(p)
    return {s: Z[f'{nm}|{s}'].astype(np.float64) for s in SETS}


def run_net(nm, smoke):
    from interp import b9_programs_across_nets as B9
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import Lasso
    from sklearn.metrics import cohen_kappa_score
    import warnings; warnings.filterwarnings('ignore')
    T = tag(smoke); Z = np.load(OUT / f'{T}_prep.npz'); meta0 = json.loads((OUT / f'{T}_prep.json').read_text())
    keys = meta0['keys']; nb9 = meta0['n_b9_pool']; skeys = meta0['std_keys']
    F = {s: Z[f'F|{s}'] for s in SETS}; S = {s: Z[f'S|{s}'] for s in SETS}; Y = {s: Z[f'y|{s}'] for s in NAT}
    base = {e: Z[f'base|{e}'] for e in EDITS}
    L = load_logits(T, nm); NC = {s: L[s].argmax(1) for s in SETS}
    R = dict(node='B10', network=nm, family=FAMILY[fam(nm)], trained=nm in TRAINED, smoke=smoke)
    R['accuracy'] = {s: dict(acc=float((NC[s] == Y[s]).mean()), ci=B9.wilson(int((NC[s] == Y[s]).sum()), len(Y[s]))) for s in NAT}
    allc = np.concatenate([NC[s] == Y[s] for s in NAT]); R['accuracy']['all'] = dict(acc=float(allc.mean()), ci=B9.wilson(int(allc.sum()), len(allc)), N=len(allc))
    R['call_frac_test'] = (np.bincount(NC['test'], minlength=5) / len(NC['test'])).tolist()
    R['net_flip_rate'] = {e: float((NC[e] != NC['test'][base[e]]).mean()) for e in EDITS}
    log(f'{nm}: accuracy all {allc.mean() * 100:.2f} %, test {R["accuracy"]["test"]["acc"] * 100:.2f} %; call fractions {np.round(R["call_frac_test"], 3).tolist()}; '
        f'network flip rates % ' + json.dumps({e: round(v * 100, 2) for e, v in R['net_flip_rate'].items()}))
    calls, meta = {}, []
    ev = ['dev', 'test'] + EDITS

    def add(name, family, bits, pred_fn, extra=None):
        c = {s: np.asarray(pred_fn(s), np.int8) for s in ev}
        for s in ev: calls[f'{name}|{s}'] = c[s]
        sd = float((c['dev'] == NC['dev']).mean()); st = float((c['test'] == NC['test']).mean())
        meta.append(dict(name=name, family=family, bits=float(bits), same_call_dev=sd, same_call_test=st,
                         kappa_test=float(cohen_kappa_score(c['test'], NC['test'])), **(extra or {})))
        log(f'{nm}: {name:20s} ({family:9s}) bits {bits:9.0f}  same call dev {sd * 100:6.2f}  test {st * 100:6.2f}')

    kmax = 8 if smoke else 32; R['program_terms'] = {}
    # (1) programs: three repeated fits on disjoint fit sets, full 140-input pool
    for f, fs in enumerate(FITS):
        t = time.time(); fits, Q = B9.mars_path(F[fs], L[fs], F['dev'], L['dev'], keys, kmax)
        log(f'{nm}: program fit {f} MARS path to {kmax} terms per logit in {time.time() - t:.0f} s')
        R['program_terms'][f'fit{f}'] = [[B9.term_record(tm, Q['fit']) for tm in terms] for terms, _ in fits]
        for K in KS:
            if K > kmax: continue
            parts, bits = B9.mars_prefix(fits, Q, L[fs], keys, K)
            add(f'prog{f}_K{K}', 'prog', bits, lambda s, p=parts: B9.mars_predict(p, F[s], keys).argmax(1), dict(fit=f, K=K))
    # B9 pool only (fit0)
    kb = keys[:nb9]; fits, Q = B9.mars_path(F['fit0'][:, :nb9], L['fit0'], F['dev'][:, :nb9], L['dev'], kb, kmax)
    for K in KS:
        if K > kmax: continue
        parts, bits = B9.mars_prefix(fits, Q, L['fit0'], kb, K)
        add(f'b9pool_K{K}', 'b9pool', bits, lambda s, p=parts: B9.mars_predict(p, F[s][:, :nb9], kb).argmax(1), dict(K=K))
    # controls (K16 or kmax)
    kc = min(16, kmax)
    perm = np.random.default_rng(3176).permutation(len(L['fit0']))
    fs_, Qs = B9.mars_path(F['fit0'], L['fit0'][perm], F['dev'], L['dev'], keys, kc)
    p_, b_ = B9.mars_prefix(fs_, Qs, L['fit0'][perm], keys, kc); add(f'shuf_K{kc}', 'shuffled', b_, lambda s, p=p_: B9.mars_predict(p, F[s], keys).argmax(1))
    rk = np.sort(np.random.default_rng(3177).choice(len(keys), 6, replace=False)); kr = [keys[i] for i in rk]
    fr, Qr = B9.mars_path(F['fit0'][:, rk], L['fit0'], F['dev'][:, rk], L['dev'], kr, kc)
    p_, b_ = B9.mars_prefix(fr, Qr, L['fit0'], kr, kc)
    add(f'rfeat_K{kc}', 'randfeat', b_, lambda s, p=p_: B9.mars_predict(p, F[s][:, rk], kr).argmax(1), dict(keys=kr))
    Tt = lambda y: np.log(0.95 * np.eye(5)[y] + 0.01)
    ft, Qt = B9.mars_path(F['fit0'], Tt(Y['fit0']), F['dev'], Tt(Y['dev']), keys, kc)
    p_, b_ = B9.mars_prefix(ft, Qt, Tt(Y['fit0']), keys, kc); add(f'truth_K{kc}', 'truth', b_, lambda s, p=p_: B9.mars_predict(p, F[s], keys).argmax(1))
    R['truth_terms'] = [[B9.term_record(tm, Qt['fit']) for tm in terms] for terms, _ in ft]
    log(f'{nm}: controls done')
    # (2) baselines on fit0
    ksm = 8 if smoke else 32
    fsd, Qsd = B9.mars_path(S['fit0'], L['fit0'], S['dev'], L['dev'], skeys, ksm)
    for K in KSTD:
        if K > ksm: continue
        p_, b_ = B9.mars_prefix(fsd, Qsd, L['fit0'], skeys, K)
        add(f'std_K{K}', 'std', b_, lambda s, p=p_: B9.mars_predict(p, S[s], skeys).argmax(1), dict(K=K))
    nin = len(keys); lb = float(np.log2(nin))
    for dpt in range(2, 13):
        m = DecisionTreeClassifier(max_depth=dpt, random_state=0).fit(F['fit0'], NC['fit0']); tr = m.tree_
        add(f'cart_d{dpt}', 'cart', (tr.node_count - tr.n_leaves) * (lb + 32) + tr.n_leaves * np.log2(5), lambda s, m=m: m.predict(F[s]))
    for fam_, XX, ni in ((('gbdt_std', S, len(skeys)), ('gbdt_pool', F, nin)) if len(np.unique(NC['fit0'])) > 1 else ()):
        for q in ((4, 6, 8) if fam_ == 'gbdt_std' else (6,)):
            qz = B9.quantiser(XX['fit0'], q); Fq = {s: qz(XX[s]) for s in ['fit0'] + ev}
            for lv in (4, 8, 16):
                for it in (3, 10, 30, 100):
                    m = HistGradientBoostingClassifier(learning_rate=0.5, max_iter=it, max_leaf_nodes=lv, early_stopping=False, random_state=0)
                    m.fit(Fq['fit0'], NC['fit0'])
                    add(f'{fam_}_q{q}_L{lv}_T{it}', fam_, B9.hgb_bits_q(m, ni, q), lambda s, m=m, Fq=Fq: m.predict(Fq[s]))
    mu, sdv = F['fit0'].mean(0), F['fit0'].std(0) + 1e-12; Zs = {s: (F[s] - mu) / sdv for s in ['fit0'] + ev}
    for ia, al in enumerate(np.logspace(-3, 0, 10)):
        ms = [Lasso(alpha=al, max_iter=5000).fit(Zs['fit0'], L['fit0'][:, k]) for k in range(5)]
        nz = sum(int((np.abs(m.coef_) > 0).sum()) for m in ms)
        add(f'lasso_a{ia}', 'lasso', nz * (32 + lb) + 5 * 32, lambda s, ms=ms: np.stack([m.predict(Zs[s]) for m in ms], 1).argmax(1), dict(alpha=float(al), nonzero=nz))
    R['surrogates'] = meta
    out = {f'netcall|{s}': np.asarray(NC[s], np.int8) for s in SETS}; out.update(calls)
    np.savez_compressed(OUT / f'{T}_net_{nm}.npz', **out); (OUT / f'{T}_net_{nm}.json').write_text(json.dumps(clean(R), indent=1))
    log(f'{nm}: wrote {T}_net_{nm}.npz/.json')


# ================================================================================================ aggregation
def agg(smoke):
    from interp import b9_programs_across_nets as B9
    T = tag(smoke); NB = 100 if smoke else 1000; kc = 8 if smoke else 16; kmax = 8 if smoke else 32
    P0 = json.loads((OUT / f'{T}_prep.json').read_text()); Zp = np.load(OUT / f'{T}_prep.npz')
    base = {e: Zp[f'base|{e}'] for e in EDITS}; yt = Zp['y|test']; Jt = len(yt)
    names = [n for n in TRAINED + UNTR if (OUT / f'{T}_net_{n}.json').exists()]; log('networks found:', names)
    RJ = {n: json.loads((OUT / f'{T}_net_{n}.json').read_text()) for n in names}; ZZ = {n: dict(np.load(OUT / f'{T}_net_{n}.npz')) for n in names}
    tn = [n for n in TRAINED if n in names]
    rng = np.random.default_rng(3179); C = np.stack([np.bincount(rng.integers(0, Jt, Jt), minlength=Jt) for _ in range(NB)]).astype(np.float32)
    Cs = {'test': C, **{e: C[:, base[e]] for e in EDITS}}
    bm = lambda s, v: (Cs[s] @ np.asarray(v, np.float32)) / np.maximum(Cs[s].sum(1), 1)
    ci, bootp, holm, mcnemar = B9.ci, B9.bootp, B9.holm, B9.mcnemar
    RES = dict(node='B10', procedure=__doc__, networks=names, trained=tn, edit_label=EDIT_LABEL, N_test=Jt,
               N_edit={e: int(len(base[e])) for e in EDITS}, edit_info=P0['edits'], sizes=P0['sizes'], n_pool=len(P0['keys']),
               n_b9_pool=P0['n_b9_pool'], halfplanes=P0['halfplanes'], per_network={}, matched=[], transfer={})

    def ind(n, sname, met):
        """per-jet indicator: natural same call (met = 'nat') or edit hit; plus the network-flip indicator for edits"""
        Z = ZZ[n]; nc = Z['netcall|test']; s = Z[f'{sname}|test'] if f'{sname}|test' in Z else None
        if met == 'nat': return s == nc, None
        b = base[met]; nf, sf, ag, hit = B9.edit_ind(nc[b], Z[f'netcall|{met}'], s[b], Z[f'{sname}|{met}'])
        return hit, nf

    def ext_ind(n, donor, sname, met):
        """indicator of donor network's surrogate sname against network n's calls"""
        Z, D = ZZ[n], ZZ[donor]; nc = Z['netcall|test']
        if met == 'nat': return D[f'{sname}|test'] == nc, None
        b = base[met]; nf, sf, ag, hit = B9.edit_ind(nc[b], Z[f'netcall|{met}'], D[f'{sname}|test'][b], D[f'{sname}|{met}'])
        return hit, nf

    # ---------------------------------------------------------------- per network
    for n in names:
        R, Z = RJ[n], ZZ[n]; nc = Z['netcall|test']; o = dict(family=R['family'], trained=R['trained'], accuracy=R['accuracy'],
                                                              net_flip_rate=R['net_flip_rate'], call_frac_test=R['call_frac_test'], surrogates={})
        o['net_flips'] = {e: int((Z[f'netcall|{e}'] != nc[base[e]]).sum()) for e in EDITS}
        for m in R['surrogates']:
            sname = m['name']; s = Z[f'{sname}|test']; i_ = s == nc
            d = dict(family=m['family'], bits=m['bits'], same_call_dev=m['same_call_dev'], same_call=float(i_.mean()),
                     same_call_ci=ci(bm('test', i_)), kappa=m['kappa_test'], edits={})
            for e in EDITS:
                b = base[e]; nf, sf, ag, hit = B9.edit_ind(nc[b], Z[f'netcall|{e}'], s[b], Z[f'{sname}|{e}'])
                d['edits'][e] = dict(recall=float(hit.sum() / nf.sum()) if nf.sum() else None, recall_ci=ci(bm(e, hit) / np.maximum(bm(e, nf), 1e-9)) if nf.sum() else [None, None],
                                     precision=float(hit.sum() / sf.sum()) if sf.sum() else None, kappa=B9.kappa(nf, sf), agree=float(ag.mean()),
                                     net_flips=int(nf.sum()), sur_flips=int(sf.sum()))
            o['surrogates'][sname] = d
        # fit-to-fit spread (programs K, three fits) and fit-mean intervals
        o['fit_spread'] = {}
        for K in KS:
            if K > kmax: continue
            nmK = [f'prog{f}_K{K}' for f in range(3)]
            if not all(x in o['surrogates'] for x in nmK): continue
            row = dict(bits=[o['surrogates'][x]['bits'] for x in nmK])
            for met in ['nat'] + EDITS:
                vals = [o['surrogates'][x]['same_call'] if met == 'nat' else o['surrogates'][x]['edits'][met]['recall'] for x in nmK]
                ii = [ind(n, x, met) for x in nmK]; hm = np.mean([v[0] for v in ii], 0)
                bt = bm('test' if met == 'nat' else met, hm) / (1.0 if met == 'nat' else np.maximum(bm(met, ii[0][1]), 1e-9))
                vv = [v for v in vals if v is not None]
                row[met] = dict(per_fit=vals, mean=float(np.mean(vv)) if vv else None, range=float(np.ptp(vv)) if vv else None, mean_jet_ci=ci(bt) if vv else [None, None])
                if met != 'nat':
                    row[met]['kappa_per_fit'] = [o['surrogates'][x]['edits'][met]['kappa'] for x in nmK]
            o['fit_spread'][f'K{K}'] = row
        # same call at DL <= 7 kbit (per fit: member chosen by dev same-call among that fit's K members with bits <= target)
        o['at_7kbit'] = []
        for f in range(3):
            mem = [m for m in R['surrogates'] if m['family'] == 'prog' and m.get('fit') == f and m['bits'] <= DL_TARGET]
            if mem:
                b_ = max(mem, key=lambda m: m['same_call_dev']); o['at_7kbit'].append(dict(fit=f, member=b_['name'], bits=b_['bits'],
                                                                                            same_call=o['surrogates'][b_['name']]['same_call'],
                                                                                            same_call_ci=o['surrogates'][b_['name']]['same_call_ci']))
        RES['per_network'][n] = o
        p16 = o['surrogates'].get(f'prog0_K{kc}', {})
        log(f'{n}: acc all {R["accuracy"]["all"]["acc"] * 100:.2f} %; prog0 K{kc} same call {p16.get("same_call", 0) * 100:.2f} % ({p16.get("bits", 0):.0f} bits); '
            + '; '.join(f'{e} recall {p16["edits"][e]["recall"] * 100:.1f} %' for e in EDITS if p16 and p16['edits'][e]['recall'] is not None)
            + f'; at 7 kbit {[round(v["same_call"] * 100, 2) for v in o["at_7kbit"]]}')
    # ---------------------------------------------------------------- matched DL: program fit0 vs each baseline family; Holm over networks x families
    BF = ['std', 'gbdt_std', 'gbdt_pool', 'cart', 'lasso', 'b9pool']
    for K in KS:
        if K > kmax: continue
        for met in ['nat', 'shift', 'turn10', 'soft']:
            rows = []
            for n in tn:
                R = RJ[n]; meta = {m['name']: m for m in R['surrogates']}; pn = f'prog0_K{K}'
                if pn not in meta: continue
                for fam_ in BF:
                    mem = [m for m in R['surrogates'] if m['family'] == fam_]
                    if not mem: continue
                    ok = [m for m in mem if m['bits'] <= meta[pn]['bits'] * 1.0001]
                    bsel = max(ok, key=lambda m: m['same_call_dev']) if ok else min(mem, key=lambda m: m['bits'])
                    a, nf = ind(n, pn, met); b_, _ = ind(n, bsel['name'], met)
                    if met == 'nat': dd = bm('test', a) - bm('test', b_); st = None
                    else:
                        if nf.sum() == 0: continue
                        a, b_ = a[nf], b_[nf]; dd = (bm(met, ind(n, pn, met)[0]) - bm(met, ind(n, bsel['name'], met)[0])) / np.maximum(bm(met, nf), 1e-9)
                    mc = mcnemar(a, b_)
                    rows.append(dict(network=n, K=K, metric=met, family=fam_, program_bits=meta[pn]['bits'], baseline=bsel['name'],
                                     baseline_bits=bsel['bits'], baseline_within_bits=bool(ok), prog=float(a.mean()), base=float(b_.mean()),
                                     diff=float(a.mean() - b_.mean()), diff_ci=ci(dd), mcnemar=mc))
            hp = holm([r['mcnemar']['p'] for r in rows])
            for r, h in zip(rows, hp): r['p_holm'] = h
            RES['matched'] += rows
    summ = {}
    for r in RES['matched']:
        k = f'K{r["K"]}|{r["family"]}|{r["metric"]}'; s = summ.setdefault(k, dict(n=0, prog_higher_sig=0, base_higher_sig=0, diffs={}))
        s['n'] += 1; s['diffs'][r['network']] = r['diff']
        if r['p_holm'] is not None and r['p_holm'] < 0.05: s['prog_higher_sig' if r['diff'] > 0 else 'base_higher_sig'] += 1
    RES['matched_summary'] = summ
    # ---------------------------------------------------------------- transfer
    ps = f'prog0_K{kc}'; tsum = {}; pairs = []
    for B_ in tn:
        own_names = [f'prog{f}_K{kc}' for f in range(3) if f'prog{f}_K{kc}|test' in ZZ[B_]]
        row = dict(B=B_, family=fam(B_))
        for met in ['nat'] + EDITS:
            st = 'test' if met == 'nat' else met
            own = [ind(B_, x, met) for x in own_names]; own_vals = [float(v[0].sum() / (len(v[0]) if met == 'nat' else max(v[1].sum(), 1))) for v in own]
            nf = own[0][1]
            if met != 'nat' and nf.sum() == 0: row[met] = None; continue
            den = (lambda: 1.0) if met == 'nat' else (lambda: np.maximum(bm(met, nf), 1e-9))
            own_mean_ind = np.mean([v[0] for v in own], 0)
            donors = [A for A in tn if A != B_]; same = [A for A in donors if fam(A) == fam(B_)]; other = [A for A in donors if fam(A) != fam(B_)]
            tr = {A: ext_ind(B_, A, ps, met)[0] for A in donors}
            tru = ind(B_, f'truth_K{kc}', met)[0]
            norm = (lambda v: float(v.mean())) if met == 'nat' else (lambda v: float(v.sum() / nf.sum()))
            tr_all = np.mean([tr[A] for A in donors], 0); tr_oth = np.mean([tr[A] for A in other], 0) if other else None
            d1 = (bm(st, own_mean_ind) - bm(st, tr_all)) / den(); d2 = (bm(st, tr_all) - bm(st, tru)) / den(); d3 = (bm(st, own_mean_ind) - bm(st, tru)) / den()
            rr = dict(own_per_fit=own_vals, own_mean=float(np.mean(own_vals)), own_range=float(np.ptp(own_vals)),
                      transfer={A: norm(tr[A]) for A in donors}, transfer_mean=norm(tr_all),
                      transfer_same_arch=float(np.mean([norm(tr[A]) for A in same])) if same else None,
                      transfer_other_arch=norm(tr_oth) if other else None, truth=norm(tru),
                      own_minus_transfer=float(np.mean(own_vals) - norm(tr_all)), own_minus_transfer_ci=ci(d1), own_minus_transfer_p=bootp(d1),
                      transfer_minus_truth=float(norm(tr_all) - norm(tru)), transfer_minus_truth_ci=ci(d2), transfer_minus_truth_p=bootp(d2),
                      own_minus_truth=float(np.mean(own_vals) - norm(tru)), own_minus_truth_ci=ci(d3), own_minus_truth_p=bootp(d3))
            rr['own_minus_transfer_beyond_fit_range'] = bool(rr['own_minus_transfer'] > rr['own_range'])
            # McNemar own fit0 vs best-matching donor of another architecture (paired hits among network flips / same-call jets)
            if other:
                a0 = own[0][0]; best = max(other, key=lambda A: norm(tr[A])); bb = tr[best]
                if met != 'nat': a0, bb = a0[nf], bb[nf]
                rr['mcnemar_own0_vs_best_other'] = dict(donor=best, **mcnemar(a0, bb))
            row[met] = rr
        pairs.append(row)
    for met in ['nat'] + EDITS:
        for key in ('own_minus_transfer_p', 'transfer_minus_truth_p', 'own_minus_truth_p'):
            rows = [r[met] for r in pairs if r.get(met)]
            hp = holm([r[key] for r in rows])
            for r, h in zip(rows, hp): r[key.replace('_p', '_p_holm')] = h
        rows = [r[met] for r in pairs if r.get(met) and 'mcnemar_own0_vs_best_other' in r[met]]
        hp = holm([r['mcnemar_own0_vs_best_other']['p'] for r in rows])
        for r, h in zip(rows, hp): r['mcnemar_own0_vs_best_other']['p_holm'] = h
    RES['transfer'] = dict(program=ps, rows=pairs)
    for r in pairs:
        for met in ('nat', 'shift', 'turn10', 'ctrl10'):
            v = r.get(met)
            if v: log(f'transfer {r["B"]:9s} {met:7s}: own {v["own_mean"] * 100:6.2f} (fits {[round(x * 100, 1) for x in v["own_per_fit"]]}) transfer {v["transfer_mean"] * 100:6.2f} '
                      f'(other arch {None if v["transfer_other_arch"] is None else round(v["transfer_other_arch"] * 100, 2)}) truth {v["truth"] * 100:6.2f}; '
                      f'own-transfer {v["own_minus_transfer"] * 100:+.2f} CI [{v["own_minus_transfer_ci"][0] * 100:+.2f}, {v["own_minus_transfer_ci"][1] * 100:+.2f}] Holm p {v["own_minus_transfer_p_holm"]:.3g}; beyond fit range {v["own_minus_transfer_beyond_fit_range"]}')
    # ---------------------------------------------------------------- predictions check (descriptive)
    pr = {}
    for n in tn:
        o = RES['per_network'][n]; a7 = [v['same_call'] for v in o['at_7kbit']]
        g = [r for r in RES['matched'] if r['network'] == n and r['K'] == kc and r['metric'] == 'nat' and r['family'] == 'gbdt_std']
        t = next((r for r in pairs if r['B'] == n), {})
        pr[n] = dict(same_call_at_7kbit_per_fit=a7, min_at_7kbit=min(a7) if a7 else None,
                     prog_minus_gbdt_std_K16=g[0]['diff'] if g else None, prog_minus_gbdt_std_p_holm=g[0]['p_holm'] if g else None,
                     shift_own_transfer_truth=[t['shift']['own_mean'], t['shift']['transfer_mean'], t['shift']['truth']] if t.get('shift') else None,
                     turn10_recall_fits=o['fit_spread'].get(f'K{kc}', {}).get('turn10', {}).get('per_fit'),
                     ctrl10_recall_fits=o['fit_spread'].get(f'K{kc}', {}).get('ctrl10', {}).get('per_fit'))
    RES['predictions_table'] = pr
    (OUT / f'{T}.json').write_text(json.dumps(clean(RES), indent=1)); log(f'wrote {T}.json')
    try:
        plot(RES, OUT / f'{T}.png', kc)
    except Exception as ex:
        import traceback; traceback.print_exc(); log('plot failed', repr(ex))


SHORT = dict(jedi='JEDI-\nlinear', ds='Deep\nSets', linf='Lin-\nformer', tf='trans-\nformer', pfn='PFN')
COL = dict(jedi='k', ds='tab:blue', linf='tab:orange', tf='tab:red', pfn='tab:green')


def plot(RES, path, kc=16):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for n in RES['trained']:
        o = RES['per_network'][n]; fs = o['fit_spread']; Ks = sorted(int(k[1:]) for k in fs)
        b = [np.mean(fs[f'K{K}']['bits']) for K in Ks]; v = [fs[f'K{K}']['nat']['mean'] * 100 for K in Ks]
        lo = [min(fs[f'K{K}']['nat']['per_fit']) * 100 for K in Ks]; hi = [max(fs[f'K{K}']['nat']['per_fit']) * 100 for K in Ks]
        ax[0].errorbar(b, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], marker='o', ms=3, capsize=2, color=COL[fam(n)],
                       ls='-' if n.endswith('s0') or n == 'jedi_off' else '--', lw=2.2 if n == 'jedi_off' else 1.2,
                       label=f'{FAMILY[fam(n)]} {"" if n == "jedi_off" else "seed " + n[-1]} ({o["accuracy"]["all"]["acc"] * 100:.1f} % acc.)')
    ax[0].axvline(DL_TARGET, color='0.5', lw=0.8, ls=':'); ax[0].axhline(85, color='0.5', lw=0.8, ls=':')
    ax[0].set_xscale('log'); ax[0].set_xlabel('program size (description length, bits)'); ax[0].set_ylabel('same call as the network, test jets (%)')
    ax[0].set_title('If-statement programs (K = 4-32 terms per class): agreement vs size\n(mean of 3 fits; bars = fit-to-fit range)', fontsize=9.5)
    ax[0].legend(fontsize=6.5); ax[0].grid(alpha=0.3)
    rows = [r for r in RES['transfer']['rows'] if r.get('shift')]; x = np.arange(len(rows)); w = 0.27
    own = [r['shift']['own_mean'] * 100 for r in rows]; rg = [[(r['shift']['own_mean'] - min(r['shift']['own_per_fit'])) * 100 for r in rows],
                                                            [(max(r['shift']['own_per_fit']) - r['shift']['own_mean']) * 100 for r in rows]]
    ax[1].bar(x - w, own, w, yerr=rg, capsize=2, label="network's own programs (mean of 3 fits, range)", color='tab:blue')
    ax[1].bar(x, [r['shift']['transfer_mean'] * 100 for r in rows], w, label="other networks' programs (mean)", color='tab:orange')
    ax[1].bar(x + w, [r['shift']['truth'] * 100 for r in rows], w, label='program fitted to truth labels', color='0.6')
    ax[1].set_xticks(x); ax[1].set_xticklabels([SHORT[fam(r['B'])] + ('' if r['B'] == 'jedi_off' else f'\nseed {r["B"][-1]}') for r in rows], fontsize=7.5)
    ax[1].set_ylabel("network call changes reproduced (%)")
    ax[1].set_title(f'Shift (all particles Δη + 0.06 rad): recall of each network\'s call changes (K = {kc})', fontsize=9.5)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); log('wrote', path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prep', 'torch', 'fit', 'agg', 'plot']); ap.add_argument('--nets', default='')
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == 'prep': prep(a.smoke)
    elif a.mode == 'torch': torch_nets(a.smoke)
    elif a.mode == 'fit':
        for nm in a.nets.split(','):
            t = time.time(); run_net(nm, a.smoke); log(f'{nm} done in {time.time() - t:.0f} s')
    elif a.mode == 'plot': plot(json.loads((OUT / f'{tag(a.smoke)}.json').read_text()), OUT / f'{tag(a.smoke)}.png', 8 if a.smoke else 16)
    else: agg(a.smoke)
