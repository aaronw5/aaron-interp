"""E10 (JEDI-linear 64 p, 3-feature): at which stage does each jet observable become linearly readable, and does that stage
increase with the observable's algebraic depth?

Modes (argv[1]):
  read    (default) readability of ~45 observables at the stages P, g, S, h4, h5, z (live units only):
          ridge (RidgeCV on standardised inputs) and a 1-hidden-layer MLP (64 units, seed 0) as the nonlinear ceiling,
          fixed 50/50 split (rng 0); targets clipped to the train half's 0.5-99.5 % range.
          Controls: the untrained network (Jedi(untrained=True): weights redrawn from the trained mean / std, seed 0),
          ridge on the raw inputs (flattened x), class-only (class means). Within-class R² (global probe scored inside
          each class, averaged over the 5 classes; and a ridge fitted inside each class).
          -> results/jedi/e10_layers.json
  causal  three input edits (depth 0, 1, 2) that change one observable at a fixed shallower one; the ridge probe of the
          observable at its first-appearance stage applied to edited vs original activations (Δprobe) against the
          observable recomputed on the edited jets with jetdistill.observables.compute (Δobs). Control: the whole jet
          rotated about the axis (every observable used here is rotation invariant). -> results/jedi/e10_causal.json
  plot    figure from the JSON (needs matplotlib) -> results/jedi/e10_layers.png
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
STG = ('P', 'g', 'S', 'h4', 'h5', 'z')
PER_CLASS = int(os.environ.get('E10_PER_CLASS', 4000))
NEVER = len(STG)          # first-stage index when the threshold is never reached

# (id, depth, slot-order quantity, justification)
OBS = [
    ('sum_pt', 0, 0, 'Σ pT: linear sum over particles'),
    ('n_particles', 0, 0, 'Σ 1[pT>0]: count'),
    ('n_pt_above_1', 0, 0, 'Σ 1[pT>1 GeV]: count above a cut'),
    ('n_pt_above_5', 0, 0, 'Σ 1[pT>5 GeV]: count above a cut'),
    ('n_pt_above_50', 0, 0, 'Σ 1[pT>50 GeV]: count above a cut'),
    ('n_dr_0_0p05', 0, 0, 'Σ 1[ΔR<0.05]: annulus count'),
    ('n_dr_0p1_0p2', 0, 0, 'Σ 1[0.1≤ΔR<0.2]: annulus count'),
    ('n_dr_0p4_up', 0, 0, 'Σ 1[ΔR≥0.4]: annulus count'),
    ('sum_ptr', 0, 0, 'Σ pT·ΔR (= girth·ΣpT): linear sum'),
    ('sum_ptr2', 0, 0, 'Σ pT·ΔR² (= girth2·ΣpT): linear sum'),
    ('sum_r', 0, 0, 'Σ ΔR over real particles, no pT weight: linear sum'),
    ('sum_pt_top5', 0, 1, 'pT₀+…+pT₄: linear in 5 slots'),
    ('pt_0', 0, 1, 'pT of slot 0: one input'),
    ('pt_1', 0, 1, 'pT of slot 1: one input'),
    ('girth', 1, 0, 'Σ pT ΔR / Σ pT: ratio of two sums'),
    ('girth2', 1, 0, 'Σ pT ΔR² / Σ pT: ratio of two sums'),
    ('mass', 1, 0, 'm, m² ≈ M0·M2 − |M1|²: product of sums (E6 owns the mechanism)'),
    ('mass_sq', 1, 0, 'm² ≈ M0·M2 − |M1|²: product of sums'),
    ('e2', 1, 0, 'Σ_{i<j} pTi pTj ΔRij / (ΣpT)²: pair sum over a product'),
    ('pt_dispersion', 1, 0, '√Σ pT² / Σ pT: ratio of two sums'),
    ('LHA', 1, 0, 'Σ pT √(ΔR/0.8) / Σ pT: ratio of two sums'),
    ('pt_entropy', 1, 0, 'ln ΣpT − Σ pT ln pT / ΣpT: ratio of two sums'),
    ('psi_0p3', 1, 0, 'Σ_{ΔR<0.3} pT / Σ pT: ratio of two sums'),
    ('z_dr_0p2_0p4', 1, 0, 'Σ_{0.2≤ΔR<0.4} pT / Σ pT: ratio of two sums'),
    ('z_0', 1, 1, 'pT₀ / Σ pT: slot over sum'),
    ('z_top5', 1, 1, '(pT₀+…+pT₄) / Σ pT: slot sum over sum'),
    ('pt1_over_pt0', 1, 1, 'pT₁ / pT₀: ratio of two slots'),
    ('dr01', 1, 1, 'ΔR between slots 0 and 1: two-particle nonlinear'),
    ('e3', 2, 0, 'Σ_{i<j<k} zizjzk ΔRijΔRikΔRjk: three-point'),
    ('C2', 2, 0, 'e3 / e2²: ratio of depth-1/2 quantities'),
    ('D2', 2, 0, 'e3 / e2³: ratio of depth-1/2 quantities'),
    ('N2', 2, 0, '₂e3 / e2²: three-point ratio'),
    ('C3', 2, 0, 'e4·e2 / e3²: four-point ratio'),
    ('planar_flow', 2, 0, '4 det T / (tr T)²: ratio of products of second moments'),
    ('eccentricity', 2, 0, '1 − λ2/λ1: eigenvalue ratio of the pT-weighted tensor'),
    ('mass_over_sum_pt', 2, 0, 'm / ΣpT: depth-1 quantity over a sum'),
    ('tau1', 3, 0, 'τ1: distance to a k-means axis (minimisation)'),
    ('tau2', 3, 0, 'τ2: two k-means axes'),
    ('tau21', 3, 0, 'τ2/τ1: ratio of minimised quantities'),
    ('tau32', 3, 0, 'τ3/τ2: ratio of minimised quantities'),
    ('sd_mass', 3, 0, 'soft-drop mass: C/A clustering + grooming'),
    ('sd_zg', 3, 0, 'soft-drop z_g: clustering + grooming'),
    ('sd_rg', 3, 0, 'soft-drop R_g: clustering + grooming'),
    ('sj2_zsoft', 3, 0, 'softer of 2 k-means subjets, pT share'),
    ('sj2_dr', 3, 0, 'ΔR between 2 k-means subjet axes'),
]
IDS = [o[0] for o in OBS]


def derive(Q, x):
    """the observables of OBS from the precomputed / recomputed table Q (and the particles x for the unweighted Σ ΔR)"""
    pt = x[..., 0].astype(np.float64); r = np.hypot(x[..., 1], x[..., 2]).astype(np.float64)
    D = {k: np.asarray(Q[k], np.float64) for k in IDS if k in Q}
    D['sum_ptr'] = np.asarray(Q['girth'], np.float64) * Q['sum_pt']
    D['sum_ptr2'] = np.asarray(Q['girth2'], np.float64) * Q['sum_pt']
    D['sum_r'] = (r * (pt > 0)).sum(1)
    D['mass_sq'] = np.asarray(Q['mass'], np.float64) ** 2
    return {k: D[k] for k in IDS}


def split(J, seed=0):
    p = np.random.default_rng(seed).permutation(J); return p[:J // 2], p[J // 2:]


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300))


def within(y, p, cls):
    return float(np.mean([r2(y[cls == c], p[cls == c]) for c in range(5)]))


class Probe:
    """standardised inputs -> RidgeCV (alphas 1e-3..1e3, leave-one-out on the train half); target clipped to the train
    half's 0.5-99.5 % range"""
    def __init__(self, X, t):
        from sklearn.linear_model import RidgeCV
        self.lo, self.hi = np.percentile(t, [0.5, 99.5]); t = np.clip(t, self.lo, self.hi)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X - self.mu) / self.sd, t)

    def __call__(self, X):
        return self.m.predict((X - self.mu) / self.sd)


def mlp_fit(X, t, Xte):
    from sklearn.neural_network import MLPRegressor
    mu, sd = X.mean(0), X.std(0) + 1e-12; tm, ts = t.mean(), t.std() + 1e-12
    m = MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-4, max_iter=300, early_stopping=True, n_iter_no_change=15,
                     random_state=0).fit((X - mu) / sd, (t - tm) / ts)
    return m.predict((Xte - mu) / sd) * ts + tm


def live(R):
    return {s: np.flatnonzero(R[s].std(0) > 1e-6) for s in STG}


def env():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from interp.jedi_core import Jedi, sample
    return Jedi, sample


# ------------------------------------------------------------------------------------------------------------- read
def read():
    Jedi, sample = env(); t0 = time.time()
    x, y, Q = sample(per_class=PER_CLASS, seed=0)
    O = derive(Q, x); J = len(x); tr, te = split(J)
    nets = {'trained': Jedi(64), 'untrained': Jedi(64, untrained=True, seed=0)}
    A = {k: n.run(x, keep=STG + ('logits',)) for k, n in nets.items()}
    LV = {k: live(A[k]) for k in A}
    res = dict(N=J, per_class=PER_CLASS, split='rng(0) permutation, first half train', stages=STG,
               acc={k: float((A[k]['logits'].argmax(1) == y).mean()) for k in A},
               live={k: {s: int(len(v)) for s, v in LV[k].items()} for k in LV}, obs={})
    # live units of the untrained control for other seeds (is seed 0 typical?)
    res['untrained_live_other_seeds'] = {}
    for sd in (1, 2, 3):
        Ru = Jedi(64, untrained=True, seed=sd).run(x[:4000], keep=STG + ('logits',))
        res['untrained_live_other_seeds'][sd] = {s: int((Ru[s][:, :].std(0) > 1e-6).sum()) for s in STG}
    print(f'{J} jets; accuracy {res["acc"]}; live units {res["live"]}; untrained seeds 1-3 live {res["untrained_live_other_seeds"]}', flush=True)
    Xraw = x.reshape(J, -1).astype(np.float64); Xraw = Xraw[:, Xraw[tr].std(0) > 1e-9]
    for oid, depth, slot, why in OBS:
        t = O[oid]; lo, hi = np.percentile(t[tr], [0.5, 99.5]); tc = np.clip(t, lo, hi)
        e = dict(depth=depth, slot=slot, why=why, clip=[float(lo), float(hi)], R2={})
        cm = np.array([tc[tr][y[tr] == c].mean() for c in range(5)])
        e['class_only'] = r2(tc[te], cm[y[te]])
        pr = Probe(Xraw[tr], t[tr]); e['raw_lin'] = r2(tc[te], pr(Xraw[te]))
        for k in A:
            e['R2'][k] = {}
            for s in STG:
                u = LV[k][s]
                if len(u) == 0:
                    e['R2'][k][s] = dict(lin=0.0, mlp=0.0, lin_within=0.0, mlp_within=0.0, lin_inclass=0.0); continue
                X = A[k][s][:, u]
                p = Probe(X[tr], t[tr]); pl = p(X[te])
                pm = mlp_fit(X[tr], tc[tr], X[te])
                inc = []
                for c in range(5):
                    a_, b_ = tr[y[tr] == c], te[y[te] == c]
                    inc.append(r2(tc[b_], Probe(X[a_], t[a_])(X[b_])))
                e['R2'][k][s] = dict(lin=r2(tc[te], pl), mlp=r2(tc[te], pm), lin_within=within(tc[te], pl, y[te]),
                                     mlp_within=within(tc[te], pm, y[te]), lin_inclass=float(np.mean(inc)))
        for thr in (0.9, 0.7):
            L = [e['R2']['trained'][s]['lin'] for s in STG]; M = [e['R2']['trained'][s]['mlp'] for s in STG]
            e[f'first_lin_{thr}'] = next((i for i, v in enumerate(L) if v >= thr), NEVER)
            e[f'first_mlp_{thr}'] = next((i for i, v in enumerate(M) if v >= thr), NEVER)
        res['obs'][oid] = e
        Rt = e['R2']['trained']; Ru = e['R2']['untrained']
        f9, f7 = e['first_lin_0.9'], e['first_lin_0.7']
        print(f'{oid:16s} d{depth}{"s" if slot else " "} first≥.9 {STG[f9] if f9 < NEVER else "-":3s} ≥.7 {STG[f7] if f7 < NEVER else "-":3s} | lin '
              + ' '.join(f'{Rt[s]["lin"]:5.2f}' for s in STG) + ' | mlp ' + ' '.join(f'{Rt[s]["mlp"]:5.2f}' for s in STG)
              + ' | untr lin ' + ' '.join(f'{Ru[s]["lin"]:5.2f}' for s in STG)
              + f' | raw {e["raw_lin"]:5.2f} class {e["class_only"]:5.2f} within(lin@h5) {Rt["h5"]["lin_within"]:5.2f}  [{time.time()-t0:.0f}s]', flush=True)
    res['summary'] = summary(res)
    (OUT / 'e10_layers.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['summary'], indent=1)); print(f'done {time.time()-t0:.0f}s')


def summary(res):
    from scipy.stats import spearmanr
    E = res['obs']; S = {}
    for name, keep in (('all', lambda e: True), ('no_slot', lambda e: not e['slot'])):
        ids = [k for k, e in E.items() if keep(e)]; d = np.array([E[k]['depth'] for k in ids])
        S[name] = dict(n=len(ids))
        for key in ('first_lin_0.9', 'first_lin_0.7', 'first_mlp_0.9', 'first_mlp_0.7'):
            f = np.array([E[k][key] for k in ids]); rho, p = spearmanr(d, f)
            S[name][key] = dict(spearman=float(rho), p=float(p),
                                median_by_depth={int(q): float(np.median(f[d == q])) for q in range(4)},
                                never_by_depth={int(q): int((f[d == q] == NEVER).sum()) for q in range(4)},
                                n_by_depth={int(q): int((d == q).sum()) for q in range(4)})
        for net in ('trained', 'untrained'):
            for s in STG:
                v = np.array([E[k]['R2'][net][s]['lin'] for k in ids]); rho, p = spearmanr(d, v)
                S[name][f'spearman_depth_vs_lin_{net}_{s}'] = [float(rho), float(p)]
    return S


# ------------------------------------------------------------------------------------------------------------- causal
def rotate(x, th):
    c, s = np.cos(th), np.sin(th); y = x.copy()
    y[..., 1], y[..., 2] = c * x[..., 1] - s * x[..., 2], s * x[..., 1] + c * x[..., 2]
    y[x[..., 0] <= 0] = x[x[..., 0] <= 0]
    return y


def edit_split_softest(x, k=5):
    """split each of the k softest real particles into two collinear halves (pT/2 each, same Δη, Δφ), appended in pT
    order: multiplicity +k; ΣpT, the four-momentum (mass), every IRC-safe observable unchanged. Jets need ≤ 64-k real
    particles."""
    y = x.copy(); n = (x[..., 0] > 0).sum(1)
    for j in range(len(x)):
        m = n[j]; soft = x[j, m - k:m].copy(); soft[:, 0] /= 2
        y[j, m - k:m + k] = np.repeat(soft, 2, axis=0)
    return y


def edit_soft_outward(x, ptmax=10.0, f=2.0, cap=None):
    """particles with pT < ptmax GeV moved radially outward, ΔR -> f·ΔR (direction kept): ΣpT, multiplicity and the pT
    order unchanged. cap: ΔR -> min(f·ΔR, max(ΔR, cap)), so no particle is pushed past cap (99.9 % of data ΔR < 0.44)"""
    y = x.copy(); s = (x[..., 0] > 0) & (x[..., 0] < ptmax)
    r = np.hypot(x[..., 1], x[..., 2]); rn = f * r if cap is None else np.minimum(f * r, np.maximum(r, cap))
    k = np.where(s & (r > 0), rn / np.maximum(r, 1e-12), 1.0)
    y[..., 1] = x[..., 1] * k; y[..., 2] = x[..., 2] * k
    return y


def _mass(x):
    pt, eta, phi = (x[..., i].astype(np.float64) for i in range(3))
    px, py, pz, E = (pt * np.cos(phi)).sum(1), (pt * np.sin(phi)).sum(1), (pt * np.sinh(eta)).sum(1), (pt * np.cosh(eta)).sum(1)
    return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))


def edit_isomass(x, k=1, th=np.pi / 2):
    """slot k moved by angle th around the pT centroid c of the other particles, at the radius that keeps the exact jet
    mass (small-angle: m² ⊃ pT_k Σ_j pT_j |r_k − r_j|² = pT_k (M0' |r_k − c|² + const), so the circle |r_k − c| = const is
    iso-mass; the radius is then corrected by bisection on the exact massless four-vector mass). pT unchanged."""
    y = x.copy().astype(np.float64); pt = x[..., 0].astype(np.float64)
    w = pt.copy(); w[:, k] = 0
    c = (w[..., None] * x[..., 1:3]).sum(1) / w.sum(1, keepdims=True)
    d = x[:, k, 1:3] - c; rad = np.hypot(d[:, 0], d[:, 1]); ang = np.arctan2(d[:, 1], d[:, 0]) + th
    m0 = _mass(x)

    def place(sc):
        z = y.copy(); z[:, k, 1] = c[:, 0] + sc * rad * np.cos(ang); z[:, k, 2] = c[:, 1] + sc * rad * np.sin(ang); return z
    lo, hi = np.full(len(x), 0.5), np.full(len(x), 1.5)
    for _ in range(40):
        mid = (lo + hi) / 2; m = _mass(place(mid))
        up = m < m0; lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    return place((lo + hi) / 2).astype(np.float32)


def causal():
    Jedi, sample = env(); t0 = time.time()
    from jetdistill.observables import compute
    res = json.loads((OUT / 'e10_layers.json').read_text())
    x, y, Q = sample(per_class=PER_CLASS, seed=0); O = derive(Q, x); J = len(x); tr, te = split(J)
    net = Jedi(64); A = net.run(x, keep=STG); LV = live(A)
    probes = {}

    def probe(oid, s):
        if (oid, s) not in probes: probes[oid, s] = Probe(A[s][tr][:, LV[s]], O[oid][tr])
        return probes[oid, s]

    def obs_of(xe):
        return derive(compute(xe.astype(np.float64), 64), xe)

    plan = json.loads(os.environ.get('E10_PLAN', 'null')) or [
        dict(name='split5_softest', target='n_particles', fixed=['sum_pt', 'mass', 'girth'], edit='split'),
        dict(name='soft_outward_pt<20_x1.5_cap0.4', target='girth', fixed=['sum_pt', 'n_particles'], edit='outward_cap'),
        dict(name='soft_outward_pt<10_x2_uncapped', target='girth', fixed=['sum_pt', 'n_particles'], edit='outward'),
        dict(name='isomass_move_slot123', target='D2', fixed=['mass', 'sum_pt', 'n_particles'], edit='isomass'),
    ]
    out = dict(plan=plan, checks={})
    rng = np.random.default_rng(1)
    for P_ in plan:
        tgt = P_['target']; e = res['obs'][tgt]
        f9, f7 = e['first_lin_0.9'], e['first_lin_0.7']
        st = STG[f9] if f9 < NEVER else (STG[f7] if f7 < NEVER else STG[int(np.argmax([e['R2']['trained'][s]['lin'] for s in STG]))])
        pool = te.copy()
        if P_['edit'] == 'split': pool = pool[(x[pool, :, 0] > 0).sum(1) <= 59]
        sel = np.sort(rng.choice(pool, min(2000, len(pool)), replace=False)); xs = x[sel]
        o0 = obs_of(xs)
        if P_['edit'] == 'isomass':
            # per jet, of the 9 iso-mass moves (slot k = 1, 2, 3; angle π/2, π, 3π/2) the one with the largest |Δ target|
            C = [edit_isomass(xs, k, th) for k in (1, 2, 3) for th in (np.pi / 2, np.pi, 3 * np.pi / 2)]
            D = np.stack([np.abs(obs_of(z)[tgt] - o0[tgt]) for z in C]); best = D.argmax(0)
            xe = np.stack(C)[best, np.arange(len(xs))]
        else:
            xe = {'split': lambda: edit_split_softest(xs), 'outward': lambda: edit_soft_outward(xs, 10.0, 2.0),
                  'outward_cap': lambda: edit_soft_outward(xs, 20.0, 1.5, cap=0.4)}[P_['edit']]()
        xc = rotate(xs, np.pi / 2)                     # control: every observable here is rotation invariant
        o1, oc = obs_of(xe), obs_of(xc)
        a0, a1, ac = (net.run(z, keep=STG) for z in (xs, xe, xc))
        c = dict(stage=st, n=len(sel), stage_rule='first linear R²≥0.9' if f9 < NEVER else ('first ≥0.7' if f7 < NEVER else 'best stage'),
                 d_obs=dict(mean=float((o1[tgt] - o0[tgt]).mean()), sd=float((o1[tgt] - o0[tgt]).std()),
                            sd_over_data_sd=float((o1[tgt] - o0[tgt]).std() / O[tgt].std()), data_sd=float(O[tgt].std())),
                 fixed={f: dict(mean_abs_change=float(np.abs(o1[f] - o0[f]).mean()), data_sd=float(O[f].std())) for f in P_['fixed']},
                 side={f: dict(mean_change=float((o1[f] - o0[f]).mean()), mean_abs_change=float(np.abs(o1[f] - o0[f]).mean()), data_sd=float(O[f].std()))
                       for f in ('n_particles', 'sum_pt', 'mass', 'girth', 'e2', 'D2', 'tau21', 'sum_r', 'n_dr_0p4_up') if f not in P_['fixed'] and f != tgt},
                 control_d_obs=float(np.abs(oc[tgt] - o0[tgt]).mean()),
                 calls_changed=float((net.run(xe)['logits'].argmax(1) != net.run(xs)['logits'].argmax(1)).mean()), per_stage={})
        dob = o1[tgt] - o0[tgt]
        for s in STG:
            p = probe(tgt, s); X0, X1, Xc = a0[s][:, LV[s]], a1[s][:, LV[s]], ac[s][:, LV[s]]
            dp = p(X1) - p(X0); dc = p(Xc) - p(X0)
            slope = float((dp * dob).sum() / max((dob ** 2).sum(), 1e-300))
            c['per_stage'][s] = dict(d_probe_mean=float(dp.mean()), d_probe_sd=float(dp.std()), slope=slope,
                                     corr=float(np.corrcoef(dp, dob)[0, 1]) if dob.std() > 0 and dp.std() > 0 else None,
                                     ratio_of_means=float(dp.mean() / dob.mean()) if dob.mean() != 0 else None,
                                     control_d_probe_mean=float(dc.mean()), control_d_probe_absmean=float(np.abs(dc).mean()),
                                     probe_test_r2=e['R2']['trained'][s]['lin'])
        out['checks'][P_['name']] = c
        print(f'{P_["name"]}: target {tgt} (stage {st}, {c["stage_rule"]}), n {len(sel)}; Δobs {c["d_obs"]["mean"]:+.4g} ± {c["d_obs"]["sd"]:.3g} '
              f'(data sd {c["d_obs"]["data_sd"]:.3g}); fixed ' + ', '.join(f'{f} |Δ| {v["mean_abs_change"]:.3g} (sd {v["data_sd"]:.3g})' for f, v in c['fixed'].items())
              + f'; control |Δobs| {c["control_d_obs"]:.2g}; calls changed {c["calls_changed"]*100:.1f}%', flush=True)
        for s in STG:
            q = c['per_stage'][s]
            print(f'   {s:3s} R² {q["probe_test_r2"]:.3f}  Δprobe {q["d_probe_mean"]:+.4g} ± {q["d_probe_sd"]:.3g}  slope {q["slope"]:+.3f}  '
                  f'corr {q["corr"] if q["corr"] is None else round(q["corr"], 3)}  ratio of means {q["ratio_of_means"] if q["ratio_of_means"] is None else round(q["ratio_of_means"], 3)}  '
                  f'| control Δprobe {q["control_d_probe_mean"]:+.4g} (|.| {q["control_d_probe_absmean"]:.3g})', flush=True)
        # readability of the target at fixed held quantity: ridge on the target minus its mean in 20 quantile bins of
        # the first held quantity (bins and probe fitted on the train half, R² on the test half)
        h = O[P_['fixed'][0]]; edges = np.unique(np.percentile(h[tr], np.linspace(0, 100, 21)))
        b = np.clip(np.searchsorted(edges, h, side='right') - 1, 0, len(edges) - 2)
        tc = np.clip(O[tgt], *np.percentile(O[tgt][tr], [0.5, 99.5]))
        bm = np.array([tc[tr][b[tr] == i].mean() if (b[tr] == i).any() else tc[tr].mean() for i in range(len(edges) - 1)])
        res_t = tc - bm[b]; c['binned_mean_r2'] = r2(tc[te], bm[b[te]])
        c['r2_at_fixed'] = {}; c['slope_at_fixed_probe'] = {}
        for s in STG:
            pr = Probe(A[s][tr][:, LV[s]], res_t[tr]); c['r2_at_fixed'][s] = r2(res_t[te], pr(A[s][te][:, LV[s]]))
            dpr = pr(a1[s][:, LV[s]]) - pr(a0[s][:, LV[s]])     # the held quantity is fixed by the edit: Δ residual = Δ target
            c['slope_at_fixed_probe'][s] = float((dpr * dob).sum() / max((dob ** 2).sum(), 1e-300))
        print(f'   R² of {tgt} from 20 bins of {P_["fixed"][0]} alone {c["binned_mean_r2"]:.3f}; probe of the residual (at fixed {P_["fixed"][0]}): R² / edit slope per stage '
              + ' '.join(f'{s} {c["r2_at_fixed"][s]:.3f}/{c["slope_at_fixed_probe"][s]:+.3f}' for s in STG), flush=True)
        print('   side changes: ' + ', '.join(f'{f} Δ {v["mean_change"]:+.3g} (|Δ| {v["mean_abs_change"]:.3g}, sd {v["data_sd"]:.3g})' for f, v in c['side'].items()), flush=True)
    (OUT / 'e10_causal.json').write_text(json.dumps(out, indent=1)); print(f'done {time.time()-t0:.0f}s')


# ------------------------------------------------------------------------------------------------------------- plot
def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = json.loads((OUT / 'e10_layers.json').read_text()); E = res['obs']
    COL = ['#2a78d6', '#eb6834', '#1baf7a', '#c98500']
    NAME = ['depth 0: sums', 'depth 1: ratios / products', 'depth 2: 3-point, ratios of ratios', 'depth 3: clustering / grooming']
    xs = np.arange(len(STG) + 1); lab = ['input\n(ridge)'] + list(STG)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    panels = [('trained', 'lin', 'Trained network, linear readout'), ('trained', 'mlp', 'Trained network, MLP readout'),
              ('untrained', 'lin', 'Untrained network, linear readout')]
    for ax, (net, kind, title) in zip(axs, panels):
        for d in range(4):
            ids = [k for k, e in E.items() if e['depth'] == d]
            V = np.array([[E[k]['raw_lin']] + [E[k]['R2'][net][s][kind] for s in STG] for k in ids])
            V = np.clip(V, -0.2, 1)
            med, q1, q3 = np.median(V, 0), np.percentile(V, 25, 0), np.percentile(V, 75, 0)
            xo = xs + (d - 1.5) * 0.11
            ax.errorbar(xo, med, yerr=[med - q1, q3 - med], color=COL[d], lw=2.2, marker='o', ms=6, capsize=3, elinewidth=1.3,
                        label=f'{NAME[d]} (n={len(ids)})')
        ax.axhline(0.9, color='#888', lw=1, ls='--'); ax.axhline(0.7, color='#bbb', lw=1, ls=':')
        ax.axvline(0.5, color='#ddd', lw=1)
        ax.text(1.005, 0.7, '0.7', fontsize=9, color='#888', va='center', ha='left', transform=ax.get_yaxis_transform())
        if net == 'untrained':
            ax.annotate(f'untrained S: {res["live"]["untrained"]["S"]} live unit', (3, 0.0), (2.2, 0.62), fontsize=10, color='#444',
                        arrowprops=dict(arrowstyle='->', color='#888', lw=1))
        ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=11); ax.set_title(title, fontsize=13)
        ax.set_ylim(-0.2, 1.02); ax.grid(axis='y', color='#eee'); ax.spines[['top', 'right']].set_visible(False)
        ax.text(1.005, 0.9, '0.9', fontsize=9, color='#666', va='center', ha='left', transform=ax.get_yaxis_transform())
    axs[0].set_ylabel('held-out R² (median over observables; bar = IQR)', fontsize=11)
    axs[0].legend(loc='lower right', fontsize=9.5, frameon=False)
    fig.suptitle(f'E10 JEDI-linear: readability of {len(E)} jet observables by stage ({res["N"]} jets, 50/50 split; '
                 'input point = ridge on raw particles in every panel)', fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / 'e10_layers.png', dpi=130); print('wrote', OUT / 'e10_layers.png')


if __name__ == '__main__':
    {'read': read, 'causal': causal, 'plot': plot}[sys.argv[1] if len(sys.argv) > 1 else 'read']()
