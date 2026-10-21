"""B7 (Paper B, claims B2 / B2o program fidelity under edits, B4 / B6 baselines vs description length; exploratory method):
counterfactual-guided program extraction. Do if-statement programs (hinge / tree surrogates) fitted with network-labelled
edited jets reproduce JEDI-linear's call changes under edits of a family that was NOT in the fit, at a given description
length (DL, bits)?

Network: official JEDI-linear 3-feature n64 (interp.jedi_core.Jedi, exact fixed point). Every jet below, natural or edited,
is labelled by the exact network: its 5 logits (g, q, W, Z, t) and its call (argmax).

Jets (interp.jedi_core.sample on the test file; overlaps removed in the order listed):
  natural fit  sample(4000, seed 3901) = 20,000 jets; permutation rng 0: first 2/3 'nat_fit' (MARS / tree fitting rows),
               last 1/3 'nat_dev' (selection of the matched-bit GBDT by natural same call; MARS R^2 path log only).
  natural test sample(2000, seed 3902) minus the 3901 jets.
  counterfactual base C = sample(800, seed 3903) minus the 3901 and 3902 jets; split by jet (rng 3904) into a fit half and a
               test half. Every edited jet inherits its base jet's half, so no base jet of a test row is ever in a fit.
Edit families (all edits act on the base jet's particles (pT, deta, dphi about the stored axis); padding never moved):
  cf     cf   E36 minimal call-flipping edit, regenerated on C: per real particle pT_i -> pT_i exp(u_i), angles + 0.02 w_i;
              the smallest eps (eps^2 = sum (dpT/pT)^2 + |d angle|^2 / 0.02^2) found by 200 Adam steps on the float + masks
              program that makes it call the partner class (W <-> Z, q <-> g; jets the exact network calls g, q, W or Z and
              where float and exact calls agree), with the jet four-momentum (pT, mass, axis) held by Gauss-Newton projection;
              kept when |dm|/m < 1e-3 and axis shift < 1 mrad. The exact network flips on part of them (E36: ~41 %).
         cfr  one random edit per cf jet of the same v-space norm, tangent to the same constraints, projected (E36 control).
  D      D(dc) E35 / H45 rest-frame decay-axis rotation of the two exclusive-kT prongs at fixed jet four-momentum, every
              pairwise invariant mass and N held; dc = -0.3, -0.15, +0.15, +0.3 (dc > 0 = more asymmetric energy sharing);
              inputs rebuilt about each jet's own E-scheme axis (base set C_id, rebuilt the same way); E35 acceptance.
         cn(dc) control: the same rest-frame angle about the prong axis (prong four-momenta, sharing and opening angle held).
  trans  trans   every real particle deta += 0.06 rad (B5 / B6 'trans').
         trans_c each real particle moved 0.06 rad in its own random direction (B5 'trans_c').
         dil     all angles scaled per jet by k = sqrt(M2(trans) / M2), M2 = sum pT r^2 about the stored axis: a dilation with
                 the same M2 (mass-like) increase as the translation, no displacement of the centroid.
  rot    rot   rigid rotation of the whole jet about the stored axis by |theta| ~ U(22.5, 90) deg, random sign (the rigid-rotation
              edit of claim A13; every r, pT, M0, M2, N held). Every observable of both pools is rotation invariant, so no
              surrogate below can respond to it: this family is a built-in expressibility null.
  rnd    (control pool, fit only) random non-targeted edits of the fit-half base jets: every real particle moved 0.03 rad in
              its own random direction and pT x exp(0.1 N(0,1)), re-sorted by pT; 4 draws per jet.
Observable pools (B6w): RW = 97 pT-rank-window energy-flow sums (e11_mars_all.handbuilt) + M0 = sum pT (98 inputs); HAND = mass,
  soft-drop mass, tau21, tau32, D2, N, sum pT, sum pT r, sum pT r^2, sum pT r^3 (10 inputs). Recomputed from each edited jet.
Fit conditions (rows added to nat_fit; each family's fit-half rows subsampled to at most CAP = 4000):
  nat   (i)  natural jets only (current practice)
  all   (iii) natural + all four families
  lofo_<f> (ii) natural + every family except f (f held out)
  shuf  control: as 'all' but the logits permuted among the fit rows of each family (within family)
  rnd   control: natural + as many rnd rows as 'all' adds (non-targeted extra data)
Surrogates (fitted to the network's 5 logits; call = argmax):
  prog  forward-stepwise MARS (jetdistill.mars, B6w settings, candidate gains on 8,000 random fit rows) on RW, up to 50 terms
        per logit; prefixes K = 4, 8, 16, 30, 50 refitted by least squares. DL = B6 rule (32 bits per coefficient / knot +
        log2(#inputs) + 2 per factor).
  hprog the same on HAND, K = 4, 8, 16, 30.
  gbdt  HistGBDT regressors on RW quantised to 2^q levels (q = 4, 8) x leaves {4, 8} x iterations {10, 30}, lr 0.5; DL as B6w
        gbdtq (q bits per threshold + log2(#inputs), 32 per leaf, 64 per input used). 'gbdt@K' = the member with bits <= prog K
        bits (same condition) with the highest natural dev same call (selection uses natural jets only).
  cart  CART regressor on RW, depth 4, 6, 8 (B6 tree_bits).   hgbdt  one HistGBDT (q 8, 8 leaves, 30 iterations) on HAND.
  mlp   sklearn MLPRegressor on standardised RW, hidden (8,) and (32,), early stopping; DL = 32 bits per parameter.
Metrics per (condition, surrogate, family; family test-half rows): network flip nf = exact call(edit) != call(base);
  surrogate flip sf likewise with the surrogate's own calls; edit agreement = mean(nf == sf); strict agreement additionally
  requires the same new call; Cohen kappa of (sf, nf) (the never-flip rule has kappa 0 and agreement 1 - flip rate);
  Pearson r of the margin change (margin = logit of the network's base call minus the largest other logit; network vs
  surrogate). Natural same call = % of natural test jets where surrogate and network calls agree.
Statistics: 95 % CIs from 1000 bootstrap resamples of base jets (all edits of a resampled base jet move together; natural test:
  resampled jets). Primary test per family f and surrogate m: H0 agreement(lofo_f) = agreement(nat) on f's test rows,
  H1 differs; paired cluster-bootstrap CI and p, exact McNemar on per-row agreement (rows of one base jet are not independent:
  the bootstrap p is the primary one); Holm over the families x surrogates tested. Secondary: all vs nat (in-fit), all vs
  shuf (label specificity), lofo_f vs rnd (targeted vs non-targeted extra data), prog vs matched GBDT gain difference,
  natural same-call change.
Modes: python -m interp.b7_counterfactual_programs pool   -> $AI_RESULTS/jedi/b7_pool.npz, b7_pool.json
       python -m interp.b7_counterfactual_programs fit    -> b7.json, b7.png
       (--smoke: 60 jets per class, fewer steps / terms)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('KERAS_BACKEND', 'jax')
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
DCS = (-0.3, -0.15, 0.15, 0.3)
FAMS = dict(cf=['cf', 'cfr'], D=[f'D{d:+.2f}' for d in DCS] + [f'cn{d:+.2f}' for d in DCS], trans=['trans', 'trans_c', 'dil'], rot=['rot'])
FAMLAB = dict(cf='minimal flip edits (E36) + random same-size', D='prong sharing D(dc) + cn controls (E35)',
              trans='0.06 rad translation, random-direction control, matched dilation', rot='rigid rotation about the axis (A13)')
HAND = ['mass', 'sd_mass', 'tau21', 'tau32', 'D2', 'n_particles', 'sum_pt']
PROG_K = (4, 8, 16, 30, 50); HPROG_K = (4, 8, 16, 30)
CAP = 4000
HELD = dict(cf='cf', D='D', trans='trans', rot='rot', cfmin='cf', Dmin='D')   # family whose LOFO fit holds out these rows


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ============================================================================================ pool
def observables(xx):
    from interp.b6_pareto import features
    from jetdistill.observables import compute
    rw, keys = features(xx)
    q = compute(np.asarray(xx, np.float64), 64, ids=HAND)
    pt = np.asarray(xx[..., 0], np.float64); r = np.hypot(xx[..., 1], xx[..., 2]).astype(np.float64)
    hd = np.stack([np.asarray(q[k], np.float64) for k in HAND] + [(pt * r ** n).sum(1) for n in (1, 2, 3)], 1)
    return rw.astype(np.float64), np.nan_to_num(hd, nan=0.0, posinf=0.0, neginf=0.0), keys


def randdisp(xx, mag, seed):
    rng = np.random.default_rng(seed); th = rng.uniform(0, 2 * np.pi, mag.shape); e = xx.copy()
    e[..., 1] += (mag * np.cos(th)).astype(np.float32); e[..., 2] += (mag * np.sin(th)).astype(np.float32); return e


def mode_pool(a):
    import jax, jax.numpy as jnp
    from interp.jedi_core import Jedi, sample
    from interp.b6w_symreg import indices
    from interp import e36_counterfactual_flip_jedi as E36
    from interp.e35_prong_sharing_route_jedi import make_edits
    pf, pt_, pc = (60, 60, 60) if a.smoke else (4000, 2000, 800)
    net = Jedi(64)
    xF, yF, _ = sample(pf, 3901); xT, yT, _ = sample(pt_, 3902); xC, yC, _ = sample(pc, 3903)
    iF, iT, iC = indices(pf, 3901), indices(pt_, 3902), indices(pc, 3903)
    kT = ~np.isin(iT, iF); kC = ~np.isin(iC, np.concatenate([iF, iT]))
    xT, yT = np.ascontiguousarray(xT[kT]), yT[kT]; xC, yC = np.ascontiguousarray(xC[kC]), yC[kC]
    perm = np.random.default_rng(0).permutation(len(yF)); nf = 2 * len(yF) // 3
    X = dict(nat_fit=np.ascontiguousarray(xF[perm[:nf]]), nat_dev=np.ascontiguousarray(xF[perm[nf:]]), nat_test=xT, C_st=xC)
    Y = dict(nat_fit=yF[perm[:nf]], nat_dev=yF[perm[nf:]], nat_test=yT, C_st=yC)
    J = len(xC); hp = np.random.default_rng(3904).permutation(J); fit_half = np.zeros(J, bool); fit_half[hp[:J // 2]] = True
    BASE = {}; BSET = {}
    info = dict(N_nat_fit=int(nf), N_nat_dev=int(len(yF) - nf), N_nat_test=int(len(yT)), N_C=int(J), C_removed=int((~kC).sum()),
                test_removed=int((~kT).sum()), N_C_fit_half=int(fit_half.sum()))
    log('jets', json.dumps(info))
    # ---------------- cf: minimal flip edits (E36 search on the float + masks program)
    t = time.time()
    f = E36.float_program(net); fj = jax.jit(f); step = E36.make_step(f, 0.05)
    x0 = xC.astype(np.float64)
    lo_f = np.concatenate([np.asarray(fj(jnp.asarray(xC[i:i + 1000]))) for i in range(0, J, 1000)]); c_f = lo_f.argmax(1)
    c_ex = net.run(xC)['logits'].argmax(1)
    idx = np.flatnonzero((c_f == c_ex) & np.isin(c_f, [0, 1, 2, 3])); tgt = np.array([E36.PARTNER[int(c)] for c in c_f[idx]])
    v, found, _ = E36.search(step, x0[idx], tgt, 30 if a.smoke else 200, 0.05, chunk=a.chunk, tag='cf')
    xe, _ = E36.apply_edit(x0[idx], v); dm, dax, _ = E36.check_constraints(x0[idx], xe); ok = found & (dm < 1e-3) & (dax < 1e-3)
    X['cf'] = xe[ok].astype(np.float32); BASE['cf'] = idx[ok]; BSET['cf'] = 'C_st'; vok = v[ok]; ii = idx[ok]
    rv = []
    for i0 in range(0, len(ii), 500):
        rv.append(E36.random_edits(x0[ii[i0:i0 + 500]], vok[i0:i0 + 500], 1, np.random.default_rng([3905, i0]))[0])
    rv = np.concatenate(rv) if rv else np.zeros((0, 64, 3))
    xr, _ = E36.apply_edit(x0[ii], rv); dm2, dax2, _ = E36.check_constraints(x0[ii], xr); ok2 = (dm2 < 1e-3) & (dax2 < 1e-3)
    X['cfr'] = xr[ok2].astype(np.float32); BASE['cfr'] = ii[ok2]; BSET['cfr'] = 'C_st'
    eps = E36.eps_of(x0[idx][ok], vok)
    info['cf'] = dict(n_searched=int(len(idx)), found_float=float(found.mean()), n_ok=int(ok.sum()), n_random_ok=int(ok2.sum()),
                      eps_median=float(np.median(eps)) if len(eps) else None, seconds=round(time.time() - t))
    log('cf', json.dumps(info['cf']))
    # ---------------- D(dc), cn(dc) (E35)
    t = time.time()
    XD, acc, _, dinfo, _, nn, _ = make_edits(xC, 3906, a.smoke)
    X['C_id'] = XD['id']; Y['C_id'] = yC
    for dc in DCS:
        m = np.flatnonzero(acc[dc] & (nn >= 3))
        for kind, nm in (('D', 'D'), ('cn', 'cn')):
            s = f'{nm}{dc:+.2f}'; X[s] = XD[(dc, kind)][m]; BASE[s] = m; BSET[s] = 'C_id'
    info['D'] = dict(accepted={f'{dc:+.2f}': int((acc[dc] & (nn >= 3)).sum()) for dc in DCS}, seconds=round(time.time() - t)); log('D', json.dumps(info['D']))
    del XD
    # ---------------- trans, trans_c, dil, rot
    real = xC[..., 0] > 0; rf = real.astype(np.float32); allJ = np.arange(J)
    tr = xC.copy(); tr[..., 1] += np.float32(0.06) * rf; X['trans'] = tr
    X['trans_c'] = randdisp(xC, 0.06 * rf, 3907)
    M2 = lambda q: (q[..., 0].astype(np.float64) * (q[..., 1].astype(np.float64) ** 2 + q[..., 2].astype(np.float64) ** 2)).sum(1)
    k = np.sqrt(M2(tr) / np.maximum(M2(xC), 1e-12)); dl = xC.copy(); dl[..., 1:] = (xC[..., 1:] * k[:, None, None]).astype(np.float32); X['dil'] = dl
    rng = np.random.default_rng(3908); th = rng.uniform(np.pi / 8, np.pi / 2, J) * rng.choice([-1.0, 1.0], J)
    c, s_ = np.cos(th)[:, None].astype(np.float32), np.sin(th)[:, None].astype(np.float32); ro = xC.copy()
    ro[..., 1] = xC[..., 1] * c - xC[..., 2] * s_; ro[..., 2] = xC[..., 1] * s_ + xC[..., 2] * c; X['rot'] = ro
    for s in ('trans', 'trans_c', 'dil', 'rot'): BASE[s] = allJ; BSET[s] = 'C_st'
    info['dil_k'] = dict(mean=float(k.mean()), median=float(np.median(k)))
    # ---------------- rnd control pool (fit half only, 4 draws)
    fj_ = np.flatnonzero(fit_half); R = []; RB = []
    for d in range(4):
        rg = np.random.default_rng([3909, d]); e = randdisp(xC[fj_], 0.03 * rf[fj_], 3910 + d)
        e[..., 0] = np.where(real[fj_], e[..., 0] * np.exp(0.1 * rg.normal(size=e[..., 0].shape)), 0).astype(np.float32)
        o = np.argsort(-e[..., 0], axis=1, kind='stable'); R.append(np.take_along_axis(e, o[..., None], 1)); RB.append(fj_)
    X['rnd'] = np.concatenate(R); BASE['rnd'] = np.concatenate(RB); BSET['rnd'] = 'C_st'
    # ---------------- network + observables
    out = {}; keys = None; acc_ = {}
    for s, xx in X.items():
        t = time.time(); xx = np.ascontiguousarray(xx, np.float32)
        L = net.run(xx)['logits']; rw, hd, keys = observables(xx)
        out[f'L|{s}'] = L.astype(np.float32); out[f'rw|{s}'] = rw; out[f'hd|{s}'] = hd
        if s in BASE: out[f'base|{s}'] = BASE[s]
        if s in Y: out[f'y|{s}'] = Y[s]; acc_[s] = float((L.argmax(1) == Y[s]).mean())
        log(f'set {s}: {len(xx)} jets; network + observables {time.time() - t:.0f} s' + (f'; accuracy {acc_[s] * 100:.2f} %' if s in Y else ''))
    nb = {s: out[f'L|{BSET[s]}'].argmax(1)[BASE[s]] for s in BASE}
    info['flip_rate'] = {s: float((out[f'L|{s}'].argmax(1) != nb[s]).mean()) for s in BASE}
    info['accuracy'] = acc_
    info['accuracy_3901'] = float(((np.concatenate([out['L|nat_fit'], out['L|nat_dev']]).argmax(1)) == np.concatenate([Y['nat_fit'], Y['nat_dev']])).mean())
    info['base_set'] = BSET; info['N'] = {s: int(len(x)) for s, x in X.items()}
    log('network accuracy: sample 3901', f'{info["accuracy_3901"] * 100:.2f} %', '; sample 3902', f'{acc_["nat_test"] * 100:.2f} %')
    log('flip rates', json.dumps({s: round(v * 100, 2) for s, v in info['flip_rate'].items()}))
    np.savez_compressed(OUT / f'b7{"smoke" if a.smoke else ""}_pool.npz', **out, fit_half=fit_half, rw_keys=np.array(keys))
    (OUT / f'b7{"smoke" if a.smoke else ""}_pool.json').write_text(json.dumps(info, indent=1, default=float))
    log('wrote pool')


# ============================================================================================ fit
def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (n - r) * p[i])); q[i] = run
    return q


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(a_only=n10, b_only=n01, p=1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue))


def mars_fit(Xf, Lf, Xd, Ld, keys, kmax, Ks, tag, nsel):
    from jetdistill import mars
    from interp.e11_mars_all import knots_of
    from interp.b6_pareto import term_bits
    Q = {'fit': {k: Xf[:, i] for i, k in enumerate(keys)}, 'dev': {k: Xd[:, i] for i, k in enumerate(keys)}}
    knots = knots_of(Q['fit'], keys); NF = len(Lf); sel = np.random.default_rng(1).choice(NF, min(nsel, NF), replace=False)
    mars.MAX_TERMS, mars.STALL = kmax, 10 ** 6; fits = []
    for k in range(5):
        t = time.time(); c = 1.0 - min(Lf[:, k].min(), Ld[:, k].min()) + 0.5 * Lf[:, k].std()
        terms, _, path = mars.fit_neuron(dict(fit=Lf[:, k] + c, dev=Ld[:, k] + c), Q, keys, knots, sel)
        fits.append((terms, c)); log(f'  {tag} MARS logit {CL[k]}: {len(terms)} terms (natural dev R^2 path end {path[-1] if path else None}); {time.time() - t:.0f} s')
    progs = {}
    for K in Ks:
        parts = [(terms[:K], mars._lstsq(terms[:K], Q['fit'], Lf[:, k] + c, NF), c) for k, (terms, c) in enumerate(fits)]
        bits = sum(32 + sum(sum(term_bits(t, len(keys))) for t in tr) for tr, _, _ in parts)
        progs[K] = (parts, bits, sum(len(p[0]) for p in parts))

    def predict(parts, Xa):
        Qd = {k: Xa[:, i] for i, k in enumerate(keys)}; J = len(Xa)
        return np.stack([np.stack([np.ones(J)] + [mars.term_value(t, Qd) for t in tr], 1) @ co - c for tr, co, c in parts], 1)
    return progs, predict


class Data:
    def __init__(self, path):
        Z = np.load(path); self.Z = Z
        self.sets = sorted({k.split('|', 1)[1] for k in Z.files if k.startswith('L|')})
        self.L = {s: Z[f'L|{s}'].astype(np.float64) for s in self.sets}; self.rw = {s: Z[f'rw|{s}'] for s in self.sets}
        self.hd = {s: Z[f'hd|{s}'] for s in self.sets}; self.base = {s: Z[f'base|{s}'] for s in self.sets if f'base|{s}' in Z.files}
        self.fit_half = Z['fit_half']; self.keys = [str(k) for k in Z['rw_keys']]
        self.bset = json.loads(Path(str(path).replace('.npz', '.json')).read_text())['base_set']
        self.call = {s: self.L[s].argmax(1) for s in self.sets}


def margin(L, c0):
    J = len(L); own = L[np.arange(J), c0]; o = L.copy(); o[np.arange(J), c0] = -np.inf; return own - o.max(1)


def build_condition(D, cond, rng_seed=11):
    """-> fit rows: rw, hd, logits (natural fit + family fit-half rows, capped)"""
    rng = np.random.default_rng(rng_seed)
    parts = [('nat_fit', np.arange(len(D.L['nat_fit'])), None)]
    fams = [] if cond in ('nat', 'rnd') else [f for f in FAMS if not (cond.startswith('lofo_') and cond[5:] == f)]
    added = 0
    for f in fams:
        rows = [(s, np.flatnonzero(D.fit_half[D.base[s]])) for s in FAMS[f] if s in D.base]
        n = sum(len(r) for _, r in rows); keep = n if n <= CAP else CAP
        allr = np.concatenate([np.stack([np.full(len(r), i), r], 1) for i, (_, r) in enumerate(rows)]); pick = allr[rng.choice(n, keep, replace=False)]
        perm = rng.permutation(keep) if cond == 'shuf' else None
        for i, (s, _) in enumerate(rows):
            sel = pick[pick[:, 0] == i, 1]; parts.append((s, sel, f))
        if perm is not None: parts[-len(rows):] = [(s, sel, (f, 'shuf')) for s, sel, _ in parts[-len(rows):]]
        added += keep
    if cond == 'rnd':
        nall = sum(min(CAP, sum(int(D.fit_half[D.base[s]].sum()) for s in FAMS[f] if s in D.base)) for f in FAMS)
        r = rng.choice(len(D.L['rnd']), min(nall, len(D.L['rnd'])), replace=False); parts.append(('rnd', r, 'rnd')); added = len(r)
    RW = np.concatenate([D.rw[s][r] for s, r, _ in parts]); HD = np.concatenate([D.hd[s][r] for s, r, _ in parts])
    L = [D.L[s][r] for s, r, _ in parts]
    if cond == 'shuf':      # permute logits among the fit rows of each family (within family)
        i0 = len(parts[0][1]); L = np.concatenate(L); fam_of = np.concatenate([np.full(len(r), str(g)) for _, r, g in parts])
        for f in FAMS:
            m = np.flatnonzero(fam_of == str((f, 'shuf')))
            if len(m): L[m] = L[m][rng.permutation(len(m))]
    else:
        L = np.concatenate(L)
    return RW, HD, L, dict(N_fit=int(len(L)), N_natural=int(len(parts[0][1])), N_added=int(added),
                           by_set={s: int(len(r)) for s, r, _ in parts})


def family_rows(D, f):
    """test-half rows of family f: list of (set, row idx), base C index per row"""
    out = []
    for s in FAMS[f]:
        if s not in D.base: continue
        r = np.flatnonzero(~D.fit_half[D.base[s]]); out.append((s, r))
    return out


def fit_surrogates(D, cond, a):
    """-> dict name -> (predict-fn on a set name -> logits, bits, extra)"""
    import warnings
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neural_network import MLPRegressor
    from interp.b6_pareto import tree_bits
    from interp.b6w_symreg import quantiser, hgb_bits_q
    warnings.filterwarnings('ignore')
    RW, HD, L, cinfo = build_condition(D, cond); log(f'condition {cond}: {json.dumps(cinfo)}')
    S = {}; F = RW.shape[1]
    t = time.time()
    progs, pred = mars_fit(RW, L, D.rw['nat_dev'], D.L['nat_dev'], D.keys, max(a.prog_k), a.prog_k, f'{cond} RW', a.nsel)
    for K, (parts, bits, nt) in progs.items():
        S[f'prog{K}'] = (lambda s, parts=parts: pred(parts, D.rw[s]), bits, dict(family='prog', K=K, terms=nt))
    hk = [f'hand{i}' for i in range(HD.shape[1])]
    hprogs, hpred = mars_fit(HD, L, D.hd['nat_dev'], D.L['nat_dev'], hk, max(a.hprog_k), a.hprog_k, f'{cond} HAND', a.nsel)
    for K, (parts, bits, nt) in hprogs.items():
        S[f'hprog{K}'] = (lambda s, parts=parts: hpred(parts, D.hd[s]), bits, dict(family='hprog', K=K, terms=nt))
    log(f'  MARS {time.time() - t:.0f} s'); t = time.time()
    for q in (4, 8):
        qf = quantiser(RW, q)
        for lv in (4, 8):
            for it in (3, 5, 10, 30):
                rs = [HistGradientBoostingRegressor(max_leaf_nodes=lv, max_iter=it, learning_rate=0.5, early_stopping=False, random_state=0).fit(qf(RW), L[:, k]) for k in range(5)]
                S[f'gbdt{q}_L{lv}_T{it}'] = (lambda s, rs=rs, qf=qf: np.stack([r.predict(qf(D.rw[s])) for r in rs], 1), sum(hgb_bits_q(r, F, q) for r in rs),
                                            dict(family='gbdt', q=q, leaves=lv, iters=it))
    qh = quantiser(HD, 8)
    rs = [HistGradientBoostingRegressor(max_leaf_nodes=8, max_iter=30, learning_rate=0.5, early_stopping=False, random_state=0).fit(qh(HD), L[:, k]) for k in range(5)]
    S['hgbdt8_L8_T30'] = (lambda s, rs=rs: np.stack([r.predict(qh(D.hd[s])) for r in rs], 1), sum(hgb_bits_q(r, HD.shape[1], 8) for r in rs), dict(family='hgbdt'))
    for d in (4, 6, 8):
        r = DecisionTreeRegressor(max_depth=d, random_state=0).fit(RW, L)
        S[f'cart_d{d}'] = (lambda s, r=r: r.predict(D.rw[s]), tree_bits(r.tree_, F, 5), dict(family='cart', depth=d))
    log(f'  trees {time.time() - t:.0f} s'); t = time.time()
    mu, sd = RW.mean(0), np.where(RW.std(0) > 0, RW.std(0), 1.0)
    for h in ((8,), (32,)):
        m = MLPRegressor(hidden_layer_sizes=h, max_iter=300, early_stopping=True, random_state=0).fit((RW - mu) / sd, L)
        npar = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
        S[f'mlp{h[0]}'] = (lambda s, m=m: m.predict((D.rw[s] - mu) / sd), 32.0 * npar + 64 * F, dict(family='mlp', hidden=list(h)))
    log(f'  MLP {time.time() - t:.0f} s')
    return S, cinfo


def boot_counts(groups, B, seed):
    """cluster bootstrap: per resample, multiplicity of each unique group -> (B, n_groups), group index per row"""
    u, inv = np.unique(groups, return_inverse=True); rng = np.random.default_rng(seed)
    C = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(B)]).astype(np.float32)
    return C, inv, len(u)


def gsum(v, inv, ng): return np.bincount(inv, weights=np.asarray(v, np.float64), minlength=ng)


def ci(b): return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def kappa_of(nf, sf):
    po = (nf == sf).mean(); pn, ps = nf.mean(), sf.mean(); pe = pn * ps + (1 - pn) * (1 - ps)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def kappa_boot(C, inv, ng, nf, sf):
    n = C @ gsum(np.ones(len(nf)), inv, ng); a = C @ gsum(nf == sf, inv, ng) / n; pn = C @ gsum(nf, inv, ng) / n; ps = C @ gsum(sf, inv, ng) / n
    pe = pn * ps + (1 - pn) * (1 - ps); return np.where(pe < 1, (a - pe) / np.maximum(1 - pe, 1e-12), 0.0)


def corr_boot(C, inv, ng, x, y):
    n = C @ gsum(np.ones(len(x)), inv, ng); sx, sy = C @ gsum(x, inv, ng), C @ gsum(y, inv, ng)
    sxx, syy, sxy = C @ gsum(x * x, inv, ng), C @ gsum(y * y, inv, ng), C @ gsum(x * y, inv, ng)
    cov = sxy / n - sx * sy / n ** 2; vx = sxx / n - (sx / n) ** 2; vy = syy / n - (sy / n) ** 2
    return cov / np.sqrt(np.maximum(vx * vy, 1e-30))


def paired_boot(C, inv, ng, d):
    n = C @ gsum(np.ones(len(d)), inv, ng); b = C @ gsum(d, inv, ng) / n
    p = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / len(b)))
    return dict(mean=float(np.mean(d)), ci=ci(b), p=p)


def mode_fit(a):
    D = Data(OUT / f'b7{"smoke" if a.smoke else ""}_pool.npz'); pinfo = json.loads((OUT / f'b7{"smoke" if a.smoke else ""}_pool.json').read_text())
    conds = ['nat', 'all'] + [f'lofo_{f}' for f in FAMS] + ['shuf', 'rnd']
    FR = {f: family_rows(D, f) for f in FAMS}
    FR['cfmin'] = [(s_, r_) for s_, r_ in FR['cf'] if s_ == 'cf']; FR['Dmin'] = [(s_, r_) for s_, r_ in FR['D'] if s_.startswith('D')]
    # network per family test rows
    NET = {}
    for f, rows in FR.items():
        nb = np.concatenate([D.call[D.bset[s]][D.base[s][r]] for s, r in rows]); ne = np.concatenate([D.call[s][r] for s, r in rows])
        Lb = np.concatenate([D.L[D.bset[s]][D.base[s][r]] for s, r in rows]); Le = np.concatenate([D.L[s][r] for s, r in rows])
        grp = np.concatenate([D.base[s][r] for s, r in rows]); sub = np.concatenate([np.full(len(r), s) for s, r in rows])
        C, inv, ng = boot_counts(grp, a.B, 17)
        nf = ne != nb; dm = margin(Le, nb) - margin(Lb, nb)
        NET[f] = dict(nb=nb, ne=ne, nf=nf, dm=dm, C=C, inv=inv, ng=ng, sub=sub, N=int(len(nf)), N_base=int(ng))
        bf = C @ gsum(nf, inv, ng) / (C @ gsum(np.ones(len(nf)), inv, ng))
        log(f'family {f}: {len(nf)} test rows on {ng} base jets; network flip rate {nf.mean() * 100:.2f} % {[round(x * 100, 2) for x in ci(bf)]}; '
            + ', '.join(f'{s} {nf[sub == s].mean() * 100:.1f} %' for s, _ in FR[f]))
    nt = D.call['nat_test']; Nt = len(nt); Ct = np.stack([np.bincount(np.random.default_rng([18, b]).integers(0, Nt, Nt), minlength=Nt) for b in range(a.B)]).astype(np.float32)
    RES = dict(procedure=__doc__, pool=pinfo, conditions={}, network={}, rows=[])
    for f in FR:
        n = NET[f]; RES['network'][f] = dict(N_rows=n['N'], N_base=n['N_base'], flip=float(n['nf'].mean()),
                                            flip_by_set={s: float(n['nf'][n['sub'] == s].mean()) for s, _ in FR[f]},
                                            never_flip_agreement=float(1 - n['nf'].mean()))
    PER = {}     # (cond, model) -> {fam: agree bool, strict bool}, natural same-call bool
    for cond in conds:
        t = time.time(); S, cinfo = fit_surrogates(D, cond, a); RES['conditions'][cond] = cinfo
        cache_sets = {}
        def P(name, s):
            k = (name, s)
            if k not in cache_sets: cache_sets[k] = np.asarray(S[name][0](s), np.float64)
            return cache_sets[k]
        sc_dev = {}
        for name, (fn, bits, ex) in S.items():
            row = dict(cond=cond, model=name, bits=float(bits), **ex)
            pt_ = P(name, 'nat_test').argmax(1); sc = pt_ == nt; row['same_call'] = [float(sc.mean())] + ci(Ct @ sc.astype(np.float32) / Nt)
            sc_dev[name] = float((P(name, 'nat_dev').argmax(1) == D.call['nat_dev']).mean()); row['same_call_dev'] = sc_dev[name]
            PER[(cond, name)] = dict(nat=sc)
            row['fam'] = {}
            for f, rows in FR.items():
                n = NET[f]
                Sb = np.concatenate([P(name, D.bset[s])[D.base[s][r]] for s, r in rows]); Se = np.concatenate([P(name, s)[r] for s, r in rows])
                sb, se = Sb.argmax(1), Se.argmax(1); sf = sb != se; ag = sf == n['nf']; st = ag & (~n['nf'] | (se == n['ne']))
                sdm = margin(Se, n['nb']) - margin(Sb, n['nb'])
                ns = n['C'] @ gsum(np.ones(len(ag)), n['inv'], n['ng'])
                ba = n['C'] @ gsum(ag, n['inv'], n['ng']) / ns; bk = kappa_boot(n['C'], n['inv'], n['ng'], n['nf'], sf)
                ok = np.isfinite(sdm) & np.isfinite(n['dm'])
                r_ = float(np.corrcoef(n['dm'][ok], sdm[ok])[0, 1]) if sdm[ok].std() > 0 else 0.0
                br = corr_boot(n['C'], n['inv'], n['ng'], np.where(ok, n['dm'], 0), np.where(ok, sdm, 0)) if sdm[ok].std() > 0 else np.zeros(a.B)
                row['fam'][f] = dict(agree=[float(ag.mean())] + ci(ba), strict=float(st.mean()), kappa=[kappa_of(n['nf'], sf)] + ci(bk),
                                     corr_dmargin=[r_] + ci(br), flip=float(sf.mean()),
                                     by_set={s: dict(agree=float(ag[n['sub'] == s].mean()), kappa=kappa_of(n['nf'][n['sub'] == s], sf[n['sub'] == s]),
                                                     flip=float(sf[n['sub'] == s].mean())) for s, _ in rows})
                PER[(cond, name)][f] = ag
            RES['rows'].append(row)
        # matched-bit GBDT per prog K (natural dev same call, bits <= prog bits)
        gb = [n for n in S if n.startswith('gbdt')]
        RES['conditions'][cond]['matched'] = {}
        for K in a.prog_k:
            pb = S[f'prog{K}'][1]; ok = [n for n in gb if S[n][1] <= pb]
            if ok:
                m = max(ok, key=lambda n: sc_dev[n]); RES['conditions'][cond]['matched'][f'gbdt@{K}'] = m
        log(f'condition {cond} done in {time.time() - t:.0f} s: ' + '; '.join(
            f'{r["model"]} bits {r["bits"]:.0f} sc {r["same_call"][0] * 100:.1f} ' + ' '.join(f'{f}:k{r["fam"][f]["kappa"][0]:.2f}' for f in FAMS)
            for r in RES['rows'] if r['cond'] == cond and r['model'] in ('prog8', 'prog30', 'prog50', 'mlp32', 'gbdt8_L8_T30')))
        cache_sets.clear()
    # ---------------- tests
    R = {(r['cond'], r['model']): r for r in RES['rows']}
    def matched(cond, K): return RES['conditions'][cond]['matched'].get(f'gbdt@{K}')
    models_primary = [f'prog{K}' for K in a.prog_k] + ['gbdt@16', 'gbdt@30', 'hprog16', 'cart_d6', 'mlp32']
    def resolve(cond, m): return matched(cond, int(m[5:])) if m.startswith('gbdt@') else m
    def comp(c1, c2, m, f):
        m1, m2 = resolve(c1, m), resolve(c2, m)
        if m1 is None or m2 is None or (c1, m1) not in PER or (c2, m2) not in PER: return None
        n = NET[f]; a1, a2 = PER[(c1, m1)][f], PER[(c2, m2)][f]
        d = paired_boot(n['C'], n['inv'], n['ng'], a1.astype(float) - a2.astype(float))
        k1, k2 = R[(c1, m1)]['fam'][f]['kappa'][0], R[(c2, m2)]['fam'][f]['kappa'][0]
        return dict(family=f, model=m, members=[m1, m2], cond=[c1, c2], d_agree=d, mcnemar=mcnemar(a1, a2), kappa=[k1, k2], d_kappa=k1 - k2)
    tests = dict(primary_lofo_vs_nat=[], all_vs_nat=[], all_vs_shuf=[], lofo_vs_rnd=[], prog_vs_gbdt_gain=[], natural_same_call=[])
    for f in FR:
        h = HELD[f]
        for m in models_primary:
            c = comp(f'lofo_{h}', 'nat', m, f)
            if c: tests['primary_lofo_vs_nat'].append(c)
            for key, (c1, c2) in (('all_vs_nat', ('all', 'nat')), ('all_vs_shuf', ('all', 'shuf')), ('lofo_vs_rnd', (f'lofo_{h}', 'rnd'))):
                c = comp(c1, c2, m, f)
                if c: tests[key].append(c)
        for K in (8, 16, 30, 50):
            g1, g0 = matched(f'lofo_{h}', K), matched('nat', K)
            if g1 is None or g0 is None or ('nat', f'prog{K}') not in PER: continue
            n = NET[f]
            d = (PER[(f'lofo_{h}', f'prog{K}')][f].astype(float) - PER[('nat', f'prog{K}')][f]) - (PER[(f'lofo_{h}', g1)][f].astype(float) - PER[('nat', g0)][f])
            tests['prog_vs_gbdt_gain'].append(dict(family=f, K=K, gbdt=[g0, g1], d=paired_boot(n['C'], n['inv'], n['ng'], d)))
    for key in ('primary_lofo_vs_nat', 'all_vs_nat', 'all_vs_shuf', 'lofo_vs_rnd'):
        if not tests[key]: continue
        h = holm([t['d_agree']['p'] for t in tests[key]]); hm = holm([t['mcnemar']['p'] for t in tests[key]])
        for t, q, qm in zip(tests[key], h, hm): t['holm_boot'] = float(q); t['holm_mcnemar'] = float(qm)
    for cond in conds[1:]:
        for m in ('prog8', 'prog16', 'prog30', 'prog50', 'mlp32'):
            if (cond, m) not in PER: continue
            d = PER[(cond, m)]['nat'].astype(np.float32) - PER[('nat', m)]['nat']
            b = Ct @ d / Nt; tests['natural_same_call'].append(dict(cond=cond, model=m, d=float(d.mean()), ci=ci(b),
                                                                   p=float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / len(b)))))
    RES['tests'] = tests; RES['models_primary'] = models_primary
    for t in tests['primary_lofo_vs_nat']:
        log(f"LOFO vs nat  {t['family']:5s} {t['model']:8s}: agree {t['d_agree']['mean'] * 100:+.2f} {[round(x * 100, 2) for x in t['d_agree']['ci']]} "
            f"Holm {t['holm_boot']:.3g}; kappa {t['kappa'][1]:.3f} -> {t['kappa'][0]:.3f}")
    name = f'b7{"smoke" if a.smoke else ""}'
    (OUT / f'{name}.json').write_text(json.dumps(RES, indent=1, default=float))
    np.savez_compressed(OUT / f'{name}_perjet.npz', **{f'{c}|{m}|{f}': v for (c, m), dd in PER.items() for f, v in dd.items()},
                        **{f'net|{f}|{k}': NET[f][k] for f in NET for k in ('nf', 'dm', 'sub')})
    log(f'wrote {OUT}/{name}.json')
    try:
        plot(RES, OUT / f'{name}.png')
    except Exception as ex:  # noqa: BLE001
        log('plot failed:', ex)


def plot(RES, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = {(r['cond'], r['model']): r for r in RES['rows']}
    panels = [('cfmin', 'cf', 'minimal flip edits (E36 search)'), ('Dmin', 'D', 'prong-sharing edit D(dc), |dc| 0.15-0.3'),
              ('trans', 'trans', '0.06 rad shift, random shift, dilation')]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (f, h, title) in zip(axs, panels):
        for cond, col, lab, ls in (('nat', '#555555', 'fitted on natural jets only', '-'),
                                   (f'lofo_{h}', '#1f6fb4', '+ edited jets of the other families (this family held out)', '-'),
                                   ('rnd', '#7a3fa0', '+ random non-targeted edits (control)', '--'),
                                   ('all', '#d0661b', '+ edited jets of all families (this family in the fit)', ':')):
            rows = sorted([r for (c, m), r in R.items() if c == cond and r.get('family') == 'prog'], key=lambda r: r['bits'])
            if not rows: continue
            x = [r['bits'] / 1000 for r in rows]; y = [r['fam'][f]['kappa'][0] for r in rows]
            lo = [r['fam'][f]['kappa'][0] - r['fam'][f]['kappa'][1] for r in rows]; hi = [r['fam'][f]['kappa'][2] - r['fam'][f]['kappa'][0] for r in rows]
            ax.errorbar(x, y, yerr=[lo, hi], color=col, ls=ls, marker='o', ms=4, lw=1.6, capsize=2, label=lab)
        ax.axhline(0, color='k', lw=0.8, ls=':'); ax.axhline(0.4, color='#bbbbbb', lw=0.8)
        ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels(['2', '4', '8', '16']); ax.minorticks_off(); ax.set_xlabel('program size (kbit; K = 4, 8, 16, 30, 50 terms per logit)')
        ax.set_title(f"{title}\nnetwork changes its call on {RES['network'][f]['flip'] * 100:.0f} % of jets", fontsize=10)
    axs[0].set_ylabel("Cohen's kappa: program call changes vs network\n(0 = never-flip rule, 1 = all match)")
    axs[0].legend(fontsize=7.5, loc='upper left'); axs[0].set_ylim(-0.05, 0.8)
    fig.suptitle('Programs fitted with edited jets of other families (blue) match the network on a held-out family no better than natural-only fits (grey),'
                 '\nexcept a rise on the prong-sharing edit at 30-50 terms; on minimal flip edits kappa stays below 0.2 for every fit', fontsize=10.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['pool', 'fit', 'plot'])
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--B', type=int, default=1000); ap.add_argument('--nsel', type=int, default=8000)
    a = ap.parse_args()
    a.prog_k = (2, 4) if a.smoke else PROG_K; a.hprog_k = (2, 4) if a.smoke else HPROG_K
    if a.smoke: a.B = 100
    log('args', vars(a))
    if a.mode == 'pool': mode_pool(a)
    elif a.mode == 'fit': mode_fit(a)
    else: plot(json.loads((OUT / f'b7{"smoke" if a.smoke else ""}.json').read_text()), OUT / f'b7{"smoke" if a.smoke else ""}.png')
    log('done')
