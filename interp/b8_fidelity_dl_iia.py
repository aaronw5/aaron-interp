"""B8 (Paper B, claims B4, B2m, B2o; stage 2): end-to-end program fidelity vs description length (DL), scored on the
network's causal behaviour (five real input edits and stage-level interchange interventions), with an energy-flow-polynomial
(EFP) baseline, a standard-observable baseline and gradient-boosted trees.

Network: official JEDI-linear 3-feature n64 (interp.jedi_core.Jedi, exact fixed point). Every program is fitted to the
network's outputs (logits / unit values), never to truth labels; a program's call = argmax of its 5 logits (g, q, W, Z, t).

Jets (interp.jedi_core.sample, test file):
  fit sample  sample(per_class 4000, seed 808) = 20,000 jets; permutation rng 0: 2/3 'fit' (fitting rows), 1/3 'dev'
              (selection only: GBDT member, baseline variable choice, DAS step / stage choice).
  test sample sample(per_class 2000, seed 809) minus every jet of the fit sample and of sample(6000, seed 0) (the fit jets of
              the stored E11 / B4 nested programs).
Programs (DL = bits: every stored number costs its bit width; structure bits log2(#inputs) + 2 per hinge factor, as B6):
  P    the E11 / B4 unit-level nested program (rank-window (pT, r) hinge formulas for P, g, S units on the physics set, then
       S -> h4 -> h5 -> z formulas, z through the network's own last layer d7; fitted on sample(6000, 0) to the network's unit
       values). Levels: K = 30, 16, 8 terms per unit at 32-bit numbers; K = 8, 4, 2 with every number rounded to b = 10
       significant bits (cost b + 8 bits each: sign + exponent); K = 2, b = 8 with every live unit of E1 mean-ablation cost
       < 1 % of calls replaced by its mean on 3,000 fit jets (dropped objects).
  STD  MARS (jetdistill.mars, B6/B7 settings) on 10 standard observables: m, m_SD, tau21, tau32, D2, N, sum pT,
       sum pT r^n (n = 1, 2, 3); prefixes K = 2, 4, 8, 16, 30 terms per logit, least-squares refit.
  EFP  MARS on the 54 connected EFPs of degree d <= 5 (energyflow 1.4 EFPSet, hadronic measure, beta 1, normed; computed
       in the volume's venv-ef because the job env has no energyflow) plus jet m, sum pT, N (57 inputs); K = 2 .. 50.
  GBDT HistGradientBoosting regressors (one per logit) on the union of STD and EFP inputs quantised to 2^q levels,
       q in {4, 8} x leaves {4, 8, 16} x iterations {3, 10, 30, 100}, lr 0.5; DL as B6w hgb_bits_q.
  Controls: STD K30, EFP K30 and GBDT (q 8, 8 leaves, 30 iterations) fitted on the fit rows' logits permuted (rng 8099).
Edits on test jets (padding never moved; every program recomputes its inputs from the edited particles):
  trans    every real particle's deta += 0.06 rad (the W->Z flow edit; invariant mass and every EFP unchanged).
  soft     one extra particle with pT = 0.5 x the softest real pT, at r = 0.4 sqrt(U), uniform azimuth, in the first empty
           slot (jets with N < 64; it is then the softest particle, so the pT order is kept).
  swap     pT values of slot 13 and slot 14 exchanged (jets with N >= 15): the particle set is that of the A13 angle swap,
           but the softer of the two now sits in slot 13 (the rank-window edge of S0's 14 hardest particles).
  D        prong-sharing edit dc = +0.3 (E35 / H45: rest-frame rotation of the two exclusive-kT prongs' decay axis at fixed
           jet four-momentum, pairwise masses and N held), on a random subset C of test jets, inputs rebuilt about the E-scheme
           axis; base = the same jets rebuilt (C_id).  Control cn: same angle about the prong axis (sharing held).
  cf       E36 minimal call-flipping edit (per-particle pT / angle, four-momentum held) searched on the float + masks program
           towards the partner class (W<->Z, q<->g), on subset C; control cfr: a random edit of the same size, same constraints.
Scores on test jets: (1) natural same call %; (2) per edit, Cohen kappa between the program's and the network's call-change
  indicators (0 = the never-change rule, 1 = identical), plus the ratio of mean margin changes (program / network; margin =
  logit of the network's base call minus the largest other logit); (3) interchange-intervention accuracy (IIA): base jet b,
  source jet s (random test pairs); set program variable V to its value on s and run the program; do the same on the network
  at the aligned location; IIA = fraction of pairs where both calls agree; effective IIA = the same over pairs where the
  network's call changes. P variables: S units S29, S1, S0, S19, S30, all 19 live S units, the mass variable (h4 units 3, 9,
  18: the M2-follower and m-like units of E6), z windows z0, z7, z10, z14 and the four together; aligned location = the same
  units. STD / EFP / GBDT variables: the 2 inputs whose swap changes the program's call most often (dev pairs); aligned
  location = a learned 1-dim DAS direction (distributed alignment search: S' = S_b + ((S_s - S_b) . w) w) at S or at h4
  (chosen on dev pairs), trained on 4,000 fit-jet pairs to make the network's patched call equal the program's
  counterfactual call. Controls: permuted alignment (V aligned to random live units of the same stage and count, or a
  random live direction; 5 draws), shuffled-label programs, GBDT at matched DL.
Statistics: 95 % bootstrap over test jets (pairs for IIA), B = 1000; matched DL: for each P level and baseline family the
  member with bits <= P's bits and the highest dev same call (smallest member if none fits, flagged); paired bootstrap of
  P - baseline (same resamples), exact McNemar on paired per-jet indicators; Benjamini-Hochberg over P levels x families x
  scores.
Modes: python -m interp.b8_fidelity_dl_iia pool   (jax env)    -> $AI_RESULTS/jedi/b8_pool.npz/.json
       /j-jepa-vol/aaron-interp/venv-ef/bin/python -m interp.b8_fidelity_dl_iia efp -> b8_efp.npz
       python -m interp.b8_fidelity_dl_iia main   (jax env)    -> b8.json, b8_perjet.npz, b8.png
       (--smoke: tiny sample, files b8smoke_*)
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
HAND = ['mass', 'sd_mass', 'tau21', 'tau32', 'D2', 'n_particles', 'sum_pt']
STD_NAMES = ['m', 'm_SD', 'tau21', 'tau32', 'D2', 'N', 'sum pT', 'sum pT r', 'sum pT r^2', 'sum pT r^3']
EDITS = ['trans', 'soft', 'swap', 'D', 'cf']
CTRL_EDITS = ['cn', 'cfr']
EDIT_LABEL = dict(trans='0.06 rad translation', soft='add one soft particle', swap='pT swap of ranks 13/14',
                  D='prong sharing dc +0.3', cf='minimal call flips (E36)', cn='control: prong-axis rotation (sharing held)',
                  cfr='control: random edit, same size as cf')
P_LEVELS = [('P30', 30, 32, 0.0), ('P16', 16, 32, 0.0), ('P8', 8, 32, 0.0), ('P8r10', 8, 10, 0.0), ('P4r10', 4, 10, 0.0),
            ('P2r10', 2, 10, 0.0), ('P2r8d1', 2, 8, 1.0)]
P_VARS = {'S29 (sum pT r^2 + sum r^2 + soft count)': ('S', [29]), 'S1 (sum_{rank>=1} (r-0.2)+)': ('S', [1]),
          'S0 (sum pT of 14 hardest)': ('S', [0]), 'S19 (soft multiplicity)': ('S', [19]), 'S30 (sum r, core)': ('S', [30]),
          'all 19 live S units': ('S', None), 'mass variable (h4 3, 9, 18)': ('h4', [3, 9, 18]),
          'z0 window (75 GeV)': ('z', [0]), 'z7 window (94-98 GeV)': ('z', [7]), 'z10 window (135-148 GeV)': ('z', [10]),
          'z14 window (94-98 GeV)': ('z', [14]), 'z windows 0, 7, 10, 14': ('z', [0, 7, 10, 14])}
WID = dict(S=64, h4=64, h5=32, z=16)


def fmt(x, f='.3f'): return 'NA' if x is None or not np.isfinite(x) else format(x, f)


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def tag(a): return 'b8smoke' if a.smoke else 'b8'


def indices(per_class, seed):
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def std_obs(xx):
    from jetdistill.observables import compute
    q = compute(np.asarray(xx, np.float64), 64, ids=HAND)
    pt = np.asarray(xx[..., 0], np.float64); r = np.hypot(xx[..., 1], xx[..., 2]).astype(np.float64)
    hd = np.stack([np.asarray(q[k], np.float64) for k in HAND] + [(pt * r ** n).sum(1) for n in (1, 2, 3)], 1)
    return np.nan_to_num(hd, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================================ pool (jax env)
def mode_pool(a):
    import jax, jax.numpy as jnp
    from interp.jedi_core import Jedi, sample
    from interp import e36_counterfactual_flip_jedi as E36
    from interp.e35_prong_sharing_route_jedi import make_edits
    pf, pt_, nc = (60, 40, 120) if a.smoke else (4000, 2000, 3000)
    net = Jedi(64)
    xF, yF, _ = sample(pf, 808); xT, yT, _ = sample(pt_, 809)
    iF, iT, i0 = indices(pf, 808), indices(pt_, 809), indices(6000, 0)
    keep = ~np.isin(iT, np.concatenate([iF, i0])); xT, yT = np.ascontiguousarray(xT[keep]), yT[keep]
    perm = np.random.default_rng(0).permutation(len(yF)); nf = 2 * len(yF) // 3
    X = dict(fit=np.ascontiguousarray(xF[perm[:nf]]), dev=np.ascontiguousarray(xF[perm[nf:]]), test=xT)
    Y = dict(fit=yF[perm[:nf]], dev=yF[perm[nf:]], test=yT)
    J = len(xT); real = xT[..., 0] > 0; nn = real.sum(1); BASE, BSET = {}, {}
    info = dict(N_fit=int(nf), N_dev=int(len(yF) - nf), N_test=int(J), test_removed=int((~keep).sum()))
    log('jets', json.dumps(info))
    # trans
    tr = xT.copy(); tr[..., 1] += np.float32(0.06) * real; X['trans'] = tr; BASE['trans'] = np.arange(J); BSET['trans'] = 'test'
    # soft: one extra particle in the first empty slot
    js = np.flatnonzero(nn < 64); rng = np.random.default_rng(8094); so = xT[js].copy()
    ptmin = np.where(real[js], xT[js][..., 0], np.inf).min(1); r_ = 0.4 * np.sqrt(rng.uniform(size=len(js))); ph = rng.uniform(0, 2 * np.pi, len(js))
    so[np.arange(len(js)), nn[js], 0] = 0.5 * ptmin; so[np.arange(len(js)), nn[js], 1] = r_ * np.cos(ph); so[np.arange(len(js)), nn[js], 2] = r_ * np.sin(ph)
    X['soft'] = so.astype(np.float32); BASE['soft'] = js; BSET['soft'] = 'test'
    info['soft'] = dict(N=int(len(js)), pt_new_median=float(np.median(0.5 * ptmin)), pt_new_over_sumpt_median=float(np.median(0.5 * ptmin / xT[js][..., 0].sum(1))))
    # swap: pT of slots 13 and 14 exchanged
    jw = np.flatnonzero(nn >= 15); sw = xT[jw].copy(); sw[:, [13, 14], 0] = xT[jw][:, [14, 13], 0]; X['swap'] = sw; BASE['swap'] = jw; BSET['swap'] = 'test'
    info['swap'] = dict(N=int(len(jw)), rel_dpt_median=float(np.median((xT[jw][:, 13, 0] - xT[jw][:, 14, 0]) / xT[jw][:, 13, 0])))
    # subset C for D and cf
    C = np.sort(np.random.default_rng(8091).choice(J, min(nc, J), replace=False)); xC = xT[C]
    t = time.time(); XD, acc, _, dinfo, _, nnC, _ = make_edits(xC, 8092, a.smoke)
    dc = min([k for k in acc], key=lambda d: abs(d - 0.3)); m = np.flatnonzero(acc[dc] & (nnC >= 3))
    X['C_id'] = XD['id']; Y['C_id'] = yT[C]
    X['D'] = XD[(dc, 'D')][m]; BASE['D'] = m; BSET['D'] = 'C_id'; X['cn'] = XD[(dc, 'cn')][m]; BASE['cn'] = m; BSET['cn'] = 'C_id'
    info['D'] = dict(dc=float(dc), N_C=int(len(C)), accepted=int(len(m)), seconds=round(time.time() - t)); log('D', json.dumps(info['D']))
    del XD
    # cf: E36 minimal flips (same search as B7)
    t = time.time(); f = E36.float_program(net); fj = jax.jit(f); step = E36.make_step(f, 0.05); x0 = xC.astype(np.float64)
    lo_f = np.concatenate([np.asarray(fj(jnp.asarray(xC[i:i + 1000]))) for i in range(0, len(xC), 1000)]); c_f = lo_f.argmax(1)
    c_ex = net.run(xC)['logits'].argmax(1)
    idx = np.flatnonzero((c_f == c_ex) & np.isin(c_f, [0, 1, 2, 3])); tgt = np.array([E36.PARTNER[int(c)] for c in c_f[idx]])
    v, found, _ = E36.search(step, x0[idx], tgt, 30 if a.smoke else 200, 0.05, chunk=500, tag='cf')
    xe, _ = E36.apply_edit(x0[idx], v); dm, dax, _ = E36.check_constraints(x0[idx], xe); ok = found & (dm < 1e-3) & (dax < 1e-3)
    X['cf'] = xe[ok].astype(np.float32); BASE['cf'] = C[idx[ok]]; BSET['cf'] = 'test'; vok = v[ok]; ii = idx[ok]
    rv = [E36.random_edits(x0[ii[i0:i0 + 500]], vok[i0:i0 + 500], 1, np.random.default_rng([8095, i0]))[0] for i0 in range(0, len(ii), 500)]
    rv = np.concatenate(rv) if rv else np.zeros((0, 64, 3))
    xr, _ = E36.apply_edit(x0[ii], rv); dm2, dax2, _ = E36.check_constraints(x0[ii], xr); ok2 = (dm2 < 1e-3) & (dax2 < 1e-3)
    X['cfr'] = xr[ok2].astype(np.float32); BASE['cfr'] = C[ii[ok2]]; BSET['cfr'] = 'test'
    info['cf'] = dict(n_searched=int(len(idx)), found_float=float(found.mean()) if len(found) else None, n_ok=int(ok.sum()), n_random_ok=int(ok2.sum()),
                      eps_median=float(np.median(E36.eps_of(x0[idx][ok], vok))) if ok.any() else None, seconds=round(time.time() - t))
    log('cf', json.dumps(info['cf']))
    # network on every set; stage values on test and on DAS fit / dev subsets
    out = {}; acc_ = {}
    for s, xx in X.items():
        xx = np.ascontiguousarray(xx, np.float32); X[s] = xx; t = time.time()
        keep_ = ('S', 'h4', 'h5', 'z', 'logits') if s == 'test' else ('logits',)
        R = net.run(xx, keep=keep_)
        out[f'L|{s}'] = R['logits'].astype(np.float32); out[f'x|{s}'] = xx; out[f'std|{s}'] = std_obs(xx)
        if s == 'test':
            for k in ('S', 'h4', 'h5', 'z'): out[f'{k}|test'] = R[k].astype(np.float32)
        if s in BASE: out[f'base|{s}'] = BASE[s]
        if s in Y: out[f'y|{s}'] = Y[s]; acc_[s] = float((R['logits'].argmax(1) == Y[s]).mean())
        log(f'set {s}: {len(xx)} jets; network + standard observables {time.time() - t:.0f} s' + (f'; accuracy {acc_[s] * 100:.2f} %' if s in Y else ''))
    nd_f, nd_d = (200, 100) if a.smoke else (4000, 1500)
    rf = np.sort(np.random.default_rng(8096).choice(len(X['fit']), min(nd_f, len(X['fit'])), replace=False))
    rd = np.sort(np.random.default_rng(8097).choice(len(X['dev']), min(nd_d, len(X['dev'])), replace=False))
    for nm, s, rr in (('dasfit', 'fit', rf), ('dasdev', 'dev', rd)):
        R = net.run(X[s][rr], keep=('S', 'h4', 'logits')); out[f'rows|{nm}'] = rr
        out[f'S|{nm}'] = R['S'].astype(np.float32); out[f'h4|{nm}'] = R['h4'].astype(np.float32)
    nb = {s: out[f'L|{BSET[s]}'].argmax(1)[BASE[s]] for s in BASE}
    info['flip_rate'] = {s: float((out[f'L|{s}'].argmax(1) != nb[s]).mean()) for s in BASE}
    info['accuracy'] = acc_; info['accuracy_808'] = float((np.concatenate([out['L|fit'], out['L|dev']]).argmax(1) == np.concatenate([Y['fit'], Y['dev']])).mean())
    info['base_set'] = BSET; info['N'] = {s: int(len(x)) for s, x in X.items()}; info['C_rows'] = C.tolist()
    log('network accuracy: sample 808', f'{info["accuracy_808"] * 100:.2f} %', '; test (809)', f'{acc_["test"] * 100:.2f} %')
    log('flip rates %', json.dumps({s: round(v_ * 100, 2) for s, v_ in info['flip_rate'].items()}))
    np.savez_compressed(OUT / f'{tag(a)}_pool.npz', **out)
    (OUT / f'{tag(a)}_pool.json').write_text(json.dumps(info, indent=1, default=float))
    log('wrote pool')


# ============================================================================================ efp (venv-ef)
def mode_efp(a):
    import energyflow as ef
    Z = np.load(OUT / f'{tag(a)}_pool.npz'); sets = sorted(k[2:] for k in Z.files if k.startswith('x|'))
    es = ef.EFPSet('d<=5', 'p==1', measure='hadr', beta=1, normed=True, coords='ptyphim')
    log(f'energyflow {ef.__version__}; EFPSet d<=5 connected: {len(es.graphs())} graphs')
    out = {}
    for s in sets:
        t = time.time(); xx = Z[f'x|{s}']
        jets = [j[j[:, 0] > 0][:, [0, 1, 2]].astype(np.float64) for j in xx]
        out[f'efp|{s}'] = np.asarray(es.batch_compute(jets, n_jobs=1), np.float32); log(f'EFPs {s}: {out[f"efp|{s}"].shape} in {time.time() - t:.0f} s')
    out['graph_info'] = np.asarray([[int(d) for d in sp[:4]] for sp in es.specs]) if hasattr(es, 'specs') else np.zeros((0, 4))
    np.savez_compressed(OUT / f'{tag(a)}_efp.npz', **out); log('wrote EFPs')


# ============================================================================================ main helpers
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for r in range(n - 1, -1, -1):
        i = o[r]; prev = min(prev, p[i] * n / (r + 1)); q[i] = prev
    return q


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(a_only=n10, b_only=n01, p=1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue))


def ci(b): b = np.asarray(b, float); b = b[np.isfinite(b)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None]


def bootp(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / max(len(b), 1))) if len(b) else 1.0


def counts(N, B, seed): rng = np.random.default_rng(seed); return np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(B)]).astype(np.float32)


def kappa(nf, sf):
    po = (nf == sf).mean(); pn, ps = nf.mean(), sf.mean(); pe = pn * ps + (1 - pn) * (1 - ps)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def kappa_b(C, nf, sf):
    n = C.sum(1); a = C @ (nf == sf).astype(np.float32) / n; pn = C @ nf.astype(np.float32) / n; ps = C @ sf.astype(np.float32) / n
    pe = pn * ps + (1 - pn) * (1 - ps); return np.where(pe < 1, (a - pe) / np.maximum(1 - pe, 1e-12), 0.0)


def margin(L, c0):
    J = len(L); own = L[np.arange(J), c0]; o = L.copy(); o[np.arange(J), c0] = -np.inf; return own - o.max(1)


def mars_fit(Xf, Lf, Xd, Ld, keys, Ks, tagn, nsel):
    from jetdistill import mars
    from interp.e11_mars_all import knots_of
    from interp.b6_pareto import term_bits
    Q = {'fit': {k: Xf[:, i] for i, k in enumerate(keys)}, 'dev': {k: Xd[:, i] for i, k in enumerate(keys)}}
    knots = knots_of(Q['fit'], keys); NF = len(Lf); sel = np.random.default_rng(1).choice(NF, min(nsel, NF), replace=False)
    mars.MAX_TERMS, mars.STALL = max(Ks), 10 ** 6; fits = []
    for k in range(5):
        t = time.time(); c = 1.0 - min(Lf[:, k].min(), Ld[:, k].min()) + 0.5 * Lf[:, k].std()
        terms, _, path = mars.fit_neuron(dict(fit=Lf[:, k] + c, dev=Ld[:, k] + c), Q, keys, knots, sel)
        fits.append((terms, c)); log(f'  {tagn} MARS logit {CL[k]}: {len(terms)} terms (dev R^2 path end {path[-1] if path else None}); {time.time() - t:.0f} s')
    progs = {}
    for K in Ks:
        parts = [(terms[:K], mars._lstsq(terms[:K], Q['fit'], Lf[:, k] + c, NF), c) for k, (terms, c) in enumerate(fits)]
        bits = sum(32 + sum(sum(term_bits(t, len(keys))) for t in tr) for tr, _, _ in parts)
        progs[K] = (parts, float(bits))

    def predict(parts, Xa):
        Qd = {k: Xa[:, i] for i, k in enumerate(keys)}; J = len(Xa)
        return np.stack([np.stack([np.ones(J)] + [mars.term_value(t, Qd) for t in tr], 1) @ co - c for tr, co, c in parts], 1)
    return progs, predict


# ---------------------------------------------------------------- nested program P
def round_sig(x, b):
    if x is None or b >= 32: return x
    m, e = np.frexp(float(x)); return float(np.ldexp(np.round(np.ldexp(m, b)), e - b))


def load_P(K):
    from interp import e11_mars_all as E
    pre = 'e11' if K == 30 else f'b4m{K}'
    src = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
    rd = lambda n: json.loads((src / f'{pre}_{n}.json').read_text())
    fb = {'P': rd('fit_a_P'), **{s: rd(f'fit_b_{s}') for s in E.STG[1:]}}
    for s in E.STG: assert [u['unit'] for u in fb[s]['units']] == E.UNITS[s], (K, s)
    return fb


def level_P(fb0, b, drop_cost, mean_vals):
    """rounded / pruned copy of a stored nested program and its DL in bits"""
    import copy
    from interp import e11_mars_all as E
    fb = copy.deepcopy(fb0); nb = 32 if b >= 32 else b + 8; bits = 0.0; dropped = []
    for s in E.STG:
        nin = len(fb[s]['keys']); lb = float(np.log2(max(nin, 2)))
        for i, u in enumerate(fb[s]['units']):
            if drop_cost > 0 and E.LIVE[s][u['unit']] < drop_cost:
                u['terms'] = []; u['intercept'] = round_sig(float(mean_vals[s][i]), b); bits += nb; dropped.append(f'{s}{u["unit"]}'); continue
            u['intercept'] = round_sig(u['intercept'], b); bits += nb
            for t in u['terms']:
                t['coef'] = round_sig(t['coef'], b); bits += nb
                for kk, tk in (('kind', 't'), ('kind2', 't2')):
                    if kk == 'kind2' and not t.get('q2'): continue
                    if t.get(tk) is not None: t[tk] = round_sig(t[tk], b)
                    bits += lb + 2 + (0 if t[kk] == 'lin' else nb)
    return fb, bits, dropped


def physics(xx):
    from interp import e11_mars_all as E
    from jetdistill.observables import compute
    q = compute(np.asarray(xx, np.float64), 64)
    ph = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in q.items()}; ph.update(E.handbuilt(xx, False)); return ph


def chain_from(fb, V, start):
    """continue the nested program after stage `start` (stages after S never read physics)"""
    from interp import e11_mars_all as E
    V = dict(V)
    for s in E.STG[E.STG.index(start) + 1:]:
        V[s] = E.evaluate(fb[s]['units'], E.chain_inputs(s, V[E.PREV[s]], None), E.RELU[s])
    return V


# ---------------------------------------------------------------- network tails
def exact_tail(net, S_ref, L_ref):
    import jax, jax.numpy as jnp
    Ly = net.L; KB = {}
    for k in ('d4', 'd5', 'd6', 'd7'):     # the E22 make_tail (exact quantizers; checked there against run_jetwise to 1e-6)
        K, b = Ly[k].get_fused_qkernel_and_qbias(False, Ly[k].moving_mean, Ly[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))

    def dense(k, v, act=True):
        o = jnp.einsum(Ly[k].equation, Ly[k].iq(v, training=False), KB[k][0]) + KB[k][1]
        return Ly[k].activation(o) if act and Ly[k].activation is not None else o

    def f0(S, m4, v4, m5, v5, mz, vz):
        h4 = dense('d4', S); h4 = h4 * (1 - m4) + m4 * v4
        h5 = dense('d5', h4); h5 = h5 * (1 - m5) + m5 * v5
        z = dense('d6', h5, act=False); z = z * (1 - mz) + mz * vz
        return dense('d7', jax.nn.relu(z)).reshape(S.shape[0], -1)
    f = jax.jit(f0)
    for shp in ((), (1,)):
        try:
            def T(S, h4=None, m4=None, z=None, mz=None, shp=shp):
                J = len(S); sh = lambda W: (J,) + shp + (W,)
                a4 = (np.zeros(64, np.float32), np.zeros(sh(64), np.float32)) if h4 is None else (np.asarray(m4, np.float32), h4.reshape(sh(64)).astype(np.float32))
                az = (np.zeros(16, np.float32), np.zeros(sh(16), np.float32)) if z is None else (np.asarray(mz, np.float32), z.reshape(sh(16)).astype(np.float32))
                return np.asarray(f(S.reshape(sh(64)).astype(np.float32), a4[0], a4[1], np.zeros(32, np.float32), np.zeros(sh(32), np.float32), az[0], az[1]), np.float64).reshape(J, -1)
            d = float(np.abs(T(S_ref[:500]) - L_ref[:500]).max())
            log(f'exact tail shape {shp}: max |logit diff| vs network {d:.2e}')
            if d < 1e-4: return T, d
        except Exception as e:  # noqa: BLE001
            log(f'tail shape {shp} failed: {e!r}')
    raise SystemExit('no exact tail reproduces the network')


def float_tails(net):
    import jax, jax.numpy as jnp
    KB = {}
    for k in ('d4', 'd5', 'd6', 'd7'):
        K, b = net.KB[k]; b = np.asarray(b).ravel(); KB[k] = (jnp.asarray(np.asarray(K).reshape(-1, b.size), jnp.float32), jnp.asarray(b, jnp.float32))
    d = lambda k, v: v @ KB[k][0] + KB[k][1]
    from_h4 = lambda h4: d('d7', jax.nn.relu(d('d6', jax.nn.relu(d('d5', h4)))))
    from_S = lambda S: from_h4(jax.nn.relu(d('d4', S)))
    return dict(S=from_S, h4=from_h4)


def das_train(ft, st, Hb, Hs, tgt, Vb, Vs, vt, live, seed, steps):
    import jax, jax.numpy as jnp
    lm = jnp.asarray(np.isin(np.arange(Hb.shape[1]), live).astype(np.float32)); f = ft[st]

    def patched(w, B_, S_):
        u = w * lm; u = u / (jnp.linalg.norm(u) + 1e-9); return f(B_ + ((S_ - B_) @ u)[:, None] * u[None, :])

    def loss(w, B_, S_, T_): return -jnp.mean(jax.nn.log_softmax(patched(w, B_, S_))[jnp.arange(T_.shape[0]), T_])
    key = ('vg', st)
    if key not in das_train.cache:
        das_train.cache[key] = (jax.jit(jax.value_and_grad(loss)), jax.jit(lambda w, B_, S_: patched(w, B_, S_).argmax(1)))
    vg, pc = das_train.cache[key]
    Hb_, Hs_, T_ = jnp.asarray(Hb, jnp.float32), jnp.asarray(Hs, jnp.float32), jnp.asarray(tgt, jnp.int32)
    Vb_, Vs_ = jnp.asarray(Vb, jnp.float32), jnp.asarray(Vs, jnp.float32)
    rng = np.random.default_rng(seed); w = jnp.asarray(rng.normal(size=Hb.shape[1]).astype(np.float32)); m = jnp.zeros_like(w); v = jnp.zeros_like(w)
    best = (-1.0, None, 0); lr = 0.05
    for t in range(1, steps + 1):
        L, g = vg(w, Hb_, Hs_, T_); m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w = w - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t % 25 == 0 or t == steps:
            sc = float((np.asarray(pc(w, Vb_, Vs_)) == vt).mean())
            if sc > best[0]:
                u = np.asarray(w) * np.asarray(lm); best = (sc, u / max(np.linalg.norm(u), 1e-9), t)
    return best[1], dict(dev_match=best[0], step=best[2], loss_last=float(L))


das_train.cache = {}


# ============================================================================================ main
def mode_main(a):
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.ensemble import HistGradientBoostingRegressor
    from interp.jedi_core import Jedi
    from interp import e11_mars_all as E
    from interp.e11_verify import Last
    from interp.b6w_symreg import quantiser, hgb_bits_q
    Z = np.load(OUT / f'{tag(a)}_pool.npz'); pinfo = json.loads((OUT / f'{tag(a)}_pool.json').read_text())
    EF = None
    try:
        EF = np.load(OUT / f'{tag(a)}_efp.npz')
    except Exception as e:  # noqa: BLE001
        log('EFP file missing:', e)
    BSET = pinfo['base_set']; sets = sorted(k[2:] for k in Z.files if k.startswith('L|'))
    L = {s: Z[f'L|{s}'].astype(np.float64) for s in sets}; call = {s: L[s].argmax(1) for s in sets}
    base = {s: Z[f'base|{s}'] for s in BSET}
    STD = {s: Z[f'std|{s}'] for s in sets}
    if EF is not None:
        EFX = {s: np.concatenate([EF[f'efp|{s}'].astype(np.float64), STD[s][:, [0, 5, 6]]], 1) for s in sets}
        UN = {s: np.concatenate([STD[s], EF[f'efp|{s}'].astype(np.float64)], 1) for s in sets}
    RES = dict(procedure=__doc__, pool=pinfo, energyflow_in_job_env=False, energyflow_source='venv-ef (volume) via mode efp', programs=[], tests={})
    acc_t = float((call['test'] == Z['y|test']).mean()); RES['accuracy_test'] = acc_t
    log(f'network accuracy on test (809) {acc_t * 100:.2f} % (N {len(call["test"])}); sample 808 {pinfo["accuracy_808"] * 100:.2f} %')
    Bn = a.B; Nt = len(call['test']); Ct = counts(Nt, Bn, 11)
    # ---------- edit rows
    FAM = {}
    for f in EDITS + CTRL_EDITS:
        bs = BSET[f]; nb = call[bs][base[f]]; ne = call[f]; Lb = L[bs][base[f]]; Le = L[f]
        FAM[f] = dict(nb=nb, ne=ne, nf=nb != ne, dm=margin(Le, nb) - margin(Lb, nb), C=counts(len(nb), Bn, 100 + len(FAM)), bs=bs)
        log(f'edit {f}: N {len(nb)}, network call changes {FAM[f]["nf"].mean() * 100:.2f} %, mean margin change {FAM[f]["dm"].mean():+.3f}')
    RES['network'] = {f: dict(N=int(len(FAM[f]['nf'])), flip=float(FAM[f]['nf'].mean()), flip_ci=ci(FAM[f]['C'] @ FAM[f]['nf'].astype(np.float32) / len(FAM[f]['nf'])),
                              dmargin=float(FAM[f]['dm'].mean())) for f in FAM}
    # ---------- IIA pairs
    NP = min(a.npairs, Nt); rng = np.random.default_rng(8093); bi = rng.choice(Nt, NP, replace=False); si = rng.choice(Nt, NP, replace=True)
    same = si == bi; si[same] = (si[same] + 1) % Nt
    Cp = counts(NP, Bn, 12)
    net = Jedi(64); last = Last(net)
    Sx = {k: Z[f'{k}|test'].astype(np.float64) for k in ('S', 'h4', 'h5', 'z')}
    T, tail_diff = exact_tail(net, Sx['S'], L['test']); RES['tail_check_max_diff'] = tail_diff
    nbase = call['test'][bi]
    # ---------- program store: name -> dict(family, bits, logits per set, iia)
    PROG = {}
    def add(name, fam, bits, pred_sets, extra=None):
        PROG[name] = dict(name=name, family=fam, bits=float(bits), L={s: np.asarray(v, np.float64) for s, v in pred_sets.items()}, extra=extra or {})
    # ---------- P levels
    t0 = time.time(); xs = {s: Z[f'x|{s}'] for s in sets}
    PH = {}
    for s in ['test'] + EDITS + CTRL_EDITS + ['C_id']:
        t = time.time(); PH[s] = physics(xs[s]); log(f'physics set {s} ({len(xs[s])} jets) {time.time() - t:.0f} s')
    rfm = np.random.default_rng(8098).choice(len(xs['fit']), min(3000, len(xs['fit'])), replace=False); PHf = physics(xs['fit'][rfm])
    stored = {K: load_P(K) for K in sorted({lv[1] for lv in P_LEVELS})}
    PV = {}
    for name, K, b, dc in P_LEVELS:
        from interp.e11_verify import chain
        means = None
        if dc > 0:
            Vf = chain(stored[K], PHf); means = {s: Vf[s].mean(0) for s in E.STG}
        fb, bits, dropped = level_P(stored[K], b, dc, means)
        preds = {}
        for s in PH:
            V = chain(fb, PH[s]); preds[s] = last.jax_logits(V['z'])
            if s == 'test': PV[name] = (fb, V)
        add(name, 'P', bits, preds, dict(K=K, bits_per_number=32 if b >= 32 else b + 8, dropped_units=dropped))
        log(f'{name}: K {K}, number bits {32 if b >= 32 else b + 8}, dropped {len(dropped)} units; DL {bits:.0f} bits; same call test '
            f'{(preds["test"].argmax(1) == call["test"]).mean() * 100:.2f} %')
    del PH, PHf
    log(f'P levels {time.time() - t0:.0f} s')
    # ---------- baselines
    perm = np.random.default_rng(8099).permutation(len(L['fit']))
    for fam, XX, keys, Ks in (('STD', STD, STD_NAMES, a.std_k), ('EFP', EFX if EF is not None else None, None, a.efp_k)):
        if XX is None: log(f'{fam}: skipped (no EFP file)'); continue
        keys = keys or [f'EFP{i}' for i in range(XX['fit'].shape[1] - 3)] + ['m', 'sum pT', 'N']
        for shuf in (False, True):
            Lf = L['fit'][perm] if shuf else L['fit']
            Ks_ = (30,) if shuf else Ks
            progs, pred = mars_fit(XX['fit'], Lf, XX['dev'], L['dev'], keys, Ks_, fam + (' shuffled' if shuf else ''), a.nsel)
            for K, (parts, bits) in progs.items():
                nm = f'{fam}{K}' + ('_shuf' if shuf else '')
                add(nm, fam + ('_shuf' if shuf else ''), bits, {s: pred(parts, XX[s]) for s in sets}, dict(K=K, inputs=keys))
                PROG[nm]['predict'] = (lambda Xa, parts=parts, pred=pred: pred(parts, Xa)); PROG[nm]['X'] = XX
                log(f'{nm}: DL {bits:.0f} bits; same call test {(PROG[nm]["L"]["test"].argmax(1) == call["test"]).mean() * 100:.2f} %, dev {(PROG[nm]["L"]["dev"].argmax(1) == call["dev"]).mean() * 100:.2f} %')
    if EF is not None:
        F = UN['fit'].shape[1]
        for q in (4, 8):
            qf = quantiser(UN['fit'], q)
            for lv in (4, 8, 16):
                for it in a.gb_iters:
                    rs = [HistGradientBoostingRegressor(max_leaf_nodes=lv, max_iter=it, learning_rate=0.5, early_stopping=False, random_state=0).fit(qf(UN['fit']), L['fit'][:, k]) for k in range(5)]
                    nm = f'GBDT_q{q}_L{lv}_T{it}'; pr = lambda Xa, rs=rs, qf=qf: np.stack([r.predict(qf(Xa)) for r in rs], 1)
                    add(nm, 'GBDT', sum(hgb_bits_q(r, F, q) for r in rs), {s: pr(UN[s]) for s in sets}, dict(q=q, leaves=lv, iters=it, inputs=STD_NAMES + [f'EFP{i}' for i in range(F - 10)]))
                    PROG[nm]['predict'] = pr; PROG[nm]['X'] = UN
        qf = quantiser(UN['fit'], 8)
        rs = [HistGradientBoostingRegressor(max_leaf_nodes=8, max_iter=30, learning_rate=0.5, early_stopping=False, random_state=0).fit(qf(UN['fit']), L['fit'][perm][:, k]) for k in range(5)]
        pr = lambda Xa, rs=rs: np.stack([r.predict(qf(Xa)) for r in rs], 1)
        add('GBDT_shuf', 'GBDT_shuf', sum(hgb_bits_q(r, F, 8) for r in rs), {s: pr(UN[s]) for s in sets}, dict(inputs=STD_NAMES + [f'EFP{i}' for i in range(F - 10)])); PROG['GBDT_shuf']['predict'] = pr; PROG['GBDT_shuf']['X'] = UN
        log('GBDT: ' + '; '.join(f'{n} {p["bits"]:.0f} b sc {(p["L"]["test"].argmax(1) == call["test"]).mean() * 100:.1f}' for n, p in PROG.items() if p['family'].startswith('GBDT')))
    # ---------- natural + edit scores
    PER = {}
    for nm, p in PROG.items():
        sc = p['L']['test'].argmax(1) == call['test']; p['same_call'] = [float(sc.mean())] + ci(Ct @ sc.astype(np.float32) / Nt)
        p['same_call_dev'] = float((p['L']['dev'].argmax(1) == call['dev']).mean()) if 'dev' in p['L'] else None
        PER[nm] = dict(nat=sc); p['edits'] = {}
        for f in EDITS + CTRL_EDITS:
            n = FAM[f]; Pb = p['L'][n['bs']][base[f]]; Pe = p['L'][f]; sf = Pb.argmax(1) != Pe.argmax(1); ag = sf == n['nf']
            sdm = margin(Pe, n['nb']) - margin(Pb, n['nb']); C = n['C']; N_ = len(sf)
            bk = kappa_b(C, n['nf'], sf); rb = (C @ sdm.astype(np.float32)) / np.where(np.abs(C @ n['dm'].astype(np.float32)) > 1e-12, C @ n['dm'].astype(np.float32), np.nan)
            p['edits'][f] = dict(kappa=[kappa(n['nf'], sf)] + ci(bk), agree=float(ag.mean()), prog_flip=float(sf.mean()),
                                 ratio=[float(sdm.mean() / n['dm'].mean()) if abs(n['dm'].mean()) > 1e-12 else None] + ci(rb),
                                 prog_dmargin=float(sdm.mean()))
            PER[nm][f] = dict(sf=sf, ag=ag, bk=bk)
    log('natural + edit scores done')
    # ---------- IIA
    t0 = time.time(); live = {s: [int(u) for u in np.flatnonzero(Sx[s].std(0) > 1e-9)] for s in ('S', 'h4', 'z')}
    RES['live_units'] = {s: len(v) for s, v in live.items()}
    def net_swap(st, units):
        if st == 'S':
            Sp = Sx['S'][bi].copy(); Sp[:, units] = Sx['S'][si][:, units]; return T(Sp).argmax(1)
        m = np.zeros(WID[st]); m[units] = 1
        if st == 'h4': return T(Sx['S'][bi], h4=Sx['h4'][si], m4=m).argmax(1)
        return T(Sx['S'][bi], z=Sx['z'][si], mz=m).argmax(1)
    def iia_stats(pc, nc, nb_):
        ag = pc == nc; eff = nc != nb_; n_e = Cp @ eff.astype(np.float32)
        b_all = Cp @ ag.astype(np.float32) / NP; b_eff = (Cp @ (ag & eff).astype(np.float32)) / np.where(n_e > 0, n_e, np.nan)
        return dict(iia=[float(ag.mean())] + ci(b_all), iia_eff=[float(ag[eff].mean()) if eff.any() else None] + ci(b_eff),
                    n_eff=int(eff.sum()), net_change=float(eff.mean()), prog_change=None), b_eff, ag, eff
    NETSW = {}; rngp = np.random.default_rng(8100)
    for vn, (st, units) in P_VARS.items():
        u = units if units is not None else E.UNITS[st]
        NETSW[vn] = dict(al=net_swap(st, u), perm=[])
        pool = [x for x in E.UNITS[st] if x not in u] if st != 'z' else [x for x in live['z'] if x not in u]
        if len(pool) < len(u): pool = [x for x in live[st] if x not in u]
        for d in range(a.nperm):
            pu = list(rngp.choice(pool, len(u), replace=False)) if len(pool) >= len(u) else list(rngp.choice(live[st], len(u), replace=False))
            NETSW[vn]['perm'].append((pu, net_swap(st, pu)))
    log(f'network unit swaps {time.time() - t0:.0f} s')
    IIA_B = {}
    for name, K, b, dc in P_LEVELS:
        fb, V = PV[name]; p = PROG[name]; p['iia'] = {}; IIA_B[name] = {}
        pbase = last.jax_logits(V['z'][bi]).argmax(1)
        for vn, (st, units) in P_VARS.items():
            u = units if units is not None else E.UNITS[st]; cols = [E.UNITS[st].index(x) for x in u]
            Vb = {s: V[s][bi] for s in E.STG}; Vb[st] = Vb[st].copy(); Vb[st][:, cols] = V[st][si][:, cols]
            pc = last.jax_logits(chain_from(fb, Vb, st)['z']).argmax(1)
            r, beff, ag, eff = iia_stats(pc, NETSW[vn]['al'], nbase); r['prog_change'] = float((pc != pbase).mean())
            r['iia_eff_noswap_program'] = float((pbase[eff] == NETSW[vn]['al'][eff]).mean()) if eff.any() else None
            pr_e, pr_b = [], []
            for pu, ncp in NETSW[vn]['perm']:
                rr, bb, _, _ = iia_stats(pc, ncp, nbase); pr_e.append(rr['iia_eff'][0]); pr_b.append(bb)
            if units is None:      # every live S unit is in V: no disjoint unit set of the same size exists
                pr_e, pr_b = [None], [np.full(Bn, np.nan)]
            pbm = np.nanmean(np.stack(pr_b), 0)
            r['perm'] = dict(iia_eff=[float(np.nanmean([x for x in pr_e if x is not None])) if any(x is not None for x in pr_e) else None] + ci(pbm), draws=[x for x in pr_e],
                             units=[[int(x) for x in pu] for pu, _ in NETSW[vn]['perm']])
            dd = beff - pbm; r['aligned_minus_perm'] = dict(d=float(r['iia_eff'][0] - r['perm']['iia_eff'][0]) if None not in (r['iia_eff'][0], r['perm']['iia_eff'][0]) else None, ci=ci(dd), p=bootp(dd) if units is not None else None)
            p['iia'][vn] = r; IIA_B[name][vn] = beff
        effs = [p['iia'][v]['iia_eff'][0] for v in P_VARS if p['iia'][v]['iia_eff'][0] is not None]
        p['iia_eff_mean'] = [float(np.mean(effs))] + ci(np.nanmean(np.stack([IIA_B[name][v] for v in P_VARS]), 0))
        p['iia_eff_perm_mean'] = float(np.mean([p['iia'][v]['perm']['iia_eff'][0] for v in P_VARS if p['iia'][v]['perm']['iia_eff'][0] is not None]))
        log(f'{name} IIA: mean effective {p["iia_eff_mean"][0]:.3f} (permuted {p["iia_eff_perm_mean"]:.3f}); ' +
            '; '.join(f'{v.split(" (")[0]} {fmt(p["iia"][v]["iia_eff"][0])}/{fmt(p["iia"][v]["perm"]["iia_eff"][0])} (n {p["iia"][v]["n_eff"]})' for v in P_VARS))
    # baselines: DAS at S / h4
    ft = float_tails(net)
    fl_ok = float((np.asarray(ft['S'](Sx['S'][:2000].astype(np.float32))).argmax(1) == call['test'][:2000]).mean()); RES['float_tail_call_match'] = fl_ok
    log(f'float tail (DAS training) reproduces the exact call on {fl_ok * 100:.2f} % of 2000 test jets')
    D_ = {nm: dict(rows=Z[f'rows|{nm}'], S=Z[f'S|{nm}'].astype(np.float64), h4=Z[f'h4|{nm}'].astype(np.float64)) for nm in ('dasfit', 'dasdev')}
    rq = np.random.default_rng(8101); nfp, ndp = len(D_['dasfit']['rows']), len(D_['dasdev']['rows'])
    fpb, fps = rq.integers(0, nfp, nfp), rq.integers(0, nfp, nfp); dpb, dps = rq.integers(0, ndp, ndp), rq.integers(0, ndp, ndp)
    live_h4 = live['h4']; live_S = live['S']
    for nm, p in PROG.items():
        if 'predict' not in p: continue
        XX = p['X']; pr = p['predict']
        Xd = XX['dev'][D_['dasdev']['rows']]; Xf = XX['fit'][D_['dasfit']['rows']]; nfe = Xd.shape[1]
        cb = pr(Xd[dpb]).argmax(1); ch = []
        for c in range(nfe):
            Xm = Xd[dpb].copy(); Xm[:, c] = Xd[dps][:, c]; ch.append(float((pr(Xm).argmax(1) != cb).mean()))
        top = list(np.argsort(ch)[::-1][:2]); p['iia'] = {}; IIA_B[nm] = {}
        pbase = p['L']['test'][bi].argmax(1)
        for c in top:
            vn = f'input {c} ({p["extra"].get("inputs", [None] * (c + 1))[c] if p["extra"].get("inputs") else c})'
            Xm = Xf[fpb].copy(); Xm[:, c] = Xf[fps][:, c]; tf_ = pr(Xm).argmax(1)
            Xm = Xd[dpb].copy(); Xm[:, c] = Xd[dps][:, c]; td_ = pr(Xm).argmax(1)
            best = None
            for st in ('S', 'h4'):
                lv_ = live_S if st == 'S' else live_h4
                w, inf = das_train(ft, st, D_['dasfit'][st][fpb], D_['dasfit'][st][fps], tf_, D_['dasdev'][st][dpb], D_['dasdev'][st][dps], td_, lv_, 8102 + c, a.das_steps)
                if best is None or inf['dev_match'] > best[2]['dev_match']: best = (st, w, inf)
            st, w, inf = best
            Xt = XX['test'][bi].copy(); Xt[:, c] = XX['test'][si][:, c]; pc = pr(Xt).argmax(1)
            def net_dir(st, w):
                Hb, Hs = Sx[st][bi], Sx[st][si]; Hp = Hb + ((Hs - Hb) @ w)[:, None] * w[None, :]
                return (T(Hp) if st == 'S' else T(Sx['S'][bi], h4=Hp, m4=np.ones(64))).argmax(1)
            r, beff, ag, eff = iia_stats(pc, net_dir(st, w), nbase); r['prog_change'] = float((pc != pbase).mean())
            r['das'] = dict(stage=st, **inf); r['prog_change_dev'] = ch[c]
            pr_e, pr_b = [], []
            lv_ = live_S if st == 'S' else live_h4
            for d in range(a.nperm):
                u = np.zeros(64); u[lv_] = rngp.normal(size=len(lv_)); u /= np.linalg.norm(u)
                rr, bb, _, _ = iia_stats(pc, net_dir(st, u), nbase); pr_e.append(rr['iia_eff'][0]); pr_b.append(bb)
            pbm = np.nanmean(np.stack(pr_b), 0)
            r['perm'] = dict(iia_eff=[float(np.nanmean([x for x in pr_e if x is not None])) if any(x is not None for x in pr_e) else None] + ci(pbm), draws=pr_e)
            dd = beff - pbm; r['aligned_minus_perm'] = dict(d=(r['iia_eff'][0] - r['perm']['iia_eff'][0]) if None not in (r['iia_eff'][0], r['perm']['iia_eff'][0]) else None, ci=ci(dd), p=bootp(dd))
            p['iia'][vn] = r; IIA_B[nm][vn] = beff
        effs = [r['iia_eff'][0] for r in p['iia'].values() if r['iia_eff'][0] is not None]
        p['iia_eff_mean'] = [float(np.mean(effs)) if effs else None] + ci(np.nanmean(np.stack(list(IIA_B[nm].values())), 0))
        pe_ = [r['perm']['iia_eff'][0] for r in p['iia'].values() if r['perm']['iia_eff'][0] is not None]
        p['iia_eff_perm_mean'] = float(np.mean(pe_)) if pe_ else None
        log(f'{nm} IIA: ' + '; '.join(f'{v}: eff {r["iia_eff"][0]} (n {r["n_eff"]}), perm {r["perm"]["iia_eff"][0]}, DAS {r["das"]["stage"]} dev {r["das"]["dev_match"]:.3f}' for v, r in p['iia'].items()))
    log(f'IIA done {time.time() - t0:.0f} s')
    # ---------- matched-DL tests
    fams = [f for f in ('STD', 'EFP', 'GBDT') if any(p['family'] == f for p in PROG.values())]
    tests = []
    for name, *_ in P_LEVELS:
        pb = PROG[name]['bits']
        for fam in fams:
            mem = [n for n, p in PROG.items() if p['family'] == fam]
            ok = [n for n in mem if PROG[n]['bits'] <= pb]; above = not ok
            m = max(ok, key=lambda n: PROG[n]['same_call_dev']) if ok else min(mem, key=lambda n: PROG[n]['bits'])
            d = PER[name]['nat'].astype(np.float32) - PER[m]['nat']; bd = Ct @ d / Nt
            tests.append(dict(P=name, family=fam, member=m, member_above_P_bits=above, score='natural same call', d=float(d.mean()), ci=ci(bd), p=bootp(bd),
                              mcnemar=mcnemar(PER[name]['nat'], PER[m]['nat'])))
            for f in EDITS:
                bk = PER[name][f]['bk'] - PER[m][f]['bk']
                tests.append(dict(P=name, family=fam, member=m, member_above_P_bits=above, score=f'kappa {f}', d=PROG[name]['edits'][f]['kappa'][0] - PROG[m]['edits'][f]['kappa'][0],
                                  ci=ci(bk), p=bootp(bk), mcnemar=mcnemar(PER[name][f]['ag'], PER[m][f]['ag'])))
            if 'iia_eff_mean' in PROG[m] and PROG[m]['iia_eff_mean'][0] is not None:
                bp = np.nanmean(np.stack([IIA_B[name][v] for v in P_VARS]), 0); bm = np.nanmean(np.stack(list(IIA_B[m].values())), 0); dd = bp - bm
                tests.append(dict(P=name, family=fam, member=m, member_above_P_bits=above, score='effective IIA (mean over variables)',
                                  d=PROG[name]['iia_eff_mean'][0] - PROG[m]['iia_eff_mean'][0], ci=ci(dd), p=bootp(dd), mcnemar=None))
    q = bh([t['p'] for t in tests]); qm = bh([t['mcnemar']['p'] if t['mcnemar'] else t['p'] for t in tests])
    for t, a_, b_ in zip(tests, q, qm): t['bh'] = float(a_); t['bh_mcnemar'] = float(b_)
    RES['tests']['matched_dl'] = tests; RES['tests']['n_bh'] = len(tests)
    pt = [(name, v, PROG[name]['iia'][v]['aligned_minus_perm']['p']) for name, *_ in P_LEVELS for v in P_VARS if PROG[name]['iia'][v]['aligned_minus_perm']['p'] is not None]
    qq = bh([x[2] for x in pt])
    RES['tests']['P_aligned_vs_perm'] = [dict(P=n, var=v, p=p, bh=float(q_), d=PROG[n]['iia'][v]['aligned_minus_perm']['d'], ci=PROG[n]['iia'][v]['aligned_minus_perm']['ci'])
                                         for (n, v, p), q_ in zip(pt, qq)]
    for t in tests:
        if t['P'] in ('P30', 'P8', 'P2r10'):
            log(f"{t['P']} vs {t['family']} ({t['member']}{' ABOVE' if t['member_above_P_bits'] else ''}) {t['score']}: {t['d']:+.3f} {[round(x, 3) if x is not None else None for x in t['ci']]} BH {t['bh']:.3g}")
    # ---------- write
    RES['programs'] = [{k: v for k, v in p.items() if k not in ('L', 'predict', 'X')} for p in PROG.values()]
    (OUT / f'{tag(a)}.json').write_text(json.dumps(RES, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)))
    np.savez_compressed(OUT / f'{tag(a)}_perjet.npz', **{f'{n}|nat': v['nat'] for n, v in PER.items()},
                        **{f'{n}|{f}|ag': v[f]['ag'] for n, v in PER.items() for f in EDITS + CTRL_EDITS}, pairs_base=bi, pairs_src=si)
    log(f'wrote {OUT}/{tag(a)}.json')
    try:
        plot(RES, OUT / f'{tag(a)}.png')
    except Exception as ex:  # noqa: BLE001
        log('plot failed:', repr(ex))


STYLE = dict(P=('nested program (rank-window hinges -> S -> mass -> z)', '#1c5cab', 'o'), STD=('MARS on standard observables', '#d0661b', 's'),
             EFP=('MARS on EFPs (d<=5) + m, pT, N', '#2a9d4b', '^'), GBDT=('gradient-boosted trees on STD + EFP', '#7a3fa0', 'D'))


def plot(RES, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    P = RES['programs']; net = RES['network']
    panels = [('nat', 'natural jets: same call as network (%)')] + [(f, f'{EDIT_LABEL[f]}\n(network changes {net[f]["flip"] * 100:.0f} % of calls)') for f in EDITS] + [('iia', 'interchange: effective IIA\n(dashed: permuted alignment)')]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.2)); axs = axs.ravel()
    for ax, (k, title) in zip(axs, panels):
        for fam, (lab, col, mk) in STYLE.items():
            rows = sorted([p for p in P if p['family'] == fam], key=lambda p: p['bits'])
            if not rows: continue
            x = [p['bits'] / 1000 for p in rows]
            if k == 'nat': y = [p['same_call'][0] * 100 for p in rows]; e = [[(p['same_call'][0] - p['same_call'][1]) * 100 for p in rows], [(p['same_call'][2] - p['same_call'][0]) * 100 for p in rows]]
            elif k == 'iia':
                rows = [p for p in rows if p.get('iia_eff_mean') and p['iia_eff_mean'][0] is not None]; x = [p['bits'] / 1000 for p in rows]
                y = [p['iia_eff_mean'][0] for p in rows]; e = [[p['iia_eff_mean'][0] - p['iia_eff_mean'][1] for p in rows], [p['iia_eff_mean'][2] - p['iia_eff_mean'][0] for p in rows]]
                ax.plot(x, [p['iia_eff_perm_mean'] for p in rows], color=col, ls='none' if fam == 'GBDT' else '--', marker='_' if fam == 'GBDT' else None, ms=7, lw=1.2)
            else:
                y = [p['edits'][k]['kappa'][0] for p in rows]; e = [[p['edits'][k]['kappa'][0] - p['edits'][k]['kappa'][1] for p in rows], [p['edits'][k]['kappa'][2] - p['edits'][k]['kappa'][0] for p in rows]]
            ax.errorbar(x, y, yerr=e, color=col, marker=mk, ms=4, lw=0 if fam == 'GBDT' else 1.5, elinewidth=1, capsize=2, label=lab + (' (each grid member)' if fam == 'GBDT' else ''))
        for p in P:
            if p['family'].endswith('_shuf'):
                xv = p['bits'] / 1000
                yv = p['same_call'][0] * 100 if k == 'nat' else (p['edits'][k]['kappa'][0] if k in EDITS else (p.get('iia_eff_mean') or [None])[0])
                if yv is not None: ax.plot([xv], [yv], marker='x', color='k', ms=6, ls='none', label='shuffled-label fits (control)' if p['family'] == 'STD_shuf' else None)
        ax.set_xscale('log'); ax.set_xlabel('description length (kbit)'); ax.set_title(title, fontsize=10)
        if k in EDITS: ax.axhline(0, color='k', lw=0.7, ls=':'); ax.set_ylabel("Cohen's kappa of call changes")
    axs[0].set_ylabel('same call (%)'); axs[-2].set_ylabel('effective IIA (pairs where the network call changes)')
    h, l = axs[0].get_legend_handles_labels(); axs[-1].axis('off'); axs[-1].legend(h, l, fontsize=9, loc='center', frameon=False)
    fig.suptitle('At equal description length the nested program matches the network on natural jets as well as the baselines (above ~60 kbit), '
                 'and follows its call changes more closely\nunder the 0.06 rad translation and under unit-level interchange; '
                 'the pT swap at ranks 13/14 changes too few calls (0.4 %) to separate programs', fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['pool', 'efp', 'main', 'plot'])
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--B', type=int, default=1000); ap.add_argument('--nsel', type=int, default=8000)
    ap.add_argument('--npairs', type=int, default=3000); ap.add_argument('--nperm', type=int, default=5); ap.add_argument('--das_steps', type=int, default=300)
    a = ap.parse_args()
    a.std_k = (2, 4, 8, 16, 30); a.efp_k = (2, 4, 8, 16, 30, 50); a.gb_iters = (3, 10, 30, 100)
    if a.smoke: a.B, a.npairs, a.das_steps, a.std_k, a.efp_k, a.gb_iters, a.nsel = 100, 200, 50, (2, 4), (2, 4), (3, 10), 2000
    log('args', vars(a))
    if a.mode == 'pool': mode_pool(a)
    elif a.mode == 'efp': mode_efp(a)
    elif a.mode == 'main': mode_main(a)
    else: plot(json.loads((OUT / f'{tag(a)}.json').read_text()), OUT / f'{tag(a)}.png')
    log('done')
