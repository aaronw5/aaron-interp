"""E22v2: second skeptic rerun of E22 on a fresh sample; checks the rotation and probe contrasts and the CI method.

Sample: sample(2000, SEED) minus the jets of sample(6000, 0) (E11 fits, probe fit), sample(2000, 11) (E22) and sample(2000, 91)
(E22v); 5000 ordered (base, source) pairs, pair seed 3.  L = the network; H = the E11 (b) program (as interp/e22_iia_jedi.py).
Parts
 (1) one-unit swaps at S, h4, h5, z (E22's main arm) on E(k) = pairs where L's call changes; ru_prog (program runner-up on the
     base) as the swap-free predictor; CIs from (a) the pair bootstrap used in E22 and (b) a jet bootstrap: multinomial weights
     w_j over jets, pair weight w_base * w_source (pigeonhole bootstrap for pairs that share jets).
 (2) random directions (E22 baseline ii), DRAWS directions per stage; one-sided p (directions >= unit-basis mean) and
     Holm over the 4 stages.
 (3) probe alignment (E22 baseline iii): ridge from the Q observables, fitted on sample(6000, 0).  'probe_raw' = E22's target
     (raw stage value; for z the pre-ReLU value); 'probe_matched' = the target the E11 formulas were fitted to (z clipped
     below at -0.2 sd of its positive part, as jetdistill.mars.fit_neuron; S, h4, h5 are post-ReLU in both).  Post-ReLU
     test R^2 of formula and both probes; McNemar formula vs probe on E(k), Holm within stage (as E22) and over all units.
Output: $AI_RESULTS/jedi/e22_verify2.json.   Usage: python -m interp.e22_verify2 [--seed 137] [--pairs 5000] [--draws 1000]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample
from interp.e11_mars_all import handbuilt, indices, evaluate, LIVE, UNITS, RELU

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'
POST = ('S', 'h4', 'h5', 'z'); NXT = dict(S='h4', h4='h5', h5='z', z=None)
NAMED = [('S', 0), ('S', 1), ('S', 29), ('S', 50), ('S', 19), ('z', 0), ('z', 1), ('z', 7), ('h4', 49), ('h5', 26)]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=137); ap.add_argument('--pairs', type=int, default=5000)
ap.add_argument('--draws', type=int, default=1000); ap.add_argument('--boot', type=int, default=500)
A = ap.parse_args(); log('args', vars(A))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); n = len(ps); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * ps[i])); adj[i] = run
    return adj


def mcnemar(a, b):
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    k = min(n10, n01)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    mx = max(lp); return dict(n10=n10, n01=n01, p=float(min(1.0, 2 * math.exp(mx) * sum(math.exp(v - mx) for v in lp))))


# ------------------------------------------------------------------ data, network
xt, yt, Qt = sample(2000, A.seed)
i0 = np.concatenate([indices(6000, 0), indices(2000, 11), indices(2000, 91)]); i1 = indices(2000, A.seed); keep = ~np.isin(i1, i0)
xt, yt = xt[keep], yt[keep]; Qt = {k: np.asarray(v[keep], np.float64) for k, v in Qt.items()}; J = len(yt)
phys = {**Qt, **handbuilt(xt)}
fb = {'P': json.loads((OUT / 'e11_fit_a_P.json').read_text())}
for s in ('g', 'S', 'h4', 'h5', 'z'): fb[s] = json.loads((OUT / f'e11_fit_b_{s}.json').read_text())
fa_S_keys = json.loads((OUT / 'e11_fit_a_S.json').read_text())['keys']
net = Jedi(64)
R = net.run(xt, keep=POST + ('logits',)); call = R['logits'].argmax(1); acc = float((call == yt).mean())
log(f'seed {A.seed}: jets {J} (overlap removed {int((~keep).sum())}); network accuracy {acc:.4f}')


def make_stage_fns():
    import jax, jax.numpy as jnp
    L = net.L; KB = {}
    for k in ('d4', 'd5', 'd6', 'd7'):
        K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))

    def mk(k, act):
        def f(v):
            z = jnp.einsum(L[k].equation, L[k].iq(v, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z
        return jax.jit(f)
    return dict(h4=mk('d4', True), h5=mk('d5', True), z=mk('d6', False)), jax.jit(lambda z: mk('d7', True)(jax.nn.relu(z)))


SF, D7 = make_stage_fns()


def tail(stage, V):
    cur = np.asarray(V, np.float32); s = stage
    while NXT[s] is not None:
        nx = NXT[s]; cur = np.asarray(SF[nx](cur), np.float32); s = nx
    return np.asarray(D7(cur), np.float64)


chk = {s: float(np.abs(tail(s, R[s]) - R['logits']).max()) for s in POST}
rr = np.random.default_rng(5).permutation(J)[:1000]; j0 = int(np.argmax(R['S'].std(0)))
Sv = R['S'][rr].copy(); Sv[:, j0] = R['S'][rr[::-1], j0]; m = np.zeros(64); m[j0] = 1
chk['S_swap_vs_run_jetwise'] = float(np.abs(tail('S', Sv) - net.run_jetwise(xt[rr], {'S': (m, Sv)})['logits']).max())
log('tail check max |diff|', chk); assert max(chk.values()) < 1e-5, chk

CH = {}; prev = None
for s in ('P', 'g', 'S', 'h4', 'h5', 'z'):
    d = dict(phys) if s in ('P', 'S') else {}
    if prev is not None: d.update({f'{prev}:{j}': CH[prev][:, i] for i, j in enumerate(UNITS[prev])})
    CH[s] = evaluate(fb[s]['units'], d, RELU[s]); prev = s


def H_from(stage, V):
    s = stage; cur = V
    while NXT[s] is not None:
        nx = NXT[s]; cur = evaluate(fb[nx]['units'], {f'{s}:{j}': cur[:, i] for i, j in enumerate(UNITS[s])}, RELU[nx]); s = nx
    z = np.zeros((len(cur), 16)); z[:, UNITS['z']] = cur
    return np.asarray(D7(z.astype(np.float32)), np.float64)


rng = np.random.default_rng(3); ib = rng.integers(0, J, A.pairs); isr = rng.integers(0, J, A.pairs)
while (bad := ib == isr).any(): isr[bad] = rng.integers(0, J, int(bad.sum()))
NB = call[ib]; HBlog = H_from('S', CH['S'][ib]); HB = HBlog.argmax(1); ru_prog = np.argsort(HBlog, 1)[:, -2]
HB_all = H_from('S', CH['S']).argmax(1)
res = dict(meta=dict(seed=A.seed, jets=J, overlap_removed=int((~keep).sum()), pairs=A.pairs, pair_seed=3, draws=A.draws, boot=A.boot,
                     network_acc=acc, program_acc=float((HB_all == yt).mean()), program_same_call_all_jets=float((HB_all == call).mean()),
                     program_same_call_bases=float((HB == NB).mean()), tail_check=chk,
                     unique_jets_in_pairs=int(len(np.unique(np.concatenate([ib, isr]))))))
log('meta', json.dumps(res['meta']))

# bootstrap weights: pairs (E22) and jets (pigeonhole)
RB = np.random.default_rng(4)
WP = np.stack([np.bincount(RB.integers(0, A.pairs, A.pairs), minlength=A.pairs) for _ in range(A.boot)]).astype(np.float32)
WJ = np.stack([np.bincount(RB.integers(0, J, J), minlength=J) for _ in range(A.boot)]).astype(np.float32)
WJ = WJ[:, ib] * WJ[:, isr]


def eff_boot(agree, E, W):
    num = W @ (agree & E).astype(np.float32); den = W @ E.astype(np.float32)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def q(b): return [float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))]


# ------------------------------------------------------------------ (1) one-unit swaps
Lc, Hc, per = {}, {}, {}
for s in POST:
    U = UNITS[s]; n = len(U); Lc[s] = np.zeros((n, A.pairs), int); Hc[s] = np.zeros((n, A.pairs), int)
    for i, j in enumerate(U):
        VL = R[s][ib].copy(); VL[:, j] = R[s][isr, j]; Lc[s][i] = tail(s, VL).argmax(1)
        V = CH[s][ib].copy(); V[:, i] = CH[s][isr, i]; Hc[s][i] = H_from(s, V).argmax(1)
    rows = []; bp, bj, bru = [], [], []
    for i, j in enumerate(U):
        E = Lc[s][i] != NB; ag = Hc[s][i] == Lc[s][i]; agr = ru_prog == Lc[s][i]
        r = dict(unit=j, cost=LIVE[s][j], n_eff=int(E.sum()), prog=float(ag[E].mean()) if E.any() else float('nan'),
                 ru_prog=float(agr[E].mean()) if E.any() else float('nan'))
        b1, b2, b3 = eff_boot(ag, E, WP), eff_boot(ag, E, WJ), eff_boot(agr, E, WJ)
        r['ci_pair'] = q(b1); r['ci_jet'] = q(b2); r['mcnemar_vs_ru_prog'] = mcnemar(ag[E], agr[E])
        bp.append(b1); bj.append(b2); bru.append(b3); rows.append(r)
    ok = np.array([r['n_eff'] >= 20 for r in rows]); vals = np.array([r['prog'] for r in rows])[ok]
    st = dict(mean=float(np.nanmean(vals)), ci_pair=q(np.nanmean(np.stack(bp)[ok], 0)), ci_jet=q(np.nanmean(np.stack(bj)[ok], 0)),
              ru_prog_mean=float(np.nanmean(np.array([r['ru_prog'] for r in rows])[ok])),
              prog_minus_ru_prog_ci_jet=q(np.nanmean(np.stack(bj)[ok] - np.stack(bru)[ok], 0)),
              range=[float(np.nanmin(vals)), float(np.nanmax(vals))], n_tested=int(ok.sum()))
    per[s] = dict(units=rows, stage=st)
    log(f'stage {s}: mean {st["mean"]:.3f} pair CI {np.round(st["ci_pair"], 3).tolist()} jet CI {np.round(st["ci_jet"], 3).tolist()}; '
        f'ru_prog {st["ru_prog_mean"]:.3f}; prog - ru_prog jet CI {np.round(st["prog_minus_ru_prog_ci_jet"], 3).tolist()}; range {np.round(st["range"], 3).tolist()}')
for s, j in NAMED:
    r = per[s]['units'][UNITS[s].index(j)]
    log(f'{s}{j}: |E| {r["n_eff"]} prog {r["prog"]:.3f} pair {np.round(r["ci_pair"], 3).tolist()} jet {np.round(r["ci_jet"], 3).tolist()} ru_prog {r["ru_prog"]:.3f} McNemar {r["mcnemar_vs_ru_prog"]}')
tests = [(s, k) for s in POST for k, r in enumerate(per[s]['units']) if r['n_eff'] >= 20]
adj = holm([per[s]['units'][k]['mcnemar_vs_ru_prog']['p'] for s, k in tests])
for (s, k), a in zip(tests, adj): per[s]['units'][k]['mcnemar_vs_ru_prog']['p_holm'] = float(a)
hi = sum(a < .05 and per[s]['units'][k]['mcnemar_vs_ru_prog']['n10'] > per[s]['units'][k]['mcnemar_vs_ru_prog']['n01'] for (s, k), a in zip(tests, adj))
res['per_unit'] = per; res['holm_vs_ru_prog'] = dict(n_tested=len(tests), prog_higher=int(hi)); log(f'Holm vs ru_prog: prog higher {hi}/{len(tests)}')

# ------------------------------------------------------------------ (2) random directions
rot = {}
for s in POST:
    U = UNITS[s]; n = len(U); sd = R[s][:, U].std(0); sd = np.where(sd > 0, sd, 1.0); rr_ = np.random.default_rng(500 + POST.index(s)); v = []
    for d in range(A.draws):
        r = rr_.normal(size=n); r /= np.linalg.norm(r)
        cL = ((R[s][isr][:, U] - R[s][ib][:, U]) / sd) @ r; cH = ((CH[s][isr] - CH[s][ib]) / sd) @ r
        VL = R[s][ib].copy(); VL[:, U] = VL[:, U] + sd * r * cL[:, None]; VH = CH[s][ib] + sd * r * cH[:, None]
        if RELU[s]: VL = np.maximum(VL, 0); VH = np.maximum(VH, 0)
        cl = tail(s, VL).argmax(1); ch = H_from(s, VH).argmax(1); E = cl != NB
        v.append([float((cl == ch)[E].mean()) if E.sum() >= 20 else np.nan, int(E.sum())])
    v = np.array(v); real = per[s]['stage']['mean']; fin = np.isfinite(v[:, 0])
    rot[s] = dict(rot_mean=float(np.nanmean(v[:, 0])), rot_interval=np.nanquantile(v[:, 0], [.025, .975]).tolist(), n_dirs=int(fin.sum()),
                  n_eff_median=float(np.median(v[:, 1])), unit_mean=real,
                  p_unit_gt=float((1 + (v[fin, 0] >= real).sum()) / (1 + fin.sum())), p_unit_lt=float((1 + (v[fin, 0] <= real).sum()) / (1 + fin.sum())))
    log(f'rotation {s}: directions {rot[s]["rot_mean"]:.3f} {np.round(rot[s]["rot_interval"], 3).tolist()} (|E| median {rot[s]["n_eff_median"]:.0f}); unit {real:.3f}; '
        f'p(unit > dirs) {rot[s]["p_unit_gt"]:.4f}')
adj = holm([rot[s]['p_unit_gt'] for s in POST])
for s, a in zip(POST, adj): rot[s]['p_unit_gt_holm4'] = float(a)
res['rotation'] = rot; log('rotation Holm (4 stages):', {s: round(rot[s]['p_unit_gt_holm4'], 4) for s in POST})

# ------------------------------------------------------------------ (3) probes: raw (E22) and matched-target
xf, yf, Qf = sample(6000, 0); Rf = net.run(xf, keep=POST); del xf
qk = [k for k in fa_S_keys if not k.startswith('H:') and k in Qf and k in Qt]
Xf = np.stack([np.asarray(Qf[k], np.float64) for k in qk], 1); Xt = np.stack([Qt[k] for k in qk], 1); del Qf
mu, sg = Xf.mean(0), Xf.std(0); sg = np.where(sg > 0, sg, 1); Xf = (Xf - mu) / sg; Xt = (Xt - mu) / sg
perm = np.random.default_rng(0).permutation(len(Xf)); nf = 2 * len(Xf) // 3; fi, di = perm[:nf], perm[nf:]
XtX = Xf[fi].T @ Xf[fi]


def ridge(Y):
    ym = Y[fi].mean(0); best = None
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        Wr = np.linalg.solve(XtX + lam * np.eye(len(qk)), Xf[fi].T @ (Y[fi] - ym)); err = ((Xf[di] @ Wr + ym - Y[di]) ** 2).mean()
        if best is None or err < best[0]: best = (err, lam, Wr)
    return best[1], Xt @ best[2] + ym


def r2h(p, t):  # post-ReLU R^2, as mars.py / E11
    h = np.maximum(t, 0); return 1 - ((h - np.maximum(p, 0)) ** 2).mean(0) / np.maximum(h.var(0), 1e-12)


probe = {}; allp = []
for s in POST:
    Y = Rf[s][:, UNITS[s]]
    if s == 'z':
        Ym = Y.copy()
        for c in range(Y.shape[1]):
            zp = Y[:, c][Y[:, c] > 0]; fl = -0.2 * (zp.std() if len(zp) > 50 else Y[:, c].std()); Ym[:, c] = np.maximum(Y[:, c], fl)
    else:
        Ym = Y
    lam_r, Pr = ridge(Y); lam_m, Pm = ridge(Ym)
    if RELU[s]: Pr = np.maximum(Pr, 0); Pm = np.maximum(Pm, 0)
    tgt = R[s][:, UNITS[s]]; R2 = dict(formula=r2h(CH[s], tgt), probe_raw=r2h(Pr, tgt), probe_matched=r2h(Pm, tgt))
    rows = []
    for i, j in enumerate(UNITS[s]):
        cl = Lc[s][i]; E = cl != NB; okf = (Hc[s][i] == cl)[E]; r = dict(unit=j, n_eff=int(E.sum()), formula=float(okf.mean()) if E.any() else float('nan'))
        for tag, P in (('probe_raw', Pr), ('probe_matched', Pm)):
            base = CH[s][ib].copy(); base[:, i] = P[ib, i]; V = base.copy(); V[:, i] = P[isr, i]; ch = H_from(s, V).argmax(1)
            okp = (ch == cl)[E]; r[tag] = float(okp.mean()) if E.any() else float('nan'); r['mcnemar_' + tag] = mcnemar(okf, okp)
        for k, v in R2.items(): r['r2_post_relu_' + k] = float(v[i])
        rows.append(r)
        if s == 'z' or (s, j) in NAMED:
            log(f'probe {s}{j}: R2 formula {r["r2_post_relu_formula"]:.3f} raw {r["r2_post_relu_probe_raw"]:.3f} matched {r["r2_post_relu_probe_matched"]:.3f}; '
                f'IIA_eff formula {r["formula"]:.3f} raw {r["probe_raw"]:.3f} {r["mcnemar_probe_raw"]} matched {r["probe_matched"]:.3f} {r["mcnemar_probe_matched"]}')
    for tag in ('probe_raw', 'probe_matched'):
        a = holm([r['mcnemar_' + tag]['p'] for r in rows])
        for r, x in zip(rows, a): r['mcnemar_' + tag]['p_holm_stage'] = float(x)
    probe[s] = dict(lambda_raw=lam_r, lambda_matched=lam_m, n_observables=len(qk), units=rows,
                    formula_lower_than={tag: int(sum(r['mcnemar_' + tag]['p_holm_stage'] < .05 and r['mcnemar_' + tag]['n10'] < r['mcnemar_' + tag]['n01'] for r in rows)) for tag in ('probe_raw', 'probe_matched')},
                    formula_higher_than={tag: int(sum(r['mcnemar_' + tag]['p_holm_stage'] < .05 and r['mcnemar_' + tag]['n10'] > r['mcnemar_' + tag]['n01'] for r in rows)) for tag in ('probe_raw', 'probe_matched')})
    log(f'probe stage {s}: formula higher (Holm within stage) raw {probe[s]["formula_higher_than"]["probe_raw"]}, matched {probe[s]["formula_higher_than"]["probe_matched"]}; '
        f'formula lower raw {probe[s]["formula_lower_than"]["probe_raw"]}, matched {probe[s]["formula_lower_than"]["probe_matched"]} of {len(rows)}')
res['probe'] = probe

p = OUT / 'e22_verify2.json'; p.write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {p}'); log('done')
