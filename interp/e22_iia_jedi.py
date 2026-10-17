"""E22 (JEDI-linear 64 p, 3-feature; claims A5, A6, B2, B3): causal abstraction of the E11 (b) nested program.

High-level model H: the E11 (b) program (results/jedi/e11_fit_b_*.json; P from e11_fit_a_P.json): P^ <- physics, g^ <- P^,
S^ <- g^ + physics, h4^ <- S^, h5^ <- h4^, z^ <- h5^, then the network's own last layer d7 on relu(z^) makes the call.
Low-level model L: the network (interp.jedi_core.Jedi); interventions at S, h4, h5 or z are run with the network's own layers
from that stage on (make_tail below, checked against Jedi.run_jetwise to 1e-6).
Sample: held-out E11 test jets, sample(per_class=2000, seed=11) minus the jets of the fitting sample sample(6000, 0)
(8848 jets); 5000 random ordered (base, source) pairs, base != source, pair seed 0.
Interchange intervention on variable V aligned to unit k of stage s:
  L: network on the base jet with unit k of s set to its value on the source jet;
  H: program on the base jet with V = formula k of s set to its chain value on the source jet, downstream formulas recomputed.
IIA(V) = fraction of pairs whose argmax calls agree between L and H; effective IIA = the same on E(V), the pairs where the
intervention changes L's call (relative to the network's call on the base jet). 95 % bootstrap CI over pairs (1000 resamples).
Joint: all live units of S / h4 / h5 / z at once, S + z at once, and random subsets (each live unit of S and z with p = 0.5;
also of all four stages with p = 0.25), 200 draws, against a misaligned subset of the same size.
Baselines: (i) permuted alignment: H's V_k against L's unit pi(k), pi a random derangement of the stage's live units, 200
draws; also the fixed-L version (L unit k, H variable j != k, averaged over j) on the same pairs; (ii) random rotations:
200 random unit directions r in the standardised live-unit space of a stage, V' = V_b + sd * r (r . (V_s - V_b) / sd) in L and H
(clipped at 0 for the ReLU stages S, h4, h5); (iii) probe alignment: unit k's formula replaced by a ridge probe from the Q
observable library (380 finite observables) fitted on the E11 fitting jets; McNemar of formula vs probe on E(V);
(iv) the untrained network (Jedi(untrained=True, seed=0)) with its E11 control fits (single-stage (a) fits on physics,
untrained tail), next to the same construction for the trained network with the E11 (a) fits.
Tests: H0 effective IIA(V) = permuted-alignment IIA, H1 greater; one-sided permutation p over the 200 draws (floor 1/201),
and a paired bootstrap-normal p of [IIA(V) - fixed-L null] on the same pairs; Holm across the per-unit variables with |E| >= 20.
Output: $AI_RESULTS/jedi/e22_iia.json, e22_iia.png.   Usage: python -m interp.e22_iia_jedi [--pairs 5000] [--draws 200]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample
from interp.e11_mars_all import handbuilt, indices, evaluate, LIVE, UNITS, RELU

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
POST = ('S', 'h4', 'h5', 'z'); NXT = dict(S='h4', h4='h5', h5='z', z=None); WID = dict(S=64, h4=64, h5=32, z=16)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


ap = argparse.ArgumentParser(); ap.add_argument('--pairs', type=int, default=5000); ap.add_argument('--draws', type=int, default=200)
ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--prefix', default='e22')
A = ap.parse_args(); log('args', vars(A))
RB = np.random.default_rng(1)


# ------------------------------------------------------------------ stats
def boot_ci(c):
    c = np.asarray(c, float); n = len(c)
    if n == 0: return [float('nan')] * 2
    idx = RB.integers(0, n, size=(A.boot, n), dtype=np.int32); m = c[idx].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); n = len(ps); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * ps[i])); adj[i] = run
    return adj


def mcnemar(a, b):
    """a, b bool (correct under alignment 1 / 2): exact two-sided binomial on the discordant pairs"""
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    k = min(n10, n01)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    mx = max(lp); return dict(n10=n10, n01=n01, p=float(min(1.0, 2 * math.exp(mx) * sum(math.exp(v - mx) for v in lp))))


def p_paired(d):
    """one-sided p of mean(d) > 0 (normal approximation, bootstrap-equivalent SE)"""
    d = np.asarray(d, float)
    if len(d) < 2: return 1.0
    s = d.std(ddof=1) / math.sqrt(len(d))
    if s <= 0: return 0.0 if d.mean() > 0 else 1.0
    return float(0.5 * math.erfc((d.mean() / s) / math.sqrt(2)))


def summ(cl, ch, base_call):
    """cl, ch: L and H calls; base_call: the network's call on the base jet"""
    agree = cl == ch; E = cl != base_call
    return dict(iia=float(agree.mean()), iia_ci=boot_ci(agree), n_eff=int(E.sum()), frac_eff=float(E.mean()),
                iia_eff=float(agree[E].mean()) if E.any() else float('nan'), iia_eff_ci=boot_ci(agree[E]),
                h_changed=float((ch != HB).mean()), iia_noneff=float(agree[~E].mean()) if (~E).any() else float('nan'))


# ------------------------------------------------------------------ network tails
def make_tail(net):
    import jax, jax.numpy as jnp
    L = net.L; KB = {}
    for k in ('d4', 'd5', 'd6', 'd7'):
        K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))

    def dense(k, v, act=True):
        z = jnp.einsum(L[k].equation, L[k].iq(v, training=False), KB[k][0]) + KB[k][1]
        return L[k].activation(z) if act and L[k].activation is not None else z

    def f(S, m4, v4, m5, v5, mz, vz):
        h4 = dense('d4', S); h4 = h4 * (1 - m4) + m4 * v4
        h5 = dense('d5', h4); h5 = h5 * (1 - m5) + m5 * v5
        z = dense('d6', h5, act=False); z = z * (1 - mz) + mz * vz
        return dense('d7', jax.nn.relu(z)).reshape(S.shape[0], -1)
    return jax.jit(f)


class Tail:
    """logits of the network from stage values; V = {stage: (J, W) full stage array}; the earliest given stage is the
    entry point (later stages given are overrides of the whole stage)"""
    def __init__(self, net, R, shp):
        self.f = make_tail(net); self.R = R; self.shp = shp

    def __call__(self, V, rows):
        J = len(rows); sh = lambda W: (J,) + tuple(d if d != 'W' else W for d in self.shp)
        S = V.get('S', self.R['S'][rows]); args = [S.reshape(sh(64)).astype(np.float32)]
        for s in ('h4', 'h5', 'z'):
            if s in V:
                args += [np.ones(WID[s], np.float32), V[s].reshape(sh(WID[s])).astype(np.float32)]
            else:
                args += [np.zeros(WID[s], np.float32), np.zeros(sh(WID[s]), np.float32)]
        out = []
        for i in range(0, J, 2500):
            a = [args[0][i:i + 2500]] + [x if x.ndim == 1 else x[i:i + 2500] for x in args[1:]]
            out.append(np.asarray(self.f(*a), np.float64))
        return np.concatenate(out)


def enable_tail(net, R, x):
    for shp in (('W',), (1, 'W')):
        try:
            T = Tail(net, R, shp); rows = np.arange(len(x))
            d0 = max(float(np.abs(T({s: R[s][rows]}, rows) - R['logits']).max()) for s in POST)
            # one per-unit S intervention against run_jetwise
            j = int(np.argmax(R['S'].std(0))); rr = np.random.default_rng(5).permutation(len(x))[:1000]
            Sv = R['S'][rr].copy(); Sv[:, j] = R['S'][rr[::-1], j]; m = np.zeros(64); m[j] = 1
            lo = net.run_jetwise(x[rr], {'S': (m, Sv)})['logits']; d1 = float(np.abs(T({'S': Sv}, rr) - lo).max())
            zv = R['z'][rr].copy(); zv[:, 0] = R['z'][rr[::-1], 0]; mz = np.zeros(16); mz[0] = 1
            lo = net.run_jetwise(x[rr], {'z': (mz, zv)})['logits']; d2 = float(np.abs(T({'z': zv}, rr) - lo).max())
            log(f'tail shape {shp}: max |diff| base {d0:.2e}, S-unit swap vs run_jetwise {d1:.2e}, z-unit swap {d2:.2e}')
            if max(d0, d1, d2) < 1e-5: return T, dict(shape=list(map(str, shp)), max_diff_base=d0, max_diff_S_swap=d1, max_diff_z_swap=d2)
        except Exception as e:  # noqa: BLE001
            log(f'tail shape {shp} failed: {e!r}')
    raise SystemExit('no tail reproduces the network')


# ------------------------------------------------------------------ data
xt, yt, Qt = sample(2000, 11); i0, i1 = indices(6000, 0), indices(2000, 11); keep = ~np.isin(i1, i0)
xt, yt = xt[keep], yt[keep]; Qt = {k: np.asarray(v[keep], np.float64) for k, v in Qt.items()}; J = len(yt)
phys = {**Qt, **handbuilt(xt)}
fb = {'P': json.loads((OUT / 'e11_fit_a_P.json').read_text())}
for s in ('g', 'S', 'h4', 'h5', 'z'): fb[s] = json.loads((OUT / f'e11_fit_b_{s}.json').read_text())
fa = {s: json.loads((OUT / f'e11_fit_a_{s}.json').read_text()) for s in POST}
net = Jedi(64)
R = net.run(xt, keep=POST + ('logits',)); call = R['logits'].argmax(1)
acc = float((call == yt).mean()); log(f'test jets {J} (overlap removed {int((~keep).sum())}); network accuracy {acc:.4f}')
T, tailinfo = enable_tail(net, R, xt)

# chain (formula) values per jet
CH = {}; prev = None
for s in ('P', 'g', 'S', 'h4', 'h5', 'z'):
    d = dict(phys) if s in ('P', 'S') else {}
    if prev is not None: d.update({f'{prev}:{j}': CH[prev][:, i] for i, j in enumerate(UNITS[prev])})
    CH[s] = evaluate(fb[s]['units'], d, RELU[s]); prev = s
    assert [u['unit'] for u in fb[s]['units']] == UNITS[s]


def H_from(stage, V, rows):
    """H's logits from stage values V (J, n_live) of the program (live units of stage, chain order); later stages recomputed"""
    s = stage; cur = V
    while NXT[s] is not None:
        nx = NXT[s]; cur = evaluate(fb[nx]['units'], {f'{s}:{j}': cur[:, i] for i, j in enumerate(UNITS[s])}, RELU[nx]); s = nx
    z = np.zeros((len(rows), 16)); z[:, UNITS['z']] = cur
    return T({'z': z}, rows)


rng = np.random.default_rng(0); ib = rng.integers(0, J, A.pairs); isr = rng.integers(0, J, A.pairs)
while (bad := ib == isr).any(): isr[bad] = rng.integers(0, J, int(bad.sum()))
NB = call[ib]                                                      # network call on the base jet
HB = H_from('S', CH['S'][ib], ib).argmax(1)                         # program call on the base jet
HB_all = H_from('S', CH['S'], np.arange(J)).argmax(1)
res = dict(meta=dict(jets=J, overlap_removed=int((~keep).sum()), sample='sample(2000, 11) minus sample(6000, 0)', pairs=A.pairs, pair_seed=0,
                     draws=A.draws, boot=A.boot, network_acc=acc, network_per_class_acc=[float((call[yt == c] == c).mean()) for c in range(5)],
                     program_acc=float((HB_all == yt).mean()), program_same_call_all_jets=float((HB_all == call).mean()),
                     program_same_call_bases=float((HB == NB).mean()), tail=tailinfo, units=UNITS, costs={s: LIVE[s] for s in POST}))
log(f'program on all test jets: acc {res["meta"]["program_acc"]:.4f}, same call {res["meta"]["program_same_call_all_jets"]:.4f}; on the {A.pairs} bases same call {res["meta"]["program_same_call_bases"]:.4f}')


def L_unit(stage, j, rows_b, rows_s, src=None):
    V = R[stage][rows_b].copy(); V[:, j] = R[stage][rows_s, j] if src is None else src; return T({stage: V}, rows_b)


def H_unit(stage, i, base, srcv):
    V = base.copy(); V[:, i] = srcv; return H_from(stage, V, ib)


# ------------------------------------------------------------------ (1) per-unit interchange + (i) permuted alignment
per = {}; Lc = {}; Hc = {}
for s in POST:
    U = UNITS[s]; n = len(U); Lc[s] = np.stack([L_unit(s, j, ib, isr).argmax(1) for j in U])
    Hc[s] = np.stack([H_unit(s, i, CH[s][ib], CH[s][isr, i]).argmax(1) for i in range(n)])
    E = Lc[s] != NB[None]                                          # E[j]: pairs where L's intervention on unit U[j] changes the call
    M = np.full((n, n), np.nan)                                    # M[i, j]: agreement of H_i with L_j on E_j
    for j in range(n):
        if E[j].sum() > 0: M[:, j] = (Hc[s][:, E[j]] == Lc[s][j, E[j]][None]).mean(1)
    rngp = np.random.default_rng(100 + POST.index(s)); null = np.full((A.draws, n), np.nan)
    for d in range(A.draws):
        while True:
            pi = rngp.permutation(n)
            if not (pi == np.arange(n)).any(): break
        null[d] = M[np.arange(n), pi]
    rows = []
    for i, j in enumerate(U):
        e = E[i]; ag = Hc[s][i] == Lc[s][i]; r = summ(Lc[s][i], Hc[s][i], NB)
        nd = null[:, i]; nd = nd[np.isfinite(nd)]
        fixed = (Hc[s][np.arange(n) != i][:, e] == Lc[s][i, e][None]).mean(0) if e.any() else np.array([])   # per-pair mean over j != i
        r.update(unit=j, cost=LIVE[s][j], perm_null_mean=float(nd.mean()) if len(nd) else float('nan'),
                 perm_null_interval=[float(np.quantile(nd, .025)), float(np.quantile(nd, .975))] if len(nd) else [float('nan')] * 2,
                 p_perm=float((1 + (nd >= r['iia_eff']).sum()) / (1 + len(nd))) if e.any() else float('nan'),
                 fixedL_null=float(fixed.mean()) if e.any() else float('nan'), fixedL_null_ci=boot_ci(fixed),
                 p_paired=p_paired(ag[e].astype(float) - fixed) if e.sum() >= 2 else float('nan'))
        rows.append(r)
        log(f'{s}{j} (cost {LIVE[s][j]:.2f} %): IIA {r["iia"]:.4f}; |E| {r["n_eff"]}; effective IIA {r["iia_eff"]:.3f} [{r["iia_eff_ci"][0]:.3f}, {r["iia_eff_ci"][1]:.3f}]; '
            f'permuted {r["perm_null_mean"]:.3f} [{r["perm_null_interval"][0]:.3f}, {r["perm_null_interval"][1]:.3f}] p {r["p_perm"]:.4f}; fixed-L null {r["fixedL_null"]:.3f}, p_paired {r["p_paired"]:.2e}')
    real = np.array([r['iia_eff'] for r in rows]); ok = np.isfinite(real)
    stage_null = np.nanmean(null[:, ok], 1)
    per[s] = dict(units=rows, perm_matrix=np.where(np.isfinite(M), M, -1).round(4).tolist(),
                  stage_mean_iia_eff=float(np.nanmean(real)), stage_perm_null_mean=float(stage_null.mean()),
                  stage_perm_null_interval=[float(np.quantile(stage_null, .025)), float(np.quantile(stage_null, .975))],
                  stage_p_perm=float((1 + (stage_null >= np.nanmean(real)).sum()) / (1 + A.draws)))
    log(f'stage {s}: mean effective IIA {per[s]["stage_mean_iia_eff"]:.3f}; permuted {per[s]["stage_perm_null_mean"]:.3f} {per[s]["stage_perm_null_interval"]} p {per[s]["stage_p_perm"]:.4f}')
tests = [(s, k) for s in POST for k, r in enumerate(per[s]['units']) if r['n_eff'] >= 20]
for key in ('p_perm', 'p_paired'):
    adj = holm([per[s]['units'][k][key] for s, k in tests])
    for (s, k), a in zip(tests, adj): per[s]['units'][k][key + '_holm'] = float(a)
res['per_unit'] = per; res['n_tested_holm'] = len(tests)
log(f'Holm over {len(tests)} per-unit variables with |E| >= 20: p_perm_holm < 0.05 for {sum(per[s]["units"][k]["p_perm_holm"] < .05 for s, k in tests)}, '
    f'p_paired_holm < 0.05 for {sum(per[s]["units"][k]["p_paired_holm"] < .05 for s, k in tests)}')

# ------------------------------------------------------------------ (2) joint interventions
joint = {}
for s in POST:
    V = R[s][ib].copy(); V[:, UNITS[s]] = R[s][isr][:, UNITS[s]]
    joint[f'all_{s}'] = summ(T({s: V}, ib).argmax(1), H_from(s, CH[s][isr], ib).argmax(1), NB)
VS = R['S'][ib].copy(); VS[:, UNITS['S']] = R['S'][isr][:, UNITS['S']]
joint['all_S_plus_all_z'] = summ(T({'S': VS, 'z': R['z'][isr]}, ib).argmax(1), H_from('z', CH['z'][isr], ib).argmax(1), NB)
joint['all_S_plus_all_z']['note'] = 'all 16 z units set from the source: L equals the network on the source, H the program on the source'
joint['no_intervention'] = summ(NB, HB, NB)
for k, v in joint.items(): log(f'joint {k}: IIA {v["iia"]:.4f} {v["iia_ci"]}; |E| {v["n_eff"]}; effective {v["iia_eff"]:.4f} {v["iia_eff_ci"]}')


def mixed(subsets, rows_b, rows_s, aligned=None):
    """random-subset intervention over several stages; subsets {stage: unit indices (into UNITS)}; returns L, H calls.
    aligned: H's subsets (default the same; a misaligned control passes other subsets of the same sizes)"""
    aligned = aligned or subsets; V = {}
    for s in [s for s in POST if s in subsets]:
        v = R[s][rows_b].copy(); u = np.array(UNITS[s])[subsets[s]]; v[:, u] = R[s][rows_s][:, u]; V[s] = v
    # L: entry at the earliest stage; later stages: the network's own value except the subset (needs a stage-by-stage run)
    lo = L_mixed(V, subsets, rows_b, rows_s)
    # H
    s = 'S'; cur = CH['S'][rows_b].copy()
    if 'S' in aligned: cur[:, aligned['S']] = CH['S'][rows_s][:, aligned['S']]
    while NXT[s] is not None:
        nx = NXT[s]; cur = evaluate(fb[nx]['units'], {f'{s}:{j}': cur[:, i] for i, j in enumerate(UNITS[s])}, RELU[nx]); s = nx
        if s in aligned: cur[:, aligned[s]] = CH[s][rows_s][:, aligned[s]]
    z = np.zeros((len(rows_b), 16)); z[:, UNITS['z']] = cur
    return lo.argmax(1), T({'z': z}, rows_b).argmax(1)


def L_mixed(V, subsets, rows_b, rows_s):
    """network with the subset units of each stage set to the source values and every other unit computed by the network"""
    S = V.get('S', R['S'][rows_b])
    st = [s for s in ('h4', 'h5', 'z') if s in subsets]
    if not st: return T({'S': S}, rows_b)
    # run stage by stage: compute h4 from S with the network, then override the subset, and so on
    vals = {'S': S}
    for s in ('h4', 'h5', 'z'):
        full = forward_stage(s, vals, rows_b)
        if s in subsets:
            u = np.array(UNITS[s])[subsets[s]]; full = full.copy(); full[:, u] = R[s][rows_s][:, u]
        vals[s] = full
    return T({'z': vals['z']}, rows_b)


def _stage_fns(netx):
    import jax, jax.numpy as jnp
    L = netx.L; KB = {}
    for k in ('d4', 'd5', 'd6'):
        K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))

    def mk(k, act):
        def f(v):
            z = jnp.einsum(L[k].equation, L[k].iq(v, training=False), KB[k][0]) + KB[k][1]
            return (L[k].activation(z) if act and L[k].activation is not None else z).reshape(v.shape[0], -1)
        return jax.jit(f)
    return dict(h4=mk('d4', True), h5=mk('d5', True), z=mk('d6', False))


SF = _stage_fns(net); PRV = dict(h4='S', h5='h4', z='h5')


def forward_stage(s, vals, rows):
    v = vals[PRV[s]]; sh = (len(v),) + tuple(d if d != 'W' else v.shape[1] for d in T.shp)
    return np.asarray(SF[s](v.reshape(sh).astype(np.float32)), np.float64)


# check: stage-by-stage with no override reproduces the network
_r = np.arange(min(J, 2000)); _v = {'S': R['S'][_r]}
for _s in ('h4', 'h5', 'z'): _v[_s] = forward_stage(_s, _v, _r)
res['meta']['stagewise_check'] = float(max(np.abs(_v[s] - R[s][_r]).max() for s in ('h4', 'h5', 'z'))); log(f'stage-by-stage check max |diff| {res["meta"]["stagewise_check"]:.2e}')

rs = np.random.default_rng(7); sub = {}
for name, stages, p in (('S_and_z_half', ('S', 'z'), 0.5), ('all_stages_quarter', POST, 0.25)):
    ra, rn = [], []
    for d in range(A.draws):
        S_ = {s: np.flatnonzero(rs.random(len(UNITS[s])) < p) for s in stages}
        S_ = {s: v for s, v in S_.items() if len(v)}
        mis = {s: rs.choice(len(UNITS[s]), len(v), replace=False) for s, v in S_.items()}
        cl, ch = mixed(S_, ib, isr); _, chm = mixed(S_, ib, isr, aligned=mis)
        E = cl != NB
        ra.append([float((cl == ch).mean()), float((cl == ch)[E].mean()) if E.any() else np.nan, int(E.sum())])
        rn.append([float((cl == chm).mean()), float((cl == chm)[E].mean()) if E.any() else np.nan])
    ra, rn = np.array(ra), np.array(rn); dd = ra[:, 1] - rn[:, 1]
    sub[name] = dict(p_unit=p, stages=list(stages), iia_median=float(np.median(ra[:, 0])), iia_interval=np.quantile(ra[:, 0], [.025, .975]).tolist(),
                     iia_eff_mean=float(np.nanmean(ra[:, 1])), iia_eff_interval=np.nanquantile(ra[:, 1], [.025, .975]).tolist(), n_eff_median=float(np.median(ra[:, 2])),
                     misaligned_iia_eff_mean=float(np.nanmean(rn[:, 1])), misaligned_iia_eff_interval=np.nanquantile(rn[:, 1], [.025, .975]).tolist(),
                     frac_draws_aligned_gt_misaligned=float(np.nanmean(dd > 0)), p_sign=float((1 + np.nansum(dd <= 0)) / (1 + np.isfinite(dd).sum())))
    log(f'random subsets {name}: IIA median {sub[name]["iia_median"]:.4f}; effective {sub[name]["iia_eff_mean"]:.3f} {sub[name]["iia_eff_interval"]} '
        f'(|E| median {sub[name]["n_eff_median"]:.0f}); misaligned {sub[name]["misaligned_iia_eff_mean"]:.3f} {sub[name]["misaligned_iia_eff_interval"]}; p {sub[name]["p_sign"]:.4f}')
joint['random_subsets'] = sub; res['joint'] = joint

# ------------------------------------------------------------------ (ii) random rotations
rot = {}
for s in POST:
    U = UNITS[s]; n = len(U); sd = R[s][:, U].std(0); sd = np.where(sd > 0, sd, 1.0); rr = np.random.default_rng(200 + POST.index(s)); vals = []
    for d in range(A.draws):
        r = rr.normal(size=n); r /= np.linalg.norm(r)
        cL = ((R[s][isr][:, U] - R[s][ib][:, U]) / sd) @ r; cH = ((CH[s][isr] - CH[s][ib]) / sd) @ r
        VL = R[s][ib].copy(); VL[:, U] = VL[:, U] + sd * r * cL[:, None]
        VH = CH[s][ib] + sd * r * cH[:, None]
        if RELU[s]: VL = np.maximum(VL, 0); VH = np.maximum(VH, 0)
        cl = T({s: VL}, ib).argmax(1); ch = H_from(s, VH, ib).argmax(1); E = cl != NB
        vals.append([float((cl == ch).mean()), float((cl == ch)[E].mean()) if E.any() else np.nan, int(E.sum())])
    v = np.array(vals); real = np.array([r_['iia_eff'] for r_ in per[s]['units']]); w = np.array([r_['n_eff'] for r_ in per[s]['units']])
    rot[s] = dict(iia_eff_mean=float(np.nanmean(v[:, 1])), iia_eff_interval=np.nanquantile(v[:, 1], [.025, .975]).tolist(), iia_mean=float(v[:, 0].mean()),
                  n_eff_median=float(np.median(v[:, 2])), unit_iia_eff_mean=float(np.nanmean(real)),
                  unit_iia_eff_weighted=float(np.nansum(real * w) / w[np.isfinite(real)].sum()),
                  p_rot=float((1 + (v[:, 1] >= np.nanmean(real)).sum()) / (1 + np.isfinite(v[:, 1]).sum())))
    log(f'rotations {s}: effective IIA {rot[s]["iia_eff_mean"]:.3f} {rot[s]["iia_eff_interval"]} (|E| median {rot[s]["n_eff_median"]:.0f}); unit-aligned mean {rot[s]["unit_iia_eff_mean"]:.3f}; p {rot[s]["p_rot"]:.4f}')
res['rotation'] = rot

# ------------------------------------------------------------------ (iii) probe alignment
xf, yf, Qf = sample(6000, 0); Rf = net.run(xf, keep=POST); del xf
qk = [k for k in fa['S']['keys'] if not k.startswith('H:') and k in Qf and k in Qt]
Xf = np.stack([np.asarray(Qf[k], np.float64) for k in qk], 1); Xt = np.stack([Qt[k] for k in qk], 1); del Qf
mu, sg = Xf.mean(0), Xf.std(0); sg = np.where(sg > 0, sg, 1); Xf = (Xf - mu) / sg; Xt = (Xt - mu) / sg
perm = np.random.default_rng(0).permutation(len(Xf)); nf = 2 * len(Xf) // 3; fi, di = perm[:nf], perm[nf:]
XtX = Xf[fi].T @ Xf[fi]; probe = {}; PV = {}
for s in POST:
    Y = Rf[s][:, UNITS[s]]; ym = Y[fi].mean(0); best = None
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        W = np.linalg.solve(XtX + lam * np.eye(len(qk)), Xf[fi].T @ (Y[fi] - ym))
        pd_ = Xf[di] @ W + ym; err = ((pd_ - Y[di]) ** 2).mean()
        if best is None or err < best[0]: best = (err, lam, W)
    _, lam, W = best; P = Xt @ W + ym
    if RELU[s]: P = np.maximum(P, 0)
    PV[s] = P; tgt = R[s][:, UNITS[s]]
    r2p = 1 - ((P - tgt) ** 2).mean(0) / np.maximum(tgt.var(0), 1e-12); r2f = 1 - ((CH[s] - tgt) ** 2).mean(0) / np.maximum(tgt.var(0), 1e-12)
    rows = []
    U = UNITS[s]
    for i, j in enumerate(U):
        base = CH[s][ib].copy(); base[:, i] = P[ib, i]
        ch = H_unit(s, i, base, P[isr, i]).argmax(1); cl = Lc[s][i]; E = cl != NB
        okf = (Hc[s][i] == cl)[E]; okp = (ch == cl)[E]; r = summ(cl, ch, NB); mc = mcnemar(okf, okp)
        r.update(unit=j, probe_r2_test=float(r2p[i]), formula_r2_test_chain=float(r2f[i]), formula_iia_eff=per[s]['units'][i]['iia_eff'], mcnemar=mc)
        rows.append(r)
        log(f'probe {s}{j}: R² probe {r2p[i]:.3f} / formula chain {r2f[i]:.3f}; effective IIA probe {r["iia_eff"]:.3f} {r["iia_eff_ci"]} vs formula {r["formula_iia_eff"]:.3f}; McNemar {mc}')
    adj = holm([r['mcnemar']['p'] for r in rows]); [r.update(mcnemar_p_holm_stage=float(a)) for r, a in zip(rows, adj)]
    probe[s] = dict(ridge_lambda=lam, n_observables=len(qk), units=rows)
res['probe'] = probe

# ------------------------------------------------------------------ (iv) untrained network vs trained, single-stage (a) fits
ctrl = {}
ut = {**json.loads((OUT / 'e11_control_untrained_P_g_S_z.json').read_text())['stages'], **json.loads((OUT / 'e11_control_untrained_h4_h5.json').read_text())['stages']}
unet = Jedi(64, untrained=True, seed=0); RU = unet.run(xt, keep=POST + ('logits',)); TU, tinfo_u = enable_tail(unet, RU, xt); callU = RU['logits'].argmax(1)
for tag, netR, TT, fits, NBx in (('untrained', RU, TU, {s: (ut[s]['units_by_spread'], ut[s]['units']) for s in POST}, callU[ib]),
                                  ('trained_a', R, T, {s: (UNITS[s], fa[s]['units']) for s in POST}, NB)):
    ctrl[tag] = {}
    for s in POST:
        U, fu = fits[s]; F = evaluate(fu, phys, RELU[s]); rows = []
        for i, j in enumerate(U):
            VL = netR[s][ib].copy(); VL[:, j] = netR[s][isr, j]
            VH = netR[s][ib].copy(); VH[:, U] = F[ib]; VH[:, j] = F[isr, i]
            cl = TT({s: VL}, ib).argmax(1); ch = TT({s: VH}, ib).argmax(1); E = cl != NBx; ag = cl == ch
            VH0 = netR[s][ib].copy(); VH0[:, U] = F[ib]; hb = TT({s: VH0}, ib).argmax(1)
            rows.append(dict(unit=int(j), n_eff=int(E.sum()), iia=float(ag.mean()), iia_eff=float(ag[E].mean()) if E.any() else float('nan'),
                             iia_eff_ci=boot_ci(ag[E]), base_same_call=float((hb == NBx).mean()), test_r2=float(fu[i]['test_r2'])))
        ctrl[tag][s] = rows
        log(f'{tag} {s}: ' + '; '.join(f'{s}{r["unit"]} |E| {r["n_eff"]} eff {r["iia_eff"]:.3f}' for r in rows[:20]))
ctrl['untrained_meta'] = dict(tail=tinfo_u, acc=float((callU == yt).mean()), call_distribution=np.bincount(callU, minlength=5).tolist())
res['controls'] = ctrl

p = OUT / f'{A.prefix}_iia.json'; p.write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {p}')


# ------------------------------------------------------------------ figure
def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = dict(S='#1c5cab', h4='#7aa6d9', h5='#e39b3b', z='#b23a48')
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for s in POST:
        rr_ = [r for r in per[s]['units'] if r['n_eff'] >= 20]
        x = [r['cost'] for r in rr_]; yv = [r['iia_eff'] for r in rr_]
        lo_ = [r['iia_eff'] - r['iia_eff_ci'][0] for r in rr_]; hi_ = [r['iia_eff_ci'][1] - r['iia_eff'] for r in rr_]
        ax[0].errorbar(x, yv, yerr=[lo_, hi_], fmt='o', ms=4, color=col[s], ecolor=col[s], elinewidth=0.7, alpha=0.85, label=f'{s} units')
        for r in rr_:
            if r['cost'] >= 10 or (s, r['unit']) in (('S', 0), ('S', 1), ('S', 29), ('z', 0), ('z', 7)):
                ax[0].annotate(f'{s}{r["unit"]}', (r['cost'], r['iia_eff']), fontsize=7.5, xytext=(3, 3), textcoords='offset points')
    pn = np.mean([per[s]['stage_perm_null_mean'] for s in POST])
    ax[0].axhline(pn, color='#888888', ls='--', lw=1); ax[0].text(0.2, pn + 0.015, 'permuted alignment (mean of stages)', fontsize=8, color='#555555')
    ax[0].axhline(0.8, color='#bbbbbb', ls=':', lw=1)
    ax[0].set_xscale('log'); ax[0].set_xlabel('E1 cost of the unit: % of calls changed when set to its mean'); ax[0].set_ylabel('effective IIA (pairs where the swap changes the call)')
    ax[0].set_ylim(-0.02, 1.02); ax[0].legend(fontsize=8, loc='lower right', frameon=False); ax[0].set_title('Per-unit interchange: program call vs network call', fontsize=10, loc='left')
    labels = ['unit-aligned', 'permuted', 'rotated dir.', 'probe-aligned', 'trained (a)', 'untrained (a)']
    w = 0.13
    for k, s in enumerate(POST):
        v = [per[s]['stage_mean_iia_eff'], per[s]['stage_perm_null_mean'], rot[s]['iia_eff_mean'],
             float(np.nanmean([r['iia_eff'] for r in probe[s]['units']])),
             float(np.nanmean([r['iia_eff'] for r in ctrl['trained_a'][s] if r['n_eff'] >= 20] or [np.nan])),
             float(np.nanmean([r['iia_eff'] for r in ctrl['untrained'][s] if r['n_eff'] >= 20] or [np.nan]))]
        for m, vv in enumerate(v):
            ax[1].bar(k + (m - 2.5) * w, vv, w, color=plt.cm.Blues(0.3 + 0.12 * m) if m else '#1c5cab', label=labels[m] if k == 0 else None)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(POST); ax[1].set_ylim(0, 1); ax[1].set_ylabel('mean effective IIA over the stage\'s live units')
    ax[1].legend(fontsize=7.5, frameon=False, ncol=2, loc='upper left'); ax[1].set_title('Alignments and controls per stage', fontsize=10, loc='left')
    for a_ in ax:
        for sp in ('top', 'right'): a_.spines[sp].set_visible(False)
    fig.suptitle(f'E22: does the E11 program predict the network\'s call after swapping one unit from another jet? ({A.pairs} jet pairs)', fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); q = OUT / f'{A.prefix}_iia.png'; fig.savefig(q, dpi=150); log(f'wrote {q}')


try:
    plot()
except Exception as e:  # noqa: BLE001
    log(f'plot failed: {e!r}')
log('done')
