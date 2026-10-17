"""E22v: skeptic rerun of E22 (interchange-intervention accuracy of the E11 (b) program) on a fresh sample.

Sample: sample(2000, SEED) minus the jets of the E11 fitting sample sample(6000, 0) and of the E22 sample sample(2000, 11); 5000 random ordered (base, source)
pairs (pair seed 1).  L = the network; H = the E11 (b) program (same construction as interp/e22_iia_jedi.py, re-implemented
here so that the module does not run E22 on import).
Arms, per live unit k of stage s (S, h4, h5, z), all scored on E(k) = pairs where L's swap changes the network's base call:
  prog      H: formula k <- chain value on the source, downstream formulas recomputed (E22's main arm)
  fnet      formula values at stage s (unit k from the source), the NETWORK's layers downstream   [(b) chain formulas]
  fnet_a    as fnet with the single-stage (a) physics fits (E22's "trained_a" arm)
  nform     the NETWORK's stage-s vector after L's swap, the program's FORMULAS downstream
  perm      permuted alignment (200 derangements), as E22
Trivial predictors of L's new call (no information about which unit is swapped):
  src       the network's call on the source jet
  ru_net    the runner-up class of the network's logits on the base jet
  ru_prog   the runner-up class of the program's logits on the base jet
Also: effective IIA restricted to bases where the program's base call equals the network's; random-subset joint swaps (as E22).
Tests: McNemar (exact) of prog vs each trivial predictor on E(k), Holm over units with |E| >= 20; stage means with a 95 %
bootstrap over pairs (500 resamples).  Output: $AI_RESULTS/jedi/e22_verify.json
Usage: python -m interp.e22_verify [--seed 91] [--pairs 5000] [--draws 200]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample
from interp.e11_mars_all import handbuilt, indices, evaluate, LIVE, UNITS, RELU

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'
POST = ('S', 'h4', 'h5', 'z'); NXT = dict(S='h4', h4='h5', h5='z', z=None); WID = dict(S=64, h4=64, h5=32, z=16)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=91); ap.add_argument('--pairs', type=int, default=5000)
ap.add_argument('--draws', type=int, default=200); ap.add_argument('--boot', type=int, default=500); ap.add_argument('--subset_draws', type=int, default=100)
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


# ------------------------------------------------------------------ data, network, tail
xt, yt, Qt = sample(2000, A.seed); i0, i1 = np.concatenate([indices(6000, 0), indices(2000, 11)]), indices(2000, A.seed); keep = ~np.isin(i1, i0)
xt, yt = xt[keep], yt[keep]; Qt = {k: np.asarray(v[keep], np.float64) for k, v in Qt.items()}; J = len(yt)
phys = {**Qt, **handbuilt(xt)}; del Qt
fb = {'P': json.loads((OUT / 'e11_fit_a_P.json').read_text())}
for s in ('g', 'S', 'h4', 'h5', 'z'): fb[s] = json.loads((OUT / f'e11_fit_b_{s}.json').read_text())
fa = {s: json.loads((OUT / f'e11_fit_a_{s}.json').read_text()) for s in POST}
net = Jedi(64)
R = net.run(xt, keep=POST + ('logits',)); call = R['logits'].argmax(1); acc = float((call == yt).mean())
log(f'seed {A.seed}: jets {J} (overlap with sample(6000, 0) and sample(2000, 11) removed {int((~keep).sum())}); network accuracy {acc:.4f}')


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


SF, D7 = make_stage_fns(); PRV = dict(h4='S', h5='h4', z='h5')


def tail(stage, V):
    """network logits from a full stage-s array V (J, W): the network's own layers after s"""
    cur = np.asarray(V, np.float32); s = stage
    while NXT[s] is not None:
        nx = NXT[s]; cur = np.asarray(SF[nx](cur), np.float32); s = nx
    return np.asarray(D7(cur), np.float64)


# exact check against the network and run_jetwise
chk = {s: float(np.abs(tail(s, R[s]) - R['logits']).max()) for s in POST}
rr = np.random.default_rng(5).permutation(J)[:1000]; j0 = int(np.argmax(R['S'].std(0)))
Sv = R['S'][rr].copy(); Sv[:, j0] = R['S'][rr[::-1], j0]; m = np.zeros(64); m[j0] = 1
chk['S_swap_vs_run_jetwise'] = float(np.abs(tail('S', Sv) - net.run_jetwise(xt[rr], {'S': (m, Sv)})['logits']).max())
log('tail check max |diff|', chk)
assert max(chk.values()) < 1e-5, chk

CH = {}; prev = None
for s in ('P', 'g', 'S', 'h4', 'h5', 'z'):
    d = dict(phys) if s in ('P', 'S') else {}
    if prev is not None: d.update({f'{prev}:{j}': CH[prev][:, i] for i, j in enumerate(UNITS[prev])})
    CH[s] = evaluate(fb[s]['units'], d, RELU[s]); prev = s
FA = {s: evaluate(fa[s]['units'], phys, RELU[s]) for s in POST}


def H_from(stage, V):
    """program logits from live-unit values V (J, n_live) of stage s; downstream formulas; d7 on relu(z)"""
    s = stage; cur = V
    while NXT[s] is not None:
        nx = NXT[s]; cur = evaluate(fb[nx]['units'], {f'{s}:{j}': cur[:, i] for i, j in enumerate(UNITS[s])}, RELU[nx]); s = nx
    z = np.zeros((len(cur), 16)); z[:, UNITS['z']] = cur
    return np.asarray(D7(z.astype(np.float32)), np.float64)


rng = np.random.default_rng(1); ib = rng.integers(0, J, A.pairs); isr = rng.integers(0, J, A.pairs)
while (bad := ib == isr).any(): isr[bad] = rng.integers(0, J, int(bad.sum()))
NB = call[ib]; HBlog = H_from('S', CH['S'][ib]); HB = HBlog.argmax(1)
HB_all = H_from('S', CH['S']).argmax(1)
ru_net = np.argsort(R['logits'][ib], 1)[:, -2]; ru_prog = np.argsort(HBlog, 1)[:, -2]; src = call[isr]
base_agree = HB == NB
res = dict(meta=dict(seed=A.seed, jets=J, overlap_removed=int((~keep).sum()), pairs=A.pairs, pair_seed=1, draws=A.draws, boot=A.boot,
                     network_acc=acc, program_acc=float((HB_all == yt).mean()), program_same_call_all_jets=float((HB_all == call).mean()),
                     program_same_call_bases=float(base_agree.mean()), tail_check=chk))
log(f'program acc {res["meta"]["program_acc"]:.4f}; same call all jets {res["meta"]["program_same_call_all_jets"]:.4f}; bases {base_agree.mean():.4f}')

# bootstrap weights over pairs (resample counts), for stage means and per-unit CIs
RB = np.random.default_rng(2); W = np.stack([np.bincount(RB.integers(0, A.pairs, A.pairs), minlength=A.pairs) for _ in range(A.boot)]).astype(np.float32)


def eff(agree, E):
    """effective IIA, its bootstrap CI, and the bootstrap draws (for stage means)"""
    if E.sum() == 0: return float('nan'), [float('nan')] * 2, np.full(A.boot, np.nan)
    num = W @ (agree & E).astype(np.float32); den = W @ E.astype(np.float32); b = num / np.maximum(den, 1)
    return float(agree[E].mean()), [float(np.quantile(b, .025)), float(np.quantile(b, .975))], b


# ------------------------------------------------------------------ per-unit arms
per = {}; boots = {}
for s in POST:
    U = UNITS[s]; n = len(U); Lc = np.zeros((n, A.pairs), int); arms = {k: np.zeros((n, A.pairs), int) for k in ('prog', 'fnet', 'fnet_a', 'nform')}
    for i, j in enumerate(U):
        VL = R[s][ib].copy(); VL[:, j] = R[s][isr, j]; Lc[i] = tail(s, VL).argmax(1)
        V = CH[s][ib].copy(); V[:, i] = CH[s][isr, i]; arms['prog'][i] = H_from(s, V).argmax(1)
        VF = R[s][ib].copy(); VF[:, U] = CH[s][ib]; VF[:, j] = CH[s][isr, i]; arms['fnet'][i] = tail(s, VF).argmax(1)
        VA = R[s][ib].copy(); VA[:, U] = FA[s][ib]; VA[:, j] = FA[s][isr, i]; arms['fnet_a'][i] = tail(s, VA).argmax(1)
        arms['nform'][i] = H_from(s, VL[:, U]).argmax(1) if s != 'z' else Lc[i]
    E = Lc != NB[None]
    # permuted alignment, as E22
    M = np.full((n, n), np.nan)
    for j in range(n):
        if E[j].any(): M[:, j] = (arms['prog'][:, E[j]] == Lc[j, E[j]][None]).mean(1)
    rngp = np.random.default_rng(300 + POST.index(s)); null = np.full((A.draws, n), np.nan)
    for d in range(A.draws):
        while True:
            pi = rngp.permutation(n)
            if not (pi == np.arange(n)).any(): break
        null[d] = M[np.arange(n), pi]
    rows = []; bs = {k: [] for k in ('prog', 'fnet', 'fnet_a', 'nform', 'src', 'ru_net', 'ru_prog', 'prog_base_agree')}
    for i, j in enumerate(U):
        e = E[i]; r = dict(unit=j, cost=LIVE[s][j], n_eff=int(e.sum()), n_eff_base_agree=int((e & base_agree).sum()))
        preds = {**{k: arms[k][i] for k in arms}, 'src': src, 'ru_net': ru_net, 'ru_prog': ru_prog}
        for k, pr in preds.items():
            v, ci, b = eff(pr == Lc[i], e); r[k] = v; r[k + '_ci'] = ci; bs[k].append(b)
        v, ci, b = eff(arms['prog'][i] == Lc[i], e & base_agree); r['prog_base_agree'] = v; r['prog_base_agree_ci'] = ci; bs['prog_base_agree'].append(b)
        r['iia_all'] = float((arms['prog'][i] == Lc[i]).mean())
        r['prog_changed_on_E'] = float((arms['prog'][i] != HB)[e].mean()) if e.any() else float('nan')
        nd = null[:, i]; nd = nd[np.isfinite(nd)]
        r['perm_mean'] = float(nd.mean()) if len(nd) else float('nan')
        r['perm_interval'] = [float(np.quantile(nd, .025)), float(np.quantile(nd, .975))] if len(nd) else [float('nan')] * 2
        okp = (arms['prog'][i] == Lc[i])[e]
        for k in ('src', 'ru_net', 'ru_prog', 'fnet', 'nform'):
            r['mcnemar_prog_vs_' + k] = mcnemar(okp, (preds[k] == Lc[i])[e])
        rows.append(r)
        log(f'{s}{j} |E| {r["n_eff"]}: prog {r["prog"]:.3f} {np.round(r["prog_ci"], 3).tolist()} | fnet {r["fnet"]:.3f} fnet_a {r["fnet_a"]:.3f} nform {r["nform"]:.3f} | '
            f'src {r["src"]:.3f} ru_net {r["ru_net"]:.3f} ru_prog {r["ru_prog"]:.3f} | perm {r["perm_mean"]:.3f} | base-agree {r["prog_base_agree"]:.3f}')
    ok = np.array([r['n_eff'] >= 20 for r in rows])
    st = {}
    for k, lst in bs.items():
        vals = np.array([r[k] for r in rows])[ok]; bb = np.stack(lst)[ok]
        st[k] = dict(mean=float(np.nanmean(vals)), ci=np.nanquantile(np.nanmean(bb, 0), [.025, .975]).tolist())
    st['perm'] = dict(mean=float(np.nanmean(null[:, ok])), interval=np.nanquantile(np.nanmean(null[:, ok], 1), [.025, .975]).tolist())
    # stage-level paired bootstrap of prog - each trivial predictor and of fnet - prog
    for k in ('src', 'ru_net', 'ru_prog', 'fnet', 'nform'):
        dd = np.nanmean(np.stack(bs['prog'])[ok] - np.stack(bs[k])[ok], 0)
        st[f'prog_minus_{k}'] = dict(mean=float(st['prog']['mean'] - st[k]['mean']), ci=np.nanquantile(dd, [.025, .975]).tolist(),
                                     p_two_sided=float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()) + 1 / A.boot)))
    per[s] = dict(units=rows, stage=st, n_tested=int(ok.sum()))
    log(f'stage {s}: ' + '; '.join(f'{k} {v["mean"]:.3f}' for k, v in st.items() if 'mean' in v and not k.startswith('prog_minus')))
    log(f'stage {s} prog minus: ' + '; '.join(f'{k} {v["mean"]:+.3f} {np.round(v["ci"], 3).tolist()}' for k, v in st.items() if k.startswith('prog_minus')))

tests = [(s, k) for s in POST for k, r in enumerate(per[s]['units']) if r['n_eff'] >= 20]
for b in ('src', 'ru_net', 'ru_prog', 'fnet', 'nform'):
    adj = holm([per[s]['units'][k]['mcnemar_prog_vs_' + b]['p'] for s, k in tests])
    for (s, k), a in zip(tests, adj): per[s]['units'][k]['mcnemar_prog_vs_' + b]['p_holm'] = float(a)
    hi = sum(a < .05 and per[s]['units'][k]['mcnemar_prog_vs_' + b]['n10'] > per[s]['units'][k]['mcnemar_prog_vs_' + b]['n01'] for (s, k), a in zip(tests, adj))
    lo = sum(a < .05 and per[s]['units'][k]['mcnemar_prog_vs_' + b]['n10'] < per[s]['units'][k]['mcnemar_prog_vs_' + b]['n01'] for (s, k), a in zip(tests, adj))
    res.setdefault('holm_counts', {})[b] = dict(n_tested=len(tests), prog_higher=int(hi), prog_lower=int(lo))
    log(f'Holm over {len(tests)}: prog vs {b}: prog higher {hi}, prog lower {lo}')
res['per_unit'] = per

# ------------------------------------------------------------------ random-subset joint swaps (as E22), with trivial predictors
rs = np.random.default_rng(17); sub = {}


def mixed(subsets, aligned):
    VL = {'S': R['S'][ib].copy()}
    if 'S' in subsets: u = np.array(UNITS['S'])[subsets['S']]; VL['S'][:, u] = R['S'][isr][:, u]
    cur = VL['S'].astype(np.float32)
    for s in ('h4', 'h5', 'z'):
        cur = np.asarray(SF[s](cur), np.float64)
        if s in subsets: u = np.array(UNITS[s])[subsets[s]]; cur[:, u] = R[s][isr][:, u]
        cur = cur.astype(np.float32)
    cl = np.asarray(D7(cur), np.float64).argmax(1)
    s = 'S'; c = CH['S'][ib].copy()
    if 'S' in aligned: c[:, aligned['S']] = CH['S'][isr][:, aligned['S']]
    while NXT[s] is not None:
        nx = NXT[s]; c = evaluate(fb[nx]['units'], {f'{s}:{j}': c[:, i] for i, j in enumerate(UNITS[s])}, RELU[nx]); s = nx
        if s in aligned: c[:, aligned[s]] = CH[s][isr][:, aligned[s]]
    z = np.zeros((A.pairs, 16)); z[:, UNITS['z']] = c
    return cl, np.asarray(D7(z.astype(np.float32)), np.float64).argmax(1)


for name, stages, p in (('S_and_z_half', ('S', 'z'), 0.5), ('all_stages_quarter', POST, 0.25)):
    out = []
    for d in range(A.subset_draws):
        S_ = {s: np.flatnonzero(rs.random(len(UNITS[s])) < p) for s in stages}; S_ = {s: v for s, v in S_.items() if len(v)}
        mis = {s: rs.choice(len(UNITS[s]), len(v), replace=False) for s, v in S_.items()}
        cl, ch = mixed(S_, S_); _, chm = mixed(S_, mis); E = cl != NB
        if not E.any(): continue
        out.append([(cl == ch)[E].mean(), (cl == chm)[E].mean(), (cl == src)[E].mean(), (cl == ru_net)[E].mean(), E.sum()])
    o = np.array(out)
    sub[name] = dict(draws=len(o), prog=float(o[:, 0].mean()), prog_interval=np.quantile(o[:, 0], [.025, .975]).tolist(),
                     misaligned=float(o[:, 1].mean()), src=float(o[:, 2].mean()), ru_net=float(o[:, 3].mean()), n_eff_median=float(np.median(o[:, 4])),
                     frac_prog_gt_src=float((o[:, 0] > o[:, 2]).mean()), frac_prog_gt_misaligned=float((o[:, 0] > o[:, 1]).mean()))
    log(f'subsets {name}: ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sub[name].items()}))
res['random_subsets'] = sub

p = OUT / 'e22_verify.json'; p.write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {p}')
log('done')
