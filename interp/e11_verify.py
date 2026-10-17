"""E11 verification (skeptic, 2026-10-16): the nested hinge-formula program of interp/e11_mars_all.py, re-evaluated from the
STORED formulas (results/jedi/e11_fit_*.json, e11hp_fit_*.json); nothing is refitted, no network is trained.

Modes (python -m interp.e11_verify <mode>; writes results/jedi/e11_verify_<mode>.json):
  core   held-out test jets of E11 (sample(2000, seed 11) minus the seed-0 fitting sample):
         1 leakage / exactness: index disjointness, content hashes, term / if-statement count, last layer on the program's z
           four ways (E11's all-stage override; z-only override; jax d7 alone; float64 round/wrap + K·h + b), wrap events
         3 disagreements: classes, per-stage R² / normalised RMSE on agreeing vs disagreeing jets, first diverging stage
           (cumulative (b) P..k), chain restarted from the network's true values at stage k
         4 single (b) at h4: reproduction, per-class shift, transitions, unit attribution (single unit, leave-one-out),
           S-input mismatch (formula fitted on chain S^, applied to true S)
  fresh  fresh sample(4000, seed 31) minus every jet of seeds 0 and 11: same call, accuracy, per class, rejection at 50 %
         signal efficiency (one-vs-rest P_k; W/Z/t vs g+q with P_k/(P_k+P_g+P_q); g vs q), paired bootstrap; (b) chain,
         (a) z-on-physics, half-plane (e11hp) versions; checks 3 and 4 (aggregate) repeated
  edit   1,000 jets per class of the fresh sample (rng 5): translation +0.06 rad along η (padding fixed, as E6v) and
         pT × 0.8; the physics inputs recomputed from the edited particles (jetdistill.observables.compute + handbuilt);
         class flows of the network vs the programs
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import e11_mars_all as E
from interp.jedi_core import Jedi
from jetdistill import network as NW

OUT = WOUT = E.OUT; STG, UNITS, LIVE, RELU, PREV, CL = E.STG, E.UNITS, E.LIVE, E.RELU, E.PREV, E.CL
T0 = time.time()
LIMIT = 0          # --limit k: smoke test of the code on k random jets per set (not a result)


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def rnd(v, k=4):
    if isinstance(v, dict): return {kk: rnd(vv, k) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)): return [rnd(vv, k) for vv in v]
    if isinstance(v, (float, np.floating)): return float(round(float(v), k)) if np.isfinite(v) else (None if np.isnan(v) else float(v))
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.ndarray): return rnd(v.tolist(), k)
    return v


# ------------------------------------------------------------------ data and formulas
def load(sets, need_q):
    """{name: sorted row indices} -> {name: dict(idx, x, y, Q)} from the test file, each member read once"""
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz')
    y = np.asarray(d['y']); X = d['x']
    if LIMIT: sets = {n: (ix if n == 'fitdev' else np.sort(np.random.default_rng(9).choice(ix, min(LIMIT, len(ix)), replace=False))) for n, ix in sets.items()}
    out = {n: dict(idx=ix, x=np.asarray(X[ix], np.float32), y=y[ix].astype(int)) for n, ix in sets.items()}
    del X
    for n in need_q: out[n]['Q'] = {}
    if need_q:
        for k in d.files:
            if not k.startswith('Q_'): continue
            a = d[k]
            for n in need_q: out[n]['Q'][k[2:]] = np.asarray(a[out[n]['idx']], np.float64)
    desc = ', '.join(f'{n} {len(v["y"])}' for n, v in out.items()); log(f'loaded {desc}; test file {len(y)} jets')
    return out, y


def formulas(prefix):
    rd = lambda n: json.loads((OUT / f'{prefix}_{n}.json').read_text())
    fa = {s: rd(f'fit_a_{s}') for s in STG}; fb = {'P': fa['P'], **{s: rd(f'fit_b_{s}') for s in STG[1:]}}
    for s in STG:
        assert [u['unit'] for u in fa[s]['units']] == UNITS[s] and [u['unit'] for u in fb[s]['units']] == UNITS[s], s
    return fa, fb


def count_terms(fb):
    n_t = n_if = n_prod = 0; inputs = set(); phys_inputs = set()
    for s in STG:
        for u in fb[s]['units']:
            for t in u['terms']:
                n_t += 1; n_if += (t['kind'] != 'lin') + (bool(t.get('q2')) and t.get('kind2') != 'lin'); n_prod += bool(t.get('q2'))
                for q in (t['q'], t.get('q2')):
                    if q: inputs.add(q); (phys_inputs.add(q) if E.family(q) is not None else None)
    return dict(units=sum(len(UNITS[s]) for s in STG), terms=n_t, if_statements=int(n_if), products=n_prod, distinct_inputs=len(inputs),
                distinct_physics_inputs=len(phys_inputs))


def physics(x, Q, hp):
    return {**Q, **E.handbuilt(x, hp)}


def chain(fb, phys, start=None, true=None):
    """the (b) program: P <- physics, g <- P^, S <- g^ + physics, h4 <- S^, h5 <- h4^, z <- h5^ (live units only).
    start=k: stages up to k take the network's TRUE live values (true[s]), the chain continues from there"""
    V = {}
    for i, s in enumerate(STG):
        if start is not None and i <= STG.index(start): V[s] = true[s][:, UNITS[s]]; continue
        V[s] = E.evaluate(fb[s]['units'], E.chain_inputs(s, None if s == 'P' else V[PREV[s]], phys), RELU[s])
    return V


class Last:
    """the network's last layer d7 on z: ReLU, the layer's input quantizer (fixed point), K·h + b"""

    def __init__(self, net):
        import jax, jax.numpy as jnp
        L7 = net.L['d7']; K, b = L7.get_fused_qkernel_and_qbias(False, L7.moving_mean, L7.moving_variance)
        K, b = jnp.asarray(K), jnp.asarray(b); act = L7.activation
        self.jnp = jnp
        self.f = jax.jit(lambda z: (lambda o: act(o) if act is not None else o)(jnp.einsum(L7.equation, L7.iq(jax.nn.relu(z), training=False), K) + b))
        q = L7.iq.quantizer; self.k, self.i, self.fr = (np.asarray(getattr(q, a)).ravel() for a in ('k', 'i', 'f'))
        self.K64 = np.asarray(net.KB['d7'][0], np.float64).reshape(-1, 5); self.b64 = np.asarray(net.KB['d7'][1], np.float64).ravel()
        self.info = dict(equation=L7.equation, activation=str(act), k_bits=self.k.tolist(), i_bits=self.i.astype(int).tolist(), f_bits=self.fr.astype(int).tolist(),
                         round_mode=str(getattr(q, 'round_mode', None)), overflow_mode=str(getattr(q, 'overflow_mode', None)))
        try:
            self.info['iq_config'] = str(L7.iq.get_config())[:800]
        except Exception as e:  # noqa: BLE001
            self.info['iq_config'] = repr(e)
        self.shape3 = False

    def jax_logits(self, z):
        z = np.asarray(z, np.float32)
        try:
            o = self.f(self.jnp.asarray(z[:, None, :] if self.shape3 else z))
        except Exception:  # noqa: BLE001
            self.shape3 = True; o = self.f(self.jnp.asarray(z[:, None, :]))
        return np.asarray(o, np.float64).reshape(len(z), -1)

    def np_logits(self, z):
        return NW.logits_from_h(np.maximum(np.asarray(z, np.float64), 0), self.K64, self.b64, self.i, self.fr)

    def wraps(self, z):
        """jets with any neuron at or above the unsigned fixed-point range 2^i (it would wrap)"""
        sc = 2.0 ** self.fr; return (np.floor(np.maximum(np.asarray(z, np.float64), 0) * sc + 0.5) / sc >= 2.0 ** self.i).any(1)


WIDTH = {'P': 64, 'g': 64, 'S': 64, 'h4': 64, 'h5': 32, 'z': 16}


def masks():
    return {s: np.isin(np.arange(WIDTH[s]), UNITS[s]).astype(np.float32) for s in STG}


MASK = None


def sub(net, x, N, O):
    """E11's stage substitution: {stage: live-unit values (J, n_live)}; non-live units keep the network's values"""
    ov = {}
    for s, F in O.items():
        v = N[s].copy(); v[:, UNITS[s]] = F; ov[s] = (MASK[s], v)
    return net.run_jetwise(x, ov)['logits']


def paired(c, cn, y):
    b = int(((c == y) & (cn != y)).sum()); w = int(((c != y) & (cn == y)).sum()); J = len(y)
    return dict(program_right_network_wrong=b, program_wrong_network_right=w, acc_diff=(b - w) / J, se=float(np.sqrt(max(b + w - (b - w) ** 2 / J, 0)) / J))


def callstats(lo, y, cn):
    c = lo.argmax(1)
    return dict(same_call=float((c == cn).mean()), acc=float((c == y).mean()), per_class_acc=[float((c[y == k] == k).mean()) for k in range(5)],
                paired_vs_network=paired(c, cn, y), confusion=[[int(((y == a) & (c == b)).sum()) for b in range(5)] for a in range(5)])


def softmax(L):
    e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def rej50(score, sig, bkg):
    thr = np.quantile(score[sig], 0.5); fpr = float((score[bkg] >= thr).mean()); eff = float((score[sig] >= thr).mean())
    return (1 / fpr if fpr > 0 else np.inf), eff


def tasks(P, y):
    qcd = P[:, 0] + P[:, 1]; T = {}
    for k in range(5): T[f'{CL[k]} vs rest'] = (P[:, k], y == k, y != k)
    for k in (2, 3, 4): T[f'{CL[k]} vs QCD(g+q)'] = (P[:, k] / np.maximum(P[:, k] + qcd, 1e-300), y == k, y <= 1)
    T['g vs q'] = (P[:, 0] / np.maximum(qcd, 1e-300), y == 0, y == 1); T['q vs g'] = (P[:, 1] / np.maximum(qcd, 1e-300), y == 1, y == 0)
    return T


def rejections(lo_p, lo_n, y, B=300, seed=7):
    Pp, Pn = softmax(lo_p), softmax(lo_n); Tp, Tn = tasks(Pp, y), tasks(Pn, y); out = {}
    rng = np.random.default_rng(seed); boots = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    for name in Tp:
        rp, ep = rej50(*Tp[name]); rn, en = rej50(*Tn[name]); lr = []
        for ix in boots:
            sp, sg, bg = (a[ix] for a in Tp[name]); sn = Tn[name][0][ix]
            a_, _ = rej50(sp, sg, bg); b_, _ = rej50(sn, sg, bg)
            if np.isfinite(a_) and np.isfinite(b_): lr.append(np.log(a_ / b_))
        lr = np.array(lr); pc = lambda q: np.exp(np.percentile(lr, q)).tolist() if len(lr) else [np.nan, np.nan]
        out[name] = dict(program=rp, network=rn, ratio=rp / rn if np.isfinite(rp) and np.isfinite(rn) else np.nan, eff_program=ep, eff_network=en,
                         ratio_boot_16_84=pc([16, 84]), ratio_boot_2p5_97p5=pc([2.5, 97.5]), n_boot_finite=int(len(lr)), n_sig=int(Tp[name][1].sum()), n_bkg=int(Tp[name][2].sum()))
    return out


# ------------------------------------------------------------------ check 3
def r2sub(F, T, m):
    h = np.maximum(T[m], 0); p = np.maximum(F[m], 0); return float(1 - ((h - p) ** 2).mean() / max(h.var(), 1e-12))


def per_stage(V, N, ag, dg):
    rows = {}
    for s in STG:
        T = N[s][:, UNITS[s]]; F = V[s]; cost = np.array([LIVE[s][j] for j in UNITS[s]]); cw = cost / cost.sum()
        sd = np.maximum(np.maximum(T, 0).std(0), 1e-12); err = (np.maximum(F, 0) - np.maximum(T, 0)) / sd
        r = {}
        for name, m in (('agree', ag), ('disagree', dg)):
            r2 = np.array([r2sub(F[:, i], T[:, i], m) for i in range(T.shape[1])]); nr = np.sqrt((err[m] ** 2).mean(0))
            r[name] = dict(r2_median=float(np.median(r2)), r2_cost_weighted=float(cw @ r2), nrmse_median=float(np.median(nr)), nrmse_cost_weighted=float(cw @ nr))
        r['nrmse_ratio_disagree_over_agree'] = r['disagree']['nrmse_cost_weighted'] / max(r['agree']['nrmse_cost_weighted'], 1e-12)
        nr_d = np.sqrt((err[dg] ** 2).mean(0)); nr_a = np.sqrt((err[ag] ** 2).mean(0)); ratio = nr_d / np.maximum(nr_a, 1e-12)
        top = np.argsort(-(ratio * cw))[:5]
        r['units_largest_ratio_x_cost'] = [dict(unit=f'{s}:{UNITS[s][i]}', cost=float(cost[i]), nrmse_agree=float(nr_a[i]), nrmse_disagree=float(nr_d[i])) for i in top]
        rows[s] = r
    return rows


def disagreement(net, x, y, N, cn, V, fb, phys, last, cumul=True):
    """check 3 on one jet set: who, where the chain diverges, which restart fixes the call"""
    lo = last.jax_logits(V['z']); c = lo.argmax(1); dg = c != cn; ag = ~dg; res = dict(n=int(len(y)), n_disagree=int(dg.sum()), frac=float(dg.mean()))
    res['by_true_class'] = {CL[k]: dict(n=int((dg & (y == k)).sum()), frac_of_class=float(dg[y == k].mean())) for k in range(5)}
    res['transition_network_to_program'] = [[int((dg & (cn == a) & (c == b)).sum()) for b in range(5)] for a in range(5)]
    res['on_disagreeing'] = dict(network_right=int((cn[dg] == y[dg]).sum()), program_right=int((c[dg] == y[dg]).sum()), neither=int(((cn[dg] != y[dg]) & (c[dg] != y[dg])).sum()))
    P = softmax(N['logits']); srt = np.sort(P, 1); marg = srt[:, -1] - srt[:, -2]
    res['network_top2_margin_median'] = dict(agree=float(np.median(marg[ag])), disagree=float(np.median(marg[dg])))
    res['per_stage'] = per_stage(V, N, ag, dg)
    # restart: stages up to k take the network's true values, formulas from k+1 on
    rs = {}
    for k in STG[:-1]:
        cr = last.jax_logits(chain(fb, phys, start=k, true=N)['z']).argmax(1)
        rs[k] = dict(same_call=float((cr == cn).mean()), fixed_of_disagreeing=float((cr[dg] == cn[dg]).mean()), broken_of_agreeing=float((cr[ag] != cn[ag]).mean()))
    res['restart_from_true'] = rs
    if cumul:   # E11's cumulative (b) P..k (non-live units from the network): first k where the call leaves the network's
        calls = {}
        for i, s in enumerate(STG): calls[s] = sub(net, x, N, {k: V[k] for k in STG[:i + 1]}).argmax(1)
        first = np.full(len(y), -1)
        for i, s in enumerate(STG):
            first[(first < 0) & (calls[s] != cn)] = i
        res['cumulative_same_call'] = {s: float((calls[s] == cn).mean()) for s in STG}
        res['first_diverging_stage_of_disagreeing'] = {s: int((first[dg] == i).sum()) for i, s in enumerate(STG)}
        res['cumulative_equals_final_program_call_on_disagreeing'] = {s: float((calls[s][dg] == c[dg]).mean()) for s in STG}
    return res


# ------------------------------------------------------------------ check 4
def h4_single(net, x, y, N, cn, V, fb, phys, detail=True):
    S_true = N['S'][:, UNITS['S']]; Vs = V['S']
    fS = lambda Sin: E.evaluate(fb['h4']['units'], E.chain_inputs('h4', Sin, phys), True)
    F1 = fS(S_true); T = N['h4'][:, UNITS['h4']]; sd = np.maximum(T.std(0), 1e-12)
    st = lambda lo: callstats(lo, y, cn)
    res = dict(single_b_true_S=st(sub(net, x, N, {'h4': F1})), chain_values_only_at_h4=st(sub(net, x, N, {'h4': V['h4']})),
               cumulative_b_h4=st(sub(net, x, N, {k: V[k] for k in STG[:4]})))
    c1 = sub(net, x, N, {'h4': F1}).argmax(1)
    res['single_b_transitions_network_to_sub'] = [[int(((cn == a) & (c1 == b)).sum()) for b in range(5)] for a in range(5)]
    res['net_per_class_acc'] = [float((cn[y == k] == k).mean()) for k in range(5)]
    # bias of the h4 formula outputs (spread units), true S vs chain S^ as input
    cost = np.array([LIVE['h4'][j] for j in UNITS['h4']]); cw = cost / cost.sum()
    bias = lambda F, m: (F[m] - T[m]).mean(0) / sd
    res['h4_bias_cost_weighted_abs'] = {nm: {CL[k]: float(cw @ np.abs(bias(F, y == k))) for k in range(5)} for nm, F in (('true_S', F1), ('chain_S', V['h4']))}
    res['h4_r2_cost_weighted'] = {nm: float(cw @ np.array([r2sub(F[:, i], T[:, i], np.ones(len(y), bool)) for i in range(len(UNITS['h4']))]))
                                 for nm, F in (('true_S', F1), ('chain_S', V['h4']))}
    # S input: chain S^ vs true S (spread units), cost-weighted |mean| and RMS
    sdS = np.maximum(S_true.std(0), 1e-12); dS = (Vs - S_true) / sdS
    res['S_chain_minus_true'] = {f'S:{UNITS["S"][i]}': dict(mean=float(dS[:, i].mean()), rms=float(np.sqrt((dS[:, i] ** 2).mean())),
                                                              mean_by_class=[float(dS[y == k, i].mean()) for k in range(5)]) for i in range(len(UNITS['S']))}
    if not detail: return res
    # S-input swaps: start from the formula on chain S^ and give it ONE unit's true S value (and the reverse)
    sw = {}
    for i, j in enumerate(UNITS['S']):
        A = Vs.copy(); A[:, i] = S_true[:, i]; B = S_true.copy(); B[:, i] = Vs[:, i]
        ca = sub(net, x, N, {'h4': fS(A)}).argmax(1); cb = sub(net, x, N, {'h4': fS(B)}).argmax(1)
        sw[f'S:{j}'] = dict(chain_S_with_this_unit_true=dict(acc=float((ca == y).mean()), same=float((ca == cn).mean())),
                            true_S_with_this_unit_chain=dict(acc=float((cb == y).mean()), same=float((cb == cn).mean())))
        log(f'  S swap S:{j}: chain S^ + true S:{j} acc {sw[f"S:{j}"]["chain_S_with_this_unit_true"]["acc"]:.4f}; true S + chain S:{j} acc {sw[f"S:{j}"]["true_S_with_this_unit_chain"]["acc"]:.4f}')
    res['S_swap'] = sw
    # unit attribution: only unit j substituted; all but unit j substituted
    ua = {}
    for i, j in enumerate(UNITS['h4']):
        v = N['h4'].copy(); v[:, j] = F1[:, i]; co = net.run_jetwise(x, {'h4': (MASK['h4'], v)})['logits'].argmax(1)
        v = N['h4'].copy(); v[:, UNITS['h4']] = F1; v[:, j] = N['h4'][:, j]; cl = net.run_jetwise(x, {'h4': (MASK['h4'], v)})['logits'].argmax(1)
        ua[f'h4:{j}'] = dict(cost=LIVE['h4'][j], only_this=dict(acc=float((co == y).mean()), same=float((co == cn).mean()), per_class_acc=[float((co[y == k] == k).mean()) for k in range(5)]),
                             all_but_this=dict(acc=float((cl == y).mean()), same=float((cl == cn).mean())),
                             bias_sd_true_S=float(((F1[:, i] - T[:, i]).mean()) / sd[i]), bias_sd_chain_S=float(((V['h4'][:, i] - T[:, i]).mean()) / sd[i]),
                             r2_true_S=r2sub(F1[:, i], T[:, i], np.ones(len(y), bool)), r2_chain_S=r2sub(V['h4'][:, i], T[:, i], np.ones(len(y), bool)))
    res['unit_attribution'] = ua
    return res


# ------------------------------------------------------------------ modes
def mode_core(net, last):
    i0, i1 = E.indices(6000, 0), E.indices(2000, 11); keep = ~np.isin(i1, i0); it = i1[keep]
    J0 = len(i0); perm = np.random.default_rng(0).permutation(J0); nf = 2 * J0 // 3; ifit, idev = i0[perm[:nf]], i0[perm[nf:]]
    D, yall = load(dict(fitdev=i0, test=it), need_q=['test'])
    x, y, Q = D['test']['x'], D['test']['y'], D['test']['Q']
    res = dict(check1=dict(n_seed11=int(len(i1)), overlap_removed=int((~keep).sum()), n_test=int(len(it)), n_fit=int(len(ifit)), n_dev=int(len(idev)),
                           test_cap_fit=int(np.isin(it, ifit).sum()), test_cap_dev=int(np.isin(it, idev).sum()), fit_cap_dev=int(np.isin(ifit, idev).sum()),
                           test_per_class=[int((y == k).sum()) for k in range(5)]))
    hs = lambda X: {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in X}
    hf, ht = hs(D['fitdev']['x']), hs(x)
    res['check1']['content_duplicates_test_vs_fitdev'] = len(hf & ht); res['check1']['distinct_test_rows'] = len(ht)
    log('check 1 indices:', res['check1'])
    N = net.run(x, keep=STG + ('logits',)); cn = N['logits'].argmax(1); res['network'] = dict(acc=float((cn == y).mean()), per_class_acc=[float((cn[y == k] == k).mean()) for k in range(5)])
    log(f'network acc {res["network"]["acc"]:.4f}')
    res['check1']['last_layer'] = last.info
    # network's own logits from its own z through the standalone last layer
    lz_j, lz_n = last.jax_logits(N['z']), last.np_logits(N['z'])
    res['check1']['network_logits_from_own_z'] = dict(jax_max_abs=float(np.abs(lz_j - N['logits']).max()), np_same_call=float((lz_n.argmax(1) == cn).mean()),
                                                      np_max_abs=float(np.abs(lz_n - N['logits']).max()))
    for prefix in ('e11', 'e11hp'):
        hp = prefix == 'e11hp'; fa, fb = formulas(prefix); phys = physics(x, Q, hp)
        V = chain(fb, phys); r = dict(size=count_terms(fb))
        lo_e11 = sub(net, x, N, {k: V[k] for k in STG})                 # E11's cumulative (b) at z (every stage overridden)
        xs = x.copy(); xs[...] = 0                                        # particles erased: the override must make them irrelevant
        lo_zonly = net.run_jetwise(x, {'z': (MASK['z'], V['z'])})['logits']; lo_zonly_noparticles = net.run_jetwise(xs, {'z': (MASK['z'], V['z'])})['logits']
        lo_j, lo_n = last.jax_logits(V['z']), last.np_logits(V['z'])
        r['logits_agreement'] = dict(e11_vs_zonly_max_abs=float(np.abs(lo_e11 - lo_zonly).max()), zonly_vs_zeroed_particles_max_abs=float(np.abs(lo_zonly - lo_zonly_noparticles).max()),
                                     zonly_vs_jax_d7_max_abs=float(np.abs(lo_zonly - lo_j).max()), jax_vs_np64_same_call=float((lo_j.argmax(1) == lo_n.argmax(1)).mean()),
                                     jax_vs_np64_max_abs=float(np.abs(lo_j - lo_n).max()))
        r['program'] = callstats(lo_j, y, cn); r['program_np64'] = callstats(lo_n, y, cn); r['program_e11_path'] = callstats(lo_e11, y, cn)
        r['wrap_jets'] = dict(program=int(last.wraps(V['z']).sum()), network=int(last.wraps(N['z']).sum()))
        Fa_z = E.evaluate(fa['z']['units'], phys, False); r['z_on_physics'] = callstats(last.jax_logits(Fa_z), y, cn)
        log(f'{prefix}: program same {r["program"]["same_call"]:.4f} acc {r["program"]["acc"]:.4f} (paired diff {r["program"]["paired_vs_network"]["acc_diff"]:+.4f} ± '
            f'{r["program"]["paired_vs_network"]["se"]:.4f}); z-on-physics same {r["z_on_physics"]["same_call"]:.4f} acc {r["z_on_physics"]["acc"]:.4f}; logits {r["logits_agreement"]}; size {r["size"]}')
        r['check3'] = disagreement(net, x, y, N, cn, V, fb, phys, last)
        log(f'{prefix} check 3: disagree {r["check3"]["n_disagree"]}; restart {r["check3"]["restart_from_true"]}; first diverging {r["check3"]["first_diverging_stage_of_disagreeing"]}')
        r['check4'] = h4_single(net, x, y, N, cn, V, fb, phys, detail=not hp)
        log(f'{prefix} check 4: single (b) h4 {r["check4"]["single_b_true_S"]["same_call"]:.4f} / {r["check4"]["single_b_true_S"]["acc"]:.4f}; chain values at h4 '
            f'{r["check4"]["chain_values_only_at_h4"]["same_call"]:.4f} / {r["check4"]["chain_values_only_at_h4"]["acc"]:.4f}; cumulative {r["check4"]["cumulative_b_h4"]["acc"]:.4f}')
        res[prefix] = r
        (WOUT / 'e11_verify_core.json').write_text(json.dumps(rnd(res, 5), indent=1, ensure_ascii=False))
    return res


def mode_fresh(net, last):
    i0, i1, i2 = E.indices(6000, 0), E.indices(2000, 11), E.indices(4000, 31); keep = ~np.isin(i2, np.concatenate([i0, i1])); ifr = i2[keep]
    D, _ = load(dict(fresh=ifr), need_q=['fresh']); x, y, Q = D['fresh']['x'], D['fresh']['y'], D['fresh']['Q']
    res = dict(n_seed31=int(len(i2)), removed_overlap=int((~keep).sum()), n=int(len(ifr)), per_class=[int((y == k).sum()) for k in range(5)],
               overlap_check=dict(with_seed0=int(np.isin(ifr, i0).sum()), with_seed11=int(np.isin(ifr, i1).sum())))
    N = net.run(x, keep=STG + ('logits',)); cn = N['logits'].argmax(1)
    res['network'] = dict(acc=float((cn == y).mean()), per_class_acc=[float((cn[y == k] == k).mean()) for k in range(5)])
    log(f'fresh: {res["n"]} jets (removed {res["removed_overlap"]}), network acc {res["network"]["acc"]:.4f}')
    for prefix in ('e11', 'e11hp'):
        hp = prefix == 'e11hp'; fa, fb = formulas(prefix); phys = physics(x, Q, hp); V = chain(fb, phys)
        lo_p = last.jax_logits(V['z']); Fa_z = E.evaluate(fa['z']['units'], phys, False); lo_a = last.jax_logits(Fa_z)
        r = dict(program=callstats(lo_p, y, cn), z_on_physics=callstats(lo_a, y, cn), wrap_jets=dict(program=int(last.wraps(V['z']).sum()), network=int(last.wraps(N['z']).sum())))
        r['rejection50_program'] = rejections(lo_p, N['logits'], y); r['rejection50_z_on_physics'] = rejections(lo_a, N['logits'], y)
        r['stage_r2_chain'] = {s: float(np.median([r2sub(V[s][:, i], N[s][:, UNITS[s]][:, i], np.ones(len(y), bool)) for i in range(len(UNITS[s]))])) for s in STG}
        log(f'{prefix} fresh: program same {r["program"]["same_call"]:.4f} acc {r["program"]["acc"]:.4f} (Δ {r["program"]["paired_vs_network"]["acc_diff"]:+.4f} ± {r["program"]["paired_vs_network"]["se"]:.4f}) '
            f'per class {np.round(r["program"]["per_class_acc"], 4).tolist()} | z-on-physics same {r["z_on_physics"]["same_call"]:.4f} acc {r["z_on_physics"]["acc"]:.4f}')
        log(f'{prefix} R50 ratio program/network: ' + '; '.join(f'{k} {v["program"]:.2f}/{v["network"]:.2f} = {v["ratio"]:.3f} [{v["ratio_boot_2p5_97p5"][0]:.3f}, {v["ratio_boot_2p5_97p5"][1]:.3f}]' for k, v in r['rejection50_program'].items()))
        log(f'{prefix} R50 ratio z-on-physics/network: ' + '; '.join(f'{k} {v["ratio"]:.3f}' for k, v in r['rejection50_z_on_physics'].items()))
        r['check3'] = disagreement(net, x, y, N, cn, V, fb, phys, last, cumul=False)
        log(f'{prefix} fresh check 3: disagree {r["check3"]["n_disagree"]}; restart {r["check3"]["restart_from_true"]}')
        r['check4'] = h4_single(net, x, y, N, cn, V, fb, phys, detail=False)
        log(f'{prefix} fresh check 4: single (b) h4 {r["check4"]["single_b_true_S"]["same_call"]:.4f} / {r["check4"]["single_b_true_S"]["acc"]:.4f}; chain values at h4 '
            f'{r["check4"]["chain_values_only_at_h4"]["acc"]:.4f}; cumulative {r["check4"]["cumulative_b_h4"]["acc"]:.4f}; per class single {np.round(r["check4"]["single_b_true_S"]["per_class_acc"], 4).tolist()}')
        res[prefix] = r
        (WOUT / 'e11_verify_fresh.json').write_text(json.dumps(rnd(res, 5), indent=1, ensure_ascii=False))
    return res


def mode_edit(net, last, per_class):
    from jetdistill.observables import compute
    i0, i1, i2 = E.indices(6000, 0), E.indices(2000, 11), E.indices(4000, 31); ifr = i2[~np.isin(i2, np.concatenate([i0, i1]))]
    D, yall = load(dict(fresh=ifr), need_q=['fresh']); yf = D['fresh']['y']; rng = np.random.default_rng(5)
    pick = np.sort(np.concatenate([rng.choice(np.flatnonzero(yf == c), per_class, replace=False) for c in range(5)]))
    x0 = D['fresh']['x'][pick]; y = yf[pick]; Qs = {k: v[pick] for k, v in D['fresh']['Q'].items()}; del D
    xs = x0.copy(); rl = xs[..., 0] > 0; xs[..., 1] += np.float32(0.06) * rl                      # E6v edit_shift(x, 0.06, 0): padding fixed
    xp = x0.copy(); xp[..., 0] *= np.float32(0.8)                                                # E6 edit_pt(x, 0.8)
    X = dict(unedited=x0, shift_eta_0p06=xs, pt_x0p8=xp); Qr = {}
    for k, v in X.items():
        t = time.time(); q = compute(v.astype(np.float64), 64); Qr[k] = {kk: np.asarray(vv, np.float32).astype(np.float64) for kk, vv in q.items()}
        log(f'observables recomputed for {k}: {len(q)} in {time.time() - t:.0f} s')
    common = [k for k in Qs if k in Qr['unedited']]
    dq = {k: float(np.abs(Qr['unedited'][k] - Qs[k]).max() / max(np.abs(Qs[k]).max(), 1e-12)) for k in common}
    res = dict(n=int(len(y)), per_class=per_class, recompute_vs_stored=dict(n_keys=len(common), max_rel_diff=float(max(dq.values())),
                                                                            keys_rel_diff_gt_1em5=sorted(k for k, v in dq.items() if v > 1e-5)[:30]))
    NR = {k: net.run(v, keep=STG + ('logits',)) for k, v in X.items()}; C = {k: v['logits'].argmax(1) for k, v in NR.items()}
    def flows(c0, c1):
        return {CL[a]: dict(unedited=[float((c0[y == a] == b).mean()) for b in range(5)], edited=[float((c1[y == a] == b).mean()) for b in range(5)],
                            changed=float((c1[y == a] != c0[y == a]).mean())) for a in range(5)}
    res['network'] = {k: flows(C['unedited'], C[k]) for k in X if k != 'unedited'}
    res['network_acc'] = {k: float((C[k] == y).mean()) for k in X}
    mi = lambda xx: E.handbuilt(xx, False)['H:√(M0·M2)']
    res['physics_shift'] = {k: dict(mass_mean_rel=float(np.mean(Qr[k]['mass'] / np.maximum(Qr['unedited']['mass'], 1e-9) - 1)),
                                    sqrt_M0M2_mean_rel=float(np.mean(mi(X[k]) / np.maximum(mi(x0), 1e-9) - 1)),
                                    sumpt_r2_all_mean_rel=float(np.mean(E.handbuilt(X[k])['H:ΣpT·r²[all]'] / np.maximum(E.handbuilt(x0)['H:ΣpT·r²[all]'], 1e-12) - 1)))
                            for k in X if k != 'unedited'}
    log('network flows:', {k: dict(W_to_Z=v['W']['edited'][3], W_to_Z_unedited=v['W']['unedited'][3], Z_to_t=v['Z']['edited'][4], Z_to_t_unedited=v['Z']['unedited'][4]) for k, v in res['network'].items()})
    log('physics shifts:', res['physics_shift'])
    for prefix in ('e11', 'e11hp'):
        hp = prefix == 'e11hp'; fa, fb = formulas(prefix); r = {}
        ph_stored = physics(x0, Qs, hp); Vst = chain(fb, ph_stored); cst = last.jax_logits(Vst['z']).argmax(1)
        PH = {k: physics(X[k], Qr[k], hp) for k in X}; V = {k: chain(fb, PH[k]) for k in X}
        Cp = {k: last.jax_logits(V[k]['z']).argmax(1) for k in X}; Ca = {k: last.jax_logits(E.evaluate(fa['z']['units'], PH[k], False)).argmax(1) for k in X}
        r['program_stored_vs_recomputed_inputs_same_call'] = float((cst == Cp['unedited']).mean())
        r['program'] = {k: flows(Cp['unedited'], Cp[k]) for k in X if k != 'unedited'}; r['z_on_physics'] = {k: flows(Ca['unedited'], Ca[k]) for k in X if k != 'unedited'}
        r['same_call_with_network'] = {k: dict(program=float((Cp[k] == C[k]).mean()), z_on_physics=float((Ca[k] == C[k]).mean())) for k in X}
        r['acc'] = {k: dict(program=float((Cp[k] == y).mean()), z_on_physics=float((Ca[k] == y).mean())) for k in X}
        # per-stage response: mean shift (spread units) of chain vs network live units, and per-jet correlation of the shifts
        resp = {}
        for k in X:
            if k == 'unedited': continue
            rs = {}
            for s in STG:
                Tn0, Tn1 = NR['unedited'][s][:, UNITS[s]], NR[k][s][:, UNITS[s]]; sd = np.maximum(Tn0.std(0), 1e-12)
                dn = (Tn1 - Tn0) / sd; dp = (V[k][s] - V['unedited'][s]) / sd; cost = np.array([LIVE[s][j] for j in UNITS[s]]); cw = cost / cost.sum()
                mn, mp = dn.mean(0), dp.mean(0)
                cor = np.array([np.corrcoef(dn[:, i], dp[:, i])[0, 1] if dn[:, i].std() > 1e-9 and dp[:, i].std() > 1e-9 else np.nan for i in range(len(UNITS[s]))])
                rs[s] = dict(net_mean_abs_shift_cw=float(cw @ np.abs(mn)), program_mean_abs_shift_cw=float(cw @ np.abs(mp)),
                             corr_of_unit_mean_shifts=float(np.corrcoef(mn, mp)[0, 1]) if mn.std() > 0 and mp.std() > 0 else None,
                             per_jet_corr_median=float(np.nanmedian(cor)), slope_program_on_net=float((mn @ mp) / max(mn @ mn, 1e-12)))
            resp[k] = rs
        r['stage_response'] = resp
        log(f'{prefix} program flows: ' + ', '.join(f'{k}: W->Z {v["W"]["edited"][3]:.3f} (unedited {v["W"]["unedited"][3]:.3f}), Z->t {v["Z"]["edited"][4]:.3f} (unedited {v["Z"]["unedited"][4]:.3f})' for k, v in r['program'].items()))
        log(f'{prefix} z-on-physics flows: ' + ', '.join(f'{k}: W->Z {v["W"]["edited"][3]:.3f}, Z->t {v["Z"]["edited"][4]:.3f}' for k, v in r['z_on_physics'].items()))
        log(f'{prefix} same call with network: {r["same_call_with_network"]}; stored vs recomputed {r["program_stored_vs_recomputed_inputs_same_call"]:.4f}')
        log(f'{prefix} stage response: ' + json.dumps(rnd(resp, 3)))
        res[prefix] = r
        (WOUT / 'e11_verify_edit.json').write_text(json.dumps(rnd(res, 5), indent=1, ensure_ascii=False))
    return res


def mode_nonlive(net, last):
    """coordinator addition: every NON-live unit of every stage set to its mean over the 20,000 fit jets (live units = the
    program's chain values), for each cumulative row P..k; k = z is the fully standalone program"""
    i0, i1 = E.indices(6000, 0), E.indices(2000, 11); it = i1[~np.isin(i1, i0)]
    perm = np.random.default_rng(0).permutation(len(i0)); ifit = np.sort(i0[perm[:2 * len(i0) // 3]])
    D, _ = load(dict(fit=ifit, test=it), need_q=['test']); x, y, Q = D['test']['x'], D['test']['y'], D['test']['Q']
    Nf = net.run(D['fit']['x'], keep=STG); mean = {s: Nf[s].mean(0) for s in STG}; del Nf
    N = net.run(x, keep=STG + ('logits',)); cn = N['logits'].argmax(1); J = len(y)
    res = dict(n_test=int(J), n_fit=int(len(ifit)), network_acc=float((cn == y).mean()), n_nonlive={s: int(WIDTH[s] - len(UNITS[s])) for s in STG})
    one = {s: np.ones(WIDTH[s], np.float32) for s in STG}
    for prefix in ('e11', 'e11hp'):
        hp = prefix == 'e11hp'; fa, fb = formulas(prefix); phys = physics(x, Q, hp); V = chain(fb, phys); r = {}
        for i, k in enumerate(STG):
            ov_net, ov_mean = {}, {}
            for s in STG[:i + 1]:
                v = N[s].copy(); v[:, UNITS[s]] = V[s]; ov_net[s] = (MASK[s], v)                    # E11: non-live = network
                w = np.tile(mean[s], (J, 1)); w[:, UNITS[s]] = V[s]; ov_mean[s] = (one[s], w)        # non-live = fit-jet mean
            ln = net.run_jetwise(x, ov_net)['logits']; lm = net.run_jetwise(x, ov_mean)['logits']
            r[k] = dict(nonlive_network=callstats(ln, y, cn), nonlive_fit_mean=callstats(lm, y, cn), calls_differ=int((ln.argmax(1) != lm.argmax(1)).sum()),
                        max_abs_logit_diff=float(np.abs(ln - lm).max()))
            log(f'{prefix} cumulative P..{k}: non-live = network {r[k]["nonlive_network"]["same_call"]:.4f} / {r[k]["nonlive_network"]["acc"]:.4f}; '
                f'non-live = fit mean {r[k]["nonlive_fit_mean"]["same_call"]:.4f} / {r[k]["nonlive_fit_mean"]["acc"]:.4f}; calls differ {r[k]["calls_differ"]}, max |Δlogit| {r[k]["max_abs_logit_diff"]:.2e}')
        ov_all = {}
        for s_ in STG:
            w = np.tile(mean[s_], (J, 1)); w[:, UNITS[s_]] = V[s_]; ov_all[s_] = (one[s_], w)
        l_all = net.run_jetwise(x, ov_all)['logits']; l_all0 = net.run_jetwise(np.zeros_like(x), ov_all)['logits']
        r['standalone_all_stages'] = dict(stats=callstats(l_all, y, cn), zeroed_particles_max_abs_logit_diff=float(np.abs(l_all - l_all0).max()),
                                          vs_jax_d7_max_abs=float(np.abs(l_all - last.jax_logits(V['z'])).max()))
        log(f'{prefix} standalone (all stages, non-live = fit mean): {r["standalone_all_stages"]["stats"]["same_call"]:.4f} / {r["standalone_all_stages"]["stats"]["acc"]:.4f}; '
            f'zeroed particles max |Δlogit| {r["standalone_all_stages"]["zeroed_particles_max_abs_logit_diff"]:.2e}; vs d7 alone {r["standalone_all_stages"]["vs_jax_d7_max_abs"]:.2e}')
        res[prefix] = r
        (WOUT / 'e11_verify_nonlive.json').write_text(json.dumps(rnd(res, 5), indent=1, ensure_ascii=False))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['core', 'fresh', 'edit', 'nonlive']); ap.add_argument('--per-class', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=0)
    A = ap.parse_args(); log('args', vars(A)); LIMIT = A.limit
    if LIMIT: WOUT = OUT / 'e11v_smoke'; WOUT.mkdir(exist_ok=True)
    net = Jedi(64); last = Last(net); MASK = masks()
    log('last layer:', {k: v for k, v in last.info.items() if k != 'iq_config'})
    {'core': lambda: mode_core(net, last), 'fresh': lambda: mode_fresh(net, last), 'edit': lambda: mode_edit(net, last, A.per_class), 'nonlive': lambda: mode_nonlive(net, last)}[A.mode]()
    log('done')
