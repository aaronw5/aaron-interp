"""E8 verification (skeptic): classes as conjunctions of neuron windows on JEDI-linear's own z and S (claim A6).

Network: JEDI-linear 3-feature n64, fixed-point forward (jedi_core.Jedi(64)); call = argmax of the logits.
Sample: sample(per_class=7800, seed=23); jets that are also in E8's sample(per_class=6000, seed=4) (matched by the bytes of the
  (64, 3) particle array) are dropped; 6,000 per class kept at random (rng 23). Split: random 2/3 fit / 1/3 evaluation (rng 1).
Checks: run_jetwise override of z / S with the network's own values (0 calls changed expected); the tail copy
  (S: d4 -> d5 -> d6 -> relu -> d7; z: relu -> d7) vs run_jetwise on every rule imposition.
Part A (E8's rules, values from E8's JSON, on the fresh evaluation jets): precision / coverage; rule units set to E8's in-rule
  medians on jets called otherwise (share now called c, by source call); singles and pairs; dose 0.25-1; necessity (rule units
  at E8's other-call medians on in-rule jets called c); the same with in-rule medians recomputed on the fresh fit jets; at z:
  additivity of the logit change (rule vs sum of single units); mass quantiles of the jets inside each window.
Part B (refit): E8's tree procedure (3-deep sklearn tree, min_samples_leaf 100, 19-knot 5-95 % grid + on/off; z grid from the
  positive values) refit on the fresh fit jets -> rule, precision / coverage on the fresh evaluation jets, imposition.
Part C (controls not in E8):
  C1 exhaustive subsets: every set of K units (K = the rule's size) from the 16 z / 19 live S units set to (i) the class-c medians
     of fit jets called c, (ii) the medians of fit jets called c inside E8's rule (the rule's own value source) -> share of
     other-called jets switched; rank of the rule's own unit set among all sets.
  C2 basis: the same tree + imposition on 5 random orthogonal rotations of the standardized units (plain 19-knot grid, no on/off),
     and on the identity rotation with the same plain grid (neuron basis, plain grid); imposition = the rule's K rotated coordinates
     set to their in-rule medians, rotated back.
  C3 window entry: on other-called jets, each rule unit outside its window is moved just inside it (to the 5 % / 95 % quantile of
     the fit jets called c inside the rule: "inner"; to the window edge: "edge"); units inside are untouched. Also "donor": the rule
     units copied jointly from a random fit jet called c inside the rule. Share now called c, against the rule's natural precision.
Output: $AI_RESULTS/jedi/e8_verify.json (E8V_SMOKE=1: e8_verify_smoke.json, 800 per class)."""
import gc, hashlib, json, os, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get('E8V_SMOKE') == '1'
PER = 800 if SMOKE else 6000
SEED = 23
CL = ['g', 'q', 'W', 'Z', 't']
DEPTH, MIN_LEAF, N_KNOTS, ONOFF = 3, (20 if SMOKE else 100), 19, 1e-6
N_ROT = 2 if SMOKE else 5
DOSES = (0.25, 0.5, 0.75, 1.0)
OUTF = OUT / ('e8_verify_smoke.json' if SMOKE else 'e8_verify.json')
t0 = time.time()


def log(*a):
    print(' '.join(str(v) for v in a) + f'  [{time.time() - t0:.0f} s]', flush=True)


def pct(v):
    return round(float(v) * 100, 2)


E8 = json.loads((OUT / 'e8_conjunctions.json').read_text())
LIVE = [int(u) for u in E8['liveS']]
UNITS = dict(z=list(range(16)), S=LIVE)

# ------------------------------------------------------------------ fresh sample, disjoint from E8's
hsh = lambda r: hashlib.blake2b(np.ascontiguousarray(r).tobytes(), digest_size=16).digest()
x4 = sample(per_class=6000, seed=4)[0]
H4 = {hsh(r) for r in x4}; n4 = len(x4); del x4; gc.collect()
xa, ya, Qa = sample(per_class=PER + 1800, seed=SEED)
dup = np.array([hsh(r) in H4 for r in xa])
rng = np.random.default_rng(SEED)
avail = [int(((ya == c) & ~dup).sum()) for c in range(5)]
keep = np.sort(np.concatenate([rng.choice(np.flatnonzero((ya == c) & ~dup), min(PER, avail[c]), replace=False) for c in range(5)]))
x, y = np.ascontiguousarray(xa[keep]), np.asarray(ya[keep])
Q = {k: np.asarray(Qa[k], np.float64)[keep] for k in ('mass', 'sum_pt', 'n_particles')}
n_overlap_kept = int(sum(hsh(r) in H4 for r in x))
del xa, Qa, H4; gc.collect()
log(f'sample: sample(per_class={PER + 1800}, seed={SEED}) = {len(dup)} jets; {int(dup.sum())} also in the E8 sample ({n4} jets) dropped; '
    f'non-overlapping per class {avail}; kept {len(x)} ({PER}/class); overlap of kept jets with E8: {n_overlap_kept}')

net = Jedi(64)
R = net.run(x, keep=('S', 'z', 'logits')); call = R['logits'].argmax(1)
BASE = dict(z=R['z'], S=R['S'])
J = len(x); perm = np.random.default_rng(1).permutation(J); nf = 2 * J // 3
fit, ev = np.sort(perm[:nf]), np.sort(perm[nf:])
xe, ce, cf = x[ev], call[ev], call[fit]
log(f'network accuracy: all {(call == y).mean() * 100:.2f} %, eval {(ce == y[ev]).mean() * 100:.2f} %; calls on eval {np.bincount(ce, minlength=5).tolist()}')

import jax, jax.numpy as jnp
L = net.L; KB = {k: (jnp.asarray(v[0]), jnp.asarray(v[1])) for k, v in net.KB.items()}


def dense(k, t, act=True):
    o = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1]
    return L[k].activation(o) if act and L[k].activation is not None else o


TAIL = dict(S=jax.jit(lambda S: dense('d7', jax.nn.relu(dense('d6', dense('d5', dense('d4', S)), act=False)))),
            z=jax.jit(lambda z: dense('d7', jax.nn.relu(z))))


def logits_of(stage, V):
    return np.asarray(TAIL[stage](jnp.asarray(V, jnp.float32)), np.float64).reshape(len(V), -1)


MISMATCH = []


def rj_calls(stage, Vev):
    return net.run_jetwise(xe, {stage: (np.ones(Vev.shape[1]), Vev)})['logits'].argmax(1)


def calls_rows(stage, idx, rows, check=False):
    """calls of eval jets idx when their stage values are replaced by rows (len(idx), width); the tail is run on those rows only
    (dense layers act per jet); check: the full eval matrix through run_jetwise, mismatches on idx recorded"""
    c = logits_of(stage, rows).argmax(1)
    if check:
        V = BASE[stage][ev].copy(); V[idx] = rows
        MISMATCH.append(int((rj_calls(stage, V)[idx] != c).sum()))
    return c


CHECK = {}
for st in ('z', 'S'):
    c_rj = rj_calls(st, BASE[st][ev]); c_t = logits_of(st, BASE[st][ev]).argmax(1)
    idx = np.flatnonzero(ce != 0); c_sub = logits_of(st, BASE[st][ev][idx]).argmax(1)
    CHECK[st] = dict(run_jetwise_changed=int((c_rj != ce).sum()), tail_changed=int((c_t != ce).sum()), tail_on_subset_changed=int((c_sub != ce[idx]).sum()))
    log(f'check {st}: own-value override via run_jetwise changes {CHECK[st]["run_jetwise_changed"]} calls; tail {CHECK[st]["tail_changed"]}; '
        f'tail on a row subset {CHECK[st]["tail_on_subset_changed"]}')

# ------------------------------------------------------------------ helpers (as E8)
from sklearn.tree import DecisionTreeClassifier


def grids(V, positive_part=False, onoff=True):
    G = []
    for j in range(V.shape[1]):
        v = V[:, j]
        if positive_part and (v > 0).sum() >= 50: v = v[v > 0]
        q = np.quantile(v, np.linspace(.05, .95, N_KNOTS))
        if onoff: q = np.concatenate([q[np.abs(q) > 2 * ONOFF], [ONOFF]])
        G.append(np.unique(q))
    return G


def binify(V, G):
    return np.stack([np.searchsorted(G[j], V[:, j], side='right') for j in range(V.shape[1])], 1)


def leaves_of(tree, G):
    T = tree.tree_; out = []

    def walk(node, cond):
        if T.children_left[node] == -1:
            out.append((node, {k: list(v) for k, v in cond.items()})); return
        f, b = int(T.feature[node]), int(np.floor(T.threshold[node])); t = float(G[f][min(b, len(G[f]) - 1)])
        c = {k: list(v) for k, v in cond.items()}; w = c.setdefault(f, [None, None])
        w[1] = t if w[1] is None else min(w[1], t); walk(T.children_left[node], c)
        c = {k: list(v) for k, v in cond.items()}; w = c.setdefault(f, [None, None])
        w[0] = t if w[0] is None else max(w[0], t); walk(T.children_right[node], c)
    walk(0, {}); return out


def inside(V, cond):
    m = np.ones(len(V), bool)
    for f, (lo, hi) in cond.items():
        if lo is not None: m &= V[:, int(f)] >= lo
        if hi is not None: m &= V[:, int(f)] < hi
    return m


def pc(mask, pos):
    n, tp = int(mask.sum()), int((mask & pos).sum()); P = tp / max(n, 1); C = tp / max(int(pos.sum()), 1)
    return P, C, (2 * P * C / (P + C) if P + C > 0 else 0.0), n


def cond_str(cond, names):
    parts = []
    for f, (lo, hi) in sorted(cond.items(), key=lambda kv: int(kv[0])):
        nm = names[int(f)]
        if lo == ONOFF: parts.append(f'{nm} > 0' + (f' and {nm} < {hi:.3g}' if hi is not None else ''))
        elif hi == ONOFF: parts.append((f'{lo:.3g} <= ' if lo is not None else '') + f'{nm} <= 0')
        elif lo is not None and hi is not None: parts.append(f'{lo:.3g} <= {nm} < {hi:.3g}')
        elif lo is not None: parts.append(f'{nm} >= {lo:.3g}')
        else: parts.append(f'{nm} < {hi:.3g}')
    return ' and '.join(parts)


def describe(Vf, Ve, positive_part, onoff):
    G = grids(Vf, positive_part, onoff); Bf = binify(Vf, G); out = {}
    for c in range(5):
        pos_f, pos_e = cf == c, ce == c
        if pos_f.sum() < MIN_LEAF or (~pos_f).sum() < MIN_LEAF: out[c] = None; continue
        tr = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=0).fit(Bf, pos_f)
        best = None
        for node, cond in leaves_of(tr, G):
            Pf, Cf, Ff, _ = pc(inside(Vf, cond), pos_f)
            if best is None or Ff > best[0]: best = (Ff, cond, Pf, Cf)
        Pe, Ce, Fe, ne = pc(inside(Ve, best[1]), pos_e)
        out[c] = dict(rule=best[1], precision=Pe, coverage=Ce, prec_fit=best[2], cov_fit=best[3], n_in_eval=ne)
    return out


def impose(stage, units, values, c, src=None, which='other', check=False):
    """eval jets with src != c ('other') or == c ('same'): units set to values ((K,) or (n, K)) -> % called c, by source, n"""
    src = ce if src is None else src
    idx = np.flatnonzero(src != c if which == 'other' else src == c)
    rows = BASE[stage][ev][idx].copy(); rows[:, list(units)] = values
    ca = calls_rows(stage, idx, rows, check)
    by = {CL[k]: pct((ca[src[idx] == k] == c).mean()) for k in range(5) if k != c and (src[idx] == k).sum()} if which == 'other' else None
    return pct((ca == c).mean()), by, len(idx)


RES = dict(N=J, per_class=PER, seed=SEED, sample_note=f'sample(per_class={PER + 1800}, seed={SEED}) minus the jets of E8 sample(6000, 4); '
           f'{PER}/class kept (rng {SEED}); split 2/3 fit / 1/3 eval (rng 1)', n_dropped_overlap=int(dup.sum()), overlap_kept=n_overlap_kept,
           n_fit=len(fit), n_eval=len(ev), acc_all=float((call == y).mean()), acc_eval=float((ce == y[ev]).mean()),
           calls_eval=np.bincount(ce, minlength=5).tolist(), checks=CHECK, liveS=LIVE)

# ------------------------------------------------------------------ part A: E8's rules on the fresh jets
A = {}
for st in ('z', 'S'):
    cols = UNITS[st]; names = [f'{st}{u}' for u in cols]; Vf, Ve, Va = BASE[st][fit][:, cols], BASE[st][ev][:, cols], BASE[st][:, cols]
    A[st] = {}
    for c in range(5):
        d = E8[st]['description'][CL[c]]; iv = E8[st]['intervention'][CL[c]]
        cond = {int(k): v for k, v in d['rule'].items()}; cu = [int(u) for u in iv['rule_units']]
        med_in, med_o = np.array(iv['med_in_rule']), np.array(iv['med_other'])
        Pe, Ce, _, ne = pc(inside(Ve, cond), ce == c); Pa, Ca, _, na = pc(inside(Va, cond), call == c)
        r = dict(rule=cond_str(cond, names), e8=dict(precision=pct(d['precision']), coverage=pct(d['coverage']), switched=iv['rule'],
                 necessity=iv['necessity']['rule_to_other_med'], dose=iv['dose'], subsets=iv['subsets']),
                 precision=pct(Pe), coverage=pct(Ce), n_in_eval=ne, precision_all=pct(Pa), coverage_all=pct(Ca))
        r['switched'], r['by_source'], r['n_other'] = impose(st, cu, med_in, c, check=True)
        inf = inside(Vf, cond) & (cf == c)
        med_in_fresh = np.median(BASE[st][fit][inf][:, cu], 0)
        r['switched_fresh_medians'], _, _ = impose(st, cu, med_in_fresh, c)
        r['med_in_fresh'] = med_in_fresh.tolist(); r['med_in_e8'] = med_in.tolist()
        r['subsets'] = {}
        for k_ in range(1, len(cu)):
            for sub in combinations(range(len(cu)), k_):
                r['subsets'][' & '.join(f'{st}{cu[j]}' for j in sub)] = impose(st, [cu[j] for j in sub], med_in[list(sub)], c)[0]
        singles = [r['subsets'][f'{st}{u}'] for u in cu] if len(cu) > 1 else [r['switched']]
        r['sum_of_singles'] = round(float(sum(singles)), 2); r['max_single'] = max(singles)
        idx = np.flatnonzero(ce != c); r['dose'] = {}
        for f_ in DOSES:
            rows = BASE[st][ev][idx].copy(); rows[:, cu] = rows[:, cu] + f_ * (med_in - rows[:, cu])
            r['dose'][str(f_)] = pct((calls_rows(st, idx, rows) == c).mean())
        src = np.where((ce == c) & inside(Ve, cond), c, -1)
        r['necessity'], _, r['n_called_c_in_rule'] = impose(st, cu, med_o, c, src=src, which='same')
        if st == 'z':      # additivity of the logit change through d7
            rows0 = BASE['z'][ev][idx]; L0 = logits_of('z', rows0)
            rr = rows0.copy(); rr[:, cu] = med_in; dR = logits_of('z', rr) - L0; dS = np.zeros_like(dR)
            for j, u in enumerate(cu):
                r1 = rows0.copy(); r1[:, u] = med_in[j]; dS += logits_of('z', r1) - L0
            r['additivity_max_abs_diff'] = float(np.abs(dR - dS).max()); r['logit_change_scale'] = float(np.abs(dR).mean())
        r['windows_mass_q25_50_75'] = {}
        for f, (lo, hi) in cond.items():
            m = inside(Va, {f: (lo, hi)}); r['windows_mass_q25_50_75'][names[f]] = [round(float(np.percentile(Q['mass'][m], p)), 1) for p in (25, 50, 75)] if m.sum() else None
        A[st][CL[c]] = r
        log(f'A {st} {CL[c]} [{r["rule"]}]: precision {r["precision"]} (E8 {r["e8"]["precision"]}) coverage {r["coverage"]} (E8 {r["e8"]["coverage"]}); '
            f'switched {r["switched"]} (E8 {r["e8"]["switched"]}; fresh medians {r["switched_fresh_medians"]}); singles {singles} sum {r["sum_of_singles"]}; '
            f'dose {list(r["dose"].values())}; necessity {r["necessity"]} (E8 {r["e8"]["necessity"]}); by source {r["by_source"]}; '
            f'additivity {r.get("additivity_max_abs_diff")} (scale {r.get("logit_change_scale")}); windows {r["windows_mass_q25_50_75"]}')
RES['A_e8_rules_fresh'] = A
RES['tail_check'] = dict(n_runs=len(MISMATCH), mismatched_calls=int(sum(MISMATCH)))
log(f'tail vs run_jetwise: {len(MISMATCH)} runs, {sum(MISMATCH)} mismatched calls')
json.dump(RES, open(OUTF, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

# ------------------------------------------------------------------ part B: refit on the fresh fit jets
B = {}
for st in ('z', 'S'):
    cols = UNITS[st]; names = [f'{st}{u}' for u in cols]; Vf, Ve = BASE[st][fit][:, cols], BASE[st][ev][:, cols]
    D = describe(Vf, Ve, positive_part=(st == 'z'), onoff=True); B[st] = {}
    for c in range(5):
        if D[c] is None: B[st][CL[c]] = None; continue
        cond = D[c]['rule']; cu = [cols[f] for f in cond]
        med = np.median(BASE[st][fit][inside(Vf, cond) & (cf == c)][:, cu], 0)
        sw, by, _ = impose(st, cu, med, c)
        e8u = sorted(int(u) for u in E8[st]['intervention'][CL[c]]['rule_units'])
        B[st][CL[c]] = dict(rule=cond_str(cond, names), units=sorted(cu), same_units_as_e8=sorted(cu) == e8u, precision=pct(D[c]['precision']),
                            coverage=pct(D[c]['coverage']), switched=sw, by_source=by)
        log(f'B {st} {CL[c]} refit [{B[st][CL[c]]["rule"]}] precision {B[st][CL[c]]["precision"]} coverage {B[st][CL[c]]["coverage"]} switched {sw}; '
            f'same units as E8 {B[st][CL[c]]["same_units_as_e8"]} (E8 {e8u})')
RES['B_refit'] = B
json.dump(RES, open(OUTF, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

# ------------------------------------------------------------------ part C1: every K-subset of units
C1 = {}
for st in ('z', 'S'):
    cols = UNITS[st]; Vf = BASE[st][fit][:, cols]; C1[st] = {}
    for c in range(5):
        iv = E8[st]['intervention'][CL[c]]; cond = {int(k): v for k, v in E8[st]['description'][CL[c]]['rule'].items()}
        cu = sorted(int(u) for u in iv['rule_units']); K = len(cu)
        idx = np.flatnonzero(ce != c); rows0 = BASE[st][ev][idx]
        vals = dict(class_medians=np.median(BASE[st][fit][cf == c], 0),
                    in_rule_medians=np.median(BASE[st][fit][inside(Vf, cond) & (cf == c)], 0))
        C1[st][CL[c]] = {}
        for scheme, v in vals.items():
            res = []
            for sub in combinations(cols, K):
                rows = rows0.copy(); rows[:, list(sub)] = v[list(sub)]
                res.append((sub, float((calls_rows(st, idx, rows) == c).mean())))
            s = np.array([r_[1] for r_ in res]); rule_s = dict(res)[tuple(cu)]
            has = np.array([bool(set(r_[0]) & set(cu)) for r_ in res])
            order = np.argsort(-s)
            C1[st][CL[c]][scheme] = dict(K=K, n_sets=len(res), rule=pct(rule_s), rank=int((s > rule_s).sum()) + 1, n_ge=int((s >= rule_s).sum()),
                                        percentile=pct((s < rule_s).mean()), best=[[f'{st}{u}' for u in res[i][0]] + [pct(s[i])] for i in order[:5]],
                                        median_all=pct(np.median(s)), mean_all=pct(s.mean()), mean_no_rule_unit=pct(s[~has].mean()) if (~has).any() else None,
                                        max_no_rule_unit=pct(s[~has].max()) if (~has).any() else None, mean_with_rule_unit=pct(s[has].mean()))
            log(f'C1 {st} {CL[c]} {scheme}: rule {pct(rule_s)} % rank {C1[st][CL[c]][scheme]["rank"]}/{len(res)}; best {C1[st][CL[c]][scheme]["best"][:3]}; '
                f'mean all {pct(s.mean())}, no rule unit {C1[st][CL[c]][scheme]["mean_no_rule_unit"]} (max {C1[st][CL[c]][scheme]["max_no_rule_unit"]})')
RES['C1_all_subsets'] = C1
json.dump(RES, open(OUTF, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

# ------------------------------------------------------------------ part C2: rotated bases
C2 = {}
for st in ('z', 'S'):
    cols = UNITS[st]; Vf, Ve = BASE[st][fit][:, cols], BASE[st][ev][:, cols]
    mu, sd = Vf.mean(0), Vf.std(0); sd = np.where(sd > 0, sd, 1.0); C2[st] = {}
    for r in range(N_ROT + 1):
        Rm = np.eye(len(cols)) if r == 0 else np.linalg.qr(np.random.default_rng(100 + r).standard_normal((len(cols), len(cols))))[0]
        Yf, Ye = ((Vf - mu) / sd) @ Rm, ((Ve - mu) / sd) @ Rm
        back = lambda Y: (Y @ Rm.T) * sd + mu
        rt = np.flatnonzero(ce != 0); full = BASE[st][ev][rt].copy(); full[:, cols] = back(Ye[rt])
        rt_changed = int((calls_rows(st, rt, full) != ce[rt]).sum())
        D = describe(Yf, Ye, positive_part=False, onoff=False); key = 'identity_plain_grid' if r == 0 else f'rotation_{r}'; C2[st][key] = dict(roundtrip_calls_changed=rt_changed)
        for c in range(5):
            if D[c] is None: C2[st][key][CL[c]] = None; continue
            cond = D[c]['rule']; rc = sorted(cond)
            med = np.median(Yf[inside(Yf, cond) & (cf == c)][:, rc], 0)
            idx = np.flatnonzero(ce != c); Y2 = Ye[idx].copy(); Y2[:, rc] = med
            full = BASE[st][ev][idx].copy(); full[:, cols] = back(Y2)
            sw = pct((calls_rows(st, idx, full, check=(r == 1 and c == 0)) == c).mean())
            C2[st][key][CL[c]] = dict(n_coords=len(rc), precision=pct(D[c]['precision']), coverage=pct(D[c]['coverage']), switched=sw,
                                      rule=cond_str(cond, ([f'{st}{u}' for u in cols] if r == 0 else [f'y{j}' for j in range(len(cols))])))
        log(f'C2 {st} {key} (round trip changes {rt_changed} calls): ' + '; '.join(
            f'{CL[c]} {C2[st][key][CL[c]]["n_coords"]} coords P {C2[st][key][CL[c]]["precision"]} C {C2[st][key][CL[c]]["coverage"]} sw {C2[st][key][CL[c]]["switched"]}'
            for c in range(5) if C2[st][key][CL[c]]))
RES['C2_rotated_basis'] = C2
json.dump(RES, open(OUTF, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

# ------------------------------------------------------------------ part C3: entering the window
C3 = {}
for st in ('z', 'S'):
    cols = UNITS[st]; Vf, Ve = BASE[st][fit][:, cols], BASE[st][ev][:, cols]; C3[st] = {}
    for c in range(5):
        cond = {int(k): v for k, v in E8[st]['description'][CL[c]]['rule'].items()}; cu = [cols[f] for f in cond]
        vin = BASE[st][fit][inside(Vf, cond) & (cf == c)]
        idx = np.flatnonzero(ce != c); rows0 = BASE[st][ev][idx]
        already = inside(rows0[:, cols], cond)
        r = dict(n_other=len(idx), n_already_inside=int(already.sum()),
                 already_inside_called_c=pct((logits_of(st, rows0[already]).argmax(1) == c).mean()) if already.any() else None,
                 natural_precision=A[st][CL[c]]['precision'])
        for variant in ('inner', 'edge'):
            rows = rows0.copy()
            for f, (lo, hi) in cond.items():
                u = cols[f]
                if lo is not None:
                    m = rows[:, u] < lo; rows[m, u] = np.quantile(vin[:, u], .05) if variant == 'inner' else lo
                if hi is not None:
                    m = rows[:, u] >= hi; rows[m, u] = np.quantile(vin[:, u], .95) if variant == 'inner' else hi - 1e-3 * max(1.0, abs(hi))
            ok = inside(rows[:, cols], cond)
            ca = calls_rows(st, idx, rows)
            r[variant] = dict(all_inside_after=bool(ok.all()), switched=pct((ca == c).mean()),
                              switched_moved_only=pct((ca[~already] == c).mean()) if (~already).any() else None,
                              mean_abs_change=float(np.abs(rows[:, cu] - rows0[:, cu]).mean()))
        donor = vin[np.random.default_rng(7 + c).integers(0, len(vin), len(idx))][:, cu]
        rows = rows0.copy(); rows[:, cu] = donor; ca = calls_rows(st, idx, rows)
        r['donor'] = dict(switched=pct((ca == c).mean()), mean_abs_change=float(np.abs(rows[:, cu] - rows0[:, cu]).mean()))
        r['median_imposed'] = A[st][CL[c]]['switched_fresh_medians']
        C3[st][CL[c]] = r
        log(f'C3 {st} {CL[c]}: natural precision {r["natural_precision"]}; already inside {r["n_already_inside"]} (called c {r["already_inside_called_c"]}); '
            f'inner {r["inner"]}; edge {r["edge"]}; donor {r["donor"]}; median imposed {r["median_imposed"]}')
RES['C3_window_entry'] = C3
RES['tail_check'] = dict(n_runs=len(MISMATCH), mismatched_calls=int(sum(MISMATCH)))
RES['time_s'] = time.time() - t0
json.dump(RES, open(OUTF, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
log(f'tail vs run_jetwise: {len(MISMATCH)} runs, {sum(MISMATCH)} mismatched calls; wrote {OUTF}; done')
