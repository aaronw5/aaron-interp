"""E8 (JEDI-linear 64 p, 3-feature; claim A6): classes as conjunctions of neuron windows, imposed on the network's own z and S.

Mirrors J33b (done on the formula's objects) and G206 (ParT) on JEDI's own pre-ReLU neurons z and second-round objects S.
Sample: sample(per_class=6000, seed=4) = 30,000 test jets; random 2/3 fit (rng 0) / 1/3 evaluation (10,000).
Network: fixed-point forward (jedi_core.Jedi), stages S, z, logits kept; the call is argmax of the logits.
Checks: a per-jet z override with the network's own z, and an S override with its own S, change 0 calls (as E12).

Step 1 (description), per class c and level L in {z: the 16 pre-ReLU neurons, S: the 19 live S units (E1)}:
  a 3-deep decision tree (sklearn, gini, min_samples_leaf 100) on the fit jets predicting call == c from the units, with
  thresholds restricted to each unit's 5-95 % quantile grid (19 knots) plus 0 (on/off = relu(z) > 0; S > 0): each unit is
  binned by that grid and the tree splits the bins, so every split is "unit >= t" for a grid value t. The rule for class c is
  the leaf with the highest F1 (precision x coverage) on the fit jets, written as windows lo <= unit < hi per unit (1-3 units).
  Reported on the evaluation jets: precision = share of jets inside the rule that the network calls c; coverage = share of
  the network's c calls inside the rule; also the union of the tree's positive leaves. Per unit in any rule: the two poles
  (unit <= 0 vs > 0): n, true-label and call composition, median mass / sum pT / multiplicity; per window: the mass range
  (25-75 %) of the jets inside it; E6's mass windows where known (z0 75 GeV, z7/z14 94-98, z10 135-148).
Step 2 (intervention at z, run_jetwise override of stage z, the network's own d7 makes the call): on evaluation jets called
  otherwise, the rule's neurons are set to the in-rule medians of fit jets called c (other neurons untouched) -> share now
  called c (also split by the original call). Dose: each rule neuron moved a fraction f = 0.25, 0.5, 0.75, 1 of the way from
  the jet's own z to that median. Controls: the same number of random neurons set to class-c medians (10 draws from all 16;
  also 10 draws from the neurons outside the rule); the rule imposed on Jedi(untrained=True, seed 0) (its own calls; values =
  its own class-c medians when it calls c, and the trained medians); ceiling: all 16 neurons set to class-c medians.
  Necessity: evaluation jets called c inside the rule get the rule's neurons set to the medians of fit jets called otherwise
  -> share still called c (ceiling: all 16 neurons).
Step 3: the same at the object level S (tree on the 19 live S units; per-jet S override; the network's own h4/h5/z/d7
  compute; random live-S-unit control, 10 draws; ceilings: all 19 live units, all 64 units).
Metrics: precision / coverage per class and level; % of other-called jets switched into c (rule, random, untrained, ceiling);
  dose curves; sufficiency gap = ceiling - rule.  Output: $AI_RESULTS/jedi/e8_conjunctions.json (figure: interp/e8_fig.py).
"""
import json, os, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
LOGD = Path('logs'); LOGD.mkdir(exist_ok=True)
SMOKE = os.environ.get('E8_SMOKE') == '1'
PER = 600 if SMOKE else 6000
SEED = 4
CLASSES = ['g', 'q', 'W', 'Z', 't']
DEPTH, MIN_LEAF = 3, (20 if SMOKE else 100)
N_RAND = 3 if SMOKE else 10
DOSES = (0.25, 0.5, 0.75, 1.0)
N_KNOTS = 19
KNOWN = {0: 'W mass window, peak 75 GeV (E6)', 7: 'Z mass window 94-98 GeV (E6)', 14: 'Z mass window 94-98 GeV (E6)',
         10: 'top mass window 135-148 GeV (E6)'}
LOGF = open(LOGD / ('e8_smoke.log' if SMOKE else 'e8.log'), 'w')
t0 = time.time()


def log(*a):
    s = ' '.join(str(v) for v in a); print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


def pct(v):
    return round(float(v) * 100, 2)


# ------------------------------------------------------------------ network, sample, checks
net = Jedi(64)
x0, y0, _ = sample(per_class=2000, seed=0)
acc0 = float((net.run(x0)['logits'].argmax(1) == y0).mean())
log(f'check: seed-0 sample 10,000 jets, accuracy {acc0*100:.2f} % (E1: 81.29 %)')
del x0, y0

x, y, Q = sample(per_class=PER, seed=SEED); J = len(x)
R = net.run(x, keep=('S', 'z', 'logits'))
call = R['logits'].argmax(1)
perm = np.random.default_rng(0).permutation(J); nf = 2 * J // 3
fit, ev = np.sort(perm[:nf]), np.sort(perm[nf:])
xe, ye, ce = x[ev], y[ev], call[ev]
acc_all, acc_ev = float((call == y).mean()), float((ce == ye).mean())
log(f'sample: {J} jets (per_class {PER}, seed {SEED}); fit {len(fit)}, eval {len(ev)}; network accuracy on all {acc_all*100:.2f} %, '
    f'on eval {acc_ev*100:.2f} %; calls on eval {np.bincount(ce, minlength=5).tolist()}  [{time.time()-t0:.0f} s]')

LIVE_S = [0, 1, 13, 17, 19, 20, 29, 30, 31, 40, 43, 50, 51, 52, 56, 58, 60, 62, 63]      # E1: mean ablation changes >= 1 % of calls
liveS = LIVE_S
if (OUT / 'e1_e3.json').exists():
    e1 = json.loads((OUT / 'e1_e3.json').read_text())['E1']
    liveS = [int(k) for k in np.flatnonzero(np.array(e1['S']['changed']) > 0.01)]
log(f'live S units ({len(liveS)}): {liveS}' + ('' if liveS == LIVE_S else '  (differs from the E1 list in the script)'))
Q = {k: np.asarray(Q[k], np.float64) for k in ('mass', 'sum_pt', 'n_particles')}

BASE = dict(z=R['z'], S=R['S'])                       # (J, 16), (J, 64): the network's own values
UNITS = dict(z=list(range(16)), S=liveS)             # units the trees see
NAME = dict(z=lambda k: f'z{k}', S=lambda k: f'S{k}')

import jax, jax.numpy as jnp


def make_tail(nt):
    """the network's own computation downstream of an S or z override, as jedi_core.forward writes it (same layers,
    quantizers and fused weights): from S: d4 -> d5 -> d6 -> relu -> d7; from z: relu -> d7.  Used for the many control
    runs; checked below against run_jetwise (identical calls) on the identity and on every rule imposition."""
    L = nt.L; KB = {k: (jnp.asarray(v[0]), jnp.asarray(v[1])) for k, v in nt.KB.items()}

    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1]
        return L[k].activation(z) if act and L[k].activation is not None else z

    def from_S(S):
        return dense('d7', jax.nn.relu(dense('d6', dense('d5', dense('d4', S)), act=False)))

    def from_z(z):
        return dense('d7', jax.nn.relu(z))
    return dict(S=jax.jit(from_S), z=jax.jit(from_z))


TAIL = {'net': make_tail(net)}


def calls_rj(stage, own, nt=None):
    """calls of the evaluation jets through run_jetwise with stage fully overridden by own (n_eval, units)"""
    nt = nt or net
    return nt.run_jetwise(xe, {stage: (np.ones(own.shape[1]), own)})['logits'].argmax(1)


def calls_after(stage, V, idx=None, nt=None, check=False):
    """calls of the (10,000) evaluation jets when stage is replaced by V (n_eval, units) or, with idx, when the jets in idx
    get the rows V and the others keep their own values; nt = network (default: trained); check: also run_jetwise and
    count mismatches"""
    nt = nt or net; key = 'net' if nt is net else 'netU'
    own = (BASE if nt is net else BASEU)[stage][ev].copy()
    if idx is None: own = np.asarray(V, np.float64)
    else: own[idx] = V
    if not USE_TAIL: return calls_rj(stage, own, nt)
    lo = np.asarray(TAIL[key][stage](jnp.asarray(own, jnp.float32)), np.float64).reshape(len(own), -1); c = lo.argmax(1)
    if check:
        c2 = calls_rj(stage, own, nt); MISMATCH.append(int((c2 != c).sum()))
    return c


MISMATCH = []; USE_TAIL = True
for st in ('z', 'S'):
    c_id = calls_after(st, BASE[st][ev], check=True)
    c_rj = calls_rj(st, BASE[st][ev])
    log(f'check: {st} override with the network\'s own {st}: run_jetwise calls changed {(c_rj != ce).mean()*100:.3f} %; '
        f'tail from {st}: calls changed {(c_id != ce).mean()*100:.3f} %, tail vs run_jetwise mismatches {MISMATCH[-1]}')
    assert (c_rj != ce).mean() == 0.0
if sum(MISMATCH) > 0:
    USE_TAIL = False; log('tail differs from run_jetwise: every intervention below runs through run_jetwise')

# ------------------------------------------------------------------ untrained control network
netU = Jedi(64, untrained=True, seed=0)
RU = netU.run(x, keep=('S', 'z', 'logits')); callU = RU['logits'].argmax(1); cUe = callU[ev]
BASEU = dict(z=RU['z'], S=RU['S'])
TAIL['netU'] = make_tail(netU)
log(f'untrained network (seed 0): accuracy {(callU == y).mean()*100:.2f} %; calls on eval {np.bincount(cUe, minlength=5).tolist()}; '
    f'z units with nonzero variance {(RU["z"][fit].std(0) > 1e-9).sum()}, live-S units with nonzero variance {(RU["S"][fit][:, liveS].std(0) > 1e-9).sum()}')
for st in ('z', 'S'):
    cU_id = calls_after(st, BASEU[st][ev], nt=netU, check=True)
    log(f'check: untrained {st} override with its own {st}: calls changed {(cU_id != cUe).mean()*100:.3f} %, tail vs run_jetwise mismatches {MISMATCH[-1]}')

# ------------------------------------------------------------------ trees on grid-binned units
from sklearn.tree import DecisionTreeClassifier


ONOFF = 1e-6     # the on/off threshold: unit >= ONOFF  <=>  unit > 0 (relu(z) > 0; S > 0) on the fixed-point grid (step >= 2^-12)


def grids(V, positive_part=False):
    """per column: unique thresholds = 5-95 % quantiles (19 knots) + the on/off threshold ONOFF; positive_part (z):
    quantiles of the values > 0 only, since d7 reads relu(z) and every threshold below 0 is the same state for the network"""
    G = []
    for j in range(V.shape[1]):
        v = V[:, j]
        if positive_part: v = v[v > 0] if (v > 0).sum() >= 50 else v
        q = np.quantile(v, np.linspace(.05, .95, N_KNOTS)); q = q[np.abs(q) > 2 * ONOFF]
        G.append(np.unique(np.concatenate([q, [ONOFF]])))
    return G


def binify(V, G):
    return np.stack([np.searchsorted(G[j], V[:, j], side='right') for j in range(V.shape[1])], 1)


def leaves_of(tree, G):
    """every leaf as a list of windows {col: [lo, hi]} (lo <= unit < hi; None = unbounded) + the leaf id"""
    T = tree.tree_; out = []

    def walk(node, cond):
        if T.children_left[node] == -1:
            out.append((node, {k: list(v) for k, v in cond.items()})); return
        f, b = int(T.feature[node]), int(np.floor(T.threshold[node])); t = float(G[f][min(b, len(G[f]) - 1)])
        c = {k: list(v) for k, v in cond.items()}; w = c.setdefault(f, [None, None])
        w[1] = t if w[1] is None else min(w[1], t); walk(T.children_left[node], c)                 # bin <= b  <=>  unit < G[b]
        c = {k: list(v) for k, v in cond.items()}; w = c.setdefault(f, [None, None])
        w[0] = t if w[0] is None else max(w[0], t); walk(T.children_right[node], c)                # bin > b   <=>  unit >= G[b]
    walk(0, {}); return out


def inside(V, cond):
    m = np.ones(len(V), bool)
    for f, (lo, hi) in cond.items():
        if lo is not None: m &= V[:, f] >= lo
        if hi is not None: m &= V[:, f] < hi
    return m


def pc(mask, pos):
    """precision, coverage, F1 of a membership mask against positives"""
    n, tp = int(mask.sum()), int((mask & pos).sum()); P = tp / max(n, 1); C = tp / max(int(pos.sum()), 1)
    return P, C, (2 * P * C / (P + C) if P + C > 0 else 0.0), n, tp


def cond_str(cond, stage, G=None):
    units = UNITS[stage]; parts = []
    for f, (lo, hi) in sorted(cond.items()):
        nm = NAME[stage](units[f])
        if lo == ONOFF and hi is not None: parts.append(f'0 < {nm} < {hi:.3g}')
        elif lo == ONOFF: parts.append(f'{nm} > 0')
        elif hi == ONOFF and lo is not None: parts.append(f'{lo:.3g} <= {nm} <= 0')
        elif hi == ONOFF: parts.append(f'{nm} <= 0')
        elif lo is not None and hi is not None: parts.append(f'{lo:.3g} <= {nm} < {hi:.3g}')
        elif lo is not None: parts.append(f'{nm} >= {lo:.3g}')
        else: parts.append(f'{nm} < {hi:.3g}')
    return ' and '.join(parts)


def describe(stage, Vfit, Vev, calls_fit, calls_ev, label):
    """per class: the 3-deep tree, its best-F1 leaf (rule), precision / coverage on eval; all leaves"""
    G = grids(Vfit, positive_part=(stage == 'z')); Bf, Be = binify(Vfit, G), binify(Vev, G); out = {}
    for c in range(5):
        pos_f, pos_e = calls_fit == c, calls_ev == c
        if pos_f.sum() < MIN_LEAF or (~pos_f).sum() < MIN_LEAF:
            out[c] = dict(rule=None, note=f'class called on {int(pos_f.sum())} fit jets: no tree'); continue
        tr = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=0).fit(Bf, pos_f)
        LV = []
        for node, cond in leaves_of(tr, G):
            mf, me = inside(Vfit, cond), inside(Vev, cond)
            Pf, Cf, Ff, nfj, _ = pc(mf, pos_f); Pe, Ce, Fe, nej, _ = pc(me, pos_e)
            LV.append(dict(node=int(node), cond={int(k): v for k, v in cond.items()}, n_fit=nfj, n_eval=nej, prec_fit=Pf, cov_fit=Cf, f1_fit=Ff,
                           prec=Pe, cov=Ce, f1=Fe, positive=bool(tr.tree_.value[node][0].argmax() == 1)))
        best = max(LV, key=lambda d: d['f1_fit'])
        pred_e = tr.predict(Be); Pu, Cu, Fu, _, _ = pc(pred_e, pos_e)
        out[c] = dict(rule=best['cond'], rule_str=cond_str(best['cond'], stage), n_units=len(best['cond']), n_cond=sum((lo is not None) + (hi is not None) for lo, hi in best['cond'].values()),
                      precision=best['prec'], coverage=best['cov'], f1=best['f1'], prec_fit=best['prec_fit'], cov_fit=best['cov_fit'],
                      n_in_rule_eval=best['n_eval'], union_precision=Pu, union_coverage=Cu, union_f1=Fu, n_leaves=len(LV),
                      leaves=sorted(LV, key=lambda d: -d['f1_fit']))
        log(f'  {label} {CLASSES[c]}: rule [{out[c]["rule_str"]}]  precision {best["prec"]*100:.1f} %  coverage {best["cov"]*100:.1f} % '
            f'(fit {best["prec_fit"]*100:.1f} / {best["cov_fit"]*100:.1f}); tree union {Pu*100:.1f} / {Cu*100:.1f} %; {len(LV)} leaves')
    return out, G


# ------------------------------------------------------------------ interventions
def impose(stage, cond_units, values, src_calls, c, nt=None, which='other', check=False):
    """set cond_units of stage to values (per unit) on eval jets with src_calls != c ('other') or == c ('same'); returns
    the share of those jets called c afterwards and the per-source-call shares"""
    base = (BASE if (nt or net) is net else BASEU)[stage][ev]
    idx = np.flatnonzero(src_calls != c if which == 'other' else src_calls == c)
    V = base[idx].copy(); V[:, cond_units] = values
    ca = calls_after(stage, V, idx, nt, check=check)[idx]
    by = {CLASSES[k]: pct((ca[src_calls[idx] == k] == c).mean()) if (src_calls[idx] == k).sum() else None for k in range(5) if k != c} if which == 'other' else None
    return pct((ca == c).mean()), by, len(idx)


def run_level(stage, D, all_units_sets):
    """D = describe() output for the trained network at this level; all_units_sets: {name: list of unit indices} ceilings"""
    base = BASE[stage]; bf = base[fit]; cf = call[fit]; out = {}
    units_pool = np.array(UNITS[stage])
    for c in range(5):
        d = D[c]
        if d['rule'] is None: out[c] = dict(note=d['note']); continue
        cond = d['rule']; cu = [UNITS[stage][f] for f in cond]                      # unit indices in the layer
        in_rule_f = inside(bf[:, UNITS[stage]], cond)
        med_in = np.median(bf[(cf == c) & in_rule_f][:, cu], 0)                      # in-rule medians of jets called c
        med_c = np.median(bf[cf == c], 0)                                            # class-c medians, all units
        med_o = np.median(bf[cf != c], 0)                                            # medians of jets called otherwise
        r = dict(rule_units=cu, med_in_rule=med_in.tolist(), med_class=med_c[cu].tolist(), med_other=med_o[cu].tolist(),
                 n_fit_in_rule_called_c=int(((cf == c) & in_rule_f).sum()))
        r['rule'], r['rule_by_source'], r['n_other'] = impose(stage, cu, med_in, ce, c, check=True)
        r['rule_classmed'], _, _ = impose(stage, cu, med_c[cu], ce, c)
        # every positive leaf of the tree (predicted c, >= 1 % coverage on fit) imposed the same way
        r['leaves_imposed'] = []
        for lf in d['leaves']:
            if not lf['positive'] or lf['cov_fit'] < 0.01: continue
            lcu = [UNITS[stage][int(f)] for f in lf['cond']]; lin = inside(bf[:, UNITS[stage]], {int(k): v for k, v in lf['cond'].items()})
            if ((cf == c) & lin).sum() == 0: continue
            lmed = np.median(bf[(cf == c) & lin][:, lcu], 0)
            sw, _, _ = impose(stage, lcu, lmed, ce, c)
            r['leaves_imposed'].append(dict(rule=cond_str({int(k): v for k, v in lf['cond'].items()}, stage), precision=lf['prec'], coverage=lf['cov'], switched=sw))
        # dose
        idx = np.flatnonzero(ce != c); dose = {}
        for f_ in DOSES:
            V = base[ev][idx].copy(); V[:, cu] = V[:, cu] + f_ * (med_in - V[:, cu])
            dose[str(f_)] = pct((calls_after(stage, V, idx)[idx] == c).mean())
        r['dose'] = dose
        # random controls
        K = len(cu); rg = np.random.default_rng(1000 + c)
        rand_all, rand_out, units_all, units_out = [], [], [], []
        others = np.array([u for u in units_pool if u not in cu])
        for dd in range(N_RAND):
            ua = rg.choice(units_pool, K, replace=False); rand_all.append(impose(stage, list(ua), med_c[ua], ce, c)[0]); units_all.append([int(u) for u in ua])
            if len(others) >= K:
                uo = rg.choice(others, K, replace=False); rand_out.append(impose(stage, list(uo), med_c[uo], ce, c)[0]); units_out.append([int(u) for u in uo])
        n_hit = sum(bool(set(u) & set(cu)) for u in units_all)
        r['random'] = dict(mean=float(np.mean(rand_all)), min=float(np.min(rand_all)), max=float(np.max(rand_all)), draws=rand_all, units=units_all, K=K,
                           n_draws_with_rule_unit=n_hit)
        r['random_outside_rule'] = dict(mean=float(np.mean(rand_out)), min=float(np.min(rand_out)), max=float(np.max(rand_out)), draws=rand_out, units=units_out) if rand_out else None
        # AND test: each single rule unit and each pair imposed alone (same in-rule medians); conjunction vs its parts
        r['subsets'] = {}
        for k_ in range(1, len(cu)):
            for sub in combinations(range(len(cu)), k_):
                us_ = [cu[j] for j in sub]
                r['subsets'][' & '.join(NAME[stage](u) for u in us_)] = impose(stage, us_, med_in[list(sub)], ce, c)[0]
        singles = [r['subsets'][NAME[stage](u)] for u in cu] if len(cu) > 1 else [r['rule']]
        r['and_test'] = dict(sum_of_singles=round(float(sum(singles)), 2), max_single=round(float(max(singles)), 2),
                             max_pair=(round(float(max(v for kk, v in r['subsets'].items() if kk.count('&') == 1)), 2) if len(cu) == 3 else None))
        # ceilings
        r['ceiling'] = {}
        for nm, us in all_units_sets.items():
            r['ceiling'][nm], _, _ = impose(stage, list(us), med_c[us], ce, c, check=True)
        # necessity: jets called c inside the rule -> rule units at the other-call medians
        in_rule_e = inside(base[ev][:, UNITS[stage]], cond); src = np.where((ce == c) & in_rule_e, c, -1)
        r['necessity'] = {}
        r['necessity']['rule_to_other_med'], _, r['n_called_c_in_rule'] = impose(stage, cu, med_o[cu], src, c, which='same')
        us = list(all_units_sets[list(all_units_sets)[0]])
        r['necessity']['all_units_to_other_med'], _, _ = impose(stage, us, med_o[us], src, c, which='same')
        # untrained network: trained rule's units; its own calls
        nU_c = int((callU[fit] == c).sum()); rU = dict(n_fit_called_c=nU_c, n_other=int((cUe != c).sum()))
        if (cUe != c).sum() > 0:
            rU['trained_medians'], _, _ = impose(stage, cu, med_in, cUe, c, nt=netU)
            if nU_c >= 20:
                mU = np.median(BASEU[stage][fit][callU[fit] == c], 0)
                rU['own_medians'], _, _ = impose(stage, cu, mU[cu], cUe, c, nt=netU)
                rU['own_ceiling'], _, _ = impose(stage, us, mU[us], cUe, c, nt=netU)
            # rank-matched: each rule unit set to the untrained unit's own quantile at the rank the trained in-rule median
            # has among the trained fit jets (same position in the distribution, the untrained network's scale)
            qv = [float((bf[:, u] < med_in[j]).mean()) for j, u in enumerate(cu)]
            vU = np.array([np.quantile(BASEU[stage][fit][:, u], q) for u, q in zip(cu, qv)])
            rU['rank_matched'], _, _ = impose(stage, cu, vU, cUe, c, nt=netU)
            rU['rank_matched_values'] = vU.tolist(); rU['rank_of_trained_median'] = qv
            rU['trained_ceiling'], _, _ = impose(stage, us, med_c[us], cUe, c, nt=netU)    # all units at the trained class medians
        r['untrained'] = rU
        ceil0 = r['ceiling'][list(all_units_sets)[0]]
        r['gap'] = ceil0 - r['rule']
        out[c] = r
        log(f'  {stage} {CLASSES[c]}: imposed on {r["n_other"]} jets called otherwise -> called {CLASSES[c]}: rule {r["rule"]:.1f} % '
            f'(class medians {r["rule_classmed"]:.1f}); dose {[dose[str(f_)] for f_ in DOSES]}; random K={K}: {r["random"]["mean"]:.1f} % '
            f'[{r["random"]["min"]:.1f}-{r["random"]["max"]:.1f}] (outside rule {round(r["random_outside_rule"]["mean"], 2) if r["random_outside_rule"] else None}); '
            f'subsets {r["subsets"]}; and_test {r["and_test"]}; '
            f'other leaves {[(l["rule"], round(l["precision"]*100), round(l["coverage"]*100), l["switched"]) for l in r["leaves_imposed"]]}; '
            f'ceiling {r["ceiling"]}; gap {r["gap"]:.1f}; necessity: {r["n_called_c_in_rule"]} jets, rule->other med still c '
            f'{r["necessity"]["rule_to_other_med"]:.1f} % (all units {r["necessity"]["all_units_to_other_med"]:.1f}); untrained {rU}; '
            f'by source {r["rule_by_source"]}  [{time.time()-t0:.0f} s]')
    return out


# ------------------------------------------------------------------ poles and windows in physical units
def poles(stage, units):
    out = {}
    m_, pt_, n_ = Q['mass'], Q['sum_pt'], Q['n_particles']
    for u in sorted(set(units)):
        v = BASE[stage][:, u]; d = {}
        for side, msk in (('off', v <= 0), ('on', v > 0)):
            if msk.sum() == 0: d[side] = dict(n=0); continue
            d[side] = dict(n=int(msk.sum()), truth=[pct((y[msk] == k).mean()) for k in range(5)], calls=[pct((call[msk] == k).mean()) for k in range(5)],
                           mass_med=float(np.median(m_[msk])), sum_pt_med=float(np.median(pt_[msk])), n_med=float(np.median(n_[msk])))
        d['known'] = KNOWN.get(u) if stage == 'z' else None
        out[NAME[stage](u)] = d
    return out


def window_ranges(stage, D):
    """per rule window: mass 25/50/75 % of the jets inside it (and outside), on all jets"""
    out = {}
    for c in range(5):
        if D[c]['rule'] is None: continue
        for f, (lo, hi) in D[c]['rule'].items():
            u = UNITS[stage][f]; v = BASE[stage][:, u]; m = np.ones(J, bool)
            if lo is not None: m &= v >= lo
            if hi is not None: m &= v < hi
            out[f'{CLASSES[c]}:{NAME[stage](u)}'] = dict(lo=lo, hi=hi, n=int(m.sum()), mass_q=[float(np.percentile(Q['mass'][m], p)) for p in (25, 50, 75)],
                                                       mass_q_out=[float(np.percentile(Q['mass'][~m], p)) for p in (25, 50, 75)] if (~m).sum() else None,
                                                       sum_pt_med=float(np.median(Q['sum_pt'][m])), n_med=float(np.median(Q['n_particles'][m])), known=KNOWN.get(u) if stage == 'z' else None)
    return out


RES = dict(N=J, per_class=PER, seed=SEED, split='random 2/3 fit, 1/3 eval (rng 0)', n_fit=len(fit), n_eval=len(ev), acc_check_seed0=acc0,
           acc_net_all=acc_all, acc_net_eval=acc_ev, calls_eval=np.bincount(ce, minlength=5).tolist(), classes=CLASSES, liveS=liveS,
           tree=dict(depth=DEPTH, min_samples_leaf=MIN_LEAF, knots=N_KNOTS, grid='5-95 % quantiles + 0'), doses=list(DOSES), n_random=N_RAND,
           untrained=dict(acc=float((callU == y).mean()), calls_eval=np.bincount(cUe, minlength=5).tolist(),
                          z_varying=int((RU['z'][fit].std(0) > 1e-9).sum()), liveS_varying=int((RU['S'][fit][:, liveS].std(0) > 1e-9).sum())))

for stage in ('z', 'S'):
    cols = UNITS[stage]
    log(f'--- step 1 ({stage}): 3-deep trees on {len(cols)} units, grid thresholds')
    D, G = describe(stage, BASE[stage][fit][:, cols], BASE[stage][ev][:, cols], call[fit], ce, stage)
    log(f'--- step 1 control ({stage}): same trees on the untrained network\'s own {stage} and calls')
    DU, _ = describe(stage, BASEU[stage][fit][:, cols], BASEU[stage][ev][:, cols], callU[fit], cUe, f'untrained {stage}')
    log(f'--- step {"2" if stage == "z" else "3"} ({stage}): imposing the rules')
    sets = {'all16': list(range(16))} if stage == 'z' else {'live19': liveS, 'all64': list(range(64))}
    I = run_level(stage, D, sets)
    rule_units = [u for c in range(5) if D[c]['rule'] for f in D[c]['rule'] for u in [cols[f]]]
    RES[stage] = dict(grid={NAME[stage](cols[j]): [round(float(t), 4) for t in G[j]] for j in range(len(cols))},
                      description={CLASSES[c]: D[c] for c in range(5)}, description_untrained={CLASSES[c]: {k: v for k, v in DU[c].items() if k != 'leaves'} for c in range(5)},
                      intervention={CLASSES[c]: I[c] for c in range(5)}, poles=poles(stage, rule_units), windows=window_ranges(stage, D))
    json.dump(RES, open(OUT / ('e8_smoke.json' if SMOKE else 'e8_conjunctions.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

# ------------------------------------------------------------------ summary
log('--- summary: class | level | rule | precision / coverage | switched: rule / random (mean) / untrained / ceiling | gap')
summ = []
for stage in ('z', 'S'):
    for c in range(5):
        d, i = RES[stage]['description'][CLASSES[c]], RES[stage]['intervention'][CLASSES[c]]
        if d['rule'] is None: continue
        ceil = i['ceiling'][list(i['ceiling'])[0]]
        u = i['untrained'].get('own_medians', i['untrained'].get('trained_medians'))
        row = dict(cls=CLASSES[c], level=stage, rule=d['rule_str'], precision=pct(d['precision']), coverage=pct(d['coverage']), switched=i['rule'],
                   random=round(i['random']['mean'], 2), untrained=u, ceiling=ceil, gap=round(ceil - i['rule'], 2), necessity_still_c=i['necessity']['rule_to_other_med'])
        summ.append(row)
        log(f'  {CLASSES[c]:2s} {stage:2s} [{d["rule_str"]}]  {row["precision"]:.1f} / {row["coverage"]:.1f} %  switched {row["switched"]:.1f} / {row["random"]:.1f} / {u} / {ceil:.1f} %  gap {row["gap"]:.1f}')
RES['summary'] = summ
RES['tail_vs_run_jetwise_mismatches'] = dict(n_checked_runs=len(MISMATCH), total_mismatched_calls=int(sum(MISMATCH)))
log(f'check: tail vs run_jetwise on {len(MISMATCH)} runs: {sum(MISMATCH)} mismatched calls in total')
RES['time_s'] = time.time() - t0
json.dump(RES, open(OUT / ('e8_smoke.json' if SMOKE else 'e8_conjunctions.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
log(f'done in {time.time()-t0:.0f} s')
