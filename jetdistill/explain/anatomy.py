"""What each neuron, its if-statements, groups of jets and each class score look at, shown on the jets themselves
(the 60,000 explanation jets):
  jet image  = pT share of the particles in (Δη, Δφ), each jet rotated so that its 2nd-hardest particle points along +Δη
               and flipped so that its 3rd-hardest particle lies above (2- and 3-prong structure becomes visible);
  profile    = share of the jet pT at each distance ΔR from the jet axis;  pT sharing = mean pT share of the hardest particles;
  plus the true-class mix and the mean mass, width and total pT.
Per neuron: jets where it is off / on and low / on and high; every if-statement (the jets that pass it, what it adds);
groups of jets formed only from what each if-statement adds (k-means, 10 groups, split further where mixed); regimes (a
depth-3 tree on the neuron's own observables). Per class score: what each neuron adds for jets of each true class."""
import numpy as np
from ..config import CLASSES as CL
from .. import formula as F
from .pack import neurons_and_logits
from ..observables import library

NB = 16; RB = [0, .025, .05, .1, .15, .2, .3, .4, .6, 1.0]
CTX = {}


def term(t, Q):
    return F.basis(t, Q)


def aligned_images(X):
    pt, eta, phi = X[..., 0].astype(np.float64), X[..., 1].astype(np.float64), X[..., 2].astype(np.float64)
    z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-9)
    th = np.arctan2(phi[:, 1], eta[:, 1]); c, s = np.cos(-th)[:, None], np.sin(-th)[:, None]
    e2, p2 = eta * c - phi * s, eta * s + phi * c
    flip = np.where(p2[:, 2] < 0, -1.0, 1.0)[:, None]; p2 = p2 * flip
    ix = np.clip(((e2 + 0.8) / 1.6 * NB).astype(int), 0, NB - 1); iy = np.clip(((p2 + 0.8) / 1.6 * NB).astype(int), 0, NB - 1)
    dr = np.hypot(eta, phi); rb = np.clip(np.searchsorted(RB, dr, side='right') - 1, 0, len(RB) - 2)
    return z, ix, iy, rb, pt


def profile(sel, z, ix, iy, rb, Q, y):
    """summary of a set of jets (boolean mask)"""
    n = int(sel.sum())
    if n < 20: return None
    img = np.zeros((NB, NB)); zz = z[sel]; np.add.at(img, (iy[sel].ravel(), ix[sel].ravel()), zz.ravel()); img /= n
    rad = np.zeros(len(RB) - 1); np.add.at(rad, rb[sel].ravel(), zz.ravel()); rad /= n
    zs = -np.sort(-zz, 1)[:, :min(8, zz.shape[1])].mean(0)
    return dict(n=n, frac=round(n / len(y), 4), classes=[round(float((y[sel] == c).mean()), 4) for c in range(5)],
                image=[[round(float(v), 5) for v in row] for row in img], radial=[round(float(v), 4) for v in rad], zshare=[round(float(v), 4) for v in zs],
                means={q: round(float(Q[q][sel].mean()), 4) for q in ('mass', 'width', 'sum_pt') if q in Q},
                pt_rank=[round(float(v), 2) for v in CTX['PT'][0][sel][:, :min(8, CTX['PT'][0].shape[1])].mean(0)],                     # mean pT [GeV] of the hardest particles
                pt_spec=[round(float(v), 4) for v in np.histogram(np.clip(CTX['PT'][0][sel][CTX['PT'][0][sel] > 0], 1, 999), CTX['PT'][1])[0] / n],  # particles per jet in each pT bin
                jetpt_hist=[round(float(v), 5) for v in np.histogram(np.clip(CTX['PT'][2][sel], CTX['PT'][3][0], CTX['PT'][3][-1]), CTX['PT'][3])[0] / len(y)],   # share of ALL jets per bin
                leadpt_hist=[round(float(v), 5) for v in np.histogram(np.clip(CTX['PT'][4][sel], CTX['PT'][5][0], CTX['PT'][5][-1]), CTX['PT'][5])[0] / len(y)])


def hist(Q, y, q, passing=None, nb=30):
    """the quantity's distribution for each true class (1st-99th percentile range, each class normalized to 1), and for all
    jets and for the jets that pass the if-statement (both as shares of ALL jets, so the passing part is drawn to scale)"""
    x = Q[q].astype(np.float64); lo, hi = np.quantile(x, [.01, .99])
    if hi <= lo: hi = lo + 1e-6
    e = np.linspace(lo, hi, nb + 1); xc = np.clip(x, lo, hi); out = dict(edges=[round(float(v), 6) for v in e],
        per_class=[[round(float(v), 4) for v in np.histogram(xc[y == c], e)[0] / max((y == c).sum(), 1)] for c in range(5)],
        all=[round(float(v), 5) for v in np.histogram(xc, e)[0] / len(x)])
    if passing is not None: out['passing'] = [round(float(v), 5) for v in np.histogram(xc[passing], e)[0] / len(x)]
    return out


def group_object(sel, nr, terms, fires, adds, h, Q, y, PRED, wrow, z, ix, iy, rb, image=True):
    """one set of jets evaluated as an object: per if-statement pass rate / mean added / typical range, the neuron value,
    formula decision + accuracy on these jets, what the neuron adds to each class score, jet profile, a worked example"""
    pr = profile(sel, z, ix, iy, rb, Q, y)
    if not pr: return None
    KT = len(terms); pf = PRED[sel]; ii = np.flatnonzero(sel); r = ii[np.argmin(np.abs(h[ii] - np.median(h[ii])))]
    example = dict(start=round(float(nr['intercept']), 4), values=[{'q': t['q'], 'v': round(float(Q[t['q']][r]), 5), **({'q2': t['q2'], 'v2': round(float(Q[t['q2']][r]), 5)} if t.get('q2') else {})} for t in terms],
                   adds=[round(float(adds[r, i]), 4) for i in range(KT)], rest=0.0, z=round(float(nr['intercept'] + adds[r].sum()), 4), neuron=round(float(h[r]), 4), n_rest=0,
                   true_class=CL[y[r]], formula_class=CL[PRED[r]])
    keys = ('n', 'frac', 'classes', 'means', 'radial', 'pt_rank', 'zshare', 'pt_spec', 'jetpt_hist', 'leadpt_hist') + (('image',) if image else ())
    return dict(pass_rate=[round(float(v), 3) for v in fires[sel].mean(0)], pattern=[bool(v >= 0.5) for v in fires[sel].mean(0)],
                mean_neuron=round(float(h[sel].mean()), 3), neuron_on=round(float((h[sel] > 0).mean()), 3), example=example,
                added_by_each=[round(float(adds[sel, i].mean()), 3) for i in range(KT)],
                added_p10_p90=[[round(float(np.quantile(adds[sel, i], .1)), 3), round(float(np.quantile(adds[sel, i], .9)), 3)] for i in range(KT)],
                formula_decides=[round(float((pf == c).mean()), 3) for c in range(5)], formula_right=round(float((pf == y[sel]).mean()), 3),
                adds_to_scores=[round(float((h[sel] * wrow[c]).mean()), 3) for c in range(5)], **{k2: pr[k2] for k2 in keys if k2 in pr})


def split_by_mixed(sel, fires, adds, depth, obj, min_n, share_min=0.15, max_depth=3):
    """split a set of jets by its MIXED if-statements (on for 10-90 % of them): 2-means on what those if-statements add;
    recurse while the mixed ones carry >= share_min of the neuron's average |input| and the set has >= min_n jets"""
    from sklearn.cluster import KMeans
    ii = np.flatnonzero(sel); pr = fires[ii].mean(0); w = np.abs(adds[ii]).mean(0)
    mixed = np.flatnonzero((pr > 0.1) & (pr < 0.9) & (w > 0)); share = float(w[mixed].sum() / max(w.sum(), 1e-12))
    if depth >= max_depth or len(ii) < 2 * min_n or share < share_min or len(mixed) == 0: return []
    lab = KMeans(n_clusters=2, n_init=4, random_state=0).fit(adds[ii][:, mixed]).labels_; out = []
    if min((lab == 0).sum(), (lab == 1).sum()) < min_n // 2: return []          # split only if BOTH halves are big enough: every jet stays in a group
    for k in range(2):
        s2 = np.zeros(len(sel), bool); s2[ii[lab == k]] = True
        o = obj(s2)
        if not o: return []
        pr2 = fires[s2].mean(0)
        o['split_on'] = [dict(index=int(i), passes=round(float(pr2[i]), 3)) for i in mixed[np.argsort(-np.abs(pr2[mixed] - pr[mixed]))][:4]]
        o['mixed_share'] = round(share, 3); o['children'] = split_by_mixed(s2, fires, adds, depth + 1, obj, min_n, share_min, max_depth); out.append(o)
    return out


def neuron_regimes(nr, h, Q, y, pred, wrow, z, ix, iy, rb, depth=3):
    """the neuron's REGIMES with explicit boundaries: a depth-3 regression tree for the neuron's value on the quantities its
    if-statements use (all of them); each leaf = a regime: its conditions, share of jets, value (mean, p10-p90), class mix,
    formula accuracy, what the neuron adds to each class score there.  r2 = how well the regimes reproduce the neuron."""
    from sklearn.tree import DecisionTreeRegressor
    qs = sorted({t['q'] for t in nr['terms']} | {t['q2'] for t in nr['terms'] if t.get('q2')})
    X = np.stack([Q[q].astype(np.float64) for q in qs], 1)
    m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(50, int(0.02 * len(h))), random_state=0).fit(X, h)
    T = m.tree_; leaf = m.apply(X); r2 = float(1 - ((h - m.predict(X)) ** 2).mean() / max(h.var(), 1e-12)); out = []
    def walk(i, conds):
        if T.children_left[i] == -1:
            sel = leaf == i; hv = h[sel]; pr = profile(sel, z, ix, iy, rb, Q, y)
            out.append(dict(conditions=conds, frac=round(float(sel.mean()), 4), mean=round(float(hv.mean()), 3), p10=round(float(np.quantile(hv, .1)), 3),
                            p90=round(float(np.quantile(hv, .9)), 3), on=round(float((hv > 0).mean()), 3), classes=[round(float((y[sel] == c).mean()), 4) for c in range(5)],
                            formula_right=round(float((pred[sel] == y[sel]).mean()), 3), adds_to_scores=[round(float(hv.mean() * wrow[c]), 3) for c in range(5)],
                            means=pr['means'] if pr else None, pt_rank=pr['pt_rank'] if pr else None)); return
        q = qs[T.feature[i]]; t = float(T.threshold[i])
        walk(T.children_left[i], conds + [[q, '<=', round(t, 5)]]); walk(T.children_right[i], conds + [[q, '>', round(t, 5)]])
    walk(0, []); out.sort(key=lambda r: -r['mean']); return out, round(r2, 3)


def readable(t):
    p = lambda q, k, th: q if k == 'lin' else (f'{q} > {th:.3g}' if k == 'gt' else f'{q} < {th:.3g}')
    return p(t['q'], t['kind'], t.get('t')) + (' and ' + p(t['q2'], t['kind2'], t.get('t2')) if t.get('q2') else '')


def build(f, normalized, n, last, jets, X):
    """jets: dict(Q=observables, y=true class) of the explanation jets; X: their particles (J, n, 3)"""
    Q = dict(jets['Q']); y = jets['y']
    z, ix, iy, rb, PT = aligned_images(X); PTS = -np.sort(-PT, 1); PBIN = np.logspace(0, 3, 22)
    JPT = PT.sum(1); LPT = PTS[:, 0]; JB = np.linspace(*np.quantile(JPT, [.005, .995]), 31); LB = np.linspace(*np.quantile(LPT, [.005, .995]), 31)
    CTX['PT'] = (PTS, PBIN, JPT, JB, LPT, LB)
    allp = profile(np.ones(len(y), bool), z, ix, iy, rb, Q, y)
    desc = {k: o.desc for k, o in library(n).items()}
    H, Lg = neurons_and_logits(f, Q, last); PRED = Lg.argmax(1); K7 = last[0]; normT = {nr['neuron']: nr for nr in normalized[0]}
    neurons = []
    for nr in f:
        j = nr['neuron']; h = H[:, j]; on = h > 0; hi = on & (h >= np.quantile(h[on], 0.75)) if on.any() else on
        groups = dict(off=profile(~on, z, ix, iy, rb, Q, y), on_low=profile(on & ~hi, z, ix, iy, rb, Q, y), on_high=profile(hi, z, ix, iy, rb, Q, y))
        terms = sorted(normT[j]['terms'], key=lambda t: -abs(t['share'])); tl = []
        for k, t in enumerate(terms):
            v = term(t, Q); ps = v != 0; add = t['coef'] * v; pr = profile(ps, z, ix, iy, rb, Q, y)
            if pr and k >= 6: pr.pop('image')                              # jet images only for the 6 strongest (page size)
            tl.append(dict(test=readable(t), share=round(float(t['share']), 3), passes=pr, coef=float(t['coef']), kind=t['kind'], t=t.get('t'), q=t['q'],
                           q2=t.get('q2'), kind2=t.get('kind2'), t2=t.get('t2'), meaning=desc.get(t['q'], t['q']), meaning2=desc.get(t.get('q2'), t.get('q2')) if t.get('q2') else None,
                           mean_added_when_passing=round(float(add[ps].mean()), 4) if ps.any() else 0.0,
                           added_p10_p90=[round(float(np.quantile(add[ps], .1)), 4), round(float(np.quantile(add[ps], .9)), 4)] if ps.sum() > 10 else None,
                           hist=hist(Q, y, t['q'], ps), hist2=hist(Q, y, t['q2'], ps) if t.get('q2') else None,
                           q2_median_when_passing=round(float(np.median(Q[t['q2']][ps])), 5) if t.get('q2') and ps.any() else None))
        if not terms:                                                     # a neuron without terms is a constant: nothing to group
            neurons.append(dict(neuron=j, intercept=float(nr['intercept']), groups=groups, terms=tl, combo_tests=[], combos=[], regimes=[], regime_r2=None)); continue
        fires = np.stack([term(t, Q) != 0 for t in terms], 1); adds = np.stack([t['coef'] * term(t, Q) for t in terms], 1)
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=10, n_init=4, random_state=0).fit(adds).labels_; combos = []
        obj = lambda sel, im=True: group_object(sel, nr, terms, fires, adds, h, Q, y, PRED, K7[j], z, ix, iy, rb, image=im)
        for k in np.argsort(-np.bincount(lab, minlength=10)):
            sel = lab == k; o = obj(sel)
            if not o: continue
            o['children'] = split_by_mixed(sel, fires, adds, 0, lambda s2: obj(s2, False), max(150, int(0.0025 * len(y))), share_min=0.05, max_depth=6); combos.append(o)
        regimes, r2 = neuron_regimes(nr, h, Q, y, PRED, K7[j], z, ix, iy, rb)
        neurons.append(dict(neuron=j, intercept=float(nr['intercept']), groups=groups, terms=tl, combo_tests=[t['test'] for t in tl], combos=combos, regimes=regimes, regime_r2=r2))
    comp = {c: sorted([dict(neuron=nn['neuron'], regime=k, conditions=r['conditions'], frac=r['frac'], adds=r['adds_to_scores'][c], weighted=round(r['frac'] * r['adds_to_scores'][c], 4),
                            classes=r['classes'], value=r['mean']) for nn in neurons for k, r in enumerate(nn['regimes']) if abs(r['adds_to_scores'][c]) > 1e-4], key=lambda x: -abs(x['weighted'])) for c in range(5)}
    scores = []
    for c in range(5):
        top = Lg[:, c] >= np.quantile(Lg[:, c], 0.9)
        scores.append(dict(cls=CL[c], bias=float(last[1][c]), contrib_by_true_class=[[round(float((H[y == k, j] * K7[j, c]).mean()), 4) for j in range(H.shape[1])] for k in range(5)],
                           top10=profile(top, z, ix, iy, rb, Q, y), composition=comp[c][:12]))
    return dict(formula=F.tag(f), n_particles=n, n_jets=len(y), image_range=0.8, image_bins=NB, radial_bins=RB, pt_bins=[round(float(v), 3) for v in PBIN],
                jetpt_bins=[round(float(v), 1) for v in JB], leadpt_bins=[round(float(v), 1) for v in LB], all=allp, neurons=neurons, class_scores=scores)
