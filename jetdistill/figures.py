"""The figures of the slides, from the results of this repository (whole test file unless stated).

  motivation      four network neurons against the observable each follows most closely (density, bin means, rank correlation)
  hinge           one network neuron against one observable with the best single hinge a·max(0, x − t) (per jet, test jets)
  linear          four network neurons with the straight line fitted where the neuron is on (and inside a window of the observable)
  mars_steps      step 1 in one observable: every candidate hinge and the least-squares best, first and second hinge
  mars_hinges     the fit of one neuron after 1, 2 and 4 hinges
  fidelity        step 2 without and with the neuron-closeness term R (λ = 0, 0.01): one neuron against ΣzΔR², R² of every neuron
  massnet         network neurons against the jet mass (the network never receives it)
  mass_nomass     network and formula neurons against the jet mass (default: a formula without mass observables)
  neuron_hist     formula neuron against network neuron, jet by jet, for step 2 with λ = 0 or 0.01
  setups          agreement with the network for each setup (formula tuned on the network and its smaller version)
  targets         agreement with the network for each tuning target
  control         R² of step 1 for the trained and the untrained network; orientation share (analysis/control.py)
  inputs          R² of step 1 on the observables, of step 1 and of boosted trees on the raw inputs (trained network)
  convergence     agreement during 3000 steps of step 2, training and validation jets (analysis/convergence.py)
  term            one if-statement of one neuron: what it adds, and the observable's distribution per class
  equations       the definitions on the jet-quantities slide
Output: results/figures/<name>.png
Usage: python -m jetdistill.figures <name> [N] ...   (python -m jetdistill.figures all: every figure whose results exist)"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from . import pipeline as P, formula as F, tuning, mars
from .config import RESULTS, CLASSES, FIDELITY_LAMBDA, N_TUNE_FIT, MASSES, SETUPS
from .network import Network
from .observables import symbol

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'cm'})
OUT = RESULTS / 'figures'
SETUPS_NO_MASS = {k: bool(v.no_mass) for k, v in SETUPS.items()}
NAVY, GREEN, GRAY, LIGHT, RED, BLUE, ORANGE = '#1D2433', '#2f855a', '#A7B0BD', '#7FB89A', '#c53030', '#2b6cb0', '#c05621'
CCOL = ['#2f855a', '#2b6cb0', '#c05621', '#6b46c1', '#c53030']
AXIS = {'mass': 'jet mass [GeV]', 'girth': r'$\sum_i p_{T,i}\,\Delta R_i\,/\,\sum_i p_{T,i}$', 'girth2': r'$\sum_i p_{T,i}\,\Delta R_i^2\,/\,\sum_i p_{T,i}$',
        'width': r'$\sum_i p_{T,i}\,\Delta R_i^2\,/\,\sum_i p_{T,i}$', 'e2': r'$e_2=\sum_{i<k} z_i z_k\,\Delta R_{ik}$', 'sum_pt': r'$\sum_i p_{T,i}$ [GeV]',
        'log_sum_pt': r'$\ln(\sum_i p_{T,i}\,/\,\mathrm{GeV})$', 'z_dr_0p2_0p4': r'pT fraction at $0.2 \leq \Delta R < 0.4$'}


def axis(q): return AXIS.get(q, symbol(q))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(OUT / f'{name}.png', facecolor='white'); plt.close(fig); print(OUT / f'{name}.png')


def binned(x, h, lo, hi, nb=35, min_count=50):
    e = np.linspace(lo, hi, nb + 1); c = .5 * (e[1:] + e[:-1]); k = np.digitize(x, e) - 1
    return c, np.array([h[k == i].mean() if (k == i).sum() > min_count else np.nan for i in range(nb)])


def hexpanel(ax, x, h, j, lab):
    lo, hi = np.quantile(x, [.002, .995]); top = np.quantile(h, .998)
    ax.hexbin(np.clip(x, lo, hi), h, gridsize=55, bins='log', cmap='Blues', mincnt=1, extent=(lo, hi, 0, top))
    ax.set_xlabel(lab, fontsize=13); ax.set_xlim(lo, hi); ax.set_ylim(-0.2, top); ax.tick_params(labelsize=11)
    return lo, hi


# ---- motivation ----
def motivation(n=8, panels=((10, 'girth'), (3, 'z_dr_0p2_0p4'), (5, 'e2'), (2, 'mass'))):
    from scipy.stats import spearmanr
    T = P.jets(n, 'full_test'); fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.3), dpi=150)
    for ax, (j, q) in zip(axs, panels):
        x, h = T['Q'][q], T['H'][:, j].astype(float); lo, hi = hexpanel(ax, x, h, j, axis(q))
        c, mu = binned(x, h, lo, hi); ax.plot(c, mu, 'o', color=NAVY, ms=4.5, label='mean in each bin'); rho = spearmanr(x[::5], h[::5]).statistic
        ax.set_title(f'network neuron {j}   (rank correlation {rho:+.2f})', fontsize=14); ax.legend(fontsize=11, frameon=False, loc='upper left' if rho > 0 else 'upper right')
    axs[0].set_ylabel('neuron value', fontsize=13); save(fig, f'motivation_n{n}')


def linear(n=64, panels=((12, 'mass'), (9, 'width'), (6, 'mass'), (2, 'log_sum_pt'))):
    """per panel: the straight line with the best R² among (a) all jets with the neuron on, (b) jets with the neuron on inside a
    window of 3-6 deciles of the observable (the window is shaded)"""
    T = P.jets(n, 'full_test'); fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.4), dpi=150)
    def fit(x, h):
        if len(h) < 2000 or np.ptp(x) == 0: return -1, 0, 0
        A = np.stack([x, np.ones_like(x)], 1); c = np.linalg.lstsq(A, h, rcond=None)[0]; return 1 - ((h - A @ c) ** 2).sum() / max(((h - h.mean()) ** 2).sum(), 1e-12), c[0], c[1]
    for ax, (j, q) in zip(axs, panels):
        x, h = T['Q'][q], T['H'][:, j].astype(float); on = h > 0; qs = np.quantile(x, np.linspace(0, 1, 11))
        best = max([(fit(x[on], h[on]), None, None)] + [(fit(x[s], h[s]), qs[a], qs[a + w]) for a in range(8) for w in (3, 4, 5, 6) if a + w <= 10
                                                        for s in [on & (x >= qs[a]) & (x <= qs[a + w])]], key=lambda r: r[0][0])
        (r2, sl, ic), a, b = best; lo, hi = hexpanel(ax, x, h, j, axis(q)); a_, b_ = (lo if a is None else a), (hi if b is None else b)
        g = np.linspace(max(a_, lo), min(b_, hi), 50); ax.plot(g, sl * g + ic, color=RED, lw=2.6, label=f'straight line, R² = {r2:.2f}')
        if a is not None: ax.axvspan(max(a, lo), min(b, hi), color=RED, alpha=0.06)
        ax.set_title(f'network neuron {j}' + (f'  ({a:.3g} < x < {b:.3g}, neuron on)' if a is not None else '  (jets with neuron on)'), fontsize=13)
        ax.legend(fontsize=12, frameon=False, loc='upper right' if sl < 0 else 'upper left')
    axs[0].set_ylabel('neuron value', fontsize=13); save(fig, f'linear_n{n}')


def hinge(n=8, j=3, q='girth'):
    T = P.jets(n, 'full_test'); x, h = T['Q'][q], T['H'][:, j].astype(float); best = (-np.inf,)
    for t in np.quantile(x, np.linspace(.01, .99, 99)):
        for d in (1, -1):
            b = np.maximum(0, d * (x - t)); bb = (b * b).sum()
            if bb == 0: continue
            a = (b * h).sum() / bb; r2 = 1 - ((h - a * b) ** 2).sum() / ((h - h.mean()) ** 2).sum()
            if r2 > best[0]: best = (r2, t, d, a)
    r2, t, d, a = best; zero = float((h[d * (x - t) <= 0] == 0).mean())
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160); lo, hi = hexpanel(ax, x, h, j, axis(q)); c, mu = binned(x, h, lo, hi)
    ax.plot(c, mu, 'o', color=NAVY, ms=4.5, label='mean in each bin'); g = np.linspace(lo, hi, 300)
    ax.plot(g, a * np.maximum(0, d * (g - t)), color=RED, lw=2.6, label=f'max(0, {a * d:.3g}·(x − {t:.3g})), R² = {r2:.2f}' if d > 0 else f'max(0, {a:.3g}·({t:.3g} − x)), R² = {r2:.2f}')
    ax.set_ylabel('neuron value', fontsize=13); ax.set_title(f'network neuron {j}: {100 * zero:.0f}% of jets on the flat side give exactly 0', fontsize=13)
    ax.legend(fontsize=12, frameon=False, loc='upper left' if d > 0 else 'upper right'); save(fig, f'hinge_n{n}_{j}_{q}')


# ---- step 1 illustrated ----
def _hinges(n, j, q, n_fit=150000):
    fit = P.sub(P.jets(n, 'fit'), n_fit); T = P.jets(n, 'full_test')
    x, h = fit['Q'][q], fit['H'][:, j].astype(float); knots = np.quantile(x, np.linspace(.05, .95, 19))
    hb = lambda v, k, s: np.maximum(0, s * (v - k)); cands = [(k, s) for k in knots for s in (1, -1)]
    def lsq(ch):
        A = np.stack([np.ones(len(x))] + [hb(x, *c) for c in ch], 1); b = np.linalg.lstsq(A, h, rcond=None)[0]; return b, ((A @ b - h) ** 2).sum()
    xt, ht = T['Q'][q], T['H'][:, j].astype(float); lo, hi = np.quantile(xt, [.002, .95]); c, mu = binned(xt, ht, lo, hi, 40)
    g = np.linspace(lo, hi, 400); curve = lambda ch, b: b[0] + sum(bb * hb(g, *cc) for bb, cc in zip(b[1:], ch))
    return cands, lsq, curve, g, c, mu, lo, hi


def mars_steps(n=8, j=3, q='log_sum_pt'):
    cands, lsq, curve, g, c, mu, lo, hi = _hinges(n, j, q); fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), dpi=160, sharey=True); chosen = []
    for step, ax in enumerate(axs, 1):
        trials = [(lsq(chosen + [cc]), cc) for cc in cands if cc not in chosen]; (bb, _), best = min(trials, key=lambda r: r[0][1])
        ax.plot(c, mu, 'o', color=NAVY, ms=4.5, label=f"network's neuron {j} (bin means, test jets)", zorder=5)
        for (b, _), cc in trials[::2]: ax.plot(g, curve(chosen + [cc], b), ls=':', lw=1.1, color='#8a94a3')
        ax.plot([], [], ls=':', color='#8a94a3', label='other candidate hinges'); chosen.append(best)
        ax.plot(g, curve(chosen, bb), color=RED, lw=2.6, label=f'best: {step} hinge{"s" if step > 1 else ""}')
        for cc in chosen: ax.axvline(cc[0], color=RED, ls='--', lw=0.8)
        ax.set_title('First hinge: all candidate thresholds (dotted) and the least-squares best (solid)' if step == 1 else 'Second hinge', fontsize=13)
        ax.set_xlabel(axis(q), fontsize=14); ax.tick_params(labelsize=12); ax.set_xlim(lo, hi); ax.legend(fontsize=11.5, frameon=False, loc='upper left')
    axs[0].set_ylabel('neuron value', fontsize=14); axs[0].set_ylim(-0.6, np.nanmax(mu) * 1.3); save(fig, f'mars_steps_n{n}_{j}_{q}')


def mars_hinges(n=8, j=10, q='girth2'):
    cands, lsq, curve, g, c, mu, lo, hi = _hinges(n, j, q); chosen, fits = [], {}
    for step in range(1, 5):
        (b, _), best = min(((lsq(chosen + [cc]), cc) for cc in cands if cc not in chosen), key=lambda r: r[0][1]); chosen.append(best); fits[step] = (list(chosen), b)
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=170); ax.plot(c, mu, 'o', color=NAVY, ms=5, label=f"network's neuron {j} (mean per bin, test jets)")
    for step, col in ((1, ORANGE), (2, BLUE), (4, GREEN)):
        ch, b = fits[step]; ax.plot(g, curve(ch, b), color=col, lw=2.3, label=f'{step} hinge{"s" if step > 1 else ""}')
        for cc in ch: ax.axvline(cc[0], color=col, ls=':', lw=0.9)
    ax.set_xlabel(axis(q), fontsize=14); ax.set_ylabel('neuron value', fontsize=14); ax.set_xlim(lo, hi); ax.set_ylim(-0.2, np.nanmax(mu) * 1.15)
    ax.tick_params(labelsize=12); ax.legend(fontsize=12, frameon=False, loc='upper left'); save(fig, f'mars_hinges_n{n}_{j}_{q}')


# ---- step 2 without / with R ----
def fidelity(n=8, j=6, setup='all', K=100):
    """step 1 (first K terms per neuron) and step 2 tuned toward the network's probabilities with λ = 0 and λ = 0.01 (not
    pruned; cached in results/analysis/)"""
    last = Network(n).last; T = P.jets(n, 'full_test'); cache = RESULTS / 'analysis'; cache.mkdir(parents=True, exist_ok=True)
    fit = dev = None; f1 = None; Fs = {}
    for lam in (0.0, FIDELITY_LAMBDA):
        path = cache / f'fidelity_{setup}_n{n}_lam{lam:g}.json'
        if not path.exists():
            if fit is None: fit = P.sub(P.jets(n, 'fit'), N_TUNE_FIT); dev = P.jets(n, 'dev')
            f0 = tuning.refit([dict(nr, terms=nr['terms'][:K]) for nr in mars.load(setup, n)], fit['Q'], fit['Z'])
            F.save(tuning.train(f0, fit['Q'], dev['Q'], 'probabilities', fit, dev, last, lam=lam)[0], path)
        Fs[f'tuned on the network, λ = {lam:g}'] = F.load(path)
    Fs = {'step 1 (fitted to the neurons)': [dict(nr, terms=nr['terms'][:K]) for nr in mars.load(setup, n)], **Fs}
    col = dict(zip(Fs, (BLUE, RED, GREEN))); H = T['H'].astype(float); Hf = {k: F.hidden(f, T['Q']) for k, f in Fs.items()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(20, 6.6), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.25]))
    x = T['Q']['girth2']; e = np.array([0, .004, .008, .012, .016, .02, .025, .03, .04, .06]); c = .5 * (e[1:] + e[:-1]); k = np.digitize(x, e) - 1; ok = (k >= 0) & (k < len(c))
    prof = lambda h: np.bincount(k[ok], h[ok], len(c)) / np.maximum(np.bincount(k[ok], minlength=len(c)), 1)
    a.plot(c, prof(H[:, j]), color=NAVY, lw=3, marker='o', label='network')
    for nm, h in Hf.items(): a.plot(c, prof(h[:, j]), color=col[nm], lw=2.2, marker='o', ms=5, ls='--' if nm.endswith('λ = 0') else '-', label=nm)
    a.set_xlabel(AXIS['girth2'], fontsize=14); a.set_ylabel(f'mean value of neuron {j}', fontsize=14); a.set_title(f'neuron {j} ({n} particles)', fontsize=15)
    a.legend(fontsize=12, frameon=False, loc='upper left'); a.tick_params(labelsize=12)
    r2 = {nm: 1 - ((H - h) ** 2).sum(0) / np.maximum(((H - H.mean(0)) ** 2).sum(0), 1e-12) for nm, h in Hf.items()}; w = 0.27; idx = np.arange(H.shape[1])
    for i, (nm, r) in enumerate(r2.items()):
        b.bar(idx + (i - 1) * w, np.clip(r, -1, 1), w, color=col[nm], label=nm)
        for jj in np.where(r < -1)[0]: b.text(jj + (i - 1) * w, -1.02, f'{r[jj]:.1f}', ha='center', va='top', fontsize=10, color=col[nm])
    b.axhline(0, color='#6A7482', lw=1); b.set_ylim(-1.25, 1.05); b.set_xticks(idx); b.set_xlabel('neuron', fontsize=14); b.set_ylabel('R² against the network neuron (clipped at −1)', fontsize=13)
    b.set_title('every neuron: formula vs network', fontsize=15); b.legend(fontsize=12, frameon=False, loc='lower left'); b.tick_params(labelsize=12)
    save(fig, f'fidelity_n{n}')


# ---- neurons against the jet mass ----
def _mass_profile(m):
    bins = np.arange(40, 160.5, 1.0); c = .5 * (bins[1:] + bins[:-1]); k = np.digitize(m, bins) - 1; ok = (k >= 0) & (k < len(c))
    return bins, c, lambda h: np.bincount(k[ok], h[ok], len(c)) / np.maximum(np.bincount(k[ok], minlength=len(c)), 1)


def _mass_lines(ax, top):
    for nm, v in (('$m_W$', MASSES['m_W']), ('$m_Z$', MASSES['m_Z'])): ax.axvline(v, color='#8A94A3', ls='--', lw=1.2); ax.text(v + 0.8, top * 0.95, nm, color='#6A7482', fontsize=13)


def massnet(n=64, neurons=(0, 7, 10)):
    """the network's own neurons against the jet mass (mean per 1 GeV), although the network never receives the mass"""
    T = P.jets(n, 'full_test'); bins, c, prof = _mass_profile(T['Q']['mass']); fig, ax = plt.subplots(figsize=(10, 5.4), dpi=150); top = 0
    for j, col in zip(neurons, (NAVY, GREEN, ORANGE, BLUE, RED)):
        p = prof(T['H'][:, j].astype(float)); top = max(top, p.max() * 1.15); ax.plot(c, p, color=col, lw=2.4, label=f'network neuron {j}')
    _mass_lines(ax, top); ax.set_ylim(0, top); ax.set_xlim(bins[0], bins[-1]); ax.set_xlabel('jet mass [GeV]', fontsize=14); ax.set_ylabel('mean neuron value', fontsize=14)
    ax.legend(fontsize=12, frameon=False, loc='upper left'); ax.tick_params(labelsize=12); save(fig, f'massnet_n{n}')


def mass_nomass(n=64, setup='nomass', tag=None, neurons=(0, 7)):
    """network and formula neurons against the jet mass. setup 'nomass' (default): a formula that uses no mass observable;
    tag: a formula of the setup (default: the smaller version of the formula tuned on the network)"""
    neurons = (neurons,) if isinstance(neurons, int) else neurons
    reg = P.formulas(setup, n); tag = str(tag or next(t for t, m in reg.items() if m['family'] == 'network' and m['role'] == 'step4'))
    f = P.load_formula(setup, n, tag); nomass = not any('mass' in q or q in ('m01', 'm012') for q in F.observables_used(f))
    if SETUPS_NO_MASS.get(setup): assert nomass, 'the formula uses a mass observable'
    T = P.jets(n, 'full_test'); Hf = F.hidden(f, T['Q']); bins, c, prof = _mass_profile(T['Q']['mass'])
    fig, axs = plt.subplots(1, len(neurons), figsize=(7 * len(neurons), 5.4), dpi=150); axs = np.atleast_1d(axs)
    for ax, j in zip(axs, neurons):
        pn, pf = prof(T['H'][:, j].astype(float)), prof(Hf[:, j]); top = max(pn.max(), pf.max()) * 1.15
        ax.plot(c, pn, color=NAVY, lw=2.6, label=f'network neuron {j}'); ax.plot(c, pf, color=GREEN, lw=2.2, label=f'formula neuron {j}' + ('\n(no mass observable)' if nomass else ''))
        _mass_lines(ax, top); ax.set_ylim(0, top); ax.set_xlim(bins[0], bins[-1]); ax.set_xlabel('jet mass [GeV]', fontsize=14); ax.set_ylabel('mean neuron value', fontsize=14)
        ax.legend(fontsize=11.5, frameon=False, loc='upper left', bbox_to_anchor=(0, 0.9)); ax.tick_params(labelsize=12)
    save(fig, f'mass_{setup}_n{n}_{tag}_' + '_'.join(map(str, neurons)))


def neuron_hist(n=8, lam=0.0, neurons=(4, 6, 8, 13, 15, 2), setup='all'):
    """formula neuron against network neuron, jet by jet (2D histograms, log colour), for the step-2 formula tuned with the
    given λ (the fidelity figure's cached formulas; run 'fidelity' first); neurons='all' for all 16"""
    path = RESULTS / 'analysis' / f'fidelity_{setup}_n{n}_lam{lam:g}.json'
    if not path.exists(): raise SystemExit(f'{path} missing: run python -m jetdistill.figures fidelity {n}')
    T = P.jets(n, 'full_test'); Hf = F.hidden(F.load(path), T['Q']); H = T['H'].astype(float)
    neurons = list(range(H.shape[1])) if neurons == 'all' else [neurons] if isinstance(neurons, int) else list(neurons); cols = min(len(neurons), 4 if len(neurons) > 6 else 3); rows = -(-len(neurons) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), dpi=130); axs = np.atleast_1d(axs).ravel()
    for ax, j in zip(axs, neurons):
        top = max(np.quantile(H[:, j], .999), np.quantile(Hf[:, j], .999), 1e-3)
        ax.hist2d(H[:, j], Hf[:, j], bins=60, range=[[0, top], [0, top]], cmap='Blues', norm=matplotlib.colors.LogNorm()); ax.plot([0, top], [0, top], color=RED, ls='--', lw=1)
        r2 = 1 - ((H[:, j] - Hf[:, j]) ** 2).sum() / max(((H[:, j] - H[:, j].mean()) ** 2).sum(), 1e-12)
        ax.set_title(f'neuron {j}   R² = {r2:.2f}', fontsize=12); ax.set_xlabel('network', fontsize=11); ax.set_ylabel('formula', fontsize=11)
    for ax in axs[len(neurons):]: ax.axis('off')
    save(fig, f'neuron_hist_n{n}_lam{lam:g}_' + ('all' if len(neurons) == H.shape[1] else '_'.join(map(str, neurons))))


# ---- summary charts ----
def _best(setup, n, family, role):
    f = P.out_dir(setup, n) / 'metrics.json'
    if not f.exists(): return None
    reg = P.formulas(setup, n); rows = {r['formula']: r for r in json.loads(f.read_text())['models'] if r.get('formula')}
    ts = [t for t, m in reg.items() if m['family'] == family and m['role'] == role and t in rows]
    return 100 * max(rows[t]['same_as_network'] for t in ts) if ts else None


def _dots(name, rows, ns=(8, 64)):
    """rows: [(label, setup, family)]: the formula tuned that way (main) and its smaller version (step 4), per network"""
    fig, axs = plt.subplots(1, len(ns), figsize=(7 * len(ns), 5.0), dpi=150, sharey=True); axs = np.atleast_1d(axs); ys = list(range(len(rows)))[::-1]
    for ax, n in zip(axs, ns):
        for y in ys: ax.axhline(y, color='#EEF0F3', lw=1, zorder=0)
        for lab, col, mk, role in (('tuned', NAVY, 'o', 'main'), ('smaller version (step 4)', GREEN, 'D', 'step4')):
            pts = [(v, y) for (_, s, fam), y in zip(rows, ys) if (v := _best(s, n, fam, role)) is not None]
            if not pts: continue
            ax.scatter(*zip(*pts), s=90, color=col, marker=mk, label=lab, zorder=3)
            for v, y in pts: ax.text(v, y + 0.2, f'{v:.1f}', ha='center', fontsize=12, color=col)
        ax.set_title(f'{n} particles', fontsize=14); ax.set_xlabel('agreement with the network [%]', fontsize=14); ax.tick_params(labelsize=12)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    axs[0].set_yticks(ys); axs[0].set_yticklabels([r[0] for r in rows], fontsize=13); axs[0].set_ylim(-0.6, len(rows) - 0.3)
    axs[-1].legend(frameon=False, fontsize=12, loc='lower right'); save(fig, name)


def setups():
    _dots('setups', [('all observables', 'all', 'network'), ('no W/Z/t mass thresholds', 'nophys', 'network'), ('no mass observables', 'nomass', 'network'),
                     ('no mass observables\nor equivalents', 'nomass_strict', 'network')])


def targets():
    _dots('targets', [('class probabilities', 'all', 'network'), ('one-hot class', 'all_agree', 'network'), ('neuron values only', 'all', 'neurons'),
                      ('true labels', 'all', 'labels')])


def _bars(ax, groups, series, ylabel):
    w = 0.8 / len(series)
    for k, (lab, col, vals) in enumerate(series):
        xs = [i + (k - (len(series) - 1) / 2) * w for i in range(len(groups))]; ax.bar(xs, vals, w * 0.9, color=col, label=lab)
        for x, v in zip(xs, vals): ax.text(x, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=13, color=NAVY)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=13); ax.set_ylim(0, 1.05); ax.set_ylabel(ylabel, fontsize=14); ax.tick_params(labelsize=12)


def control(ns=(8, 64)):
    from .analysis.control import numbers
    N = {n: numbers(n) for n in ns}; ns = [n for n in ns if N[n]]
    if not ns: raise SystemExit('control not run (python -m jetdistill.analysis.control 8)')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150, gridspec_kw=dict(width_ratios=[1.5, 1]))
    _bars(a1, [f'{n} particles' for n in ns], [('trained network', NAVY, [N[n]['physics']['trained'] for n in ns]),
                                              ('untrained (random weights)', GRAY, [N[n]['physics']['untrained'] for n in ns])], 'R² of step 1 (median over neurons)')
    a1.legend(frameon=False, fontsize=12, loc='upper right', bbox_to_anchor=(1, 1.12), ncol=2)
    o = [100 * N[ns[0]]['orientation'][k] for k in ('trained', 'untrained')]; a2.bar([0, 1], o, 0.6, color=[NAVY, GRAY])
    for x, v in zip([0, 1], o): a2.text(x, v + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=13)
    a2.set_xticks([0, 1]); a2.set_xticklabels(['trained', 'untrained'], fontsize=13); a2.set_ylim(0, 105); a2.set_xlim(-0.7, 1.7)
    a2.set_ylabel('importance on Δη- or Δφ-only\nobservables [%]', fontsize=14); a2.set_xlabel(f'{ns[0]} particles', fontsize=13); a2.tick_params(labelsize=12)
    save(fig, 'control')


def inputs(ns=(8, 64)):
    from .analysis.control import numbers
    N = {n: numbers(n) for n in ns}; ns = [n for n in ns if N[n]]
    if not ns: raise SystemExit('control not run (python -m jetdistill.analysis.control 8)')
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=150)
    _bars(ax, [f'{n} particles' for n in ns], [('MARS on the physics observables', NAVY, [N[n]['physics']['trained'] for n in ns]),
                                              ('MARS on the raw inputs (pT, Δη, Δφ)', GRAY, [N[n]['raw_inputs']['trained'] for n in ns]),
                                              ('boosted decision trees on the raw inputs', LIGHT, [N[n]['trees']['trained'] for n in ns])], 'R² (median over neurons)')
    ax.legend(frameon=False, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3); save(fig, 'control_inputs')


def convergence(ns=(8, 64), setup='all', used=800):
    fs = {n: RESULTS / 'analysis' / f'convergence_{setup}_n{n}.json' for n in ns}; ns = [n for n in ns if fs[n].exists()]
    if not ns: raise SystemExit('convergence not run (python -m jetdistill.analysis.convergence 8)')
    fig, axs = plt.subplots(1, len(ns), figsize=(7 * len(ns), 5), dpi=150); axs = np.atleast_1d(axs)
    for ax, n in zip(axs, ns):
        H = json.loads(fs[n].read_text())['history']; s = [h['step'] for h in H]
        ax.plot(s, [100 * h['train'] for h in H], color='#8A94A3', lw=2, label='training jets'); ax.plot(s, [100 * h['val'] for h in H], color=NAVY, lw=2.2, label='validation jets')
        ax.axvline(used, color=GREEN, ls=':', lw=1.5); ax.text(used + 20, ax.get_ylim()[0] + 0.05, f'steps used: {used}', color=GREEN, fontsize=12)
        ax.set_title(f'{n} particles', fontsize=14); ax.set_xlabel('Adam step', fontsize=14); ax.set_ylabel('agreement with the network [%]', fontsize=14); ax.tick_params(labelsize=12)
    axs[0].legend(frameon=False, fontsize=12, loc='lower right'); save(fig, f'convergence_{setup}')


# ---- one if-statement ----
def term(setup='all', n=8, tag=None, neuron=10, q='girth2', kind='gt'):
    """what the if-statement adds to the neuron as a function of its observable, and the observable's distribution per true
    class (from the formula's explanation data)"""
    tag = str(tag or next(t for t, m in P.formulas(setup, n).items() if m['family'] == 'network' and m['role'] == 'step4'))
    A = json.loads((P.out_dir(setup, n) / 'formulas' / tag / 'anatomy.json').read_text())
    t = next(t for t in next(x for x in A['neurons'] if x['neuron'] == neuron)['terms'] if t['q'] == q and t['kind'] == kind and not t.get('q2'))
    H = t['hist']; e = np.array(H['edges']); c = .5 * (e[1:] + e[:-1]); x = np.linspace(e[0], e[-1], 200)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)
    a1.plot(x, t['coef'] * (np.maximum(x - t['t'], 0) if kind == 'gt' else np.maximum(t['t'] - x, 0)), color=NAVY, lw=2.2); a1.axvline(t['t'], color='#8A94A3', ls=':', lw=1.4)
    a1.set_xlabel(axis(q), fontsize=14); a1.set_ylabel(f'added to neuron {neuron}', fontsize=14); a1.tick_params(labelsize=12)
    for i, lab in enumerate(CLASSES): a2.plot(c, H['per_class'][i], color=CCOL[i], lw=2, label=lab)
    a2.axvline(t['t'], color='#8A94A3', ls=':', lw=1.4); a2.set_xlabel(axis(q), fontsize=14); a2.set_ylabel('fraction of jets of each class', fontsize=14)
    a2.legend(frameon=False, fontsize=12, ncol=5); a2.tick_params(labelsize=12); save(fig, f'term_{setup}_n{n}_{tag}_{neuron}_{q}')


def equations():
    fig = plt.figure(figsize=(11, 2.88), dpi=200); fig.patch.set_facecolor('#F1F3F6')
    L = [r'$z_i = p_{T,i}\ /\ \Sigma_j\, p_{T,j}$', r'$r_i = (\Delta\eta_i, \Delta\phi_i)$, $\ \Delta R_i = |r_i|$', r'$m^2 = (\Sigma_i\, p_i)^2$',
         r'$\Sigma z\Delta R^2 = \Sigma_i\, z_i\,\Delta R_i^2$']
    R = [r'$e_2 = \Sigma_{i<k}\, z_i z_k\,\Delta R_{ik}$', r'$\tau_{21} = \tau_2\,/\,\tau_1$', r'$\lambda_1 \geq \lambda_2\ \mathrm{eigenvalues\ of}\ \Sigma_i\, z_i\, r_i r_i^{T}$',
         r'$\lambda_1 + \lambda_2 = \Sigma z\Delta R^2$']
    for col, x in ((L, 0.03), (R, 0.50)):
        for k, s in enumerate(col): fig.text(x, 0.85 - k * 0.235, s, fontsize=23, color=NAVY, va='center')
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / 'equations.png', facecolor=fig.get_facecolor()); plt.close(fig); print(OUT / 'equations.png')


FIGURES = dict(motivation=motivation, hinge=hinge, linear=linear, mars_steps=mars_steps, mars_hinges=mars_hinges, fidelity=fidelity, massnet=massnet, mass_nomass=mass_nomass, neuron_hist=neuron_hist,
               setups=setups, targets=targets, control=control, inputs=inputs, convergence=convergence, term=term, equations=equations)

if __name__ == '__main__':
    name, *args = sys.argv[1:] or ['all']
    def conv(a):
        for t in (int, float):
            try: return t(a)
            except ValueError: pass
        return a
    for nm in (FIGURES if name == 'all' else [name]):
        try: FIGURES[nm](*map(conv, args))
        except (SystemExit, FileNotFoundError, StopIteration) as e:
            if name != 'all': raise
            print(nm, 'skipped:', e or 'results missing')
