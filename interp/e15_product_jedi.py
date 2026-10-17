"""E15 (JEDI-linear 3-feature n64; claims A4, A9): product or sum — do h4/h5 multiply M0 = ΣpT and M2 = Σ pT r², or add them?

Sample: sample(per_class=2000, seed=11) = 10,000 test jets; exact fixed-point network (interp.jedi_core.Jedi), stages S, h4,
h5, z, logits; control network Jedi(untrained=True, seed=0).

A  factorial edit: pT × c (c ∈ {0.8, 1, 1.25}) × angles × k (k ∈ {0.8, 1, 1.25}), the 9 cells; real particles only (padded
   slots stay (0, 0, 0)); pT order unchanged; jets whose edited input leaves the l1 quantizer range in any cell are dropped
   (e5_causal.in_range).  M0 → c·M0, M2 → k²·M2, m → c·k·m (small angle).
   Per live unit U of S, h4, h5, z (and per mass read-out), on jets with U > 0 in all 9 cells, the four corner cells
   (c, k ≠ 1) are compared with two nulls built from the single-factor cells:
     U_lin = U(c,1) + U(1,k) − U(1,1)      exact if U = f(M0) + g(M2)   (a weighted sum a·M0 + b·M2; also log of a product)
     U_log = U(c,1)·U(1,k) / U(1,1)        exact if U = f(M0)·g(M2)     (a product M0^α·M2^β)
   Interaction I = U(c,k) − U_lin; the product predicts I = U_log − U_lin = ΔU_c·ΔU_k / U(1,1), the sum predicts I = 0.
   Per unit, in units of the unit's sd on the unedited sample: median |U − U_lin|, median |U − U_log|, the fraction of
   jet × cell pairs where U_log is closer, I projected on the product's sign s = sign((c−1)(k²−1))·sign(e_c·e_k) (mean over
   jets and corner cells; 1,000-resample Poisson bootstrap over jets), the product's prediction of the same, and
   β = Σ I·I_pred / Σ I_pred² (1 = product, 0 = sum; in between: a product with an offset, U = t + f·g, t = U(1,1)(1 − 1/β)).
   Supplementary: single-factor step ratios (U(1.25) − U(1)) / (U(1) − U(0.8)) for c and for k: linear in M0 / M2: 1.25 /
   1.5625; ∝ √M0 / ∝ √M2 (= k): 1.118 / 1.25; linear in log M0 / log M2: 1.0 / 1.0.
   Read-outs: ridge of log m and of m (GeV) on the unedited train half (as E6), evaluated on the test half, treated as units.
   Controls: the live S units (sums over particles by construction) and the live units of the untrained network.
B  S0 patch at matched M2 (run_jetwise, S0 replaced by a donor jet's S0): donor = another jet of the sample with
   |Δlog S29| < 0.05, |Δlog S50| < 0.05 (the M2-like units) and |Δlog S0| ≥ 0.25, chosen at random; jets without a donor
   dropped.  Metric: slope of Δlog m̂ (read-outs at h4, h5, z) on ½Δlog S0 within tertiles of M2 (true Σ pT r²; also of
   M2/S0 and of S29), with the product's prediction (slope 1 in every tertile) and the sum's prediction
   (log(1 + s_j (e^{Δlog S0} − 1)), s_j = share of the M0 term with a·M0 = b·M2 at the sample medians); W→Z and Z→t call
   switches per tertile and Δlog S0 bin.  Controls: S19 (soft multiplicity) patched with the same donors' S19 (x axis
   unchanged: the donor's ½Δlog S0); donors with M2 not matched; the S0 patch on the untrained network; supplementary
   S0 × f, f ∈ {0.7, 0.85, 1.2, 1.4}.
Run:  python -m interp.e15_product_jedi            (writes $AI_RESULTS/jedi/e15_product.json)
      python -m interp.e15_product_jedi --plot     (reads the json [+ e6_mass.json for E6's ρ]; writes e15_product.png and
                                                    e15_product_table.png; needs matplotlib)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
OUT_JSON = OUT_DIR / 'e15_product.json'
OUT_PNG = OUT_DIR / 'e15_product.png'
OUT_TAB = OUT_DIR / 'e15_product_table.png'
PER, SEED = 2000, 11
CS = (0.8, 1.0, 1.25); KS = (0.8, 1.0, 1.25)
CORNERS = [(ci, ki) for ci in (0, 2) for ki in (0, 2)]
LSTEP = float(np.log(1.25) - np.log(0.8))
STG = ('S', 'h4', 'h5', 'z'); KEEP = STG + ('logits',)
NBOOT = 1000
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_, T_ = 2, 3, 4
MASSLIKE = [('h4', 18), ('h5', 0), ('h5', 28), ('z', 9), ('z', 11), ('z', 12)]     # E6's mass-like units
FSCAN = (0.7, 0.85, 1.2, 1.4)
PAL = {'S': '#2a78d6', 'h4': '#eb6834', 'h5': '#1baf7a', 'z': '#eda100', 'readout': '#7b3fb8', 'untrained': '#9a9a94'}
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)


# ---------------------------------------------------------------- plot mode (reads the json; matplotlib only)
def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = json.loads(OUT_JSON.read_text())
    e6 = {}
    for p in (OUT_DIR / 'e6_mass.json', ROOT / 'results' / 'jedi' / 'e6_mass.json'):
        if p.exists():
            e6 = json.loads(p.read_text())['units']; break
    surf, ink, ink2, grid = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df'
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.0), facecolor=surf)
    for a in ax:
        a.set_facecolor(surf); a.grid(True, color=grid, lw=0.8); a.set_axisbelow(True)
        for s in ('top', 'right'): a.spines[s].set_visible(False)
        for s in ('left', 'bottom'): a.spines[s].set_color(grid)
        a.tick_params(colors=ink2, labelsize=9)
    # (a) per-unit interaction vs the product's prediction
    a = ax[0]; xs = []
    for tag, F in (('trained', R['A']['trained']), ('untrained', R['A']['untrained'])):
        for st in STG:
            for u, d in F['units'][st].items():
                if 'I' not in d: continue
                col = PAL[st] if tag == 'trained' else PAL['untrained']
                a.errorbar(d['I_pred'], d['I'], yerr=d['I_se'], fmt='o' if tag == 'trained' else 'o', ms=5 if tag == 'trained' else 4,
                           color=col, mfc=col if tag == 'trained' else surf, mew=1.2, ecolor=col, elinewidth=0.8, alpha=0.9, zorder=3)
                xs.append(d['I_pred'])
                if tag == 'trained' and (st, int(u)) in MASSLIKE:
                    a.annotate(f'{st}{u}', (d['I_pred'], d['I']), xytext=(4, 3), textcoords='offset points', color=ink, fontsize=8)
    for st in ('h4', 'h5', 'z'):
        d = R['A']['trained']['readout'][st]['m_gev']
        if 'I' in d:
            a.errorbar(d['I_pred'], d['I'], yerr=d['I_se'], fmt='*', ms=11, color=PAL['readout'], mfc=PAL['readout'], zorder=4)
            a.annotate(f'm̂ {st}', (d['I_pred'], d['I']), xytext=(5, -9), textcoords='offset points', color=PAL['readout'], fontsize=8)
    lim = max(0.02, 1.15 * max(xs + [0.01]))
    a.plot([0, lim], [0, lim], color=ink, lw=1.4, ls='--'); a.text(lim * 0.98, lim * 0.98, 'product: I = U_log − U_lin', color=ink, fontsize=9, ha='right', va='bottom')
    a.axhline(0, color=ink2, lw=1.2, ls='-.'); a.text(lim * 0.98, 0, 'sum: I = 0', color=ink2, fontsize=9, ha='right', va='bottom')
    a.set_xlim(-0.02 * lim, lim); a.set_ylim(min(-0.25 * lim, a.get_ylim()[0]), max(lim, a.get_ylim()[1]))
    a.set_xlabel('interaction predicted by a product,  ΔU_c·ΔU_k / U  [sd of the unit]', color=ink2, fontsize=10)
    a.set_ylabel('measured interaction  U(c,k) − U(c,1) − U(1,k) + U(1,1)  [sd]', color=ink2, fontsize=10)
    a.set_title('(a) factorial edit pT × c, angles × k: 4 corner cells, mean over jets', color=ink, fontsize=11, loc='left')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker='o', ls='', color=PAL[s], label=f'{s} units') for s in STG]
    hs += [Line2D([], [], marker='o', ls='', color=PAL['untrained'], mfc=surf, label='untrained network'),
           Line2D([], [], marker='*', ls='', ms=11, color=PAL['readout'], label='m̂ read-out (GeV)')]
    a.legend(handles=hs, frameon=False, fontsize=8.5, loc='upper left', labelcolor=ink)
    # (b) S0 patch at matched M2
    a = ax[1]; B = R['B']['S0_matched']; rd = B.get('plot_reader', 'h5')
    cols = ['#1f77b4', '#2ca02c', '#d62728']; mk = ['o', 's', '^']
    for i, tname in enumerate(('low', 'mid', 'high')):
        t = B['tertiles']['M2'][tname]; bn = t['bins'][rd]
        a.errorbar(bn['x'], bn['y'], yerr=bn['se'], fmt=mk[i] + '-', color=cols[i], ms=5, lw=1.6, mfc=surf, mew=1.5,
                   label=f'M2 {tname} tertile (median {t["M2_median"]:.0f} GeV·rad²): slope {t["slope"][rd]:.2f} ± {t["slope_se"][rd]:.2f}')
        a.plot(bn['x'], bn['sum_pred'], color=cols[i], lw=1, ls=':')
    C = R['B']['S19_control']; bn = C['tertiles']['M2']['all']['bins'][rd]
    a.errorbar(bn['x'], bn['y'], yerr=bn['se'], fmt='x', color=PAL['untrained'], ms=6, lw=1.2,
               label=f'control: S19 patched instead (slope {C["tertiles"]["M2"]["all"]["slope"][rd]:.2f})')
    xx = np.array([min(b['bins'][rd]['x'][0] for b in [B['tertiles']['M2'][t] for t in ('low', 'mid', 'high')]),
                   max(b['bins'][rd]['x'][-1] for b in [B['tertiles']['M2'][t] for t in ('low', 'mid', 'high')])])
    a.plot(xx, xx, color=ink, lw=1.4, ls='--'); a.text(xx[1], xx[1], 'product: Δlog m̂ = ½Δlog S0', color=ink, fontsize=9, ha='right', va='bottom')
    a.text(xx[1], 0.55 * xx[1], 'dotted: sum a·M0 + b·M2 (per tertile)', color=ink2, fontsize=8.5, ha='right')
    a.axhline(0, color=ink2, lw=0.8)
    a.set_xlabel('½ Δlog S0 applied by the patch (donor S0 / own S0)', color=ink2, fontsize=10)
    a.set_ylabel(f'Δ log m̂, read-out at {rd}', color=ink2, fontsize=10)
    a.set_title('(b) S0 replaced by a donor\'s at matched S29, S50 (M2 fixed)', color=ink, fontsize=11, loc='left')
    a.legend(frameon=False, fontsize=8, loc='upper left', labelcolor=ink)
    fig.suptitle(f'JEDI-linear 64 p, {R["N"]} test jets (seed {R["seed"]}), acc {R["acc"]:.3f}; sd = the unit\'s spread on the unedited sample',
                 color=ink2, fontsize=9, x=0.01, ha='left', y=0.995)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=150, facecolor=surf); print('wrote', OUT_PNG)

    # table of the top units
    rows = []
    for st in STG:
        for u, d in R['A']['trained']['units'][st].items():
            if 'I' in d: rows.append((st, int(u), d))
    rows.sort(key=lambda r: -abs(r[2]['I_pred']))
    top = [r for r in rows if (r[0], r[1]) in MASSLIKE] + [r for r in rows if (r[0], r[1]) not in MASSLIKE][:12]
    top += [(f'm̂ {st}', '', R['A']['trained']['readout'][st]['m_gev']) for st in ('h4', 'h5', 'z') if 'I' in R['A']['trained']['readout'][st]['m_gev']]
    hdr = ['unit', 'n jets', 'ρ (E6)', 'ρ (E15)', 'med |U−U_lin|', 'med |U−U_log|', 'frac log closer', 'I pred', 'I ± se', 'β ± se', 'step c | k']
    cells = []
    for st, u, d in top:
        rho6 = e6.get(st, {}).get(str(u), {}).get('rho') if u != '' else None
        f = lambda v, p='.3f': ('–' if v is None else format(v, p))
        cells.append([f'{st}{u}', str(d['n']), f(rho6, '.2f'), f(d['rho'], '.2f'), f(d['med_abs_lin']), f(d['med_abs_log']),
                      f(d['frac_log_closer'], '.2f'), f(d['I_pred']), f'{d["I"]:.3f} ± {d["I_se"]:.3f}', f'{f(d["beta"], ".2f")} ± {f(d["beta_se"], ".2f")}',
                      f'{f(d["step_c"], ".2f")} | {f(d["step_k"], ".2f")}'])
    fig2, a2 = plt.subplots(figsize=(13, 0.34 * len(cells) + 1.3), facecolor=surf); a2.axis('off')
    tb = a2.table(cellText=cells, colLabels=hdr, loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.25)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(grid); c.set_text_props(color=ink)
        if i == 0: c.set_facecolor('#efeeea'); c.set_text_props(weight='bold')
        elif (top[i - 1][0], top[i - 1][1]) in MASSLIKE: c.set_facecolor('#f3f7ff')
    a2.set_title('E15 factorial edit: per-unit interaction (sd units). β = 1 product, β = 0 sum. Step ratios: sum 1.25 | 1.56, √(M0·M2) 1.12 | 1.25, log 1.0 | 1.0',
                 color=ink, fontsize=9.5, loc='left')
    fig2.tight_layout(); fig2.savefig(OUT_TAB, dpi=150, facecolor=surf); print('wrote', OUT_TAB)


if '--plot' in sys.argv:
    plot(); sys.exit(0)

# ---------------------------------------------------------------- main
from interp.jedi_core import Jedi, sample  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
x, y, Q = sample(per_class=PER, seed=SEED); x = x.astype(np.float32); J = len(x)
real = x[..., 0] > 0
net = Jedi(64); unt = Jedi(64, untrained=True, seed=0)
res = dict(N=J, per_class=PER, seed=SEED, cells_c=CS, cells_k=KS, nboot=NBOOT,
           input_pt_sorted_frac=float((np.diff(x[..., 0], axis=1) <= 1e-6).all(1).mean()),
           padded_slots_frac=float((~real).mean()), padded_max_abs_value=float(np.abs(x[~real]).max()))
log(f'E15: {J} jets ({PER}/class, seed {SEED}); networks loaded; jets pT-sorted {res["input_pt_sorted_frac"]:.4f}; padded slots '
    f'{res["padded_slots_frac"]:.3f}, max |value| in a padded slot {res["padded_max_abs_value"]:.2g}')


def in_range(net_, xx):
    """e5_causal.in_range: no real particle outside its slot's l1 fixed-point input range (WRAP would wrap it)"""
    q = net_.L['l1'].iq.quantizer
    k_, i_, f_ = (np.asarray(getattr(q, a)).reshape(64, 3) for a in ('k', 'i', 'f'))
    hi = 2.0 ** i_ - 2.0 ** -f_; lo = np.where(k_ > 0, -2.0 ** i_, 0.0)
    xn = net_.inputs(xx, 64); rl = xx[..., 0] > 0
    return ~((((xn > hi[None]) | (xn < lo[None])) & rl[..., None]).any((1, 2)))


def moments(xx):
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    r2 = eta ** 2 + phi ** 2
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=pt.sum(1), M2=(pt * r2).sum(1), m=m, logm=np.log(np.maximum(m, 1e-3)))


def edit(xx, c, k):
    e = xx.copy()
    e[..., 0] = np.where(real, e[..., 0] * np.float32(c), e[..., 0])
    e[..., 1:] = np.where(real[..., None], e[..., 1:] * np.float32(k), e[..., 1:])
    return e


M = moments(x)
R0 = net.run(x, keep=KEEP); call0 = R0['logits'].argmax(1)
res['acc'] = float((call0 == y).mean())
U0 = unt.run(x, keep=KEEP); res['acc_untrained'] = float((U0['logits'].argmax(1) == y).mean())
log(f'accuracy trained {res["acc"]:.4f}, untrained {res["acc_untrained"]:.4f}')
chk = net.run_jetwise(x, {'S': (np.ones(64), R0['S'])}, keep=KEEP)
res['jetwise_identity_maxdiff'] = {k: float(np.abs(chk[k] - R0[k]).max()) for k in KEEP}
log('run_jetwise with S := own S, max |diff| per stage:', res['jetwise_identity_maxdiff'])

# split for the read-outs (fit on the train half, evaluate on the test half)
rng = np.random.default_rng(SEED); perm = rng.permutation(J)
tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:]); trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
is_te = np.zeros(J, bool); is_te[te] = True
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)


class Ridge:
    """standardised ridge (as E6): alpha chosen on a validation part of the training half, then refit on the whole half"""

    def __init__(self, X, t, idx_fit, idx_val, idx_all):
        self.mu = X[idx_all].mean(0); self.sd = X[idx_all].std(0); self.sd[self.sd < 1e-9] = np.inf
        Zs = (X - self.mu) / self.sd; self.tm = t[idx_all].mean(); self.ts = t[idx_all].std(); tt = (t - self.tm) / self.ts; best = None
        for a in ALPHAS:
            w = self._fit(Zs[idx_fit], tt[idx_fit], a); e = ((Zs[idx_val] @ w - tt[idx_val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = best[1]; self.w = self._fit(Zs[idx_all], tt[idx_all], self.alpha)

    @staticmethod
    def _fit(Zs, t, a):
        Zc = Zs - Zs.mean(0); return np.linalg.solve(Zc.T @ Zc + a * len(Zs) * np.eye(Zs.shape[1]), Zc.T @ (t - t.mean()))

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w * self.ts + self.tm


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def make_readers(RR):
    rd = {}
    for st in STG:
        rd[(st, 'logm')] = Ridge(RR[st], M['logm'], trf, trv, tr); rd[(st, 'm')] = Ridge(RR[st], M['m'], trf, trv, tr)
    return rd


readers = make_readers(R0); readers_u = make_readers(U0)
res['readout_r2'] = {st: dict(logm=r2(readers[(st, 'logm')](R0[st][te]), M['logm'][te]), m=r2(readers[(st, 'm')](R0[st][te]), M['m'][te]),
                              logm_untrained=r2(readers_u[(st, 'logm')](U0[st][te]), M['logm'][te])) for st in STG}
log('read-out held-out R² (log m | m GeV | untrained log m): ' + ', '.join(f'{st} {v["logm"]:.3f}|{v["m"]:.3f}|{v["logm_untrained"]:.3f}' for st, v in res['readout_r2'].items()))

# bootstrap weights (Poisson), shared across units for comparability
WB = np.random.default_rng(SEED + 1).poisson(1.0, size=(NBOOT, J)).astype(np.uint8)

# ================================================================ A. factorial edit
X = {}; inr = np.ones(J, bool)
for c in CS:
    for k in KS:
        X[(c, k)] = edit(x, c, k) if (c, k) != (1.0, 1.0) else x
        inr &= in_range(net, X[(c, k)])
res['A'] = dict(in_range_all_cells=float(inr.mean()), n_in_range=int(inr.sum()),
                padded_moved_max=float(max(np.abs(xe[~real] - x[~real]).max() for xe in X.values())),
                order_kept_frac=float(min((np.diff(xe[..., 0], axis=1) <= 1e-6).all(1).mean() for xe in X.values())))
log(f'A: 9 cells built; jets inside the input range in all cells {inr.mean():.4f} ({inr.sum()}); padded slots moved by at most '
    f'{res["A"]["padded_moved_max"]:.2g}; pT order kept in {res["A"]["order_kept_frac"]:.4f} of jets')
mom = {ck: moments(xe) for ck, xe in X.items()}
res['A']['moment_ratios'] = {f'c{c}k{k}': dict(M0=float(np.median(mom[(c, k)]['M0'] / M['M0'])), M2=float(np.median(mom[(c, k)]['M2'] / M['M2'])),
                                               m=float(np.median(mom[(c, k)]['m'] / np.maximum(M['m'], 1e-3))))
                             for (c, k) in X}
RA = {ck: (R0 if ck == (1.0, 1.0) else net.run(xe, keep=KEEP)) for ck, xe in X.items()}
RAU = {ck: (U0 if ck == (1.0, 1.0) else unt.run(xe, keep=KEEP)) for ck, xe in X.items()}
log(f'A: {2 * (len(X) - 1)} edited forward passes done')
res['A']['calls'] = {f'c{c}k{k}': dict(acc=float((RA[(c, k)]['logits'].argmax(1)[inr] == y[inr]).mean()),
                                       W_to_Z=float((RA[(c, k)]['logits'].argmax(1)[inr & (y == W_)] == Z_).mean()),
                                       Z_to_t=float((RA[(c, k)]['logits'].argmax(1)[inr & (y == Z_)] == T_).mean())) for (c, k) in X}


def cube(getter, jets):
    """(3, 3, n) array of one quantity in the 9 cells on the given jets (axis 0: c, axis 1: k)"""
    return np.stack([np.stack([getter((c, k))[jets] for k in KS], 0) for c in CS], 0)


def fstats(V, sd, wb, min_n=200):
    """factorial statistics of one quantity: V (3, 3, n) cell values; sd = its spread on the unedited sample; wb (NBOOT, n) weights"""
    ok = (V > 0).all((0, 1)); n = int(ok.sum()); out = dict(n=n, n_offered=int(V.shape[2]))
    if n < min_n or sd <= 1e-9: return out
    V = V[:, :, ok]; w = wb[:, ok].astype(np.float64); wsum = w.sum(1)
    B = V[1, 1]
    dc = V[2, 1] - V[0, 1]; dk = V[1, 2] - V[1, 0]
    ec = float(dc.mean() / LSTEP / sd); ek = float(dk.mean() / LSTEP / sd)
    ssign = np.sign(ec * ek) if abs(ec * ek) > 1e-12 else 1.0
    a = np.zeros(n); p = np.zeros(n); qq = np.zeros(n); rr = np.zeros(n); absl = []; absg = []; closer = []; cells = {}
    for ci, ki in CORNERS:
        c, k = CS[ci], KS[ki]
        U = V[ci, ki]; lin = V[ci, 1] + V[1, ki] - B; logn = V[ci, 1] * V[1, ki] / B
        Im = U - lin; Ip = logn - lin
        s = np.sign((c - 1) * (k * k - 1)) * ssign
        a += Im * s; p += Ip * s; qq += Im * Ip; rr += Ip * Ip
        absl.append(np.abs(Im)); absg.append(np.abs(U - logn))
        closer.append(np.where(Ip == 0, 0.5, (np.abs(U - logn) < np.abs(Im)).astype(float)))
        cells[f'c{c}k{k}'] = dict(I=float(Im.mean() / sd), I_se=float(Im.std() / np.sqrt(n) / sd), I_pred=float(Ip.mean() / sd),
                                  sign=float(s), rel_dU_c=float(np.median((V[ci, 1] - B) / B)), rel_dU_k=float(np.median((V[1, ki] - B) / B)))
    absl = np.concatenate(absl); absg = np.concatenate(absg); closer = np.concatenate(closer)
    bI = (w @ a) / (4 * wsum) / sd; bP = (w @ p) / (4 * wsum) / sd
    # β = (measured interaction projected on the product's sign) / (the product's prediction of the same): the per-jet
    # regression Σ I·I_pred / Σ I_pred² is biased under the sum because I and I_pred share the single-factor cells' noise
    bB = bI / np.where(np.abs(bP) > 1e-12, bP, np.nan); bB = bB[np.isfinite(bB)]
    s1 = (V[2, 1] - B).mean(); s0 = (B - V[0, 1]).mean(); t1 = (V[1, 2] - B).mean(); t0_ = (B - V[1, 0]).mean()
    Ip_s = float(p.mean() / 4 / sd)
    return dict(n=n, n_offered=int(V.shape[2]), sd=float(sd), e_c=ec, e_k=ek, rho=float(ek / ec) if abs(ec) > 1e-9 else None,
                U_over_sd_median=float(np.median(B) / sd),
                med_abs_lin=float(np.median(absl) / sd), med_abs_log=float(np.median(absg) / sd),
                frac_log_closer=float(closer.mean()), frac_tie=float((closer == 0.5).mean()),
                I=float(a.mean() / 4 / sd), I_se=float(bI.std()), I_ci=[float(np.percentile(bI, 2.5)), float(np.percentile(bI, 97.5))],
                I_pred=Ip_s, I_pred_se=float(bP.std()),
                beta=float(a.mean() / 4 / sd / Ip_s) if abs(Ip_s) > 1e-12 else None,
                beta_se=float(bB.std()) if len(bB) > 10 else None,
                beta_ci=[float(np.percentile(bB, 2.5)), float(np.percentile(bB, 97.5))] if len(bB) > 10 else None,
                beta_regression=float(qq.sum() / max(rr.sum(), 1e-30)),
                step_c=float(s1 / s0) if abs(s0) > 1e-9 else None, step_k=float(t1 / t0_) if abs(t0_) > 1e-9 else None, cells=cells)


def live_units(RR, st):
    sd = RR[st].std(0); on = (RR[st] > 0).mean(0)
    return [int(u) for u in np.flatnonzero((sd > 1e-6) & (on > 0.02))], sd


def factorial_block(RR, base, rdrs, tag):
    F = dict(units={}, readout={})
    for st in STG:
        units, sd = live_units(base, st); F['units'][st] = {}
        for u in units:
            V = cube(lambda ck: RR[ck][st][:, u], inr)
            F['units'][st][str(u)] = fstats(V, float(sd[u]), WB[:, inr])
    sel = inr & is_te
    for st in ('S', 'h4', 'h5', 'z'):
        row = {}
        for nm, fn, bsd in (('logm', lambda ck: rdrs[(st, 'logm')](RR[ck][st]), None),
                            ('m_gev', lambda ck: rdrs[(st, 'm')](RR[ck][st]), None),
                            ('exp_logm', lambda ck: np.exp(rdrs[(st, 'logm')](RR[ck][st])), None)):
            v0 = fn((1.0, 1.0)); sdv = float(v0[te].std())
            row[nm] = fstats(cube(fn, sel), sdv, WB[:, sel])
            if nm == 'logm':      # interaction in log units against the sum's prediction (product predicts 0)
                V = cube(fn, sel); lam = float(np.median(M['M0']) / np.median(M['M2']))
                I_log = []; I_sum = []
                for ci, ki in CORNERS:
                    c, k = CS[ci], KS[ki]
                    I_log.append(V[ci, ki] - V[ci, 1] - V[1, ki] + V[1, 1])
                    Vs = lambda cc, kk: np.log(cc * lam * M['M0'][sel] + kk * kk * M['M2'][sel])
                    I_sum.append(Vs(c, k) - Vs(c, 1) - Vs(1, k) + Vs(1, 1))
                I_log = np.stack(I_log, 1); I_sum = np.stack(I_sum, 1)
                row['logm_interaction'] = dict(I_log_mean_per_cell={f'c{CS[ci]}k{KS[ki]}': float(I_log[:, i].mean()) for i, (ci, ki) in enumerate(CORNERS)},
                                               I_log_se_per_cell={f'c{CS[ci]}k{KS[ki]}': float(I_log[:, i].std() / np.sqrt(len(I_log))) for i, (ci, ki) in enumerate(CORNERS)},
                                               I_sum_pred_per_cell={f'c{CS[ci]}k{KS[ki]}': float(I_sum[:, i].mean()) for i, (ci, ki) in enumerate(CORNERS)},
                                               beta_vs_sum=float((I_log * I_sum).sum() / max((I_sum ** 2).sum(), 1e-30)))
                row['logm_interaction']['transfer_dlogm'] = {f'c{c}k{k}': float((fn((c, k))[sel] - v0[sel]).mean()) for (c, k) in X}
                row['logm_interaction']['true_dlogm'] = {f'c{c}k{k}': float((mom[(c, k)]['logm'][sel] - M['logm'][sel]).mean()) for (c, k) in X}
        F['readout'][st] = row
    n_live = {st: len(F['units'][st]) for st in STG}
    log(f'A [{tag}]: live units {n_live}; per-unit statistics done')
    return F


res['A']['trained'] = factorial_block(RA, R0, readers, 'trained')
res['A']['untrained'] = factorial_block(RAU, U0, readers_u, 'untrained')


def propagation(RR, F, net_):
    """On a downstream unit's on-region (the jets with the unit > 0 in all 9 cells) the unit is linear in the upstream stage, so its
    interaction equals W·I(upstream) up to the layer's input rounding.  Upstream units that stay on in all 9 cells pass their own
    interaction on (inherited); upstream units that switch off in some cell make new interaction (switch)."""
    out = {}
    for up, dn, key in (('S', 'h4', 'd4'), ('h4', 'h5', 'd5'), ('h5', 'z', 'd6')):
        Wm = net_.KB[key][0]                                                                  # (in, out)
        Vup = np.stack([np.stack([RR[(c, k)][up][inr] for k in KS], 0) for c in CS], 0)      # (3, 3, n, in)
        I_up = np.stack([Vup[ci, ki] - Vup[ci, 1] - Vup[1, ki] + Vup[1, 1] for ci, ki in CORNERS], 1)   # (n, 4, in)
        on_up = (Vup > 0).all((0, 1))
        Vdn = np.stack([np.stack([RR[(c, k)][dn][inr] for k in KS], 0) for c in CS], 0)
        I_dn = np.stack([Vdn[ci, ki] - Vdn[ci, 1] - Vdn[1, ki] + Vdn[1, 1] for ci, ki in CORNERS], 1)
        on_dn = (Vdn > 0).all((0, 1)); rows = {}
        for u_s, d in F['units'][dn].items():
            if 'I' not in d: continue
            u = int(u_s); jets = on_dn[:, u]
            if jets.sum() < 200: continue
            s = np.array([d['cells'][f'c{CS[ci]}k{KS[ki]}']['sign'] for ci, ki in CORNERS])
            pred = I_up[jets] @ Wm[:, u]; inh = (I_up[jets] * on_up[jets][:, None, :]) @ Wm[:, u]; sw = pred - inh
            meas = I_dn[jets][:, :, u]; sd = d['sd']
            rows[u_s] = dict(n=int(jets.sum()), I=float((meas * s).mean() / sd), I_from_upstream=float((pred * s).mean() / sd),
                             I_inherited=float((inh * s).mean() / sd), I_switch=float((sw * s).mean() / sd),
                             resid_median_abs=float(np.median(np.abs(meas - pred)) / sd), resid_max_abs=float(np.abs(meas - pred).max() / sd),
                             n_upstream_switching_mean=float((~on_up[jets]).sum(1).mean()))
        out[dn] = rows
    return out


res['A']['trained']['propagation'] = propagation(RA, res['A']['trained'], net)
res['A']['untrained']['propagation'] = propagation(RAU, res['A']['untrained'], unt)
for dn in ('h4', 'h5', 'z'):
    P_ = res['A']['trained']['propagation'][dn]
    top = sorted(P_.items(), key=lambda kv: -abs(kv[1]['I']))[:8]
    log(f'A propagation into {dn} (sd units, projected on the product sign): I measured = W·I(upstream) [residual median|max] = inherited (upstream on in all cells) + switch')
    for u_s, r_ in top + [(str(u), P_[str(u)]) for st_, u in MASSLIKE if st_ == dn and str(u) in P_ and (str(u), P_[str(u)]) not in top]:
        log(f'   {dn}{u_s:<3s} n {r_["n"]:5d}: {r_["I"]:+.4f} = {r_["I_from_upstream"]:+.4f} [{r_["resid_median_abs"]:.4f}|{r_["resid_max_abs"]:.3f}] '
            f'= {r_["I_inherited"]:+.4f} + {r_["I_switch"]:+.4f}; upstream units switching per jet {r_["n_upstream_switching_mean"]:.1f}')


def fmt(v, p='.3f'):
    return '  n/a' if v is None else format(v, p)


def show_units(F, tag):
    log(f'A [{tag}] per unit (sd units): n | e_c e_k rho | med|U−lin| med|U−log| frac log closer | I ± se (pred) | beta ± se | step c, k')
    for st in STG:
        rows = [(int(u), d) for u, d in F['units'][st].items() if 'I' in d]
        rows.sort(key=lambda r: -abs(r[1]['I_pred']))
        for u, d in rows[:14 if st != 'z' else 16]:
            log(f'   {st}{u:<3d} n {d["n"]:5d} | {d["e_c"]:+.2f} {d["e_k"]:+.2f} {fmt(d["rho"], "+.2f")} | {d["med_abs_lin"]:.3f} {d["med_abs_log"]:.3f} '
                f'{d["frac_log_closer"]:.2f} | {d["I"]:+.4f} ± {d["I_se"]:.4f} ({d["I_pred"]:+.4f}) | {fmt(d["beta"], "+.2f")} ± {fmt(d["beta_se"], ".2f")} | '
                f'{fmt(d["step_c"], ".2f")} {fmt(d["step_k"], ".2f")}')
        skipped = [int(u) for u, d in F['units'][st].items() if 'I' not in d]
        if skipped: log(f'   {st}: {len(skipped)} live units with < 200 jets on in all cells: {skipped}')
    for st in ('S', 'h4', 'h5', 'z'):
        r_ = F['readout'][st]
        for nm in ('logm', 'm_gev', 'exp_logm'):
            d = r_[nm]
            if 'I' in d:
                log(f'   read-out {nm:8s} at {st:2s} n {d["n"]:5d} | med|U−lin| {d["med_abs_lin"]:.3f} med|U−log| {d["med_abs_log"]:.3f} frac log closer {d["frac_log_closer"]:.2f} | '
                    f'I {d["I"]:+.4f} ± {d["I_se"]:.4f} (pred {d["I_pred"]:+.4f}) | beta {fmt(d["beta"], "+.2f")} ± {fmt(d["beta_se"], ".2f")} | step {fmt(d["step_c"], ".2f")} {fmt(d["step_k"], ".2f")}')
        li = r_['logm_interaction']
        log(f'   log m̂ at {st}: interaction in log units per corner cell (product predicts 0; sum prediction in brackets): '
            + ' '.join(f'{k_} {v_:+.4f}±{li["I_log_se_per_cell"][k_]:.4f} [{li["I_sum_pred_per_cell"][k_]:+.4f}]' for k_, v_ in li['I_log_mean_per_cell'].items())
            + f' | beta vs sum {li["beta_vs_sum"]:+.2f}')
        log(f'   log m̂ at {st}: transfer Δlog m̂ (true Δlog m): ' + ' '.join(f'{k_} {v_:+.3f} ({li["true_dlogm"][k_]:+.3f})' for k_, v_ in li['transfer_dlogm'].items() if k_ != 'c1.0k1.0'))


show_units(res['A']['trained'], 'trained'); show_units(res['A']['untrained'], 'untrained')
log('A: calls per cell (in-range jets): ' + ' '.join(f'{k_}: acc {v_["acc"]:.3f} W→Z {v_["W_to_Z"]:.2f} Z→t {v_["Z_to_t"]:.2f};' for k_, v_ in res['A']['calls'].items()))

# ================================================================ B. S0 patch at matched M2
S = R0['S']
lS = lambda col, SS=S: np.log(np.maximum(SS[:, col], 0.1))
lS0, lS29, lS50 = lS(0), lS(29), lS(50)
res['B'] = dict(S_on=dict(S0=float((S[:, 0] > 0).mean()), S29=float((S[:, 29] > 0).mean()), S50=float((S[:, 50] > 0).mean()), S19=float((S[:, 19] > 0).mean())))


def pick_donors(match, tol, far_key, far, okj, rng_, chunk=500):
    """random donor per jet (−1 if none): |Δ key| < tol for every matching key, |Δ far_key| ≥ far, donor ≠ jet, both in okj"""
    D = np.full(J, -1)
    for i0 in range(0, J, chunk):
        sl = slice(i0, min(i0 + chunk, J)); m_ = np.abs(far_key[sl, None] - far_key[None, :]) >= far
        for kv in match: m_ &= np.abs(kv[sl, None] - kv[None, :]) < tol
        m_ &= okj[None, :] & okj[sl, None]
        m_[np.arange(m_.shape[0]), np.arange(i0, i0 + m_.shape[0])] = False
        r_ = rng_.random(m_.shape) * m_
        D[sl] = np.where(m_.any(1), r_.argmax(1), -1)
    return D


rngB = np.random.default_rng(SEED + 2)
okS0 = S[:, 0] > 0
D_match = pick_donors([lS29, lS50], 0.05, lS0, 0.25, okS0, rngB)
D_free = pick_donors([], 0.05, lS0, 0.25, okS0, rngB)
res['B']['donors'] = dict(matched_frac=float((D_match >= 0).mean()), unmatched_frac=float((D_free >= 0).mean()),
                          matched_dlogS0_abs_median=float(np.median(np.abs(lS0[D_match[D_match >= 0]] - lS0[D_match >= 0]))),
                          unmatched_dlogS0_abs_median=float(np.median(np.abs(lS0[D_free[D_free >= 0]] - lS0[D_free >= 0]))),
                          matched_dlogS29_abs_median=float(np.median(np.abs(lS29[D_match[D_match >= 0]] - lS29[D_match >= 0]))),
                          unmatched_dlogS29_abs_median=float(np.median(np.abs(lS29[D_free[D_free >= 0]] - lS29[D_free >= 0]))),
                          matched_dlogM2_abs_median=float(np.median(np.abs(np.log(M['M2'][D_match[D_match >= 0]] / M['M2'][D_match >= 0])))),
                          unmatched_dlogM2_abs_median=float(np.median(np.abs(np.log(M['M2'][D_free[D_free >= 0]] / M['M2'][D_free >= 0])))))
log(f'B: donors: matched (|Δlog S29|,|Δlog S50| < 0.05, |Δlog S0| ≥ 0.25) for {res["B"]["donors"]["matched_frac"]:.3f} of jets, median |Δlog S0| '
    f'{res["B"]["donors"]["matched_dlogS0_abs_median"]:.3f}, median |Δlog M2(true)| {res["B"]["donors"]["matched_dlogM2_abs_median"]:.3f}; '
    f'unmatched for {res["B"]["donors"]["unmatched_frac"]:.3f}, median |Δlog S0| {res["B"]["donors"]["unmatched_dlogS0_abs_median"]:.3f}, '
    f'median |Δlog M2| {res["B"]["donors"]["unmatched_dlogM2_abs_median"]:.3f}')


def patched(net_, base, unit, D, values=None):
    """S[unit] := values (per jet) or the donor's S[unit]; jets with D < 0 keep their own value"""
    Sp = base['S'].copy(); has = D >= 0
    Sp[has, unit] = base['S'][D[has], unit] if values is None else values[has]
    m_ = np.zeros(64); m_[unit] = 1
    return net_.run_jetwise(x, {'S': (m_, Sp)}, keep=KEEP)


LAM = float(np.median(S[:, 0]) / np.median(M['M2']))          # a·M0 = b·M2 at the medians (S0 stands for M0)
share = (LAM * S[:, 0]) / np.maximum(LAM * S[:, 0] + M['M2'], 1e-9)
res['B']['sum_model'] = dict(lambda_S0_per_M2=LAM, share_median=float(np.median(share)), share_q25=float(np.percentile(share, 25)),
                             share_q75=float(np.percentile(share, 75)))


def bins_of(xv, yv, sp, nb=6):
    """quantile bins of xv (negative and positive sides separately): mean x, mean y, se, sum prediction"""
    out = dict(x=[], y=[], se=[], sum_pred=[], n=[])
    for side in (xv < 0, xv > 0):
        if side.sum() < 10: continue
        q = np.percentile(xv[side], np.linspace(0, 100, nb + 1)); b = np.clip(np.searchsorted(q, xv[side], side='right') - 1, 0, nb - 1)
        for i in range(nb):
            sel_ = b == i
            if sel_.sum() < 5: continue
            out['x'].append(float(xv[side][sel_].mean())); out['y'].append(float(yv[side][sel_].mean()))
            out['se'].append(float(yv[side][sel_].std() / np.sqrt(sel_.sum()))); out['sum_pred'].append(float(sp[side][sel_].mean())); out['n'].append(int(sel_.sum()))
    o = np.argsort(out['x']); return {k_: [v_[i] for i in o] for k_, v_ in out.items()}


def patch_stats(Rp, base, rdrs, D, dl, tag, bins=True):
    """dl = log(S0_donor / S0_own) (the x axis), evaluated on test-half jets with a donor"""
    has = (D >= 0) & is_te; xh = 0.5 * dl[has]; n = int(has.sum())
    dy = {}; m0 = {}
    for st in ('h4', 'h5', 'z'):
        dy[st] = rdrs[(st, 'logm')](Rp[st][has]) - rdrs[(st, 'logm')](base[st][has])
        m0[st] = rdrs[(st, 'm')](base[st][has]); dy[st + '_gev'] = np.log(np.maximum(rdrs[(st, 'm')](Rp[st][has]), 5.0) / np.maximum(m0[st], 5.0))
    sp = np.log(1 + share[has] * (np.exp(dl[has]) - 1))                  # sum model: Δlog(a S0 + b M2) for the applied ΔS0
    out = dict(n=n, tertiles={}, calls={})
    groups = {'M2': M['M2'][has], 'M2_over_S0': M['M2'][has] / np.maximum(S[has, 0], 1e-3), 'S29': S[has, 29]}
    w_all = WB[:, has].astype(np.float64)
    for gname, gv in groups.items():
        qs = np.percentile(gv, [100 / 3, 200 / 3]); tert = np.digitize(gv, qs)
        out['tertiles'][gname] = {}
        for ti, tname in enumerate(('low', 'mid', 'high', 'all')):
            sel_ = np.ones(n, bool) if tname == 'all' else tert == ti
            if sel_.sum() < 30: continue
            xs_ = xh[sel_]; w = w_all[:, sel_]
            row = dict(n=int(sel_.sum()), M2_median=float(np.median(M['M2'][has][sel_])), S0_median=float(np.median(S[has, 0][sel_])),
                       share_median=float(np.median(share[has][sel_])), x_mean_abs=float(np.abs(xs_).mean()),
                       slope={}, slope_se={}, slope_ols={}, bins={})
            row['sum_pred_slope'] = float((xs_ @ sp[sel_]) / (xs_ @ xs_))
            for key, yv in dy.items():
                yy = yv[sel_]
                row['slope'][key] = float((xs_ @ yy) / (xs_ @ xs_))
                bs = (w @ (xs_ * yy)) / np.maximum(w @ (xs_ * xs_), 1e-30); row['slope_se'][key] = float(bs.std())
                A_ = np.c_[xs_, np.ones(len(xs_))]; row['slope_ols'][key] = float(np.linalg.lstsq(A_, yy, rcond=None)[0][0])
                if bins: row['bins'][key] = bins_of(xs_, yy, sp[sel_])
            out['tertiles'][gname][tname] = row
    # calls
    c0 = base['logits'].argmax(1)[has]; c1 = Rp['logits'].argmax(1)[has]; yy_ = y[has]; dlh = dl[has]
    edges = [(-np.inf, -0.5), (-0.5, -0.25), (0.25, 0.5), (0.5, np.inf)]
    qs = np.percentile(M['M2'][has], [100 / 3, 200 / 3]); tert = np.digitize(M['M2'][has], qs)
    for ti, tname in enumerate(('low', 'mid', 'high', 'all')):
        selt = np.ones(n, bool) if tname == 'all' else tert == ti; out['calls'][tname] = {}
        for lo, hi in edges:
            sb = selt & (dlh >= lo) & (dlh < hi)
            wj = sb & (yy_ == W_) & (c0 == W_); zj = sb & (yy_ == Z_) & (c0 == Z_)
            out['calls'][tname][f'[{lo},{hi})'] = dict(n=int(sb.sum()), calls_changed=float((c1[sb] != c0[sb]).mean()) if sb.sum() else None,
                                                     W_to_Z=float((c1[wj] == Z_).mean()) if wj.sum() >= 20 else None, n_W=int(wj.sum()),
                                                     Z_to_t=float((c1[zj] == T_).mean()) if zj.sum() >= 20 else None, n_Z=int(zj.sum()),
                                                     Z_to_W=float((c1[zj] == W_).mean()) if zj.sum() >= 20 else None)
    out['acc_after'] = float((c1 == yy_).mean()); out['acc_before'] = float((c0 == yy_).mean()); out['calls_changed'] = float((c1 != c0).mean())
    log(f'B [{tag}]: n {n}; acc {out["acc_before"]:.3f} → {out["acc_after"]:.3f}, calls changed {out["calls_changed"]:.3f}; slope of Δlog m̂ on ½Δlog S0 by M2 tertile '
        f'(product 1.0; sum prediction in brackets):')
    for gname in groups:
        for key in ('h4', 'h5', 'z', 'h5_gev', 'z_gev'):
            tt = out['tertiles'][gname]
            log(f'   by {gname:11s} {key:7s} ' + ' '.join(f'{tn} {tt[tn]["slope"][key]:+.3f}±{tt[tn]["slope_se"][key]:.3f} [{tt[tn]["sum_pred_slope"]:+.3f}]'
                                                         for tn in ('low', 'mid', 'high', 'all') if tn in tt))
    cc = out['calls']
    for tname in ('low', 'mid', 'high'):
        log(f'   calls M2 {tname:4s}: ' + ' '.join(f'Δlog S0 {b_}: n {v_["n"]} changed {fmt(v_["calls_changed"], ".2f")} W→Z {fmt(v_["W_to_Z"], ".2f")} (n_W {v_["n_W"]}) '
                                                  f'Z→t {fmt(v_["Z_to_t"], ".2f")} Z→W {fmt(v_["Z_to_W"], ".2f")} (n_Z {v_["n_Z"]});' for b_, v_ in cc[tname].items()))
    return out


dl_match = np.where(D_match >= 0, lS0[np.maximum(D_match, 0)] - lS0, 0.0)
dl_free = np.where(D_free >= 0, lS0[np.maximum(D_free, 0)] - lS0, 0.0)
RP = patched(net, R0, 0, D_match); res['B']['S0_matched'] = patch_stats(RP, R0, readers, D_match, dl_match, 'S0 patch, matched donors')
res['B']['S0_matched']['plot_reader'] = 'h5'
RP19 = patched(net, R0, 19, D_match)
d19 = np.where(D_match >= 0, np.log(np.maximum(S[np.maximum(D_match, 0), 19], 0.1)) - np.log(np.maximum(S[:, 19], 0.1)), 0.0)
res['B']['S19_control'] = patch_stats(RP19, R0, readers, D_match, dl_match, 'S19 patch (control), same donors; x = the donor\'s ½Δlog S0')
res['B']['S19_control']['dlogS19_abs_median'] = float(np.median(np.abs(d19[D_match >= 0])))
res['B']['S19_control']['slope_on_own_dlogS19'] = {st: float((d19[(D_match >= 0) & is_te] @ (readers[(st, 'logm')](RP19[st][(D_match >= 0) & is_te]) - readers[(st, 'logm')](R0[st][(D_match >= 0) & is_te])))
                                                              / max((d19[(D_match >= 0) & is_te] ** 2).sum(), 1e-30)) for st in ('h4', 'h5', 'z')}
log(f'   S19 control: median |Δlog S19| applied {res["B"]["S19_control"]["dlogS19_abs_median"]:.3f}; slope of Δlog m̂ on Δlog S19 itself: '
    + ', '.join(f'{k_} {v_:+.3f}' for k_, v_ in res['B']['S19_control']['slope_on_own_dlogS19'].items()))
RPf = patched(net, R0, 0, D_free); res['B']['S0_unmatched'] = patch_stats(RPf, R0, readers, D_free, dl_free, 'S0 patch, donors with M2 NOT matched', bins=False)

# supplementary: S0 × f (every jet; same statistics with dl = log f)
res['B']['S0_scale'] = {}
allj = np.arange(J)
for f_ in FSCAN:
    Rf = patched(net, R0, 0, allj, values=S[:, 0] * f_)
    res['B']['S0_scale'][str(f_)] = patch_stats(Rf, R0, readers, allj, np.full(J, np.log(f_)), f'S0 × {f_}', bins=False)

# untrained network
Su = U0['S']; sdSu = Su.std(0); onSu = (Su > 0).mean(0)
live_Su = [int(u) for u in np.flatnonzero((sdSu > 1e-6) & (onSu > 0.02))]
res['B']['untrained'] = dict(live_S_units=live_Su, S0_live=bool(0 in live_Su), S0_sd=float(sdSu[0]), S0_on=float(onSu[0]))
log(f'B [untrained]: live S units {live_Su}; S0 sd {sdSu[0]:.3g}, on {onSu[0]:.3f}')
if 0 in live_Su:
    others = [u for u in live_Su if u != 0][:2]
    lSu = lambda col: np.log(np.maximum(Su[:, col], 0.1))
    Du = pick_donors([lSu(u) for u in others], 0.05, lSu(0), 0.25, Su[:, 0] > 0, rngB)
    dlu = np.where(Du >= 0, lSu(0)[np.maximum(Du, 0)] - lSu(0), 0.0)
    res['B']['untrained']['matched_on'] = others; res['B']['untrained']['donor_frac'] = float((Du >= 0).mean())
    res['B']['untrained']['S0_patch'] = patch_stats(patched(unt, U0, 0, Du), U0, readers_u, Du, dlu, f'untrained S0 patch, donors matched on S{others}', bins=False)
else:
    # S0 is dead in the untrained network: set it to the trained donors' S0 values (off its own manifold) and read out with its own ridge
    Rpu = patched(unt, U0, 0, D_match, values=S[np.maximum(D_match, 0), 0])
    res['B']['untrained']['note'] = 'S0 dead in the untrained network; S0 set to the trained network\'s donor S0 (x = Δlog S0 of the trained donors)'
    res['B']['untrained']['S0_patch'] = patch_stats(Rpu, U0, readers_u, D_match, dl_match, 'untrained network, S0 := trained donor S0', bins=False)

res['wall_s'] = float(time.time() - t0)
OUT_JSON.write_text(json.dumps(res))
log(f'done -> {OUT_JSON}')
