"""E15 (JEDI-linear 3-feature n64; claims A4, A9): product or sum — do h4/h5 multiply M0 = ΣpT and M2 = Σ pT r², or add them?

Sample: sample(per_class=2000, seed=11) = 10,000 test jets; exact fixed-point network (interp.jedi_core.Jedi), stages S, h4,
h5, z, logits; control network Jedi(untrained=True, seed=0).  Only real particles are edited (padded slots stay (0, 0, 0)); pT
order unchanged; jets whose edited input leaves the l1 quantizer range in any cell are dropped (e5_causal.in_range).
Hypotheses, as functions of the true moments: sum V = λ·M0 + M2 with λ = median M2 / median M0 (the two terms equal at the
sample medians, which reproduces E6's slope ratio ρ = 1); product V = √(M0·M2).

A   factorial edit as planned: pT × c (c ∈ {0.8, 1, 1.25}) × angles × k (k ∈ {0.8, 1, 1.25}), the 9 cells.
    M0 → c·M0, M2 → c·k²·M2 (pT × c scales M2 as well), m → c·k·m.  Every U that is homogeneous in pT is exactly log-additive
    in (c, k); the sum and the product are both homogeneous of degree 1 (sum → c·(λM0 + k²M2), product → c·k·√(M0·M2)), so this
    design measures pT-homogeneity and gives the same prediction for both hypotheses (the 'models' rows show it).
A2  separable factorial (added): pT × c, angles × k/√c.  M0 → c·M0 and M2 → k²·M2 exactly (c moves M0 at fixed M2, k moves M2
    at fixed M0), m → √c·k·m.  The sum is then exactly linear-additive in (c, k) and the product exactly log-additive.
    Per live unit U of S, h4, h5, z (and per mass read-out), on jets with U > 0 in all 9 cells, the four corner cells
    (c, k ≠ 1) are compared with the nulls built from the single-factor cells:
      U_lin = U(c,1) + U(1,k) − U(1,1)       exact for the sum in A2
      U_log = U(c,1)·U(1,k) / U(1,1)         exact for the product in A2 (and for both in A)
    Interaction I = U(c,k) − U_lin; the product predicts I = U_log − U_lin = ΔU_c·ΔU_k / U(1,1), the sum predicts 0.
    Per unit, in units of the unit's sd on the unedited sample: median |U − U_lin|, median |U − U_log|, the fraction of
    jet × corner pairs where U_log is closer, I projected on the product's sign s = sign((c−1)(k²−1))·sign(e_c·e_k) (mean over
    jets and corners; 1,000-resample Poisson bootstrap over jets), the product's prediction of the same, and β = I / I_pred.
    Read-out log m̂ (ridge of log m on h4, h5, z, fit on the unedited train half as E6, evaluated on the test half): interaction
    in log units per corner; product 0; sum log(λM0 + M2) on the true moments (−0.025 at the corners); β_sum = measured / sum
    prediction, both projected on −sign((c−1)(k−1)) (1 = sum, 0 = product).
    Controls: the live S units, the model rows (sum, product, true m on the true moments of each cell), the untrained network.
    Propagation: on a downstream unit's on-region it is linear in the upstream stage, so its I = W·I(upstream) up to input
    rounding; split into the part from upstream units on in all cells (inherited) and from units that switch (switch).
B   S0 patch at matched M2 (run_jetwise, S0 replaced by a donor jet's S0): donor = another jet of the sample with
    |Δlog S29| < 0.05, |Δlog S50| < 0.05 and |Δlog S0| ≥ 0.25, chosen at random; jets without a donor keep S0.
    Metric: slope of Δlog m̂ (read-outs at h4, h5, z) on ½Δlog S0 within tertiles of M2 (true Σ pT r²; edges from all 10,000 jets),
    product prediction 1 in every tertile (S0 the only M0 carrier); sum prediction log(1 + s_j(e^{Δlog S0} − 1)), s_j = λ_S·S0 /
    (λ_S·S0 + M2), λ_S = median M2 / median S0; W→Z and Z→t call switches per tertile and Δlog S0 bin.  Controls: S19 (soft
    multiplicity) patched with the same donors' S19 (x axis: the donor's ½Δlog S0); donors with M2 not matched; the S0 patch
    on the untrained network; supplementary S0 × f, f ∈ {0.7, 0.85, 1.2, 1.4}.
T   response by M2 tertile (A and A2 cells, B patch; test half): per jet, Δlog m̂ / Δlog c under pT × c at k = 1 (A: both
    hypotheses 1), under angles × k at c = 1 (M2 only: product 1, sum ≈ 2(1 − s)) and under pT × c at fixed M2 (A2: product ½,
    sum ≈ s); each divided by the pT × c response of the same tertile (removes a tertile-dependent read-out gain); the B slope
    divided the same way.  Product: flat across tertiles; sum: M0-only response falls and M2-only response rises with M2.
Run 1 (2026-10-16, 20:22 UTC, job e15-run) had λ inverted (share of the M0 term 0.87 instead of 0.5 at the medians) and design A
only; its json is kept as e15_product_run1.json.
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
MIN_N = 30
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_, T_ = 2, 3, 4
MASSLIKE = [('h4', 18), ('h5', 0), ('h5', 28), ('z', 9), ('z', 11), ('z', 12)]     # E6's mass-like units
MODELS = ('sum', 'prod', 'm')
MODEL_LABEL = {'sum': 'sum λM0 + M2', 'prod': 'product √(M0·M2)', 'm': 'true m'}
DESIGNS = {'A': 'pT × c, angles × k (as planned): M0 × c, M2 × c·k²',
           'A2': 'pT × c, angles × k/√c (separable): M0 × c, M2 × k²'}
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
    from matplotlib.lines import Line2D
    R = json.loads(OUT_JSON.read_text())
    e6 = {}
    for p in (OUT_DIR / 'e6_mass.json', ROOT / 'results' / 'jedi' / 'e6_mass.json'):
        if p.exists():
            e6 = json.loads(p.read_text())['units']; break
    surf, ink, ink2, grid = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df'
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), facecolor=surf, gridspec_kw=dict(width_ratios=[1.05, 1.05, 0.9]))
    for a in ax:
        a.set_facecolor(surf); a.grid(True, color=grid, lw=0.8); a.set_axisbelow(True)
        for s in ('top', 'right'): a.spines[s].set_visible(False)
        for s in ('left', 'bottom'): a.spines[s].set_color(grid)
        a.tick_params(colors=ink2, labelsize=9)
    # (a) per-unit interaction vs the product's prediction, separable design A2
    a = ax[0]; xs = []; ys = []
    D2 = R['A2']
    for tag in ('untrained', 'trained'):
        F = D2[tag]
        for st in STG:
            for u, d in F['units'][st].items():
                if 'I' not in d: continue
                col = PAL[st] if tag == 'trained' else PAL['untrained']
                a.errorbar(d['I_pred'], d['I'], yerr=d['I_se'], fmt='o', ms=5 if tag == 'trained' else 4,
                           color=col, mfc=col if tag == 'trained' else surf, mew=1.1, ecolor=col, elinewidth=0.8, alpha=0.85, zorder=3)
                xs.append(d['I_pred']); ys.append(d['I'])
                if tag == 'trained' and (st, int(u)) in MASSLIKE:
                    a.annotate(f'{st}_{u}', (d['I_pred'], d['I']), xytext=(4, 3), textcoords='offset points', color=ink, fontsize=8)
    lo_ = min(xs + ys + [0.0]); hi_ = max(xs + ys + [0.05])
    lim = (lo_ - 0.06 * (hi_ - lo_), hi_ + 0.06 * (hi_ - lo_))
    a.plot(lim, lim, color=ink, lw=1.3, ls='--'); a.axhline(0, color=ink2, lw=1.1, ls='-.')
    a.text(0.62 * lim[1], 0.62 * lim[1] - 0.04 * (lim[1] - lim[0]), 'product: I = U_log − U_lin', color=ink, fontsize=8.5, ha='left', va='top', rotation=0)
    a.text(lim[1], 0.01 * (lim[1] - lim[0]), 'sum: I = 0 ', color=ink2, fontsize=8.5, ha='right', va='bottom')
    a.set_xlim(*lim); a.set_ylim(*lim)
    a.set_xlabel('interaction a product predicts, ΔU_c·ΔU_k / U   [sd of the unit]', color=ink2, fontsize=9.5)
    a.set_ylabel('measured interaction U(c,k) − U(c,1) − U(1,k) + U(1,1)   [sd]', color=ink2, fontsize=9.5)
    a.set_title('(a) per unit: M0 × c at fixed M2, M2 × k² at fixed M0', color=ink, fontsize=10, loc='left')
    hs = [Line2D([], [], marker='o', ls='', color=PAL[s], label=f'{s} units') for s in STG]
    hs += [Line2D([], [], marker='o', ls='', color=PAL['untrained'], mfc=surf, label='untrained network')]
    a.legend(handles=hs, frameon=False, fontsize=8, loc='upper left', labelcolor=ink)
    # (b) S0 patch at matched M2
    a = ax[1]; B = R['B']['S0_matched']; rd = B.get('plot_reader', 'h5')
    cols = ['#1f77b4', '#2ca02c', '#d62728']; mk = ['o', 's', '^']
    xr = []
    for i, tname in enumerate(('low', 'mid', 'high')):
        t = B['tertiles']['M2'][tname]; bn = t['bins'][rd]
        a.errorbar(bn['x'], bn['y'], yerr=bn['se'], fmt=mk[i] + '-', color=cols[i], ms=5, lw=1.6, mfc=surf, mew=1.5,
                   label=f'M2 {tname} third (median {t["M2_median"]:.1f} GeV): slope {t["slope"][rd]:.2f} ± {t["slope_se"][rd]:.2f}')
        a.plot(bn['x'], bn['sum_pred'], color=cols[i], lw=1.1, ls=':')
        xr += [bn['x'][0], bn['x'][-1]]
    C = R['B']['S19_control']; bn = C['tertiles']['M2']['all']['bins'][rd]
    a.errorbar(bn['x'], bn['y'], yerr=bn['se'], fmt='x', color=PAL['untrained'], ms=6, lw=1.2,
               label=f'control, S19 patched instead: slope {C["tertiles"]["M2"]["all"]["slope"][rd]:.3f}')
    xx = np.array([min(xr), max(xr)])
    a.plot(xx, xx, color=ink, lw=1.3, ls='--'); a.text(xx[1], xx[1], 'product, S0 the only M0 carrier ', color=ink, fontsize=8.5, ha='right', va='bottom')
    a.plot([], [], color=ink2, ls=':', label='sum λ·S0 + M2, per third (dotted)')
    a.axhline(0, color=ink2, lw=0.8)
    a.set_xlabel('½ Δlog S0 set by the patch (donor S0 / own S0)', color=ink2, fontsize=9.5)
    a.set_ylabel(f'Δ log m̂   (read-out at {rd})', color=ink2, fontsize=9.5)
    a.set_title('(b) S0 replaced by a donor\'s with S29, S50 within 5 %', color=ink, fontsize=10, loc='left')
    a.legend(frameon=False, fontsize=7.8, loc='lower right', labelcolor=ink)
    # (c) responses by M2 tertile, normalised by the pT × c response
    a = ax[2]; T = R['T']['sources']; xt = np.arange(3); tn = ('low', 'mid', 'high')
    for key, lab, col, off in (('c0_over_c', 'M0 × c at fixed M2', '#1f77b4', -0.08), ('k_over_c', 'M2 × k² at fixed M0', '#d62728', 0.08)):
        for src, mk_, ls_ in (('logm_h5', 'o', '-'), ('logm_z', 's', '-')):
            v = [T[src][t][key] for t in tn]; e = [T[src][t][key + '_se'] for t in tn]
            a.errorbar(xt + off, v, yerr=e, fmt=mk_ + ls_, color=col, ms=5, lw=1.5, mfc=surf if src == 'logm_z' else col, mew=1.3,
                       label=f'{lab}: m̂ at {src[-2:].strip("_")}')
        a.plot(xt, [T['model_sum'][t][key] for t in tn], color=col, lw=1.2, ls=':')
        a.plot(xt, [T['model_prod'][t][key] for t in tn], color=col, lw=1.2, ls='--')
    a.plot([], [], color=ink2, ls='--', label='product √(M0·M2)'); a.plot([], [], color=ink2, ls=':', label='sum λM0 + M2')
    a.set_xticks(xt); a.set_xticklabels([f'{t} M2\n({T["model_sum"][t]["M2_median"]:.1f} GeV)' for t in tn], color=ink2, fontsize=8.5)
    a.set_ylabel('Δlog m̂ per Δlog c (or Δlog k), ÷ response to pT × c', color=ink2, fontsize=9.5)
    a.set_title('(c) one moment moved, by M2 third', color=ink, fontsize=10, loc='left')
    a.legend(frameon=False, fontsize=7.6, loc='best', labelcolor=ink)
    fig.suptitle(f'JEDI-linear 3-feature n64, {R["N"]} test jets (seed {R["seed"]}), accuracy {R["acc"]:.3f}; '
                 'M2 = Σ pT·r² in GeV (r in rad); sd = the unit\'s spread on the unedited jets', color=ink2, fontsize=9, x=0.01, ha='left', y=0.995)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=140, facecolor=surf); print('wrote', OUT_PNG)

    # table of the top units (design A2) with design A's interaction alongside
    rows = []
    for st in STG:
        for u, d in D2['trained']['units'][st].items():
            if 'I' in d: rows.append((st, str(u), d))
    rows.sort(key=lambda r: -abs(r[2]['I_pred']))
    ml = [(s, str(u)) for s, u in MASSLIKE]
    top = [r for r in rows if (r[0], r[1]) in ml] + [r for r in rows if (r[0], r[1]) not in ml][:12]
    top += [(f'm̂ ({st})', '', D2['trained']['readout'][st]['m_gev']) for st in ('h4', 'h5', 'z') if 'I' in D2['trained']['readout'][st]['m_gev']]
    top += [(MODEL_LABEL[nm], '', D2['trained']['models'][nm]) for nm in ('sum', 'prod')]
    f = lambda v, p='.3f': ('–' if v is None else format(v, p))
    hdr = ['unit', 'stage', 'n jets', 'ρ (E6)', 'med |U−U_lin|', 'med |U−U_log|', 'frac log closer', 'I pred (product)', 'I ± se', 'β = I / I pred', 'I ± se, design A']
    cells = []
    for st, u, d in top:
        rho6 = e6.get(st, {}).get(u, {}).get('rho') if u != '' else None
        dA = R['A']['trained']['units'].get(st, {}).get(u, {}) if u != '' else {}
        cells.append([f'{st}_{u}' if u != '' else st, st if u != '' else '–', str(d['n']), f(rho6, '.2f'), f(d['med_abs_lin']), f(d['med_abs_log']),
                      f(d['frac_log_closer'], '.2f'), f(d['I_pred']), f'{d["I"]:.3f} ± {d["I_se"]:.3f}', f(d['beta'], '.2f'),
                      f'{dA["I"]:.3f} ± {dA["I_se"]:.3f}' if 'I' in dA else '–'])
    fig2, a2 = plt.subplots(figsize=(14, 0.27 * len(cells) + 0.8), facecolor=surf); a2.axis('off')
    tb = a2.table(cellText=cells, colLabels=hdr, loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.25)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(grid); c.set_text_props(color=ink)
        if i == 0: c.set_facecolor('#efeeea'); c.set_text_props(weight='bold')
        elif (top[i - 1][0], top[i - 1][1]) in ml: c.set_facecolor('#f3f7ff')
    a2.set_title('E15, design A2 (M0 × c at fixed M2, M2 × k² at fixed M0); interaction in sd of the unit; product: β = 1, sum: β = 0. '
                 f'{R["N"]} jets, seed {R["seed"]}', color=ink, fontsize=9.5, loc='left')
    fig2.tight_layout(); fig2.savefig(OUT_TAB, dpi=150, facecolor=surf); print('wrote', OUT_TAB)


if '--plot' in sys.argv:
    plot(); sys.exit(0)

# ---------------------------------------------------------------- main
from interp.jedi_core import Jedi, sample  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
x, y, Q = sample(per_class=PER, seed=SEED); x = x.astype(np.float32); J = len(x)
del Q
real = x[..., 0] > 0
net = Jedi(64); unt = Jedi(64, untrained=True, seed=0)
res = dict(N=J, per_class=PER, seed=SEED, cells_c=CS, cells_k=KS, nboot=NBOOT, min_n=MIN_N, designs=DESIGNS,
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
    """pT × c and (Δη, Δφ) × k on real particles only"""
    e = xx.copy()
    e[..., 0] = np.where(real, e[..., 0] * np.float32(c), e[..., 0])
    e[..., 1:] = np.where(real[..., None], e[..., 1:] * np.float32(k), e[..., 1:])
    return e


M = moments(x)
LAM_M = float(np.median(M['M2']) / np.median(M['M0']))          # sum model λ·M0 + M2: equal terms at the sample medians
share_M = LAM_M * M['M0'] / (LAM_M * M['M0'] + M['M2'])
T_EDGES = np.percentile(M['M2'], [100 / 3, 200 / 3]); TERT = np.digitize(M['M2'], T_EDGES)
res['sum_model'] = dict(lambda_M0=LAM_M, share_median=float(np.median(share_M)), share_q25=float(np.percentile(share_M, 25)),
                        share_q75=float(np.percentile(share_M, 75)), M0_median=float(np.median(M['M0'])), M2_median=float(np.median(M['M2'])),
                        M2_tertile_edges=[float(v) for v in T_EDGES],
                        share_median_by_M2_tertile=[float(np.median(share_M[TERT == t])) for t in range(3)])
log(f'sum model: λ = {LAM_M:.4g} (median M2 {np.median(M["M2"]):.3g} GeV, median M0 {np.median(M["M0"]):.4g} GeV); share of the M0 term median '
    f'{np.median(share_M):.3f} (by M2 tertile {", ".join(f"{v:.3f}" for v in res["sum_model"]["share_median_by_M2_tertile"])}); M2 tertile edges {T_EDGES}')


def models(mm):
    return dict(sum=LAM_M * mm['M0'] + mm['M2'], prod=np.sqrt(mm['M0'] * mm['M2']), m=np.maximum(mm['m'], 1e-3))


R0 = net.run(x, keep=KEEP); call0 = R0['logits'].argmax(1)
res['acc'] = float((call0 == y).mean())
U0 = unt.run(x, keep=KEEP); res['acc_untrained'] = float((U0['logits'].argmax(1) == y).mean())
log(f'accuracy trained {res["acc"]:.4f}, untrained {res["acc_untrained"]:.4f}')
chk = net.run_jetwise(x, {'S': (np.ones(64), R0['S'])}, keep=KEEP)
res['jetwise_identity_maxdiff'] = {k: float(np.abs(chk[k] - R0[k]).max()) for k in KEEP}
del chk
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
        return ((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.w * self.ts + self.tm


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

# bootstrap weights (Poisson), shared across units and analyses
WB = np.random.default_rng(SEED + 1).poisson(1.0, size=(NBOOT, J)).astype(np.uint8)


def cube(getter, jets):
    """(3, 3, n) float64 array of one quantity in the 9 cells on the given jets (axis 0: c, axis 1: k)"""
    return np.stack([np.stack([np.asarray(getter((c, k))[jets], np.float64) for k in KS], 0) for c in CS], 0)


def inter(V, ci, ki):
    return V[ci, ki] - V[ci, 1] - V[1, ki] + V[1, 1]


def fstats(V, sd, wb, min_n=MIN_N):
    """factorial statistics of one quantity: V (3, 3, n) cell values; sd = its spread on the unedited sample; wb (NBOOT, n) weights"""
    ok = (V > 0).all((0, 1)); n = int(ok.sum()); out = dict(n=n, n_offered=int(V.shape[2]))
    if n < min_n or sd <= 1e-9: return out
    V = V[:, :, ok]; w = wb[:, ok].astype(np.float64); wsum = np.maximum(w.sum(1), 1.0)
    B = V[1, 1]
    dc = V[2, 1] - V[0, 1]; dk = V[1, 2] - V[1, 0]
    ec = float(dc.mean() / LSTEP / sd); ek = float(dk.mean() / LSTEP / sd)
    ssign = np.sign(ec * ek) if abs(ec * ek) > 1e-12 else 1.0
    a = np.zeros(n); p = np.zeros(n); absl = []; absg = []; closer = []; cells = {}
    for ci, ki in CORNERS:
        c, k = CS[ci], KS[ki]
        U = V[ci, ki]; lin = V[ci, 1] + V[1, ki] - B; logn = V[ci, 1] * V[1, ki] / B
        Im = U - lin; Ip = logn - lin
        s = np.sign((c - 1) * (k * k - 1)) * ssign
        a += Im * s; p += Ip * s
        absl.append(np.abs(Im)); absg.append(np.abs(U - logn))
        closer.append(np.where(Ip == 0, 0.5, (np.abs(U - logn) < np.abs(Im)).astype(float)))
        cells[f'c{c}k{k}'] = dict(I=float(Im.mean() / sd), I_se=float(Im.std() / np.sqrt(n) / sd), I_pred=float(Ip.mean() / sd),
                                  sign=float(s), rel_dU_c=float(np.median((V[ci, 1] - B) / B)), rel_dU_k=float(np.median((V[1, ki] - B) / B)))
    absl = np.concatenate(absl); absg = np.concatenate(absg); closer = np.concatenate(closer)
    bI = (w @ a) / (4 * wsum) / sd; bP = (w @ p) / (4 * wsum) / sd
    Ip_s = float(p.mean() / 4 / sd); I_s = float(a.mean() / 4 / sd)
    okb = np.abs(bP) > 1e-12; bB = bI[okb] / bP[okb]
    s1 = (V[2, 1] - B).mean(); s0 = (B - V[0, 1]).mean(); t1 = (V[1, 2] - B).mean(); t0_ = (B - V[1, 0]).mean()
    return dict(n=n, n_offered=int(V.shape[2]), sd=float(sd), e_c=ec, e_k=ek, rho=float(ek / ec) if abs(ec) > 1e-9 else None,
                U_over_sd_median=float(np.median(B) / sd),
                med_abs_lin=float(np.median(absl) / sd), med_abs_log=float(np.median(absg) / sd),
                frac_log_closer=float(closer.mean()), frac_tie=float((closer == 0.5).mean()),
                I=I_s, I_se=float(bI.std()), I_ci=[float(np.percentile(bI, 2.5)), float(np.percentile(bI, 97.5))],
                I_over_se=float(I_s / bI.std()) if bI.std() > 0 else None,
                I_pred=Ip_s, I_pred_se=float(bP.std()),
                beta=float(I_s / Ip_s) if abs(Ip_s) > 1e-6 else None,
                beta_se=float(bB.std()) if (len(bB) > 10 and abs(Ip_s) > 1e-6) else None,
                step_c=float(s1 / s0) if abs(s0) > 1e-9 else None, step_k=float(t1 / t0_) if abs(t0_) > 1e-9 else None, cells=cells)


def log_interaction(Vl, Vs, Vp, wb):
    """interaction of a log quantity (3, 3, n) per corner, against the sum model's (Vs) and the product's (Vp) log values on the
    same jets; pooled projection on σ = −sign((c−1)(k−1)) (the sign of the sum's log interaction in A2); β_sum = I / I_sum"""
    w = wb.astype(np.float64); wsum = np.maximum(w.sum(1), 1.0); n = Vl.shape[2]
    out = dict(n=int(n), I_log_mean_per_cell={}, I_log_se_per_cell={}, I_sum_pred_per_cell={}, I_prod_pred_per_cell={})
    a = np.zeros(n); b = np.zeros(n); pp = np.zeros(n)
    for ci, ki in CORNERS:
        key = f'c{CS[ci]}k{KS[ki]}'; sg = -np.sign((CS[ci] - 1) * (KS[ki] - 1))
        Il = inter(Vl, ci, ki); Is = inter(Vs, ci, ki); Ip = inter(Vp, ci, ki)
        out['I_log_mean_per_cell'][key] = float(Il.mean()); out['I_log_se_per_cell'][key] = float(Il.std() / np.sqrt(n))
        out['I_sum_pred_per_cell'][key] = float(Is.mean()); out['I_prod_pred_per_cell'][key] = float(Ip.mean())
        a += sg * Il / 4; b += sg * Is / 4; pp += sg * Ip / 4
    bI = (w @ a) / wsum; bS = (w @ b) / wsum
    out.update(I_proj=float(a.mean()), I_proj_se=float(bI.std()), I_sum_proj=float(b.mean()), I_prod_proj=float(pp.mean()),
               mean_abs_I_per_cell=float(np.mean([abs(v) for v in out['I_log_mean_per_cell'].values()])))
    if abs(b.mean()) > 1e-4:
        out['beta_sum'] = float(a.mean() / b.mean()); out['beta_sum_se'] = float((bI / bS).std())
    else:
        out['beta_sum'] = None; out['beta_sum_se'] = None
    return out


def live_units(RR, st):
    sd = RR[st].std(0); on = (RR[st] > 0).mean(0)
    return [int(u) for u in np.flatnonzero((sd > 1e-6) & (on > 0.02))], sd


def run_cells(net_, base, X):
    out = {}
    for ck, xe in X.items():
        out[ck] = base if ck == (1.0, 1.0) else {k_: v_.astype(np.float32) for k_, v_ in net_.run(xe, keep=KEEP).items()}
    return out


def build(dname):
    ang = (lambda c, k: k) if dname == 'A' else (lambda c, k: k / np.sqrt(c))
    X = {}; inr_ = np.ones(J, bool)
    for c in CS:
        for k in KS:
            X[(c, k)] = edit(x, c, ang(c, k)) if (c, k) != (1.0, 1.0) else x
            inr_ &= in_range(net, X[(c, k)])
    info = dict(edit=DESIGNS[dname], angle_factor={f'c{c}k{k}': float(ang(c, k)) for c in CS for k in KS},
                in_range_all_cells=float(inr_.mean()), n_in_range=int(inr_.sum()),
                padded_moved_max=float(max(np.abs(xe[~real] - x[~real]).max() for xe in X.values())),
                order_kept_frac=float(min((np.diff(xe[..., 0], axis=1) <= 1e-6).all(1).mean() for xe in X.values())))
    mom_ = {ck: moments(xe) for ck, xe in X.items()}
    info['moment_ratios'] = {f'c{c}k{k}': dict(M0=float(np.median(mom_[(c, k)]['M0'] / M['M0'])), M2=float(np.median(mom_[(c, k)]['M2'] / M['M2'])),
                                               m=float(np.median(mom_[(c, k)]['m'] / np.maximum(M['m'], 1e-3))))
                             for (c, k) in X}
    log(f'{dname} [{DESIGNS[dname]}]: 9 cells built; jets inside the input range in all cells {inr_.mean():.4f} ({inr_.sum()}); padded slots moved by '
        f'at most {info["padded_moved_max"]:.2g}; pT order kept in {info["order_kept_frac"]:.4f} of jets; median moment ratios: '
        + ' '.join(f'{k_} M0 {v_["M0"]:.3f} M2 {v_["M2"]:.3f} m {v_["m"]:.3f};' for k_, v_ in info['moment_ratios'].items()))
    RA = run_cells(net, R0, X); RAU = run_cells(unt, U0, X)
    del X
    log(f'{dname}: 16 edited forward passes done')
    info['calls'] = {f'c{c}k{k}': dict(acc=float((RA[(c, k)]['logits'].argmax(1)[inr_] == y[inr_]).mean()),
                                       W_to_Z=float((RA[(c, k)]['logits'].argmax(1)[inr_ & (y == W_)] == Z_).mean()),
                                       Z_to_t=float((RA[(c, k)]['logits'].argmax(1)[inr_ & (y == Z_)] == T_).mean())) for (c, k) in RA}
    return info, inr_, mom_, RA, RAU


def factorial_block(RR, base, rdrs, tag, inr_, mom_):
    F = dict(units={}, readout={}, models={})
    for st in STG:
        units, sd = live_units(base, st); F['units'][st] = {}
        for u in units:
            F['units'][st][str(u)] = fstats(cube(lambda ck: RR[ck][st][:, u], inr_), float(sd[u]), WB[:, inr_])
    sel = inr_ & is_te
    MV = {ck: models(mm) for ck, mm in mom_.items()}
    for nm in MODELS:
        v0 = MV[(1.0, 1.0)][nm]
        F['models'][nm] = fstats(cube(lambda ck: MV[ck][nm], sel), float(v0[te].std()), WB[:, sel])
    Vs = np.log(cube(lambda ck: MV[ck]['sum'], sel)); Vp = np.log(cube(lambda ck: MV[ck]['prod'], sel))
    for nm in MODELS:
        F['models'][nm]['log_interaction'] = log_interaction(np.log(cube(lambda ck: MV[ck][nm], sel)), Vs, Vp, WB[:, sel])
    for st in STG:
        row = {}
        fl = lambda ck: rdrs[(st, 'logm')](RR[ck][st])
        for nm, fn in (('logm', fl), ('m_gev', lambda ck: rdrs[(st, 'm')](RR[ck][st])), ('exp_logm', lambda ck: np.exp(fl(ck)))):
            v0 = fn((1.0, 1.0)); row[nm] = fstats(cube(fn, sel), float(v0[te].std()), WB[:, sel])
        row['logm_interaction'] = li = log_interaction(cube(fl, sel), Vs, Vp, WB[:, sel])
        v0 = fl((1.0, 1.0))
        li['transfer_dlogm'] = {f'c{c}k{k}': float((fl((c, k))[sel] - v0[sel]).mean()) for (c, k) in RR}
        li['true_dlogm'] = {f'c{c}k{k}': float((mom_[(c, k)]['logm'][sel] - M['logm'][sel]).mean()) for (c, k) in RR}
        F['readout'][st] = row
    log(f'{tag}: live units { {st: len(F["units"][st]) for st in STG} }; per-unit statistics done')
    return F


def propagation(RR, F, net_, inr_):
    """On a downstream unit's on-region (the jets with the unit > 0 in all 9 cells) the unit is linear in the upstream stage, so its
    interaction equals W·I(upstream) up to the layer's input rounding.  Upstream units that stay on in all 9 cells pass their own
    interaction on (inherited); upstream units that switch off in some cell make new interaction (switch)."""
    out = {}
    for up, dn, key in (('S', 'h4', 'd4'), ('h4', 'h5', 'd5'), ('h5', 'z', 'd6')):
        Wm = net_.KB[key][0]                                                                  # (in, out)
        Vup = cube(lambda ck: RR[ck][up], inr_)                                               # (3, 3, n, in)
        I_up = np.stack([inter(Vup, ci, ki) for ci, ki in CORNERS], 1)                       # (n, 4, in)
        on_up = (Vup > 0).all((0, 1))
        Vdn = cube(lambda ck: RR[ck][dn], inr_)
        I_dn = np.stack([inter(Vdn, ci, ki) for ci, ki in CORNERS], 1)
        on_dn = (Vdn > 0).all((0, 1)); rows = {}
        for u_s, d in F['units'][dn].items():
            if 'I' not in d: continue
            u = int(u_s); jets = on_dn[:, u]
            if jets.sum() < MIN_N: continue
            s = np.array([d['cells'][f'c{CS[ci]}k{KS[ki]}']['sign'] for ci, ki in CORNERS])
            pred = I_up[jets] @ Wm[:, u]; inh = (I_up[jets] * on_up[jets][:, None, :]) @ Wm[:, u]; sw = pred - inh
            meas = I_dn[jets][:, :, u]; sd = d['sd']
            rows[u_s] = dict(n=int(jets.sum()), I=float((meas * s).mean() / sd), I_from_upstream=float((pred * s).mean() / sd),
                             I_inherited=float((inh * s).mean() / sd), I_switch=float((sw * s).mean() / sd),
                             resid_median_abs=float(np.median(np.abs(meas - pred)) / sd), resid_max_abs=float(np.abs(meas - pred).max() / sd),
                             n_upstream_switching_mean=float((~on_up[jets]).sum(1).mean()))
        out[dn] = rows
        del Vup, Vdn, I_up, I_dn
    return out


def fmt(v, p='.3f'):
    return '  n/a' if v is None else format(v, p)


def show_units(F, tag):
    log(f'{tag} per unit (sd units): n | e_c e_k rho | med|U−lin| med|U−log| frac log closer | I ± se (pred) | beta | step c, k')
    for st in STG:
        rows = [(int(u), d) for u, d in F['units'][st].items() if 'I' in d]
        rows.sort(key=lambda r: -abs(r[1]['I_pred']))
        ml = [r for r in rows if (st, r[0]) in MASSLIKE]
        for u, d in rows[:14 if st != 'z' else 16] + [r for r in ml if r not in rows[:14 if st != 'z' else 16]]:
            log(f'   {st}{u:<3d} n {d["n"]:5d} | {d["e_c"]:+.2f} {d["e_k"]:+.2f} {fmt(d["rho"], "+.2f")} | {d["med_abs_lin"]:.3f} {d["med_abs_log"]:.3f} '
                f'{d["frac_log_closer"]:.2f} | {d["I"]:+.4f} ± {d["I_se"]:.4f} ({d["I_pred"]:+.4f}) | {fmt(d["beta"], "+.2f")} | '
                f'{fmt(d["step_c"], ".2f")} {fmt(d["step_k"], ".2f")}')
        skipped = [(int(u), d['n']) for u, d in F['units'][st].items() if 'I' not in d]
        if skipped: log(f'   {st}: {len(skipped)} live units with < {MIN_N} jets on in all cells (unit, n): {skipped}')
    for nm in MODELS:
        d = F['models'][nm]; li = d['log_interaction']
        log(f'   model {MODEL_LABEL[nm]:18s} n {d["n"]:5d} | med|U−lin| {d["med_abs_lin"]:.4f} med|U−log| {d["med_abs_log"]:.4f} frac log closer '
            f'{d["frac_log_closer"]:.2f} | I {d["I"]:+.4f} (pred {d["I_pred"]:+.4f}) beta {fmt(d["beta"], "+.2f")} | log interaction (proj.) '
            f'{li["I_proj"]:+.4f}, sum {li["I_sum_proj"]:+.4f}, product {li["I_prod_proj"]:+.4f}')
    for st in STG:
        r_ = F['readout'][st]
        for nm in ('logm', 'm_gev', 'exp_logm'):
            d = r_[nm]
            if 'I' in d:
                log(f'   read-out {nm:8s} at {st:2s} n {d["n"]:5d} | med|U−lin| {d["med_abs_lin"]:.3f} med|U−log| {d["med_abs_log"]:.3f} frac log closer '
                    f'{d["frac_log_closer"]:.2f} | I {d["I"]:+.4f} ± {d["I_se"]:.4f} (pred {d["I_pred"]:+.4f}) | beta {fmt(d["beta"], "+.2f")}')
        li = r_['logm_interaction']
        log(f'   log m̂ at {st}: interaction (log units) per corner [sum | product]: '
            + ' '.join(f'{k_} {v_:+.4f}±{li["I_log_se_per_cell"][k_]:.4f} [{li["I_sum_pred_per_cell"][k_]:+.4f}|{li["I_prod_pred_per_cell"][k_]:+.4f}]'
                       for k_, v_ in li['I_log_mean_per_cell'].items())
            + f' | projected {li["I_proj"]:+.4f} ± {li["I_proj_se"]:.4f} (sum {li["I_sum_proj"]:+.4f}, product {li["I_prod_proj"]:+.4f}); '
              f'beta_sum {fmt(li["beta_sum"], "+.2f")} ± {fmt(li["beta_sum_se"], ".2f")}')
        log(f'   log m̂ at {st}: Δlog m̂ (true Δlog m): ' + ' '.join(f'{k_} {v_:+.3f} ({li["true_dlogm"][k_]:+.3f})' for k_, v_ in li['transfer_dlogm'].items() if k_ != 'c1.0k1.0'))


# ================================================================ A, A2: factorial edits
DES = {}
for dname in ('A', 'A2'):
    info, inr_, mom_, RA, RAU = build(dname)
    res[dname] = info
    res[dname]['trained'] = factorial_block(RA, R0, readers, f'{dname} [trained]', inr_, mom_)
    res[dname]['untrained'] = factorial_block(RAU, U0, readers_u, f'{dname} [untrained]', inr_, mom_)
    res[dname]['trained']['propagation'] = propagation(RA, res[dname]['trained'], net, inr_)
    res[dname]['untrained']['propagation'] = propagation(RAU, res[dname]['untrained'], unt, inr_)
    for dn in ('h4', 'h5', 'z'):
        P_ = res[dname]['trained']['propagation'][dn]
        top = sorted(P_.items(), key=lambda kv: -abs(kv[1]['I']))[:8]
        top += [(str(u), P_[str(u)]) for st_, u in MASSLIKE if st_ == dn and str(u) in P_ and str(u) not in dict(top)]
        log(f'{dname} propagation into {dn} (sd units, projected on the product sign): I measured = W·I(upstream) [residual median|max] = inherited + switch')
        for u_s, r_ in top:
            log(f'   {dn}{u_s:<3s} n {r_["n"]:5d}: {r_["I"]:+.4f} = {r_["I_from_upstream"]:+.4f} [{r_["resid_median_abs"]:.4f}|{r_["resid_max_abs"]:.3f}] '
                f'= {r_["I_inherited"]:+.4f} + {r_["I_switch"]:+.4f}; upstream units switching per jet {r_["n_upstream_switching_mean"]:.1f}')
    show_units(res[dname]['trained'], f'{dname} [trained]'); show_units(res[dname]['untrained'], f'{dname} [untrained]')
    log(f'{dname}: calls per cell (in-range jets): ' + ' '.join(f'{k_}: acc {v_["acc"]:.3f} W→Z {v_["W_to_Z"]:.2f} Z→t {v_["Z_to_t"]:.2f};' for k_, v_ in info['calls'].items()))
    DES[dname] = dict(inr=inr_, LM={st: {ck: readers[(st, 'logm')](RA[ck][st]) for ck in RA} for st in ('h4', 'h5', 'z')},
                      MOD={nm: {ck: np.log(models(mm)[nm]) for ck, mm in mom_.items()} for nm in MODELS})
    del RA, RAU, mom_
    OUT_JSON.write_text(json.dumps(res))       # partial result after each design

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
hm, hf = D_match >= 0, D_free >= 0
res['B']['donors'] = dict(matched_frac=float(hm.mean()), unmatched_frac=float(hf.mean()),
                          matched_dlogS0_abs_median=float(np.median(np.abs(lS0[D_match[hm]] - lS0[hm]))),
                          unmatched_dlogS0_abs_median=float(np.median(np.abs(lS0[D_free[hf]] - lS0[hf]))),
                          matched_dlogS29_abs_median=float(np.median(np.abs(lS29[D_match[hm]] - lS29[hm]))),
                          unmatched_dlogS29_abs_median=float(np.median(np.abs(lS29[D_free[hf]] - lS29[hf]))),
                          matched_dlogM2_abs_median=float(np.median(np.abs(np.log(M['M2'][D_match[hm]] / M['M2'][hm])))),
                          unmatched_dlogM2_abs_median=float(np.median(np.abs(np.log(M['M2'][D_free[hf]] / M['M2'][hf])))),
                          matched_dlogM0_abs_median=float(np.median(np.abs(np.log(M['M0'][D_match[hm]] / M['M0'][hm])))))
log(f'B: donors: matched (|Δlog S29|,|Δlog S50| < 0.05, |Δlog S0| ≥ 0.25) for {hm.mean():.3f} of jets, median |Δlog S0| '
    f'{res["B"]["donors"]["matched_dlogS0_abs_median"]:.3f}, median |Δlog M2(true)| {res["B"]["donors"]["matched_dlogM2_abs_median"]:.3f}; '
    f'unmatched for {hf.mean():.3f}, median |Δlog S0| {res["B"]["donors"]["unmatched_dlogS0_abs_median"]:.3f}, '
    f'median |Δlog M2| {res["B"]["donors"]["unmatched_dlogM2_abs_median"]:.3f}')


def patched(net_, base, unit, D, values=None):
    """S[unit] := values (per jet) or the donor's S[unit]; jets with D < 0 keep their own value"""
    Sp = base['S'].copy(); has = D >= 0
    Sp[has, unit] = base['S'][D[has], unit] if values is None else values[has]
    m_ = np.zeros(64); m_[unit] = 1
    return net_.run_jetwise(x, {'S': (m_, Sp)}, keep=KEEP)


LAM_S = float(np.median(M['M2']) / np.median(S[:, 0]))          # sum model λ_S·S0 + M2: equal terms at the medians (S0 stands for M0)
share = (LAM_S * S[:, 0]) / np.maximum(LAM_S * S[:, 0] + M['M2'], 1e-9)
res['B']['sum_model'] = dict(lambda_S0=LAM_S, share_median=float(np.median(share)), share_q25=float(np.percentile(share, 25)),
                             share_q75=float(np.percentile(share, 75)), share_median_by_M2_tertile=[float(np.median(share[TERT == t])) for t in range(3)])
log(f'B: sum model λ_S = {LAM_S:.4g}; share of the S0 term median {np.median(share):.3f}, by M2 tertile '
    + ', '.join(f'{v:.3f}' for v in res['B']['sum_model']['share_median_by_M2_tertile']))


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
    dy = {}
    for st in ('h4', 'h5', 'z'):
        dy[st] = rdrs[(st, 'logm')](Rp[st][has]) - rdrs[(st, 'logm')](base[st][has])
        dy[st + '_gev'] = np.log(np.maximum(rdrs[(st, 'm')](Rp[st][has]), 5.0) / np.maximum(rdrs[(st, 'm')](base[st][has]), 5.0))
    sp = np.log(1 + share[has] * (np.exp(dl[has]) - 1))                  # sum model: Δlog(λ_S S0 + M2) for the applied ΔS0
    out = dict(n=n, tertiles={}, calls={})
    groups = {'M2': (M['M2'][has], T_EDGES), 'M2_over_S0': (M['M2'][has] / np.maximum(S[has, 0], 1e-3), None), 'S29': (S[has, 29], None)}
    w_all = WB[:, has].astype(np.float64)
    for gname, (gv, edges) in groups.items():
        qs = edges if edges is not None else np.percentile(gv, [100 / 3, 200 / 3]); tert = np.digitize(gv, qs)
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
    for gname in groups:      # spread of the slopes across the three tertiles (product: equal)
        tt = out['tertiles'][gname]
        if all(t in tt for t in ('low', 'mid', 'high')):
            out['tertiles'][gname]['high_over_low'] = {key: float(tt['high']['slope'][key] / tt['low']['slope'][key]) if abs(tt['low']['slope'][key]) > 1e-9 else None
                                                      for key in dy}
            out['tertiles'][gname]['high_over_low_sum_pred'] = float(tt['high']['sum_pred_slope'] / tt['low']['sum_pred_slope'])
    # calls
    c0 = base['logits'].argmax(1)[has]; c1 = Rp['logits'].argmax(1)[has]; yy_ = y[has]; dlh = dl[has]
    edges = [(-np.inf, -0.5), (-0.5, -0.25), (0.25, 0.5), (0.5, np.inf)]
    tert = TERT[has]
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
    log(f'B [{tag}]: n {n}; acc {out["acc_before"]:.3f} → {out["acc_after"]:.3f}, calls changed {out["calls_changed"]:.3f}; slope of Δlog m̂ on ½Δlog S0 by tertile '
        f'(product 1.0 in every tertile; sum prediction in brackets):')
    for gname in groups:
        for key in ('h4', 'h5', 'z', 'h5_gev', 'z_gev'):
            tt = out['tertiles'][gname]
            log(f'   by {gname:11s} {key:7s} ' + ' '.join(f'{tn} {tt[tn]["slope"][key]:+.3f}±{tt[tn]["slope_se"][key]:.3f} [{tt[tn]["sum_pred_slope"]:+.3f}]'
                                                         for tn in ('low', 'mid', 'high', 'all') if tn in tt)
                + (f' | high/low {fmt(tt["high_over_low"][key])} [sum {tt["high_over_low_sum_pred"]:.3f}, product 1]' if 'high_over_low' in tt else ''))
    cc = out['calls']
    for tname in ('low', 'mid', 'high'):
        log(f'   calls M2 {tname:4s}: ' + ' '.join(f'Δlog S0 {b_}: n {v_["n"]} changed {fmt(v_["calls_changed"], ".2f")} W→Z {fmt(v_["W_to_Z"], ".2f")} (n_W {v_["n_W"]}) '
                                                  f'Z→t {fmt(v_["Z_to_t"], ".2f")} Z→W {fmt(v_["Z_to_W"], ".2f")} (n_Z {v_["n_Z"]});' for b_, v_ in cc[tname].items()))
    return out


dl_match = np.where(hm, lS0[np.maximum(D_match, 0)] - lS0, 0.0)
dl_free = np.where(hf, lS0[np.maximum(D_free, 0)] - lS0, 0.0)
RP = patched(net, R0, 0, D_match); res['B']['S0_matched'] = patch_stats(RP, R0, readers, D_match, dl_match, 'S0 patch, matched donors')
res['B']['S0_matched']['plot_reader'] = 'h5'
RP_dLM = {st: readers[(st, 'logm')](RP[st]) - readers[(st, 'logm')](R0[st]) for st in ('h4', 'h5', 'z')}
del RP
RP19 = patched(net, R0, 19, D_match)
d19 = np.where(hm, np.log(np.maximum(S[np.maximum(D_match, 0), 19], 0.1)) - np.log(np.maximum(S[:, 19], 0.1)), 0.0)
res['B']['S19_control'] = patch_stats(RP19, R0, readers, D_match, dl_match, 'S19 patch (control), same donors; x = the donor\'s ½Δlog S0')
res['B']['S19_control']['dlogS19_abs_median'] = float(np.median(np.abs(d19[hm])))
h19 = hm & is_te
res['B']['S19_control']['slope_on_own_dlogS19'] = {st: float((d19[h19] @ (readers[(st, 'logm')](RP19[st][h19]) - readers[(st, 'logm')](R0[st][h19])))
                                                              / max((d19[h19] ** 2).sum(), 1e-30)) for st in ('h4', 'h5', 'z')}
log(f'   S19 control: median |Δlog S19| applied {res["B"]["S19_control"]["dlogS19_abs_median"]:.3f}; slope of Δlog m̂ on Δlog S19 itself: '
    + ', '.join(f'{k_} {v_:+.3f}' for k_, v_ in res['B']['S19_control']['slope_on_own_dlogS19'].items()))
del RP19
RPf = patched(net, R0, 0, D_free); res['B']['S0_unmatched'] = patch_stats(RPf, R0, readers, D_free, dl_free, 'S0 patch, donors with M2 NOT matched', bins=False)
del RPf

# supplementary: S0 × f (every jet; same statistics with dl = log f)
res['B']['S0_scale'] = {}
allj = np.arange(J)
for f_ in FSCAN:
    Rf = patched(net, R0, 0, allj, values=S[:, 0] * f_)
    res['B']['S0_scale'][str(f_)] = patch_stats(Rf, R0, readers, allj, np.full(J, np.log(f_)), f'S0 × {f_}', bins=False)
    del Rf

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
    del Rpu
OUT_JSON.write_text(json.dumps(res))

# ================================================================ T. one moment moved, by M2 tertile, normalised by the pT × c response
selT = DES['A']['inr'] & DES['A2']['inr'] & is_te
xB = 0.5 * dl_match
srcs = {f'logm_{st}': (DES['A']['LM'][st], DES['A2']['LM'][st], RP_dLM[st]) for st in ('h4', 'h5', 'z')}
dB_sum = np.log(1 + share * (np.exp(dl_match) - 1))
for nm in MODELS:
    srcs[f'model_{nm}'] = (DES['A']['MOD'][nm], DES['A2']['MOD'][nm], dB_sum if nm == 'sum' else 0.5 * dl_match)
res['T'] = dict(n=int(selT.sum()), n_patch=int((selT & hm).sum()), M2_tertile_edges=[float(v) for v in T_EDGES], sources={})
for name, (LA, LA2, dB) in srcs.items():
    g = (LA[(1.25, 1.0)] - LA[(0.8, 1.0)]) / LSTEP           # pT × c (A): both hypotheses 1
    rk = (LA[(1.0, 1.25)] - LA[(1.0, 0.8)]) / LSTEP          # angles × k at c = 1: M2 only
    r0 = (LA2[(1.25, 1.0)] - LA2[(0.8, 1.0)]) / LSTEP        # pT × c, angles × c^-1/2 (A2): M0 only
    rows = {}; boots = {}
    for ti, tn in enumerate(('low', 'mid', 'high', 'all')):
        sel_ = selT & ((TERT == ti) if tn != 'all' else np.ones(J, bool)); selb = sel_ & hm
        w = WB[:, sel_].astype(np.float64); ws = np.maximum(w.sum(1), 1.0); wb_ = WB[:, selb].astype(np.float64)
        bg = w @ g[sel_] / ws; bk = w @ rk[sel_] / ws; b0 = w @ r0[sel_] / ws
        xs_, ys_ = xB[selb], dB[selb]
        sl = float((xs_ @ ys_) / (xs_ @ xs_)); bs = (wb_ @ (xs_ * ys_)) / np.maximum(wb_ @ (xs_ * xs_), 1e-30)
        mg, mk, m0 = float(g[sel_].mean()), float(rk[sel_].mean()), float(r0[sel_].mean())
        rows[tn] = dict(n=int(sel_.sum()), n_patch=int(selb.sum()), M2_median=float(np.median(M['M2'][sel_])), share_median=float(np.median(share_M[sel_])),
                        gain_c=mg, gain_c_se=float(bg.std()), resp_k=mk, resp_k_se=float(bk.std()), resp_c0=m0, resp_c0_se=float(b0.std()),
                        k_over_c=mk / mg, k_over_c_se=float((bk / bg).std()), c0_over_c=m0 / mg, c0_over_c_se=float((b0 / bg).std()),
                        S0_slope=sl, S0_slope_se=float(bs.std()), S0_slope_over_c=sl / mg, S0_slope_over_c_se=float((bs / bg).std()))
        boots[tn] = dict(k_over_c=bk / bg, c0_over_c=b0 / bg, S0_slope_over_c=bs / bg)
    rows['high_over_low'] = {key: dict(value=rows['high'][key] / rows['low'][key], se=float((boots['high'][key] / boots['low'][key]).std()))
                             for key in ('k_over_c', 'c0_over_c', 'S0_slope_over_c')}
    res['T']['sources'][name] = rows
log(f'T: {res["T"]["n"]} test-half jets in range in both designs ({res["T"]["n_patch"]} with an S0 donor); per M2 tertile (low | mid | high; '
    f'high/low): response to pT × c, M2-only response ÷ it, M0-only response ÷ it, S0-patch slope ÷ it')
for name, rows in res['T']['sources'].items():
    hl = rows['high_over_low']
    log(f'   {name:12s} gain_c ' + ' '.join(f'{rows[t]["gain_c"]:.3f}' for t in ('low', 'mid', 'high'))
        + ' | k/c ' + ' '.join(f'{rows[t]["k_over_c"]:.3f}±{rows[t]["k_over_c_se"]:.3f}' for t in ('low', 'mid', 'high')) + f' ({hl["k_over_c"]["value"]:.3f}±{hl["k_over_c"]["se"]:.3f})'
        + ' | c0/c ' + ' '.join(f'{rows[t]["c0_over_c"]:.3f}±{rows[t]["c0_over_c_se"]:.3f}' for t in ('low', 'mid', 'high')) + f' ({hl["c0_over_c"]["value"]:.3f}±{hl["c0_over_c"]["se"]:.3f})'
        + ' | S0/c ' + ' '.join(f'{rows[t]["S0_slope_over_c"]:.3f}±{rows[t]["S0_slope_over_c_se"]:.3f}' for t in ('low', 'mid', 'high'))
        + f' ({hl["S0_slope_over_c"]["value"]:.3f}±{hl["S0_slope_over_c"]["se"]:.3f})')

res['wall_s'] = float(time.time() - t0)
OUT_JSON.write_text(json.dumps(res))
log(f'done -> {OUT_JSON}')
