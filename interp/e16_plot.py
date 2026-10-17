"""E16 figures: one card per live unit (card_<stage><unit>.png) and the summary table (atlas_summary.png), from
$AI_RESULTS/jedi/e16/atlas_<stage>.json and images_<stage>.npz (written by interp.e16_atlas run).  numpy + matplotlib only."""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi' / os.environ.get('E16_DIR', 'e16')
CL = ['g', 'q', 'W', 'Z', 't']
CCOL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']      # class identity, fixed order (validated, light surface)
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seq', ['#f3f2fb', '#b7b0ec', '#6250d6', '#2a1f7a', '#120c3d'])
STAGE_NAME = dict(a='first-layer particle cut', P='first-round object P (pooled cut)', g='jet context g', S='second-round object S (pooled u)',
                  h4='dense unit h4', h5='dense unit h5', z='neuron z (pre-ReLU, read by the last layer)')
OBS10 = ['mass', 'sum_pt', 'n_particles', 'girth', 'tau21', 'tau32', 'D2', 'z_0', 'dr01', 'sd_mass']
OBS_LAB = dict(mass='jet mass [GeV]', sum_pt='ΣpT [GeV]', n_particles='multiplicity', girth='girth Σz·ΔR [rad]', tau21='τ21', tau32='τ32',
               D2='D2', z_0='leading pT / ΣpT', dr01='ΔR of 2 hardest [rad]', sd_mass='soft-drop mass [GeV]')
ROWS = ['pT per bin [GeV]', 'particles per bin', 'pT per bin [GeV]']
FRAME = ['network frame', 'network frame', 'aligned frame']
plt.rcParams.update({'font.size': 10.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF})


def fmt(v):
    v = float(v)
    if v == 0: return '0'
    a = abs(v)
    if a >= 100: return f'{v:.0f}'
    if a >= 10: return f'{v:.1f}'
    if a >= 1: return f'{v:.2f}'
    if a >= 0.01: return f'{v:.3f}'
    return f'{v:.2g}'


def pct(v):
    return f'{100 * v:.1f} %'


def labels_all():
    """short patterns of every unit already in the atlas, to label upstream / downstream lists"""
    L = {}
    for st in ('a', 'P', 'g', 'S', 'h4', 'h5', 'z'):
        p = OUT / f'atlas_{st}.json'
        if not p.exists(): continue
        for u in json.loads(p.read_text())['units']:
            L[u['id']] = u['pattern']
    return L


def short(s, n=58):
    return s if len(s) <= n else s[:n - 1] + '…'


def mixbar(ax, mix, n=None):
    left = 0
    for c in range(5):
        w = mix[c]
        if w <= 0: continue
        ax.barh(0, w, left=left, color=CCOL[c], height=0.8, edgecolor=SURF, linewidth=1.5)
        if w >= 0.12: ax.text(left + w / 2, 0, f'{CL[c]} {100 * w:.0f}', ha='center', va='center', fontsize=9, color=INK)
        left += w
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.axis('off')


def halfplane(ax, alpha, beta):
    ap, ae, af = alpha
    if abs(ae) < 1e-9 and abs(af) < 1e-9: return
    nrm = np.hypot(ae, af); n = np.array([ae, af]) / nrm; d = -beta / nrm
    p0 = d * n; t = np.array([-n[1], n[0]])
    pts = np.array([p0 - 0.6 * t, p0 + 0.6 * t])
    ax.plot(pts[:, 0], pts[:, 1], color='#e34948', lw=1.4, ls='--')
    ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4)
    ax.annotate('', xy=p0 + 0.15 * n, xytext=p0, arrowprops=dict(arrowstyle='->', color='#e34948', lw=1.4))


def images_block(fig, u, imgs, x0, y_top, w, h, gap=0.008, vgap=0.03, cut=None):
    names = u['bin_names']; nb = len(names)
    H = imgs
    for r in range(3):
        stack = H[:, r]
        vmax = float(stack.max()) if stack.max() > 0 else 1.0
        norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)
        for j in range(nb):
            ax = fig.add_axes([x0 + j * (w + gap), y_top - (r + 1) * h - r * vgap, w, h])
            im = ax.imshow(np.maximum(stack[j].T, vmax * 1e-3), origin='lower', extent=[-0.4, 0.4, -0.4, 0.4], cmap=SEQ, norm=norm,
                           interpolation='nearest')
            ax.set_xticks([-0.3, 0, 0.3]); ax.set_yticks([-0.3, 0, 0.3]); ax.tick_params(labelsize=8.5, length=2)
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(GRID)
            if j == 0:
                ax.set_ylabel("Δφ' [rad]" if r == 2 else 'Δφ [rad]', fontsize=9.5)
            else:
                ax.set_yticklabels([])
            if r >= 1:
                ax.set_xlabel("Δη' [rad]" if r == 2 else 'Δη [rad]', fontsize=9.5)
            else:
                ax.set_xticklabels([])
            if r == 0:
                b = u['bins'][names[j]]
                ax.set_title(f'{names[j]}  ({b["n"]} jets)\nvalue {fmt(b["act_range"][0])} to {fmt(b["act_range"][1])}', fontsize=10, color=INK)
            if cut is not None and r < 2: halfplane(ax, *cut)
        cax = fig.add_axes([x0 + nb * (w + gap) + 0.002, y_top - (r + 1) * h - r * vgap + 0.01, 0.007, h - 0.02])
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8); cb.outline.set_visible(False)
        cb.set_label(ROWS[r] + ' per jet', fontsize=9, color=MUTED)
        fig.text(0.006, y_top - (r + 0.5) * h - r * vgap, FRAME[r], rotation=90, va='center', ha='left', fontsize=10, color=INK)
    # class mix under each column
    yb = y_top - 3 * h - 2 * vgap - 0.062
    for j in range(nb):
        ax = fig.add_axes([x0 + j * (w + gap), yb, w, 0.025])
        mixbar(ax, u['bins'][names[j]]['class_mix'])
    fig.text(x0, yb - 0.012, 'class mix of the jets in each bin (%; g q W Z t)', fontsize=9, color=MUTED, va='top')
    return yb


def hist_panel(ax, u):
    ed = np.array(u['hist']['edges']); c = (ed[1:] + ed[:-1]) / 2
    for k in range(5):
        cnt = np.array(u['hist']['counts'][k], float)
        ax.step(c, np.maximum(cnt, 0.8), where='mid', color=CCOL[k], lw=1.6, label=CL[k])
    ax.set_yscale('log'); ax.set_ylim(0.8, None)
    ax.set_xlabel(f'{u["id"]} value' + (' (pre-ReLU)' if u['stage'] == 'z' else ''), fontsize=9.5); ax.set_ylabel('jets', fontsize=9.5)
    ax.legend(ncol=5, fontsize=8.5, frameon=False, loc='upper right', handlelength=1.2, columnspacing=0.8)
    for bn, b in u['bins'].items():
        ax.axvline(b['act_range'][0], color=MUTED, lw=0.7, ls=':')
    ttl = f'activation per class; on in {pct(u["on_frac"])} of jets'
    if u['stage'] == 'z': ttl += f', negative in {pct(u["neg_frac"])}'
    ax.set_title(ttl, fontsize=10, loc='left', color=INK)
    ax.grid(axis='y', color=GRID, lw=0.6)


def curve_panel(ax, u):
    b = u['desc']['best1']; q = np.array(b['curve']['q']); f = np.array(b['curve']['f']); sd = np.array(b['curve']['sd'])
    ax.fill_between(q, f - sd, f + sd, color='#b7b0ec', alpha=0.5, lw=0)
    ax.plot(q, f, color='#2a1f7a', lw=2, marker='o', ms=3.5)
    try:
        from interp.e16_atlas import unit_of
        un = unit_of(b['name']).strip()
    except Exception:
        un = ''
    ax.set_xlabel(short(b['label'], 48) + (f' [{un}]' if un else ''), fontsize=9.5); ax.set_ylabel(f'mean {u["id"]} (± SD)', fontsize=9.5)
    ax.set_title(f'response to the best single observable (held-out R² {b["R2"]:.2f})', fontsize=10, loc='left', color=INK)
    ax.grid(color=GRID, lw=0.6)
    if u['stage'] == 'z': ax.axhline(0, color=MUTED, lw=0.8)


def obs_table(ax, u):
    names = list(u['bins']); cols = names + ['all on-jets', 'all jets']
    rows = []
    for o in OBS10:
        rows.append([fmt(u['bins'][n]['med'][o]) for n in names] + [fmt(u['med_on'][o]), fmt(u['med_all'][o])])
    ax.axis('off')
    t = ax.table(cellText=rows, rowLabels=[OBS_LAB[o] for o in OBS10], colLabels=cols, loc='upper left', cellLoc='right', colLoc='right',
                 bbox=[0.30, 0.0, 0.70, 1.0])
    t.auto_set_font_size(False); t.set_fontsize(9.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.5)
        if r == 0: cell.set_text_props(color=MUTED, fontweight='bold')
        if c == -1: cell.set_text_props(color=INK, ha='right'); cell._loc = 'right'
    ax.set_title('median observables per activation bin', fontsize=10, loc='left', color=INK)


def flow_panel(ax, s, title):
    M = np.array(s['flow'], float); n = M.sum()
    F = M / max(n, 1)
    ax.imshow(F, cmap=SEQ, vmin=0, vmax=max(F.max(), 1e-9))
    for i in range(5):
        for j in range(5):
            if M[i, j] > 0:
                ax.text(j, i, f'{100 * F[i, j]:.0f}' if F[i, j] >= 0.005 else '·', ha='center', va='center', fontsize=8.5,
                        color='white' if F[i, j] > 0.5 * F.max() else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(CL); ax.set_yticks(range(5)); ax.set_yticklabels(CL)
    ax.set_xlabel('call after', fontsize=9); ax.set_ylabel('call before', fontsize=9); ax.tick_params(labelsize=9, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=8.6, loc='left', color=INK)


def hinge_panel(ax, u):
    up = u['upstream']; T = np.array(up['table']); n = np.array(up['table_n'])
    T = np.where(n > 0, T, np.nan)
    im = ax.imshow(T, origin='lower', aspect='auto', cmap=SEQ, vmin=0, vmax=np.nanmax(T) if np.isfinite(T).any() else 1)
    pe, re_ = up['pT_edges'], up['r_edges']
    ax.set_yticks(np.arange(len(pe) - 1)); ax.set_yticklabels([f'{fmt(pe[i])}–{fmt(pe[i + 1]) if pe[i + 1] < 1e4 else "∞"}' for i in range(len(pe) - 1)], fontsize=8)
    ax.set_xticks(np.arange(len(re_) - 1)); ax.set_xticklabels([f'{re_[i]:g}' for i in range(len(re_) - 1)], fontsize=8)
    ax.set_xlabel('r = ΔR to the axis [rad], bin lower edge', fontsize=8.5); ax.set_ylabel('particle pT [GeV]', fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5)
    ax.set_title(f'per-particle u{u["unit"]}: mean in (pT, r) cells', fontsize=9.5, loc='left', color=INK)


def dlogit_panel(ax, u):
    d = u['downstream']; m = np.array(d['dlogit_mean']); sd = np.array(d['dlogit_sd'])
    ax.bar(range(5), m, color=CCOL, width=0.65, edgecolor=SURF)
    ax.errorbar(range(5), m, yerr=sd, fmt='none', ecolor=MUTED, lw=0.8, capsize=2)
    for i in range(5): ax.text(i, m[i] + (0.02 if m[i] >= 0 else -0.02) * max(np.abs(m).max(), 1e-9) * 3, f'{m[i]:+.2f}', ha='center',
                               va='bottom' if m[i] >= 0 else 'top', fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels(CL)
    ax.set_ylabel('Δ logit (mean ± SD)', fontsize=9)
    ax.set_title(f'unit at p90 ({fmt(d["p90"])}) minus p10 ({fmt(d["p10"])}), 2,000 jets\ncalls differ in {pct(d["calls_differ"])}', fontsize=9.5, loc='left', color=INK)
    ax.grid(axis='y', color=GRID, lw=0.6)


def text_block(fig, x, y, lines, size=9.5, width=None, title=None):
    if title:
        fig.text(x, y, title, fontsize=10, color=INK, fontweight='bold', va='top'); y -= 0.022
    for ln in lines:
        fig.text(x, y, ln, fontsize=size, color=INK if not ln.startswith('  ') else MUTED, va='top', family='DejaVu Sans')
        y -= 0.0185
    return y


def card(u, imgs, labs, path, cut=None):
    fig = plt.figure(figsize=(21, 13.5), dpi=80)
    st = u['stage']
    # ---- header
    fig.text(0.012, 0.988, f'{u["id"]}  ·  {STAGE_NAME[st]}', fontsize=17, fontweight='bold', color=INK, va='top')
    fig.text(0.012, 0.957, f'mean ablation changes {pct(u["cost"])} of calls (accuracy {pct(u["acc_ablated"])});  on in {pct(u["on_frac"])} of jets'
             + (f';  negative in {pct(u["neg_frac"])}' if st == 'z' else ''), fontsize=11.5, color=MUTED, va='top')
    d = u['desc']
    fig.text(0.012, 0.934, f'pattern: {u["pattern"]}', fontsize=13, color=INK, va='top')
    fig.text(0.012, 0.911, f'held-out R² of the description: {d["R2_1"]:.2f} with 1 observable, {d["R2_2"]:.2f} with 2 ('
             f'{" + ".join(short(l, 40) for l in d.get("best2", {}).get("labels", [d["best1"]["label"]]))})', fontsize=10.5, color=MUTED, va='top')
    cl = u['clusters']
    ctext = f'category: {u["category"]}  —  largest k-means cluster of the on-jets holds {pct(cl.get("main_share", 1))} of the activation mass (k = {cl.get("k")}'
    if cl.get('silhouette'): ctext += '; silhouette ' + ', '.join(f'k={k}: {v:.2f}' for k, v in cl['silhouette'].items())
    ctext += ')'
    fig.text(0.012, 0.889, ctext, fontsize=10.5, color=MUTED, va='top')
    # ---- images (left), histogram + response curve + table (right)
    nb = len(u['bin_names'])
    w = 0.084 if nb <= 3 else 0.078
    h = w * 21 / 13.5
    yb = images_block(fig, u, imgs, 0.052, 0.835, w, h, cut=cut)
    xr = 0.052 + nb * (w + 0.008) + 0.075
    av = 0.985 - xr
    ax = fig.add_axes([xr, 0.665, 0.5 * av - 0.04, 0.165]); hist_panel(ax, u)
    ax = fig.add_axes([xr + 0.5 * av + 0.03, 0.665, 0.5 * av - 0.035, 0.165]); curve_panel(ax, u)
    if u['upstream'].get('kind') == 'hinge':
        ax = fig.add_axes([xr, 0.375, 0.66 * av, 0.235]); obs_table(ax, u)
        ax = fig.add_axes([xr + 0.74 * av, 0.40, 0.24 * av, 0.19]); hinge_panel(ax, u)
    else:
        ax = fig.add_axes([xr, 0.375, av, 0.235]); obs_table(ax, u)
    # ---- bottom band: weak range / negative pole / clusters | upstream | downstream | steering
    ytop = 0.288
    lines = []
    if u.get('weak_vs_top'):
        lines.append('weak range (50–90 %) vs top 1 %, medians: ' + '; '.join(f'{OBS_LAB[v["obs"]].split(" [")[0]} {fmt(v["med_ref"])} → {fmt(v["med"])}'
                                                                   for v in u['weak_vs_top']))
        lines.append(f'  class-mix overlap of the two bins: {pct(u.get("weak_class_overlap", 0))}')
    if st == 'z' and 'lowest 1 %' in u['bins']:
        cm = u['bins']['lowest 1 %']['class_mix']; tp = int(np.argmax(cm))
        lines.append(f'negative pole (lowest 1 %): {CL[tp]} {100 * cm[tp]:.0f} %; ' + '; '.join(
            f'{OBS_LAB[v["obs"]].split(" [")[0]} {fmt(v["med"])} (all {fmt(v["med_ref"])})' for v in u['neg_vs_all']))
        if u.get('neg_desc'): lines.append(f'  negative side alone: {short(u["neg_desc"]["pattern"], 80)} (R² {u["neg_desc"]["R2_2"]:.2f})')
    lines.append('k-means clusters of the on-jets: share of activation mass; classes ≥ 5 %;')
    lines.append('  medians; the cluster\'s own best observable (R², |ρ| with the unit\'s best)')
    for j, c in enumerate(cl.get('clusters', [])[:4]):
        mix = ' '.join(f'{CL[i]}{100 * v:.0f}' for i, v in enumerate(c['class_mix']) if v >= 0.05)
        lines.append(f'{j + 1}: {pct(c["share_mass"])} ({c["n"]} jets); {mix}; m {fmt(c["med"]["mass"])} GeV, ΣpT {fmt(c["med"]["sum_pt"])} GeV, n {fmt(c["med"]["n_particles"])}')
        ob = c.get('own_best')
        if ob: lines.append(f'  tracks {short(ob["label"], 40)} (R² {ob["R2"]:.2f}, |ρ| {c.get("rho_with_unit_best", 1):.2f})')
    text_block(fig, 0.012, ytop, lines, size=9.5, title='weak activations, negative pole, clusters')
    up = u['upstream']; ul = []
    if up['kind'] == 'cut':
        ul.append(f'l1 cut a{u["unit"]}:'); ul.append(f'  {up["cut"]}'); ul.append(f'pooled over {up["slots"]} slots: P = Σ a / 64')
    else:
        if up['kind'] == 'hinge':
            ul.append(f'per-particle u{u["unit"]}, pooled over {up["n_slots"]} slots:'); ul.append(f'  {short(up["window"], 50)}; S = Σ u / 16')
            if up.get('e5'):
                hh = up['e5']['hinge']; v = 'ln pT' if hh['form'] == 'ln' else 'pT'
                ul.append(f'  E5 hinge: ReLU({hh["a"]:.2f} {v} {hh["b"]:+.1f} r {hh["c"]:+.1f})')
            ul.append(f'  context share of the input spread²: {pct(up["ctx_share"])}')
        ul.append('inputs by |weight × spread| (quantized input):')
        for e in up['units'][:5]:
            ul.append(f'  {e["unit"]:>5s} w {e["w"]:+.2f}, {e["score"]:.2f}: {short(labs.get(e["unit"], ""), 34)}')
    xu = 0.43
    text_block(fig, xu, ytop, ul, size=9.3, title='upstream')
    dl = []
    dn = u['downstream']
    if 'weights' in dn:
        dl.append('last-layer weight on ReLU(z) (× its spread):')
        W_ = dn['weights']
        dl.append('  ' + ',  '.join(f'{e["cls"]} {e["w"]:+.2f} ({e["w_x_spread"]:+.2f})' for e in W_[:3]))
        dl.append('  ' + ',  '.join(f'{e["cls"]} {e["w"]:+.2f} ({e["w_x_spread"]:+.2f})' for e in W_[3:]))
    elif dn.get('units'):
        dl.append('read by (|weight × spread|):')
        for e in dn.get('units', [])[:4]:
            dl.append(f'  {e["unit"]:>5s} w {e["w"]:+.2f}, {e["score"]:.2f}: {short(labs.get(e["unit"], ""), 26)}')
    xd = 0.645
    text_block(fig, xd, ytop, dl, size=9.3, title='downstream')
    ax = fig.add_axes([xd + 0.015, 0.035, 0.15, 0.095]); dlogit_panel(ax, u)
    s = u['steer']; c = u.get('control', {})
    xs = 0.842
    if st == 'z' and 'steer_neg' in u:
        ax = fig.add_axes([xs, 0.165, 0.11, 0.075])
        flow_panel(ax, s, f'steer to p99 = {fmt(s["value"])} ({s["n"]} off jets)\nchanged {pct(s["changed"])} (control {c.get("unit", "-")} {pct(c.get("changed", 0))})')
        sn = u['steer_neg']; cn = u.get('control_neg', {})
        ax = fig.add_axes([xs, 0.03, 0.11, 0.075])
        flow_panel(ax, sn, f'steer to p1 = {fmt(sn["value"])} ({sn["n"]} on-jets)\nchanged {pct(sn["changed"])} (control {cn.get("unit", "-")} {pct(cn.get("changed", 0))})')
    else:
        ax = fig.add_axes([xs, 0.035, 0.12, 0.195])
        flow_panel(ax, s, f'steer to p99 = {fmt(s["value"])} ({s["n"]} {"off jets" if s["rule"] == "off jets" else "jets < p10"})\n'
                          f'calls changed {pct(s["changed"])} (control {c.get("unit", "-")} {pct(c.get("changed", 0))})')
    fig.savefig(path, dpi=80)
    plt.close(fig)


def card_a(u, imgs, labs, path):
    fig = plt.figure(figsize=(18, 7.5), dpi=80)
    fig.text(0.012, 0.975, f'{u["id"]}  ·  {STAGE_NAME["a"]} (reference)', fontsize=16, fontweight='bold', color=INK, va='top')
    fig.text(0.012, 0.925, f'cut: {u["cut"]};  on for {pct(u["on_frac"])} of real particles;  mean ablation changes {pct(u["cost"])} of calls',
             fontsize=12, color=MUTED, va='top')
    for r, (H, lab) in enumerate(zip(imgs, ['passing particles: pT per bin per jet [GeV]', 'passing particles per bin per jet'])):
        ax = fig.add_axes([0.035 + r * 0.29, 0.09, 0.62 * 7.5 / 18, 0.70])
        vmax = max(float(H.max()), 1e-9)
        im = ax.imshow(np.maximum(H.T, vmax * 1e-3), origin='lower', extent=[-0.4, 0.4, -0.4, 0.4], cmap=SEQ, norm=LogNorm(vmax * 1e-3, vmax))
        halfplane(ax, u['alpha'], u['beta'])
        ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4)
        ax.set_xlabel('Δη [rad]'); ax.set_ylabel('Δφ [rad]'); ax.set_title(lab, fontsize=10.5, loc='left')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
    ax = fig.add_axes([0.62, 0.50, 0.17, 0.33])
    e = np.array(u['pt_edges']); c = np.sqrt(e[1:] * e[:-1])
    on = np.array(u['pt_hist_on']); al = np.array(u['pt_hist_all'])
    ax.step(c, np.maximum(al, 0.8), where='mid', color=MUTED, lw=1.5, label='all real particles')
    ax.step(c, np.maximum(on, 0.8), where='mid', color='#6250d6', lw=1.8, label='passing the cut')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('particle pT [GeV]'); ax.set_ylabel('particles')
    ax.legend(frameon=False, fontsize=9); ax.grid(color=GRID, lw=0.6); ax.set_title('pT of passing particles', fontsize=10.5, loc='left')
    s = u['steer']; c_ = u.get('control', {})
    lines = [f'pooled into P{u["unit"]} over {u["slots_pooled"]} slots ({u["window"]}); {u["downstream"]["pooled"]}',
             'read by the second-layer statements s (|weight × spread|):']
    for d in u['downstream']['units'][:5]:
        lines.append(f'  {d["unit"]:>4s}  w {d["w"]:+.3f}  score {d["score"]:.3f}')
    lines.append(f'steering: cut value set to its p99 ({fmt(s["value"])}) on the real particles')
    lines.append(f'  where it is off ({pct(s["particles_set"])} of particles), {s["n"]} jets: {pct(s["changed"])} of calls change')
    lines.append(f'  control {c_.get("unit", "-")}, same procedure: {pct(c_.get("changed", 0))}')
    text_block(fig, 0.62, 0.36, lines, size=10.5)
    ax = fig.add_axes([0.84, 0.50, 0.13, 0.33]); flow_panel(ax, s, 'steered calls (% of jets)')
    fig.savefig(path, dpi=80); plt.close(fig)


def cards(st, only=None):
    p = OUT / f'atlas_{st}.json'
    d = json.loads(p.read_text())
    IM = np.load(OUT / f'images_{st}.npz')
    labs = labels_all()
    n = 0
    for u in d['units']:
        if only and u['id'] not in only: continue
        path = OUT / f'card_{u["id"]}.png'
        if st == 'a':
            card_a(u, IM[u['id']], labs, path)
        else:
            cut = None
            if st == 'P' and u['upstream'].get('kind') == 'cut': cut = (u['upstream']['alpha'], u['upstream']['beta'])
            card(u, IM[u['id']], labs, path, cut=cut)
        n += 1
    print(f'{n} cards for stage {st} -> {OUT}', flush=True)


# ------------------------------------------------------------------ summary table
def summary_png(res, path):
    rows = res['rows']
    cols = [('stage', 0.035), ('unit', 0.04), ('calls\nchanged', 0.045), ('pattern (best ≤ 2 observables)', 0.25), ('R²\n1 / 2', 0.05),
            ('main\nshare', 0.045), ('category', 0.07), ('offshoot clusters (share: shift vs main)', 0.19), ('negative pole (z)', 0.12),
            ('steering (p99 on off jets)', 0.155)]
    nrow = len(rows) + 1
    H = 0.19 * nrow + 2.8
    fig = plt.figure(figsize=(30, H), dpi=72)
    top = 1 - 1.75 / H; rh = 0.19 / H
    T = res['total']; Tf = res['total_forced_k2']; n = sum(T.values())
    fig.text(0.005, 1 - 0.25 / H, 'E16 neuron atlas, JEDI-linear 3-feature n64: one row per live unit (20,000 test jets)', fontsize=18,
             fontweight='bold', color=INK, va='top')
    tl = '; '.join(f'{st}: {t["one main job"]} one / {t["related jobs"]} related / {t["unrelated jobs"]} unrelated / {t["unexplained"]} unexplained'
                   for st, t in res['tally'].items())
    fig.text(0.005, 1 - 0.62 / H, f'all {n} units: one main job {T["one main job"]}, related jobs {T["related jobs"]}, unrelated jobs {T["unrelated jobs"]}, '
             f'unexplained {T["unexplained"]}  (k ≥ 2 forced: {Tf["one main job"]} / {Tf["related jobs"]} / {Tf["unrelated jobs"]} / {Tf["unexplained"]}).',
             fontsize=12.5, color=INK, va='top', fontweight='bold')
    fig.text(0.005, 1 - 0.88 / H, 'per stage (one / related / unrelated / unexplained): ' + tl.replace(' one / ', '/').replace(' related / ', '/').replace(' unrelated / ', '/').replace(' unexplained', ''),
             fontsize=11.5, color=INK, va='top')
    th = res['thresholds']
    fig.text(0.005, 1 - 1.15 / H, f'thresholds: unexplained = {th["unexplained"]}; one main job = {th["one_main_job"]}; related = {th["related"]}.',
             fontsize=10.5, color=MUTED, va='top')
    x = 0.005
    xs = []
    for name, w in cols:
        xs.append(x); x += w
    for (name, w), x0 in zip(cols, xs):
        fig.text(x0, top, name, fontsize=10.5, fontweight='bold', color=MUTED, va='bottom')
    prev = None
    for i, r in enumerate(rows):
        y = top - (i + 1) * rh
        if r['stage'] != prev:
            fig.add_artist(plt.Line2D([0.005, 0.995], [y + rh * 0.95, y + rh * 0.95], color=MUTED, lw=0.8, transform=fig.transFigure))
            prev = r['stage']
        elif i % 2 == 0:
            fig.add_artist(plt.Rectangle((0.005, y - rh * 0.25), 0.99, rh, color='#f1f0ec', transform=fig.transFigure, zorder=-1))
        vals = [r['stage'], r['id'], pct(r['cost']), short(r['pattern'], 95),
                '' if r.get('R2') is None else f'{r["R2"][0]:.2f} / {r["R2"][1]:.2f}',
                '' if r.get('main_share') is None else f'{100 * r["main_share"]:.0f} %', r['category'],
                short(r.get('offshoot', ''), 70), short(r.get('negative', ''), 46), short(r.get('steering', ''), 60)]
        for v, x0 in zip(vals, xs):
            fig.text(x0, y, v, fontsize=9.5, color=INK, va='bottom')
    fig.savefig(path, dpi=72); plt.close(fig)
