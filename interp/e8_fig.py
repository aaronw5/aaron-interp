"""E8 figure and rule table. Reads results/jedi/e8_conjunctions.json; writes results/jedi/e8_conjunctions.png (a, b: share of
other-called jets switched into the class by the rule / random units / untrained network / all-unit ceiling at z and at S;
c: dose curves) and results/jedi/e8_rules_table.png (class, rule with mass ranges, precision, coverage, switched %).
Run with any Python that has matplotlib:  python interp/e8_fig.py [results/jedi/e8_conjunctions.json]"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/jedi/e8_conjunctions.json')
R = json.loads(src.read_text()); CL = R['classes']; OUT = src.parent
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948')
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})
CCOL = [C['green'], C['aqua'], C['blue'], C['violet'], C['orange']]          # g q W Z t


def untr(i):
    u = i.get('untrained', {}); return u.get('own_medians', u.get('trained_medians'))


fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor=BG)
for ax, stage, ttl in zip(axs[:2], ('z', 'S'), ('a  imposed on the 16 neurons z', 'b  imposed on the 19 live objects S')):
    I = R[stage]['intervention']; xs = np.arange(5); w = 0.2
    rule = [I[c]['rule'] for c in CL]; rnd = [I[c]['random']['mean'] for c in CL]
    rlo = [I[c]['random']['mean'] - I[c]['random']['min'] for c in CL]; rhi = [I[c]['random']['max'] - I[c]['random']['mean'] for c in CL]
    ceil_key = list(I[CL[0]]['ceiling'])[0]; ceil = [I[c]['ceiling'][ceil_key] for c in CL]
    un = [untr(I[c]) if untr(I[c]) is not None else 0 for c in CL]
    ax.bar(xs - 1.5 * w, rule, w, color=C['blue'], label=f'rule: 2-3 unit windows set to class medians')
    ax.bar(xs - 0.5 * w, rnd, w, color=C['yellow'], yerr=[rlo, rhi], capsize=2, ecolor=INK2, label='same number of random units (10 draws, min-max)')
    ax.bar(xs + 0.5 * w, un, w, color=C['magenta'], label='rule on the untrained network')
    ax.bar(xs + 1.5 * w, ceil, w, color=GRID, edgecolor=INK2, label=f'ceiling: all {"16 neurons" if stage == "z" else "19 units"} set to class medians')
    for xi, v in zip(xs, rule): ax.text(xi - 1.5 * w, v + 1, f'{v:.0f}', ha='center', va='bottom', fontsize=8, color=INK)
    for xi, v in zip(xs, ceil): ax.text(xi + 1.5 * w, v + 1, f'{v:.0f}', ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([f'{c}\n{R[stage]["description"][c]["n_units"]} units' for c in CL]); ax.set_ylim(0, 105)
    ax.set_ylabel('% of jets called otherwise now called the class'); ax.set_title(ttl, loc='left', fontsize=10, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_facecolor(BG); ax.legend(frameon=False, fontsize=7.5, loc='upper left', labelcolor=INK2)
ax = axs[2]; doses = [0.0] + [float(d) for d in R['doses']]
for k, c in enumerate(CL):
    for stage, ls in (('z', '-'), ('S', '--')):
        d = R[stage]['intervention'][c]['dose']
        ax.plot(doses, [0.0] + [d[str(f)] for f in R['doses']], ls, color=CCOL[k], marker='o', ms=4, lw=1.8, label=f'{c} at {stage}')
ax.set_xlabel('fraction of the way from the jet\'s own values to the class medians (rule units only)')
ax.set_ylabel('% now called the class'); ax.set_ylim(0, 105); ax.set_xticks(doses)
ax.set_title('c  dose: solid z, dashed S', loc='left', fontsize=10, color=INK); ax.grid(color=GRID, lw=0.8); ax.set_facecolor(BG)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper left', labelcolor=INK2)
fig.suptitle(f'JEDI-linear 64p: each class as a conjunction of unit windows (3-deep tree), imposed on the network\'s own z and S; '
             f'{R["n_eval"]:,} evaluation jets (rules fit on {R["n_fit"]:,}; seed {R["seed"]})', x=0.01, ha='left', fontsize=11, color=INK)
fig.tight_layout(); out = OUT / 'e8_conjunctions.png'; fig.savefig(out, dpi=150, facecolor=BG); print(out)


# ------------------------------------------------------------------ rule table (site/tables.py style)
def table(fname, title, cols, rows, widths, note=None, h=0.42):
    fig, ax = plt.subplots(figsize=(sum(widths), 0.9 + h * len(rows) + (0.5 if note else 0)), dpi=160); ax.axis('off')
    t = ax.table(cellText=rows, colLabels=cols, colWidths=[w / sum(widths) for w in widths], loc='upper center', cellLoc='left', colLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.75)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#d8d4ca'); cell.PAD = 0.03
        if r == 0: cell.set_facecolor('#eeebe3'); cell.set_text_props(weight='bold')
        elif r % 2 == 0: cell.set_facecolor('#f8f7f3')
    ax.set_title(title, loc='left', fontsize=11, weight='bold', pad=8)
    if note: fig.text(0.01, 0.01, note, fontsize=8, color='#555')
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); print(fname)


KNOWN = {'z0': 'W window, peak 75 GeV', 'z7': 'Z window 94-98 GeV', 'z14': 'Z window 94-98 GeV', 'z10': 'top window 135-148 GeV'}


def rule_text(stage, c):
    d = R[stage]['description'][c]; W = R[stage]['windows']; units = R['liveS'] if stage == 'S' else list(range(16)); parts = []
    for f, (lo, hi) in sorted(d['rule'].items(), key=lambda kv: int(kv[0])):
        nm = f'{"S" if stage == "S" else "z"}{units[int(f)]}'
        cond = (f'{lo:.3g} ≤ {nm} < {hi:.3g}' if lo is not None and hi is not None else f'{nm} ≥ {lo:.3g}' if lo is not None else f'{nm} < {hi:.3g}')
        w = W.get(f'{c}:{nm}'); mq = w['mass_q'] if w else None
        parts.append(cond + (f'  [m {mq[0]:.0f}–{mq[2]:.0f}' + (f'; {KNOWN[nm]}' if nm in KNOWN else '') + ']' if mq else ''))
    return '\n'.join(parts)


rows = []
for stage in ('z', 'S'):
    for c in CL:
        d, i = R[stage]['description'][c], R[stage]['intervention'][c]
        ceil = i['ceiling'][list(i['ceiling'])[0]]; u = untr(i)
        rows.append([c, stage, rule_text(stage, c), f'{d["precision"]*100:.0f} %', f'{d["coverage"]*100:.0f} %',
                     f'{i["rule"]:.0f} %  /  {i["random"]["mean"]:.0f} %  /  {u if u is None else f"{u:.0f} %"}  /  {ceil:.0f} %',
                     f'{i["necessity"]["rule_to_other_med"]:.0f} %'])
table(OUT / 'e8_rules_table.png', f'JEDI-linear: each class as a conjunction of unit windows (E8; {R["n_eval"]:,} evaluation jets, rules fit on {R["n_fit"]:,}; seed {R["seed"]})',
      ['class', 'level', 'rule (3-deep tree, best leaf); [mass 25–75 % of jets in the window, GeV; E6 window]', 'precision', 'coverage',
       'other jets switched in: rule / random / untrained / ceiling', 'still the class after\nrule → other medians'],
      rows, [0.6, 0.6, 5.6, 0.9, 0.9, 3.4, 1.6], h=0.95,
      note='precision = share of jets inside the rule the network calls the class; coverage = share of the network\'s calls inside the rule. '
           'Switched: evaluation jets called otherwise, rule units set to the in-rule medians of jets called the class; random = same number of units (mean of 10); '
           'ceiling = all 16 neurons (z) / 19 live units (S) set to class medians.')
