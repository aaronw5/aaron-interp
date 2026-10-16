"""results/sweep/*.json (interp/e1_e3_sweep.py) -> results/sweep/summary.md, one row per model.

  python -m interp.sweep_summary [results/sweep]
"""
import json, sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/sweep')
FAM = ['3-feature', '3-feature-perminv', '16-feature', '16-feature-perminv']
ST = ['a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z']
R = [json.loads(p.read_text()) for p in D.glob('*.json') if p.name != 'official_acc.json']
R.sort(key=lambda r: (r['untrained'], FAM.index(r['family']), r['n']))
L = ['# E1–E3 on official JEDI-linear models (A7 / E9 prep)', '',
     'Procedure (interp/e1_e3_sweep.py):',
     '- Test sample: the 260,000-jet validation archive, which is the official test set.',
     '- Each model sees the first n slots of each jet. The 3-feature models get (pT, Δη, Δφ); the 16-feature models get all '
     '16 jetConstituentList columns (interp/prep_16f.py).',
     '- acc: accuracy on all 260,000 jets. official: the JEDI-linear summary value.',
     '- E1–E3 run on a class-balanced random subsample: 4,000 jets per class (20,000 total), rng seed 0, the same jets for every model.',
     '- live l1: first-layer units that are > 0 on at least one real particle (out of 64).',
     '- >1% per stage: edit = one unit set to its mean over real data; metric = fraction of the 20,000 calls (argmax) that '
     'change; the entry counts the units whose edit changes more than 1% of calls. Stage widths: a, P, g, s, u, S, h4 = 64; h5 = 32; z = 16.',
     '- identity, all slots: max |pool·(b·|A| + w·Σ_A x) − P| over jets and channels. A = slots passing the cut; P is the '
     'network\'s own pooled first layer, fixed point included. rel = that error divided by max |P|.',
     '- identity, A ∩ K: the same with A restricted to K, the slots the pooling layer\'s HGQ input quantizer keeps '
     '(> 0 bits per slot and channel; the 0-bit entries are fixed zeros).',
     '- zero-bit: the fraction of input-quantizer entries with 0 bits, for layers qs, l3 and qs1.',
     '- calls: the fraction of the 20,000 sample calls per class (g, q, W, Z, t).',
     '- untrained: control. Every kernel, bias and BN scale is redrawn from a Gaussian with the trained mean and spread '
     '(seed 0); the number formats are kept.', '',
     '| model | acc | official | Δ pt | live l1 | >1% a/P/g/s/u/S/h4/h5/z | identity all slots max (rel) | identity A ∩ K max (rel) '
     '| zero-bit qs / l3 / qs1 | calls g/q/W/Z/t |',
     '|---|---|---|---|---|---|---|---|---|---|']
for r in R:
    name = f'{r["family"]} n{r["n"]}' + (' untrained' if r['untrained'] else '')
    off = r.get('official_acc'); d = '' if off is None or r['untrained'] else f'{(r["acc_full"] - off) * 100:+.2f}'
    m1 = '/'.join(str(r['E1'][s]['n_matter_1pct']) for s in ST)
    e3 = r['E3']; zb = r.get('zero_bits', {})
    mk = e3.get('masked', {}); z = lambda k: zb.get(k, {}).get('frac', float('nan'))
    L.append(f'| {name} | {r["acc_full"]:.4f} | {"" if off is None else f"{off:.4f}"} | {d} | {r["E1"]["a"]["live"]} | {m1} | '
             f'{e3["max_err"]:.3g} ({e3["rel"]:.2g}) | {mk.get("max_err", float("nan")):.3g} ({mk.get("rel", float("nan")):.2g}) | '
             f'{z("qs"):.2f} / {z("l3"):.2f} / {z("qs1"):.2f} | {"/".join(f"{v:.2f}" for v in r.get("call_frac", []))} |')
O = D / 'official_acc.json'
if O.exists():
    o = json.loads(O.read_text())
    L += ['', '## Reported test accuracy of the 20 official variants (not run here)', '',
          'Source: the JEDI-linear summary files (' + o['_source'] + '). Test set: 260,000 jets.', '',
          '| family | n8 | n16 | n32 | n64 | n128 |', '|---|---|---|---|---|---|']
    for f in FAM:
        L.append(f'| {f} | ' + ' | '.join(f'{o[f][str(k)] * 100:.2f}' for k in (8, 16, 32, 64, 128)) + ' |')
    best = max(((f, k, o[f][str(k)]) for f in FAM for k in (8, 16, 32, 64, 128)), key=lambda t: t[2])
    best3 = max(((f, k, o[f][str(k)]) for f in FAM[:2] for k in (8, 16, 32, 64, 128)), key=lambda t: t[2])
    L += ['', f'Highest reported: {best[0]} n{best[1]} ({best[2] * 100:.2f}%). Highest 3-feature: {best3[0]} n{best3[1]} '
          f'({best3[2] * 100:.2f}%). 3-feature n64: {o["3-feature"]["64"] * 100:.2f}%.']
(D / 'summary.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
