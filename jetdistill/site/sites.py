"""The whole site: a landing page comparing the setups, and a selector (setup, particle count) on top of every page.
  site/index.html                 landing page, one section per group of setups (+ the untrained-network control)
  site/quantities.html            every jet quantity: symbol, formula, definition, setups that offer it
  site/<setup>/n<N>/index.html    the page of a setup (site/page.py), with the selector injected
Usage: python -m jetdistill.site.sites [site dir]"""
import json, sys
from pathlib import Path
from ..config import SETUPS, RESULTS
from .page import SITE

GROUPS = [('Main result', ['all']), ('Tuned for agreement', ['all_agree']), ('Mass experiments', ['nophys', 'nomass', 'nomass_strict'])]
NS = (8, 64)
TITLE = 'Distilling the Latent Space of a Real-Time Jet Tagger into Interpretable Physics Equations'
SUMMARY = (('network', 'main', 'tuned on the network'), ('network', 'step4', 'smaller version'), ('neurons', 'step4', 'tuned on the neuron values, smaller'),
           ('labels', 'step4', 'tuned on true labels, smaller'))


def available(site, e, n): return (Path(site) / e / f'n{n}' / 'index.html').exists()


def selector(site, exp, n):
    def link(e, m, text, on):
        if not available(site, e, m): return f'<span style="color:#98a2b3;margin-right:14px">{text} <small>(not run)</small></span>'
        st = 'font-weight:700;color:#1d2433;text-decoration:none;border-bottom:2px solid #1f4e79' if on else 'color:#1f4e79;text-decoration:none'
        return f'<a href="../../{e}/n{m}/index.html" style="{st};margin-right:14px">{text}</a>'
    setups = '<span style="color:#d0d5dd;margin-right:10px">·</span>'.join(''.join(link(e, n, SETUPS[e].label, e == exp) for e in es) for _, es in GROUPS)
    parts = ''.join(link(exp, m, f'{m} particles', m == n) for m in NS)
    return ('<nav style="font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:#fff;border-bottom:1px solid #e4e7ec;padding:10px 16px;position:sticky;top:0;z-index:50">'
            f'<a href="../../index.html" style="color:#1f4e79;text-decoration:none;margin-right:18px">← Overview</a><span style="color:#667085;margin-right:8px">Setup:</span>{setups}'
            f'<span style="color:#667085;margin:0 8px 0 10px">|</span>{parts}</nav>')


def summary(e, n):
    base = RESULTS / e / f'n{n}'; reg = json.loads((base / 'formulas.json').read_text()); M = json.loads((base / 'metrics.json').read_text())
    rows = {r['formula']: r for r in M['models'] if r.get('formula')}; out = []
    for fam, role, lab in SUMMARY:
        ts = sorted((t for t, m in reg.items() if m['family'] == fam and m['role'] == role), key=lambda t: -rows[t]['same_as_network'])
        if ts: r = rows[ts[0]]; out.append((lab, r['terms'], r['same_as_network'], r['accuracy']))
    return M['models'][0]['accuracy'], out


def control_html():
    """the untrained-network control (analysis/control.py): how well each kind of fit describes the neurons of the trained and
    of the untrained network (median over neurons, R² on test jets)"""
    from ..analysis.control import numbers
    pc = lambda v: f'{100 * v:.0f}%'; keys = ('physics', 'orientation', 'raw_inputs', 'trees')
    rows = [(n, [[N[k]['trained'], N[k]['untrained']] for k in keys]) for n in NS if (N := numbers(n))]
    if not rows: return '<p class="cnt">Not run (python -m jetdistill.analysis.control 8, and 64).</p>'
    labels = ['R², step 1 on the physics observables', 'orientation share of the step-1 formula', 'R², step 1 on the raw inputs (pT, Δη, Δφ)',
              'R², boosted decision trees on the raw inputs']
    head = ''.join(f'<th>{n} particles, trained</th><th>{n} particles, untrained</th>' for n, _ in rows)
    body = ''.join(f'<tr><td>{lab}</td>' + ''.join(''.join(f'<td>{"–" if v is None else pc(v)}</td>' for v in V[i]) for _, V in rows) + '</tr>' for i, lab in enumerate(labels))
    return ('<p>Step 1 applied to the same network with random weights (drawn with each layer’s trained mean and spread; same architecture and number formats) and to the trained network. '
            'For the untrained 8-particle network, the formulas use almost only observables that depend on the jet orientation (Δη or Δφ alone).</p>'
            f'<table><tr><th></th>{head}</tr>{body}</table>'
            '<p class="cnt">R² of each neuron, median over the neurons, test jets; all fits on training jets. Orientation share: share of the formula’s term importance on Δη- or Δφ-only observables.</p>')


def quantities_page(site):
    """site/quantities.html: every jet quantity the formulas may use, with its symbol, formula, definition, the particle counts
    it exists for and the setups that offer it"""
    import html as H
    from .. import pipeline as P
    from ..analysis.control import category, CATS
    from ..mars import strict_equivalents
    from ..observables import library, mass_ids, symbol, CODE_RENAME
    ids = {}
    for n in NS:
        strict = strict_equivalents(P.jets(n, 'fit')['Q'], mass_ids(n)) if any(available(site, 'nomass_strict', m) for m in NS) else set()
        for q, o in library(n).items():
            x = ids.setdefault(q, dict(o=o, n=[], strict=[], mass=q in mass_ids(n))); x['n'].append(n)
            if q in strict: x['strict'].append(n)
    def offer(x):
        if x['mass']: return 'all observables; no W/Z/t mass thresholds; tuned for agreement'
        if x['strict']: return 'all except “no mass observables or exact equivalents”' + ('' if len(x['strict']) == len(x['n']) else f' ({", ".join(map(str, x["strict"]))} particles)')
        return 'all setups'
    rows = {c: [] for c in CATS}
    for q, x in ids.items(): rows[category(q)].append((q, x))
    body = ''.join(f'<h2>{c} ({len(rows[c])})</h2><table><tr><th>symbol</th><th>formula</th><th>definition</th><th>particles</th><th>setups that offer it</th><th>code name</th></tr>'
                   + ''.join(f'<tr><td><b>{H.escape(symbol(q))}</b></td><td>{H.escape(x["o"].label)}</td><td>{H.escape(x["o"].desc)}</td><td>{", ".join(map(str, x["n"]))}</td>'
                             f'<td>{offer(x)}</td><td><code>{CODE_RENAME.get(q, q)}</code></td></tr>' for q, x in sorted(rows[c])) + '</table>' for c in CATS)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jet Quantities</title>
<style>:root{{--ink:#1d2433;--mut:#667085;--line:#e4e7ec;--bg:#f7f8fa;--acc:#1f4e79}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:30px 16px 60px}}h1{{font-size:24px;margin:0 0 6px}}h2{{font-size:18px;margin:26px 0 8px}}.cnt{{color:var(--mut);font-size:13px}}a{{color:var(--acc)}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid var(--line)}}td,th{{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{background:#f2f4f7;font-weight:600}}code{{font-size:12px;color:var(--mut)}}
@media(max-width:700px){{td,th{{display:block}}}}</style></head><body><main>
<div class="cnt"><a href="index.html">← Overview</a></div>
<h1>Jet Quantities</h1>
<p>Every quantity the formulas may test: {sum(8 in x["n"] for x in ids.values())} for the 8-particle network and {sum(64 in x["n"] for x in ids.values())} for the 64-particle network. Each is computed from the pT, Δη and Δφ (relative to the jet axis) of the particles the network sees, hardest first; empty slots have pT = 0.</p>
<p class="cnt">Notation: zᵢ = pTᵢ / Σⱼ pTⱼ; ΔRᵢ = √(Δηᵢ² + Δφᵢ²); ΔRᵢⱼ = distance between particles i and j; masses treat particles as massless. λ₁ ≥ λ₂: eigenvalues of Σᵢ zᵢ (Δηᵢ, Δφᵢ)(Δηᵢ, Δφᵢ)ᵀ (λ₁ + λ₂ = Σᵢ zᵢΔRᵢ²).</p>
<p class="cnt">Setups: “no mass observables” removes every quantity in GeV of mass and every ratio built from masses; “no mass observables or exact equivalents” also removes quantities with |correlation| &gt; 0.98 with (m/ΣpT)² on the training jets.</p>
{body}
<p class="cnt" style="margin-top:20px">Symbol: how the quantity is written on the pages. Code name: its name in the exported Python files.</p></main></body></html>'''
    (Path(site) / 'quantities.html').write_text(page)


def landing(site):
    def cell(e, n):
        if not available(site, e, n): return '<td class="cnt">not run</td>'
        netacc, fs = summary(e, n)
        lines = ''.join(f'<div><b>{t}</b> if-statements, {lab}: {100 * s:.2f}% same class as the network, accuracy {100 * a:.2f}%</div>' for lab, t, s, a in fs)
        return f'<td><a href="{e}/n{n}/index.html">open the page</a><div class="cnt">network accuracy {100 * netacc:.2f}%</div>{lines}</td>'
    sections = [f'<h2>{name}</h2><table><tr><th style="width:30%">setup</th><th>8 particles</th><th>64 particles</th></tr>'
                + ''.join(f'<tr><td><b>{SETUPS[e].label}</b><div class="cnt">{SETUPS[e].page_desc}</div></td>{"".join(cell(e, n) for n in NS)}</tr>' for e in es) + '</table>' for name, es in GROUPS]
    sections.append(f'<h2>Control: untrained network</h2>{control_html()}')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>
<style>:root{{--ink:#1d2433;--mut:#667085;--line:#e4e7ec;--bg:#f7f8fa;--acc:#1f4e79}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:36px 16px 60px}}h1{{margin:0 0 6px;font-size:26px}}h2{{font-size:19px;margin:30px 0 10px}}.cnt{{color:var(--mut);font-size:13px}}a{{color:var(--acc)}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid var(--line);border-radius:10px}}td,th{{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{background:#f2f4f7;font-weight:600}}
@media(max-width:700px){{td,th{{display:block}}}}</style></head><body><main>
<h1>{TITLE}</h1><div class="cnt">The JEDI-linear jet tagger’s last hidden layer, written as if-statements on jet physics quantities. <a href="quantities.html">Jet quantities used</a></div>
<p>A small neural network sorts particle jets into gluon (g), light quark (q), W, Z and top (t). It decides from 16 numbers, its last hidden layer. Each of those numbers is rewritten here as a sum of if-statements on physics quantities of the jet, tuned so that the network’s own last step, fed with them, makes the same decisions as the network, while each formula number stays close to the network’s.</p>
<p>The method was run in several setups that differ in which quantities and thresholds it may use or in what the tuning aims at. Each setup has its own page for 8 and 64 particles per jet.</p>
{"".join(sections)}
<p class="cnt" style="margin-top:24px">All numbers on the whole test file unless stated. “Same class as the network” = share of test jets on which the formula gives the network’s answer.</p>
<p class="cnt">Model: JEDI-linear (github.com/calad0i/JEDI-linear).</p></main></body></html>'''
    (Path(site) / 'index.html').write_text(html)


def main(site=SITE):
    site = Path(site)
    for _, es in GROUPS:
        for e in es:
            for n in NS:
                p = site / e / f'n{n}' / 'index.html'
                if not p.exists(): continue
                h = p.read_text()
                if '<nav ' in h[:h.index('<main>')]: h = h[:h.index('<nav ')] + h[h.index('</nav>') + 6:]
                i = h.index('>', h.index('<body')) + 1; p.write_text(h[:i] + selector(site, e, n) + h[i:])
    landing(site); quantities_page(site); (site / '.nojekyll').write_text('')


if __name__ == '__main__':
    main(*sys.argv[1:])
