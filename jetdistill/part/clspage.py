"""The pages of the class fit and the attention fit (one each): what was fitted, the paper's metrics on the test jets next to
ParT and the jet-level formula, per class, the tuning path, what the fit uses, the files.

  python -m jetdistill.part.clspage cls|attn OUT_DIR"""
import html, json, shutil, sys, pathlib
import numpy as np
from ..config import RESULTS, CLASSES
from .clsfit import particle_features, basis, PFEAT, CTX
from .attnfit import start_scores, HEADS

CSS = """:root{--ink:#1d2433;--mut:#667085;--line:#e4e7ec;--bg:#f7f8fa;--card:#fff;--acc:#1f4e79;--good:#2f855a;--bad:#c05621}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:18px 16px 60px}h1{font-size:22px;margin:0 0 6px}h2{font-size:17px;margin:22px 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0;overflow-x:auto}
.cnt{color:var(--mut);font-size:12.5px}table{border-collapse:collapse}td,th{padding:4px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;background:#f2f4f7}.num{text-align:right;font-variant-numeric:tabular-nums}.best{font-weight:700}a{color:var(--acc)}
.kpi{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin:4px 8px 4px 0;min-width:150px}.kpi b{font-size:20px;display:block}
pre{font:12px/1.55 ui-monospace,Menlo,monospace;background:#f6f8fa;border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow-x:auto;white-space:pre}
.bars td{padding:2px 8px}.bar{height:10px;background:var(--acc);border-radius:3px}.up{color:var(--good)}.dn{color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:10px}"""

TITLE = dict(cls='Class fit: formulas for the input of ParT’s class attention',
             attn='Attention fit: formulas with their own class attention')
WHAT = dict(cls="""<p>ParT ends with two <b>class-attention blocks</b>: a class token attends to the particles’ embeddings (after ParT’s 8 particle-attention
blocks), and its final state gives the 128 neurons and the class scores. Here every particle’s embedding is replaced by a <b>formula of that particle’s
own inputs</b> and 20 jet quantities, and ParT’s own (frozen) class blocks do the rest.</p>
<ul><li>per particle: 18 inputs (ParT’s 17 and the pT rank) and 20 jet quantities, each as the value and hinge terms max(0, x − t), max(0, t − x) at 9 thresholds: 551 terms</li>
<li>the formula gives the particle’s position along the 32 main directions of ParT’s normalized embeddings (these 32 keep 96.6 % agreement when ParT’s own embeddings are cut to them)</li>
<li>fit: least squares to ParT’s embeddings, then tuned through the class blocks toward ParT’s class probabilities, plus terms keeping the neurons
(λ = 0.01) and the embeddings (λ<sub>e</sub> = 1) close to ParT’s</li></ul>""",
            attn="""<p>Here the formula replaces ParT’s class attention <i>and</i> everything before it: 128 neurons from the particles through an attention written out
as formulas, built like ParT’s two class blocks, then ParT’s own last layer.</p>
<ul><li><b>layer 1</b> (as ParT’s first class block, where the class token is a constant): 8 heads; head h gives particle i the score
s<sub>hi</sub> = a fixed start (hardest, uniform, near the axis, charged, displaced, leptons, photons, most energetic) + a formula of the particle’s terms; the class token scores
<b>itself</b> with a fitted constant b<sub>h</sub>; weights α<sub>h</sub> = softmax over [b<sub>h</sub>, s<sub>h1</sub>, …, s<sub>hn</sub>]. The head returns the weighted sum of the
particles’ terms and its <b>self-weight</b> α<sub>cls,h</sub> = 1 − Σ<sub>i</sub> α<sub>hi</sub> (how much the particles count at all, which a plain average over particles cannot say)</li>
<li><b>layer 2</b> (as ParT’s second class block, where the class token is block 1’s output): the same particles; the scores are bilinear,
s<sub>hi</sub> = φ<sub>i</sub>·A<sub>h</sub>·[z₁, 1], with z₁ layer 1’s 16-number summary of the jet; the self-score b<sub>h</sub>·[z₁, 1] depends on the jet too</li>
<li>neurons: linear in both layers’ pooled terms and self-weights; tuned toward ParT’s class probabilities (+ λ = 0.01 on the neurons)</li></ul>"""
)


def pct(x): return f'{100 * x:.2f}%'


def chart(path, w=560, h=200):
    """SVG of the validation agreement over the tuning steps"""
    s = [p[0] for p in path]; a = [100 * p[1] for p in path]; lo, hi = min(a) - .5, max(a) + .5; X = lambda v: 40 + (w - 60) * v / max(s); Y = lambda v: h - 25 - (h - 40) * (v - lo) / (hi - lo)
    pts = ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in zip(s, a)); b = int(np.argmax(a))
    ticks = ''.join(f'<text x="34" y="{Y(v) + 4:.1f}" font-size="10" text-anchor="end" fill="#667">{v:.0f}%</text><line x1="40" x2="{w - 20}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#eef0f3"/>'
                    for v in np.arange(np.ceil(lo), hi, max(1, round((hi - lo) / 5))))
    return (f'<svg viewBox="0 0 {w} {h}" style="max-width:{w}px;width:100%">{ticks}<polyline points="{pts}" fill="none" stroke="#1f4e79" stroke-width="2"/>'
            + ''.join(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.5" fill="#1f4e79"/>' for x, y in zip(s, a))
            + f'<circle cx="{X(s[b]):.1f}" cy="{Y(a[b]):.1f}" r="5" fill="none" stroke="#2f855a" stroke-width="2"/>'
            + f'<text x="{X(s[b]):.1f}" y="{Y(a[b]) - 9:.1f}" font-size="11" text-anchor="middle" fill="#2f855a">best {a[b]:.2f}% (step {s[b]})</text>'
            + f'<text x="{w / 2}" y="{h - 4}" font-size="11" text-anchor="middle" fill="#667">tuning step (0 = least-squares start)</text></svg>')


def groups(knots):
    """the term columns of each input (basis order: 1, then per input its value and hinges)"""
    out, c = [], 1
    for kn in knots: out.append(np.arange(c, c + 1 + 2 * len(kn))); c += 1 + 2 * len(kn)
    return out


def bars(imp, names, top=14):
    o = np.argsort(-imp)[:top]; mx = imp[o[0]]
    return '<table class="bars">' + ''.join(f'<tr><td>{html.escape(names[i])}</td><td style="width:220px"><div class="bar" style="width:{200 * imp[i] / mx:.0f}px"></div></td>'
                                            f'<td class="num">{100 * imp[i]:.1f}%</td></tr>' for i in o) + '</table>'


def usage(kind, net='full', n=3000):
    """what the fit uses, on n validation jets: each input's share of the fitted variation (cls: of the embeddings;
    attn: of each head's scores), and for attn the heads' self-weights"""
    import torch
    from ..pipeline import jets
    d = RESULTS / f'{kind}_fit' / net; meta = json.loads((d / f'{kind}_fit.json').read_text()); Pz = np.load(d / f'{kind}_fit.npz')
    knots = [np.asarray(k, np.float32) for k in meta['knots']]; G = groups(knots); names = PFEAT + CTX
    J = jets(net, 'dev'); F, ok = particle_features(J, np.arange(n)); Ft = torch.from_numpy(F).half().float(); B = basis(Ft, knots).numpy()
    def share(Wf):                                                            # Wf: (K, d) plain coefficients -> each input's share of the variance over the real particles
        Br = B[ok]; v = np.array([((Br[:, g] - Br[:, g].mean(0)) @ Wf[g]).var(0).sum() for g in G]); return v / v.sum()
    if kind == 'cls': return dict(overall=share(Pz['W'])), names
    M = Pz['M']; A1 = M @ Pz['A']; out = dict(heads1=[share(A1[:, [h]]) for h in range(8)])
    if meta['layers'] == 2:                                                   # layer 2: the score directions differ per jet; their average over the jets
        p = {k: torch.from_numpy(Pz[k]) for k in Pz.files}; Mt = p['M']; m = torch.from_numpy(ok)
        Phi = basis(Ft, knots) @ Mt; sc = (start_scores(Ft) + Phi @ p['A']).masked_fill(~m[..., None], -1e9); sc = torch.cat([p['b_cls'].expand(n, 1, -1), sc], 1); al = torch.softmax(sc, 1)
        X = torch.cat([torch.einsum('nph,npk->nhk', al[:, 1:], Phi), al[:, 0, :, None]], 2).reshape(n, -1); z1 = torch.cat([X @ p['W1'] + p['c1'], torch.ones(n, 1)], 1)
        Q2 = torch.einsum('khr,nr->kh', p['A2'], z1 / n).numpy(); out['heads2'] = [share(M @ Q2[:, [h]]) for h in range(8)]
        sc2 = (start_scores(Ft) + torch.einsum('npk,khr,nr->nph', Phi, p['A2'], z1)).masked_fill(~m[..., None], -1e9); sc2 = torch.cat([(z1 @ p['b2_cls'].T)[:, None], sc2], 1)
        out['self2'] = torch.softmax(sc2, 1)[:, 0].numpy()
    out['self1'] = al[:, 0].numpy() if meta['layers'] == 2 else None; out['mult'] = ok.sum(1)
    return out, names


def page(kind, outdir, net='full'):
    d = RESULTS / f'{kind}_fit' / net; meta = json.loads((d / f'{kind}_fit.json').read_text()); r = json.loads((d / 'metrics_full_test.json').read_text())
    base = json.loads((RESULTS / 'all_plus' / net / 'metrics.json').read_text()); bm = next(m for m in base['models'] if m.get('terms') == 1083)
    rows = [('ParT (ParT_full)', r['part'], None), (f'jet-level formula, all_plus ({bm["terms"]} terms)', bm['paper'], bm['same_as_network']), (TITLE[kind].split(':')[0], r, r['agreement'])]
    rej_cols = list(r['rej'])
    t = ('<table><tr><th>model</th><th class="num">same class as ParT</th><th class="num">accuracy</th><th class="num">AUC</th>'
         + ''.join(f'<th class="num">Rej<sub>{int(r["rej"][c]["eff"] * 1000) / 10:g}%</sub> {c}</th>' for c in rej_cols) + '</tr>')
    for name, m, ag in rows:
        t += (f'<tr><td>{html.escape(name)}</td><td class="num">{"—" if ag is None else pct(ag)}</td><td class="num">{m["accuracy"]:.4f}</td><td class="num">{m["auc"]:.4f}</td>'
              + ''.join(f'<td class="num">{m["rej"][c]["rej"]:.0f}</td>' for c in rej_cols) + '</tr>')
    t += '</table>'
    pc = '<table><tr><th>class</th><th class="num">same class as ParT (jets ParT puts in it)</th><th class="num">accuracy (true jets of it)</th></tr>' + ''.join(
        f'<tr><td>{CLASSES[p["c"]]}</td><td class="num">{pct(p["agreement"])}</td><td class="num">{pct(p["accuracy"])}</td></tr>' for p in r['per_class']) + '</table>'
    U, names = usage(kind, net)
    if kind == 'cls':
        use = f'<p class="cnt">Each input’s share of the variation of the fitted embeddings (32 directions, {3000} validation jets).</p>' + bars(U['overall'], names)
    else:
        cards = ''.join(f'<div class="card"><b>head {h + 1}</b> <span class="cnt">start: {HEADS[h]}</span><br><span class="cnt">layer 1 score from</span>{bars(U["heads1"][h], names, 5)}'
                        + (f'<span class="cnt">layer 2 score from (average over jets)</span>{bars(U["heads2"][h], names, 5)}' if 'heads2' in U else '')
                        + (f'<span class="cnt">self-weight α<sub>cls</sub>: layer 1 mean {U["self1"][:, h].mean():.3f} (corr. with multiplicity {np.corrcoef(U["self1"][:, h], U["mult"])[0, 1]:+.2f})'
                           f'{"; layer 2 mean %.3f" % U["self2"][:, h].mean() if "self2" in U else ""}</span>' if U.get('self1') is not None else '') + '</div>' for h in range(8))
        use = f'<p class="cnt">Per head: which inputs set the learned part of the attention scores (share of its variation; the fixed start score not counted), and how much attention the class token keeps on itself.</p><div class="grid">{cards}</div>'
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(d / f'{kind}_fit.npz', outdir / f'{kind}_fit.npz'); shutil.copy(d / f'{kind}_fit.json', outdir / f'{kind}_fit.json')
    shutil.copy(pathlib.Path(__file__).with_name('clseval.py'), outdir / 'clseval.py')
    n_coef = sum(np.load(d / f'{kind}_fit.npz')[k].size for k in np.load(d / f'{kind}_fit.npz').files if k not in ('M', 'V', 'mu'))
    dl = r['agreement'] - bm['same_as_network']
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE[kind].split(':')[0]}</title><style>{CSS}</style></head><body><main>
<p class="cnt"><a href="../../index.html">← all setups</a></p><h1>{html.escape(TITLE[kind])}</h1>
<p class="cnt">ParT_full · JetClass · {r['n']:,} test jets · tuned on {meta['n_fit']:,} training jets, best of the validation checks (100,000 jets) · {n_coef:,} coefficients</p>
<div class="card">{WHAT[kind]}</div>
<div><span class="kpi">same class as ParT<b>{pct(r['agreement'])}</b><span class="{'up' if dl > 0 else 'dn'}">{100 * dl:+.2f} pt vs the jet-level formula</span></span>
<span class="kpi">accuracy<b>{r['accuracy']:.4f}</b><span class="cnt">ParT {r['part']['accuracy']:.4f}</span></span>
<span class="kpi">AUC<b>{r['auc']:.4f}</b><span class="cnt">ParT {r['part']['auc']:.4f}</span></span></div>
<h2>The ParT paper’s metrics (test jets)</h2><div class="card">{t}<p class="cnt">AUC: macro one-vs-one; Rej<sub>X%</sub>: 1/false-positive rate on QCD at signal efficiency X, with score<sub>S</sub>/(score<sub>S</sub> + score<sub>QCD</sub>) — as in the ParT paper.</p></div>
<h2>Per class</h2><div class="card">{pc}</div>
<h2>Tuning</h2><div class="card">{chart(meta['path'])}<p class="cnt">Validation agreement with ParT every 50 steps (Adam, lr {meta['lr']}, on whitened terms); the best step is kept.
Stopped when flat.</p></div>
<h2>What the fit uses</h2><div class="card">{use}</div>
<h2>Files</h2><div class="card"><a href="{kind}_fit.npz">{kind}_fit.npz</a> (coefficients{', M: the whitening, plain = M @ A' if kind == 'attn' else ', V, mu: the 32 directions'}) ·
<a href="{kind}_fit.json">{kind}_fit.json</a> (thresholds, inputs, tuning path) · <a href="clseval.py">clseval.py</a> (the evaluation: the forward pass, the metrics)
<p class="cnt">Code: branch part-features, jetdistill/part/{'clsfit' if kind == 'cls' else 'attnfit'}.py.</p></div>
</main></body></html>"""
    (outdir / 'index.html').write_text(doc); print('page:', outdir / 'index.html', f'{len(doc) // 1024} kB')


if __name__ == '__main__':
    page(sys.argv[1], sys.argv[2])
