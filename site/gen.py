"""Render site/content.py into one tabbed HTML page (PNG figures embedded).  python site/gen.py [out.html]"""
import base64, html, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'site'))
import content as C

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'site' / 'index.html'
e = html.escape


def img(path):
    p = ROOT / path
    if not p.exists(): return f'<p class="mut">figure missing: {e(path)}</p>'
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{e(path)}">'


def table(cols, rows):
    h = ''.join(f'<th>{e(c)}</th>' for c in cols)
    b = ''.join('<tr>' + ''.join(f'<td>{e(str(v))}</td>' for v in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def claim(c):
    d = c['detail']
    conf = f'<span class="bar"><i style="width:{int(c["confidence"]*100)}%"></i></span> {c["confidence"]:.2f}'
    tabs_ = ''.join(f'<h4>{e(t)}</h4>{table(cols, rows)}' for t, cols, rows in d.get('tables', []))
    figs = ''.join(f'<figure>{img(f)}<figcaption>{e(cap)}</figcaption></figure>' for f, cap in d.get('figures', []))
    return f'''<details class="claim"><summary><span class="cid">{e(c['id'])}</span> {e(c['statement'])}
<span class="meta">{e(c['status'])} · confidence {conf} · {e(c['evidence'])}</span></summary>
<div class="body">
<h4>What the claim is</h4><p>{e(d['what'])}</p>
<h4>Procedure</h4><p>{e(d['procedure'])}</p>
<h4>Result</h4><p>{e(d['result'])}</p>
{tabs_}{figs}
<h4>Verification</h4><p>{e(d['verification'])}</p>
<p class="mut">Files: {e(d['files'])}</p>
</div></details>'''


def paper(p):
    claims = ''.join(claim(c) for c in p['claims'])
    log = ''.join(f'<li><b>{e(v)}</b> — {e(t)}</li>' for v, t in p['thesis_log'])
    opens = ''.join(f'<li>{e(o)}</li>' for o in p['open'])
    return f'''<section id="tab-{p['key']}" class="tab" hidden>
<h2>{e(p['title'])}</h2>
<h3>Thesis (current)</h3><p class="thesis">{e(p['thesis'])}</p>
<details><summary>Thesis history</summary><ul>{log}</ul></details>
<h3>Claims (open each for procedure, numbers, tables, figures)</h3>{claims}
<h3>Open</h3><ul>{opens}</ul>
</section>'''


ov = C.OVERVIEW
overview = f'''<section id="tab-overview" class="tab">
<h2>{e(ov['title'])}</h2><p>{e(ov['intro'])}</p>
<h3>Running</h3>{table(['what', 'state'], ov['status'])}
<h3>Decisions</h3>{table(['date', 'decision'], ov['decisions'])}
<h3>Papers</h3>{table(['tab', 'title', 'claims verified / total'], [(p['tab'], p['title'], f"{sum(1 for c in p['claims'] if c['status'].startswith('verified'))} / {len(p['claims'])}") for p in C.PAPERS])}
</section>'''

tabs = [('overview', 'Overview')] + [(p['key'], p['tab']) for p in C.PAPERS]
nav = ''.join(f'<button data-tab="{k}" role="tab">{e(t)}</button>' for k, t in tabs)
page = f'''<title>Tagger Interp Results</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1f24;--mut:#5f6673;--line:#e1ddd3;--acc:#1f5f8b;--card:#ffffff;--bar:#cfd9e2}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15171c;--fg:#e8e6e0;--mut:#9aa1ad;--line:#2d313a;--acc:#7db3dc;--card:#1d2027;--bar:#2f3a46;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#15171c;--fg:#e8e6e0;--mut:#9aa1ad;--line:#2d313a;--acc:#7db3dc;--card:#1d2027;--bar:#2f3a46;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;margin:0}}
main{{max-width:1100px;margin:0 auto;padding-block:20px;padding-inline:16px}}
h1{{font-size:1.5rem;margin:0;text-wrap:balance}} .sub{{color:var(--mut);margin:2px 0 14px}}
h2{{font-size:1.2rem;margin:18px 0 8px;text-wrap:balance}} h3{{font-size:.95rem;margin:22px 0 6px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}}
nav{{position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding:6px 0;z-index:2}}
nav button{{background:none;border:1px solid transparent;border-radius:6px;padding:6px 10px;font:inherit;color:var(--mut);cursor:pointer}}
nav button[aria-selected="true"]{{color:var(--acc);border-color:var(--line);background:var(--card)}}
p{{max-width:78ch}} .thesis{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px}}
figure{{margin:12px 0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px}} figure img{{max-width:100%;height:auto;display:block}}
figcaption{{font-size:.88rem;color:var(--mut);margin-top:8px;max-width:90ch}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}} .tw{{overflow-x:auto;margin:6px 0}}
th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--mut);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}}
.bar{{display:inline-block;width:60px;height:8px;background:var(--bar);border-radius:4px;vertical-align:middle;margin-right:4px}} .bar i{{display:block;height:100%;background:var(--acc);border-radius:4px}}
.mut{{color:var(--mut)}} details{{margin:6px 0}} summary{{cursor:pointer;color:var(--acc)}} ul{{padding-left:20px}} li{{margin:3px 0}}
details.claim{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;margin:8px 0}} details.claim summary{{color:var(--fg);list-style:none;display:block}}
details.claim summary::before{{content:'▸ ';color:var(--acc)}} details.claim[open] summary::before{{content:'▾ '}}
.cid{{font-weight:700;color:var(--acc);margin-right:4px}} .meta{{display:block;font-size:.82rem;color:var(--mut);margin-top:2px}}
details.claim .body{{border-top:1px solid var(--line);margin-top:8px;padding-top:4px}} h4{{font-size:.82rem;margin:14px 0 4px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}}
</style>
<main>
<h1>Tagger Interp Results</h1>
<p class="sub">Updated {e(C.UPDATED)}. Measurements, figures and claim status per paper; theses change only with a dated entry.</p>
<nav role="tablist">{nav}</nav>
{overview}
{''.join(paper(p) for p in C.PAPERS)}
</main>
<script>
(function(){{
  var bs=document.querySelectorAll('nav button'),tabs=document.querySelectorAll('.tab');
  function show(k){{tabs.forEach(function(t){{t.hidden=(t.id!=='tab-'+k)}});bs.forEach(function(b){{b.setAttribute('aria-selected',b.dataset.tab===k)}});
    try{{localStorage.setItem('tab',k)}}catch(e){{}}}}
  bs.forEach(function(b){{b.addEventListener('click',function(){{show(b.dataset.tab);history.replaceState(null,'','#'+b.dataset.tab)}})}});
  var k=(location.hash||'').replace('#','');var ok=Array.prototype.some.call(bs,function(b){{return b.dataset.tab===k}});
  if(!ok){{try{{k=localStorage.getItem('tab')||'overview'}}catch(e){{k='overview'}}}}
  if(!Array.prototype.some.call(bs,function(b){{return b.dataset.tab===k}}))k='overview';
  show(k);
}})();
</script>'''
OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(page)
print(OUT, len(page) // 1024, 'KB')
