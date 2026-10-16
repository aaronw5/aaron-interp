"""Check the written explanations against the numbers (no contradictions allowed).
For every neuron and class-score text: each claim that it raises / lowers / does not enter a class's score must match the
sign of that neuron's weight share in that class score (|share| > 0.02 = enters); each number quoted next to a class
("5.89 on g", "g (5.89)", "mean 5.89 for gluons") must match that class's mean value within 0.02 (or 2 %).
Usage (each prints every problem and exits 1 if there is any):
  python -m jetdistill.explain.check texts   <pack.json> <explain.json>
  python -m jetdistill.explain.check combos  <combos_pack.json> <combos.json>
  python -m jetdistill.explain.check numbers <combos_pack.json> <combos.json>
  python -m jetdistill.explain.check nolabels <explain.json or combos.json>"""
import json, re, sys

CL = {'g': 'g', 'gluon': 'g', 'gluons': 'g', 'q': 'q', 'quark': 'q', 'quarks': 'q', 'light-quark': 'q', 'w': 'W', 'z': 'Z',
      't': 't', 'top': 't', 'tops': 't'}
CW = r'(gluons?|quarks?|light-quark|tops?|[gqWZt])'
UP = r'(raises|raise|increases|boosts|pushes up|adds to|favou?rs|supports|votes? for|helps)'
DN = r'(lowers|lower|reduces|decreases|suppresses|pushes down|subtracts from|penali[sz]es|votes? against|counts against)'
NO = r'(does not enter|doesn\'t enter|not used by|ignored by|no weight in|does not feed|not in)'


def classes_in(s):
    return [CL[m.lower() if m not in 'WZ' else m.lower()] for m in re.findall(CW + r'(?=[\s,/)\-]|$)', s)]


def check_text(txt, shares, means, where):
    probs = []; txt = txt.replace('−', '-')
    for sent in re.split(r'(?<=[.;])\s+', txt):
        for pat, kind in ((UP, 'up'), (DN, 'down'), (NO, 'none')):
            for m in re.finditer(pat + r'([^.;]{0,90}?)score', sent, flags=re.I):
                for c in classes_in(m.group(2)):
                    s = shares.get(c)
                    if s is None: continue
                    bad = (kind == 'up' and s <= 0) or (kind == 'down' and s >= 0) or (kind == 'none' and abs(s) > 0.02)
                    if bad: probs.append(f'{where}: says it {kind} the {c} score, but its share there is {s:+.3f}: "{sent.strip()[:160]}"')
        if means:
            for m in re.finditer(r'(?<![\d.\-–])(\d+\.\d+)(?![\d]*\s*[-–]\s*\d)\s*(?:on|for|in)\s+(?:true[- ])?' + CW + r'(?=[\s,/)\-;.]|$)', sent):
                c = CL.get(m.group(2).lower()); v = float(m.group(1)); pre = sent[max(0, m.start() - 14):m.start()]
                if re.search(r'[\d.]\s*[-–]\s*$', pre): continue                       # the end of a range "1.7-2.1 for W/Z/t"
                bm = re.search(r'(≤|<=|<|below|under|at most|≥|>=|>|above|over|at least)\s*$', pre)
                if bm and c in means:                                                    # a bound: check the inequality
                    le = bm.group(1) in ('≤', '<=', '<', 'below', 'under', 'at most')
                    if (le and means[c] > v + 0.02) or (not le and means[c] < v - 0.02):
                        probs.append(f'{where}: says {bm.group(1)} {v} for {c}, but its mean for {c} is {means[c]:.3f}: "{sent.strip()[:160]}"')
                    continue
                tol = 0.25 * abs(means.get(c, 0)) + 0.02 if re.search(r'(about|around|~|≈|≤|≥|<|>|roughly|near)\s*$', pre) else max(0.02, 0.02 * abs(means.get(c, 0)))
                if c in means and abs(v - means[c]) > tol and 'AUC' not in pre:
                    probs.append(f'{where}: quotes {v} for {c}, but its mean for {c} is {means[c]:.3f}: "{sent.strip()[:160]}"')
    return probs


def main(pack_f, expl_f):
    P = json.load(open(pack_f)); E = json.load(open(expl_f)); probs = []
    for n in P['neurons']:
        j = str(n['neuron']); x = E['neurons'].get(j, {}); sh = n['weight_share_in_each_class_score']; mn = n['mean_value_by_true_class']
        txt = ' '.join(str(x.get(k, '')) for k in ('title', 'measures', 'role', 'detects', 'effect', 'explanation', 'interpretation'))
        probs += check_text(txt, sh, mn, f'neuron {j}')
    for s in P['class_scores']:
        x = E.get('class_scores', {}).get(s['cls'], {}); txt = ' '.join(str(v) for v in x.values())
        probs += check_text(txt, {}, s['mean_score_by_true_class'], f"score {s['cls']}")
    for p in probs: print(p)
    print(f'{len(probs)} problem(s)'); return 1 if probs else 0



def check_combos(pack_f, combos_f):
    """combination texts: 'mostly X' / 'mainly X' / 'dominated by X' must name the largest true class (>= 35 %); 'the formula calls
    them X' / 'decided as X' must name the class the formula decides most; 'the neuron is off / on' must match its on-rate."""
    P = json.load(open(pack_f)); C = json.load(open(combos_f)); probs = []; names = ['g', 'q', 'W', 'Z', 't']
    for n in P['neurons']:
        for i, pat in enumerate(n['patterns']):
            x = C.get('neurons', {}).get(str(n['neuron']), {}).get(str(i))
            if not x: probs.append(f"neuron {n['neuron']} pattern {i}: no text"); continue
            txt = ' '.join(str(v) for v in x.values()); top = names[max(range(5), key=lambda c: pat['classes'][c])]; dec = names[max(range(5), key=lambda c: pat['formula_decides'][c])]
            for m in re.finditer(r'(?i:mostly|mainly|dominated by|predominantly)\s+(?:(?i:true)[- ])?(?i:(gluons?|quarks?|light-quark|tops?)|([gqWZt]))(?=[\s,/)\-;.]|$)', txt):
                c = CL.get((m.group(1) or m.group(2) or "").lower())
                if c and (c != top or pat['classes'][names.index(c)] < 0.35) and not re.search(r'\b(and|/|or)\s*$', txt[m.end():m.end() + 6]):
                    probs.append(f"neuron {n['neuron']} pattern {i}: says mostly {c}, but the largest class is {top} ({pat['classes'][names.index(top)]:.2f})")
            for m in re.finditer(r'(?i:formula (?:calls|decides|labels|tags) (?:them|these|most of them)?\s*(?:as)?\s*|decided as\s+)(?i:(gluons?|quarks?|light-quark|tops?)|([gqWZt]))(?=[\s,/)\-;.]|$)', txt):
                c = CL.get((m.group(1) or m.group(2) or "").lower())
                if c and c != dec: probs.append(f"neuron {n['neuron']} pattern {i}: says the formula calls them {c}, but it decides {dec} most")
            if re.search(r'neuron (is )?(off|zero)\b', txt, flags=re.I) and pat['neuron_on'] > 0.5: probs.append(f"neuron {n['neuron']} pattern {i}: says the neuron is off, but it is on for {pat['neuron_on']:.2f}")
    for p in probs: print(p)
    print(f'{len(probs)} problem(s)'); return 1 if probs else 0



def check_numbers(pack_f, text_f):
    """every number quoted in the text must match some number in the pack (within 1.5 % or 0.011), or be a percentage of
    a pack fraction, or a simple rounded threshold that appears in a rule"""
    P = open(pack_f).read(); nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', P)]; T = json.load(open(text_f)); probs = []
    def ok(v):
        for w in nums:
            if abs(v - w) <= max(0.011, 0.015 * abs(w)) or abs(v - 100 * w) <= max(0.6, 0.015 * abs(100 * w)) or abs(-v - w) <= max(0.011, 0.015 * abs(w)): return True
        return False
    def walk(o, path):
        if isinstance(o, dict): [walk(v, f'{path}/{k}') for k, v in o.items()]
        elif isinstance(o, list): [walk(v, f'{path}/{i}') for i, v in enumerate(o)]
        elif isinstance(o, str):
            o = o.replace('−', '-')
            for m in re.finditer(r'(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])', o):
                v = float(m.group(1))
                if v in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16) and '.' not in m.group(1): continue            # small counts / indices
                if not ok(v): probs.append(f'{path}: {v} not found in the pack: "…{o[max(0, m.start() - 60):m.end() + 30]}…"')
    walk(T, ''); [print(p) for p in probs]; print(f'{len(probs)} problem(s)'); return 1 if probs else 0



def check_nolabels(text_f):
    """names must describe physics, not classes: neuron titles, class-score titles excepted, and group names may not contain a
    class name; no text may call anything '<class>-like' or '<class>-likeness'"""
    T = json.load(open(text_f)); probs = []
    CLASSW = r'\b(gluons?|quarks?|light-quark|tops?|bosons?|W|Z|g|q|t)\b'
    LIKE = r'\b(gluon|quark|top|boson|W|Z|q|g|t|W/Z|two-prong boson)[- ]?like(ness)?\b|likeness'
    for j, x in (T.get('neurons') or {}).items():
        if isinstance(x, dict) and 'title' in x and re.search(CLASSW, x['title']): probs.append(f"neuron {j} title names a class: {x['title']}")
        for k, v in (x.items() if isinstance(x, dict) else []):
            if k in ('title', 'measures', 'role', 'name') and isinstance(v, str) and re.search(LIKE, v, flags=re.I): probs.append(f"neuron {j} {k}: class-likeness wording: \"{v[:120]}\"")
            if isinstance(v, dict):                                              # group texts: neurons -> group -> {name, jets, why}
                if 'name' in v and re.search(CLASSW, v['name']): probs.append(f"neuron {j} group {k} name names a class: {v['name']}")
                for kk, vv in v.items():
                    if isinstance(vv, str) and re.search(LIKE, vv, flags=re.I): probs.append(f"neuron {j} group {k} {kk}: class-likeness wording: \"{vv[:120]}\"")
    for c, x in (T.get('class_scores') or {}).items():
        for k, v in x.items():
            if isinstance(v, str) and re.search(LIKE, v, flags=re.I): probs.append(f"score {c} {k}: class-likeness wording: \"{v[:120]}\"")
    if isinstance(T.get('summary'), str) and re.search(LIKE, T['summary'], flags=re.I): probs.append('summary: class-likeness wording')
    for p in probs: print(p)
    print(f'{len(probs)} problem(s)'); return 1 if probs else 0



if __name__ == '__main__':
    what, *files = sys.argv[1:]
    sys.exit(dict(texts=main, combos=check_combos, numbers=check_numbers, nolabels=check_nolabels)[what](*files))
