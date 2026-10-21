"""H58 verify (loop D, skeptic): fresh-sample rerun of the H58 matched-50 % fingerprint contrasts plus the controls H58 lacks.

Fresh sample (disjoint from H55 / H55v / H58)
  jc   JetClass test_20M *_103.root (H55 / H55v: *_100, H58: *_102), Hbb / Wqq / Zqq / Tbqq / QCD, 3,000 random rows each
       (default_rng 5892); networks ParT_full, ParT_kin, PN_full, PN_kin (no permuted copies: the contrasts below do not use them).
  top  Top Landscape test split, 3,000 top + 3,000 QCD random rows (default_rng 5893), rows of H55 (5503) and H58 (5803) excluded;
       networks ParT_kin, PN_kin, PFN, EFN, LorentzNet.
  Edits, observables, classical taggers, snap and network calls: the unchanged H55 / H58 code (interp.h58_fingerprint_rerun).
Extra analysis (--part extra), run on the fresh sample and on H58's own stored sample (hls4ml, JetClass, Top), trained nets only
  (a) the H58 main contrasts d_shape, d_groom, d_sm (reproduction; same scaling = RMS over trained nets + @50 taggers).
  (b) window/efficiency-matched control for d_shape: msdpub_rand@50 = the SAME published m_SD window as the shape@50 tagger
      (65-105 / 105-210 GeV) and a per-jet random number u_j < q fixed across all edits (q set for 50 % signal efficiency).
      It has the shape tagger's window and efficiency but a cut that no edit can move. d_tau = cos(net, shape@50) -
      cos(net, msdpub_rand@50) isolates the tau21 / tau32 cut; d_width = cos(net, msdpub_rand@50) - cos(net, msd@50) is the
      window-width part of d_shape. Also shape_narrow@50 = shortest m_SD interval with 70 % of signal + tau cut to 50 %.
  (c) edit dependence: each contrast recomputed with the columns of {decay}, {gout}, {gin}, {gin, gout} removed (scaling
      recomputed), with bootstrap CIs (H58 gave leave-one-out point values only).
  (d) direct grooming test: r = chg(gout) / chg(gin) on the matched gin / gout jets (gin: 3 soft particles soft drop keeps;
      gout: the same ungroomed-mass change, soft drop removes them). msd window: r = 0 by construction; ungroomed m window r = 1.
  Paired jet bootstrap (B draws, the scaling recomputed per draw); two-sided bootstrap p; Holm within each family over cells.
Run: ./part_job.sh d-h58-verify 'python -m interp.h58_verify --part edits --ds jc && ... --ds top && --part nets --group jc_part
     && ... jc_pn && ... top_main && ... top_lnet && python -m interp.h58_verify --part extra'
Outputs: $AI_RESULTS/physics/h58_verify_work/ (tables, logits), $AI_RESULTS/physics/h58_verify.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h58_fingerprint_rerun as G  # noqa: E402

H = G.H
PHYS = H.RES / 'physics'
VWORK = PHYS / 'h58_verify_work'
ORIG = PHYS / 'h58_work'
log = H.log
B_EX, CHUNK = 20000, 1000
DROPS = dict(none=(), no_decay=('decay',), no_gout=('gout',), no_gin=('gin',), no_gin_gout=('gin', 'gout'))


def load_ds_v(ds, per):
    seed = H.SEEDS[ds]
    if ds == 'jc':
        import uproot
        from interp.part_core import read_root, cat, sub, PRE, CL
        from interp.h1_irc_panel import jc_table
        rng = np.random.default_rng(seed); Js = []; files = {}
        for c in H.JC_CL:
            f = G.TEST20M / f'{PRE[c]}_103.root'; n = uproot.open(f)['tree'].num_entries
            idx = np.sort(rng.choice(n, per, replace=False)); parts = []
            for a in range(0, n, 20000):
                loc = idx[(idx >= a) & (idx < a + 20000)] - a
                if len(loc): parts.append(sub(read_root(str(f), start=a, stop=min(a + 20000, n)), loc))
            Js.append(cat(parts)); files[c] = f.name; log(f'JetClass {c}: {per} random rows of {n} ({f.name})')
        J0 = cat(Js); T = jc_table(J0); y = J0['label'].astype(int)
        return T, y, dict(sample=f'JetClass test_20M *_103.root, {per} random rows per class (default_rng {seed})', files=files, classes=list(CL))
    assert ds == 'top'
    from interp.h12_boost_panel import table_from_full
    from interp import data_top as D
    d = D.DST / 'test'; ylab = np.load(d / 'label.npy')
    old55 = H._rows(ylab, 2000, 5503); r58 = np.random.default_rng(5803)
    old58 = np.concatenate([r58.choice(np.setdiff1d(np.flatnonzero(ylab == c), old55), 3000, replace=False) for c in (0, 1)])
    old = np.union1d(old55, old58); rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(ylab == c), old), per, replace=False) for c in (0, 1)]))
    assert not np.isin(idx, old).any()
    F = np.asarray(np.load(d / 'full.npy', mmap_mode='r')[idx]); jet = np.asarray(np.load(d / 'jet.npy', mmap_mode='r')[idx])
    log(f'top: {len(idx)} rows, none of the {len(old)} H55 / H58 rows')
    nn = np.minimum(jet[:, 5].astype(np.int64), F.shape[1]); y = ylab[idx].astype(int); T = table_from_full(F, nn); del F
    T['sgn'] = np.where(T['ax_eta'] < 0, -1.0, 1.0)
    return T, y, dict(sample=f'top test split, {per} per class (default_rng {seed}), H55 + H58 rows excluded', classes=['QCD', 'top'])


def setup_fresh():
    H.SEEDS.update(jc=5892, top=5893); H.PER.update(jc=3000, top=3000)
    H.OUT = VWORK; G.WORK = VWORK; G.NKEEP = 0; H.load_ds = load_ds_v; VWORK.mkdir(parents=True, exist_ok=True)


# ================================================================================================================ extra analysis
def control_calls(Ocl, rows, ysig, kind):
    """msdpub_rand@50 (published m_SD window, edit-invariant random thinning to 50 %) and shape_narrow@50"""
    b = Ocl['base']; lo, hi = H.WIN_T if kind == 'T' else H.WIN_V; ms = b['msd'][rows]
    fw = float(((ms > lo) & (ms < hi))[ysig].mean()); a, z = lo, hi
    if fw < 0.55:
        a, z = G.shortest(ms[ysig], 0.7); fw = float(((ms > a) & (ms < z))[ysig].mean())
    u = np.random.default_rng(5899).random(len(rows)); q = min(1.0, 0.5 / fw)
    out = {'msdpub_rand@50': {c: (o['msd'][rows] > a) & (o['msd'][rows] < z) & (u < q) for c, o in Ocl.items()}}
    info = {'msdpub_rand@50': dict(window=[a, z], q=q)}
    v = 'tau32' if kind == 'T' else 'tau21'; a2, z2 = G.shortest(ms[ysig], 0.7)
    inw = (ms >= a2) & (ms <= z2) & ysig; fw2 = float(inw[ysig].mean()) if ysig.any() else 1.0
    c50 = float(np.quantile(b[v][rows][inw], min(1.0, 0.5 / fw2)))
    out['shape_narrow@50'] = {c: (o['msd'][rows] >= a2) & (o['msd'][rows] <= z2) & (o[v][rows] <= c50) for c, o in Ocl.items()}
    info['shape_narrow@50'] = dict(window=[a2, z2], var=v, cut=c50)
    for tg, cl in out.items(): info[tg].update(eff_sig=float(cl['base'][ysig].mean()), eff_bkg=float(cl['base'][~ysig].mean()))
    return out, info


def load_set(work, ds):
    p = work / f'h55_{ds}_obs.npz'
    if not p.exists(): return None
    Zo = np.load(p); O = {}
    for k in Zo.files:
        c, q = k.split('|'); O.setdefault(c, {})[q] = Zo[k]
    Zt = np.load(work / f'h55_{ds}_tab.npz'); y = Zt['y'].astype(int)
    oks = {c: Zt[f'{c}|ok'] for c in H.CONDS if f'{c}|ok' in Zt.files}
    LO, acc = {}, {}
    for q in sorted(work.glob(f'h58_{ds}_*_lo.npz')):
        Zl = np.load(q)
        for k in Zl.files:
            mn, c, tk = k.split('|')
            if '_perm' in mn: continue
            LO.setdefault(tk, {}).setdefault(mn, {})[c] = Zl[k].astype(np.float64)
        a = json.loads(Path(str(q).replace('_lo', '_acc').replace('.npz', '.json')).read_text())
        acc.update({k: v for k, v in a.items() if not k.startswith('_') and '_perm' not in k})
    return O, y, oks, LO, acc


def extra_ds(work, ds, rng, B):
    S_ = load_set(work, ds)
    if S_ is None: return None
    O, y, oks, LO, acc = S_; edits = [e for e in H.DS_EDITS[ds] if e in oks]; E = len(edits); out = dict(N=int(len(y)), accuracy=acc, tasks={})
    for task, (sig, bkg) in G.tasks_of(ds).items():
        kind = G.TASK_KIND.get(task)
        if kind not in ('V', 'T') or task not in LO: continue
        rows = np.flatnonzero(np.isin(y, [sig] + bkg)); ysig = y[rows] == sig; n = len(rows)
        okr = {c: oks[c][rows] for c in oks}; Ocl, _ = G.snap({c: O[c] for c in ['base'] + edits})
        cc, cinfo = G.window_calls(Ocl, rows, ysig, kind); xc, xinfo = control_calls(Ocl, rows, ysig, kind)
        nets = sorted(LO[task]); names, CH, FL, OK = [], [], [], []
        for mn in nets:
            s = {c: LO[task][mn][c][rows] for c in ['base'] + edits}; thr = float(np.median(s['base'][ysig]))
            ch, fl, ok = H.response_arrays({c: v > thr for c, v in s.items()}, okr, edits); names.append(mn); CH.append(ch); FL.append(fl); OK.append(ok)
        tn = list(G.M50[kind]) + list(xc)
        for tg in tn:
            ch, fl, ok = H.response_arrays((cc.get(tg) or xc[tg]), okr, edits); names.append(tg); CH.append(ch); FL.append(fl); OK.append(ok)
        Rn = len(names); A = G.stackA(np.stack(CH), np.stack(FL), np.stack(OK)); nN = len(nets)
        ref = list(range(nN + len(G.M50[kind])))                         # H58 scaling: trained nets + @50 taggers (controls excluded)
        ix = {t: names.index(t) for t in tn}
        colsets = {k: [j for j in range(2 * E) if edits[j % E] not in d] for k, d in DROPS.items()}
        con = {'d_shape': (G.SHAPE1[kind], G.MSD[kind]), 'd_groom': (G.MSD[kind], G.MASS[kind]), 'd_sm': (G.SHAPE1[kind], G.MASS[kind]),
               'd_tau': (G.SHAPE1[kind], 'msdpub_rand@50'), 'd_width': ('msdpub_rand@50', G.MSD[kind]),
               'd_shape_narrow': ('shape_narrow@50', G.MSD[kind])}
        gi, go = (edits.index('gin'), edits.index('gout')) if 'gin' in edits else (None, None)

        def stats_of(F):                                                  # F (b, R, 2E) -> dict of (b, nN) arrays
            r = {}
            for k, cols in colsets.items():
                C = G.cosines(F, ref, cols)
                for cn, (u, v) in con.items():
                    if k != 'none' and cn not in ('d_shape', 'd_groom', 'd_sm', 'd_tau'): continue
                    r[f'{cn}|{k}'] = C[:, :nN, ix[u]] - C[:, :nN, ix[v]]
            if gi is not None:
                den = np.where(F[:, :, gi] > 0, F[:, :, gi], np.nan); r['ratio_gout_gin'] = F[:, :, go] / den
            for e in ('decay', 'gout', 'gin', 'soft'):
                if e in edits: r[f'chg_{e}'] = F[:, :, edits.index(e)]
            return r
        F1 = G.fp_batch(np.ones((1, n)), A, Rn, E); P = stats_of(F1); boot = {}
        for b0 in range(0, B, CHUNK):
            Wm = rng.multinomial(n, np.full(n, 1.0 / n), size=CHUNK).astype(np.float64)
            for k, v in stats_of(G.fp_batch(Wm, A, Rn, E)).items(): boot.setdefault(k, []).append(v.astype(np.float32))
        boot = {k: np.concatenate(v) for k, v in boot.items()}
        T = dict(kind=kind, n=n, n_sig=int(ysig.sum()), classical={**{t: cinfo[t] for t in G.M50[kind]}, **xinfo}, nets={}, taggers={})
        for a, mn in enumerate(nets):
            r = {}
            for k in P:
                if k.startswith(('ratio', 'chg')): continue
                db = boot[k][:, a].astype(np.float64)
                r[k] = dict(delta=float(P[k][0, a]), ci=np.nanpercentile(db, [2.5, 97.5]).tolist(), p_boot=G.p_two(db), p_z=G.p_z(float(P[k][0, a]), float(np.std(db))))
            for k in [k for k in P if k.startswith(('ratio', 'chg'))]:
                db = boot[k][:, a].astype(np.float64)
                ent = dict(value=float(P[k][0, a]), ci=np.nanpercentile(db, [2.5, 97.5]).tolist())
                if k == 'ratio_gout_gin':
                    ent.update(p_vs_0=G.p_two(db[np.isfinite(db)]), p_vs_1=G.p_two(db[np.isfinite(db)] - 1.0))
                r[k] = ent
            T['nets'][mn] = r
        for t in tn:
            T['taggers'][t] = {k: float(P[k][0, ix[t]]) for k in P if k.startswith(('ratio', 'chg'))}
        out['tasks'][task] = T
        log(f'[{work.name} {ds}] {task}: ' + '; '.join(f'{mn} d_groom {T["nets"][mn]["d_groom|none"]["delta"]:+.3f} -> no_decay '
                                                     f'{T["nets"][mn]["d_groom|no_decay"]["delta"]:+.3f}, d_tau {T["nets"][mn]["d_tau|none"]["delta"]:+.3f}'
                                                     for mn in nets))
    return out


def part_extra(B):
    rng = np.random.default_rng(5898); R = dict(node='H58v', B=B, drops={k: list(v) for k, v in DROPS.items()}, samples={})
    for tag, work, dss in (('fresh', VWORK, ('jc', 'top')), ('h58', ORIG, ('hls', 'jc', 'top'))):
        R['samples'][tag] = {ds: r for ds in dss for r in [extra_ds(work, ds, rng, B)] if r is not None}
    tests = []
    for tag, S in R['samples'].items():
        for ds, D in S.items():
            for task, T in D['tasks'].items():
                for mn, r in T['nets'].items():
                    for k, v in r.items():
                        if 'p_boot' in v: tests.append((k, v['p_boot'], v))
                        if 'p_vs_0' in v: tests.append((k + '|vs0', v['p_vs_0'], v)); tests.append((k + '|vs1', v['p_vs_1'], v))
    fams = sorted({t[0] for t in tests})
    for f in fams:
        ix = [i for i, t in enumerate(tests) if t[0] == f]; hp = H.holm([tests[i][1] for i in ix])
        for i, h in zip(ix, hp):
            key = 'p_holm' if not f.endswith(('|vs0', '|vs1')) else ('p_holm_vs0' if f.endswith('|vs0') else 'p_holm_vs1')
            tests[i][2][key] = h
    R['holm_families'] = {f: sum(t[0] == f for t in tests) for f in fams}
    (PHYS / 'h58_verify.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h58_verify.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['edits', 'nets', 'extra'])
    ap.add_argument('--ds'); ap.add_argument('--group'); ap.add_argument('--B', type=int, default=B_EX)
    a = ap.parse_args()
    if a.part == 'edits':
        setup_fresh(); H.part_edits(a.ds, 0)
    elif a.part == 'nets':
        setup_fresh(); G.GROUPS[a.group]()
    else:
        part_extra(a.B)
    log('done')


if __name__ == '__main__':
    main()
