"""H3 verification (loop D).  Fresh samples; reruns the key H3 edits and adds two controls H3 did not have.

  (1) W-pair mass inside top.  H3's control shifts the non-W pair (b + harder W daughter) by the same Δm, but leaves the W-pair
      mass free: at fixed jet mass the W pair then moves the other way.  Here: the W-pair mass reached under H3's control is logged,
      and a second control shifts the non-W pair by the same Δm while holding the W pair at its original mass (and the jet mass).
  (2) H→bb displaced tracks.  H3 calls lifetime x0.35 'count at S>2 kept', but scaling d0 by 0.35 lowers d0/σ, so the count at
      S>2 / S>3 falls.  Here: a dose series (x0.35, x0.1, x0.03) and a count-preserving edit that lowers every track with
      S = |d0|/σ > 3.5 to S = 3.5 (d0, dz scaled by 3.5/S; the count at S>2 and S>3 is unchanged, the flight distance falls).
  Plus reruns: boost x0.9 vs rotation (JEDI W/Z, ParT_full Wqq), keep-2-displaced and made-prompt (ParT_full Hbb).

Samples.  JEDI: 2,000 jets per class drawn with seed 601 from the hls4ml test set, excluding the 20,000 jets of H3's seed-501
sample.  ParT: 3,000 random rows per class (default_rng(702)) of each class's SECOND test file (*_101.root; H3 used *_100).
  ./job.sh d-h3-verifyj 'python -m interp.h3_verify jedi'      -> $AI_RESULTS/physics/h3_verify_jedi.json
  ./part_job.sh d-h3-verify 'python -m interp.h3_verify part'  -> $AI_RESULTS/physics/h3_verify_part.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.h3_d3_fullstats as H
from interp.h3_d3_fullstats import boot_ci, mcnemar, signflip, holm, logsoft, mass_of, boost, rotate, top_split, move_set, drop_prong

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
H.ITERS = 30
ITERS2, TOL = 40, 2.0


def spread(pt, de, dp, inS, target):
    wp = pt * inS; c = (wp[..., None] * np.stack([de, dp], -1)).sum(1) / np.maximum(wp.sum(1), 1e-9)[:, None]
    f = (target / np.maximum(mass_of(pt, de, dp, inS), 1e-3))[:, None]
    return (np.where(inS, c[:, None, 0] + f * (de - c[:, None, 0]), de), np.where(inS, c[:, None, 1] + f * (dp - c[:, None, 1]), dp))


def move_two(pt, de, dp, inC, tC, inW, tW):
    """shift pair C to tC while holding pair W at tW and the jet mass fixed (alternating steps)"""
    m_top = mass_of(pt, de, dp); de, dp = de.copy(), dp.copy()
    for _ in range(ITERS2):
        de, dp = spread(pt, de, dp, inC, tC); de, dp = spread(pt, de, dp, inW, tW)
        s = (m_top / np.maximum(mass_of(pt, de, dp), 1e-3))[:, None]; de, dp = de * s, dp * s
    return de, dp


def pair_masses(pt, de, dp, TS):
    a = TS['a']; i, j, b, hw = TS['i'], TS['j'], TS['b'], TS['hardW']; lw = i + j - hw
    g = lambda u: (a == u[:, None]) & (pt > 0)
    return dict(W=mass_of(pt, de, dp, g(i) | g(j)), C=mass_of(pt, de, dp, g(b) | g(hw)), third=mass_of(pt, de, dp, g(b) | g(lw)),
                jet=mass_of(pt, de, dp))


def wshift_all(pt, de, dp, t, TS):
    a = TS['a']; inW = ((a == TS['i'][:, None]) | (a == TS['j'][:, None])) & (pt > 0)
    inC = ((a == TS['b'][:, None]) | (a == TS['hardW'][:, None])) & (pt > 0)
    mW0 = mass_of(pt, de, dp, inW); mC0 = mass_of(pt, de, dp, inC); tC = mC0 + (t - TS['mW']); ok = TS['ok'] & (tC > 5)
    deE, dpE, mWe, _ = move_set(pt, de, dp, inW, np.full(len(pt), t))
    deA, dpA, mCa, _ = move_set(pt, de, dp, inC, np.maximum(tC, 5.0))
    deB, dpB = move_two(pt, de, dp, inC, np.maximum(tC, 5.0), inW, mW0)
    PE, PA, PB = (pair_masses(pt, x, y, TS) for x, y in ((deE, dpE), (deA, dpA), (deB, dpB)))
    okE = np.abs(PE['W'] - t) < TOL; okA = np.abs(PA['C'] - tC) < TOL
    okB = (np.abs(PB['C'] - tC) < TOL) & (np.abs(PB['W'] - mW0) < TOL) & (np.abs(PB['jet'] - mass_of(pt, de, dp)) < 0.5)
    P0 = pair_masses(pt, de, dp, TS)
    med = lambda P, m: {k: float(np.median(v[m])) for k, v in P.items()}
    return dict(edit=(deE, dpE), ctrlA=(deA, dpA), ctrlB=(deB, dpB), ok=ok, okE=okE, okA=okA, okB=okB, P0=P0, PE=PE, PA=PA, PB=PB, med=med)


def paired(L0, L1, LC, y, key):
    """key-call change of edit (L1) and control (LC) vs unedited L0; edit − control with CI, McNemar, Δlogp sign-flip"""
    n = len(y); c0, c1 = L0.argmax(1), L1.argmax(1); I0, I1 = key(c0), key(c1)
    lp0, lp1 = logsoft(L0)[np.arange(n), y], logsoft(L1)[np.arange(n), y]
    r = dict(n=int(n), key_before=float(I0.mean()), key_after=float(I1.mean()), key_change=float(I1.mean() - I0.mean()),
             key_change_ci=boot_ci(I1.astype(float) - I0), mcnemar_vs_unedited=mcnemar(I0, I1), dlogp=float((lp1 - lp0).mean()),
             dlogp_ci=boot_ci(lp1 - lp0))
    if LC is not None:
        cC = LC.argmax(1); IC = key(cC); lpC = logsoft(LC)[np.arange(n), y]
        r.update(ctrl_key_after=float(IC.mean()), ctrl_key_change=float(IC.mean() - I0.mean()), ctrl_key_change_ci=boot_ci(IC.astype(float) - I0),
                 diff=float(I1.mean() - IC.mean()), diff_ci=boot_ci(I1.astype(float) - IC), mcnemar_vs_ctrl=mcnemar(IC, I1),
                 dlogp_diff=float((lp1 - lpC).mean()), dlogp_diff_ci=boot_ci(lp1 - lpC), dlogp_perm_p=signflip(lp1 - lpC))
    return r


def finish(rows):
    f = [k for k, r in rows.items() if 'mcnemar_vs_ctrl' in r]
    for k, a in zip(f, holm([rows[k]['mcnemar_vs_ctrl']['p'] for k in f])): rows[k]['p_holm_vs_ctrl'] = float(a)
    for k, a in zip(f, holm([rows[k]['dlogp_perm_p'] for k in f])): rows[k]['p_holm_dlogp_vs_ctrl'] = float(a)
    g = list(rows)
    for k, a in zip(g, holm([rows[k]['mcnemar_vs_unedited']['p'] for k in g])): rows[k]['p_holm_vs_unedited'] = float(a)
    for k, r in rows.items():
        s = f"{k} n={r['n']}: {100 * r['key_before']:.1f} -> {100 * r['key_after']:.1f} (Δ {100 * r['key_change']:+.1f} CI {100 * r['key_change_ci'][0]:+.1f}..{100 * r['key_change_ci'][1]:+.1f})"
        if 'diff' in r: s += f" | ctrl {100 * r['ctrl_key_after']:.1f}; edit-ctrl {100 * r['diff']:+.1f} CI {100 * r['diff_ci'][0]:+.1f}..{100 * r['diff_ci'][1]:+.1f}, Holm p {r['p_holm_vs_ctrl']:.2g}"
        log(s)
    return rows


def wblock(run, pt, de, dp, top_rows, L0, y, cls_idx, mk_input, label):
    """W-shift block for one model family. mk_input(rows_local_mask, de, dp) -> network input; L0 unedited logits of top jets"""
    TS = top_split(pt, de, dp); log(f'{label}: top jets with a W pair {int(TS["ok"].sum())}/{len(pt)}'); out = {}; rows = {}
    for t in (60.0, 100.0, 120.0):
        R = wshift_all(pt, de, dp, t, TS); m = R['ok'] & R['okE'] & R['okA'] & R['okB']
        LE = run(mk_input(m, *[v[m] for v in R['edit']])); LA = run(mk_input(m, *[v[m] for v in R['ctrlA']])); LB = run(mk_input(m, *[v[m] for v in R['ctrlB']]))
        key = lambda c: c == cls_idx
        out[f'{t:g}'] = dict(n=int(m.sum()), n_ok=int(R['ok'].sum()), masses=dict(unedited=R['med'](R['P0'], m), edit=R['med'](R['PE'], m),
                             ctrl_H3=R['med'](R['PA'], m), ctrl_Wfixed=R['med'](R['PB'], m)),
                             mW_abs_shift_ctrl_H3_median=float(np.median(np.abs(R['PA']['W'][m] - R['P0']['W'][m]))))
        log(f'{label} W->{t:g}: n {int(m.sum())}; pair masses {out[f"{t:g}"]["masses"]}')
        for k in (f'{label} W->{t:g} vs H3 control (W free)', f'{label} W->{t:g} vs W-fixed control'):
            for mod in LE:
                rows[f'{k} [{mod}]'] = paired(L0[mod][m], LE[mod], LB[mod] if 'W-fixed' in k else LA[mod], y[m], key)
        for mod in LE:
            rows[f'{label} H3 control (W free) vs W-fixed control, target {t:g} [{mod}]'] = paired(L0[mod][m], LA[mod], LB[mod], y[m], key)
    return out, rows


# ------------------------------------------------------------------------------------------------------------- JEDI
def run_jedi():
    from interp.jedi_core import Jedi
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); yall = np.asarray(d['y'])
    r501 = np.random.default_rng(501); old = np.concatenate([r501.choice(np.flatnonzero(yall == c), 4000, replace=False) for c in range(5)])
    used = np.zeros(len(yall), bool); used[old] = True; rng = np.random.default_rng(601)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero((yall == c) & ~used), 2000, replace=False) for c in range(5)]))
    x = np.asarray(d['x'][idx], np.float64); y = yall[idx].astype(int); CL = ['g', 'q', 'W', 'Z', 't']
    net = Jedi(64, '3-feature'); run1 = lambda X: net.run(np.asarray(X, np.float32))['logits']
    L0 = run1(x); acc = float((L0.argmax(1) == y).mean()); log(f'JEDI fresh sample {len(y)} jets (overlap with H3 sample: {int(used[idx].sum())}); accuracy {100 * acc:.2f} %')
    pt, de, dp = x[..., 0], x[..., 1], x[..., 2]; rows = {}
    for cl in ('W', 'Z'):
        r = np.flatnonzero(y == CL.index(cl)); key = lambda c, k=CL.index(cl): c == k
        rows[f'boost x0.9 vs rotation, {cl} jets [jedi]'] = paired(L0[r], run1(np.stack(boost(pt[r], de[r], dp[r], 0.9), -1)),
                                                                    run1(np.stack(rotate(pt[r], de[r], dp[r], 0.9), -1)), y[r], key)
    top = np.flatnonzero(y == 4)
    mk = lambda m, de_, dp_: np.stack([pt[top][m], de_, dp_], -1)
    W, wr = wblock(lambda X: {'jedi': run1(X)}, pt[top], de[top], dp[top], top, {'jedi': L0[top]}, y[top], 4, mk, 'JEDI')
    rows.update(wr)
    return dict(model='JEDI-linear 64p 3-feature', sample=dict(per_class=2000, seed=601, N=len(y), excludes='H3 seed-501 jets'),
                accuracy=acc, own_class=[float((L0.argmax(1)[y == k] == k).mean()) for k in range(5)], wshift=W, comparisons=finish(rows))


# ------------------------------------------------------------------------------------------------------------- ParT
def run_part():
    from interp.part_core import ParTNetwork, kin, build, sub, cat, read_root, files, PRE, CL
    from jetdistill.config import DATA
    CL = list(CL); per = 3000; rng = np.random.default_rng(702); Js = []
    for c in CL:
        f = next(x['file'] for x in files() if x['file'].split('/')[-1] == PRE[c] + '_101.root')
        p = next(q for q in (Path(f), DATA.parent / f, DATA / 'raw' / 'test' / Path(f).name) if q.exists())
        rows = np.sort(rng.choice(100000, per, replace=False)); parts = []
        for s in range(0, 100000, 25000):
            sel = rows[(rows >= s) & (rows < s + 25000)] - s
            if len(sel): parts.append(sub(read_root(p, start=s, stop=s + 25000), sel))
        Js.append(cat(parts)); log(f'read {c} from {p.name}: {per} random rows (first {rows[:3].tolist()})')
    J0 = cat(Js); del Js; y = J0['label'].astype(int); N = len(y)
    nets = {n: ParTNetwork(n) for n in ('full', 'kin')}
    run = lambda J, models=('full', 'kin'): {n: nets[n].run(J)['logits'].astype(np.float32) for n in models}
    pt, de, dp, ms = (np.asarray(a, np.float64) for a in kin(J0)); Jb = build(J0, pt, de, dp, ms)
    Lraw, L0 = run(J0), run(Jb)
    acc = {n: dict(raw=float((Lraw[n].argmax(1) == y).mean()), rebuilt=float((L0[n].argmax(1) == y).mean()),
                   own_class=[float((L0[n].argmax(1)[y == k] == k).mean()) for k in range(10)]) for n in L0}
    log(f'ParT accuracy {acc}'); rows = {}
    # boost x0.9 vs rotation, Wqq (full and kin)
    r = np.flatnonzero(y == CL.index('Wqq')); Jr = sub(J0, r); key = lambda c: c == CL.index('Wqq')
    LB_ = run(build(Jr, *boost(pt[r], de[r], dp[r], 0.9), ms[r])); LR_ = run(build(Jr, *rotate(pt[r], de[r], dp[r], 0.9), ms[r]))
    for n in ('full', 'kin'): rows[f'boost x0.9 vs rotation, Wqq [{n}]'] = paired(L0[n][r], LB_[n], LR_[n], y[r], key)
    # W shift in top
    top = np.flatnonzero(y == CL.index('Tbqq')); Jt = sub(J0, top)
    mk = lambda m, de_, dp_: build(sub(Jt, m), pt[top][m], de_, dp_, ms[top][m])
    W, wr = wblock(run, pt[top], de[top], dp[top], top, {n: L0[n][top] for n in L0}, y[top], CL.index('Tbqq'), mk, 'ParT')
    rows.update(wr)
    # H->bb displaced-track dose series (ParT_full)
    hb = np.flatnonzero(y == CL.index('Hbb')); Jh = sub(Jb, hb); key = lambda c: c == CL.index('Hbb')

    def S_(J):
        m = (J['mask'] > 0) & (J['charge'] != 0); return np.where(m, np.abs(J['d0val']) / np.maximum(J['d0err'], 1e-6), 0)

    def with_f(J, f):
        E = dict(J); E['d0val'] = (J['d0val'] * f).astype(np.float32); E['dzval'] = (J['dzval'] * f).astype(np.float32); return E
    Sg = S_(Jh); d2 = Sg > 2; rk = np.argsort(-Sg, 1).argsort(1)
    V = {'lifetime x0.35': with_f(Jh, np.where(d2, 0.35, 1.0)), 'lifetime x0.1': with_f(Jh, np.where(d2, 0.1, 1.0)),
         'lifetime x0.03': with_f(Jh, np.where(d2, 0.03, 1.0)),
         'S capped at 3.5 (count at S>2, S>3 kept)': with_f(Jh, np.where(Sg > 3.5, 3.5 / np.maximum(Sg, 1e-9), 1.0)),
         'keep 2 most displaced': with_f(Jh, np.where(d2 & (rk >= 2), 0.0, 1.0)),
         'displaced made prompt': with_f(Jh, np.where(d2, 0.0, 1.0))}
    LV = {k: run(v, ('full',))['full'] for k, v in V.items()}
    cnt = {k: dict(nS2=float((S_(v) > 2).sum(1).mean()), nS3=float((S_(v) > 3).sum(1).mean()),
                   sumS_over2=float(np.where(S_(v) > 2, S_(v), 0).sum(1).mean()), median_S_of_S2_tracks=float(np.median(S_(v)[S_(v) > 2])) if (S_(v) > 2).any() else 0.0)
           for k, v in [('unedited', Jh)] + list(V.items())}
    log(f'Hbb track counts {cnt}')
    for k in V: rows[f'Hbb {k} (vs unedited) [full]'] = paired(L0['full'][hb], LV[k], None, y[hb], key)
    rows['Hbb keep 2 most displaced vs S capped at 3.5 [full]'] = paired(L0['full'][hb], LV['keep 2 most displaced'], LV['S capped at 3.5 (count at S>2, S>3 kept)'], y[hb], key)
    rows['Hbb lifetime x0.1 vs S capped at 3.5 [full]'] = paired(L0['full'][hb], LV['lifetime x0.1'], LV['S capped at 3.5 (count at S>2, S>3 kept)'], y[hb], key)
    return dict(sample=dict(per_class=per, seed=702, N=N, files='*_101.root (H3 used *_100)'), accuracy=acc, wshift=W, hbb_track_counts=cnt,
                comparisons=finish(rows))


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    return o


if __name__ == '__main__':
    which = sys.argv[1]; RES.mkdir(parents=True, exist_ok=True)
    out = run_jedi() if which == 'jedi' else run_part()
    f = RES / f'h3_verify_{which}.json'; f.write_text(json.dumps(jsonable(out), indent=1)); log(f'wrote {f}; DONE')
