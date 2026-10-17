"""B5 verification on a fresh test sample (no refit): the stored B5 programs (b5_programs.json: axis20, axis50, inv50, all50)
and mars30 (E11) against the network under the B5 edits, plus three controls B5 did not run.

Sample: sample(per_class 2000, seed 23) minus every jet of sample(6000, 0) (fit / dev), sample(2000, 11) (B5 test) and
sample(2000, 91) (B5 confirmation).
Edits (real particles only; padding untouched; physics recomputed from the edited particles):
  trans      Δη += 0.06 (as B5)                 trans_neg  Δη −= 0.06 (sign check)
  trans_rd   rigid 0.06 rad translation in a per-jet uniformly random direction (rng 201): matched rigid control for trans
  trans_c    each particle 0.06 rad in its own random direction (as B5, rng 101)
  rot        45° about the axis
  wta        re-axis on the WTA axis            wta_c      same distance, random direction (rng 104 as B5)
  wta_m      re-axis in a random direction with the length chosen so that M2 = Σ pT r² about the new axis changes by the same
             amount as under wta (M2-matched control; rng 202; jets with no real solution in 50 draws keep the wta_c axis and
             are flagged)
  hardest, hardest_c as B5
Metrics as B5 (% calls changed, W→Z, paired surrogate − network difference with bootstrap CI, McNemar, "reproduces"), plus
network-only paired McNemar tests between edits and the ΔM2 diagnostics for wta / wta_c / wta_m.
    python -m interp.b5_verify [--per-class 2000 --seed 23]   -> AI_RESULTS/jedi/b5_verify.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import e11_mars_all as E
from interp.jedi_core import sample, Jedi
from interp.b5_compact_jedi import extras, to_cols, prog_logits, mcnemar, holm

OUT = E.OUT; T0 = time.time(); W_, Z_ = 2, 3; B = 1000
PROGS = ('axis20', 'axis50', 'inv50', 'all50')


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def moments(x):
    x = np.asarray(x, np.float64); pt, e, p = x[..., 0], x[..., 1], x[..., 2]
    return pt.sum(1), np.stack([(pt * e).sum(1), (pt * p).sum(1)], 1), (pt * (e ** 2 + p ** 2)).sum(1)


def main(a):
    from jetdistill.observables import compute
    from interp.b4_baselines_jedi import edit_shift, rotate, randdisp
    from interp.e17_axis_jedi import axis_set, reaxis, wta_selftest
    from interp.e11_verify import Last, chain
    i0, i1, i2, i3 = E.indices(6000, 0), E.indices(2000, 11), E.indices(2000, 91), E.indices(a.per_class, a.seed)
    keep = ~np.isin(i3, np.concatenate([i0, i1, i2]))
    x, y, _ = sample(a.per_class, a.seed); x, y = x[keep], y[keep]; J = len(x); real = x[..., 0] > 0
    log(f'fresh sample seed {a.seed}: {J} jets ({int((~keep).sum())} removed as overlapping seeds 0 / 11 / 91)')
    net = Jedi(64); last = Last(net)
    st = wta_selftest(); assert st['ok'], st
    A, ainfo = axis_set(x); log('axes', ainfo)
    M0, M1, M2 = moments(x)
    sets = dict(base=x, trans=edit_shift(x, 0.06), trans_neg=edit_shift(x, -0.06), trans_c=randdisp(x, 0.06 * real, 101), rot=rotate(x, 45.0))
    th = np.random.default_rng(201).uniform(0, 2 * np.pi, J); e = x.copy()
    e[..., 1] += (0.06 * np.cos(th)[:, None] * real).astype(np.float32); e[..., 2] += (0.06 * np.sin(th)[:, None] * real).astype(np.float32); sets['trans_rd'] = e
    rng = np.random.default_rng(104); diag = {}
    for nm in ('wta', 'hardest'):
        ae, ap = A[nm]; d = np.hypot(ae, ap); t = rng.uniform(0, 2 * np.pi, J)
        sets[nm] = reaxis(x, ae, ap); sets[nm + '_c'] = (reaxis(x, d * np.cos(t), d * np.sin(t)), (d * np.cos(t), d * np.sin(t)))
        if nm == 'wta': wc_axis = (d * np.cos(t), d * np.sin(t))
    sets['wta_c'], sets['hardest_c'] = sets['wta_c'][0], sets['hardest_c'][0]
    # M2-matched random-direction re-axis: M2(a) = M2 - 2 a.M1 + M0 |a|^2 ;  want M2(a) - M2 = dM2_wta
    aw = np.stack(A['wta'], 1); dM2w = -2 * (aw * M1).sum(1) + M0 * (aw ** 2).sum(1)
    r2 = np.random.default_rng(202); am = np.stack(wc_axis, 1).copy(); ok = np.zeros(J, bool)
    for _ in range(50):
        todo = np.flatnonzero(~ok)
        if not len(todo): break
        t = r2.uniform(0, 2 * np.pi, len(todo)); u = np.stack([np.cos(t), np.sin(t)], 1); b = (u * M1[todo]).sum(1)
        disc = b ** 2 + M0[todo] * dM2w[todo]; s = np.where(disc >= 0, (b + np.sqrt(np.maximum(disc, 0))) / M0[todo], -1)
        good = (disc >= 0) & (s > 0) | (np.abs(dM2w[todo]) < 1e-12)
        s = np.where(np.abs(dM2w[todo]) < 1e-12, 0, s); am[todo[good]] = u[good] * s[good, None]; ok[todo[good]] = True
    sets['wta_m'] = reaxis(x, am[:, 0], am[:, 1])
    dc = np.stack(wc_axis, 1); dM2c = -2 * (dc * M1).sum(1) + M0 * (dc ** 2).sum(1); dM2m = -2 * (am * M1).sum(1) + M0 * (am ** 2).sum(1)
    sq = np.sqrt(M0 * M2)
    for k, dd in (('wta', dM2w), ('wta_c', dM2c), ('wta_m', dM2m)):
        rel = np.sqrt(M0 * np.maximum(M2 + dd, 0)) / sq - 1
        diag[k] = dict(dM2_median=float(np.median(dd)), dsqrtM0M2_rel_median=float(np.median(rel)), dsqrtM0M2_rel_q10_90=np.percentile(rel, [10, 90]).tolist())
    diag['wta_m_matched_fraction'] = float(ok.mean()); diag['wta_m_disp_median'] = float(np.median(np.hypot(*am.T)))
    diag['wta_disp_median'] = float(np.median(np.hypot(*A['wta']))); diag['centroid_offset_median'] = float(np.median(np.hypot(*(M1 / M0[:, None]).T)))
    diag['frac_abs_dM2_wta_lt_wta_c'] = float(np.mean(np.abs(dM2w) < np.abs(dM2c)))
    diag['cos_wta_dir_vs_M1_median'] = float(np.median((aw * M1).sum(1) / np.maximum(np.hypot(*aw.T) * np.hypot(*M1.T), 1e-12)))
    log('M2 diagnostics', diag)
    progs = json.loads((OUT / 'b5_programs.json').read_text())['programs']
    rd = lambda n: json.loads((OUT / f'e11_{n}.json').read_text())
    fb = {'P': rd('fit_a_P'), **{s: rd(f'fit_b_{s}') for s in E.STG[1:]}}
    calls = {m: {} for m in ('network', 'mars30') + PROGS}; phys_chk = None
    for nm, xx in sets.items():
        t = time.time(); q = compute(np.asarray(xx, np.float64), 64)
        ph = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in q.items()}; ph.update(E.handbuilt(xx, False)); ph.update(extras(xx)); del q
        keys = list(ph); Q = to_cols(ph, keys); del ph
        if nm == 'base': phys_chk = len(keys)
        calls['network'][nm] = net.run(xx)['logits'].argmax(1)
        calls['mars30'][nm] = last.jax_logits(chain(fb, Q)['z']).argmax(1)
        for p in PROGS: calls[p][nm] = prog_logits(progs[p]['parts'], Q).argmax(1)
        del Q; log(f'set {nm}: physics ({phys_chk} keys), network, programs ({time.time() - t:.0f} s)')
    bi = np.random.default_rng(7).integers(0, J, (B, J)).astype(np.int32)
    ci = lambda v: [float(np.mean(v)), *map(float, np.percentile(np.asarray(v, np.float32)[bi].mean(1), [2.5, 97.5]))]
    Wm = y == W_; bw = np.random.default_rng(8).integers(0, Wm.sum(), (B, int(Wm.sum())))
    nb = calls['network']['base']
    res = dict(procedure=__doc__, seed=a.seed, N=J, N_W=int(Wm.sum()), axes=ainfo, M2_diagnostics=diag,
               network=dict(acc=ci(nb == y), edits={}), models={}, network_pairs={})
    edits = [k for k in sets if k != 'base']
    for e in edits:
        f = calls['network'][e] != nb
        res['network']['edits'][e] = dict(flip=ci(f), WtoZ=float((calls['network'][e][Wm] == Z_).mean()))
        log(f'network {e}: flip {res["network"]["edits"][e]["flip"]}, W→Z {res["network"]["edits"][e]["WtoZ"]:.3f}')
    for e1, e2 in (('trans', 'trans_rd'), ('trans', 'trans_neg'), ('trans', 'trans_c'), ('wta', 'wta_c'), ('wta', 'wta_m'), ('wta_m', 'wta_c'), ('hardest', 'hardest_c')):
        f1, f2 = calls['network'][e1] != nb, calls['network'][e2] != nb
        d = (f1.astype(np.float32) - f2)[bi].mean(1)
        res['network_pairs'][f'{e1} - {e2}'] = dict(diff=[float(f1.mean() - f2.mean()), *map(float, np.percentile(d, [2.5, 97.5]))], mcnemar=mcnemar(f1, f2))
        log(f'network pair {e1} - {e2}: {res["network_pairs"][f"{e1} - {e2}"]}')
    pv, cells = [], []
    for m in ('mars30',) + PROGS:
        C = calls[m]; r = dict(same_call=ci(C['base'] == nb), edits={})
        for e in edits:
            nf, sf = calls['network'][e] != nb, C[e] != C['base']
            d = (sf.astype(np.float32) - nf)[bi].mean(1); fl = ci(sf); ne = res['network']['edits'][e]['flip']
            wz_s, wz_n = (C[e][Wm] == Z_).astype(np.float32), (calls['network'][e][Wm] == Z_).astype(np.float32); dd = (wz_s - wz_n)[bw].mean(1)
            ag = (nf == sf) & (~nf | (C[e] == calls['network'][e]))
            ent = dict(flip=fl, flip_diff=[float(sf.mean() - nf.mean()), *map(float, np.percentile(d, [2.5, 97.5]))], mcnemar=mcnemar(sf, nf),
                       WtoZ=float(wz_s.mean()), WtoZ_diff=[float(wz_s.mean() - wz_n.mean()), *map(float, np.percentile(dd, [2.5, 97.5]))],
                       agree_minus_never_flip=float(ag.mean() - (1 - nf.mean())))
            ent['reproduces'] = bool(fl[1] <= ne[2] and ne[1] <= fl[2] and ent['flip_diff'][1] <= 0 <= ent['flip_diff'][2])
            r['edits'][e] = ent; pv.append(ent['mcnemar']['p']); cells.append((m, e))
        res['models'][m] = r
        log(f'{m:8s} same call {r["same_call"][0] * 100:.2f} | ' + ' '.join(f'{e} {v["flip"][0] * 100:.1f}{"*" if v["reproduces"] else ""}/WZ{v["WtoZ"] * 100:.0f}' for e, v in r['edits'].items()))
    for (m, e), q in zip(cells, holm(pv)): res['models'][m]['edits'][e]['mcnemar']['p_holm'] = float(q)
    res['n_holm'] = len(pv)
    p = OUT / 'b5_verify.json'; p.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float)); log('wrote', p)
    np.savez_compressed(OUT / 'b5_verify_calls.npz', y=y, **{f'{m}|{s}': np.asarray(v, np.int8) for m, d in calls.items() for s, v in d.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--per-class', type=int, default=2000); ap.add_argument('--seed', type=int, default=23)
    main(ap.parse_args()); log('done')
