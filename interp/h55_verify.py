"""H55v (loop D, skeptic for H55): fresh-sample rerun of the classical-tagger fingerprint panel on hls4ml and JetClass, plus
controls H55 did not run.

Rerun (same code, interp.h55_classical_fingerprint, fresh seeds): hls4ml jedi_core.sample(2000, 5591); JetClass Hbb/Wqq/Zqq/
Tbqq/QCD 2,000 random rows each (default_rng 5592). Same nine edits, same classical taggers, same networks and permuted
copies, same stats (part_stats) written to $AI_RESULTS/physics/h55_verify_work/.

Extra controls (--part extra), per task, on the same jets:
  (a) degenerate-null check: permuted copies whose unedited log-odds are constant (std < 1e-6) or whose calls never change
      have an all-zero fingerprint and cosine 0 by construction; the null is recomputed without them; rank p =
      (1 + #{perm cos >= net cos}) / (n_perm + 1) (exact permutation-style p, no normal approximation).
  (b) paired tagger contrast: Delta_shape-mass = cos(net, best m_SD+shape tagger) - cos(net, plain mass window), 95 %
      bootstrap CI over jets (B = 500, standardisation recomputed per draw), p = P_boot(Delta <= 0); the direct test of
      "nearest is an m_SD + shape tagger, not the plain mass window".
  (c) matched working point: each network's call threshold set so that its unedited signal efficiency equals that of the
      classical tagger it is compared with (instead of 50 %); cosine recomputed (column scale fixed to the main analysis);
      nearest tagger under matched working points.
  (d) leave-one-edit-out: nearest tagger with each edit's (chg, flow) columns removed.
  (e) chg-only and flow-only fingerprints.
Run
  ./part_job.sh d-h55-verify 'python -m interp.h55_verify --part edits --ds jc && python -m interp.h55_verify --part edits --ds hls
      && python -m interp.h55_verify --part nets --ds jc && python -m interp.h55_verify --part nets --ds hls_small'
  ./job.sh d-h55-verify-jedi 1 4 'python -m interp.h55_verify --part nets --ds hls_jedi'
  AI_PY=conda ./job.sh d-h55-verify-stats 2 8 'python -m interp.h55_verify --part stats && python -m interp.h55_verify --part extra'
  -> $AI_RESULTS/physics/h55_verify.json (extra), h55_verify_work/ (tables, logits, fingerprint json, figures)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h55_classical_fingerprint as H  # noqa: E402

H.SEEDS.update(hls=5591, jc=5592)
WORK = H.RES / 'physics' / 'h55_verify_work'
H.OUT = WORK
SHAPE = {'W_vs_QCD': ['V_tau21', 'V_DDT', 'V_D2'], 'Z_vs_QCD': ['V_tau21', 'V_DDT', 'V_D2'], 't_vs_QCD': ['top_tau32'],
         'Top_vs_QCD': ['top_tau32']}
MASS = {'W_vs_QCD': 'mass_V', 'Z_vs_QCD': 'mass_V', 't_vs_QCD': 'mass_top', 'Top_vs_QCD': 'mass_top'}
NB = 500


def fpv(CH, FL, OK, w, cols):
    """(R, n, E) arrays -> (R, 2E') fingerprints over edit indices cols"""
    num_c = np.einsum('n,rne->re', w, CH[:, :, cols]); num_f = np.einsum('n,rne->re', w, FL[:, :, cols])
    den = np.maximum(np.einsum('n,rne->re', w, OK[:, :, cols]), 1e-12)
    return np.concatenate([num_c / den, num_f / den], 1)


def cos_to(F, s, a, b):
    keep = s > 1e-12; x, z = F[a, keep] / s[keep], F[b, keep] / s[keep]
    na, nb = np.linalg.norm(x), np.linalg.norm(z)
    return float(x @ z / (na * nb)) if na > 0 and nb > 0 else 0.0


def scale(F, ref):
    return np.sqrt((F[ref] ** 2).mean(0))


def part_extra():
    out = dict(seeds=dict(hls=H.SEEDS['hls'], jc=H.SEEDS['jc']), datasets={})
    rng = np.random.default_rng(5599)
    for ds in ('hls', 'jc'):
        Zo = np.load(WORK / f'h55_{ds}_obs.npz'); O = {}
        for k in Zo.files:
            c, q = k.split('|'); O.setdefault(c, {})[q] = Zo[k]
        Zt = np.load(WORK / f'h55_{ds}_tab.npz'); y = Zt['y'].astype(int)
        oks = {c: Zt[f'{c}|ok'] for c in H.CONDS if f'{c}|ok' in Zt.files}
        L = {}; acc = {}
        for g in ('jedi', 'small', 'nets'):
            q = WORK / f'h55_{ds}_{g}_logits.npz'
            if q.exists():
                Zl = np.load(q)
                for k in Zl.files:
                    mn, c = k.split('|'); L.setdefault(mn, {})[c] = Zl[k]
                acc.update(json.loads((WORK / f'h55_{ds}_{g}_acc.json').read_text()))
        edits = [e for e in H.DS_EDITS[ds] if e in oks]; E = len(edits)
        tasks = H.DS_TASKS[ds] or H.jc_tasks(); D = dict(accuracy={k: v for k, v in acc.items() if '_perm' not in k}, tasks={})
        D['perm_accuracy'] = {k: v['acc'] for k, v in acc.items() if '_perm' in k and isinstance(v, dict)}
        for task, (sig, bkg) in tasks.items():
            if task not in H.TASK_TAGGERS or not H.TASK_TAGGERS[task]: continue
            rows = np.flatnonzero(np.isin(y, [sig] + bkg)); ysig = y[rows] == sig; n = len(rows)
            okr = {c: oks[c][rows] for c in oks}; Ocl = {c: O[c] for c in ['base'] + edits}
            ccalls, cinfo = H.classical_calls(ds, task, sig, bkg, Ocl, y, rows)
            tg = list(ccalls); names, kinds, CH, FL, OK, S = [], [], [], [], [], {}
            for t in tg:
                ch, fl, ok = H.response_arrays(ccalls[t], okr, edits); names.append(t); kinds.append('classical'); CH.append(ch); FL.append(fl); OK.append(ok)
            degen = []
            for mn, lc in L.items():
                s = {c: H.logodds(lc[c][rows], sig, bkg) for c in ['base'] + edits}; S[mn] = s
                thr = float(np.median(s['base'][ysig])); calls = {c: v > thr for c, v in s.items()}
                ch, fl, ok = H.response_arrays(calls, okr, edits)
                kind = 'perm' if '_perm' in mn else 'net'
                if kind == 'perm' and (np.std(s['base']) < 1e-6 or ch.sum() == 0): degen.append(mn)
                names.append(mn); kinds.append(kind); CH.append(ch); FL.append(fl); OK.append(ok)
            CH, FL, OK = np.stack(CH), np.stack(FL), np.stack(OK); kinds = np.array(kinds)
            ref = np.flatnonzero(kinds != 'perm'); allc = list(range(E))
            cl = [i for i in range(len(names)) if kinds[i] == 'classical']; nets = [i for i in range(len(names)) if kinds[i] == 'net']
            perms = [i for i in range(len(names)) if kinds[i] == 'perm']; pnd = [i for i in perms if names[i] not in degen]
            F0 = fpv(CH, FL, OK, np.ones(n), allc); s0 = scale(F0, ref)
            Wb = rng.multinomial(n, np.full(n, 1.0 / n), size=NB).astype(np.float64)
            Fb = [fpv(CH, FL, OK, w, allc) for w in Wb]; sb = [scale(F, ref) for F in Fb]
            T = dict(n=n, n_sig=int(ysig.sum()), classical_eff={t: cinfo[t] for t in tg}, degenerate_perms=degen,
                     n_perm=len(perms), n_perm_nondegenerate=len(pnd), nets={})
            for i in nets:
                r = dict(cos={}, null_all={}, null_nondegen={})
                for t in cl:
                    c0 = cos_to(F0, s0, i, t); pc = np.array([cos_to(F0, s0, p, t) for p in perms]); pn = np.array([cos_to(F0, s0, p, t) for p in pnd])
                    cb = np.array([cos_to(Fb[b], sb[b], i, t) for b in range(NB)])
                    r['cos'][names[t]] = dict(cos=c0, ci=np.percentile(cb, [2.5, 97.5]).tolist())
                    r['null_all'][names[t]] = dict(mean=float(pc.mean()), rank_p=float((1 + (pc >= c0).sum()) / (len(pc) + 1)))
                    if len(pn) > 1:
                        z = (c0 - pn.mean()) / max(np.hypot(np.std(cb), np.std(pn, ddof=1)), 1e-12)
                        r['null_nondegen'][names[t]] = dict(mean=float(pn.mean()), sd=float(np.std(pn, ddof=1)), z=float(z),
                                                            rank_p=float((1 + (pn >= c0).sum()) / (len(pn) + 1)))
                r['nearest'] = max(r['cos'], key=lambda k: r['cos'][k]['cos'])
                # (b) paired contrast shape vs mass
                if task in SHAPE:
                    mi = names.index(MASS[task]); shp = [names.index(k) for k in SHAPE[task]]
                    bi = max(shp, key=lambda k: cos_to(F0, s0, i, k))
                    d0 = cos_to(F0, s0, i, bi) - cos_to(F0, s0, i, mi)
                    db = np.array([cos_to(Fb[b], sb[b], i, bi) - cos_to(Fb[b], sb[b], i, mi) for b in range(NB)])
                    r['shape_minus_mass'] = dict(shape=names[bi], delta=float(d0), ci=np.percentile(db, [2.5, 97.5]).tolist(),
                                                 p_boot=float((1 + (db <= 0).sum()) / (NB + 1)))
                # (c) matched working point
                s = S[names[i]]; mw = {}
                for t in cl:
                    eff = cinfo[names[t]]['eff_sig']; thr = float(np.quantile(s['base'][ysig], 1 - eff))
                    calls = {c: v > thr for c, v in s.items()}; ch, fl, ok = H.response_arrays(calls, okr, edits)
                    Fm = np.concatenate([F0[[t]], fpv(ch[None], fl[None], ok[None], np.ones(n), allc)], 0)
                    pm = []
                    for p in pnd:
                        sp = S[names[p]]; tp = float(np.quantile(sp['base'][ysig], 1 - eff)); cp = {c: v > tp for c, v in sp.items()}
                        a, b2, c2 = H.response_arrays(cp, okr, edits); Fp = np.concatenate([F0[[t]], fpv(a[None], b2[None], c2[None], np.ones(n), allc)], 0)
                        pm.append(cos_to(Fp, s0, 1, 0))
                    pm = np.array(pm); cm = cos_to(Fm, s0, 1, 0)
                    mw[names[t]] = dict(eff_sig=eff, cos=cm, perm_mean=float(pm.mean()) if len(pm) else None,
                                        rank_p=float((1 + (pm >= cm).sum()) / (len(pm) + 1)) if len(pm) else None,
                                        chg_by_edit={e: float(Fm[1, j]) for j, e in enumerate(edits)})
                r['matched_wp'] = mw; r['matched_wp_nearest'] = max(mw, key=lambda k: mw[k]['cos'])
                # (d) leave one edit out
                loo = {}
                for j, e in enumerate(edits):
                    cols = [k for k in allc if k != j]; Fl = fpv(CH, FL, OK, np.ones(n), cols); sl = scale(Fl, ref)
                    cc = {names[t]: cos_to(Fl, sl, i, t) for t in cl}; loo[e] = dict(nearest=max(cc, key=cc.get), cos=cc)
                r['leave_one_out'] = loo
                # (e) chg-only, flow-only
                for part, sl_ in (('chg_only', slice(0, E)), ('flow_only', slice(E, 2 * E))):
                    Fx = F0[:, sl_]; sx = scale(Fx, ref); cc = {names[t]: cos_to(Fx, sx, i, t) for t in cl}
                    r[part] = dict(nearest=max(cc, key=cc.get), cos=cc)
                r['fingerprint_chg'] = {e: float(F0[i, j]) for j, e in enumerate(edits)}
                T['nets'][names[i]] = r
            T['classical_chg'] = {names[t]: {e: float(F0[t, j]) for j, e in enumerate(edits)} for t in cl}
            T['perm_chg_median'] = {e: float(np.median(F0[perms, j])) for j, e in enumerate(edits)}
            T['perm_nondegen_chg_median'] = {e: float(np.median(F0[pnd, j])) for j, e in enumerate(edits)} if pnd else None
            D['tasks'][task] = T; H.log(f'[{ds}] {task} done; degenerate perms {len(degen)}/{len(perms)}')
        out['datasets'][ds] = D
    (H.RES / 'physics' / 'h55_verify.json').write_text(json.dumps(out, indent=1, default=float)); H.log('wrote h55_verify.json')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['edits', 'nets', 'stats', 'extra']); ap.add_argument('--ds')
    a = ap.parse_args(); WORK.mkdir(parents=True, exist_ok=True)
    if a.part == 'edits': H.part_edits(a.ds, 0)
    elif a.part == 'nets': dict(hls_jedi=H.nets_hls_jedi, hls_small=H.nets_hls_small, jc=H.nets_jc)[a.ds](0)
    elif a.part == 'stats': H.part_stats(0)
    else: part_extra()
    H.log('done')


if __name__ == '__main__':
    main()
