"""E24w verify, part 2: per-jet call disagreement between the prong-preserving (P) and prong-destroying (D) edits, with
two matched nulls.

Question: E24w reports that JEDI-linear changes the same share of calls under P and D. Part 1 (interp.e24w_verify) found that,
per jet, JEDI's calls under P and D disagree more often (7.7 %) than under two independent D draws (4.8 %). Is that excess
a response to prong integrity, or to where an intact prong sits (JEDI's first layer is not rotation invariant)?
Sample: interp.jedi_core.sample(per_class=2000, seed=3001), N = 10,000 (fresh). Selection and edits as interp.e24w_prong_jedi
(functions imported): exclusive-kt 2 subjets, z_s >= 0.1, >= 2 soft particles, tau21 < 0.5; within-jet a; every edit ends with
the a-holding whole-jet rotation beta.
  D, D2  two independent softer-subjet scrambles (theta ~ U(0, 2 pi)).
  P      softer subjet rotated rigidly by alpha with m_P = m_D (as E24w).
  P2     softer subjet rotated rigidly by -alpha (mirror; same angle to the hard prong, so the same m to O(R^2)), |m_P2/m_D - 1|
         <= 1 % required.
Analysis set: jets where P, P2, D, D2 all reach tol.
Read-outs per model (paired, same jets): % calls changed vs unedited; disagreement rates P!=D, P!=P2, D!=D2, P2!=D;
  E_loc = (P!=D) - (P!=P2): disagreement beyond moving an intact prong to its mirror position;
  E_draw = (P!=D) - (D!=D2): disagreement beyond redrawing the scramble.
  H0 (no integrity use): E_loc = 0. H1: E_loc > 0. 95 % jet bootstrap, paired bootstrap p.
Models: JEDI-linear official, + g held, + S58/S31 held; Deep Sets s0; tree Tht (held features + tau21, positive control) fit
  on sample(4000, 3002) minus the main jets.
  ./job.sh e24w-verify2 'python -m interp.e24w_verify_p2'  ->  $AI_RESULTS/jedi/e24w_verify_p2.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
EDITS = ['P', 'P2', 'D', 'D2']
SEED, POOL_SEED, RNG_SEED, PER = 3001, 3002, 3003, 2000


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def main():
    import interp.e24_azimuth_prong as E24
    import interp.e24w_prong_jedi as W
    from interp.jedi_core import sample, Jedi
    from jetdistill.config import RESULTS
    from sklearn.ensemble import HistGradientBoostingClassifier
    ci, mcnemar, clean, boot_p = E24.ci, E24.mcnemar, E24.clean, E24.boot_p
    HP = W.HP
    R = dict(node='E24w-verify-p2', sample=f'interp.jedi_core.sample(per_class={PER}, seed={SEED})',
             pool=f'interp.jedi_core.sample(per_class=4000, seed={POOL_SEED}) minus main jets', models={})
    save = lambda: (OUT / 'e24w_verify_p2.json').write_text(json.dumps(clean(R), indent=1))
    rng = np.random.default_rng(RNG_SEED)
    x, y, Q = sample(PER, SEED); N = len(x); R['N'] = N
    j = Jedi(64); pf = W.PF(j)
    base = j.run(x, keep=('P', 'S', 'logits'))
    acc = float((base['logits'].argmax(1) == y).mean()); R['jedi_acc_full_sample'] = acc
    log(f'N {N}; JEDI acc {acc:.4f}')
    t21, _ = E24.substructure(x, with_d2=False)
    lab, two = W.kt2(x); soft = lab == 1; real = x[..., 0] > 0
    ptp = x[..., 0].astype(np.float64); zs = (ptp * soft).sum(1) / ptp.sum(1); ns = soft.sum(1)
    sel = two & (zs >= 0.1) & (ns >= 2) & (t21 < 0.5); idx = np.flatnonzero(sel)
    R['selection'] = dict(n_selected=int(sel.sum()), per_class={CL[c]: int((y[sel] == c).sum()) for c in range(5)})
    log('selection', json.dumps(R['selection']))
    xs, ys = x[idx], y[idx]; J = len(xs); rs = real[idx]; ss = soft[idx]; b0 = {k: v[idx] for k, v in base.items()}
    # within-jet a (as E24w v2)
    nw = min(1500, J); wj = rng.choice(J, nw, replace=False); dP, dS = [], []
    for k in range(6):
        xe = W.rot(xs[wj], rs[wj], rng.uniform(-np.pi, np.pi, nw)) if k < 4 else W.scramble(xs[wj], ss[wj], rng)
        rr = j.run(xe.astype(np.float32), keep=('P', 'S'))
        dP.append(rr['P'][:, HP] - b0['P'][wj][:, HP]); dS.append(rr['S'][:, [58, 31]] - b0['S'][wj][:, [58, 31]])
    dP = np.concatenate(dP); dS = np.concatenate(dS); jid = np.tile(np.arange(nw), 6); trw = jid < nw // 2
    sd = {u: base['S'][:, u].std() for u in (58, 31)}; Wd = {}
    for q, u in enumerate((58, 31)):
        Wd[u] = np.linalg.lstsq(dP[trw], dS[trw, q], rcond=None)[0]
    zw = np.stack([dP @ Wd[58] / sd[58], dP @ Wd[31] / sd[31]], 1)
    ev, V = np.linalg.eigh(zw.T @ zw / len(zw)); v1 = V[:, 1]
    if v1[0] < 0: v1 = -v1
    wa = Wd[58] / sd[58] * v1[0] + Wd[31] / sd[31] * v1[1]
    afun = lambda P: P[:, HP] @ wa; sda = float(afun(base['P']).std()); tol = 0.01 * sda
    a0 = afun(b0['P']); fa = lambda xe: afun(pf(xe))
    agrid = lambda xe, mk, angs: np.concatenate([fa(W.rot_grid(xe[c:c + 20], mk[c:c + 20], angs)).reshape(-1, len(angs))
                                                 for c in range(0, len(xe), 20)])
    fm = lambda xe: W.moments(xe)['m']
    out = {e: np.array(xs, np.float64) for e in EDITS}; okE = {e: np.zeros(J, bool) for e in EDITS}
    for e in ('D', 'D2'):
        for k in range(12):
            todo = np.flatnonzero(~okE[e])
            if len(todo) == 0: break
            xe = W.scramble(xs[todo], ss[todo], rng)
            bt, ok = W.search(xe, rs[todo], a0[todo], fa, agrid, rng)
            xe = W.rot(xe, rs[todo], bt); ok = ok & (np.abs(fa(xe) - a0[todo]) <= tol)
            out[e][todo[ok]] = xe[ok]; okE[e][todo[ok]] = True
        log(e, 'ok', int(okE[e].sum()), '/', J)
    mD = np.where(okE['D'], fm(out['D']), np.nan); alpha = np.full(J, np.nan)
    for t in range(3):
        todo = np.flatnonzero(okE['D'] & okE['D2'] & ~okE['P'])
        if len(todo) == 0: break
        al, ok1 = W.search(xs[todo], ss[todo], mD[todo], fm, lambda xe, mk, angs: W.mass_grid(xe, mk, angs), rng,
                           exclude=np.pi / 18, iters=24, t=t)
        xe = W.rot(xs[todo], ss[todo], al); ok1 &= np.abs(fm(xe) / mD[todo] - 1) <= 0.01
        bt, ok2 = W.search(xe, rs[todo], a0[todo], fa, agrid, rng)
        xe = W.rot(xe, rs[todo], bt); ok = ok1 & ok2 & (np.abs(fa(xe) - a0[todo]) <= tol)
        out['P'][todo[ok]] = xe[ok]; okE['P'][todo[ok]] = True; alpha[todo[ok]] = al[ok]
    log('P ok', int(okE['P'].sum()))
    # P2: mirror angle -alpha, then beta
    todo = np.flatnonzero(okE['P'])
    for k in range(3):
        xe = W.rot(xs[todo], ss[todo], -alpha[todo]); ok1 = np.abs(fm(xe) / mD[todo] - 1) <= 0.01
        bt, ok2 = W.search(xe, rs[todo], a0[todo], fa, agrid, rng, t=k)
        xe = W.rot(xe, rs[todo], bt); ok = ok1 & ok2 & (np.abs(fa(xe) - a0[todo]) <= tol)
        out['P2'][todo[ok]] = xe[ok]; okE['P2'][todo[ok]] = True
        todo = todo[~ok]
        if len(todo) == 0: break
    log('P2 ok', int(okE['P2'].sum()))
    K = np.flatnonzero(okE['P'] & okE['P2'] & okE['D'] & okE['D2']); n = len(K)
    R['acceptance'] = dict(J=J, **{e: int(okE[e].sum()) for e in EDITS}, analysis=n,
                           per_class={CL[c]: int((ys[K] == c).sum()) for c in range(5)},
                           P_alpha_deg_median_abs=float(np.median(np.abs(np.degrees(alpha[K])))))
    log('analysis', n, json.dumps(R['acceptance']))
    X0 = xs[K].astype(np.float32); Y = ys[K]; XE = {e: out[e][K].astype(np.float32) for e in EDITS}
    SUB = {'none': E24.substructure(X0, with_d2=False)[0]}; chk = {}
    mo0 = W.moments(X0); r0, th0, re0 = W.polar(X0)
    for e in EDITS:
        SUB[e] = E24.substructure(XE[e], with_d2=False)[0]; mo = W.moments(XE[e]); r1, _, _ = W.polar(XE[e])
        chk[e] = dict(max_rel_dM0=float(np.max(np.abs(mo['M0'] / mo0['M0'] - 1))), max_rel_dM2=float(np.max(np.abs(mo['M2'] / mo0['M2'] - 1))),
                      max_abs_dr=float(np.max(np.abs(r1 - r0)[re0])), padding_changed=int((XE[e][~re0] != X0[~re0]).sum()),
                      mean_rel_dm=ci(mo['m'] / mo0['m'] - 1), dtau21=ci(SUB[e] - SUB['none']),
                      med_abs_rel_m_vs_D=float(np.median(np.abs(mo['m'] / W.moments(XE['D'])['m'] - 1))))
    R['edit_checks'] = chk
    log('checks', json.dumps(clean({e: dict(dm=chk[e]['mean_rel_dm']['mean'], dt=chk[e]['dtau21']['mean'],
                                            mD=chk[e]['med_abs_rel_m_vs_D']) for e in EDITS})))
    save()
    lgt = {}
    res = {'none': j.run(X0, keep=('g', 'S', 'logits'))}
    for e in EDITS: res[e] = j.run(XE[e], keep=('logits',))
    lgt['jedi'] = {k: v['logits'] for k, v in res.items()}
    g0, S0 = res['none']['g'], res['none']['S']; mS = np.zeros(64); mS[[58, 31]] = 1
    for arm, ov in (('g_held', {'g': (np.ones(64), g0)}), ('S58_S31_held', {'S': (mS, S0)})):
        lgt['jedi|' + arm] = {'none': lgt['jedi']['none'], **{e: j.run_jetwise(XE[e], ov)['logits'] for e in EDITS}}
    run, _, _, _ = E24.get_model('dsets_s0')
    lgt['dsets_s0'] = {'none': run(X0)['logits'], **{e: run(XE[e])['logits'] for e in EDITS}}
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y_all = np.asarray(d['y'])
    im = E24.sample_idx(y_all, PER, SEED); assert np.array_equal(y_all[im], y)
    xp, yp, _ = sample(4000, POOL_SEED); ip = E24.sample_idx(y_all, 4000, POOL_SEED); kp = ~np.isin(ip, im)
    xp, yp = xp[kp], yp[kp]; tp = E24.substructure(xp, with_d2=False)[0]; R['pool_n'] = int(len(xp))

    def feats(xx, t):
        mo = W.moments(xx); return np.stack([np.log(mo['M0']), np.log(mo['M2'] / mo['M0']), mo['Mr'] / mo['M0'], mo['N'], t], 1)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(feats(xp, tp), yp)
    R['tht_full_sample_acc'] = float((clf.predict(feats(x, t21)) == y).mean())
    lgt['Tht'] = {k: np.log(np.maximum(clf.predict_proba(feats(X0 if k == 'none' else XE[k], SUB[k])), 1e-9)) for k in ['none'] + EDITS}
    for name, Lg in lgt.items():
        C = {k: v.argmax(1) for k, v in Lg.items()}; c0 = C['none']; ch = {e: C[e] != c0 for e in EDITS}
        dis = lambda a_, b_: (C[a_] != C[b_]).astype(float)
        M = dict(calls_changed={e: ci(ch[e]) for e in EDITS},
                 disagree={f'{a_}!={b_}': ci(dis(a_, b_)) for a_, b_ in (('P', 'D'), ('P', 'P2'), ('D', 'D2'), ('P2', 'D'), ('P2', 'D2'))})
        for a_, b_ in (('D', 'P'), ('P2', 'P'), ('D2', 'D')):
            mc = mcnemar(ch[a_], ch[b_]); mc['p_one_sided'] = W.p_upper(mc['n10'], mc['n01'])
            M[f'{a_}_minus_{b_}'] = dict(calls=ci(ch[a_].astype(float) - ch[b_]), mcnemar=mc)
        eloc = dis('P', 'D') - dis('P', 'P2'); edraw = dis('P', 'D') - dis('D', 'D2')
        eloc2 = (dis('P', 'D') + dis('P2', 'D2')) / 2 - (dis('P', 'P2') + dis('D', 'D2')) / 2
        M['E_loc'] = dict(ci(eloc), p=boot_p(eloc)); M['E_draw'] = dict(ci(edraw), p=boot_p(edraw))
        M['E_sym'] = dict(ci(eloc2), p=boot_p(eloc2))
        R['models'][name] = M
        log(f'{name:18s} changed', {e: round(M['calls_changed'][e]['mean'] * 100, 2) for e in EDITS},
            'disagree', {k: round(v['mean'] * 100, 2) for k, v in M['disagree'].items()},
            'E_loc', round(M['E_loc']['mean'] * 100, 2), [round(v * 100, 2) for v in M['E_loc']['ci']], f"p {M['E_loc']['p']:.3g}",
            'E_sym', round(M['E_sym']['mean'] * 100, 2), [round(v * 100, 2) for v in M['E_sym']['ci']],
            'D-P', round(M['D_minus_P']['calls']['mean'] * 100, 2), [round(v * 100, 2) for v in M['D_minus_P']['calls']['ci']])
    save()
    np.savez_compressed(OUT / 'e24w_verify_p2_calls.npz', idx=idx[K], y=Y, tau21=np.stack([SUB[k] for k in ['none'] + EDITS]),
                        **{f'{nm_}|{e}': np.asarray(L[e], np.float32) for nm_, L in lgt.items() for e in ['none'] + EDITS})
    log('wrote', OUT / 'e24w_verify_p2.json')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); main()
