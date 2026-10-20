"""E24w verify: rerun of the E24w prong test on a fresh sample, plus controls the E24w run did not have.

Sample: interp.jedi_core.sample(per_class=3000, seed=2901), N = 15,000 (no overlap requirement with E24w's seed 2801).
Same selection (exclusive-kt to 2 subjets, z_s >= 0.1, >= 2 soft particles, tau21 < 0.5), same within-jet definition of a,
same edits as interp.e24w_prong_jedi (functions imported from it): R (rigid rotation, |beta| >= 30 deg), D (softer-subjet
particles theta ~ U(0, 2 pi)), P (softer subjet rotated rigidly by alpha with m_P = m_D), each ending with the a-holding
whole-jet rotation beta. D' is not rerun.
Added controls:
  D2    an independent second D draw on the same jets (same procedure, fresh angles). D2 - D is the draw-to-draw null for a
        D - P contrast (same prong destruction, different draw).
  smooth classifiers on the held features (sklearn MLPClassifier 64-64, standardised inputs, early stopping), fit on
        sample(4000, 2902) minus the main jets: Mhm (Th + log m), Mhmt (Th + log m + tau21), Mht (Th + tau21). Trees Thm, Thmt,
        Tht as in E24w. These calibrate a logit-level read-out:
  two-prong score s = logsumexp(logit_W, logit_Z) - logsumexp(logit_q, logit_g) (log-probabilities for trees / MLPs).
        Signed D - P of s on the same jets, divided by sd(s) over the unedited analysis jets. A tau21 use (W/Z less likely
        when the prong is scattered) predicts D - P < 0; no prong use predicts 0. Reported for every model with 95 % jet
        bootstrap and paired bootstrap p, for all analysis jets and for true W/Z jets.
Networks: official JEDI-linear (+ g held, + S58/S31 held), Deep Sets s0, PFN s0.
  ./job.sh e24w-verify 'python -m interp.e24w_verify'  ->  $AI_RESULTS/jedi/e24w_verify.json, e24w_verify_calls.npz
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
EDITS = ['R', 'P', 'D', 'D2']
SEED, POOL_SEED, RNG_SEED = 2901, 2902, 2903


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def score(L):
    """two-prong score from logits or log-probabilities (J, 5): log P(W or Z) - log P(q or g)"""
    L = np.asarray(L, np.float64); lse = lambda A: np.log(np.exp(A - A.max(1, keepdims=True)).sum(1)) + A.max(1)
    return lse(L[:, [2, 3]]) - lse(L[:, [0, 1]])


def main():
    import interp.e24_azimuth_prong as E24
    import interp.e24w_prong_jedi as W
    from interp.jedi_core import sample, Jedi
    from jetdistill.config import RESULTS
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    ci, mcnemar, clean, boot_p = E24.ci, E24.mcnemar, E24.clean, E24.boot_p
    HP = W.HP
    R = dict(node='E24w-verify', sample=f'interp.jedi_core.sample(per_class=3000, seed={SEED})',
             pool=f'interp.jedi_core.sample(per_class=4000, seed={POOL_SEED}) minus main jets', models={})
    save = lambda: (OUT / 'e24w_verify.json').write_text(json.dumps(clean(R), indent=1))
    rng = np.random.default_rng(RNG_SEED)
    x, y, Q = sample(3000, SEED); N = len(x); R['N'] = N
    j = Jedi(64); pf = W.PF(j)
    base = j.run(x, keep=('P', 'g', 'S', 'logits'))
    acc = float((base['logits'].argmax(1) == y).mean()); R['jedi_acc_full_sample'] = acc
    R['pf_exact_max_abs_diff'] = float(np.max(np.abs(pf(x[:2000]) - base['P'][:2000])))
    log(f'N {N}; JEDI acc {acc:.4f}; replica {R["pf_exact_max_abs_diff"]:.1e}')
    # selection
    t21, _ = E24.substructure(x, with_d2=False)
    lab, two = W.kt2(x); soft = lab == 1; real = x[..., 0] > 0
    ptp = x[..., 0].astype(np.float64); zs = (ptp * soft).sum(1) / ptp.sum(1); ns = soft.sum(1)
    sel = two & (zs >= 0.1) & (ns >= 2) & (t21 < 0.5); idx = np.flatnonzero(sel)
    R['selection'] = dict(n_selected=int(sel.sum()), per_class={CL[c]: int((y[sel] == c).sum()) for c in range(5)},
                          soft_label_on_padding=int((soft & ~real).sum()))
    log('selection', json.dumps(R['selection']))
    xs, ys = x[idx], y[idx]; J = len(xs); rs = real[idx]; ss = soft[idx]; b0 = {k: v[idx] for k, v in base.items()}
    # within-jet a (as E24w v2)
    nw = min(1500, J); wj = rng.choice(J, nw, replace=False); dP, dS = [], []
    for k in range(6):
        xe = W.rot(xs[wj], rs[wj], rng.uniform(-np.pi, np.pi, nw)) if k < 4 else W.scramble(xs[wj], ss[wj], rng)
        rr = j.run(xe.astype(np.float32), keep=('P', 'S'))
        dP.append(rr['P'][:, HP] - b0['P'][wj][:, HP]); dS.append(rr['S'][:, [58, 31]] - b0['S'][wj][:, [58, 31]])
    dP = np.concatenate(dP); dS = np.concatenate(dS); jid = np.tile(np.arange(nw), 6); trw = jid < nw // 2; tew = ~trw
    sd = {u: base['S'][:, u].std() for u in (58, 31)}; Wd = {}; nm = {}
    for q, u in enumerate((58, 31)):
        w = np.linalg.lstsq(dP[trw], dS[trw, q], rcond=None)[0]; Wd[u] = w; pq = dP @ w
        nm[f'S{u}_within_r2_test'] = float(1 - ((dS[tew, q] - pq[tew]) ** 2).sum() / (dS[tew, q] ** 2).sum())
    zw = np.stack([dP @ Wd[58] / sd[58], dP @ Wd[31] / sd[31]], 1)
    ev, V = np.linalg.eigh(zw.T @ zw / len(zw)); v1 = V[:, 1]
    if v1[0] < 0: v1 = -v1
    wa = Wd[58] / sd[58] * v1[0] + Wd[31] / sd[31] * v1[1]
    afun = lambda P: P[:, HP] @ wa; sda = float(afun(base['P']).std()); tol = 0.01 * sda
    R['naming'] = dict(nm, sd_a=sda, tol=tol); log('naming', json.dumps(clean(R['naming'])))
    # edits
    a0 = afun(b0['P']); fa = lambda xe: afun(pf(xe))
    agrid = lambda xe, mk, angs: np.concatenate([fa(W.rot_grid(xe[c:c + 20], mk[c:c + 20], angs)).reshape(-1, len(angs))
                                                 for c in range(0, len(xe), 20)])
    fm = lambda xe: W.moments(xe)['m']
    out = {e: np.array(xs, np.float64) for e in EDITS}; okE = {e: np.zeros(J, bool) for e in EDITS}
    beta, ok = W.search(xs, rs, a0, fa, agrid, rng, exclude=np.pi / 6)
    xe = W.rot(xs, rs, beta); okE['R'] = ok & (np.abs(fa(xe) - a0) <= tol); out['R'][okE['R']] = xe[okE['R']]
    log('R ok', int(okE['R'].sum()), '/', J)
    for e in ('D', 'D2'):
        for k in range(12):
            todo = np.flatnonzero(~okE[e])
            if len(todo) == 0: break
            xe = W.scramble(xs[todo], ss[todo], rng)
            bt, ok = W.search(xe, rs[todo], a0[todo], fa, agrid, rng)
            xe = W.rot(xe, rs[todo], bt); ok = ok & (np.abs(fa(xe) - a0[todo]) <= tol)
            out[e][todo[ok]] = xe[ok]; okE[e][todo[ok]] = True
        log(e, 'ok', int(okE[e].sum()), '/', J)
    mD = np.where(okE['D'], fm(out['D']), np.nan)
    for t in range(3):
        todo = np.flatnonzero(okE['D'] & ~okE['P'])
        if len(todo) == 0: break
        al, ok1 = W.search(xs[todo], ss[todo], mD[todo], fm, lambda xe, mk, angs: W.mass_grid(xe, mk, angs), rng,
                           exclude=np.pi / 18, iters=24, t=t)
        xe = W.rot(xs[todo], ss[todo], al); ok1 &= np.abs(fm(xe) / mD[todo] - 1) <= 0.01
        bt, ok2 = W.search(xe, rs[todo], a0[todo], fa, agrid, rng)
        xe = W.rot(xe, rs[todo], bt); ok = ok1 & ok2 & (np.abs(fa(xe) - a0[todo]) <= tol)
        out['P'][todo[ok]] = xe[ok]; okE['P'][todo[ok]] = True
    log('P ok', int(okE['P'].sum()))
    K = np.flatnonzero(okE['P'] & okE['D']); n = len(K); SM = {e: okE[e][K] for e in EDITS}
    R['acceptance'] = dict(J=J, **{e: int(okE[e].sum()) for e in EDITS}, analysis=n, R_in=int(SM['R'].sum()), D2_in=int(SM['D2'].sum()),
                           per_class={CL[c]: int((ys[K] == c).sum()) for c in range(5)})
    X0 = xs[K].astype(np.float32); Y = ys[K]; XE = {e: out[e][K].astype(np.float32) for e in EDITS}
    # edit checks (independent recomputation)
    SUB = {'none': E24.substructure(X0, with_d2=False)[0]}; chk = {}
    mo0 = W.moments(X0); r0, th0, re0 = W.polar(X0)
    for e in EDITS:
        SUB[e] = E24.substructure(XE[e], with_d2=False)[0]; s_ = SM[e]; mo = W.moments(XE[e]); r1, th1, _ = W.polar(XE[e])
        moved = (np.abs(W.wrap(th1 - th0)) > 1e-6) & re0
        chk[e] = dict(n=int(s_.sum()), max_rel_dM0=float(np.max(np.abs(mo['M0'] / mo0['M0'] - 1)[s_])),
                      max_rel_dM2=float(np.max(np.abs(mo['M2'] / mo0['M2'] - 1)[s_])),
                      max_abs_dr=float(np.max(np.abs(r1 - r0)[re0 & s_[:, None]])),
                      padding_changed=int((XE[e][s_][~re0[s_]] != X0[s_][~re0[s_]]).sum()),
                      mean_rel_dm=ci((mo['m'] / mo0['m'] - 1)[s_]), dtau21=ci((SUB[e] - SUB['none'])[s_]),
                      frac_particles_moved=float(moved[s_].sum() / re0[s_].sum()))
    mP, mDD = W.moments(XE['P'])['m'], W.moments(XE['D'])['m']
    chk['P_vs_D_mass_med_abs_rel'] = float(np.median(np.abs(mP / mDD - 1)))
    chk['D_minus_P_tau21'] = ci(SUB['D'] - SUB['P'])
    R['edit_checks'] = chk; log('checks', json.dumps(clean({e: dict(dm=chk[e]['mean_rel_dm']['mean'], dt=chk[e]['dtau21']['mean']) for e in EDITS})))
    save()
    # networks
    lgt = {}
    res = {'none': j.run(X0, keep=('g', 'S', 'logits'))}
    for e in EDITS: res[e] = j.run(XE[e], keep=('logits',))
    lgt['jedi'] = {k: v['logits'] for k, v in res.items()}
    g0, S0 = res['none']['g'], res['none']['S']; mS = np.zeros(64); mS[[58, 31]] = 1
    for arm, ov in (('g_held', {'g': (np.ones(64), g0)}), ('S58_S31_held', {'S': (mS, S0)})):
        lgt['jedi|' + arm] = {'none': lgt['jedi']['none'], **{e: j.run_jetwise(XE[e], ov)['logits'] for e in EDITS}}
    full_acc = {'jedi': acc}
    for name in ('dsets_s0', 'pfn_s0'):
        run, _, _, _ = E24.get_model(name); full_acc[name] = float((run(x)['logits'].argmax(1) == y).mean())
        lgt[name] = {'none': run(X0)['logits'], **{e: run(XE[e])['logits'] for e in EDITS}}
    log('networks done')
    # trees and MLPs on held features
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y_all = np.asarray(d['y'])
    im = E24.sample_idx(y_all, 3000, SEED); assert np.array_equal(y_all[im], y)
    xp, yp, _ = sample(4000, POOL_SEED); ip = E24.sample_idx(y_all, 4000, POOL_SEED); kp = ~np.isin(ip, im)
    xp, yp = xp[kp], yp[kp]; tp = E24.substructure(xp, with_d2=False)[0]; R['pool_n'] = int(len(xp))

    def feats(xx, t, which):
        mo = W.moments(xx); h = [np.log(mo['M0']), np.log(mo['M2'] / mo['M0']), mo['Mr'] / mo['M0'], mo['N']]
        lm = np.log(np.maximum(mo['m'], 1e-6))
        return np.stack({'hm': h + [lm], 'hmt': h + [lm, t], 'ht': h + [t]}[which], 1)
    for which in ('hm', 'hmt', 'ht'):
        for kind in ('T', 'M'):
            clf = (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0) if kind == 'T' else
                   make_pipeline(StandardScaler(), MLPClassifier((64, 64), max_iter=300, early_stopping=True, random_state=0)))
            clf.fit(feats(xp, tp, which), yp); nmc = kind + which
            full_acc[nmc] = float((clf.predict(feats(x, t21, which)) == y).mean())
            lgt[nmc] = {'none': clf.predict_proba(feats(X0, SUB['none'], which)), **{e: clf.predict_proba(feats(XE[e], SUB[e], which)) for e in EDITS}}
            lgt[nmc] = {k: np.log(np.maximum(v, 1e-9)) for k, v in lgt[nmc].items()}
            log(nmc, 'acc', round(full_acc[nmc], 4))
    R['full_sample_acc'] = full_acc
    # statistics
    WZ = (Y == 2) | (Y == 3)
    for name, Lg in lgt.items():
        C = {k: v.argmax(1) for k, v in Lg.items()}; c0 = C['none']; ch = {e: C[e] != c0 for e in EDITS}
        sc = {k: score(v) for k, v in Lg.items()}; sds = float(sc['none'].std())
        cen = lambda L: L - L.mean(1, keepdims=True); dl = {e: np.linalg.norm(cen(Lg[e]) - cen(Lg['none']), axis=1) for e in EDITS}
        M = dict(acc_unedited=ci(c0 == Y), calls_changed={e: ci(ch[e][SM[e]]) for e in EDITS}, sd_score=sds,
                 dscore_over_sd={e: ci(((sc[e] - sc['none']) / sds)[SM[e]]) for e in EDITS},
                 dlogit_l2={e: ci(dl[e][SM[e]]) for e in EDITS})
        for a_, b_ in (('D', 'P'), ('D2', 'D'), ('D', 'R'), ('P', 'R'), ('D2', 'P')):
            s_ = SM[a_] & SM[b_]; mc = mcnemar(ch[a_][s_], ch[b_][s_]); mc['p_one_sided'] = W.p_upper(mc['n10'], mc['n01'])
            dsc = (sc[a_] - sc[b_]) / sds
            M[f'{a_}_minus_{b_}'] = dict(n=int(s_.sum()), calls=ci((ch[a_].astype(float) - ch[b_])[s_]), mcnemar=mc,
                                         score=ci(dsc[s_]), score_p=boot_p(dsc[s_]),
                                         score_WZ=ci(dsc[s_ & WZ]), score_WZ_p=boot_p(dsc[s_ & WZ]),
                                         dlogit=ci((dl[a_] - dl[b_])[s_]))
        R['models'][name] = M
        log(f'{name:20s} changed', {e: round(M['calls_changed'][e]['mean'] * 100, 2) for e in EDITS},
            'D-P', round(M['D_minus_P']['calls']['mean'] * 100, 2), [round(v * 100, 2) for v in M['D_minus_P']['calls']['ci']],
            'p', f"{M['D_minus_P']['mcnemar']['p']:.2g}", '| D2-D', round(M['D2_minus_D']['calls']['mean'] * 100, 2),
            '| score D-P/sd', round(M['D_minus_P']['score']['mean'], 4), [round(v, 4) for v in M['D_minus_P']['score']['ci']],
            'P-R', round(M['P_minus_R']['score']['mean'], 4), '| D2-D', round(M['D2_minus_D']['score']['mean'], 4))
    ex = lambda nmc, a_, b_: (np.argmax(lgt[nmc][a_], 1) != np.argmax(lgt[nmc]['none'], 1)).astype(float) - (np.argmax(lgt[nmc][b_], 1) != np.argmax(lgt[nmc]['none'], 1))
    R['jedi_vs_controls_D_minus_P_calls'] = {c: dict(diff=ci(ex('jedi', 'D', 'P') - ex(c, 'D', 'P')), p=boot_p(ex('jedi', 'D', 'P') - ex(c, 'D', 'P')))
                                             for c in ('Thm', 'Thmt', 'Tht', 'Mhm', 'Mhmt', 'Mht')}
    save()
    np.savez_compressed(OUT / 'e24w_verify_calls.npz', idx=idx[K], y=Y, ok=np.stack([SM[e] for e in EDITS]),
                        tau21=np.stack([SUB[k] for k in ['none'] + EDITS]),
                        **{f'{nm_}|{e}': np.asarray(L[e], np.float32) for nm_, L in lgt.items() for e in ['none'] + EDITS})
    log('wrote', OUT / 'e24w_verify.json')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); main()
