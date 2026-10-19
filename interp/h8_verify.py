"""H8 verify (loop D skeptic): rerun interp.h8_efp_ado on fresh jets and add a positive control for the surrogate comparison.

Fresh samples
  hls4ml: interp.jedi_core.sample indices with per_class 1500, seed 811 (H8: 2000, seed 801), JEDI-linear official n64 3-feature.
  JetClass: 1,000 random rows per class, seed 812 (H8: 802), ParT_full / ParT_kin / ParT_full redrawn; corrected eta build.
  The H8 pipeline (edits, observables, guided search, HGB/MLP surrogates, edit statistics) runs unchanged on these jets
  (h8.part_jedi / part_jetclass / part_efp with the sample functions patched); intermediates go to physics/h8_verify_work/.
Positive control (the H8 surrogates are translation-invariant and axis-free by construction, so their 0 % change under the WTA
  and translation edits is fixed in advance): add the axis-moment mass m_A = sqrt(M0 * M2), M0 = sum pT, M2 = sum pT (deta^2 +
  dphi^2) about the axis the network is given (A4: JEDI-linear's mass variable; un-recentred), and sum pT.
  'swap': the H8-chosen observables with m replaced by m_A (m_A appended when m was not chosen); same HGB, same fit half.
  'axis': the guided search rerun on the pool irc + m_A + sum pT.
  Prediction if the networks' WTA/translation response goes through the axis-moment mass: the 'swap'/'axis' surrogates change a
  similar share of calls as the network, and the per-jet score changes correlate. If it does not: they stay near 0 or uncorrelated.
Normalised sensitivity: mean |delta score| / sd(natural score) on the test jets of each task, trained vs redrawn networks (the
  redrawn ParT's score is near 0 for many jets, so the share of calls changed alone over-states its sensitivity).
Run
  ./part_job.sh d-h8-verifyg 'python -m interp.h8_verify --part jetclass'
  ./job.sh d-h8-verify 'python -m interp.h8_verify --part jedi && <venv-ef python> -m interp.h8_verify --part efp && <venv-ef> ... --part control'
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.h8_efp_ado as h8  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
PHYS = RES / 'physics'
WORK = PHYS / 'h8_verify_work'
h8.OUT = WORK
SEED_J, SEED_C, PER_J, PER_C = 811, 812, 1500, 1000
log = h8.log


def patch():
    import interp.h1_irc_panel as h1
    hs, js = h1.hls_sample, h1.jc_sample
    h1.hls_sample = lambda per, seed=None: hs(PER_J, seed=SEED_J)
    h1.jc_sample = lambda per, seed=None: js(PER_C, seed=SEED_C)


def axis_feats(pt, de, dp):
    pt = np.asarray(pt, np.float64); de = np.asarray(de, np.float64); dp = np.asarray(dp, np.float64)
    real = pt > 0; M0 = pt.sum(1); M2 = (pt * (de ** 2 + dp ** 2) * real).sum(1)
    M1 = np.stack([(pt * de * real).sum(1), (pt * dp * real).sum(1)], 1)
    return np.sqrt(M0 * M2), M0, (M1 ** 2).sum(1) / np.maximum(M0 * M2, 1e-300)


def control():
    WORK.mkdir(parents=True, exist_ok=True)
    R = json.loads((WORK / 'h8_efp.json').read_text()); info = h8.efp_info()
    out = dict(samples=dict(hls4ml=f'jedi_core.sample indices per_class {PER_J}, seed {SEED_J}', jetclass=f'{PER_C} random rows per class, rng {SEED_C}'),
               datasets={})
    tests = []
    for ds in ('jedi', 'jc'):
        net, OBS = h8.load_obs(ds, 0, None)
        y = net['y'].astype(int); cls = [str(c) for c in net['classes']]; N = len(y)
        names, F0, _ = h8.feature_table(OBS['natural'], info)
        AX = {}
        for c in h8.CONDS:
            if ds == 'jedi':
                x = net[f'x_{c}']; AX[c] = axis_feats(x[..., 0], x[..., 1], x[..., 2])
            else:
                AX[c] = axis_feats(net[f'pt_{c}'], net[f'de_{c}'], net[f'dp_{c}'])
        FE = {c: np.concatenate([h8.feature_table(OBS[c], info)[1], AX[c][0][:, None], AX[c][1][:, None]], 1) for c in h8.CONDS}
        names = names + ['mA', 'sumpt']; F0 = FE['natural']
        iM, iA = names.index('m'), names.index('mA')
        mrel = float(np.corrcoef(np.log(F0[:, iM] + 1e-9), np.log(F0[:, iA] + 1e-9))[0, 1])
        D = dict(N=N, corr_log_m_log_mA_natural=mrel, M1sq_over_M0M2_median={c: float(np.median(AX[c][2])) for c in h8.CONDS},
                 half_dlog_M0M2_median={c: float(np.median(0.5 * np.log(np.maximum(AX[c][0], 1e-12) ** 2 / np.maximum(AX['natural'][0], 1e-12) ** 2)))
                                        for c in h8.EDITS}, models={})
        log(f'{ds}: corr(log m, log mA) {mrel:.5f}; median |M1|^2/(M0 M2) ' + ' '.join(f'{c} {v:.4f}' for c, v in D['M1sq_over_M0M2_median'].items()))
        rng = np.random.default_rng(803); half = rng.permutation(N); fit, test = np.sort(half[:N // 2]), np.sort(half[N // 2:])
        rngAV = np.random.default_rng(804); fp = rngAV.permutation(fit); A_, V_ = np.sort(fp[:2 * len(fp) // 3]), np.sort(fp[2 * len(fp) // 3:])
        irc_cols = [j for j, nm in enumerate(names) if nm not in ('N', 'ptD', 'mA', 'sumpt')]
        axis_cols = [j for j, nm in enumerate(names) if nm not in ('N', 'ptD')]
        for mn in h8.MODELS[ds]:
            D['models'][mn] = {}
            for tk, (sg, bk) in h8.TASKS[ds].items():
                si_, bi_ = [cls.index(c) for c in sg], [cls.index(c) for c in bk]
                S = {c: h8.score(net[f'L_{mn}_{c}'], si_, bi_) for c in h8.CONDS}; s = S['natural']
                issig, isbkg = np.isin(y, si_), np.isin(y, bi_); task = issig | isbkg
                fr = fit[task[fit]]; te = test[task[test]]; tsig, tbkg = test[issig[test]], test[isbkg[test]]
                sd = float(s[te].std())
                E = dict(net_norm_sens={c: float(np.abs(S[c][te] - s[te]).mean() / sd) for c in h8.EDITS}, variants={})
                chosen_irc = R['datasets'][ds]['models'][mn][tk]['pools']['irc']['chosen']
                swap = [('mA' if nm == 'm' else nm) for nm in chosen_irc]
                if 'mA' not in swap: swap = swap + ['mA']
                cs, st = h8.guided_search(F0[:, axis_cols], [names[j] for j in axis_cols], s, A_[issig[A_]], A_[isbkg[A_]], V_[issig[V_]],
                                          V_[isbkg[V_]], np.random.default_rng(805))
                axis_chosen = [names[axis_cols[j]] for j in cs]
                for vn, ch in (('irc', chosen_irc), ('swap', swap), ('axis', axis_chosen)):
                    jj = [names.index(nm) for nm in ch]
                    mdl = h8.hgb().fit(F0[fr][:, jj], s[fr]); SS = {c: mdl.predict(FE[c][:, jj]) for c in h8.CONDS}; ss = SS['natural']
                    rb = np.random.default_rng(806)
                    V = dict(chosen=ch, ado_test=h8.ado_all(s, ss, tsig, tbkg), ado_ci=h8.ado_boot(s, ss, tsig, tbkg, rb),
                             same_call=float(((s[te] > 0) == (ss[te] > 0)).mean()),
                             sur_norm_sens={c: float(np.abs(SS[c][te] - ss[te]).mean() / sd) for c in h8.EDITS}, edits={})
                    for c in h8.EDITS:
                        V['edits'][c] = h8.edit_stats(s, S[c], ss, SS[c], te, rb)
                    if vn == 'axis': V['search_steps'] = st
                    E['variants'][vn] = V
                    if vn == 'swap' and 'redrawn' not in mn:
                        for c in h8.EDITS: tests.append((ds, mn, tk, c, V['edits'][c]['p_mcnemar']))
                    log(f'{ds} {mn} {tk} {vn}: {ch}; ADO {V["ado_test"]:.4f}; ' + ' '.join(
                        f'{c} net {V["edits"][c]["net_changed"]:.3f} sur {V["edits"][c]["sur_changed"]:.3f} r {V["edits"][c]["corr_delta"]}' for c in h8.EDITS))
                D['models'][mn][tk] = E
        out['datasets'][ds] = D
    q = h8.bh([t[4] for t in tests])
    for (ds, mn, tk, c, p), qq in zip(tests, q):
        out['datasets'][ds]['models'][mn][tk]['variants']['swap']['edits'][c]['bh_q'] = qq
    out['bh'] = dict(n_tests=len(tests), family='swap surrogate vs network, trained models x tasks x edits')
    # fresh-sample rerun summary of the unchanged H8 pipeline
    summ = {}
    for ds in ('jedi', 'jc'):
        for mn, M in R['datasets'][ds]['models'].items():
            for tk, E in M.items():
                P = E['pools']['irc']; Sx = P['surrogates']['hgb']
                summ[f'{mn}|{tk}'] = dict(chosen=P['chosen'], ado=Sx['ado_test'], ado_ci=Sx['ado_ci'], same_call=Sx['same_call'],
                                          mlp_ado=P['surrogates']['mlp']['ado_test'],
                                          edits={c: [Sx['edits'][c]['net_changed'], Sx['edits'][c]['sur_changed'], Sx['edits'][c].get('bh_q')] for c in
                                                 ('wta', 'translate', 'boost', 'split', 'split_no_overflow')})
    out['h8_rerun'] = summ
    for f in ('h8_jedi_net.json', 'h8_jc_net.json'):
        out[f.replace('.json', '')] = json.loads((WORK / f).read_text())
    PHYS.mkdir(parents=True, exist_ok=True)
    (PHYS / 'h8_verify.json').write_text(json.dumps(out, indent=1, default=float)); log('wrote h8_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['jedi', 'jetclass', 'efp', 'control']); a = ap.parse_args()
    log(f'H8 verify part {a.part}')
    WORK.mkdir(parents=True, exist_ok=True)
    if a.part in ('jedi', 'jetclass'):
        patch()
    dict(jedi=h8.part_jedi, jetclass=h8.part_jetclass, efp=h8.part_efp, control=control)[a.part](0) if a.part != 'control' else control()
    log('done')
