"""P18v2 -- second verification control for P18 (ParT stage pipeline, C4). Reuses interp/p18_stage_pipeline_part.py unchanged.

Question 1 (edit scope): P18's DAS counterfactual for the azimuth scramble edits only the per-particle inputs and keeps the base
4-vectors (pair features). Does the depth profile of the 1-d IIA hold when the counterfactual is the fully scrambled jet (pair
features scrambled too)? Edits compared: scr (per-particle inputs only, as P18) and scrF (full edited jet).
Question 2 (why an untrained net gets high 1-d IIA): spectrum of the logit-change vectors d = L_cf - L_base over the evaluation
jets for every P18 edit (per-particle counterfactual as P18, and the full edited jet): top-1 share lambda_1 / sum lambda of the
uncentred second-moment matrix of d, and the participation ratio (sum lambda)^2 / sum lambda^2.

Sample: JetClass test_20M <sample>_102.root (P18: _100, P18v: _101), 500 random rows per class (default_rng(2018)), N = 5,000:
4,000 DAS-train, 1,000 evaluation jets (random split). Models: ParT_full, ParT_kin, untrained ParT_full at torch seeds 18 (P18's
init net) and 1018 (P18v's init net). DAS settings as P18 (200 Adam steps x 128 jets, lr 0.01; IIA with 1,000 jet bootstraps;
1,000 linearised random directions). Paired bootstrap over jets of IIA(trained) - IIA(init) and of the depth change b8 - emb.

  ./sync.sh && ./part_job.sh c-p18-verify2 'python -m interp.p18v2_fullcf_part'
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_argv = sys.argv; SM = '--smoke' in _argv
sys.argv = [_argv[0]] + (['--smoke'] if SM else [])
import interp.p18_stage_pipeline_part as P  # noqa: E402
sys.argv = _argv

SEED = 2018
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUT / ('p18_verify2_smoke.json' if SM else 'p18_verify2.json')
PER = 60 if SM else 500
log = P.log
ALL_EDITS = ['phi', 'dil', 'scr', 'soft5', 'pt']


def init_net(seed):
    import torch
    from interp.part_core import ParTNetwork
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import FEATURES, CFG
    pi = ParTNetwork('full'); torch.manual_seed(seed); fresh = ParticleTransformer(input_dim=len(FEATURES['full']), **CFG)
    pi.model.load_state_dict(fresh.state_dict()); pi.model.eval(); return pi


def logits(net, trip, idx, cnt):
    out = np.zeros((len(idx), 10)); order = np.argsort(cnt[idx], kind='stable')
    for i in range(0, len(idx), 128):
        o = order[i:i + 128]; L = net.run(*P.feed(net, *P.slc(trip, idx[o]))); out[o] = L.double().cpu().numpy()
    return out


def spectrum(d):
    S = d.T @ d / len(d); lam = np.clip(np.linalg.eigvalsh(S)[::-1], 0, None)
    return dict(top1_share=float(lam[0] / max(lam.sum(), 1e-30)), participation_ratio=float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-30)),
                mean_sq_norm=float((d ** 2).sum(1).mean()))


def main():
    import torch
    from interp.part_core import CL, PRE, cat, kin, inputs, ParTNetwork
    OUT.mkdir(parents=True, exist_ok=True)
    R = dict(node='P18v2', script='interp/p18v2_fullcf_part.py', base_script='interp/p18_stage_pipeline_part.py', seed=SEED,
             files='_102.root', per_class=PER, NB=P.NB, STEPS=P.STEPS, BS=P.BS)
    save = lambda: FN.write_text(json.dumps(P.strip(R), indent=1))
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED); Js = []
    for c in CL:
        Js.append(P.load_rows(P.TEST / f'{PRE[c]}_102.root', PER, rng)); log(f'loaded {c}')
    J0 = cat(Js); del Js; N = len(J0['label']); R['N'] = N
    pt, de, dp, _ = kin(J0); pt = pt.astype(np.float64); r_ = (pt > 0)
    M2a = (pt * (de ** 2 + dp ** 2)).sum(1); M2b = (pt * (de ** 2 + (dp + 0.06 * r_) ** 2)).sum(1)
    k_match = float(np.exp(np.mean(0.5 * np.log(M2b / np.maximum(M2a, 1e-12))))); R['k_match'] = k_match; del pt, de, dp, r_
    nets = {'full': P.Net(ParTNetwork('full')), 'kin': P.Net(ParTNetwork('kin')), 'init18': P.Net(init_net(18)), 'init1018': P.Net(init_net(1018))}
    feat = dict(full='full', kin='kin', init18='full', init1018='full')
    perm = np.random.default_rng(SEED + 1).permutation(N); n_tr = N * 4 // 5
    tr_i, ev_i = np.sort(perm[:n_tr]), np.sort(perm[n_tr:]); R['N_DAS_train'] = len(tr_i); R['N_eval'] = len(ev_i)
    # inputs: base, per-particle counterfactuals (as P18 das_data) and full edited jets
    Jb = P.make_edit(J0, 'base', k_match, np.random.default_rng(SEED + 11)); D = {}
    for fk in ('full', 'kin'):
        xb, vb, mb = inputs(Jb, fk); D[fk] = {'base': (xb, vb, mb)}
        for ed in ALL_EDITS:
            Je = P.make_edit(J0, ed, k_match, np.random.default_rng(SEED + 13 + ALL_EDITS.index(ed)))
            xe, ve, me = inputs(Je, fk); D[fk][ed + 'F'] = (xe, ve, me)
            if ed == 'soft5': D[fk][ed] = (xe, ve, me)
            else:
                assert np.array_equal(me, mb), ed
                D[fk][ed] = (xe, vb, mb)
    cnt = (Jb['mask'] > 0).sum(1)
    # (2) logit-change spectra and call changes on the evaluation jets
    R['logit_change'] = {}
    for mn, net in nets.items():
        Dk = D[feat[mn]]; Lb = logits(net, Dk['base'], ev_i, cnt); o = {}
        for ed in ALL_EDITS:
            for v in (ed, ed + 'F'):
                if v == 'soft5F': continue
                Lc = logits(net, Dk[v], ev_i, cnt); s = spectrum(Lc - Lb); s['calls_changed'] = float((Lc.argmax(1) != Lb.argmax(1)).mean()); o[v] = s
        R['logit_change'][mn] = o
        log(mn, 'logit-change spectra:', {k: (round(v['top1_share'], 3), round(v['participation_ratio'], 2), round(v['calls_changed'], 3)) for k, v in o.items()})
    save()
    # (1) DAS for scr (per-particle cf) and scrF (full edited jet)
    P.EDITS = ['scr', 'scrF']; P.VARS = dict(scr='tau21', scrF='tau21')
    boot_ev = P.Boot(torch, nets['full'].dev, len(ev_i), P.NB, SEED + 22); O = {}; R['das'] = {}
    for mn, net in nets.items():
        O[mn], _, fits = P.das_model(net, D[feat[mn]], cnt, tr_i, ev_i, np.random.default_rng(SEED + 32), None, mn, boot_ev)
        steps = P.das_tests(O[mn], None); R['das'][mn] = dict(variables=O[mn], steps=steps, fits=fits)
        for ed in P.EDITS: log(f'  DAS {mn} {ed}: IIA ' + ' '.join(f'{v:.2f}' for v in steps[ed]['iia']))
        save()
    R['vs_init'] = {}
    for mn in ('full', 'kin'):
        for ctl in ('init18', 'init1018'):
            o = {}
            for ed in P.EDITS:
                A, B = O[mn][ed]['sites'], O[ctl][ed]['sites']
                cell = {s: dict(diff=float(A[s]['das']['iia'] - B[s]['das']['iia']), ci=P.ci(A[s]['das']['_bs'] - B[s]['das']['_bs'])) for s in P.SITES}
                dd = (A['b8']['das']['_bs'] - A['emb']['das']['_bs']) - (B['b8']['das']['_bs'] - B['emb']['das']['_bs'])
                o[ed] = dict(per_site=cell, trained_b8_minus_emb=float(A['b8']['das']['iia'] - A['emb']['das']['iia']),
                             control_b8_minus_emb=float(B['b8']['das']['iia'] - B['emb']['das']['iia']), did=float(np.mean(dd)), did_ci=P.ci(dd), did_p=P.p_two(dd))
            R['vs_init'][f'{mn}-{ctl}'] = o
            log(mn, ctl, {ed: (round(v['did'], 3), [round(x, 3) for x in v['did_ci']]) for ed, v in o.items()})
    R['runtime_s'] = time.time() - P.T0; save(); log('wrote', FN)


if __name__ == '__main__':
    main()
