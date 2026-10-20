"""P18v -- verification of P18 (ParT stage pipeline, C4). Reuses the functions of interp/p18_stage_pipeline_part.py unchanged on a
fresh sample and adds one control.

Fresh sample: JetClass test_20M <sample>_101.root (P18 used _100.root), PER random rows per class (numpy default_rng(1018)),
N = 10 x PER. All seeds of P18 (perm permutation, init net, rotation, edits, splits, DAS starts) replaced by 1018-based seeds.
Added control: DAS (same fit, same evaluation) on the untrained ParT_full ("init"), so that the depth profile of the 1-d IIA can be
compared with an untrained network of the same architecture (P18 ran DAS only on the weight-permuted net). For each edit and site:
IIA_trained - IIA_init with a paired jet bootstrap (1,000) and Holm over 11 sites x 5 edits; and the depth change
(b8 - emb) of trained minus the same of init (difference-in-differences) with bootstrap CI.

  ./sync.sh && ./part_job.sh c-p18-verify 'python -m interp.p18v_verify_part'
  python -m interp.p18v_verify_part --plot
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_argv = sys.argv; PLOT = '--plot' in _argv; SM = '--smoke' in _argv
sys.argv = [_argv[0]] + (['--smoke'] if SM else [])
import interp.p18_stage_pipeline_part as P  # noqa: E402
sys.argv = _argv

P.SEED = 1018
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUT / ('p18_verify_smoke.json' if SM else 'p18_verify.json')
FIG = OUT / ('p18_verify_smoke.png' if SM else 'p18_verify.png')
PER = 200 if SM else 2000
log = P.log


def diffs(O, Oc, name):
    out = {}; pp, keys = [], []
    for ed in P.EDITS:
        for s in P.SITES:
            a = O[ed]['sites'][s]['das']; b = Oc[ed]['sites'][s]['das']; dd = a['_bs'] - b['_bs']
            out[f'{ed}|{s}'] = dict(diff=float(a['iia'] - b['iia']), ci=P.ci(dd), p=P.p_two(dd)); pp.append(out[f'{ed}|{s}']['p']); keys.append(f'{ed}|{s}')
    for k, h in zip(keys, P.holm(pp)): out[k]['p_holm'] = h
    did = {}
    for ed in P.EDITS:
        A = O[ed]['sites']; B = Oc[ed]['sites']
        d = (A['b8']['das']['_bs'] - A['emb']['das']['_bs']) - (B['b8']['das']['_bs'] - B['emb']['das']['_bs'])
        did[ed] = dict(trained_b8_minus_emb=float(A['b8']['das']['iia'] - A['emb']['das']['iia']),
                       control_b8_minus_emb=float(B['b8']['das']['iia'] - B['emb']['das']['iia']),
                       did=float((A['b8']['das']['iia'] - A['emb']['das']['iia']) - (B['b8']['das']['iia'] - B['emb']['das']['iia'])),
                       did_ci=P.ci(d), did_p=P.p_two(d))
    return dict(per_cell=out, depth_did=did, control=name)


def main():
    import torch
    from interp.part_core import CL, PRE, sample, cat, kin
    OUT.mkdir(parents=True, exist_ok=True); SEED = P.SEED
    models = ['full', 'kin']
    R = dict(node='P18v', script='interp/p18v_verify_part.py', base_script='interp/p18_stage_pipeline_part.py', seed=SEED, files='_101.root',
             per_class=PER, NB=P.NB, STEPS=P.STEPS, BS=P.BS, N_DAS_train=P.N_DAS_TR, N_eval=P.N_EVAL)
    save = lambda: FN.write_text(json.dumps(P.strip(R), indent=1))
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED); Js = []
    for c in CL:
        Js.append(P.load_rows(P.TEST / f'{PRE[c]}_101.root', PER, rng)); log(f'loaded {c}')
    J0 = cat(Js); del Js; N = len(J0['label']); R['N'] = N
    M = P.make_models(models); nets = {mn: P.Net(pn) for mn, pn in M.items()}; dev = M['full'].device; allm = list(M)
    J1 = sample(per_class=1000, start=3000); J1b = P.make_edit(J1, 'base', 1.0, None)
    R['accuracy_P1_rows'] = {mn: float((M[mn].run(J1b)['logits'].argmax(1) == J1['label']).mean()) for mn in allm}; log('acc P1 rows', R['accuracy_P1_rows']); del J1, J1b
    pt, de, dp, _ = kin(J0); pt = pt.astype(np.float64); r_ = (pt > 0)
    M2a = (pt * (de ** 2 + dp ** 2)).sum(1); M2b = (pt * (de ** 2 + (dp + 0.06 * r_) ** 2)).sum(1)
    k_match = float(np.exp(np.mean(0.5 * np.log(M2b / np.maximum(M2a, 1e-12))))); R['k_match'] = k_match; del pt, de, dp, r_
    perm = np.random.default_rng(SEED).permutation(N); tr, te = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    fit, val = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    das_tr = np.sort(np.random.default_rng(SEED + 1).choice(tr, P.N_DAS_TR, replace=False))
    das_ev = np.sort(np.random.default_rng(SEED + 2).choice(te, P.N_EVAL, replace=False))
    F, Lg, TG = P.extract(J0, allm, nets, k_match, das_tr, das_ev)
    y = J0['label'].astype(int); iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    R['accuracy_sample'] = {mn: float((Lg[mn]['base'].argmax(1) == y).mean()) for mn in allm}
    R['edit_calls'] = {mn: {ed: dict(changed=float((Lg[mn][ed].argmax(1) != Lg[mn]['base'].argmax(1)).mean()),
                                     W_to_Z_base=float((Lg[mn]['base'].argmax(1)[y == iW] == iZ).mean()),
                                     W_to_Z=float((Lg[mn][ed].argmax(1)[y == iW] == iZ).mean())) for ed in P.EDITS} for mn in allm}
    R['edit_true_changes'] = {ed: {t: dict(mean=float((TG[ed][t] - TG['base'][t]).mean()), sd=float((TG[ed][t] - TG['base'][t]).std()))
                                   for t in P.TARGETS} for ed in P.EDITS}
    # correlation of the scramble's true changes across targets (edit specificity)
    dts = np.stack([TG['scr'][t] - TG['base'][t] for t in ('tau21', 'tau32', 'log_D2', 'log_minv')], 0)
    R['scr_dt_corr'] = np.corrcoef(dts).tolist()
    log('acc sample', R['accuracy_sample']); save()
    boot = P.Boot(torch, dev, len(te), P.NB, SEED + 21)
    PR, dirs, R2B = P.probes(F, TG, allm, tr, te, fit, val, boot, torch, dev)
    R['probes'] = PR; R['depth_tests'] = P.depth_tests(PR, R2B, allm); save(); del F
    boot_ev = P.Boot(torch, dev, P.N_EVAL, P.NB, SEED + 22); R['das'] = {}
    dasidx = np.concatenate([das_tr, das_ev]); Dall, cnt = P.das_data(J0, dasidx, k_match, ['full', 'kin'])
    tr_i = np.arange(len(das_tr)); ev_i = np.arange(len(das_tr), len(dasidx)); O = {}
    for mn in ['init', 'perm']:
        O[mn], _, _ = P.das_model(nets[mn], Dall['full'], cnt, tr_i, ev_i, np.random.default_rng(SEED + 31), None, mn, boot_ev)
        R['das'][mn] = dict(variables=O[mn], steps=P.das_tests(O[mn], None)); save()
    for mn in models:
        O[mn], _, _ = P.das_model(nets[mn], Dall[P.FEAT_OF(mn)], cnt, tr_i, ev_i, np.random.default_rng(SEED + 32), dirs, mn, boot_ev)
        steps = P.das_tests(O[mn], O['perm'])
        R['das'][mn] = dict(variables=O[mn], steps=steps, vs_init=diffs(O[mn], O['init'], 'init'), vs_perm_did=diffs(O[mn], O['perm'], 'perm'))
        for ed in P.EDITS:
            log(f'  DAS {mn} {ed}: ' + ' '.join(f'{v:.2f}' for v in steps[ed]['iia']) + f" | init " +
                ' '.join(f"{O['init'][ed]['sites'][s]['das']['iia']:.2f}" for s in P.SITES) + f" | DiD vs init {R['das'][mn]['vs_init']['depth_did'][ed]}")
        save()
    R['runtime_s'] = time.time() - P.T0; save(); log('wrote', FN)
    try: plot(json.loads(FN.read_text()))
    except Exception as e: log('plot failed', repr(e))


def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = res or json.loads(FN.read_text()); S = P.SITES; x = np.arange(len(S)); lab = ['emb'] + [str(l) for l in range(1, 9)] + ['c1', 'c2']
    try: old = json.loads((OUT / 'p18_stage_pipeline.json').read_text())
    except Exception: old = None
    ecol = dict(phi='#d95f02', dil='#666666', scr='#7570b3', soft5='#1b9e77', pt='#66a61e')
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for j, mn in enumerate(['full', 'kin']):
        a = ax[j]
        for ed in P.EDITS:
            d = res['das'][mn]['variables'][ed]['sites']
            a.plot(x, [d[s]['das']['iia'] for s in S], '-o', ms=3, color=ecol[ed], label=f'{ed} (trained, fresh)')
            a.fill_between(x, [d[s]['das']['iia_ci'][0] for s in S], [d[s]['das']['iia_ci'][1] for s in S], color=ecol[ed], alpha=0.2)
            a.plot(x, [res['das']['init']['variables'][ed]['sites'][s]['das']['iia'] for s in S], '--', color=ecol[ed], lw=1)
            if old: a.plot(x, old['das'][mn]['steps'][ed]['iia'], 'x', color=ecol[ed], ms=5)
        a.set_title(f'ParT_{mn}: DAS 1-d IIA (solid: fresh _101 rows; x: P18; dashed: untrained net)', fontsize=9)
        a.set_ylim(-0.1, 1.02); a.set_xticks(x); a.set_xticklabels(lab); a.grid(alpha=0.3); a.set_xlabel('site'); a.set_ylabel('logit recovery (IIA)')
        if j == 0: a.legend(fontsize=7, loc='upper right')
    a = ax[2]
    col = dict(N='#1b9e77', log_pT='#66a61e', log_r2='#a6761d', tau21='#7570b3', tau32='#e7298a', log_D2='#e6ab02', log_minv='#666666', log_mA='#d95f02')
    for t in P.TARGETS:
        a.plot(x, [res['probes']['full'][s][t]['ridge']['r2'] for s in S], '-o', ms=3, color=col[t], label=t)
        a.plot(x, [res['probes']['init'][s][t]['ridge']['r2'] for s in S], '--', color=col[t], lw=1)
        if old: a.plot(x, [old['probes']['full'][s][t]['ridge']['r2'] for s in S], 'x', color=col[t], ms=5)
    a.set_title('ParT_full ridge R^2 (solid: fresh; x: P18; dashed: untrained net, fresh)', fontsize=9); a.set_ylim(-0.05, 1.02)
    a.set_xticks(x); a.set_xticklabels(lab); a.grid(alpha=0.3); a.legend(fontsize=7, ncol=2, loc='lower right'); a.set_xlabel('site')
    fig.tight_layout(); fig.savefig(FIG, dpi=110); print('wrote', FIG)


if __name__ == '__main__':
    plot() if PLOT else main()
