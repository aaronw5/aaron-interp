"""P13 verify, part 2: how much of the 'main' feature-circuit accuracy is carried by the parts that stay live (particle attention,
transcoder skip maps, embeddings) rather than by the kept features.

Fresh sample: WToQQ/ZToQQ_111.root (test_20M), NEVAL random rows each, seed 41 (never first-k). Ranking = P13's original IG
scores (p13_scores.npz); live = features with a nonzero P13 score (a feature that never fires has exactly zero mask gradient).
Corruption: P13's resample ablation B (opposite-class pT-matched donor, errors resampled).
Specs (main = attention + skip live; strict = also skip + particle attention resampled):
  class_R1      : every live R1 SAE latent kept, nothing else          (the R1 SAE is the bottleneck to the read-out)
  class_R0R1    : every live R0 and R1 latent
  class_all     : every live class-level feature (C0, R0, C1, R1); all particle-site features resampled
  top50         : P13's top-50 features;   top50_classonly: its class-level members only
  top50_plusR1  : top-50 plus every live R1 latent
  top1000_classonly: the class-level members of P13's top-1000
  ./part_job.sh c-p13-verify2 'python -m interp.p13_verify2'
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.p13_transcoder_part import make_dict_cls, Runner, MaskH, prepset, load_rows, to_dev, jet_batches, SITES, DTC, DSAE, K_TOP, TEST, OUT  # noqa: E402
from interp.p9_eap_wz_part import boot_W, ci, mcnemar, clean, donors_B  # noqa: E402

SMOKE = '--smoke' in sys.argv; NEVAL = 100 if SMOKE else 3000; NB = 200 if SMOKE else 1000; SEED = 41
FN = OUT / f'p13_verify2{"_smoke" if SMOKE else ""}.json'; T0 = time.time()


def log(*a): print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def main():
    import torch as t
    from interp.part_core import ParTNetwork, inputs, sub, cat, CL, PRE
    iW, iZ = CL.index('Wqq'), CL.index('Zqq'); res = dict(smoke=SMOKE, seed=SEED, NEVAL=NEVAL)
    net = ParTNetwork('full'); model = net.model; dev = net.device
    for p in model.parameters(): p.requires_grad_(False)
    run = Runner(model); Dict = make_dict_cls(); blob = t.load(OUT / 'p13_dicts.pt', map_location=dev); D = {}
    for s in SITES:
        D[s] = Dict(128, DTC if s[0] in 'PC' else DSAE, K_TOP, s[0] in 'PC').to(dev); D[s].load_state_dict(blob['dicts'][s])
        for p in D[s].parameters(): p.requires_grad_(False)
    orig = np.load(OUT / 'p13_scores.npz')
    rank = sorted([(s, i, float(orig[s][i])) for s in SITES for i in range(D[s].D) if orig[s][i] != 0], key=lambda r: -abs(r[2]))
    live = {s: [i for i in range(D[s].D) if orig[s][i] != 0] for s in SITES}
    res['n_live'] = {s: len(v) for s, v in live.items()}; log('live', res['n_live'])

    rng = np.random.default_rng(SEED); files = {}; pE = []
    for c in ('Wqq', 'Zqq'):
        J, inf = load_rows(TEST / f'{PRE[c]}_111.root', NEVAL, rng); pE.append(J); files[f'eval_{c}'] = inf
    JE, yE = prepset(pE); del pE; res['samples'] = files
    don = donors_B(JE['jet_pt'], yE, iW, iZ, np.random.default_rng(SEED))

    def spec_of(feats, err=0.0, **kw):
        m = {s: (t.zeros(D[s].D, device=dev), t.tensor(float(err), device=dev)) for s in SITES}
        for s, i in feats: m[s][0][i] = 1.0
        return dict(m=m, **kw)

    cls_of = lambda f: [(s, i) for s, i in f if s[0] != 'P']
    top = lambda K: [(s, i) for s, i, _ in rank[:K]]
    allsite = lambda ss: [(s, i) for s in ss for i in live[s]]
    sets = dict(class_R1=allsite(['R1']), class_R0R1=allsite(['R0', 'R1']), class_all=allsite(['C0', 'R0', 'C1', 'R1']),
                top50=top(50), top50_classonly=cls_of(top(50)), top50_plusR1=sorted(set(top(50)) | set(allsite(['R1']))),
                top1000_classonly=cls_of(top(1000)), top1000=top(1000))
    specs = {}
    for k, f in sets.items():
        specs[f'main_{k}'] = spec_of(f); specs[f'strict_{k}'] = spec_of(f, 0.0, skip_live=False, attn_res=True)
    res['set_sizes'] = {k: len(v) for k, v in sets.items()}
    H = MaskH(D); out = {k: np.zeros((len(yE), 10), np.float32) for k in list(specs) + ['clean', 'corr']}
    with t.no_grad():
        for bb in jet_batches(np.maximum(JE['mask'].sum(1), JE['mask'][don].sum(1)), 64):
            x, v, mk = inputs(cat([sub(JE, bb), sub(JE, don[bb])]), 'full'); n = len(bb); xs = to_dev(t, dev, (x, v, mk))
            cl, dn = [a[:n] for a in xs], [a[n:] for a in xs]
            H.mode = 'donor'; Ld = run(*dn, H); Lc = run(*cl); H.mode = 'clean'
            out['clean'][bb] = Lc.cpu().numpy(); out['corr'][bb] = Ld.cpu().numpy()
            for k, sp in specs.items(): H.spec = sp; out[k][bb] = run(*cl, H).cpu().numpy()
    Wb = boot_W(len(yE), NB, np.random.default_rng(2)); sE = np.where(yE == iW, 1.0, -1.0)
    okf = lambda L: np.where(L[:, iW] > L[:, iZ], iW, iZ) == yE
    ok0, okd = okf(out['clean']), okf(out['corr'])
    res['unedited_wz_acc'] = float(ok0.mean()); res['unedited_wz_ci'] = ci(Wb @ ok0 / len(yE)); res['donor_wz_acc'] = float(okd.mean())
    dc = out['clean'][:, iW] - out['clean'][:, iZ]; dco = out['corr'][:, iW] - out['corr'][:, iZ]; R = {}
    for k in specs:
        L = out[k]; ok = okf(L); d = L[:, iW] - L[:, iZ]; num, den = sE * (d - dco), sE * (dc - dco)
        R[k] = dict(n_features=res['set_sizes'][k.split('_', 1)[1]], wz_acc=float(ok.mean()), wz_acc_ci=ci(Wb @ ok / len(yE)),
                    F=float(num.sum() / den.sum()), F_ci=ci((Wb @ num) / (Wb @ den)), mcnemar_vs_full=mcnemar(ok0, ok))
        log(f'{k:28s} n {R[k]["n_features"]:5d}  W/Z {R[k]["wz_acc"]:.4f} {R[k]["wz_acc_ci"]}  F {R[k]["F"]:.3f}')
    # paired differences of interest (bootstrap over jets)
    pd = {}
    for a, b in (('main_top50', 'main_top50_classonly'), ('main_top50_plusR1', 'main_top50'), ('main_class_all', 'main_top1000'),
                 ('main_class_R1', 'main_top50'), ('main_top1000', 'main_top1000_classonly')):
        oa, ob = okf(out[a]), okf(out[b]); pd[f'{a} - {b}'] = dict(diff=float(oa.mean() - ob.mean()), ci=ci(Wb @ (oa.astype(float) - ob) / len(yE)), mcnemar=mcnemar(ob, oa))
    res['results'] = R; res['paired'] = pd; res['seconds'] = time.time() - T0
    log(f'unedited {res["unedited_wz_acc"]:.4f} donor {res["donor_wz_acc"]:.4f}'); log('paired', pd)
    FN.write_text(json.dumps(clean(res), indent=1)); log('done')


if __name__ == '__main__':
    main()
