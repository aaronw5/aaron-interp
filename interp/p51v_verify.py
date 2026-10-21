"""P51v (loop C, round C12): verification of P51 (L-GATr final checkpoint, C6 test) on a fresh sample.

Sample. JetClass val_5M WToQQ_124.root and ZToQQ_124.root (P51 used _123; L-GATr's W/Z validation file is _121), 1,500 random
rows each, numpy default_rng(5199) (never first-k); jets with < 2 particles dropped. Edits are P51's own generator
(interp.p46_lgatr_c6.edits) with SEED = 5199 (pair plans, E6 directions) and doses FS = +-5, +-10 %; E6 skipped.

Models. L-GATr final (model_run0.pt); ParT_full and ParT_kin (released).

What is re-measured (same definitions as P51):
  R(edit) = sum_jets Delta d(edit) / sum_jets Delta d(matched dilation E4), d = logit_W - logit_Z.
  BOOST +-0.06 (exact longitudinal boost at fixed pT, masses; stored axis fixed): R, calls changed, McNemar, W->Z.
  L-GATr route split of BOOST: A full; D four-vectors boosted, 7 scalar features held; F only the scalar features from the
    boosted jet; C four-vectors + time reference boosted, features held (exact symmetry).
  E1m (pair spread changed, second pair moved until the jet invariant mass is restored) and E5 (rearrangement at fixed M2).
Added controls (not in P51):
  (i) route split of a REAL mass change for L-GATr: E4 dilation and E1 +-10 % with D (four-vectors changed, features held) and
      F (only the scalar features about the four-momentum-sum axis changed): R_D = sum dd_D / sum dd_A, R_F = sum dd_F / sum dd_A.
      R_F measures how much of L-GATr's response to a real mass change passes through its axis-relative scalar features.
  (ii) BOOST +-0.3 (5x the rapidity), routes A and C: does the frame-dependent residual grow with the boost?
  (iii) per-jet correlation of the feature route (A - D) and the time-reference route (D - C) under BOOST (cancellation test).
  (iv) rerun of the unedited input (GPU nondeterminism floor).
  ./sync.sh && ./part_job.sh c-p51-verify '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p51v_verify --stage lgatr && python -m interp.p51v_verify --stage part && python -m interp.p51v_verify --stage analyze'
-> $AI_RESULTS/part/p51_verify_lg.npz, p51_verify_part.npz, p51_verify.json
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ARGS = ap.parse_args()
_argv = sys.argv; sys.argv = [sys.argv[0], '--stage', 'none']
import interp.p46_lgatr_c6 as P46
sys.argv = _argv
SEED, FIDX, PER = 5199, 124, 1500
NB, NPERM = 2000, 5000
P46.SEED = SEED; P46.DL6 = []; P46.FS = [-0.10, -0.05, 0.05, 0.10]
ROOT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
SAMPLE = ROOT / 'p51_verify_sample.npz'
iW, iZ = P46.iW, P46.iZ
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def get_sample():
    from interp.part_core import sub, cat
    if SAMPLE.exists():
        z = np.load(SAMPLE); return {k: z[k] for k in z.files}
    import uproot
    rng = np.random.default_rng(SEED); Js = []
    for c in ('Wqq', 'Zqq'):
        p = str(P46.VAL / f'{P46.PRE[c]}_{FIDX}.root'); N = uproot.open(p)['tree'].num_entries
        Js.append(P46._read_rows((p, np.sort(rng.choice(N, PER, replace=False)))))
    WZ = cat(Js); assert (WZ['label'][:PER] == iW).all() and (WZ['label'][PER:] == iZ).all()
    ok = (WZ['mask'] > 0).sum(1) >= 2; WZ = sub(WZ, np.where(ok)[0]); log('W/Z jets', len(WZ['label']))
    ROOT.mkdir(parents=True, exist_ok=True); np.savez(SAMPLE, **WZ); return WZ


def boost_big(J, y):
    """P46's BOOST construction at rapidity y (fixed pT and mass, stored axis fixed)"""
    from interp.part_core import kin, build
    pt, de, dp, ms = kin(J); rm = pt > 0
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]; je = J['jet_eta'][:, None].astype(np.float64)
    eta = je + sg * de.astype(np.float64); p64 = np.maximum(pt.astype(np.float64), 1e-12); mT = np.sqrt(p64 ** 2 + ms.astype(np.float64) ** 2)
    yv = np.arcsinh(p64 * np.sinh(eta) / mT); eta2 = np.arcsinh(mT * np.sinh(yv + y) / p64)
    return build(J, pt, np.where(rm, sg * (eta2 - je), 0).astype(np.float32), dp, ms)


def put(S, model, edit, route, rows, L, N):
    d = np.full(N, np.nan, np.float32); p = np.full(N, -1, np.int8)
    d[rows] = L[:, iW] - L[:, iZ]; p[rows] = L.argmax(1); S[f'{model}|{edit}|{route}|d'] = d; S[f'{model}|{edit}|{route}|p'] = p


def stage_lgatr():
    import torch
    sys.argv = [sys.argv[0], '--stage', 'none']
    import interp.p44_lgatr_grades as P
    from interp.part_core import kin, build, sub, jmass
    WZ = get_sample(); N = len(WZ['label'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lg = P.LG(f'{P46.LGRUN}/model_run0.pt', dev); log('loaded L-GATr final', dev)
    S, M = {}, {}; out = {}
    for name, rows, Je, I in P46.edits(WZ):
        r = np.zeros(N, bool); r[I['rows']] = True; dl = np.full(N, np.nan); dl[I['rows']] = I['dlogM2']
        S[f'meta|{name}|rows'] = r; S[f'meta|{name}|dlogM2'] = dl; M[name] = dict(match=I['match'], geo=I['geo'])
        if name == 'base':
            Jb = Je; put(S, 'LG', name, 'A', rows, lg.logits(Je), N)
            L2 = lg.logits(Je); d0 = S['LG|base|A|d']; out['rerun_floor_maxabs'] = float(np.abs((L2[:, iW] - L2[:, iZ]) - d0).max())
            out['rerun_floor_meanabs'] = float(np.abs((L2[:, iW] - L2[:, iZ]) - d0).mean()); out['rerun_calls_changed'] = float((L2.argmax(1) != S['LG|base|A|p']).mean())
            continue
        Jbr = sub(Jb, rows); put(S, 'LG', name, 'A', rows, lg.logits(Je), N)
        if name.startswith(('BOOST', 'E4|BOOST', 'E1_', 'E4|E1_', 'E1m_', 'E5')):
            put(S, 'LG', name, 'D', rows, lg.logits(Je, Jbr), N); put(S, 'LG', name, 'F', rows, lg.logits(Jbr, Je), N)
        if name.startswith('BOOST'):
            put(S, 'LG', name, 'C', rows, lg.logits(Je, Jbr, spur_y=float(name[5:])), N)
        log(name, I['geo'], round(float(np.nanmean(S[f'LG|{name}|A|d'] - S['LG|base|A|d'])), 4))
    for y in (0.3, -0.3):
        nm = f'BOOSTBIG{y:+.1f}'; Je = boost_big(WZ, y); ALL = np.arange(N)
        M[nm] = dict(match=f'E4|BOOST{0.06 * np.sign(y):+.2f}', geo=dict(dlog_minv_maxabs=float(np.abs(np.log(np.maximum(jmass(Je), 1e-3) / np.maximum(jmass(Jb), 1e-3))).max())))
        S[f'meta|{nm}|rows'] = np.ones(N, bool)
        put(S, 'LG', nm, 'A', ALL, lg.logits(Je), N); put(S, 'LG', nm, 'C', ALL, lg.logits(Je, Jb, spur_y=y), N)
        put(S, 'LG', nm, 'D', ALL, lg.logits(Je, Jb), N); put(S, 'LG', nm, 'F', ALL, lg.logits(Jb, Je), N); log(nm, M[nm]['geo'])
    S['meta|y'] = WZ['label'].astype(int); out['meta'] = M; out['N'] = int(N)
    np.savez(ROOT / 'p51_verify_lg.npz', **S); (ROOT / 'p51_verify_lg_meta.json').write_text(json.dumps(out, indent=1, default=float)); log('wrote lg')


def stage_part():
    import torch
    from interp.p39v_verify import runner
    from interp.part_core import ParTNetwork
    from jetdistill.part.network import FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; Mdl = {}
    for nm, feat in (('ParT_full', 'full'), ('ParT_kin', 'kin')):
        net = ParTNetwork(feat); net.model.to(dev).eval(); Mdl[nm] = dict(kind='part', feat=feat, model=net.model)
    names = list(Mdl); run = runner(Mdl, FEATURES, dev); WZ = get_sample(); N = len(WZ['label']); S = {}
    for name, rows, Je, I in P46.edits(WZ):
        L = run(Je, names)
        for k in names: put(S, k, name, 'A', rows, L[k], N)
        log(name, {k: round(float(np.nanmean(S[f'{k}|{name}|A|d'] - S[f'{k}|base|A|d'])), 4) for k in names})
    np.savez(ROOT / 'p51_verify_part.npz', **S); log('wrote part')


def stage_analyze():
    S = {}
    for f in ('p51_verify_lg.npz', 'p51_verify_part.npz'):
        z = np.load(ROOT / f); S.update({k: z[k] for k in z.files})
    meta = json.loads((ROOT / 'p51_verify_lg_meta.json').read_text()); Mt = meta['meta']
    y = S['meta|y']; N = len(y)
    rb = np.random.default_rng(SEED + 3); C = np.zeros((NB, N), np.float32)
    for b in range(NB): C[b] = np.bincount(rb.integers(0, N, N), minlength=N)
    prng = np.random.default_rng(SEED + 4)
    ci, holm, signflip, mcnemar, boot_p = P46.ci, P46.holm, P46.signflip, P46.mcnemar, P46.boot_p
    P46.NPERM = NPERM
    models = ['LG', 'ParT_full', 'ParT_kin']; den_of = lambda e: Mt[e]['match'] if Mt[e]['match'].startswith('E4|') else f"E4|{Mt[e]['match']}"

    def dd(m, e, route='A'):
        k = f'{m}|{e}|{route}|d'
        if k not in S: return None
        d = S[k].astype(np.float64) - S[f'{m}|base|A|d'].astype(np.float64); return np.where(np.isfinite(d), d, 0.0)
    res = dict(node='P51v', sample=f'val_5M {{WToQQ,ZToQQ}}_{FIDX}.root, {PER}/class random, default_rng({SEED})', N=int(N), NB=NB,
               readout='d = logit_W - logit_Z', geometry={e: v['geo'] for e, v in Mt.items()},
               rerun=dict(maxabs=meta['rerun_floor_maxabs'], meanabs=meta['rerun_floor_meanabs'], calls=meta['rerun_calls_changed']),
               acc_WZ={m: float((S[f'{m}|base|A|p'] == y).mean()) for m in models}, fam={}, compare={}, routes={}, boost_big={})
    BO = {}
    fams = {'BOOST': ['BOOST+0.06', 'BOOST-0.06'], 'PHI': ['PHI+0.06', 'PHI-0.06'], 'E1_raise': ['E1_+0.05', 'E1_+0.10'],
            'E1_lower': ['E1_-0.05', 'E1_-0.10'], 'E1m_raise': ['E1m_+0.05', 'E1m_+0.10'], 'E1m_lower': ['E1m_-0.05', 'E1m_-0.10']}
    for m in models:
        o = {}; pv, pk = [], []
        for fn, es in fams.items():
            nums = [dd(m, e) for e in es]; dens = [dd(m, den_of(e)) * S[f'meta|{e}|rows'] for e in es]
            nu, de_ = sum(nums), sum(dens); b = (C @ nu) / (C @ de_); BO[(m, fn)] = b
            rowsU = np.concatenate([np.where(S[f'meta|{e}|rows'])[0] for e in es]); allnum = np.concatenate([x[S[f'meta|{e}|rows']] for x, e in zip(nums, es)])
            p0 = np.concatenate([S[f'{m}|base|A|p'][S[f'meta|{e}|rows']] for e in es]); p1 = np.concatenate([S[f'{m}|{e}|A|p'][S[f'meta|{e}|rows']] for e in es])
            pE = np.concatenate([S[f'{m}|{den_of(e)}|A|p'][S[f'meta|{e}|rows']] for e in es]); yy = y[rowsU]; w = yy == iW
            o[fn] = dict(R=float(nu.sum() / de_.sum()), R_ci=ci(b), n_pairs=int(len(yy)), mean_absdd=float(np.abs(allnum).mean()),
                         mean_absdd_E4=float(np.abs(np.concatenate([x[S[f'meta|{e}|rows']] for x, e in zip(dens, es)])).mean()),
                         calls_changed=float((p0 != p1).mean()), calls_changed_E4=float((p0 != pE).mean()),
                         WtoZ=[float((p0[w] == iZ).mean()), float((p1[w] == iZ).mean())], WtoZ_E4=float((pE[w] == iZ).mean()),
                         mcnemar=mcnemar(p0 == yy, p1 == yy), p_signflip=signflip(allnum, prng))
            pv.append(o[fn]['p_signflip']); pk.append(fn)
        up, dn = fams['E1m_raise'], fams['E1m_lower']
        nu_u, nu_d = sum(dd(m, e) for e in up), sum(dd(m, e) for e in dn)
        de_u = sum(dd(m, den_of(e)) * S[f'meta|{e}|rows'] for e in up); de_d = sum(dd(m, den_of(e)) * S[f'meta|{e}|rows'] for e in dn)
        den = de_u - de_d
        for nm_, nu in (('E1m_antisym', nu_u - nu_d), ('E1m_sym', nu_u + nu_d)):
            b = (C @ nu) / (C @ den); BO[(m, nm_)] = b; o[nm_] = dict(R=float(nu.sum() / den.sum()), R_ci=ci(b))
        n5 = dd(m, 'E5+') + dd(m, 'E5-'); r5 = S['meta|E5+|rows']; d5 = (dd(m, 'E4|E1_+0.10') - dd(m, 'E4|E1_-0.10')) * r5
        b = (C @ n5) / (C @ d5); BO[(m, 'E5')] = b
        p0 = np.concatenate([S[f'{m}|base|A|p'][r5]] * 2); p1 = np.concatenate([S[f'{m}|E5+|A|p'][r5], S[f'{m}|E5-|A|p'][r5]]); yy = np.concatenate([y[r5]] * 2)
        o['E5'] = dict(R=float(n5.sum() / d5.sum()), R_ci=ci(b), calls_changed=float((p0 != p1).mean()), mcnemar=mcnemar(p0 == yy, p1 == yy))
        for k, h in zip(pk, holm(pv)): o[k]['p_holm'] = h
        res['fam'][m] = o; log(m, {k: (round(v['R'], 4), [round(x, 4) for x in v['R_ci']]) for k, v in o.items()})
    for ref in ('ParT_full', 'ParT_kin'):
        cmp = {}
        for fn in ('BOOST', 'PHI', 'E1_raise', 'E1_lower', 'E1m_antisym', 'E1m_sym', 'E5'):
            bb = BO[('LG', fn)] - BO[(ref, fn)]; cmp[fn] = dict(diff=res['fam']['LG'][fn]['R'] - res['fam'][ref][fn]['R'], ci=ci(bb), p=boot_p(bb))
        for k, h in zip(list(cmp), holm([v['p'] for v in cmp.values()])): cmp[k]['p_holm'] = h
        res['compare'][f'LG_minus_{ref}'] = cmp
    # L-GATr route splits
    rt = {}
    for e in ('BOOST+0.06', 'BOOST-0.06', 'BOOSTBIG+0.3', 'BOOSTBIG-0.3'):
        a, d_, f_, c_ = (dd('LG', e, r) for r in 'ADFC'); p0 = S['LG|base|A|p']
        fr, sr = a - d_, d_ - c_
        rt[e] = dict(mean_dd={r: float(x.mean()) for r, x in zip('ADFC', (a, d_, f_, c_))}, mean_absdd={r: float(np.abs(x).mean()) for r, x in zip('ADFC', (a, d_, f_, c_))},
                     calls_changed={r: float((p0 != S[f'LG|{e}|{r}|p']).mean()) for r in 'ADFC'},
                     corr_feature_vs_spurion_route=float(np.corrcoef(fr, sr)[0, 1]), corr_F_vs_D_minus_C=float(np.corrcoef(f_, sr)[0, 1]),
                     mcnemar_A=mcnemar(p0 == y, S[f'LG|{e}|A|p'] == y), p_signflip_A=signflip(a, prng),
                     mean_dd_A_over_E4_match=float(a.mean() / dd('LG', Mt[e]['match'] if e.startswith('BOOSTBIG') else den_of(e)).mean()))
        rb_ = (C @ a) / N; rt[e]['mean_dd_A_ci'] = ci(rb_)
    for e in ('E4|BOOST+0.06', 'E4|BOOST-0.06', 'E1_+0.10', 'E1_-0.10', 'E4|E1_+0.10', 'E4|E1_-0.10'):
        rows = S[f'meta|{e}|rows']; a, d_, f_ = (dd('LG', e, r) * rows for r in 'ADF')
        bD, bF, bS = (C @ d_) / (C @ a), (C @ f_) / (C @ a), (C @ (d_ + f_)) / (C @ a)
        rt[e] = dict(R_D=float(d_.sum() / a.sum()), R_D_ci=ci(bD), R_F=float(f_.sum() / a.sum()), R_F_ci=ci(bF),
                     additivity=float((d_ + f_).sum() / a.sum()), additivity_ci=ci(bS), n=int(rows.sum()))
    res['routes'] = rt
    out = ROOT / 'p51_verify.json'; out.write_text(json.dumps(res, indent=1, default=float)); log('wrote', out)
    log(json.dumps(rt, default=float)[:3000])


if __name__ == '__main__':
    {'lgatr': stage_lgatr, 'part': stage_part, 'analyze': stage_analyze, 'sample': get_sample}[ARGS.stage]()
