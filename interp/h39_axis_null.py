"""H39 (loop D, claim D4m): direction part of the axis-convention response against architecture-matched nulls.

Question. H36 / H36v: recomputing Top-Landscape and quark-gluon (QG) tagger inputs about the winner-take-all (WTA) or
hardest-particle (HP) axis is exactly a rigid translation of every (deta_i, dphi_i) by the axis offset delta_j. The part of the
call change that depends on the offset pointing at the hardest particle is D = (% calls changed about the axis) - (% calls
changed by a translation of the same per-jet size |delta_j| in a random direction, averaged over directions). H36v found D in
trained networks (-1.7 to +4.0 pp) inside a band of weight-permuted ParT / PN / PFN networks (-3.0 to +3.2 pp), but permuted
EFN, PFN and PN were constant (degenerate) and there was one training seed. Here D of each trained network is compared with
networks of the same architecture that were not trained on the class labels:
  init   epoch-0 networks (NINIT per architecture; weaver ParT / ParticleNet built with torch.manual_seed(1000 + k); energyflow
         EFN / PFN built with seed k and saved untrained by interp/h39_train_ef.py --variant init, models/h39_init/)
  shuf   trained on labels independent of the class for the same number of steps (EFN / PFN: permuted labels; ParticleNet:
         labels = hash of (jet pT, eta, phi) mod 2; jobs d-h39-<ds>-ef, d-h39-<ds>-pn-shuf) -- included when finished
  s1     second training seed (seed spread) -- included when finished
H0: D(trained) is drawn from the architecture's null distribution (init + shuf networks). H1 (D4m, formula-driven reading):
D(trained) lies outside it, with the same sign in s0 and s1. Generic-translation reading: D(trained) inside, sign changing
between seeds.

Samples (random rows per class): Top Landscape test, data_top.sample(per_class=5000, seed 3901) (read through
h4_top_wmass._sample: the same rows); QG Pythia test, data_qg.sample(per_class=5000, seed 3911).
Conditions per jet: E (unedited), E2 (repeat: GPU floor), WTA, HP (inputs recomputed about the axis, h36_axis_top_qg paths),
R_k (k = 1..NR, translation by |delta_j| in a uniformly random direction, independent per k and jet; rng 39021 / 39022).
Calls: sign(log-odds) and the working points at signal efficiency eps_S = 0.3 and 0.5 (signal = top for Top, quark for QG;
threshold = the (1 - eps_S) quantile of the unedited signal-oriented score of this sample's signal jets, held fixed under edits).
Per model x axis x call: % changed about the axis, % changed under R (mean over k), per-jet d_j = 1[axis changes] - mean_k
1[R_k changes]; D = 100 mean d_j with a 95 % bootstrap interval over jets (B = 2000); p_jet (normal approximation) for D != 0.
Null per dataset x architecture x axis x call: D of the init (+ shuf) networks that are non-degenerate (unedited score std >
1e-6 and >= 0.5 % of calls changed under R). Tests of each trained network: z = (D - mean_null) / sd_null; p_pred = two-sided
t prediction-interval p (df = n_null - 1); p_emp = (1 + #{|D_null - mean| >= |D - mean|}) / (n_null + 1); Holm over models x
axes x datasets x calls. LorentzNet (Top, QG) takes four-vectors only: the axis does not enter (D = 0 by construction, not
re-scored); its unedited accuracy is reported.
Run: ./sync.sh && ./part_job.sh d-h39-run 'python -m interp.h39_axis_null'   (H39_SMOKE=1: 150 jets per class, 4 draws)
Outputs: $AI_RESULTS/physics/h39_axis_null.json, h39_perjet.npz, h39_axis_null.png
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h36_axis_top_qg as H36   # noqa: E402  (patches h5_qg_counting.kinematics with the 'ax' / 'sh' conditions)
from interp import h4_top_wmass as H4       # noqa: E402
from interp import h5_qg_counting as H5     # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H39_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h39_smoke' if SMOKE else 'h39'
PER = 150 if SMOKE else 5000
NR = 4 if SMOKE else 20
NINIT = 2 if SMOKE else 6
B = 300 if SMOKE else 2000
AX = ('WTA', 'HP')
CALLS = ('sign', 'wp30', 'wp50')
WP = dict(wp30=0.3, wp50=0.5)
V = H5.V
H36.PER_MAIN, H36.PER_CAL = PER, 0
H36.SEEDS = dict(top=3901, py=3911, hw=3911)
ARCHS = ('ParT', 'PN', 'PFN', 'EFN')


# ============================================================================================================== networks
def build_weaver(arch, seed=None):
    import torch
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from weaver.nn.model.ParticleNet import ParticleNet
    if seed is not None: torch.manual_seed(seed)
    if arch == 'ParT':
        return ParticleTransformer(input_dim=7, num_classes=2, pair_input_dim=4, use_pre_activation_pair=False, embed_dims=[128, 512, 128],
                                   pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8, num_cls_layers=2, block_params=None,
                                   cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0}, fc_params=[],
                                   activation='gelu', trim=True, for_inference=False)
    return ParticleNet(input_dims=7, num_classes=2, conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
                       fc_params=[(256, 0.1)], use_fusion=False, use_fts_bn=True, use_counts=True, for_inference=False)


def ef_from_npz(path, kind):
    """energyflow Keras weights (weights.npz, kernels (in, out)) -> torch EF module of h5_qg_counting.make_ef (as
    h14_qg_generator.add_keras_ef: the module divides angles by 0.4, so the first kernel's angle rows are multiplied by 0.4)"""
    import torch
    W = np.load(path); ks = sorted(W.files)
    kern = [W[k] for k in ks if k.endswith('_0')]; bias = [W[k] for k in ks if k.endswith('_1')]
    net = H5.make_ef(kind); lin = [m for m in list(net.phi) + list(net.F) if isinstance(m, torch.nn.Linear)]
    assert len(lin) == len(kern) == 7, (len(lin), len(kern))
    with torch.no_grad():
        for i, (l, k, b) in enumerate(zip(lin, kern, bias)):
            k = k.copy()
            if i == 0: k[-2:] *= 0.4
            assert l.weight.shape == k.T.shape, (i, l.weight.shape, k.shape)
            l.weight.copy_(torch.from_numpy(k.T)); l.bias.copy_(torch.from_numpy(b))
    return net


def _weaver_done(d, var):
    if var == 'shuf': return (d / 'net_epoch-19_state.pt').exists(), d / 'net_epoch-19_state.pt'
    return (d / 'predict_output').exists() or (d / 'net_epoch-19_state.pt').exists(), d / 'net_best_epoch_state.pt'


def registry(ds, dev):
    """list of dict(name, arch, var, kind ('weaver' | 'ef'), net, info) for one dataset"""
    import torch
    E = []; dsd = 'top' if ds == 'top' else 'qg'

    def add_w(name, arch, var, ckpt=None, seed=None, info=None):
        net = build_weaver(arch, seed)
        if ckpt is not None: net.load_state_dict(H4._strip(torch.load(ckpt, map_location='cpu')), strict=True)
        E.append(dict(name=name, arch=arch, var=var, kind='weaver', net=net.eval().to(dev), info=info or dict(checkpoint=str(ckpt), seed=seed)))

    def add_ef(name, arch, var, path, info=None):
        E.append(dict(name=name, arch=arch, var=var, kind='ef', net=ef_from_npz(path, arch.lower()).eval().to(dev), info=info or dict(weights=str(path))))
    # trained seed 0 (the H36 models)
    if ds == 'top':
        add_w('ParT_s0', 'ParT', 's0', RES / 'h4_ckpt_part_kin.pt'); add_w('PN_s0', 'PN', 's0', RES / 'h4_ckpt_pnet_kin.pt')
    else:
        add_w('ParT_s0', 'ParT', 's0', V / 'models/qg/part_kin/net_best_epoch_state.pt')
        add_w('PN_s0', 'PN', 's0', V / 'models/qg/pnet_kin/net_best_epoch_state.pt')
    for a in ('PFN', 'EFN'):
        d = V / 'models' / dsd / a.lower(); add_ef(f'{a}_s0', a, 's0', d / 'weights.npz', json.loads((d / 'info.json').read_text()))
    # trained variants (included when finished)
    for arch, wd in (('PN', 'pnet_kin'), ('ParT', 'part_kin')):
        for var in ('s1', 'shuf'):
            d = V / 'models' / dsd / f'{wd}_{var}'
            if not d.exists(): continue
            ok, ck = _weaver_done(d, var)
            if ok and ck.exists(): add_w(f'{arch}_{var}', arch, var, ck, info=dict(checkpoint=str(ck)))
            else: log(f'{ds}: {d.name} not finished ({len(list(d.glob("net_epoch-*_state.pt")))} epoch files): not included')
    for a in ('PFN', 'EFN'):
        for var in ('s1', 'shuf'):
            d = V / 'models' / dsd / f'{a.lower()}_{var}'
            if (d / 'info.json').exists(): add_ef(f'{a}_{var}', a, var, d / 'weights.npz', json.loads((d / 'info.json').read_text()))
            elif d.exists(): log(f'{ds}: {d.name} not finished: not included')
    # epoch-0 networks
    for k in range(NINIT):
        for arch in ('ParT', 'PN'): add_w(f'{arch}_init{k}', arch, 'init', seed=1000 + k)
        for a in ('PFN', 'EFN'):
            p = V / 'models' / 'h39_init' / f'{a.lower()}_init{k}' / 'weights.npz'
            if p.exists(): add_ef(f'{a}_init{k}', a, 'init', p)
            else: log(f'missing {p}')
    log(f'{ds}: {len(E)} networks: ' + ', '.join(e['name'] for e in E))
    return E


# ============================================================================================================== scoring
def score_top(E, P4, nn, cond, dev):
    """top-positive log-odds per network (weaver: class 0 = top; EF: label 1 = top)"""
    import torch
    out = {e['name']: np.zeros(len(nn)) for e in E}; d = lambda a: torch.from_numpy(a).to(dev)
    with torch.no_grad():
        for s in range(0, len(nn), 2000):
            sl = slice(s, s + 2000); c = None if cond is None else (cond[0], cond[1][sl], cond[2][sl])
            J = H36.top_J(P4[sl], nn[sl], c)
            for b in range(0, len(nn[sl]), 500):
                Jb = {k: (v[b:b + 500] if np.ndim(v) else v) for k, v in J.items()}; pts, x, v, m = H4.weaver_inputs(Jb)
                for e in E:
                    if e['kind'] != 'weaver': continue
                    o = (e['net'](d(x), v=d(v), mask=d(m)) if e['arch'] == 'ParT' else e['net'](d(pts), d(x), d(m))).float().cpu().numpy()
                    out[e['name']][s + b:s + b + len(o)] = o[:, 0] - o[:, 1]
            k3 = np.stack([J['pt'][:, :128], J['deta'][:, :128], J['dphi'][:, :128]], -1).astype(np.float16).astype(np.float32)
            pt = k3[..., 0]; real = pt > 0; z = (pt / np.maximum(pt.sum(1, keepdims=True), 1e-12)).astype(np.float32)
            ang = (k3[..., 1:3] * real[..., None]).astype(np.float32); zt, at = d(z), d(ang)
            for e in E:
                if e['kind'] != 'ef': continue
                o = e['net'](zt, at).float().cpu().numpy(); out[e['name']][sl] = o[:, 1] - o[:, 0]
    return out


def score_qg(E, Jt, cond, dev):
    """gluon-positive log-odds per network (weaver labels [isQ, isG]; EF label 1 = quark), as h5_qg_counting.Models.logit"""
    import torch
    J = len(Jt['n']); out = {e['name']: np.zeros(J) for e in E}; d = lambda a: torch.from_numpy(a).to(dev)
    with torch.no_grad():
        for s in range(0, J, 2000):
            rows = np.arange(s, min(J, s + 2000)); Jc = dict(H36.sub_jt(Jt, rows)); Jc['_cond'] = None if cond is None else (cond[0], cond[1][rows], cond[2][rows])
            P, X, Vv, M = H5.weaver_inputs(Jc, False)
            for b in range(0, len(rows), 500):
                a = [d(np.ascontiguousarray(q[b:b + 500])) for q in (P, X, Vv, M)]
                for e in E:
                    if e['kind'] != 'weaver': continue
                    o = e['net'](a[1], v=a[2], mask=a[3].bool()) if e['arch'] == 'ParT' else e['net'](a[0], a[1], a[3])
                    o = torch.log_softmax(o.float(), -1); out[e['name']][s + b:s + b + len(o)] = (o[:, 1] - o[:, 0]).cpu().numpy()
            Z, A = H5.ef_inputs(Jc); zt, at = d(Z), d(A)
            for e in E:
                if e['kind'] != 'ef': continue
                o = torch.log_softmax(e['net'](zt, at).float(), -1); out[e['name']][rows] = (o[:, 0] - o[:, 1]).cpu().numpy()
    return out


# ============================================================================================================== statistics
def holm(ps):
    return H4._holm(list(ps))


def null_test(D, vals):
    from scipy.stats import t as tdist
    v = np.asarray(vals, float); n = len(v)
    if n < 2: return dict(n_null=n)
    m, s = v.mean(), v.std(ddof=1)
    z = (D - m) / s if s > 0 else None
    tp = float(2 * tdist.sf(abs(D - m) / (s * np.sqrt(1 + 1 / n)), n - 1)) if s > 0 else (1.0 if D == m else 0.0)
    pe = float((1 + np.sum(np.abs(v - m) >= abs(D - m))) / (n + 1))
    return dict(n_null=n, null_mean=float(m), null_sd=float(s), null_min=float(v.min()), null_max=float(v.max()),
                z=float(z) if z is not None else None, p_pred=tp, p_emp=pe, outside_range=bool(D < v.min() or D > v.max()))


def analyse(ds, S, E, sc):
    from scipy.stats import norm
    y = S['y']; sgn = (lambda l: l) if ds == 'top' else (lambda l: -l)     # signal-oriented score (top / quark positive)
    ytgt = (y == 1) if ds == 'top' else (y == 0)                          # class the positive log-odds points to
    J = len(y); bi = np.random.default_rng(39051).integers(0, J, (B, J)).astype(np.int32)
    ci = lambda x: (np.percentile(x[bi].mean(1), [2.5, 97.5]) * 100).tolist()
    out = dict(n=int(J), n_by_class=np.bincount(y).tolist(), axinfo=S['axinfo'],
               offsets={a: dict(median_mrad=float(1e3 * np.median(np.hypot(*S['off'][a]))),
                                by_class={str(c): float(1e3 * np.median(np.hypot(*S['off'][a])[y == c])) for c in (0, 1)}) for a in AX},
               models={})
    pj = {}
    for e in E:
        nm = e['name']; l0 = sc['E'][nm]
        ok = np.isfinite(l0).all()
        M = dict(arch=e['arch'], variant=e['var'], info=e['info'], acc=float(np.mean((l0 > 0) == ytgt)),
                 auc=float(H4._auc(sgn(l0), (y == 1).astype(int))), score_std=float(np.std(l0)), score_mean=float(np.mean(l0)),
                 frac_positive=float(np.mean(l0 > 0)), repeat_max_abs_dl=float(np.max(np.abs(sc['E2'][nm] - l0))), calls={})
        thr = {c: float(np.quantile(sgn(l0)[y == 1], 1 - WP[c])) for c in WP}
        M['wp_threshold'] = thr
        M['bkg_rejection'] = {c: (float(1 / np.mean(sgn(l0)[y == 0] >= thr[c])) if np.mean(sgn(l0)[y == 0] >= thr[c]) > 0 else None) for c in WP}
        call = {'sign': lambda l: l > 0, **{c: (lambda l, t=thr[c]: sgn(l) >= t) for c in WP}}
        for c in CALLS:
            c0 = call[c](l0); Mc = dict(repeat_pct_changed=float(100 * np.mean(call[c](sc['E2'][nm]) != c0)), axes={})
            for a in AX:
                ch = (call[c](sc[a][nm]) != c0).astype(float)
                pr = np.mean([call[c](sc[f'R{k}_{a}'][nm]) != c0 for k in range(NR)], 0)
                dd = ch - pr; se = dd.std(ddof=1) / np.sqrt(J)
                Mc['axes'][a] = dict(pct_axis=float(100 * ch.mean()), ci_axis=ci(ch), pct_R=float(100 * pr.mean()), ci_R=ci(pr),
                                     D_pp=float(100 * dd.mean()), D_ci=ci(dd),
                                     p_jet=float(2 * norm.sf(abs(dd.mean()) / se)) if se > 0 else 1.0,
                                     ratio_axis_over_R=float(ch.mean() / pr.mean()) if pr.mean() > 0 else None,
                                     by_class={str(k): dict(axis=float(100 * ch[y == k].mean()), R=float(100 * pr[y == k].mean())) for k in (0, 1)},
                                     nondegenerate=bool(ok and np.std(l0) > 1e-6 and 100 * pr.mean() >= 0.5))
                if c == 'sign':
                    Mc['axes'][a]['acc_after'] = float(np.mean((sc[a][nm] > 0) == ytgt))
                pj[f'{ds}|d|{c}|{a}|{nm}'] = dd.astype(np.float32)
            M['calls'][c] = Mc
        out['models'][nm] = M
        log(f'[{ds}] {nm:10s} acc {M["acc"]:.4f} AUC {M["auc"]:.4f} std {M["score_std"]:.3g} | ' + ' | '.join(
            f'{c} ' + ' '.join(f'{a} {M["calls"][c]["axes"][a]["pct_axis"]:.2f}/{M["calls"][c]["axes"][a]["pct_R"]:.2f} D {M["calls"][c]["axes"][a]["D_pp"]:+.2f}'
                               for a in AX) for c in CALLS))
    # nulls per architecture and tests of the trained networks
    out['nulls'] = {}
    for arch in ARCHS:
        for c in CALLS:
            for a in AX:
                vals = {nm: M['calls'][c]['axes'][a]['D_pp'] for nm, M in out['models'].items()
                        if M['arch'] == arch and M['variant'] in ('init', 'shuf') and M['calls'][c]['axes'][a]['nondegenerate']}
                allv = {nm: M['calls'][c]['axes'][a]['D_pp'] for nm, M in out['models'].items() if M['arch'] == arch and M['variant'] in ('init', 'shuf')}
                out['nulls'][f'{arch}|{c}|{a}'] = dict(nondegenerate=vals, all=allv)
                for nm, M in out['models'].items():
                    if M['arch'] == arch and M['variant'] in ('s0', 's1'):
                        M['calls'][c]['axes'][a]['vs_null'] = null_test(M['calls'][c]['axes'][a]['D_pp'], list(vals.values()))
        s0, s1 = out['models'].get(f'{arch}_s0'), out['models'].get(f'{arch}_s1')
        if s0 and s1:
            for c in CALLS:
                for a in AX:
                    d0, d1 = s0['calls'][c]['axes'][a]['D_pp'], s1['calls'][c]['axes'][a]['D_pp']
                    s0['calls'][c]['axes'][a]['seed_spread'] = dict(D_s1=d1, diff_s1_minus_s0=d1 - d0, same_sign=bool(np.sign(d0) == np.sign(d1)))
    for c in CALLS:
        for a in AX:
            log(f'[{ds}] null {c} {a}: ' + '; '.join(f'{arch} n={len(out["nulls"][f"{arch}|{c}|{a}"]["nondegenerate"])} '
                                                     f'{np.round(sorted(out["nulls"][f"{arch}|{c}|{a}"]["nondegenerate"].values()), 2).tolist()}' for arch in ARCHS))
    return out, pj


# ============================================================================================================== main
def run_ds(ds, S, dev):
    import torch
    if ds == 'top':
        def scorer(cond, E): return score_top(E, S['P4'], S['nn'], cond, dev)
    else:
        def scorer(cond, E): return score_qg(E, S['Jt'], cond, dev)
    J = len(S['y']); log(f'{ds}: {J} jets {np.bincount(S["y"]).tolist()}; axes {S["axinfo"]}')
    E = registry(ds, dev)
    sc = {'E': scorer(None, E), 'E2': scorer(None, E)}; log(f'[{ds}] E, E2 scored')
    sg = np.where(S['jeta'] < 0, -1.0, 1.0); rng = np.random.default_rng(39021 if ds == 'top' else 39022)
    chk = {}
    if ds == 'top':   # implementation check: torch EF port vs the numpy H4.Flow used in H36 (unedited, s0)
        k3 = np.stack([H36.top_J(S['P4'], S['nn'], None)[k][:, :128] for k in ('pt', 'deta', 'dphi')], -1).astype(np.float16).astype(np.float32)
        for a in ('PFN', 'EFN'):
            f = H4.Flow(a.lower()); chk[f'{a}_s0_max_abs_vs_numpy_flow'] = float(np.max(np.abs(f.score(k3) - sc['E'][f'{a}_s0'])))
        log('EF port check', chk)
    for a in AX:
        oe, op = S['off'][a]; r = np.hypot(oe, op)
        sc[a] = scorer(('ax', S['A'][a][0], S['A'][a][1]), E)
        sc[f'TN_{a}'] = scorer(('sh', -sg * oe, -op), [e for e in E if e['var'] == 's0'])    # same shift through the stored arrays
        chk[f'max_abs_axis_minus_TN_{a}'] = float(max(np.max(np.abs(sc[a][k] - v)) for k, v in sc[f'TN_{a}'].items()))
        for k in range(NR):
            th = rng.uniform(0, 2 * np.pi, J); sc[f'R{k}_{a}'] = scorer(('sh', r * np.cos(th), r * np.sin(th)), E)
        log(f'[{ds}] {a}: axis, TN, {NR} random-direction shifts scored; max |l_axis - l_TN| {chk[f"max_abs_axis_minus_TN_{a}"]:.2e}')
    res, pj = analyse(ds, S, E, sc)
    res['checks'] = chk; res['sample'] = dict(seed=H36.SEEDS[ds], per_class=PER, idx_first5=S['idx'][:5].tolist())
    for k in ('E', 'E2', 'WTA', 'HP'):
        for nm, v in sc[k].items(): pj[f'{ds}|{k}|{nm}'] = np.asarray(v, np.float32)
    pj[f'{ds}|y'] = S['y']; pj[f'{ds}|idx'] = S['idx']
    for a in AX: pj[f'{ds}|r_{a}'] = np.hypot(*S['off'][a]).astype(np.float32)
    # unedited LorentzNet accuracy (axis does not enter its inputs: D = 0 by construction)
    try:
        if ds == 'top':
            ln = H4.LNet(dev, mass64=True).score(S['P4'], S['nn'], np.random.default_rng(605).permutation(J)); acc = float(np.mean((ln > 0) == (S['y'] == 1)))
            auc = float(H4._auc(ln, S['y']))
        else:
            M = H5.Models(dev); M.add_lnet(); Jt = dict(S['Jt']); Jt['_cond'] = None; ln = M.logit('lorentznet', Jt)
            acc = float(np.mean((ln > 0) == (S['y'] == 0))); auc = float(H4._auc(-ln, S['y']))
        res['lorentznet'] = dict(acc=acc, auc=auc, D='0 by construction: four-vector inputs, the axis does not enter')
        log(f'[{ds}] LorentzNet unedited acc {acc:.4f} AUC {auc:.4f}')
    except Exception as ex:
        log(f'[{ds}] LorentzNet failed: {ex!r}'); res['lorentznet'] = dict(error=repr(ex))
    del E
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return res, pj


def main():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'    # (torch.cuda.is_available does not create a CUDA context)
    RES.mkdir(parents=True, exist_ok=True)
    log(f'H39 start; smoke={SMOKE}; {PER}/class, {NR} random directions, {NINIT} inits per architecture; device {dev}')
    R = dict(node='H39', smoke=SMOKE, per_class=PER, n_random_directions=NR, n_init=NINIT, B=B, calls=list(CALLS), datasets={}); PJ = {}
    SS = {'top': H36.load_top(), 'py': H36.load_qg('py')}     # samples and axes (multiprocessing) before any network is on the GPU
    for ds in ('top', 'py'):
        R['datasets'][ds], pj = run_ds(ds, SS[ds], dev); PJ.update(pj)
        (RES / f'{TAG}_axis_null.json').write_text(json.dumps(R, indent=1, default=float))
    cells = [(ds, nm, c, a) for ds, D in R['datasets'].items() for nm, M in D['models'].items() if M['variant'] in ('s0', 's1')
             for c in CALLS for a in AX if 'p_pred' in M['calls'][c]['axes'][a].get('vs_null', {})]
    for key in ('p_pred', 'p_emp'):
        for (ds, nm, c, a), ph in zip(cells, holm([R['datasets'][ds]['models'][nm]['calls'][c]['axes'][a]['vs_null'][key] for ds, nm, c, a in cells])):
            R['datasets'][ds]['models'][nm]['calls'][c]['axes'][a]['vs_null'][f'holm_{key}'] = float(ph)
    R['holm_family'] = f'{len(cells)} cells (trained networks x calls x axes x datasets)'
    (RES / f'{TAG}_axis_null.json').write_text(json.dumps(R, indent=1, default=float)); np.savez_compressed(RES / f'{TAG}_perjet.npz', **PJ)
    log(f'wrote {TAG}_axis_null.json ({R["holm_family"]})')
    try: fig(R)
    except Exception as ex: log('figure failed:', repr(ex))


def fig(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dsn = {'top': 'Top Landscape (top vs QCD)', 'py': 'Quark–gluon (Pythia)'}
    cn = {'wp50': 'calls at 50 % signal efficiency', 'sign': 'calls = sign of log-odds'}
    f, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharey='row')
    for i, c in enumerate(('wp50', 'sign')):
        for j, ds in enumerate(('top', 'py')):
            ax = axs[i, j]; D = R['datasets'].get(ds)
            if D is None: continue
            for xi, arch in enumerate(ARCHS):
                for dx, a, mk in ((-0.17, 'WTA', 'o'), (0.17, 'HP', 's')):
                    x = xi + dx; nv = D['nulls'][f'{arch}|{c}|{a}']
                    vals = list(nv['nondegenerate'].values()); deg = [v for k, v in nv['all'].items() if k not in nv['nondegenerate']]
                    if vals: ax.add_patch(plt.Rectangle((x - 0.12, min(vals)), 0.24, max(vals) - min(vals) + 1e-9, color='0.85', zorder=0))
                    ax.plot([x] * len(vals), vals, ls='none', marker=mk, mfc='none', mec='0.45', ms=5, zorder=2)
                    ax.plot([x] * len(deg), deg, ls='none', marker='x', color='0.7', ms=4, zorder=2)
                    for var, col in (('s0', 'C3'), ('s1', 'C0')):
                        M = D['models'].get(f'{arch}_{var}')
                        if M is None: continue
                        v = M['calls'][c]['axes'][a]; xo = x + (0.06 if var == 's1' else 0)
                        ax.errorbar(xo, v['D_pp'], yerr=[[v['D_pp'] - v['D_ci'][0]], [v['D_ci'][1] - v['D_pp']]], fmt=mk, color=col, ms=6, capsize=2, zorder=3)
            ax.axhline(0, color='0.5', lw=0.7)
            ax.set_xticks(range(len(ARCHS))); ax.set_xticklabels(['ParT', 'ParticleNet', 'PFN', 'EFN'])
            ax.set_title(f'{dsn[ds]}, {cn[c]}', fontsize=9, loc='left')
            if j == 0: ax.set_ylabel('axis shift − random-direction shift\n(percentage points of calls changed)', fontsize=9)
    h = [plt.Line2D([], [], color='C3', marker='o', ls='none', label='trained, seed 0 (95 % CI)'),
         plt.Line2D([], [], color='0.45', marker='o', mfc='none', ls='none', label='same architecture, not trained on labels'),
         plt.Line2D([], [], color='0.7', marker='x', ls='none', label='… degenerate (< 0.5 % calls move)'),
         plt.Line2D([], [], color='k', marker='o', ls='none', mfc='none', label='circle: WTA axis'),
         plt.Line2D([], [], color='k', marker='s', ls='none', mfc='none', label='square: hardest-particle axis')]
    if any(f'{a}_s1' in D['models'] for D in R['datasets'].values() for a in ARCHS):
        h.insert(1, plt.Line2D([], [], color='C0', marker='o', ls='none', label='trained, seed 1'))
    axs[0, 1].legend(handles=h, fontsize=7, loc='upper left')
    f.suptitle('Direction part of the axis response of trained taggers vs untrained networks of the same architecture '
               '(grey band: range of the untrained networks; LorentzNet = 0 by construction)', fontsize=9.5)
    f.tight_layout(); f.savefig(RES / f'{TAG}_axis_null.png', dpi=130); log(f'wrote {TAG}_axis_null.png')


if __name__ == '__main__':
    if '--fig' in sys.argv: fig(json.loads((RES / f'{TAG}_axis_null.json').read_text()))
    else: main()
