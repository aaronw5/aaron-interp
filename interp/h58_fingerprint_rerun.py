"""H58 (loop D, round D12): completion of the H55v classical-fingerprint rerun for claim D23 ("JetClass / Top networks change
calls under the nine-edit library most like groomed-mass + substructure taggers, hls4ml networks most like a plain mass window").

What is new relative to H55 / H55v
  * fresh rows, disjoint from H55 / H55v: hls4ml 3,000 jets per class drawn with default_rng(5801) from the full_test.npz rows
    not used by H55 (seed 5501) or H55v (seed 5591); JetClass Hbb/Wqq/Zqq/Tbqq/QCD from the *_102.root test files (H55 / H55v
    used *_100), 3,000 random rows each (default_rng 5802); Top Landscape and QG test splits, 3,000 per class (default_rng 5803 /
    5804) with the H55 rows (5503 / 5504) excluded.
  * the same nine edits and observables (interp.h55_classical_fingerprint make_conds / observables, unchanged).
  * classical panel extended with an m_SD-only window and an ungroomed-m-only window, and every window / shape tagger also at a
    matched working point of 50 % signal efficiency on the task's unedited signal jets ("@50"):
      msd_V / msd_top      65 < m_SD < 105 GeV / 105 < m_SD < 210 GeV (published-window versions)
      msd_*@50, mass_*@50  shortest m_SD (resp. ungroomed m) interval containing 50 % of the unedited signal jets
      V_tau21@50, V_DDT@50, V_D2@50, top_tau32@50
                           published m_SD window and the shape variable (tau21, tau21 + 0.063 rho', D2, tau32) cut at the value
                           that gives 50 % total signal efficiency; if the published window holds < 55 % of the signal, the
                           window is replaced by the shortest m_SD interval holding 70 % and the shape cut set to reach 50 %.
    The main analysis uses the @50 rows (networks are also at 50 %), so the working point is matched; the published cuts are a
    secondary analysis.
  * null: >= 55 weight-permuted copies per architecture (every trained network, LorentzNet included); copies whose unedited task
    log-odds are constant over jets (std < 1e-6) are rejected at screening and more seeds drawn (up to MAXTRY); in the statistics,
    copies whose calls never change under any edit on a task are dropped for that task. Exact rank p = (1 + #{perm >= net}) /
    (n_perm + 1).
  * main test per network x task (W / Z / top vs QCD): paired jet bootstrap (B = 50,000 multinomial resamples; CIs from the same
    draws) of
      d_shape = cos(net, shape@50) - cos(net, msd@50)          shape = V_tau21@50 (W, Z), top_tau32@50 (top)  [pre-specified]
      d_groom = cos(net, msd@50)   - cos(net, mass@50)
      d_sm    = cos(net, shape@50) - cos(net, mass@50)          (= d_shape + d_groom; the D23 headline contrast)
    H0: d = 0; two-sided bootstrap p; Holm over all network x task x contrast tests, BH secondary.
  * leave-one-edit-out (nearest @50 tagger and the sign of each contrast with each edit's two columns removed), unstandardised
    cosine, best-of-shape contrast with the selection redone in every draw.
Fingerprint (as H55): per row (network / permuted network / classical tagger) and edit, chg = fraction of calls changed and
  flow = (calls gained - lost) / jets, on the task's jets where the edit applies; vector (chg_e, flow_e) over the dataset's edits,
  each component divided by its RMS over the trained networks + classical rows of the analysis (recomputed in every bootstrap
  draw); cosine similarity between rows. Networks call signal above the median unedited signal log-odds (50 % efficiency).
Run (results under $AI_RESULTS/physics/h58_work/, final $AI_RESULTS/physics/h58_fingerprint.json, h58_contrasts.png)
  AI_PY=conda ./job.sh d-h58-edit 4 12 'python -m interp.h58_fingerprint_rerun --part edits --ds hls && ... --ds jc && ... --ds top && ... --ds qg'
  ./job.sh d-h58-jedi 1 4 'python -m interp.h58_fingerprint_rerun --part nets --group hls_jedi'
  AI_PY=conda ./job.sh d-h58-<arch> 1 4 'python -m interp.h58_fingerprint_rerun --part nets --group hls_<dsets|linf|tf|jedis>'
  ./part_job.sh d-h58-<group> 'python -m interp.h58_fingerprint_rerun --part nets --group <jc_part|jc_pn|top_main|top_lnet|qg_main|qg_lnet>'
  AI_PY=conda ./job.sh d-h58-stats 2 8 'python -m interp.h58_fingerprint_rerun --part stats'
  --smoke N: N jets per class, 3 permuted copies per architecture, suffix _smoke.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h55_classical_fingerprint as H  # noqa: E402

PHYS = H.RES / 'physics'
WORK = PHYS / 'h58_work'
H.OUT = WORK
H.SEEDS.update(hls=5801, jc=5802, top=5803, qg=5804)
H.PER.update(hls=3000, jc=3000, top=3000, qg=3000)
OLD = dict(hls=[(5501, 2000), (5591, 2000)], top=[(5503, 2000)], qg=[(5504, 2000)])
TEST20M = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
V = H.V
NKEEP, MAXTRY, PSEED0 = 55, dict(default=160, JEDI=300), 5800
B_MAIN, B_PUB, CHUNK = 50000, 2000, 1000
log = H.log
SMOKE = 0


def sfx():
    return '_smoke' if SMOKE else ''


# ================================================================================================================ samples
def load_ds_h58(ds, per):
    seed = H.SEEDS[ds]
    if ds == 'hls':
        from jetdistill.config import RESULTS
        from interp.h1_irc_panel import hls_table
        d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz')
        y = np.asarray(d['y']); excl = []
        for s, p in OLD['hls']:
            r = np.random.default_rng(s); excl.append(np.concatenate([r.choice(np.flatnonzero(y == c), p, replace=False) for c in range(5)]))
        excl = np.unique(np.concatenate(excl)); rng = np.random.default_rng(seed)
        idx = np.sort(np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(y == c), excl), per, replace=False) for c in range(5)]))
        assert not np.isin(idx, excl).any()
        x = np.asarray(d['x'], np.float32)[idx]; log(f'hls4ml: {len(idx)} rows, none of the {len(excl)} H55 / H55v rows')
        return hls_table(x), y[idx].astype(int), dict(
            sample=f'hls4ml full_test.npz (as interp.jedi_core.sample), {per} per class, default_rng({seed}) over rows not drawn by '
                   f'H55 (seed 5501) or H55v (seed 5591)', classes=['g', 'q', 'W', 'Z', 't'])
    if ds == 'jc':
        import uproot
        from interp.part_core import read_root, cat, sub, PRE, CL
        from interp.h1_irc_panel import jc_table
        rng = np.random.default_rng(seed); Js = []; files = {}
        for c in H.JC_CL:
            f = TEST20M / f'{PRE[c]}_102.root'; n = uproot.open(f)['tree'].num_entries
            idx = np.sort(rng.choice(n, per, replace=False)); parts = []
            for a in range(0, n, 20000):
                loc = idx[(idx >= a) & (idx < a + 20000)] - a
                if len(loc): parts.append(sub(read_root(str(f), start=a, stop=min(a + 20000, n)), loc))
            Js.append(cat(parts)); files[c] = f.name; log(f'JetClass {c}: {per} random rows of {n} ({f.name})')
        J0 = cat(Js); T = jc_table(J0); y = J0['label'].astype(int)
        return T, y, dict(sample=f'JetClass test_20M *_102.root per class {H.JC_CL}, {per} random rows each (default_rng {seed})',
                          files=files, classes=list(CL))
    from interp.h12_boost_panel import table_from_full
    D = __import__('interp.data_top' if ds == 'top' else 'interp.data_qg', fromlist=['DST'])
    d = D.DST / 'test'; ylab = np.load(d / 'label.npy')
    old = np.concatenate([H._rows(ylab, p, s) for s, p in OLD[ds]]); rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(ylab == c), old), per, replace=False) for c in (0, 1)]))
    t = time.time(); F = np.asarray(np.load(d / 'full.npy', mmap_mode='r')[idx]); jet = np.asarray(np.load(d / 'jet.npy', mmap_mode='r')[idx])
    log(f'{ds} test rows {len(idx)} read ({time.time() - t:.0f} s), none of the {len(old)} H55 rows')
    nn = np.minimum(jet[:, 5].astype(np.int64), F.shape[1]); y = ylab[idx].astype(int); T = table_from_full(F, nn); del F
    if ds == 'qg':
        T['m'] = np.zeros_like(T['m']); pid = np.asarray(np.load(d / 'pid.npy', mmap_mode='r')[idx]).astype(np.float64)
        T['ext'] = dict(pid=pid * (T['pt'] > 0)); H.CAP['qg'] = T['pt'].shape[1]
    T['sgn'] = np.where(T['ax_eta'] < 0, -1.0, 1.0)
    return T, y, dict(sample=f'{ds} test split, {per} per class random rows (default_rng {seed}), H55 rows (default_rng '
                             f'{OLD[ds][0][0]}) excluded', classes=['QCD', 'top'] if ds == 'top' else ['gluon', 'quark'])


# ================================================================================================================ nets
def tasks_of(ds):
    return H.DS_TASKS[ds] or {k: v for k, v in H.jc_tasks().items() if k != 'Hbb_vs_QCD'}


def task_lo(L, ds):
    return {t: H.logodds(L, s, b) for t, (s, b) in tasks_of(ds).items()}


def run_group(ds, group, trained, factories, prep, known):
    """trained {name: f(X) -> logits}; factories {arch: g(seed) -> f}; screened permuted copies; per-task log-odds saved"""
    tabs, y = H.load_tab(WORK / f'h55_{ds}_tab{sfx()}.npz')
    t = time.time(); Xb = prep(tabs['base'][0]); log(f'[{group}] base inputs ({time.time() - t:.0f} s)')
    models = dict(trained); base = {}; screen = {}
    nkeep = 3 if SMOKE else NKEEP
    for arch, fac in factories.items():
        kept = tried = rej = 0; lim = MAXTRY.get(arch, MAXTRY['default']) if not SMOKE else 6; t = time.time()
        while kept < nkeep and tried < lim:
            s = PSEED0 + tried; tried += 1
            f = fac(s); L = np.asarray(f(Xb), np.float64); lo = task_lo(L, ds)
            if all(np.isfinite(v).all() and np.std(v) > 1e-6 for v in lo.values()):
                nm = f'{arch}_perm{s}'; models[nm] = f; base[nm] = L; kept += 1
            else:
                rej += 1; del f
        screen[arch] = dict(tried=tried, kept=kept, rejected_constant_output=rej, seeds_from=PSEED0)
        log(f'[{group}] {arch}: {kept} permuted copies kept of {tried} tried ({rej} constant output) ({time.time() - t:.0f} s)')
    Z = {}; acc = dict(_known=known, _screen=screen)
    for c, (T, ok) in tabs.items():
        t = time.time(); X = Xb if c == 'base' else prep(T)
        for mn, f in models.items():
            L = base[mn] if (c == 'base' and mn in base) else np.asarray(f(X), np.float64)
            if c == 'base':
                if L.shape[1] > 2:
                    c0 = L.argmax(1); acc[mn] = dict(acc=float((c0 == y).mean()), per_class={int(k): float((c0[y == k] == k).mean()) for k in np.unique(y)})
                else:
                    from sklearn.metrics import roc_auc_score
                    s_ = L[:, 1] - L[:, 0]; acc[mn] = dict(acc=float(((s_ > 0).astype(int) == y).mean()), auc=float(roc_auc_score(y, s_)))
            for tk, v in task_lo(L, ds).items(): Z[f'{mn}|{c}|{tk}'] = v.astype(np.float32)
        log(f'  [{group}] {c}: {len(models)} models scored ({time.time() - t:.0f} s)')
    for mn in trained: log(f'  [{group}] {mn}: unedited accuracy {acc[mn]["acc"]:.4f} {acc[mn].get("per_class") or acc[mn].get("auc")}')
    pa = [v['acc'] for k, v in acc.items() if '_perm' in k]
    if pa: log(f'  [{group}] permuted copies: accuracy {min(pa):.3f}-{max(pa):.3f}')
    np.savez(WORK / f'h58_{ds}_{group}_lo{sfx()}.npz', **Z)
    (WORK / f'h58_{ds}_{group}_acc{sfx()}.json').write_text(json.dumps(acc, indent=1, default=float))


def g_hls_jedi():
    from interp.jedi_core import Jedi
    from jetdistill import network as N
    from interp.h12_boost_panel import permute_weights

    def fac(s):
        orig = N.redraw_untrained; N.redraw_untrained = permute_weights
        try: n = Jedi(64, '3-feature', untrained=True, seed=s)
        finally: N.redraw_untrained = orig
        return lambda x, n=n: np.asarray(n.run(x)['logits'], np.float64)
    n0 = Jedi(64, '3-feature')
    run_group('hls', 'hls_jedi', {'JEDI': lambda x: np.asarray(n0.run(x)['logits'], np.float64)}, {'JEDI': fac}, H.hls_x,
              dict(JEDI='5-class acc 0.8092 (H55 sample; 260k test 0.8092)'))


SMALL_RUNS = dict(dsets='dsets-d64-n64', linf='linf-2b-d32-k8-n64', tf='tf-2b-d64-n64-e100', jedis='jedis-d64-n64')


def g_hls_small(tag):
    import torch
    from interp.h1_irc_panel import load_small, torch_logits
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    rd = V / 'models' / 'small' / SMALL_RUNS[tag] / 'seed0'; net, sh, sc = load_small(rd)

    def fac(s):
        n2, _, _ = load_small(rd); H.permute_torch(n2, s); return lambda x, n2=n2: torch_logits(n2, x, sh, sc)
    known = {tag: json.loads((rd / 'test.json').read_text())['accuracy']}
    run_group('hls', f'hls_{tag}', {tag: lambda x: torch_logits(net, x, sh, sc)}, {tag: fac}, H.hls_x, known)


def jc_prep(T):
    from interp.part_core import build
    sg = np.where(T['ax_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = H.rel(T)
    Jx = dict(jet_eta=T['ax_eta'].astype(np.float32), jet_phi=T['ax_phi'].astype(np.float32), label=np.zeros(len(de), np.int64))
    return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])


_IC = {}


def cached_inputs(J, n):
    from interp.part_core import inputs
    if _IC.get('J') is not J: _IC.clear(); _IC['J'] = J
    if n not in _IC: _IC[n] = inputs(J, n)
    return _IC[n]


def pn_logits_c(model, J, n, batch=256):
    import torch
    x, v, m = cached_inputs(J, n); N = len(x); order = np.argsort(m[:, 0].sum(1), kind='stable'); L = np.zeros((N, 10))
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, N, batch):
            idx = order[i:i + batch]; P = min(max(int(m[idx, 0].sum(1).max()), 17), x.shape[2])
            fx = torch.from_numpy(np.ascontiguousarray(x[idx, :, :P])).to(dev); mk = torch.from_numpy(np.ascontiguousarray(m[idx, :, :P])).to(dev)
            L[idx] = model(fx[:, -2:].clone(), fx.clone(), mk).double().cpu().numpy()
    return L


JC_KNOWN = dict(ParT_full='per class (H55, 2,000/class) QCD 0.776 Zqq 0.679 Wqq 0.790 Tbqq 0.952 Hbb 0.927; H18/H49 QCD 0.783 Zqq 0.689 Wqq 0.805 Tbqq 0.956',
                ParT_kin='H55 QCD 0.731 Zqq 0.558 Wqq 0.684 Tbqq 0.897', PN_full='H55 QCD 0.765 Zqq 0.649 Wqq 0.773 Tbqq 0.951',
                PN_kin='H55 QCD 0.705 Zqq 0.526 Wqq 0.669 Tbqq 0.898')


def g_jc(kind):
    from interp.part_core import ParTNetwork
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); trained, fac = {}, {}
    if kind == 'part':
        for n in ('full', 'kin'):
            net = ParTNetwork(n); trained[f'ParT_{n}'] = (lambda J, net=net, n=n: fwd(net, *cached_inputs(J, n))[1])

            def f_(s, n=n):
                net = ParTNetwork(n); H.permute_torch(net.model, s); return lambda J, net=net: fwd(net, *cached_inputs(J, n))[1]
            fac[f'ParT_{n}'] = f_
    else:
        pth = lambda n: Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt'
        for n, dims in (('full', 17), ('kin', 7)):
            pn = load_particlenet(pth(n), dims, dev); trained[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits_c(pn, J, n))

            def f_(s, n=n, dims=dims):
                pn = H.permute_torch(load_particlenet(pth(n), dims, dev), s); return lambda J, pn=pn: pn_logits_c(pn, J, n)
            fac[f'PN_{n}'] = f_
    run_group('jc', f'jc_{kind}', trained, fac, jc_prep, {k: v for k, v in JC_KNOWN.items() if k in trained})


def g_top(which):
    import torch
    from interp import h4_top_wmass as H4
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    cache = {}

    def prep(T):
        return dict(P4=H.four(T).astype(np.float32), nn=H.nreal(T), ae=T['ax_eta'], ap=T['ax_phi'])

    def jets(X):
        if cache.get('X') is X: return cache
        cache.clear(); cache['X'] = X
        J = H4.jet_inputs(X['P4'], X['nn']); m = J['mask']; P = X['P4'].astype(np.float64)
        ptc = np.where(m, J['pt'], 1.0); eta = np.arcsinh(np.where(m, P[..., 2], 0) / ptc); phi = np.arctan2(P[..., 1], P[..., 0])
        S = (P * m[..., None]).sum(1); jeta = np.arcsinh(S[:, 2] / np.hypot(S[:, 0], S[:, 1])); sg = np.where(jeta < 0, -1.0, 1.0)
        J['deta'] = np.where(m, (eta - X['ae'][:, None]) * sg[:, None], 0); J['dphi'] = np.where(m, wrap(phi - X['ap'][:, None]), 0)
        d = lambda a: torch.from_numpy(a).to(dev); batches = []
        for b in range(0, len(X['nn']), 512):
            Jb = {k: (v[b:b + 512] if np.ndim(v) else v) for k, v in J.items()}; batches.append(tuple(d(a) for a in H4.weaver_inputs(Jb)))
        cache['wb'] = batches
        cache['k3'] = np.stack([J['pt'][:, :128], J['deta'][:, :128], J['dphi'][:, :128]], -1).astype(np.float16).astype(np.float32)
        return cache

    def weaver_f(wn):
        def f(X):
            c = jets(X); out = []
            with torch.no_grad():
                for pts, x, v, mk in c['wb']:
                    o = (wn.model(x, v=v, mask=mk) if wn.kind == 'ParT' else wn.model(pts, x, mk)).float().cpu().numpy(); out.append(o[:, 0] - o[:, 1])
            return two(np.concatenate(out))                     # class 0 = top in the weaver files
        return f

    def flow_f(fl):
        return lambda X: two(fl.score(jets(X)['k3']))
    order = np.random.default_rng(H.SEEDS['top'] + 5).permutation(2 * (SMOKE or H.PER['top']))
    trained, fac, known = {}, {}, {}
    if which == 'main':
        for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
            ck = V / 'models/top' / dname / 'net_best_epoch_state.pt'; trained[f'{kind}_kin'] = weaver_f(H4.WeaverNet(kind, ck, dev))

            def f_(s, kind=kind, ck=ck):
                w2 = H4.WeaverNet(kind, ck, dev); H.permute_torch(w2.model, s); return weaver_f(w2)
            fac[f'{kind}_kin'] = f_
        for a in ('pfn', 'efn'):
            fl0 = H4.Flow(a); trained[a.upper()] = flow_f(fl0)

            def f_(s, fl0=fl0):
                f2 = copy.deepcopy(fl0); rng = np.random.default_rng(s)
                perm = lambda L: [(rng.permutation(K.ravel()).reshape(K.shape), rng.permutation(b.ravel()).reshape(b.shape)) for K, b in L]
                f2.phi, f2.F = perm(f2.phi), perm(f2.F); f2.out = perm([f2.out])[0]; return flow_f(f2)
            fac[a.upper()] = f_
        known = dict(ParT_kin='0.940 (weaver test, H2); H55 sample 0.941', PN_kin='0.940 (weaver test, H2); H55 0.935', PFN='0.9283 (H2); H55 0.923',
                     EFN='0.9218 (H2); H55 0.919')
    else:
        ln = H4.LNet(dev, mass64=True); trained['LorentzNet'] = lambda X: two(ln.score(X['P4'], X['nn'], order))

        def f_(s):
            l2 = H4.LNet(dev, mass64=True); H.permute_torch(l2.model, s); return lambda X, l2=l2: two(l2.score(X['P4'], X['nn'], order))
        fac['LorentzNet'] = f_
        try: known['LorentzNet'] = f"released test acc {json.loads((V / 'models/top/lorentznet/test-result.json').read_text()).get('test_acc')}; H55 0.942"
        except Exception: known['LorentzNet'] = 'H55 0.942'
    run_group('top', f'top_{which}', trained, fac, prep, known)


def wrap(a):
    return H.wrap(a)


def g_qg(which):
    import torch
    from interp import h36_axis_top_qg  # noqa: F401  (patches h5_qg_counting.kinematics: inputs about the _cond axis)
    from interp import h5_qg_counting as H5
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; M = H5.Models(dev)
    if which == 'main':
        for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
        for kind in ('pfn', 'efn'): H14.add_keras_ef(M, kind)
    else:
        M.add_lnet()
    srcs = list(M.m)

    def prep(T):
        real = T['pt'] > 0
        return dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], pid=(T['ext']['pid'] * real).astype(np.int64), n=real.sum(1),
                    _cond=('ax', np.asarray(T['ax_eta']), np.asarray(T['ax_phi'])))
    two = lambda s: np.stack([np.zeros_like(s), s], 1)     # class 1 = quark; Models.logit = log p(g) - log p(q)
    trained = {k: (lambda X, k=k: two(-M.logit(k, X))) for k in srcs}
    fac = {}
    for src in srcs:
        def f_(s, src=src):
            kind, arch, pidfeat, net = M.m[src]; n2 = H.permute_torch(copy.deepcopy(net), s).eval(); nm = f'{src}__p{s}'
            M.m[nm] = (kind, arch, pidfeat, n2); return lambda X, nm=nm: two(-M.logit(nm, X))
        fac[src] = f_
    run_group('qg', f'qg_{which}', trained, fac, prep, dict(part_kin='0.827 (weaver test, H2); H55 0.827', pnet_kin='H55 0.833', pfn_k='H55 0.818',
                                                             efn_k='H55 0.808', lorentznet='H55 0.851', ckpt=M.info))


GROUPS = dict(hls_jedi=g_hls_jedi, hls_dsets=lambda: g_hls_small('dsets'), hls_linf=lambda: g_hls_small('linf'), hls_tf=lambda: g_hls_small('tf'),
              hls_jedis=lambda: g_hls_small('jedis'), jc_part=lambda: g_jc('part'), jc_pn=lambda: g_jc('pn'), top_main=lambda: g_top('main'),
              top_lnet=lambda: g_top('lnet'), qg_main=lambda: g_qg('main'), qg_lnet=lambda: g_qg('lnet'))


# ================================================================================================================ classical taggers
TASK_KIND = {'W_vs_QCD': 'V', 'Z_vs_QCD': 'V', 't_vs_QCD': 'T', 'Top_vs_QCD': 'T', 'top_vs_QCD': 'T', 'q_vs_g': 'QG'}
PUB = dict(V=['V_tau21', 'V_DDT', 'V_D2', 'msd_V', 'mass_V'], T=['top_tau32', 'msd_top', 'mass_top'])
M50 = dict(V=['V_tau21@50', 'V_DDT@50', 'V_D2@50', 'msd_V@50', 'mass_V@50'], T=['top_tau32@50', 'msd_top@50', 'mass_top@50'])
SHAPE1 = dict(V='V_tau21@50', T='top_tau32@50')
SHAPES = dict(V=['V_tau21@50', 'V_DDT@50', 'V_D2@50'], T=['top_tau32@50'])
MSD = dict(V='msd_V@50', T='msd_top@50')
MASS = dict(V='mass_V@50', T='mass_top@50')
SHAPE_VAR = dict(V_tau21='tau21', V_DDT='ddt', V_D2='d2', top_tau32='tau32')


def shortest(x, frac):
    x = np.sort(x); n = len(x); k = int(math.ceil(frac * n)); w = x[k - 1:] - x[:n - k + 1]; i = int(np.argmin(w))
    return float(x[i]), float(x[i + k - 1])


def svar(o, v, rows):
    if v == 'ddt':
        ms = o['msd'][rows]; return o['tau21'][rows] + H.DDT_M * np.log(np.maximum(ms, 1e-3) ** 2 / np.maximum(o['ptj'][rows], 1e-3))
    return o[v][rows]


SNAP = dict(m=0.05, msd=0.05, ptj=0.05, tau21=1e-4, tau32=1e-4, d2=1e-4)


def snap(O):
    """observable changes below numerical precision (|dm|, |dm_SD|, |dpT| < 0.05 GeV; |d tau21|, |d tau32|, |d D2| < 1e-4) are set
    to exactly 0, so that an edit that holds a quantity fixed by construction (decay: m; zboost: m, m_SD, pT) cannot flip a window
    call on a jet sitting at the window edge through float rounding (H58 pre-run: 1-2 such jets per task under zboost)"""
    b = O['base']; out = {'base': b}; info = {}
    for c, o in O.items():
        if c == 'base': continue
        o2 = dict(o)
        for q, tol in SNAP.items():
            if q in o:
                small = (np.abs(o[q] - b[q]) < tol) & (o[q] != b[q]); o2[q] = np.where(small, b[q], o[q])
                if small.any(): info[f'{c}|{q}'] = int(small.sum())
        out[c] = o2
    return out, info


def window_calls(O, rows, ysig, kind):
    """{tagger: {cond: calls}}, info; published and matched-50 % versions"""
    b = O['base']; lo, hi = H.WIN_T if kind == 'T' else H.WIN_V; out, info = {}, {}
    pubc = dict(V_tau21=H.TAU21_CUT, V_DDT=H.DDT_CUT, top_tau32=H.TAU32_CUT)
    for tg in PUB[kind]:
        if tg.startswith(('msd_', 'mass_')):
            q = 'msd' if tg.startswith('msd_') else 'm'
            out[tg] = {c: (o[q][rows] > lo) & (o[q][rows] < hi) for c, o in O.items()}; info[tg] = dict(window=[lo, hi], var=q)
            a, z = shortest(b[q][rows][ysig], 0.5); t50 = tg + '@50'
            out[t50] = {c: (o[q][rows] >= a) & (o[q][rows] <= z) for c, o in O.items()}; info[t50] = dict(window=[a, z], var=q)
        else:
            v = SHAPE_VAR[tg]
            if tg == 'V_D2':
                inw = (b['msd'][rows] > lo) & (b['msd'][rows] < hi) & ysig; cut = float(np.median(b['d2'][rows][inw]))
            else: cut = pubc[tg]
            out[tg] = {c: (o['msd'][rows] > lo) & (o['msd'][rows] < hi) & (svar(o, v, rows) < cut) for c, o in O.items()}
            info[tg] = dict(window=[lo, hi], var=v, cut=cut)
            ms = b['msd'][rows]; a, z = lo, hi; fw = float(((ms > a) & (ms < z))[ysig].mean()); note = 'published m_SD window'
            if fw < 0.55:
                a, z = shortest(ms[ysig], 0.7); fw = float(((ms > a) & (ms < z))[ysig].mean()); note = 'shortest m_SD interval with 70 % of signal (published window < 55 %)'
            inw = (ms > a) & (ms < z) & ysig; c50 = float(np.quantile(svar(b, v, rows)[inw], min(1.0, 0.5 / fw)))
            t50 = tg + '@50'
            out[t50] = {c: (o['msd'][rows] > a) & (o['msd'][rows] < z) & (svar(o, v, rows) <= c50) for c, o in O.items()}
            info[t50] = dict(window=[a, z], var=v, cut=c50, window_note=note, window_sig_eff=fw)
    for tg, cl in out.items():
        c0 = cl['base']; info[tg].update(eff_sig=float(c0[ysig].mean()), eff_bkg=float(c0[~ysig].mean()))
    return out, info


# ================================================================================================================ fingerprint algebra
def fp_batch(Wm, A, R, E):
    """Wm (b, n) jet weights; A (n, R*3E) = [chg | flow | ok] per row -> fingerprints (b, R, 2E)"""
    M = (Wm @ A).reshape(len(Wm), R, 3 * E)
    den = np.maximum(M[..., 2 * E:], 1e-12)
    return np.concatenate([M[..., :E] / den, M[..., E:2 * E] / den], -1)


def stackA(CH, FL, OK):
    R, n, E = CH.shape
    return np.concatenate([CH, FL, OK], 2).transpose(1, 0, 2).reshape(n, R * 3 * E)


def cosines(F, ref, cols=None, scale=True):
    """F (b, R, D) -> (b, R, R); columns divided by their RMS over ref rows (per draw)"""
    if cols is not None: F = F[..., cols]
    if scale:
        s = np.sqrt((F[:, ref] ** 2).mean(1)); G = np.where(s[:, None] > 1e-12, F / np.where(s > 1e-12, s, 1.0)[:, None], 0.0)
    else: G = F
    nr = np.linalg.norm(G, axis=2); G = G / np.where(nr > 0, nr, np.inf)[..., None]
    return np.einsum('brd,bsd->brs', G, G)


def p_two(db):
    B = len(db); return float(min(1.0, 2 * min((1 + (db <= 0).sum()) / (B + 1), (1 + (db >= 0).sum()) / (B + 1))))


def p_z(d, sd):
    if sd <= 0: return 0.0 if d != 0 else 1.0
    return float(math.erfc(abs(d) / sd / math.sqrt(2)))


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; run = min(run, ps[i] * m / (r + 1)); q[i] = run
    return q.tolist()


ARCH_OF = lambda mn: mn.split('_perm')[0]


# ================================================================================================================ stats
def part_stats():
    import matplotlib; matplotlib.use('Agg')
    rng = np.random.default_rng(5810)
    R = dict(node='H58', edits=H.EDIT_WORDS, B_main=B_MAIN, B_pub=B_PUB, nkeep=NKEEP, datasets={}, tests=[])
    for ds in ('hls', 'jc', 'top', 'qg'):
        p = WORK / f'h55_{ds}_obs{sfx()}.npz'
        if not p.exists(): log(f'[{ds}] no observables, skipped'); continue
        Zo = np.load(p); O = {}
        for k in Zo.files:
            c, q = k.split('|'); O.setdefault(c, {})[q] = Zo[k]
        Zt = np.load(WORK / f'h55_{ds}_tab{sfx()}.npz'); y = Zt['y'].astype(int)
        oks = {c: Zt[f'{c}|ok'] for c in H.CONDS if f'{c}|ok' in Zt.files}
        tinfo = json.loads((WORK / f'h55_{ds}_tab{sfx()}.json').read_text())
        LO, acc, screen = {}, {}, {}
        for q in sorted(WORK.glob(f'h58_{ds}_*_lo{sfx()}.npz')):
            if not SMOKE and q.name.endswith('_smoke.npz'): continue
            Zl = np.load(q)
            for k in Zl.files:
                mn, c, tk = k.split('|'); LO.setdefault(tk, {}).setdefault(mn, {})[c] = Zl[k].astype(np.float64)
            a = json.loads(Path(str(q).replace('_lo', '_acc').replace('.npz', '.json')).read_text())
            screen.update(a.pop('_screen', {})); kn = a.pop('_known', {}); acc.update(a); acc.setdefault('_known', {}).update(kn if isinstance(kn, dict) else {})
        edits = [e for e in H.DS_EDITS[ds] if e in oks]; E = len(edits)
        D = dict(N=int(len(y)), labels=np.bincount(y).tolist(), sample=tinfo['info'].get('sample'), cond_info=tinfo['cond_info'],
                 accuracy={k: v for k, v in acc.items() if '_perm' not in k}, perm_screen=screen,
                 perm_accuracy_range={a: [float(np.min(pa)), float(np.max(pa))] for a in screen
                                      for pa in [[v['acc'] for k, v in acc.items() if '_perm' in k and ARCH_OF(k) == a]] if pa}, tasks={})
        for task, (sig, bkg) in tasks_of(ds).items():
            kind = TASK_KIND[task]
            if task not in LO: log(f'[{ds}] {task}: no logits'); continue
            rows = np.flatnonzero(np.isin(y, [sig] + bkg)); ysig = y[rows] == sig; n = len(rows)
            okr = {c: oks[c][rows] for c in oks}; Ocl, snapped = snap({c: O[c] for c in ['base'] + edits})
            if kind == 'QG': ccalls, cinfo = H.classical_calls(ds, task, sig, bkg, Ocl, y, rows)
            else: ccalls, cinfo = window_calls(Ocl, rows, ysig, kind)
            names, kinds, CH, FL, OK = [], [], [], [], []
            for tg, calls in ccalls.items():
                ch, fl, ok = H.response_arrays(calls, okr, edits); names.append(tg)
                kinds.append('c50' if (tg.endswith('@50') or kind == 'QG') else 'cpub'); CH.append(ch); FL.append(fl); OK.append(ok)
            T = dict(kind=kind, n=n, n_sig=int(ysig.sum()), edits=edits, classical=cinfo, snapped_jets_all_rows=snapped, nets={}, degenerate={}, mcnemar={})
            for mn, lc in LO[task].items():
                s = {c: lc[c][rows] for c in ['base'] + edits}; thr = float(np.median(s['base'][ysig])); calls = {c: v > thr for c, v in s.items()}
                ch, fl, ok = H.response_arrays(calls, okr, edits); k_ = 'perm' if '_perm' in mn else 'net'
                if k_ == 'perm' and (np.std(s['base']) < 1e-6 or ch.sum() == 0):
                    T['degenerate'].setdefault(ARCH_OF(mn), []).append(mn); continue
                names.append(mn); kinds.append(k_); CH.append(ch); FL.append(fl); OK.append(ok)
                if k_ == 'net':
                    T['mcnemar'][mn] = {}
                    for j, e in enumerate(edits):
                        o = okr[e].astype(bool); g_, l_ = int((calls[e] & ~calls['base'])[o].sum()), int((~calls[e] & calls['base'])[o].sum())
                        pp = H.binom_p(g_, g_ + l_); T['mcnemar'][mn][e] = dict(gained=g_, lost=l_, n_ok=int(o.sum()), p=pp)
                        R['tests'].append(dict(family='flow_mcnemar', ds=ds, task=task, model=mn, edit=e, p=pp))
            CH, FL, OK = np.stack(CH), np.stack(FL), np.stack(OK); kinds = np.array(kinds); Rn = len(names)
            idx = lambda k: [i for i in range(Rn) if kinds[i] == k]
            nets, c50, cpub, perms = idx('net'), idx('c50'), idx('cpub'), idx('perm')
            T['n_perm_used'] = {a: sum(ARCH_OF(names[i]) == a for i in perms) for a in {ARCH_OF(names[i]) for i in perms}}
            T['n_perm_degenerate'] = {a: len(v) for a, v in T['degenerate'].items()}
            # per-row chg / flow (point)
            F1 = fp_batch(np.ones((1, n)), stackA(CH, FL, OK), Rn, E)[0]
            T['chg'] = {names[i]: dict(zip(edits, F1[i, :E].round(5).tolist())) for i in nets + c50 + cpub}
            T['flow'] = {names[i]: dict(zip(edits, F1[i, E:].round(5).tolist())) for i in nets + c50 + cpub}
            T['perm_chg_median'] = {a: dict(zip(edits, np.median(F1[[i for i in perms if ARCH_OF(names[i]) == a], :E], 0).round(5).tolist()))
                                    for a in T['n_perm_used']}
            # ---------------------------------------------------------------- main (@50) analysis
            S = nets + c50; ref = list(range(len(S))); ni = list(range(len(nets))); ti = list(range(len(nets), len(S)))
            tn = [names[i] for i in c50]; A = stackA(CH[S], FL[S], OK[S])
            C0 = cosines(F1[S][None], ref)[0]
            Cb = []
            for b0 in range(0, B_MAIN if not SMOKE else 500, CHUNK):
                Wm = rng.multinomial(n, np.full(n, 1.0 / n), size=CHUNK).astype(np.float64)
                Cb.append(cosines(fp_batch(Wm, A, len(S), E), ref)[:, ni][:, :, ti].astype(np.float32))
            Cb = np.concatenate(Cb)                                          # (B, nets, taggers)
            # permuted copies: point cosines with the main scaling
            Sp = S + perms; Cp = cosines(F1[Sp][None], ref)[0]
            Fraw = F1[S]; Craw = cosines(Fraw[None], ref, scale=False)[0]
            for a, i in enumerate(nets):
                mn = names[i]; arch = mn; pi = [len(S) + k for k, j in enumerate(perms) if ARCH_OF(names[j]) == arch]
                r = dict(cos={}, nearest=None)
                for b_, t in enumerate(ti):
                    c0 = float(C0[a, t]); pc = Cp[pi, t] if pi else np.array([])
                    r['cos'][tn[b_]] = dict(cos=c0, ci=np.percentile(Cb[:, a, b_], [2.5, 97.5]).tolist(), raw_cos=float(Craw[a, t]),
                                            perm_n=len(pi), perm_mean=float(pc.mean()) if len(pc) else None,
                                            perm_q=np.percentile(pc, [2.5, 50, 97.5]).tolist() if len(pc) else None,
                                            rank_p=float((1 + (pc >= c0).sum()) / (len(pc) + 1)) if len(pc) else None)
                    if len(pc): R['tests'].append(dict(family='cos_vs_perm', ds=ds, task=task, model=mn, tagger=tn[b_], p=r['cos'][tn[b_]]['rank_p']))
                best = int(np.argmax(C0[a, ti])); nb = np.argmax(Cb[:, a, :], 1)
                r['nearest'] = dict(tagger=tn[best], boot_freq=float((nb == best).mean()),
                                    freq_all={tn[k]: float((nb == k).mean()) for k in range(len(tn))})
                if kind in ('V', 'T'):
                    js, jm, jg = tn.index(SHAPE1[kind]), tn.index(MSD[kind]), tn.index(MASS[kind])
                    con = dict(d_shape=(js, jm), d_groom=(jm, jg), d_sm=(js, jg))
                    r['contrasts'] = {}
                    for cn, (u, v) in con.items():
                        d0 = float(C0[a, ti[u]] - C0[a, ti[v]]); db = Cb[:, a, u].astype(np.float64) - Cb[:, a, v]
                        pdp = np.array([Cp[k, ti[u]] - Cp[k, ti[v]] for k in pi]) if pi else np.array([])
                        sd = float(np.std(db)); pb = p_two(db)
                        r['contrasts'][cn] = dict(delta=d0, ci=np.percentile(db, [2.5, 97.5]).tolist(), sd_boot=sd, p_boot=pb, p_z=p_z(d0, sd),
                                                  raw_delta=float(Craw[a, ti[u]] - Craw[a, ti[v]]),
                                                  perm_mean=float(pdp.mean()) if len(pdp) else None,
                                                  perm_q=np.percentile(pdp, [2.5, 50, 97.5]).tolist() if len(pdp) else None,
                                                  frac_perm_ge=float((pdp >= d0).mean()) if len(pdp) else None,
                                                  rank_p_two=float(min(1.0, 2 * min((1 + (pdp >= d0).sum()), (1 + (pdp <= d0).sum())) / (len(pdp) + 1))) if len(pdp) else None)
                        R['tests'].append(dict(family='contrast', ds=ds, task=task, model=mn, contrast=cn, p=pb, delta=d0))
                    # best-of-shape (selection redone per draw)
                    sh = [tn.index(k) for k in SHAPES[kind]]
                    d0 = float(C0[a, [ti[k] for k in sh]].max() - C0[a, ti[jm]]); db = Cb[:, a, sh].max(1).astype(np.float64) - Cb[:, a, jm]
                    r['contrasts']['d_bestshape'] = dict(shape=tn[sh[int(np.argmax(C0[a, [ti[k] for k in sh]]))]], delta=d0,
                                                         ci=np.percentile(db, [2.5, 97.5]).tolist(), p_boot=p_two(db))
                # leave one edit out
                loo = {}
                for j, e in enumerate(edits):
                    cols = [k for k in range(2 * E) if k not in (j, j + E)]; Cl = cosines(F1[S][None], ref, cols)[0]
                    ent = dict(nearest=tn[int(np.argmax(Cl[a, ti]))])
                    if kind in ('V', 'T'):
                        ent.update({cn: float(Cl[a, ti[u]] - Cl[a, ti[v]]) for cn, (u, v) in con.items()})
                    loo[e] = ent
                r['leave_one_out'] = loo
                r['loo_nearest_changed'] = int(sum(loo[e]['nearest'] != r['nearest']['tagger'] for e in edits))
                if kind in ('V', 'T'):
                    r['loo_sign_flips'] = {cn: int(sum(np.sign(loo[e][cn]) != np.sign(r['contrasts'][cn]['delta']) for e in edits)) for cn in con}
                T['nets'][mn] = r
            T['cosine_classical'] = {tn[u]: {tn[v]: float(C0[ti[u], ti[v]]) for v in range(len(ti))} for u in range(len(ti))}
            # ---------------------------------------------------------------- published-cut analysis (secondary)
            if cpub:
                S2 = nets + cpub; ref2 = list(range(len(S2))); tn2 = [names[i] for i in cpub]; ti2 = list(range(len(nets), len(S2)))
                A2 = stackA(CH[S2], FL[S2], OK[S2]); C2 = cosines(F1[S2][None], ref2)[0]; Cb2 = []
                for b0 in range(0, B_PUB if not SMOKE else 200, CHUNK):
                    Wm = rng.multinomial(n, np.full(n, 1.0 / n), size=CHUNK).astype(np.float64)
                    Cb2.append(cosines(fp_batch(Wm, A2, len(S2), E), ref2)[:, :len(nets)][:, :, ti2])
                Cb2 = np.concatenate(Cb2); pub = {}
                for a, i in enumerate(nets):
                    best = int(np.argmax(C2[a, ti2])); nb = np.argmax(Cb2[:, a, :], 1)
                    ent = dict(cos={tn2[b_]: dict(cos=float(C2[a, t]), ci=np.percentile(Cb2[:, a, b_], [2.5, 97.5]).tolist()) for b_, t in enumerate(ti2)},
                               nearest=tn2[best], boot_freq=float((nb == best).mean()))
                    sh0 = tn2.index(SHAPE1[kind].replace('@50', '')); ms0 = tn2.index(MSD[kind].replace('@50', '')); ma0 = tn2.index(MASS[kind].replace('@50', ''))
                    for cn, (u, v) in dict(d_shape=(sh0, ms0), d_groom=(ms0, ma0), d_sm=(sh0, ma0)).items():
                        db = Cb2[:, a, u] - Cb2[:, a, v]
                        ent[cn] = dict(delta=float(C2[a, ti2[u]] - C2[a, ti2[v]]), ci=np.percentile(db, [2.5, 97.5]).tolist(), p_boot=p_two(db))
                    pub[names[i]] = ent
                T['published_cuts'] = pub
            D['tasks'][task] = T
            log(f'[{ds}] {task}: {len(nets)} nets, {len(perms)} permuted (non-degenerate), {len(c50)} @50 / {len(cpub)} published taggers; '
                + '; '.join(f'{names[i]} nearest {T["nets"][names[i]]["nearest"]["tagger"]}' for i in nets))
        R['datasets'][ds] = D
    for fam in ('contrast', 'cos_vs_perm', 'flow_mcnemar'):
        ix = [i for i, t in enumerate(R['tests']) if t['family'] == fam]
        if not ix: continue
        hp = H.holm([R['tests'][i]['p'] for i in ix]); bq = bh([R['tests'][i]['p'] for i in ix])
        for i, h, q in zip(ix, hp, bq): R['tests'][i]['p_holm'] = h; R['tests'][i]['q_bh'] = q
        R[f'n_tests_{fam}'] = len(ix)
    for t in R['tests']:
        if t['family'] == 'contrast':
            R['datasets'][t['ds']]['tasks'][t['task']]['nets'][t['model']]['contrasts'][t['contrast']].update(p_holm=t['p_holm'], q_bh=t['q_bh'])
    (PHYS / f'h58_fingerprint{sfx()}.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h58_fingerprint json')
    try: fig_contrasts(R)
    except Exception as e: import traceback; traceback.print_exc(); log('figure failed', repr(e))


LAB = {'JEDI': 'JEDI-linear', 'dsets': 'Deep Sets', 'linf': 'Linformer', 'tf': 'Transformer', 'jedis': 'JEDI-style float'}
DSL = dict(hls='hls4ml', jc='JetClass', top='Top Landscape')
TL = {'W_vs_QCD': 'W', 'Z_vs_QCD': 'Z', 't_vs_QCD': 'top', 'Top_vs_QCD': 'top', 'top_vs_QCD': 'top'}


def fig_contrasts(R):
    import matplotlib.pyplot as plt
    rows = []
    for ds in ('hls', 'jc', 'top'):
        for task, T in R['datasets'].get(ds, {}).get('tasks', {}).items():
            if T['kind'] not in ('V', 'T'): continue
            for mn, r in T['nets'].items(): rows.append((ds, task, mn, r['contrasts']))
    if not rows: return
    col = dict(hls='tab:orange', jc='tab:blue', top='tab:green')
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.24 * len(rows) + 1.8), sharey=True)
    for ax, cn, ttl in ((axs[0], 'd_shape', 'cos(net, m_SD window + τ21/τ32 cut) − cos(net, m_SD window only)'),
                        (axs[1], 'd_groom', 'cos(net, m_SD window only) − cos(net, ungroomed m window)')):
        for k, (ds, task, mn, c) in enumerate(rows):
            d = c[cn]; ax.errorbar(d['delta'], k, xerr=[[d['delta'] - d['ci'][0]], [d['ci'][1] - d['delta']]], fmt='o', ms=3.5, color=col[ds], lw=1.2)
            if d.get('perm_q'): ax.plot(d['perm_q'][1], k, '|', color='0.45', ms=7, mew=1.5)
            if d.get('p_holm', 1) < 0.01: ax.text(d['ci'][1] + 0.01, k, '*', va='center', fontsize=9)
        ax.axvline(0, color='k', lw=0.8); ax.set_title(ttl, fontsize=8.5); ax.set_xlabel('difference of cosine similarities (95 % jet-bootstrap CI)', fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([f'{DSL[ds]} {TL[t]}: {LAB.get(mn, mn)}' for ds, t, mn, _ in rows], fontsize=7)
    axs[0].invert_yaxis()
    fig.suptitle('Which classical tagger each network\'s edit-response pattern is closer to (all taggers and networks at 50 % signal '
                 'efficiency)\n> 0 left: closer to m_SD + shape than to m_SD alone; > 0 right: closer to groomed than ungroomed mass. '
                 'Grey bar: median of weight-permuted copies; * Holm p < 0.01', fontsize=8.5)
    fig.tight_layout(); p = PHYS / f'h58_contrasts{sfx()}.png'; fig.savefig(p, dpi=150); log(f'wrote {p}')


# ================================================================================================================ main
def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['edits', 'nets', 'stats'])
    ap.add_argument('--ds'); ap.add_argument('--group'); ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args(); SMOKE = a.smoke; WORK.mkdir(parents=True, exist_ok=True)
    if a.part == 'edits':
        H.load_ds = load_ds_h58; H.part_edits(a.ds, a.smoke)
    elif a.part == 'nets':
        if SMOKE: H.PER.update({k: SMOKE for k in H.PER})
        GROUPS[a.group]()
    else: part_stats()
    log('done')


if __name__ == '__main__':
    main()
