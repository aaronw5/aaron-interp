"""H24 verify (loop D): independent rerun of the H24 count-response measurements on a fresh sample, plus three controls
the H24 run did not have.

Sample: interp.h14_qg_generator.load_sample, Pythia (data_qg test) and Herwig 7.1 (files 18-19), 2000 quark + 2000 gluon
jets each, random rows, seed VSEED = 7243 (H24 used 9641). The H24 edit generator is reused with its module SEED set to
VSEED, so the edit draws are also new.
Models (logit = log p(gluon) - log p(quark)):
  best checkpoints: part_kin, pnet_kin (Pythia-trained), hw_part_kin, hw_pnet_kin (Herwig-trained), part_kin_perm (null)
  epoch checkpoints 12..19 of each of the four trainings (net_epoch-<e>_state.pt): within-run spread of the count response,
  a lower bound on the run-to-run spread that H24 did not measure.
  LR(N) per generator from the H24 negative-binomial fits (results/physics/h24_base.json).
Edits (all rebuilt about the edited jet's E-scheme axis by the model input code):
  H24 originals: add_soft, add_swap, sp1..5 (exact split, dR = 0), ct1..5 (dN = 0, matched pT^D), swap
  nsp1..5   near-collinear split: same particles as sp_k, daughters z = 0.5 placed at +-delta/2 (delta = 0.01) in a random
            direction about the parent (no zero-dR pair; tests whether the sp_k response is an artefact of dR = 0)
  addk1..5  k on-manifold soft particles added (each with the jet's softest pT at a random particle's position + 0.05)
  adsw1..5  dN = 0 control: the same k particles added and the k softest original particles removed
  (addk_k - adsw_k) slope over k: on-manifold multi-particle count response; on Herwig gluon jets the interpolation at
  k = 4.48 gives an on-manifold version of the H24 'swap' toward Pythia multiplicity.
Output: $AI_RESULTS/physics/h24_verify.json, h24_verify_logits_<gen>.npz, h24_verify.png
"""
import json
import os
import time
from pathlib import Path

import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
VSEED = 7243
PER_CLASS = int(os.environ.get('H24V_PER_CLASS', 2000))
KS = (1, 2, 3, 4, 5)
EPOCHS = tuple(range(12, 20))
DELTA = 0.01
TRAIN = dict(part_kin=('part', V / 'models' / 'qg' / 'part_kin'), pnet_kin=('pn', V / 'models' / 'qg' / 'pnet_kin'),
             hw_part_kin=('part', V / 'models' / 'qg_herwig' / 'part_kin'), hw_pnet_kin=('pn', V / 'models' / 'qg_herwig' / 'pnet_kin'))
EP_CONDS = ['base', 'add_soft', 'add_swap', 'sp1', 'sp3', 'sp5', 'ct1', 'ct3', 'ct5', 'addk1', 'addk3', 'addk5', 'adsw1', 'adsw3', 'adsw5']


def _rng(gen, name):
    import zlib
    return np.random.default_rng([VSEED, 17, zlib.crc32(gen.encode()), zlib.crc32(name.encode())])


def extra_edits(L0, gen):
    """nsp_k (near-collinear split of the sp_k particles), addk_k / adsw_k (on-manifold k added soft particles and control)"""
    import interp.h24_qg_generator_models as H24
    J = len(L0); out = {}
    # the same particle choice as H24.make_edits for sp_k (its 'split' rng with SEED = VSEED)
    rng = H24._rng(gen, 'split'); okS = np.array([len(a) >= 10 for a in L0]); order = []
    for a in L0:
        n = len(a); h = n // 2; order.append(rng.permutation(np.arange(n - h, n)))
    r2 = _rng(gen, 'nsp'); TH = r2.uniform(0, 2 * np.pi, (J, 5))
    NS = {k: [] for k in KS}
    for j, a in enumerate(L0):
        for k in KS:
            if not okS[j]: NS[k].append(a); continue
            b = a.copy(); new = []
            for m, i in enumerate(order[j][:k]):
                u = 0.5 * DELTA * np.array([np.cos(TH[j, m]), np.sin(TH[j, m])])
                p = a[i].copy(); p[0] *= 0.5; d1 = p.copy(); d2 = p.copy()
                d1[1] += u[0]; d1[2] += u[1]; d2[1] -= u[0]; d2[2] -= u[1]
                b[i] = d1; new.append(d2)
            NS[k].append(np.concatenate([b, np.array(new)]))
    for k in KS: out[f'nsp{k}'] = (NS[k], okS.copy(), np.full(J, float(k)))
    r3 = _rng(gen, 'addk'); AK = {k: [] for k in KS}; AW = {k: [] for k in KS}
    for j, a in enumerate(L0):
        P = []
        for m in range(5):
            i = r3.integers(len(a)); th = r3.uniform(0, 2 * np.pi)
            p = a[i].copy(); p[1] += 0.05 * np.cos(th); p[2] += 0.05 * np.sin(th); p[0] = a[-1, 0]; P.append(p)
        P = np.array(P)
        for k in KS:
            if not okS[j]: AK[k].append(a); AW[k].append(a); continue
            AK[k].append(np.concatenate([a, P[:k]])); AW[k].append(np.concatenate([a[:-k], P[:k]]))
    for k in KS:
        out[f'addk{k}'] = (AK[k], okS.copy(), np.full(J, float(k))); out[f'adsw{k}'] = (AW[k], okS.copy(), np.zeros(J))
    for k, (Le, ok, d) in out.items():
        out[k] = ([b[np.argsort(-b[:, 0], kind='stable')] for b in Le], ok, d)
    return out


def build_net(arch):
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from weaver.nn.model.ParticleNet import ParticleNet
    if arch == 'part':
        return ParticleTransformer(input_dim=7, num_classes=2, pair_input_dim=4, use_pre_activation_pair=False,
                                   embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8,
                                   num_cls_layers=2, block_params=None,
                                   cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                                   fc_params=[], activation='gelu', trim=True, for_inference=False)
    return ParticleNet(input_dims=7, num_classes=2, conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
                       fc_params=[(256, 0.1)], use_fusion=False, use_fts_bn=True, use_counts=True, for_inference=False)


def load_models(dev):
    import torch
    from interp import h5_qg_counting as H
    M = H.Models(dev); nets = {}; info = {}
    for name, (arch, d) in TRAIN.items():
        files = [('best', d / 'net_best_epoch_state.pt')] + [(f'e{e}', d / f'net_epoch-{e}_state.pt') for e in EPOCHS]
        for tag, f in files:
            net = build_net(arch); net.load_state_dict(M._sd(f), strict=True); net.eval().to(dev)
            nets[f'{name}@{tag}'] = (arch, net); info[f'{name}@{tag}'] = str(f)
    # permuted-weight null, as H5 Models.add_permuted (seed 0)
    import copy
    g = torch.Generator().manual_seed(0); net2 = copy.deepcopy(nets['part_kin@best'][1]).cpu()
    with torch.no_grad():
        for p in net2.parameters():
            flat = p.data.view(-1); flat.copy_(flat[torch.randperm(flat.numel(), generator=g)])
    nets['part_kin_perm@best'] = ('part', net2.eval().to(dev)); info['part_kin_perm@best'] = 'part_kin best, weights permuted (seed 0)'
    log('loaded', len(nets), 'networks')
    return nets, info


def run_nets(nets, Le, which, dev, bs=500):
    import torch
    from interp import h5_qg_counting as H
    Jt = H.from_list(Le, Lpad=180); P, X, Vv, Mk = H.weaver_inputs(Jt, False); out = {}
    with torch.no_grad():
        for nm in which:
            arch, net = nets[nm]; r = []
            for s in range(0, len(X), bs):
                a = [torch.from_numpy(z[s:s + bs]).to(dev) for z in (P, X, Vv, Mk)]
                o = net(a[1], v=a[2], mask=a[3].bool()) if arch == 'part' else net(a[0], a[1], a[3])
                o = torch.log_softmax(o.float(), -1); r.append((o[:, 1] - o[:, 0]).cpu().numpy())
            out[nm] = np.concatenate(r).astype(np.float64)
    return out


def step_eval():
    import torch
    import interp.h24_qg_generator_models as H24
    from interp import h14_qg_generator as H14
    from scipy.stats import nbinom
    H24.SEED = VSEED                                   # new sample rows and new edit draws
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    nets, info = load_models(dev)
    best = [n for n in nets if n.endswith('@best')]; allm = list(nets)
    lrn = json.loads((RES / 'h24_base.json').read_text())['lrn']
    T = {g: nbinom.logpmf(np.arange(400), lrn[g]['nb_g']['r'], lrn[g]['nb_g']['p'])
         - nbinom.logpmf(np.arange(400), lrn[g]['nb_q']['r'], lrn[g]['nb_q']['p']) for g in ('py', 'hw')}
    c2 = {}
    for gen in ('py', 'hw'):
        L0, y, inf = H14.load_sample(gen, PER_CLASS, VSEED)
        if gen == 'py':
            from interp import data_qg
            log('rows shared with the H24 Pythia sample:', int(np.isin(inf['idx'], data_qg.sample(4000, 9641)['idx']).sum()))
        E = H24.make_edits(L0, y, gen); E.update(extra_edits(L0, gen))
        log(gen, len(y), 'jets; edits', {k: int(v[1].sum()) for k, v in E.items()})
        out = dict(y=y, idx=inf['idx'])
        conds = [('base', L0, np.ones(len(L0), bool), np.zeros(len(L0)))] + [(k, *v) for k, v in E.items()]
        for cn, Le, ok, dN in conds:
            which = allm if cn in EP_CONDS else best
            S = run_nets(nets, Le, which, dev)
            for nm, v in S.items(): out[f'{cn}|{nm}'] = v.astype(np.float32)
            N = np.array([len(a) for a in Le])
            for tg in ('py', 'hw'): out[f'{cn}|lrn_{tg}@best'] = T[tg][np.minimum(N, 399)].astype(np.float32)
            O = H24.basic_obs(Le); out.update({f'obs|{cn}|{k}': v.astype(np.float32) for k, v in O.items()})
            if cn != 'base': out[f'ok|{cn}'] = ok; out[f'dN|{cn}'] = dN.astype(np.float32)
            if cn == 'base':
                from interp import h5_qg_counting as H
                log(gen, 'unedited', '; '.join(f'{nm} {H.roc_metrics(out[f"base|{nm}"].astype(float), y)}' for nm in best))
            else:
                log(gen, cn, f'applied {int(ok.sum())} dptd {np.mean((O["ptd"] - out["obs|base|ptd"])[ok]):+.6f} '
                    f'dgirth {np.mean((O["girth"] - out["obs|base|girth"])[ok]):+.2e} dmass {np.mean((O["mass"] - out["obs|base|mass"])[ok]):+.4f} | '
                    + ' '.join(f'{nm} {np.mean((out[f"{cn}|{nm}"] - out[f"base|{nm}"])[ok]):+.4f}' for nm in best))
        np.savez_compressed(RES / f'h24_verify_logits_{gen}.npz', **out)
        try:
            from interp import h5_qg_counting as H
            rs = np.random.default_rng(5); sub = np.sort(rs.choice(len(L0), len(L0), replace=False))
            Ls = [L0[i] for i in sub]; c2[gen] = {'none': H.c2st(Ls, Ls)}
            for en in ('sp1', 'nsp1', 'nsp5', 'addk1', 'addk5', 'adsw5'):
                c2[gen][en] = H.c2st(Ls, [E[en][0][i] for i in sub]); log('c2st', gen, en, c2[gen][en])
        except Exception as e:
            log('c2st skipped', repr(e)); c2[gen] = dict(error=repr(e))
    (RES / 'h24_verify_models.json').write_text(json.dumps(dict(models=info, c2st=c2), indent=1))


# ------------------------------------------------------------------------------------------------------------------ stats
def _slope(D, ks):
    k = np.array(ks, float); kc = k - k.mean()
    return D @ kc / (kc ** 2).sum()


def step_stats():
    import interp.h24_qg_generator_models as H24
    from interp import h5_qg_counting as H
    H24.SEED = VSEED
    Boot = H24._boot_cls()
    MI = json.loads((RES / 'h24_verify_models.json').read_text())
    R = dict(sample=dict(per_class=PER_CLASS, seed=VSEED, delta_nsp=DELTA, epochs=list(EPOCHS)), c2st=MI['c2st'], gens={})
    tests = []; Lg = {}
    for gen in ('py', 'hw'):
        Z = np.load(RES / f'h24_verify_logits_{gen}.npz'); y = Z['y'].astype(int); J = len(y); Bt = Boot(J, seed=VSEED)
        S = {tuple(k.split('|')): Z[k].astype(float) for k in Z.files if len(k.split('|')) == 2}
        ok = {k.split('|')[1]: Z[k].astype(bool) for k in Z.files if k.startswith('ok|')}
        obs = {(k.split('|')[1], k.split('|')[2]): Z[k].astype(float) for k in Z.files if k.startswith('obs|')}
        dN = {k.split('|')[1]: Z[k].astype(float) for k in Z.files if k.startswith('dN|')}
        models = sorted({m for (c, m) in S if c == 'base'}); best = [m for m in models if m.endswith('@best')]
        g = dict(J=J, unedited={}, obs={}, metrics={}, epochs={}); Lg[gen] = dict(y=y, S=S)
        for m in models:
            if m.startswith('lrn'): continue
            g['unedited'][m] = H.roc_metrics(S[('base', m)], y)
        for c in ok:
            o = ok[c]; g['obs'][c] = dict(applied=int(o.sum()), **{f'd{k}': float((obs[(c, k)] - obs[('base', k)])[o].mean())
                                                                     for k in ('N', 'ptd', 'girth', 'mass')},
                                         abs_dmass=float(np.abs(obs[(c, 'mass')] - obs[('base', 'mass')])[o].mean()),
                                         abs_dgirth=float(np.abs(obs[(c, 'girth')] - obs[('base', 'girth')])[o].mean()))
        o5 = ok['sp1']

        def metrics(m, ks=KS):
            d0 = lambda c: S[(c, m)] - S[('base', m)]
            r = {}
            r['add_minus_swap'] = d0('add_soft') - d0('add_swap')
            st = lambda p: np.stack([d0(f'{p}{k}') for k in ks], 1)
            r['sp_minus_ct'] = _slope(st('sp'), ks) - _slope(st('ct'), ks)
            r['sp_raw'] = _slope(st('sp'), ks); r['ct_raw'] = _slope(st('ct'), ks)
            r['addk_minus_adsw'] = _slope(st('addk') - st('adsw'), ks)
            if ks == KS:
                r['nsp_minus_ct'] = _slope(st('nsp'), ks) - _slope(st('ct'), ks)
                r['nsp_raw'] = _slope(st('nsp'), ks)
                # on-manifold swap toward Pythia N (gluon jets): interpolate addk - adsw at k = gap
                gap = 4.482931944736215; f = gap - 4
                r['oswap_gluon'] = (1 - f) * (d0('addk4') - d0('adsw4')) + f * (d0('addk5') - d0('adsw5'))
                r['swap'] = d0('swap')
            return r

        okk = dict(add_minus_swap=ok['add_swap'], sp_minus_ct=o5, sp_raw=o5, ct_raw=o5, addk_minus_adsw=o5, nsp_minus_ct=o5, nsp_raw=o5,
                   oswap_gluon=o5 & (y == 0), swap=ok['swap'] & (dN['swap'] != 0) & (y == 0))
        PJ = {}
        for m in best:
            r = metrics(m); PJ[m] = r; g['metrics'][m] = {}
            for key, d in r.items():
                o = okk[key]; ci = Bt.mean_ci(d, o)[0]
                e = dict(n=int(o.sum()), mean=float(d[o].mean()), ci=ci)
                if key in ('add_minus_swap', 'sp_minus_ct', 'addk_minus_adsw', 'nsp_minus_ct', 'oswap_gluon'):
                    e['p_perm'] = Bt.signflip(d, o); tests.append((gen, key, m, e['p_perm']))
                g['metrics'][m][key] = e
            log(gen, m, ' '.join(f'{k} {v["mean"]:+.4f}' for k, v in g['metrics'][m].items()))
        # Herwig-trained - Pythia-trained, paired per jet (best checkpoints)
        g['hw_minus_py'] = {}
        for a, b in (('hw_part_kin@best', 'part_kin@best'), ('hw_pnet_kin@best', 'pnet_kin@best')):
            g['hw_minus_py'][f'{a} - {b}'] = {}
            for key in ('add_minus_swap', 'sp_minus_ct', 'addk_minus_adsw', 'nsp_minus_ct'):
                d = PJ[a][key] - PJ[b][key]; o = okk[key]
                e = dict(mean=float(d[o].mean()), ci=Bt.mean_ci(d, o)[0], p_perm=Bt.signflip(d, o))
                g['hw_minus_py'][f'{a} - {b}'][key] = e; tests.append((gen, f'hw-py {key}', a, e['p_perm']))
        # epoch checkpoints: within-run spread (k = 1, 3, 5)
        for tr in TRAIN:
            g['epochs'][tr] = {}
            for e in EPOCHS:
                m = f'{tr}@e{e}'; r = metrics(m, ks=(1, 3, 5))
                g['epochs'][tr][e] = dict(AUC=H.roc_metrics(S[('base', m)], y)['AUC'],
                                          **{k: float(r[k][okk[k]].mean()) for k in ('add_minus_swap', 'sp_minus_ct', 'addk_minus_adsw')})
        # generator shift vs within-run spread: exact permutation test over the 16 epoch values (8 per training) per arch
        g['epoch_test'] = {}
        from itertools import combinations
        for a, b in (('hw_part_kin', 'part_kin'), ('hw_pnet_kin', 'pnet_kin')):
            g['epoch_test'][f'{a} - {b}'] = {}
            for key in ('add_minus_swap', 'sp_minus_ct', 'addk_minus_adsw'):
                va = np.array([g['epochs'][a][e][key] for e in EPOCHS]); vb = np.array([g['epochs'][b][e][key] for e in EPOCHS])
                allv = np.r_[va, vb]; obsd = va.mean() - vb.mean(); n = len(va); cnt = 0; tot = 0
                for idx in combinations(range(2 * n), n):
                    m1 = np.zeros(2 * n, bool); m1[list(idx)] = True; tot += 1
                    cnt += abs(allv[m1].mean() - allv[~m1].mean()) >= abs(obsd) - 1e-12
                g['epoch_test'][f'{a} - {b}'][key] = dict(diff=float(obsd), sd_hw=float(va.std(ddof=1)), sd_py=float(vb.std(ddof=1)),
                                                          range_hw=[float(va.min()), float(va.max())], range_py=[float(vb.min()), float(vb.max())],
                                                          p_exact=float(cnt / tot))
                tests.append((gen, f'epoch perm {key}', a, cnt / tot))
        R['gens'][gen] = g
    # gluon logit gap Py - Hw (per model, best) and the fraction reproduced by the on-manifold swap on Herwig gluon jets
    R['cross'] = {}
    for m in [m for m in R['gens']['py']['metrics']]:
        yp, yh = Lg['py']['y'], Lg['hw']['y']
        gg = Lg['py']['S'][('base', m)][yp == 0].mean() - Lg['hw']['S'][('base', m)][yh == 0].mean()
        osw = R['gens']['hw']['metrics'][m]['oswap_gluon']['mean']; sw = R['gens']['hw']['metrics'][m]['swap']['mean']
        R['cross'][m] = dict(gluon_gap=float(gg), oswap=osw, frac_oswap=float(osw / gg) if abs(gg) > 1e-6 else None,
                             swap=sw, frac_swap=float(sw / gg) if abs(gg) > 1e-6 else None)
    adj = H.holm([t[3] for t in tests])
    R['holm'] = dict(n_tests=len(tests), n_pass_005=int((adj < 0.05).sum()),
                     tests=[dict(gen=t[0], test=t[1], model=t[2], p=float(t[3]), holm=float(h)) for t, h in zip(tests, adj)])
    (RES / 'h24_verify.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h24_verify.json')
    step_fig(R)


def step_fig(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    col = dict(part_kin='#1f77b4', pnet_kin='#17becf', hw_part_kin='#d62728', hw_pnet_kin='#ff7f0e')
    lab = dict(part_kin='ParT_kin, Pythia-trained', pnet_kin='PN_kin, Pythia-trained', hw_part_kin='ParT_kin, Herwig-trained',
               hw_pnet_kin='PN_kin, Herwig-trained')
    for i, (key, t) in enumerate((('sp_minus_ct', 'exact split (k = 1,3,5) minus ct control'),
                                  ('add_minus_swap', 'add_soft minus add_swap'), ('addk_minus_adsw', 'k added soft particles minus control'))):
        for tr in col:
            for gen, mk in (('py', 'o'), ('hw', 's')):
                v = [R['gens'][gen]['epochs'][tr][str(e)][key] if str(e) in R['gens'][gen]['epochs'][tr] else R['gens'][gen]['epochs'][tr][e][key]
                     for e in EPOCHS]
                ax[i].plot(EPOCHS, v, mk + '-', color=col[tr], ms=3.5, lw=0.8,
                           label=f'{lab[tr]} ({"Pythia" if gen == "py" else "Herwig"} jets)' if i == 0 else None, alpha=1 if gen == 'py' else 0.55)
        ax[i].axhline(0, color='k', lw=0.6); ax[i].set_title(t, fontsize=9.5); ax[i].set_xlabel('training epoch checkpoint')
    ax[0].set_ylabel('Δ logit per added particle (toward gluon)')
    fig.legend(loc='lower center', ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'H24 verify: count response of each checkpoint, epochs {EPOCHS[0]}–{EPOCHS[-1]} '
                 f'({R["sample"]["per_class"]} quark + {R["sample"]["per_class"]} gluon jets per generator, seed {VSEED})', fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(RES / 'h24_verify.png', dpi=130); log('wrote h24_verify.png')


def step_kmeans():
    """per-k mean logit change (vs unedited) of every edit family, all jets and by class, best checkpoints; per-k observables"""
    out = {}
    for gen in ('py', 'hw'):
        Z = np.load(RES / f'h24_verify_logits_{gen}.npz'); y = Z['y'].astype(int); o = Z['ok|sp1'].astype(bool)
        models = sorted({k.split('|')[1] for k in Z.files if k.startswith('base|') and k.endswith('@best')})
        g = {}
        for m in models:
            b = Z[f'base|{m}'].astype(float); g[m] = {}
            for fam in ('sp', 'ct', 'nsp', 'addk', 'adsw'):
                d = np.stack([Z[f'{fam}{k}|{m}'].astype(float) - b for k in KS], 1)
                g[m][fam] = dict(all=d[o].mean(0).round(5).tolist(), q=d[o & (y == 1)].mean(0).round(5).tolist(),
                                 g=d[o & (y == 0)].mean(0).round(5).tolist(), slope=float(_slope(d, KS)[o].mean()))
                if fam in ('addk', 'adsw'):
                    sl = _slope(d, KS); r = np.random.default_rng(VSEED); oi = np.flatnonzero(o)
                    bs = [sl[oi[r.integers(0, len(oi), len(oi))]].mean() for _ in range(2000)]
                    g[m][fam]['slope_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                    ctl = np.stack([Z[f'ct{k}|{m}'].astype(float) - b for k in KS], 1); dd = sl - _slope(ctl, KS)
                    bs = [dd[oi[r.integers(0, len(oi), len(oi))]].mean() for _ in range(2000)]
                    g[m][fam]['minus_ct'] = float(dd[o].mean()); g[m][fam]['minus_ct_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                    if fam == 'addk' and 'perm' not in m:
                        pn = _slope(np.stack([Z[f'addk{k}|part_kin_perm@best'].astype(float) - Z['base|part_kin_perm@best'] for k in KS], 1), KS)
                        g[m][fam]['minus_perm_null'] = float((sl - pn)[o].mean())
        g['obs'] = {}
        for fam in ('addk', 'adsw', 'nsp', 'ct'):
            g['obs'][fam] = {v: [float((Z[f'obs|{fam}{k}|{v}'] - Z[f'obs|base|{v}'])[o].mean()) for k in KS] for v in ('N', 'ptd', 'girth', 'mass')}
        out[gen] = g
        log(gen, json.dumps({m: {f: g[m][f]['slope'] for f in g[m]} for m in models}))
    (RES / 'h24_verify_k.json').write_text(json.dumps(out, indent=1)); log('wrote h24_verify_k.json')


if __name__ == '__main__':
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if step in ('eval', 'all'): step_eval()
    if step in ('stats', 'all'): step_stats()
    if step == 'kmeans': step_kmeans()
