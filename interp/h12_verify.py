"""H12 verification (loop D): rerun the key part of interp.h12_boost_panel on a fresh sample (seed 917) and cross-check the
boost with an independent implementation.
  --part hls  (jax env, CPU): hls4ml interp.h1_irc_panel.hls_sample(3000/class, seed 917) = 15,000 jets; JEDI-linear n64, n32.
  --part gpu  (part_job): JetClass jc_sample(1500/class, seed 917) = 15,000 jets, ParT_full + PN_full; Top Landscape test
              5,000 top + 5,000 QCD random rows (rng 917), ParT_kin.
Edits B, P, A, R, T at lam 0.9 and 1.1 (h12 functions) plus Bi = independent boost (velocity addition, standard boost
formula; not the arsinh-rapidity code of h12). Per model: accuracy, % calls changed [95 % bootstrap], calls agreeing between
B and Bi, McNemar B vs R / B vs P / B vs A with Holm, mean delta log-odds per binary task [95 % bootstrap].
Output $AI_RESULTS/physics/h12_verify_<part>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h12_boost_panel as H  # noqa: E402

SEED = 917
LAMS = (0.9, 1.1)


def boost_indep(T, lam):
    """independent boost: jet velocity v0 = |P|/E -> v1 = lam|P| / sqrt(lam^2|P|^2 + m^2); beta = (v1 - v0) / (1 - v0 v1)
    along P-hat; p_par' = g (p_par + beta E), E' = g (E + beta p_par); axis boosted as a massless direction."""
    real = T['pt'] > 0; px, py, pz, E = H.four(T)
    P = np.stack([px.sum(1), py.sum(1), pz.sum(1)], 1); EJ = E.sum(1); pa = np.linalg.norm(P, axis=1)
    m2 = np.maximum(EJ ** 2 - pa ** 2, 1e-12); v0 = pa / EJ; v1 = lam * pa / np.sqrt(lam ** 2 * pa ** 2 + m2)
    beta = (v1 - v0) / (1 - v0 * v1); g = 1 / np.sqrt(1 - beta ** 2); n = P / pa[:, None]
    ppar = px * n[:, :1] + py * n[:, 1:2] + pz * n[:, 2:3]
    dp = (g[:, None] - 1) * ppar + (g * beta)[:, None] * E
    q = [px + dp * n[:, :1], py + dp * n[:, 1:2], pz + dp * n[:, 2:3]]
    U = H.copyT(T); pt, eta, phi = H._from_p(*q)
    U['pt'] = np.where(real, pt, 0); U['eta'] = np.where(real, eta, 0); U['phi'] = np.where(real, phi, 0)
    a = np.stack([np.cos(T['ax_phi']), np.sin(T['ax_phi']), np.sinh(T['ax_eta'])], 1) / np.cosh(T['ax_eta'])[:, None]
    apar = (a * n).sum(1); a2 = a + ((g - 1) * apar + g * beta)[:, None] * n
    _, U['ax_eta'], U['ax_phi'] = H._from_p(a2[:, 0], a2[:, 1], a2[:, 2])
    return U


def run(name, T0, y, models, prep, tasks):
    rng = np.random.default_rng(SEED); J = len(y); out = dict(part=name, N=J, seed=SEED, labels=np.bincount(y).tolist(), models={},
                                                              boost_check={}, tests={})
    O0 = H.jet_obs(T0); X0 = prep(H.sortT(T0)); L0 = {}
    for mn, f in models.items():
        L0[mn] = f(X0); c0 = L0[mn].argmax(1)
        out['models'][mn] = dict(acc=float((c0 == y).mean()), acc_ci=H.boot_frac(c0 == y, rng), edits={})
        H.log(f'{name} {mn} accuracy {out["models"][mn]["acc"]:.4f}')
    erng = np.random.default_rng(SEED + 1000); CH = {}
    for lam in LAMS:
        B, _ = H.boost(T0, lam); Bi = boost_indep(T0, lam); D = H.displacement(T0, B)
        real = T0['pt'] > 0
        out['boost_check'][str(lam)] = dict(
            max_rel_pt=float(np.max(np.abs(B['pt'] - Bi['pt'])[real] / T0['pt'][real])),
            max_abs_eta=float(np.max(np.abs(B['eta'] - Bi['eta'])[real])), max_abs_phi=float(np.max(np.abs(H.wrap(B['phi'] - Bi['phi']))[real])),
            max_abs_axis=float(max(np.abs(B['ax_eta'] - Bi['ax_eta']).max(), np.abs(H.wrap(B['ax_phi'] - Bi['ax_phi'])).max())),
            m_rel_absmax=float(np.max(np.abs(H.jet_obs(Bi)['m'] / O0['m'] - 1))),
            jpt_ratio=[float(np.min(H.jet_obs(Bi)['jpt'] / O0['jpt'])), float(np.max(H.jet_obs(Bi)['jpt'] / O0['jpt']))],
            mA_rel=dict(mean=float(np.mean(H.jet_obs(B)['mA'] / O0['mA'] - 1)), absmax=float(np.max(np.abs(H.jet_obs(B)['mA'] / O0['mA'] - 1))),
                        p99=float(np.percentile(np.abs(H.jet_obs(B)['mA'] / O0['mA'] - 1), 99))))
        H.log(f'{name} boost check lam {lam}: {out["boost_check"][str(lam)]}')
        eds = dict(B=B, Bi=Bi, P=H.pscale(T0, lam)[0], A=H.ascale(T0, lam)[0], R=H.rotate(T0, D, erng)[0], T=H.translate(T0, D, erng)[0])
        for k, U in eds.items():
            X1 = prep(H.sortT(U)); en = f'{k}{lam}'
            for mn, f in models.items():
                L1 = f(X1); c0, c1 = L0[mn].argmax(1), L1.argmax(1); ch = c0 != c1; CH[(mn, en)] = ch
                st = dict(changed=float(ch.mean()), ci=H.boot_frac(ch, rng), acc_after=float((c1 == y).mean()), tasks={})
                if k == 'Bi': st['calls_equal_to_B'] = float((c1 == CH[(mn, 'Bcalls' + str(lam))]).mean())
                if k == 'B': CH[(mn, 'Bcalls' + str(lam))] = c1
                for tn, (s, b) in tasks.items():
                    sel = np.isin(y, [s] + list(b)); d = H.logodds(L1, s, b) - H.logodds(L0[mn], s, b); mu, ci = H.boot_mean(d[sel], rng)
                    st['tasks'][tn] = dict(dlogodds=mu, ci=ci)
                out['models'][mn]['edits'][en] = st
            H.log(f'{name} {en}: ' + ' '.join(f'{mn} {out["models"][mn]["edits"][en]["changed"]:.4f}' for mn in models))
    ps = []
    for mn in models:
        for lam in LAMS:
            for a, b in (('B', 'R'), ('B', 'T'), ('B', 'P'), ('B', 'A')):
                t = H.mcnemar(CH[(mn, f'{a}{lam}')], CH[(mn, f'{b}{lam}')]); t['diff'] = float(CH[(mn, f'{a}{lam}')].mean() - CH[(mn, f'{b}{lam}')].mean())
                k = f'{mn}|{a}{lam}_vs_{b}{lam}'; out['tests'][k] = t; ps.append((k, t['p']))
    for (k, _), h in zip(ps, H.holm([p for _, p in ps])): out['tests'][k]['holm'] = h
    return out


def part_hls():
    from interp.h1_irc_panel import hls_sample
    from interp.jedi_core import Jedi
    T, y = hls_sample(3000, SEED)
    T0 = dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], m=T['m'], ax_eta=T['ax_eta'], ax_phi=T['ax_phi'], ext={})
    nets = {f'jedi_n{n}': Jedi(n, '3-feature') for n in (64, 32)}
    models = {k: (lambda x, n=n: n.run(x)['logits']) for k, n in nets.items()}
    return run('hls', T0, y, models, H.hls_prep, H.HLS_TASKS)


def part_gpu():
    import torch
    from interp.h1_irc_panel import jc_sample, jc_table, pn_logits
    from interp.part_core import ParTNetwork, CL, inputs, build
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    import os
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); res = {}
    J0 = jc_sample(1500, SEED); y = J0['label'].astype(int); Tj = jc_table(J0)
    T0 = dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext'])

    def prep(T):
        Jx = dict(J0); Jx['jet_eta'] = T['ax_eta'].astype(np.float32); Jx['jet_phi'] = T['ax_phi'].astype(np.float32)
        sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = H.rel(T)
        return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
    net = ParTNetwork('full'); pn = load_particlenet(Path(MODELS).parent / 'particlenet' / 'ParticleNet_full.pt', 17, dev)
    models = {'ParT_full': lambda J: fwd(net, *inputs(J, 'full'))[1].astype(np.float64), 'PN_full': lambda J: pn_logits(pn, J, 'full')}
    i = CL.index
    tasks = {'W_vs_QCD': (i('Wqq'), [i('QCD')]), 'Top_vs_QCD': (i('Tbqq'), [i('QCD')]), 'Z_vs_W': (i('Zqq'), [i('Wqq')])}
    res['jetclass'] = run('jetclass', T0, y, models, prep, tasks)
    # Top Landscape, ParT_kin
    from interp import data_top as dt
    from interp.h4_top_wmass import WeaverNet
    d = dt.DST / 'test'; ylab = np.load(d / 'label.npy'); idx = H._rows(ylab, 5000, SEED)
    F = np.load(d / 'full.npy')[idx]; jet = np.load(d / 'jet.npy')[idx]
    nn = np.minimum(jet[:, 5].astype(np.int64), F.shape[1]); yt = ylab[idx].astype(int); Tt = H.table_from_full(F, nn); del F
    devt = 'cuda' if torch.cuda.is_available() else 'cpu'
    wn = WeaverNet('ParT', H.V / 'models/top' / 'part_kin' / 'net_best_epoch_state.pt', devt)

    def tprep(T):
        pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
        P4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
        return dict(P4=P4.astype(np.float32), nn=real.sum(1))
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    res['top'] = run('top', Tt, yt, {'ParT_kin': lambda X: two(wn.score(X['P4'], X['nn']))}, tprep, {'top_vs_QCD': (1, [0])})
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['hls', 'gpu']); a = ap.parse_args()
    out = part_hls() if a.part == 'hls' else part_gpu()
    H.OUT.mkdir(parents=True, exist_ok=True); p = H.OUT / f'h12_verify_{a.part}.json'
    p.write_text(json.dumps(out, indent=1, default=float)); H.log(f'wrote {p}'); H.log('done')
