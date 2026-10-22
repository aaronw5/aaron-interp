"""P59v (loop C, skeptic of P59): replication of the auxiliary-scalar mass direction of L-GATr on fresh files, plus two controls
P59 did not run.

Sample: JetClass test_20M WToQQ / ZToQQ / ZJetsToNuNu files _115 (fit) and _116 (evaluation) (not used by any earlier node),
 5,000 random rows per class per file (default_rng 7171 / 7172; never first-k); pairs and edits default_rng 7173.
 Pairs as in P59: |pT_s/pT_b - 1| <= 0.05, 0.05 <= |ln m_s - ln m_b| <= 0.3, 1,000 bases per class.
Blocks 3, 4, 6, 9, 11. Variant: aux d = 1 (one direction among the 32 auxiliary scalars).
(a) Transfer: the ORIGINAL P59 directions (p59_U_*.npz, aligned- and global-token fits) applied unchanged to the fresh pairs.
(b) Refit: DAS aux d1 aligned with 40 steps (P59's budget) and 200 steps; shuffled-pair DAS (40 steps).
(c) Null: 200 random aux d1 directions on the first 1,000 evaluation pairs: F and beta_cf (joint least squares), empirical p.
(d) Necessity: on the counterfactual jet (base dilated to m_s), set the direction to the base jet's value (all tokens, or the
    global token only) and measure the remaining fraction of the counterfactual response (slope of d_nec - d_b on d_cf - d_b);
    null: 30 random aux directions.
(e) Edit triple at fixed particle masses (2,000 W/Z jets of the evaluation file), each a 5 % change:
    D: angles about the stored axis x lambda so that m x 1.05 (pT fixed; angular size m/pT up 5 %);
    B: every particle four-momentum x 1.05 (m x 1.05, pT x 1.05, angles fixed, angular size fixed; particle masses x 1.05, i.e.
       a pion 0.140 -> 0.147 GeV);
    P: pT x 1.05 with angles rescaled so that m is unchanged (angular size down 5 %, particle masses fixed).
    ratio_B = sum dX(B) / sum dX(D), ratio_P = sum dX(P) / sum dX(D) for X = network d = l_W - l_Z and the d1 coordinate at the
    global token. Invariant-mass reader: ratio_B 1, ratio_P 0. Angular-size (m/pT) reader: ratio_B 0, ratio_P -1. pT reader:
    dX(D) = 0. Linearity check: dX(B) ~ dX(D) + dX(P).
  ./sync.sh && ./part_job.sh c-p59-verify '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p59v_verify'
-> $AI_RESULTS/part/p59_verify.json
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_argv = sys.argv; sys.argv = [_argv[0]]
import interp.p59_lgatr_das_mass as P59  # noqa: E402
sys.argv = _argv
SMOKE = '--smoke' in sys.argv
FFIT, FEVAL, SEED = 115, 116, 7171
BLOCKS = [3, 4, 6, 9, 11] if not SMOKE else [4]
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part' / f'p59_verify{"_smoke" if SMOKE else ""}.json'
ULG = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr'
log = P59.log
iW, iZ = P59.iW, P59.iZ


def dW(L):
    return L[:, iW] - L[:, iZ]


def beta2(Lp, Lt, Lb, Ls):
    y = dW(Lp) - dW(Lb); X = np.stack([dW(Lt) - dW(Lb), dW(Ls) - dW(Lb)], 1); return np.linalg.lstsq(X, y, rcond=None)[0]


def ratio_ci(num, den, Wb):
    bs = (Wb @ num) / (Wb @ den); return [float(num.sum() / den.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    import torch
    from interp.part_core import sub, jmass, sample, kin, build
    Pm = P59.params(SMOKE); Pm['STEPS'] = 40
    per = 5000 if not SMOKE else 300; npair = 1000 if not SMOKE else 40; ned = 1000 if not SMOKE else 30
    ndraw = 200 if not SMOKE else 4; nnec = 30 if not SMOKE else 2; nnull = 1000 if not SMOKE else 60
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev)
    Jf = P59.rebuild(P59.read_split(FFIT, per, np.random.default_rng(SEED)))
    Je = P59.rebuild(P59.read_split(FEVAL, per, np.random.default_rng(SEED + 1))); rng = np.random.default_rng(SEED + 2)
    out = dict(node='P59v', files=dict(fit=FFIT, eval=FEVAL), seed=SEED, blocks=BLOCKS, sets={}, block={})
    S = {}
    for split, J in (('fit', Jf), ('eval', Je)):
        m, pt, y = jmass(J).astype(float), J['jet_pt'].astype(float), J['label'].astype(int)
        b, s = P59.make_pairs(m, pt, y, 'mass', npair, rng); Jb, Js, Jt, info = P59.pair_set(J, b, s, 'mass')
        S[split] = (Jb, Js, Jt); out['sets'][split] = info; log('pairs', split, info)
    # edit triple
    yE = Je['label']; rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(yE == c), ned, replace=False) for c in (iW, iZ)]))
    Eb = sub(Je, rows); mb = jmass(Eb).astype(float); n = len(rows); c = 1.05
    lD, eD = P59.bisect(lambda x: jmass(P59.dilate(Eb, x)).astype(float), 0.8, 1.3, c * mb); ED_ = P59.dilate(Eb, lD)
    pt_, de, dp, ms = kin(Eb); EB = build(Eb, pt_ * c, de, dp, ms * c)
    cc = np.full(n, c); lP, eP = P59.bisect(lambda x: jmass(P59.dilate(Eb, x, ptscale=cc)).astype(float), 0.8, 1.2, mb); EP = P59.dilate(Eb, lP, ptscale=cc)
    ED = dict(b=Eb, D=ED_, B=EB, P=EP)
    out['edits_info'] = dict(N=int(n), max_rel_mass_err=dict(D=float(eD.max()), P=float(eP.max())),
                             dlogm={k: float(np.median(np.log(jmass(v).astype(float) / mb))) for k, v in ED.items() if k != 'b'},
                             dlogpt={k: float(np.median(np.log(v['jet_pt'] / Eb['jet_pt']))) for k, v in ED.items() if k != 'b'},
                             dlog_own_spread={k: float(np.median(np.log(P59.own_spread(v) / P59.own_spread(Eb)))) for k, v in ED.items() if k != 'b'},
                             lam_D_median=float(np.median(lD)), lam_P_median=float(np.median(lP)))
    log('edits', out['edits_info'])
    net = P59.Net('trained', dev); log('net loaded')
    J0 = sample(Pm['P1'], 3000)
    with torch.no_grad():
        L0 = np.concatenate([net.logits(net.embed(J0, np.arange(s, min(len(J0['label']), s + 250)))).float().cpu().numpy() for s in range(0, len(J0['label']), 250)])
    out['P1_sample'] = dict(N=int(len(L0)), acc=float((L0.argmax(1) == J0['label']).mean()), expected=0.8373); log('P1', out['P1_sample'])
    PSf = P59.PairSet(net, *S['fit'], 100, 'fit'); ev = P59.PairSet(net, *S['eval'], 200, 'eval')
    Jb, Js, Jt = S['eval']; nec = P59.PairSet(net, Jt, Jb, None, 200, 'nec')  # base = counterfactual jet, source = original base
    PE = {k: P59.PairSet(net, v, None, None, 200, 'ed_' + k) for k, v in ED.items()}
    yb = Jb['label'].astype(int)
    out['unedited'] = dict(acc_eval_bases_10class=float((ev.Lb.argmax(1) == yb).mean()), frac_cf_call_changed=float((ev.Lt.argmax(1) != ev.Lb.argmax(1)).mean()),
                           nec_map_identity_frac=float(np.mean([float(d['tok_aligned'].mean()) for d in nec.B])))
    dd = {k: dW(PE[k].Lb) for k in PE}; Wd = P59.boot_W(n, 1000, SEED + 3)
    out['edits_network'] = dict(ratio_B=ratio_ci(dd['B'] - dd['b'], dd['D'] - dd['b'], Wd), ratio_P=ratio_ci(dd['P'] - dd['b'], dd['D'] - dd['b'], Wd),
                                mean_dd={k: float(np.mean(dd[k] - dd['b'])) for k in ('D', 'B', 'P')},
                                calls_changed={k: float((PE[k].Lb.argmax(1) != PE['b'].Lb.argmax(1)).mean()) for k in ('D', 'B', 'P')})
    log('edits network', out['edits_network'])
    Uo = {}
    for t in ('b23', 'b456', 'b7to11'):
        f = ULG / f'p59_U_{t}.npz'
        if f.exists():
            z = np.load(f); Uo.update({k: z[k] for k in z.files})
    W = P59.boot_W(ev.n, 1000, SEED + 4); Wn = P59.boot_W(nec.n, 1000, SEED + 5)
    perm_f = np.random.default_rng(SEED + 6).permutation(PSf.n)
    nb_null = list(range(int(np.ceil(min(nnull, ev.n) / 200)))); nn = min(ev.n, len(nb_null) * 200)
    for k in BLOCKS:
        tb = time.time(); R = {}
        for ps in (PSf, ev, nec, *PE.values()): ps.goto(k)
        R['no_patch'] = P59.metrics(ev.Lb, ev.Lt, ev.Lb, W)
        dirs = {}
        for tm in ('aligned', 'global'):
            key = f'k{k}_{tm}_aux_d1__aux'
            if key in Uo: dirs[f'orig_{tm}'] = (torch.from_numpy(Uo[key]).float().to(dev), tm)
        U40, l40 = P59.train_das(PSf, k, 'aux', 1, 'aligned', PSf.Lt, Pm, SEED + 100 * k + 1); dirs['refit40'] = (U40['aux'], 'aligned')
        P200 = dict(Pm, STEPS=200); U200, l200 = P59.train_das(PSf, k, 'aux', 1, 'aligned', PSf.Lt, P200, SEED + 100 * k + 2); dirs['refit200'] = (U200['aux'], 'aligned')
        Ush, _ = P59.train_das(PSf, k, 'aux', 1, 'aligned', PSf.Lt[perm_f], Pm, SEED + 100 * k + 3); dirs['shuffled40'] = (Ush['aux'], 'aligned')
        R['loss'] = dict(refit40=l40, refit200=l200)
        R['dirs'] = {}
        for nm, (u, tm) in dirs.items():
            Lp = P59.patched_logits(ev, k, {'aux': u}, tm); r = dict(P59.metrics(Lp, ev.Lt, ev.Lb, W), **P59.joint(Lp, ev.Lt, ev.Lb, ev.Ls, W))
            if nm != 'shuffled40':
                Lg = P59.patched_logits(ev, k, {'aux': u}, 'global'); r['global_patch'] = dict(P59.metrics(Lg, ev.Lt, ev.Lb, W), **P59.joint(Lg, ev.Lt, ev.Lb, ev.Ls, W))
                r['subset'] = dict(F=P59.quick(Lp[:nn], ev.Lt[:nn], ev.Lb[:nn])[2], beta=beta2(Lp[:nn], ev.Lt[:nn], ev.Lb[:nn], ev.Ls[:nn]).tolist())
                # necessity: cf jet with the direction set to the base jet's value
                for nt in ('aligned', 'global'):
                    Ln = P59.patched_logits(nec, k, {'aux': u}, nt); num, den = dW(Ln) - dW(nec.Ls), dW(nec.Lb) - dW(nec.Ls)
                    bs = (Wn @ (num * den)) / (Wn @ (den * den))
                    r[f'necessity_{nt}'] = dict(remaining=float((num * den).sum() / (den * den).sum()), remaining_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                calls_restored=float(((Ln.argmax(1) == nec.Ls.argmax(1)) & (nec.Lb.argmax(1) != nec.Ls.argmax(1))).sum() / max(1, (nec.Lb.argmax(1) != nec.Ls.argmax(1)).sum())))
                # edit triple projections on this direction (global token)
                pr = {}
                with torch.no_grad():
                    for en in PE:
                        a, _ = zip(*[P59.proj(d['hb'], d['eb'], 'aux', u) for d in PE[en].B]); pr[en] = torch.cat(a).cpu().numpy()
                sd = pr['b'].std() + 1e-12
                r['edits_proj'] = dict(ratio_B=ratio_ci(pr['B'] - pr['b'], pr['D'] - pr['b'], Wd), ratio_P=ratio_ci(pr['P'] - pr['b'], pr['D'] - pr['b'], Wd),
                                       dproj_over_sd={e: float(np.mean(pr[e] - pr['b']) / sd) for e in ('D', 'B', 'P')},
                                       corr_logm=float(np.corrcoef(pr['b'], np.log(mb))[0, 1]), corr_logpt=float(np.corrcoef(pr['b'], np.log(Eb['jet_pt']))[0, 1]),
                                       corr_log_m_over_pt=float(np.corrcoef(pr['b'], np.log(mb / Eb['jet_pt']))[0, 1]))
            R['dirs'][nm] = r
            log(f"block {k} {nm}: IIA {r['IIA']:.3f} IIA_ch {r['IIA_ch']:.3f} F {r['F']:.3f} {np.round(r['F_ci'], 3).tolist()} beta_cf {r['beta_cf']:.3f} "
                f"{np.round(r['beta_cf_ci'], 3).tolist()} beta_src {r['beta_src']:.3f}" + ('' if nm == 'shuffled40' else
                f" | global F {r['global_patch']['F']:.3f} | nec remaining {r['necessity_aligned']['remaining']:.3f} / global {r['necessity_global']['remaining']:.3f}"
                f" | proj ratio_B {np.round(r['edits_proj']['ratio_B'], 2).tolist()} ratio_P {np.round(r['edits_proj']['ratio_P'], 2).tolist()} sd {r['edits_proj']['dproj_over_sd']}"))
        # random null (aux d1) on the first nn eval pairs
        gen = np.random.default_rng(SEED + 1000 + k); nul = []
        for _ in range(ndraw):
            u = P59.rand_U('aux', 1, gen, dev); Lq = P59.patched_logits(ev, k, u, 'aligned', nb_null)
            nul.append([P59.quick(Lq, ev.Lt[:nn], ev.Lb[:nn])[2], *beta2(Lq, ev.Lt[:nn], ev.Lb[:nn], ev.Ls[:nn])])
        nul = np.array(nul)
        R['null'] = dict(n_pairs=int(nn), draws=int(ndraw), F_q=np.percentile(nul[:, 0], [2.5, 50, 97.5, 100]).tolist(),
                         beta_cf_q=np.percentile(nul[:, 1], [2.5, 50, 97.5, 100]).tolist(), beta_src_q=np.percentile(nul[:, 2], [2.5, 50, 97.5]).tolist())
        for nm in ('orig_aligned', 'refit40'):
            if nm in R['dirs']:
                s_ = R['dirs'][nm]['subset']; R['null'][f'p_F_{nm}'] = float((1 + np.sum(nul[:, 0] >= s_['F'])) / (1 + ndraw))
                R['null'][f'p_beta_{nm}'] = float((1 + np.sum(nul[:, 1] >= s_['beta'][0])) / (1 + ndraw))
        # necessity null
        gen2 = np.random.default_rng(SEED + 2000 + k); nn_ = []
        for _ in range(nnec):
            u = P59.rand_U('aux', 1, gen2, dev); Ln = P59.patched_logits(nec, k, u, 'aligned'); num, den = dW(Ln) - dW(nec.Ls), dW(nec.Lb) - dW(nec.Ls)
            nn_.append(float((num * den).sum() / (den * den).sum()))
        R['necessity_null_remaining_q'] = np.percentile(nn_, [0, 2.5, 50, 97.5, 100]).tolist()
        log(f"block {k} null F {np.round(R['null']['F_q'], 3).tolist()} beta_cf {np.round(R['null']['beta_cf_q'], 3).tolist()} p {[(a, b) for a, b in R['null'].items() if a.startswith('p_')]} "
            f"| necessity null remaining {np.round(R['necessity_null_remaining_q'], 3).tolist()} ({time.time() - tb:.0f}s)")
        out['block'][str(k)] = R; P59.jdump(out, OUT)
    log('done')


if __name__ == '__main__':
    try: main()
    except Exception: log('ERROR\n' + traceback.format_exc()); sys.exit(1)
