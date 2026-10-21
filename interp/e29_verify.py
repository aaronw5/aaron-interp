"""E29 verifier: rerun the key part of E29 (EAP-IG + ACDC-style circuits, hand-program variants, McNemar, random-circuit null)
on a fresh sample, and add two controls the E29 run did not have.

Fresh sample: fit / discovery = sample(1000, 3901) (ACDC on rows [::5]); test = sample(1600, 3902) minus jets shared with fit;
nulls on test[::8]. Donor rng 3903 (fit) / 3904 (test). One corruption per invocation (--corr mean | resample).
Reuses the E29 classes unchanged (MNet edge-masked forward, Graph, eap, acdc, Scorer, hand_circuits).

Nulls for the ACDC circuits at tau 0.003 and 0.001 (literal node-level circuits, F on test[::8]):
 (i)   unpaired: E29's random_nodes (same count per layer from live units; u and S channels drawn independently);
 (ii)  paired: same count per layer, but the u and S units share channels (u_c -> S_c is an identity pooling edge, so a u unit
       whose S unit is corrupted carries nothing) and the P channels are drawn with their a channel kept; this is the
       comparison set that has the same structure as the found circuits (whose u and S sets coincide);
 (iii) S-swap: the found circuit with only its u/S channel set replaced by a random paired set of the same size from the live
       S channels; tests whether the specific second-round objects chosen matter given everything else;
 (iv)  g-swap (tau 0.003 only): the found circuit with its g units replaced by every other g set of the same size (exhaustive).
Also: live units on test that are not live on fit (always computed clean, outside the graph).
Usage: python -m interp.e29_verify --corr mean|resample [--smoke]  ->  $AI_RESULTS/jedi/e29_verify_<corr>.json
"""
import argparse, json, itertools, time
import numpy as np
from interp import e29_auto_circuit_jedi as E

log = E.log


def paired_random(nodes, live, rng):
    d = E.stage_sets(nodes); out = []
    for st in ('g', 's', 'h4', 'h5', 'z'):
        if d[st]: out += [(st, int(i)) for i in rng.choice(live[st], min(len(d[st]), len(live[st])), replace=False)]
    pool = sorted(set(live['S']) & set(live['u'])); k = min(len(pool), max(len(d['S']), len(d['u'])))
    ch = list(rng.choice(pool, k, replace=False)) if k else []
    out += [('S', int(c)) for c in ch[:len(d['S'])]] + [('u', int(c)) for c in ch[:len(d['u'])]]
    P = [int(c) for c in rng.choice(live['P'], min(len(d['P']), len(live['P'])), replace=False)] if d['P'] else []
    a = [c for c in P if c in live['a']]; rest = [c for c in live['a'] if c not in a]
    a += [int(c) for c in rng.choice(rest, min(len(rest), max(0, len(d['a']) - len(a))), replace=False)]
    out += [('P', c) for c in P] + [('a', c) for c in a]
    return out


def swap_S(nodes, live, rng):
    d = E.stage_sets(nodes); keep = [n for n in nodes if n[0] not in ('u', 'S')]
    pool = sorted(set(live['S']) & set(live['u'])); k = min(len(pool), max(len(d['S']), len(d['u'])))
    ch = [int(c) for c in rng.choice(pool, k, replace=False)]
    return keep + [('S', c) for c in ch[:len(d['S'])]] + [('u', c) for c in ch[:len(d['u'])]], ch[:len(d['S'])]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--corr', default='mean'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--nnull', type=int, default=500); ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args(); corr = a.corr
    taus = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    if a.smoke:
        data = E.load(40, 3901, 40, 3902); a.nnull, a.boot = 5, 50; taus = [1e-3, 3e-3]
    else:
        data = E.load(1000, 3901, 1600, 3902)
    xf, yf, ptf, xt, yt, ptt = data
    net = E.MNet(64); xinf, xint = net.xin(xf), net.xin(xt); net.dummyC(xinf[:1]); Z = net._Z0; Mone = net.ones()
    out = dict(node='E29-verify', corr=corr, sample=dict(fit=[1000, 3901], test=[1600, 3902], N_test=len(yt), N_fit=len(yf)), taus=taus)
    lo_mine = np.concatenate([np.asarray(net._logits(xint[i:i + 1000], Z, Mone), np.float64) for i in range(0, len(xint), 1000)])
    lo_ref = net.J.run(xt)['logits']; call_t = lo_ref.argmax(1)
    out['check_all_on'] = dict(max_abs_dlogit=float(np.abs(lo_mine - lo_ref).max()), calls_differ=int((lo_mine.argmax(1) != call_t).sum()))
    log('all-on check', out['check_all_on'])

    def stats(xin):
        ssum, smax, smin = {}, {}, {}
        for i in range(0, len(xin), 1000):
            r = net._full(xin[i:i + 1000], Z, Mone)
            for k in ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z'):
                v = np.asarray(r[k]); f = v.reshape(-1, v.shape[-1])
                ssum[k] = ssum.get(k, 0) + v.astype(np.float64).sum(0, keepdims=True)
                smax[k] = np.maximum(smax.get(k, -np.inf), f.max(0)); smin[k] = np.minimum(smin.get(k, np.inf), f.min(0))
        return ({k: (ssum[k] / len(xin)).astype(np.float32) for k in ssum},
                {k: [int(c) for c in np.flatnonzero(smax[k] - smin[k] > 1e-9)] for k in ssum})
    Cmean, live = stats(xinf); _, live_t = stats(xint)
    ref = json.loads((E.OUT / 'e29_auto_circuit.json').read_text())['networks']['n64'] if (E.OUT / 'e29_auto_circuit.json').exists() else None
    out['live'] = live; out['live_same_as_E29'] = (ref is not None and live == ref['live'])
    out['live_on_test_not_fit'] = {k: sorted(set(live_t[k]) - set(live[k])) for k in live}
    log('live per stage', {k: len(v) for k, v in live.items()}, 'same as E29', out['live_same_as_E29'], 'live on test only', out['live_on_test_not_fit'])
    gr = E.Graph(net, live); hc = E.hand_circuits(live); A_ = hc['handA']
    allz = [('z', i) for i in live['z']]; allh = [(st, i) for st in ('h4', 'h5', 'z') for i in live[st]]
    up = [n for n in A_ if n[0] not in ('h4', 'h5', 'z')]
    V = dict(handA=A_, handA_allz=sorted(set(A_) | set(allz)), handA_allhead=sorted(set(up) | set(allh)),
             handA_allhead_noctxS=sorted((set(up) | set(allh)) - {('u', 0), ('S', 0), ('u', 30), ('S', 30)}),
             handA_allhead_g49only=sorted((set(up) | set(allh)) - {('g', i) for i in live['g'] if i != 49}),
             handB_allhead=sorted(set(up) | set(allh) | {('u', 31), ('u', 58), ('S', 31), ('S', 58)}))
    don_f = E.donors(ptf, yf, np.random.default_rng(3903)); don_t = E.donors(ptt, yt, np.random.default_rng(3904))
    call_f = net.J.run(xf)['logits'].argmax(1)
    # ---- EAP-IG on fit jets
    evf = E.Evaluator(net, xinf, corr, Cmean, xinf[don_f], batch=1000)
    A, A0 = E.eap(net, gr, xinf, lambda i: Cmean if corr == 'mean' else evf.donorC(i, 500), call_f, 5, 500)
    Mz = {nm: (1 - gr.G[nm]).astype(np.float32) for nm in gr.G}
    out['eap'] = dict(sum_A=float(A.sum()), m_clean_minus_m_corrupt=E.metric_np(evf.logits(Mone), call_f) - E.metric_np(evf.logits(Mz), call_f))
    log('EAP-IG', out['eap'])
    # ---- test scorer
    evt = E.Evaluator(net, xint, corr, Cmean, xint[don_t], batch=1000)
    sc = E.Scorer(yt, evt.logits(Mone), evt.logits(Mz), a.boot, np.random.default_rng(3905))
    out['acc_full'] = sc.acc_full; out['acc_full_ci'] = np.percentile(sc.cf[sc.bi].mean(1), [2.5, 97.5]).tolist(); out['acc_zero'] = sc.acc_zero
    log(f'test acc full {sc.acc_full:.4f} {out["acc_full_ci"]}, all corrupted {sc.acc_zero:.4f}')

    def Fboot(cc):
        return (cc[sc.bi].mean(1) - sc.cz[sc.bi].mean(1)) / (sc.cf[sc.bi].mean(1) - sc.cz[sc.bi].mean(1))
    # ---- ACDC
    ia = np.arange(0, len(xinf), 5)
    Ca = Cmean if corr == 'mean' else E.Evaluator(net, xinf[ia], corr, Cmean, xinf[don_f[ia]]).donorC(0, len(ia))
    p0 = E.softmax(np.asarray(net._logits(xinf[ia], Ca, Mone), np.float64))
    circ = {}; ccv = {}
    for res in E.acdc(net, gr, xinf[ia], Ca, p0, A, taus):
        nodes = gr.nodes_of(res['edges']); r = sc.score(evt.logits(gr.masks_from_nodes(nodes)), keep_vec=True); ccv[res['tau']] = r.pop('_cc')
        d = E.stage_sets(nodes)
        circ[res['tau']] = dict(n_edges=len(res['edges']), n_nodes=len(nodes), node_level=r, per_stage={k: len(v) for k, v in d.items()},
                                S=d['S'], u=d['u'], g=d['g'], P=d['P'], a=d['a'], n_named_S17=len(set(d['S']) & set(E.NAMED_S17)),
                                nodes=[[s, int(i)] for s, i in nodes])
        log(f'ACDC tau {res["tau"]:g}: {len(nodes)} units, F {r["F"]:.3f} {[round(v, 3) for v in r["F_ci"]]} acc {r["acc"]:.4f}; '
            f'a {len(d["a"])} S {d["S"]} (named17 {circ[res["tau"]]["n_named_S17"]}) g {d["g"]} P {len(d["P"])}')
    out['acdc'] = {str(k): v for k, v in circ.items()}
    # ---- hand variants + McNemar + paired bootstrap of the F difference
    hv = {}
    for vn, nodes in V.items():
        r = sc.score(evt.logits(gr.masks_from_nodes(nodes)), keep_vec=True); ccv[vn] = r.pop('_cc'); r['n_nodes'] = len(nodes); hv[vn] = r
        log(f'{vn} ({len(nodes)} units): F {r["F"]:.3f} {[round(v, 3) for v in r["F_ci"]]} acc {r["acc"]:.4f}')
    out['hand'] = hv; cmp = []
    for t in taus:
        if not (60 <= circ[t]['n_nodes'] <= 170): continue
        for vn in ('handA', 'handA_allz', 'handA_allhead'):
            dF = Fboot(ccv[t]) - Fboot(ccv[vn])
            cmp.append(dict(tau=t, n=circ[t]['n_nodes'], F=circ[t]['node_level']['F'], hand=vn, hand_n=hv[vn]['n_nodes'], hand_F=hv[vn]['F'],
                            dF=circ[t]['node_level']['F'] - hv[vn]['F'], dF_ci=np.nanpercentile(dF, [2.5, 97.5]).tolist(),
                            mcnemar=E.mcnemar(ccv[t], ccv[vn])))
            log(f'ACDC {t:g} ({circ[t]["n_nodes"]}) vs {vn} ({hv[vn]["n_nodes"]}): dF {cmp[-1]["dF"]:+.3f} {[round(v, 3) for v in cmp[-1]["dF_ci"]]} McNemar {cmp[-1]["mcnemar"]}')
    out['found_vs_hand'] = cmp
    # ---- nulls on test[::8]
    sub = np.arange(0, len(xt), 8)
    evr = E.Evaluator(net, xint[sub], corr, Cmean, xint[don_t[sub]], batch=len(sub), cache=True)
    scr = E.Scorer(yt[sub], evr.logits(Mone), evr.logits(Mz), 200, np.random.default_rng(3907))
    Fs = lambda nodes: scr.score(evr.logits(gr.masks_from_nodes(nodes)))['F']
    rng = np.random.default_rng(3906 + (corr == 'resample')); nulls = {}; pl = []
    for t, kinds in ((3e-3, ('unpaired', 'paired', 'swapS', 'swapg')), (1e-3, ('paired', 'swapS'))):
        if t not in circ: continue
        nodes = [tuple(v) for v in circ[t]['nodes']]; obs = Fs(nodes); R = dict(obs_sub=obs, n_nodes=len(nodes))
        for kind in kinds:
            t1 = time.time()
            if kind == 'swapg':
                d = E.stage_sets(nodes); keep = [n for n in nodes if n[0] != 'g']; rows = []
                for comb in itertools.combinations(live['g'], len(d['g'])):
                    rows.append(dict(g=list(map(int, comb)), F=Fs(keep + [('g', int(c)) for c in comb])))
                fl = np.array([r_['F'] for r_ in rows])
                R[kind] = dict(n=len(rows), found_g=d['g'], mean=float(fl.mean()), max=float(fl.max()), min=float(fl.min()),
                               rank_of_found=int((fl > obs + 1e-12).sum()) + 1, top=sorted(rows, key=lambda r_: -r_['F'])[:8])
                log(f'null tau {t:g} {kind}: found g {d["g"]} F {obs:.3f}; all {len(rows)} g sets: mean {fl.mean():.3f} max {fl.max():.3f} rank {R[kind]["rank_of_found"]}')
                continue
            fl, named = [], []
            for _ in range(a.nnull):
                if kind == 'unpaired': fl.append(Fs(E.random_nodes(nodes, live, rng)))
                elif kind == 'paired': fl.append(Fs(paired_random(nodes, live, rng)))
                else:
                    nn, ch = swap_S(nodes, live, rng); fl.append(Fs(nn)); named.append(len(set(ch) & set(E.NAMED_S17)))
            fl = np.array(fl); p = float((1 + (fl >= obs).sum()) / (1 + len(fl)))
            R[kind] = dict(n=len(fl), mean=float(np.nanmean(fl)), q95=float(np.nanpercentile(fl, 95)), max=float(np.nanmax(fl)), p=p,
                           ci_mean=np.percentile([np.mean(np.random.default_rng(b).choice(fl, len(fl))) for b in range(200)], [2.5, 97.5]).tolist())
            if kind == 'swapS':
                named = np.array(named); R[kind]['named_S17_in_draw'] = dict(zip(*[v.tolist() for v in np.unique(named, return_counts=True)]))
                R[kind]['F_by_named'] = {int(k): float(fl[named == k].mean()) for k in np.unique(named)}
                R[kind]['spearman_F_named'] = float(__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(fl, named)[0]) if len(set(named)) > 1 else float('nan')
            pl.append((f'{t:g}_{kind}', p))
            log(f'null tau {t:g} {kind}: obs {obs:.3f}; null mean {R[kind]["mean"]:.3f} q95 {R[kind]["q95"]:.3f} max {R[kind]["max"]:.3f} p {p:.4f} '
                f'({len(fl)} draws, {time.time() - t1:.0f} s)' + (f'; F by #named S17 {R[kind]["F_by_named"]}' if kind == 'swapS' else ''))
        nulls[str(t)] = R
    out['nulls'] = nulls
    hp = E.holm([p for _, p in pl]); out['holm'] = [dict(test=k, p=p, p_holm=q) for (k, p), q in zip(pl, hp)]
    log('Holm', out['holm'])
    E.OUT.mkdir(parents=True, exist_ok=True)
    pth = E.OUT / f'e29_verify_{corr}{"_smoke" if a.smoke else ""}.json'
    pth.write_text(json.dumps(out, default=float)); log('wrote', pth)


if __name__ == '__main__':
    main()
