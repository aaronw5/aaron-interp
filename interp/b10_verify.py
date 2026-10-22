"""B10 verification: (1) rerun the B10 pipeline (interp.b10_programs_small_models) on a fresh sample, seed 3270 instead of 3170,
trained networks only, files tagged b10_verify*; (2) extra analyses, run on both the original B10 outputs (tag b10) and the rerun
(tag b10_verify), written to b10_verify_extra.json:
  A. turn10 vs ctrl10 program response (not only recall): on the same jets, per-jet flip indicator of the network, of each own K16
     program (fits 0-2) and of the truth program under the mass-lowering turn and under the mass-preserving control; flip rates,
     turn - ctrl difference with jet-bootstrap CI and exact McNemar (paired); own-program flip-kappa minus truth-program flip-kappa
     on turn10 with bootstrap CI.
  B. shift, own vs transfer vs truth with metrics that are not inflated by a program's flip rate: Cohen kappa of the flip
     indicators, precision (program flips that match the network's flip and new call), and same call on the shifted jets
     (program call on the edited jet = network call on the edited jet); plus the rank of each own fit among all 11 K16 programs
     (3 own fits + 8 donors' fit0) for each metric.
Run (repo root, ./sync.sh first):
  ./job.sh b10-verify-prep 'python -m interp.b10_verify prep && /opt/conda/bin/python -m interp.b10_verify torch'
  ./job.sh b10-verify-<fam> 'python -m interp.b10_verify fit --nets ...'
  ./job.sh b10-verify 'python -m interp.b10_verify agg && python -m interp.b10_verify extra'
"""
import argparse, json, sys, time
import numpy as np
from interp import b10_programs_small_models as B

VSEED = 3270
B.SEED = VSEED
B.tag = lambda smoke: 'b10_verify'
OUT = B.OUT
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def wkappa(W, nf, sf):
    """Cohen kappa of binary indicators nf, sf under bootstrap weights W (NB x J) -> (NB,)"""
    nf = nf.astype(np.float32); sf = sf.astype(np.float32); Ws = np.maximum(W.sum(1), 1)
    pn, ps = W @ nf / Ws, W @ sf / Ws; po = W @ (nf == sf).astype(np.float32) / Ws
    pe = pn * ps + (1 - pn) * (1 - ps); return np.where(pe < 1, (po - pe) / np.maximum(1 - pe, 1e-12), 0.0)


def kap(nf, sf):
    return float(wkappa(np.ones((1, len(nf)), np.float32), nf, sf)[0])


def extra_for(T, NB=1000):
    from interp import b9_programs_across_nets as B9
    Zp = np.load(OUT / f'{T}_prep.npz'); base = {e: Zp[f'base|{e}'] for e in B.EDITS}; Jt = len(Zp['y|test'])
    names = [n for n in B.TRAINED if (OUT / f'{T}_net_{n}.npz').exists()]
    ZZ = {n: dict(np.load(OUT / f'{T}_net_{n}.npz')) for n in names}
    rng = np.random.default_rng(3279); C = np.stack([np.bincount(rng.integers(0, Jt, Jt), minlength=Jt) for _ in range(NB)]).astype(np.float32)
    Cs = {'test': C, **{e: C[:, base[e]] for e in B.EDITS}}
    bm = lambda s, v: (Cs[s] @ np.asarray(v, np.float32)) / np.maximum(Cs[s].sum(1), 1)
    own = [f'prog{f}_K16' for f in range(3)]
    out = dict(tag=T, N_test=Jt, N_turn_ctrl=int(len(base['turn10'])), networks=names, turn_vs_ctrl={}, shift_metrics={})
    assert np.array_equal(base['turn10'], base['ctrl10'])
    # ---------------- A. turn vs ctrl
    rowsA = []
    for n in names:
        Z = ZZ[n]; b = base['turn10']; r = {}
        def flips(key):
            t = Z[f'{key}|turn10'] != Z[f'{key}|test'][b]; c = Z[f'{key}|ctrl10'] != Z[f'{key}|test'][b]; return t, c
        nt, nc_ = flips('netcall')
        for key in ['netcall'] + own + ['truth_K16']:
            t, c = flips(key); d = bm('turn10', t) - bm('turn10', c)
            r[key] = dict(turn_flip=float(t.mean()), ctrl_flip=float(c.mean()), turn_flips=int(t.sum()), ctrl_flips=int(c.sum()),
                          ratio=float(t.sum() / max(c.sum(), 1)), diff=float(t.mean() - c.mean()), diff_ci=B9.ci(d), mcnemar=B9.mcnemar(t, c))
            if key != 'netcall':
                kt, kc = wkappa(Cs['turn10'], nt, t), wkappa(Cs['turn10'], nc_, c)
                r[key].update(kappa_turn=kap(nt, t), kappa_ctrl=kap(nc_, c), kappa_turn_minus_ctrl_ci=B9.ci(kt - kc))
        # own (mean over fits) turn kappa minus truth turn kappa
        ko = np.mean([wkappa(Cs['turn10'], nt, flips(k)[0]) for k in own], 0); kt = wkappa(Cs['turn10'], nt, flips('truth_K16')[0])
        r['own_minus_truth_kappa_turn'] = dict(value=float(np.mean([r[k]['kappa_turn'] for k in own]) - r['truth_K16']['kappa_turn']),
                                               ci=B9.ci(ko - kt), p=B9.bootp(ko - kt))
        out['turn_vs_ctrl'][n] = r; rowsA.append(r)
        log(f'{T} {n}: turn/ctrl flips net {r["netcall"]["turn_flips"]}/{r["netcall"]["ctrl_flips"]}; own ' +
            ', '.join(f'{r[k]["turn_flips"]}/{r[k]["ctrl_flips"]} (p {r[k]["mcnemar"]["p"]:.2g})' for k in own) +
            f'; truth {r["truth_K16"]["turn_flips"]}/{r["truth_K16"]["ctrl_flips"]}; own-truth turn kappa {r["own_minus_truth_kappa_turn"]["value"]:+.3f} CI {r["own_minus_truth_kappa_turn"]["ci"]}')
    hp = B9.holm([r['own_minus_truth_kappa_turn']['p'] for r in rowsA])
    for r, h in zip(rowsA, hp): r['own_minus_truth_kappa_turn']['p_holm'] = h
    # ---------------- B. shift metrics: own vs transfer vs truth
    e = 'shift'; b = base[e]; rowsB = []
    for n in names:
        Z = ZZ[n]; nc0 = Z['netcall|test'][b]; nc1 = Z[f'netcall|{e}']; nf = nc1 != nc0
        def ind(D, key):
            s0, s1 = D[f'{key}|test'][b], D[f'{key}|{e}']; sf = s1 != s0; hit = nf & sf & (s1 == nc1); return sf, hit, s1 == nc1
        progs = {f'own{f}': ind(Z, k) for f, k in enumerate(own)}
        progs.update({f'donor:{A}': ind(ZZ[A], 'prog0_K16') for A in names if A != n}); progs['truth'] = ind(Z, 'truth_K16')
        met = {}
        for k, (sf, hit, same) in progs.items():
            met[k] = dict(recall=float(hit.sum() / max(nf.sum(), 1)), precision=float(hit.sum() / max(sf.sum(), 1)), kappa=kap(nf, sf),
                          same_call_edited=float(same.mean()), program_flip_rate=float(sf.mean()))
        dn = [k for k in progs if k.startswith('donor:')]; ow = [f'own{f}' for f in range(3)]
        r = dict(network_flip_rate=float(nf.mean()), per_program=met, compare={}, rank={})
        W = Cs[e]; den = np.maximum(bm(e, nf), 1e-9)
        bmet = dict(recall=lambda sf, hit, same: bm(e, hit) / den,
                    precision=lambda sf, hit, same: bm(e, hit) / np.maximum(bm(e, sf), 1e-9),
                    kappa=lambda sf, hit, same: wkappa(W, nf, sf),
                    same_call_edited=lambda sf, hit, same: bm(e, same))
        for mname, fn in bmet.items():
            bo = np.mean([fn(*progs[k]) for k in ow], 0); bt = np.mean([fn(*progs[k]) for k in dn], 0); bu = fn(*progs['truth'])
            vo = float(np.mean([met[k][mname] for k in ow])); vt = float(np.mean([met[k][mname] for k in dn])); vu = met['truth'][mname]
            vals = sorted([met[k][mname] for k in ow + dn], reverse=True)
            r['compare'][mname] = dict(own=vo, own_range=float(np.ptp([met[k][mname] for k in ow])), transfer=vt, truth=vu,
                                       own_minus_transfer=vo - vt, own_minus_transfer_ci=B9.ci(bo - bt), own_minus_transfer_p=B9.bootp(bo - bt),
                                       transfer_minus_truth=vt - vu, transfer_minus_truth_ci=B9.ci(bt - bu), transfer_minus_truth_p=B9.bootp(bt - bu))
            r['rank'][mname] = [1 + vals.index(met[k][mname]) for k in ow]   # 1 = best of 11
        out['shift_metrics'][n] = r; rowsB.append(r)
        log(f'{T} {n} shift: ' + '; '.join(f'{m} own {r["compare"][m]["own"]:.3f} tr {r["compare"][m]["transfer"]:.3f} truth {r["compare"][m]["truth"]:.3f} ranks {r["rank"][m]}' for m in bmet))
    for mname in ['recall', 'precision', 'kappa', 'same_call_edited']:
        for key in ('own_minus_transfer_p', 'transfer_minus_truth_p'):
            hp = B9.holm([r['compare'][mname][key] for r in rowsB])
            for r, h in zip(rowsB, hp): r['compare'][mname][key + '_holm'] = h
    return out


def extra(tags, name):
    res = dict(node='B10-verify', doc=__doc__, seed_original=3170, seed_verify=VSEED, results={})
    for T in tags:
        if (OUT / f'{T}_prep.npz').exists(): res['results'][T] = extra_for(T)
    (OUT / f'{name}.json').write_text(json.dumps(B.clean(res), indent=1)); log(f'wrote {name}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prep', 'torch', 'fit', 'agg', 'extra']); ap.add_argument('--nets', default='')
    ap.add_argument('--tags', default='b10,b10_verify'); ap.add_argument('--name', default='b10_verify_extra')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); log(f'B10 verify, seed {VSEED}, mode {a.mode}')
    if a.mode == 'prep':
        B.JAXNETS[:] = ['jedi_off', 'pfn_s0', 'pfn_s1']; B.prep(False)
    elif a.mode == 'torch':
        for k in [k for k in B.TORCH if k.endswith('u0')]: B.TORCH.pop(k)
        B.torch_nets(False)
    elif a.mode == 'fit':
        for nm in a.nets.split(','):
            t = time.time(); B.run_net(nm, False); log(f'{nm} done in {time.time() - t:.0f} s')
    elif a.mode == 'agg': B.agg(False)
    else: extra(a.tags.split(','), a.name)
