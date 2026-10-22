"""B9 verification (fresh jets): rerun the key parts of B9 on new samples and add a same-network refit control.

Samples (jedi_core.sample rows): fit = sample(1000, 5901); fit2 = sample(1000, 5904) minus fit rows (second, disjoint fit set,
  for the refit control); dev = sample(1000, 5902) minus fit / fit2 rows; test = sample(1000, 5903) minus fit / fit2 / dev rows.
Networks: the 7 edit-capable B9 networks (official n64, n32, perm-inv n64, HGQ-r150 s0 / s2, float s0 / s1).
Per network (same B9 functions: pool, MARS settings, DL, view with slots >= 32 zeroed for n32):
  own   program K16 (and K8) fitted to the network's 5 logits on fit;
  refit program K16 fitted to the same network's logits on fit2 (independent jets) -> fit-noise floor for "own - transfer"
        and self-recurrence (share of own terms matched in the refit program);
  truth program K16 fitted to the labels on fit (target log(0.95 onehot + 0.01));
  STD   MARS on the 10 standard observables K2-16 (matched DL member: bits <= own K16, best dev same-call).
Edit: shift (deta + 0.06 rad for every real particle) on the test jets.
Aggregation: per-network same-call / shift recall; own - refit, own - truth (jet bootstrap 1,000, McNemar); transfer pairs
  (own - transfer, transfer - truth) over all 7 networks and over the 6 networks without n32 (network bootstrap 2,000);
  recurrence share (term matched in: refit of the same network / other networks' own programs / truth program).
Run: python -m interp.b9_verify net --nets official_n64   -> $AI_RESULTS/jedi/b9_verify_net_<name>.json
     python -m interp.b9_verify agg                       -> $AI_RESULTS/jedi/b9_verify.json
"""
import argparse, json, os, sys, time, gc
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import b9_programs_across_nets as B9   # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'
NETS = ['official_n64', 'official_n32', 'perminv_n64', 'hgq150_s0', 'hgq150_s2', 'float_s0', 'float_s1']
SEEDS = dict(fit=5901, fit2=5904, dev=5902, test=5903)
log = B9.log


def load_sets(name, per=1000):
    from interp import e33_program_recurrence_jedi as E33
    from jetdistill.config import RESULTS
    _, fam, n = E33.MODELS[name]
    yall = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idx = {s: E33.rows(yall, per, sd) for s, sd in SEEDS.items()}
    prev, keep = [], {}
    for s in ('fit', 'fit2', 'dev', 'test'):
        keep[s] = ~np.isin(idx[s], np.concatenate(prev)) if prev else np.ones(len(idx[s]), bool); prev.append(idx[s])
    X3, XI, Y = {}, {}, {}
    for s, sd in SEEDS.items():
        xin, _, x3, y, _ = E33.load(per, sd, n, False); k = keep[s]
        X3[s] = np.ascontiguousarray(x3[k], np.float32); XI[s] = np.ascontiguousarray(xin[k], np.float32); Y[s] = y[k]
    return X3, XI, Y, {s: int(keep[s].sum()) for s in SEEDS}, n, idx


def run_net(name):
    R = dict(network=name)
    X3, XI, Y, R['N'], n, idx = load_sets(name)
    xT = X3['test']; real = xT[..., 0] > 0; xs = xT.copy(); xs[..., 1] += np.float32(0.06) * real
    net = B9.Runner(name)
    L = {s: net.run(XI[s]) for s in ('fit', 'fit2', 'dev', 'test')}
    L['shift'] = net.run(np.ascontiguousarray(xs[:, :n]) if n < 64 else xs)
    NC = {s: v.argmax(1) for s, v in L.items()}
    R['acc'] = {s: float((NC[s] == Y[s]).mean()) for s in ('fit', 'fit2', 'dev', 'test')}
    R['net_flip_shift'] = float((NC['shift'] != NC['test']).mean())
    log(name, 'N', R['N'], 'acc', R['acc'], 'shift flip', R['net_flip_shift'])
    sets = dict(fit=X3['fit'], fit2=X3['fit2'], dev=X3['dev'], test=xT, shift=xs)
    F, S = {}, {}
    for s, xx in sets.items():
        F[s], keys, S[s] = B9.feats(B9.view(xx, n))
    calls, meta = {}, {}

    def add(nm, bits, fn, terms=None):
        c = {s: np.asarray(fn(s), np.int8) for s in ('dev', 'test', 'shift')}
        calls[nm] = {s: c[s].tolist() for s in ('test', 'shift')}
        meta[nm] = dict(bits=float(bits), same_call_dev=float((c['dev'] == NC['dev']).mean()),
                        same_call_test=float((c['test'] == NC['test']).mean()))
        if terms is not None: meta[nm]['terms'] = terms
        log(f'{name}: {nm:10s} bits {bits:8.0f} same call dev {meta[nm]["same_call_dev"] * 100:.2f} test {meta[nm]["same_call_test"] * 100:.2f}')

    def prog(nm, Xf, Tf, Xd, Td, ks, Kmax, Ks, src):
        fits, Q = B9.mars_path(Xf, Tf, Xd, Td, ks, Kmax)
        for K in Ks:
            parts, bits = B9.mars_prefix(fits, Q, Tf, ks, K)
            terms = [[B9.term_record(t, Q['fit']) for t in tr] for tr, _, _ in parts] if src else None
            add(f'{nm}_K{K}', bits, lambda s, p=parts: B9.mars_predict(p, src[s], ks).argmax(1), terms)
    t = time.time()
    prog('own', F['fit'], L['fit'], F['dev'], L['dev'], keys, 16, (8, 16), F)
    prog('refit', F['fit2'], L['fit2'], F['dev'], L['dev'], keys, 16, (16,), F)
    Tt = lambda y: np.log(0.95 * np.eye(5)[y] + 0.01)
    prog('truth', F['fit'], Tt(Y['fit']), F['dev'], Tt(Y['dev']), keys, 16, (16,), F)
    from interp.b8_fidelity_dl_iia import STD_NAMES
    prog('std', S['fit'], L['fit'], S['dev'], L['dev'], STD_NAMES, 16, (2, 4, 8, 16), S)
    log(f'{name}: programs in {time.time() - t:.0f} s')
    R.update(meta=meta, calls=calls, netcall={s: NC[s].astype(int).tolist() for s in ('test', 'shift')}, y_test=Y['test'].tolist(),
             test_rows_hash=int(np.sum(idx['test'][:50])))
    (OUT / f'b9_verify_net_{name}.json').write_text(json.dumps(B9.clean(R)))
    log(f'{name}: wrote b9_verify_net_{name}.json')


def agg(NB=1000):
    names = [n for n in NETS if (OUT / f'b9_verify_net_{n}.json').exists()]
    RJ = {n: json.loads((OUT / f'b9_verify_net_{n}.json').read_text()) for n in names}
    for n in names: assert RJ[n]['y_test'] == RJ[names[0]]['y_test']
    A = lambda v: np.asarray(v)
    J = len(RJ[names[0]]['y_test']); rng = np.random.default_rng(5909)
    C = np.stack([np.bincount(rng.integers(0, J, J), minlength=J) for _ in range(NB)]).astype(np.float32)
    bm = lambda v: C @ np.asarray(v, np.float32) / J
    out = dict(node='B9-verify', procedure=__doc__, seeds=SEEDS, networks=names, N=RJ[names[0]]['N'], per_network={})
    def ind(n, nm):
        nc0, nc1 = A(RJ[n]['netcall']['test']), A(RJ[n]['netcall']['shift'])
        s0, s1 = A(RJ[n]['calls'][nm]['test']), A(RJ[n]['calls'][nm]['shift'])
        nf, sf, ag, hit = B9.edit_ind(nc0, nc1, s0, s1)
        return s0 == nc0, ag, hit, nf
    for n in names:
        R = RJ[n]; m = R['meta']; o = dict(acc=R['acc'], net_flip_shift=R['net_flip_shift'], surrogates={})
        for nm in m:
            same, ag, hit, nf = ind(n, nm)
            o['surrogates'][nm] = dict(bits=m[nm]['bits'], same_call=float(same.mean()), same_call_ci=B9.ci(bm(same)),
                                       shift_agree=float(ag.mean()), shift_recall=float(hit.sum() / max(nf.sum(), 1)),
                                       shift_recall_ci=B9.ci(bm(hit) / np.maximum(bm(nf), 1e-9)))
        own = ind(n, 'own_K16')
        for other in ('refit_K16', 'truth_K16'):
            oo = ind(n, other)
            o[f'own_minus_{other}'] = dict(nat=float(own[0].mean() - oo[0].mean()), nat_ci=B9.ci(bm(own[0]) - bm(oo[0])),
                                            nat_mcnemar=B9.mcnemar(own[0], oo[0]),
                                            shift_recall=float((own[2].sum() - oo[2].sum()) / max(own[3].sum(), 1)),
                                            shift_recall_ci=B9.ci((bm(own[2]) - bm(oo[2])) / np.maximum(bm(own[3]), 1e-9)),
                                            shift_mcnemar=B9.mcnemar(own[2][own[3]], oo[2][own[3]]))
        std = [k for k in m if k.startswith('std_') and m[k]['bits'] <= m['own_K16']['bits']]
        bs = max(std, key=lambda k: m[k]['same_call_dev']); sb = ind(n, bs)
        o['matched_std'] = dict(member=bs, bits=m[bs]['bits'], nat_diff=float(own[0].mean() - sb[0].mean()), nat_ci=B9.ci(bm(own[0]) - bm(sb[0])),
                                nat_mcnemar=B9.mcnemar(own[0], sb[0]), shift_agree_diff=float(own[1].mean() - sb[1].mean()),
                                shift_agree_ci=B9.ci(bm(own[1]) - bm(sb[1])), shift_mcnemar=B9.mcnemar(own[1], sb[1]))
        out['per_network'][n] = o
        log(n, 'acc', round(R['acc']['test'] * 100, 2), 'own', round(o['surrogates']['own_K16']['same_call'] * 100, 2), 'refit',
            round(o['surrogates']['refit_K16']['same_call'] * 100, 2), 'truth', round(o['surrogates']['truth_K16']['same_call'] * 100, 2),
            'shift recall own', round(o['surrogates']['own_K16']['shift_recall'] * 100, 1), 'matched std', bs, round(o['matched_std']['nat_diff'] * 100, 2),
            round(o['matched_std']['shift_agree_diff'] * 100, 2))
    # holm over networks for matched STD
    for key in ('nat_mcnemar', 'shift_mcnemar'):
        hp = B9.holm([out['per_network'][n]['matched_std'][key]['p'] for n in names])
        for n, h in zip(names, hp): out['per_network'][n]['matched_std'][key]['p_holm'] = h
    # transfer
    pairs = {}
    for b in names:
        nc0, nc1 = A(RJ[b]['netcall']['test']), A(RJ[b]['netcall']['shift'])
        def sc(src, nm):
            s0, s1 = A(RJ[src]['calls'][nm]['test']), A(RJ[src]['calls'][nm]['shift'])
            _, _, ag, hit = B9.edit_ind(nc0, nc1, s0, s1); nf = nc1 != nc0
            return float((s0 == nc0).mean()), float(hit.sum() / max(nf.sum(), 1)), float(ag.mean())
        own, refit, tru = sc(b, 'own_K16'), sc(b, 'refit_K16'), sc(b, 'truth_K16')
        for a in names:
            if a != b:
                tr = sc(a, 'own_K16'); pairs[(a, b)] = dict(own=own, refit=refit, tr=tr, truth=tru)
    def summ(fn, nets, B=2000):
        vals = [fn(pairs[(a, b)]) for a in nets for b in nets if a != b]
        r_ = np.random.default_rng(5911); bt = []
        for _ in range(B):
            dr = r_.choice(nets, len(nets), replace=True)
            vv = [fn(pairs[(a, b)]) for i, a in enumerate(dr) for j, b in enumerate(dr) if i != j and a != b]
            bt.append(np.mean(vv) if vv else np.nan)
        return dict(mean=float(np.mean(vals)), n_pairs=len(vals), ci=B9.ci(bt), p=B9.bootp(bt))
    T = {}
    for lab, nets in (('all7', names), ('no_n32', [n for n in names if n != 'official_n32'])):
        T[lab] = {}
        for k, i in (('nat', 0), ('shift_recall', 1), ('shift_agree', 2)):
            T[lab][f'{k}_own_minus_transfer'] = summ(lambda p, i=i: p['own'][i] - p['tr'][i], nets)
            T[lab][f'{k}_transfer_minus_truth'] = summ(lambda p, i=i: p['tr'][i] - p['truth'][i], nets)
            T[lab][f'{k}_own_minus_refit'] = summ(lambda p, i=i: p['own'][i] - p['refit'][i], nets)
            T[lab][f'{k}_refit_minus_transfer'] = summ(lambda p, i=i: p['refit'][i] - p['tr'][i], nets)
        for k, v in T[lab].items():
            log(f'transfer {lab} {k}: {v["mean"] * 100:+.2f} pp CI [{v["ci"][0] * 100:+.2f}, {v["ci"][1] * 100:+.2f}] p {v["p"]:.3g} ({v["n_pairs"]} pairs)')
    out['transfer'] = T
    out['transfer_pairs'] = [dict(A=a, B=b, **v) for (a, b), v in pairs.items()]
    # recurrence: share of own K16 terms matched in (refit same net, other nets' own, truth)
    def terms(n, nm): return [t for tt in RJ[n]['meta'][nm]['terms'] for t in tt]
    rec = {}
    for n in names:
        P = terms(n, 'own_K16'); cont = lambda prog, t: any(B9.term_match(t, u) for u in prog)
        others = [m for m in names if m != n]
        rec[n] = dict(n_terms=len(P), in_refit=float(np.mean([cont(terms(n, 'refit_K16'), t) for t in P])),
                      in_other_mean=float(np.mean([np.mean([cont(terms(m, 'own_K16'), t) for t in P]) for m in others])),
                      in_other_refits_mean=float(np.mean([np.mean([cont(terms(m, 'refit_K16'), t) for t in P]) for m in others])),
                      in_truth=float(np.mean([cont(terms(n, 'truth_K16'), t) for t in P])),
                      mean_count_other=float(np.mean([sum(cont(terms(m, 'own_K16'), t) for m in others) for t in P])), of=len(others))
        log('recurrence', n, json.dumps({k: round(v, 3) for k, v in rec[n].items()}))
    out['recurrence'] = rec
    (OUT / 'b9_verify.json').write_text(json.dumps(B9.clean(out), indent=1))
    log('wrote b9_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['net', 'agg']); ap.add_argument('--nets', default='')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == 'net':
        for nm in a.nets.split(','): run_net(nm); gc.collect()
    else:
        agg()
