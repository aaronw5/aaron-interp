"""E39x verification (skeptical rerun on fresh jets, fresh untrained redraw, and the controls E39x did not run).

Question: does the E39x reading hold on a new sample, and are its two headline comparisons fair?
  (i)  "4 directions of A's S under B's loss reach 87 % of B's calls, more than A's class scores": E39x compared the task-loss
       rank-4 map only with a RIDGE map from A's 5 logits.  Missing controls, run here: the same task-loss training applied to
       (a) A's 5 logits (full 5 -> 64 map, and rank 4), (b) the untrained redraw of A (rank 4 and full).  If logits under task loss
       reach the same same-call %, the 4-direction result does not show content beyond A's class scores.
  (ii) "named S role units map across networks better than random units": E39x drew random units in BOTH A and B.  B's own
       named units already carry more of B's decision than B's random units (E39x subset_own), so the gap can come from the
       choice of B's units alone.  Controls run here: random A units -> B's named units, and A's named units -> random B units.
  (iii) replication of the rank curve, logit-only ridge at S / h4, and the S DZ / h4 DW inheritance cells on fresh jets.

Networks: 8 of the E39x 10 (official 3f n64, n32, perm-inv n64, 16f n64, HGQ-r150 s0/s2, float s0/s1): all 7 non-degenerate
  edit networks plus the 16f net; n128 and HGQ s1 left out to keep memory within 1 CPU / 2 GiB.  Untrained donors: redraw seed 1
  (E39x used seed 0).
Samples: fit = sample(2000, 6941), test = sample(2000, 6942) minus rows in fit.
Everything else (ridge, RRR, clip to B's fit range, exact tail, task-loss trainer, network bootstrap, inheritance pooling)
  imported from interp.e39x_stitch_rank_jedi / interp.e39_stitch_retrains_jedi unchanged.
Output: $AI_RESULTS/jedi/e39x_verify.json (e39x_verify_smoke.json with --smoke).
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()

from interp.e39_stitch_retrains_jedi import (EDIT_NETS, ridge, apply, Tail, Boot, build_edits, resp, to_net, holm, binom_p,  # noqa: E402
                                             clean, redrawn)
from interp.e39x_stitch_rank_jedi import NetBoot, summ, mat, float_tail, make_trainer, ce  # noqa: E402

SF, ST_ = 6941, 6942
USEED = 1
NETS = ['official_n64', 'official_n32', 'perminv_n64', 'f16_n64', 'hgq150_s0', 'hgq150_s2', 'float_s0', 'float_s1']
LAYERS = ('S', 'h4')
RANKS = (1, 2, 4, 8, 64)
ROLES = ('r2', 'soft', 'hard', 'core')
EDITS4 = ('T', 'D', 'N')
RN = ('T', 'DW', 'DZ', 'N')
TASKS = ('S4', 'S64', 'logit5', 'logit4', 'untr4', 'untr64')   # task-loss map conditions (S = A's layer L)


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def rrr(M, Xfit, ranks):
    Yh = (np.asarray(Xfit, np.float64) - M['mx']) @ M['W']
    _, s, Vt = np.linalg.svd(Yh, full_matrices=False); out = {}
    for r in ranks:
        if r >= min(M['W'].shape): out[r] = dict(W=M['W'], mx=M['mx'], my=M['my'], Vt=Vt, full=True)
        else: out[r] = dict(W=M['W'] @ Vt[:r].T @ Vt[:r], mx=M['mx'], my=M['my'], Vt=Vt[:r], full=False)
    return out


def main(a):
    import jax, jax.numpy as jnp
    import interp.e33_program_recurrence_jedi as E33
    from jetdistill.config import RESULTS
    smoke = a.smoke; per = 150 if smoke else 2000; NB = 200 if smoke else 1000; NBN = 2000; NSUB = 10 if smoke else 100
    STEPS = 20 if smoke else 300
    nets = NETS if not smoke else ['official_n64', 'official_n32', 'float_s0']
    enets = [n for n in EDIT_NETS if n in nets]
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / ('e39x_verify_smoke.json' if smoke else 'e39x_verify.json')
    R = dict(node='E39x-verify', smoke=smoke, nets=nets, edit_nets=enets, seeds=dict(fit=SF, test=ST_, untrained=USEED))
    save = lambda: JS.write_text(json.dumps(clean(R)))  # noqa: E731

    yall = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idf, idt = E33.rows(yall, per, SF), E33.rows(yall, per, ST_); keep = ~np.isin(idt, idf)
    R['sample'] = dict(fit=f'sample({per}, {SF})', test=f'sample({per}, {ST_}) minus rows in fit', N_fit=len(idf), N_test=int(keep.sum()))
    cache = {}

    def data(n, f16):
        if (n, f16) not in cache:
            xf, _, _, yf, _ = E33.load(per, SF, n, f16); xt, _, x3t, yt, _ = E33.load(per, ST_, n, f16)
            cache[(n, f16)] = (xf, yf, xt[keep], yt[keep], x3t[keep])
        return cache[(n, f16)]
    _, yf, _, yt, x3t = data(64, False); J = len(yt); log(R['sample'])
    EDT = build_edits(x3t, yt, smoke); EDT = {e: EDT[e] for e in EDITS4}; R['edits'] = {k: v[3] for k, v in EDT.items()}
    BT = Boot(J, NB, 6943); yD = yt[EDT['D'][0]]
    NR = dict(T=len(EDT['T'][0]), DW=int((yD == 2).sum()), DZ=int((yD == 3).sum()), N=len(EDT['N'][0]))
    BE = {e: Boot(NR[e], NB, 6944 + i) for i, e in enumerate(RN)}
    SRC = ('S', 'h4', 'logits')

    def stages(M, nm, with_edits):
        xf, _, xt, _, _ = data(M.n, M.f16)
        Rf = M.run(xf, keep=SRC); Rt = M.run(xt, keep=SRC)
        st = {('fit', s): Rf[s].astype(np.float32) for s in SRC}; st.update({('test', s): Rt[s].astype(np.float32) for s in SRC})
        rsp = {}
        if with_edits:
            for e in EDITS4:
                r, xid, xe, _ = EDT[e]
                Ri = M.run(to_net(nm, xid), keep=SRC); Re = M.run(to_net(nm, xe), keep=SRC)
                for s in SRC: st[(e + '_id', s)] = Ri[s].astype(np.float32); st[(e + '_ed', s)] = Re[s].astype(np.float32)
                rsp.update(resp(e, Ri['logits'], Re['logits'], yt[r]))
        return st, rsp

    ST, UST, TAIL, CALL, LIM, MEAN, LIVE, FT, RESP, INFO = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for nm in nets:
        M = E33.Model(nm); ST[nm], rsp = stages(M, nm, nm in enets)
        LIM[nm] = {L: (ST[nm][('fit', L)].min(0).astype(np.float64), ST[nm][('fit', L)].max(0).astype(np.float64)) for L in LAYERS}
        MEAN[nm] = ST[nm][('fit', 'S')].astype(np.float64).mean(0)
        LIVE[nm] = np.flatnonzero(ST[nm][('fit', 'S')].std(0) > 1e-6)
        TAIL[nm] = Tail(M); CALL[nm] = ST[nm][('test', 'logits')].argmax(1)
        chk = {L: float(np.abs(TAIL[nm](L, ST[nm][('test', L)]) - ST[nm][('test', 'logits')]).max()) for L in LAYERS}
        FT[nm] = {L: float_tail(M, L) for L in LAYERS}
        if nm in enets: RESP[nm] = rsp
        INFO[nm] = dict(acc=float((CALL[nm] == yt).mean()), acc_ci=BT.ci(CALL[nm] == yt), tail_check=chk, n_live_S=int(len(LIVE[nm])),
                        resp={e: float(v.mean()) for e, v in rsp.items()})
        del M; gc.collect()
        with redrawn(USEED):
            Mu = E33.Model(nm)
        UST[nm], _ = stages(Mu, nm, nm in enets); del Mu; gc.collect(); jax.clear_caches()
        INFO[nm]['untrained_acc'] = float((UST[nm][('test', 'logits')].argmax(1) == yt).mean())
        log(f'{nm}: {json.dumps(clean(INFO[nm]))}')
    R['nets_info'] = INFO; save()
    native = {(A, B): float((CALL[A] == CALL[B]).mean()) for A in nets for B in nets}

    TR = {(L, full): make_trainer(full, L, FT[nets[0]][L][1], STEPS, 3e-3) for L in LAYERS for full in (True, False)}
    R['float_tail_check'] = {f'{nm}|{L}': float((np.asarray(TR[(L, True)][2](jnp.asarray(ST[nm][('test', L)]), FT[nm][L][0])) == CALL[nm]).mean())
                             for nm in nets for L in LAYERS}

    def run_map(B, L, Mp, X):
        lo_, hi_ = LIM[B][L]; return TAIL[B](L, np.clip(apply(Mp, X), lo_, hi_))

    def edit_resp(src, B, L, Mp, key):
        o = {}
        for e in EDITS4:
            li = run_map(B, L, Mp, src[(e + '_id', key)]); le = run_map(B, L, Mp, src[(e + '_ed', key)])
            o.update(resp(e, li, le, yt[EDT[e][0]]))
        return {k: float(v.mean()) for k, v in o.items()}

    def train_task(L, B, Mr, Mfull, Xfit, idx):
        """task-loss map initialised at Mr (rank-r RRR of Mfull, or Mfull itself) -> affine map dict"""
        Xf = jnp.asarray(np.asarray(Xfit, np.float32)); yfj = jnp.asarray(yf.astype(np.int32)); lo_, hi_ = LIM[B][L]
        full = Mr['full']
        if full: p0 = dict(W=jnp.asarray(Mr['W'], jnp.float32), c=jnp.asarray(Mr['my'], jnp.float32))
        else: p0 = dict(U=jnp.asarray(Mfull['W'] @ Mr['Vt'].T, jnp.float32), V=jnp.asarray(Mr['Vt'], jnp.float32), c=jnp.asarray(Mr['my'], jnp.float32))
        args = (FT[B][L][0], jnp.asarray(lo_, jnp.float32), jnp.asarray(hi_, jnp.float32), jnp.asarray(Mr['mx'], jnp.float32))
        p, _ = TR[(L, full)][0](p0, Xf, yfj, *args, idx)
        Wt = np.asarray(p['W'], np.float64) if full else np.asarray(p['U'], np.float64) @ np.asarray(p['V'], np.float64)
        return dict(W=Wt, mx=Mr['mx'], my=np.asarray(p['c'], np.float64))

    # ================================================================ rank + task-loss arms with the new controls
    PR = {L: {} for L in LAYERS}; SRESP = {L: {v: {} for v in ('full', 'untrained', 'logit')} for L in LAYERS}
    for L in LAYERS:
        for A in nets:
            tA = time.time()
            for B in nets:
                if A == B: continue
                rng = np.random.default_rng([6945, NETS.index(A), NETS.index(B), LAYERS.index(L)]); okB = CALL[B]
                ed = A in enets and B in enets; d = dict(native=native[(A, B)]); SB = {}
                Ml = ridge(ST[A][('fit', 'logits')], ST[B][('fit', L)], rng); Mlr = rrr(Ml, ST[A][('fit', 'logits')], (4, 5))
                lo = run_map(B, L, Ml, ST[A][('test', 'logits')]); SB['ridge_logit'] = lo.argmax(1) == okB
                if ed: SRESP[L]['logit'][(A, B)] = edit_resp(ST[A], B, L, Ml, 'logits')
                M = ridge(ST[A][('fit', L)], ST[B][('fit', L)], rng); MR = rrr(M, ST[A][('fit', L)], RANKS)
                Mu = ridge(UST[A][('fit', L)], ST[B][('fit', L)], rng); MRu = rrr(Mu, UST[A][('fit', L)], (4, 64))
                for r in RANKS: SB[f'ridge_{r}'] = run_map(B, L, MR[r], ST[A][('test', L)]).argmax(1) == okB
                for r in (4, 64): SB[f'ridge_untr{r}'] = run_map(B, L, MRu[r], UST[A][('test', L)]).argmax(1) == okB
                if ed:
                    SRESP[L]['full'][(A, B)] = edit_resp(ST[A], B, L, MR[64], L)
                    SRESP[L]['untrained'][(A, B)] = edit_resp(UST[A], B, L, MRu[64], L)
                idx = jnp.asarray(rng.integers(0, len(yf), (STEPS, 2048)).astype(np.int32))
                spec = dict(S4=(MR[4], M, ST[A], L), S64=(MR[64], M, ST[A], L), logit5=(Mlr[5], Ml, ST[A], 'logits'),
                            logit4=(Mlr[4], Ml, ST[A], 'logits'), untr4=(MRu[4], Mu, UST[A], L), untr64=(MRu[64], Mu, UST[A], L))
                d['acc'] = {}
                for t, (Mr, Mf, src, key) in spec.items():
                    Mt = train_task(L, B, Mr, Mf, src[('fit', key)], idx); lo = run_map(B, L, Mt, src[('test', key)]); c = lo.argmax(1)
                    SB[f'task_{t}'] = c == okB; d['acc'][f'task_{t}'] = float((c == yt).mean())
                d['same_B'] = {k: float(v.mean()) for k, v in SB.items()}
                for x_, y_ in (('task_S4', 'task_logit5'), ('task_S4', 'task_logit4'), ('ridge_4', 'ridge_logit'), ('task_S4', 'task_untr4')):
                    n10, n01 = int((SB[x_] & ~SB[y_]).sum()), int((~SB[x_] & SB[y_]).sum())
                    d[f'mcnemar_{x_}_vs_{y_}'] = dict(only_first=n10, only_second=n01, p=binom_p(n10, n10 + n01))
                PR[L][(A, B)] = d
            log(f'[{L}] donor {A} ({time.time() - tA:.0f} s): ' + '; '.join(
                f'{B}: ' + ' '.join(f'{k} {v:.3f}' for k, v in PR[L][(A, B)]['same_B'].items() if k in ('ridge_4', 'ridge_logit', 'task_S4', 'task_logit5', 'task_logit4', 'task_untr4'))
                for B in nets if B != A))
        R.setdefault('pairs', {})[L] = {f'{A}|{B}': v for (A, B), v in PR[L].items()}; save()

    NBt = NetBoot(nets, NBN, 6946); R['summary'] = {}
    for L in LAYERS:
        P = PR[L]; keys = list(next(iter(P.values()))['same_B']); o = {}; nd = len(nets) * (len(nets) - 1)
        V = {k: mat(nets, lambda A, B, k=k: P[(A, B)]['same_B'][k]) for k in keys}
        V['native'] = mat(nets, lambda A, B: P[(A, B)]['native'])
        for k, Vm in V.items(): o[k] = summ(Vm.sum() / nd, NBt.mean(Vm))
        for k in [k for k in keys if k.startswith('task_')]:
            Vm = mat(nets, lambda A, B, k=k: P[(A, B)]['acc'][k]); o[f'acc_{k}'] = summ(Vm.sum() / nd, NBt.mean(Vm))
        for x_, y_ in (('task_S4', 'task_logit5'), ('task_S4', 'task_logit4'), ('ridge_4', 'ridge_logit'), ('task_S4', 'task_untr4'),
                       ('task_S64', 'task_logit5'), ('task_untr64', 'task_logit5'), ('task_S4', 'ridge_4')):
            Dm = V[x_] - V[y_]; o[f'{x_}_minus_{y_}'] = summ(Dm.sum() / nd, NBt.mean(Dm), null=0.0)
            mk = f'mcnemar_{x_}_vs_{y_}'
            if mk in next(iter(P.values())):
                ks = list(P); hp = holm([P[k][mk]['p'] for k in ks])
                o[f'{x_}_vs_{y_}_holm'] = dict(n_pairs=len(ks), first_higher=int(sum(h < .05 and P[k][mk]['only_first'] > P[k][mk]['only_second'] for h, k in zip(hp, ks))),
                                              second_higher=int(sum(h < .05 and P[k][mk]['only_first'] < P[k][mk]['only_second'] for h, k in zip(hp, ks))))
        R['summary'][L] = o
        log(f'[{L}] summary: ' + json.dumps(clean({k: (v.get('est'), v.get('ci'), v.get('p_vs_null')) if 'est' in v else v for k, v in o.items()})))
    save()

    # ================================================================ subset arm at S with unit-choice controls
    try:
        ex = json.loads((OUT / 'e33x.json').read_text())['roles_units']
        named = {m: {r_: int(v[0][1:]) for r_, v in ex[m].items() if v[2] is not None and v[2] >= 0.9} for m in nets if m in ex}
        R['named_units'] = {m: {k: f'S{u}' for k, u in v.items()} for m, v in named.items()}

        def sub(B, ua, ub, A, rng):
            Mp = ridge(ST[A][('fit', 'S')][:, ua], ST[B][('fit', 'S')][:, ub], rng)
            Y = np.tile(MEAN[B], (J, 1)); Y[:, ub] = apply(Mp, ST[A][('test', 'S')][:, ua])
            lo_, hi_ = LIM[B]['S']; return float((TAIL[B]('S', np.clip(Y, lo_, hi_)).argmax(1) == CALL[B]).mean())

        def own(B, ub):
            Y = np.tile(MEAN[B], (J, 1)); Y[:, ub] = ST[B][('test', 'S')][:, ub]; return float((TAIL[B]('S', Y).argmax(1) == CALL[B]).mean())
        rows = []
        for A in nets:
            for B in nets:
                if A == B or A not in named or B not in named: continue
                shared = [k for k in ROLES if k in named[A] and k in named[B]]
                if not shared: continue
                rng = np.random.default_rng([6948, NETS.index(A), NETS.index(B)]); k = len(shared)
                ua = [named[A][s] for s in shared]; ub = [named[B][s] for s in shared]
                rA = lambda: list(rng.choice(LIVE[A], k, replace=False))  # noqa: E731
                rB = lambda: list(rng.choice(LIVE[B], k, replace=False))  # noqa: E731
                v = sub(B, ua, ub, A, rng)
                rr = np.array([sub(B, rA(), rB(), A, rng) for _ in range(NSUB)])     # E39x control: random in A and in B
                ra = np.array([sub(B, rA(), ub, A, rng) for _ in range(NSUB)])       # new: random A units -> B's named units
                rb = np.array([sub(B, ua, rB(), A, rng) for _ in range(NSUB)])       # new: A's named units -> random B units
                rows.append(dict(pair=f'{A}|{B}', roles=shared, k=k, named=v, randAB=float(rr.mean()), randA_namedB=float(ra.mean()),
                                 namedA_randB=float(rb.mean()), own_named_B=own(B, ub),
                                 p_vs_randAB=float((1 + (rr >= v).sum()) / (1 + NSUB)), p_vs_randA_namedB=float((1 + (ra >= v).sum()) / (1 + NSUB))))
            log(f'subset donor {A}: ' + '; '.join(f'{r_["pair"].split("|")[1]} k{r_["k"]} named {r_["named"]:.3f} randAB {r_["randAB"]:.3f} '
                                                 f'randA->namedB {r_["randA_namedB"]:.3f} namedA->randB {r_["namedA_randB"]:.3f} own {r_["own_named_B"]:.3f}'
                                                 for r_ in rows if r_['pair'].startswith(A + '|')))
        o = dict(rows=rows, n_pairs=len(rows))
        for key in ('named', 'randAB', 'randA_namedB', 'namedA_randB', 'own_named_B'):
            o[f'mean_{key}'] = float(np.mean([r_[key] for r_ in rows]))
        for x_, y_ in (('named', 'randAB'), ('named', 'randA_namedB'), ('named', 'namedA_randB'), ('randA_namedB', 'randAB')):
            Dn = mat(nets, lambda A, B: next((r_[x_] - r_[y_] for r_ in rows if r_['pair'] == f'{A}|{B}'), np.nan))
            msk = np.isfinite(Dn).astype(float); np.fill_diagonal(msk, 0); Dn0 = np.nan_to_num(Dn)
            o[f'{x_}_minus_{y_}'] = summ(Dn0.sum() / max(msk.sum(), 1), NBt.ratio(Dn0, msk), null=0.0)
        o['by_k'] = {int(k): {key: float(np.mean([r_[key] for r_ in rows if r_['k'] == k])) for key in ('named', 'randAB', 'randA_namedB', 'namedA_randB', 'own_named_B')}
                     | dict(n=sum(r_['k'] == k for r_ in rows)) for k in sorted({r_['k'] for r_ in rows})}
        R['subset'] = o
        log('subset: ' + json.dumps(clean({k: v for k, v in o.items() if k != 'rows'})))
    except Exception as err:  # noqa: BLE001
        import traceback; traceback.print_exc(); R['subset'] = dict(error=repr(err))
    save()

    # ================================================================ inheritance at network level (fresh jets)
    enets_nd = enets; NBe = NetBoot(enets_nd, NBN, 6949); R['inheritance'] = {}
    for L in LAYERS:
        R['inheritance'][L] = {}
        for e in RN:
            elig = {}
            for A in enets:
                for B in enets:
                    if A == B: continue
                    se = float(BE[e].means(RESP[A][e] - RESP[B][e]).std()); elig[(A, B)] = abs(float((RESP[A][e] - RESP[B][e]).mean())) > 2 * se
            rA = {n: float(RESP[n][e].mean()) for n in enets}
            E = mat(enets_nd, lambda A, B: float(elig[(A, B)])); DEN = mat(enets_nd, lambda A, B: (rA[A] - rA[B]) ** 2) * E
            o = dict(n_eligible=int(E.sum()), n_pairs=len(enets_nd) * (len(enets_nd) - 1), r_net=rA)
            for v in ('full', 'untrained', 'logit'):
                rs = {k: val[e] for k, val in SRESP[L][v].items()}
                NUM = mat(enets_nd, lambda A, B: (rs[(A, B)] - rA[B]) * (rA[A] - rA[B])) * E
                o[v] = summ(NUM.sum() / DEN.sum() if DEN.sum() > 0 else np.nan, NBe.ratio(NUM, DEN), null=0.5)
            rs1 = {k: val[e] for k, val in SRESP[L]['full'].items()}; rs2 = {k: val[e] for k, val in SRESP[L]['untrained'].items()}
            N1 = mat(enets_nd, lambda A, B: (rs1[(A, B)] - rA[B]) * (rA[A] - rA[B])) * E
            N2 = mat(enets_nd, lambda A, B: (rs2[(A, B)] - rA[B]) * (rA[A] - rA[B])) * E
            o['full_minus_untrained'] = summ((N1.sum() - N2.sum()) / DEN.sum() if DEN.sum() > 0 else np.nan, NBe.ratio_diff(N1, N2, DEN), null=0.0)
            R['inheritance'][L][e] = o
            log(f'[{L}] {e}: eligible {o["n_eligible"]}; ' + '; '.join(f'{v} {o[v]["est"]} {o[v].get("ci")} p {o[v].get("p_vs_null")}' for v in ('full', 'untrained', 'logit'))
                + f'; full-untrained {o["full_minus_untrained"]}')
    save(); log(f'wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); main(ap.parse_args())
