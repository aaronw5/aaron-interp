"""E39 skeptic verification (fresh samples + missing controls + network-level uncertainty).

1. Fresh rerun of the key arm of interp.e39_stitch_retrains_jedi on new rows: fit = sample(2000, 4939), test = sample(2000, 4940)
   minus rows shared with fit (E39 used 3939 / 3940).  Same 10 networks, same ridge affine stitch with clip to B's fit range and
   B's exact tail, at S and h4.  Untrained donors: redraw seed 0 only (E39: seeds 0-2).  Random-map / shuffled-row controls not
   rerun (E39: 20-23 %).
2. Missing control, "output-only donor": the stitch is fitted from A's 5 logits (not from A's S or h4) into B's S or h4 by the
   same ridge + clip + tail.  This map can only carry A's class scores.  If it reaches the same same-call with B and the same
   edit inheritance pattern, then the S/h4 stitch results are explained by A and B sharing output decisions and do not show that
   the layer's content is shared beyond that.
3. Network-level uncertainty for the pooled inheritance index I (E39 gave jet-bootstrap CIs only): leave-one-network-out
   jackknife over the 7 non-degenerate edit networks, on the fresh run and on the E39 run-3 rows (results/jedi/e39.json).
4. Null value of the pooled I: the eligible pair set is symmetric (|r_A - r_B| > 2 SE is symmetric in A, B), so for a stitched
   response that is the same constant c in every pair, sum_(A,B) (c - r_B)(r_A - r_B) = sum_{A<B} (r_A - r_B)^2 and the pooled
   I = 0.5 exactly.  0.5, not 0, is the no-information value; it is recomputed here numerically.
5. Per-pair regression r_stitch = a + b r_A + c r_B over the 42 non-degenerate ordered pairs, with network jackknife SEs.
Output: $AI_RESULTS/jedi/e39_verify.json
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()

import interp.e39_stitch_retrains_jedi as E39  # noqa: E402
from interp.e39_stitch_retrains_jedi import (NETS, EDIT_NETS, DEGEN, LAYERS, EDITS, RNAMES, ridge, apply, Tail, Boot,  # noqa: E402
                                             build_edits, resp, to_net, r2, holm, binom_p, clean, redrawn)


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


SF, ST_ = 4939, 4940


def jackknife(fn, nets):
    """fn(excluded_net or None) -> value; leave-one-network-out jackknife"""
    full = fn(None); loo = [fn(n) for n in nets]; loo = [v for v in loo if v is not None and np.isfinite(v)]
    if full is None or len(loo) < 3: return dict(est=full, se=None, ci=None, loo=loo)
    k = len(loo); m = np.mean(loo); se = float(np.sqrt((k - 1) / k * ((np.array(loo) - m) ** 2).sum()))
    return dict(est=float(full), se=se, ci=[float(full - 1.96 * se), float(full + 1.96 * se)], loo_min=float(min(loo)), loo_max=float(max(loo)))


def pooled_from_means(rows, key, excl=None, const=False):
    """rows: list of dict(A, B, rA, rB, rS..., elig) (means) -> pooled I over eligible rows"""
    sel = [r for r in rows if r['elig'] and r['A'] not in DEGEN and r['B'] not in DEGEN and excl not in (r['A'], r['B'])]
    if not sel: return None
    pA = np.array([r['rA'] for r in sel]); pB = np.array([r['rB'] for r in sel]); pS = np.array([r[key] for r in sel])
    if const: pS = np.full_like(pS, pS.mean())
    den = ((pA - pB) ** 2).sum()
    return float(((pS - pB) * (pA - pB)).sum() / den) if den > 0 else None


def regress(rows, key, excl=None):
    sel = [r for r in rows if r['A'] not in DEGEN and r['B'] not in DEGEN and excl not in (r['A'], r['B'])]
    X = np.array([[1.0, r['rA'], r['rB']] for r in sel]); y = np.array([r[key] for r in sel])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def analyse_rows(rows, keys, nets):
    out = {}
    for key in keys:
        o = dict(pooled=jackknife(lambda ex: pooled_from_means(rows, key, ex), nets),
                 null_const=pooled_from_means(rows, key, None, const=True),
                 n_elig=sum(r['elig'] and r['A'] not in DEGEN and r['B'] not in DEGEN for r in rows))
        for j, nm in ((1, 'b_donor'), (2, 'c_recipient')):
            o[nm] = jackknife(lambda ex: float(regress(rows, key, ex)[j]), nets)
        out[key] = o
    # difference trained stitch - controls, network jackknife
    for ctl in keys[1:]:
        out[f'diff_{keys[0]}_minus_{ctl}'] = jackknife(
            lambda ex: (lambda a, b: None if a is None or b is None else a - b)(pooled_from_means(rows, keys[0], ex), pooled_from_means(rows, ctl, ex)), nets)
    return out


def main(a):
    import interp.e33_program_recurrence_jedi as E33
    from jetdistill.config import RESULTS
    per = 2000; NB = 1000; NPERM = 2000
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / 'e39_verify.json'
    R = dict(node='E39-verify', seeds=dict(fit=SF, test=ST_), untrained_seeds=[0])
    save = lambda: JS.write_text(json.dumps(clean(R)))  # noqa: E731

    # ---- part 0: network-level re-analysis of the E39 run-3 rows
    try:
        e = json.loads((OUT / 'e39.json').read_text()); R['e39_reanalysis'] = {}
        nets7 = [n for n in EDIT_NETS if n not in DEGEN]
        for L in LAYERS:
            R['e39_reanalysis'][L] = {}
            for rn in ('T', 'DW', 'DZ', 'N', 'Kabs'):
                rows = []
                for rw in e['inheritance'][L][rn]['rows']:
                    A, B = rw['pair'].split('|')
                    rows.append(dict(A=A, B=B, rA=rw['r_A'], rB=rw['r_B'], stitch=rw['r_stitch'], untrained=rw['r_untrained_stitch'],
                                     random=rw['r_random_stitch'], elig=rw['eligible']))
                R['e39_reanalysis'][L][rn] = analyse_rows(rows, ['stitch', 'untrained', 'random'], nets7)
                o = R['e39_reanalysis'][L][rn]
                log(f'E39 rows [{L}] {rn}: I {o["stitch"]["pooled"]["est"]:.3f} netJK CI {o["stitch"]["pooled"].get("ci")}; untrained '
                    f'{o["untrained"]["pooled"]["est"]:.3f}; random {o["random"]["pooled"]["est"]:.3f}; const-null {o["stitch"]["null_const"]:.3f}; '
                    f'diff stitch-untrained {o["diff_stitch_minus_untrained"]["est"]:.3f} CI {o["diff_stitch_minus_untrained"].get("ci")}; '
                    f'b {o["stitch"]["b_donor"]["est"]:.2f}±{o["stitch"]["b_donor"]["se"]}, c {o["stitch"]["c_recipient"]["est"]:.2f}±{o["stitch"]["c_recipient"]["se"]}')
        save()
    except Exception as err:  # noqa: BLE001
        R['e39_reanalysis'] = dict(error=repr(err)); log(f'reanalysis failed: {err!r}')

    # ---- samples
    yall = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idf, idt = E33.rows(yall, per, SF), E33.rows(yall, per, ST_); keep = ~np.isin(idt, idf)
    old = np.union1d(E33.rows(yall, per, 3939), E33.rows(yall, per, 3940))
    R['sample'] = dict(N_fit=len(idf), N_test=int(keep.sum()), overlap_removed=int((~keep).sum()),
                       test_rows_also_in_e39=int(np.isin(idt[keep], old).sum()), fit_rows_also_in_e39=int(np.isin(idf, old).sum()))
    cache = {}

    def data(n, f16):
        if (n, f16) not in cache:
            xf, _, _, yf, _ = E33.load(per, SF, n, f16); xt, _, x3t, yt, _ = E33.load(per, ST_, n, f16)
            cache[(n, f16)] = (xf, yf, xt[keep], yt[keep], x3t[keep])
        return cache[(n, f16)]
    _, yf, _, yt, x3t = data(64, False); J = len(yt); log(f'samples {R["sample"]}')
    EDT = build_edits(x3t, yt, False); R['edits'] = {k: v[3] for k, v in EDT.items()}
    BT = Boot(J, NB, 4943); yD = yt[EDT['D'][0]]
    NR = dict(T=len(EDT['T'][0]), DW=int((yD == 2).sum()), DZ=int((yD == 3).sum()), N=len(EDT['N'][0]), K=len(EDT['K'][0]), Kabs=len(EDT['K'][0]))
    BE = {e_: Boot(NR[e_], NB, 4944 + i) for i, e_ in enumerate(RNAMES)}
    SRC = ('S', 'h4', 'logits')

    def stages(M, nm, xf, xt, with_edits):
        Rf = M.run(xf, keep=SRC); Rt = M.run(xt, keep=SRC)
        st = {('fit', s): Rf[s].astype(np.float32) for s in SRC}; st.update({('test', s): Rt[s].astype(np.float32) for s in SRC})
        rsp = {}
        if with_edits:
            for e_ in EDITS:
                r, xid, xe, _ = EDT[e_]
                Ri = M.run(to_net(nm, xid), keep=SRC); Re = M.run(to_net(nm, xe), keep=SRC)
                for s in SRC: st[(e_ + '_id', s)] = Ri[s].astype(np.float32); st[(e_ + '_ed', s)] = Re[s].astype(np.float32)
                rsp.update(resp(e_, Ri['logits'], Re['logits'], yt[r]))
        return st, rsp, Rf, Rt

    ST, TAIL, CALL, LIM, RESP, INFO = {}, {}, {}, {}, {}, {}
    for nm in NETS:
        M = E33.Model(nm); xf, _, xt, _, _ = data(M.n, M.f16)
        ST[nm], rsp, Rf, Rt = stages(M, nm, xf, xt, nm in EDIT_NETS)
        LIM[nm] = {L: (Rf[L].min(0), Rf[L].max(0)) for L in LAYERS}; TAIL[nm] = Tail(M); CALL[nm] = Rt['logits'].argmax(1)
        chk = {L: float(np.abs(TAIL[nm](L, Rt[L]) - Rt['logits']).max()) for L in LAYERS}
        if nm in EDIT_NETS: RESP[nm] = rsp
        INFO[nm] = dict(acc=float((CALL[nm] == yt).mean()), acc_ci=BT.ci(CALL[nm] == yt), tail_check=chk,
                        resp={e_: [float(v.mean()), BE[e_].ci(v)] for e_, v in rsp.items()})
        log(f'{nm}: acc {INFO[nm]["acc"]:.4f}; tail {chk}; resp ' + json.dumps(clean({k: v[0] for k, v in INFO[nm]['resp'].items()})))
        del M; gc.collect()
    R['nets_info'] = INFO; save()
    native = {(A, B): float((CALL[A] == CALL[B]).mean()) for A in NETS for B in NETS}

    def stitch(src, B, L, Mp, with_edits):
        lo_, hi_ = LIM[B][L]; f = lambda X: TAIL[B](L, np.clip(apply(Mp, X), lo_, hi_))  # noqa: E731
        o = dict(lo=f(src['test']))
        if with_edits:
            o['resp'] = {}
            for e_ in EDITS:
                o['resp'].update(resp(e_, f(src[e_ + '_id']), f(src[e_ + '_ed']), yt[EDT[e_][0]]))
        return o

    def callm(lo, A, B):
        c = lo.argmax(1); dis = CALL[A] != CALL[B]
        return dict(acc=float((c == yt).mean()), same_B=float((c == CALL[B]).mean()), same_A=float((c == CALL[A]).mean()),
                    dis_n=int(dis.sum()), dis_follow_A=float((c[dis] == CALL[A][dis]).mean()), dis_follow_B=float((c[dis] == CALL[B][dis]).mean()),
                    mcnemar_p=binom_p(int(((CALL[B] == yt) & (c != yt)).sum()), int(((CALL[B] == yt) != (c == yt)).sum()))), c

    PAIRS = {k: {L: {} for L in LAYERS} for k in ('stitch', 'logit', 'untrained')}
    SAME = {k: {L: {} for L in LAYERS} for k in PAIRS}; SR = {k: {L: {} for L in LAYERS} for k in PAIRS}
    for L in LAYERS:
        for A in NETS:
            for B in NETS:
                if A == B: continue
                ed = A in EDIT_NETS and B in EDIT_NETS
                for kind, s in (('stitch', L), ('logit', 'logits')):
                    rng = np.random.default_rng([4945, NETS.index(A), NETS.index(B), LAYERS.index(L), kind == 'logit'])
                    src = {k[0]: v for k, v in ST[A].items() if k[1] == s}
                    Mp = ridge(src['fit'], ST[B][('fit', L)], rng); o = stitch(src, B, L, Mp, ed)
                    m, c = callm(o['lo'], A, B); m['map_r2_test'] = r2(apply(Mp, src['test']), ST[B][('test', L)].astype(np.float64))
                    m['native'] = native[(A, B)]
                    PAIRS[kind][L][(A, B)] = m; SAME[kind][L][(A, B)] = (c == CALL[B]).astype(np.float32)
                    if ed: SR[kind][L][(A, B)] = o['resp']
            log(f'[{L}] donor {A}: stitch same_B ' + ' '.join(f'{PAIRS["stitch"][L][(A, B)]["same_B"]:.3f}' for B in NETS if B != A)
                + ' | logit-donor ' + ' '.join(f'{PAIRS["logit"][L][(A, B)]["same_B"]:.3f}' for B in NETS if B != A))
    save()

    # ---- untrained donors (seed 0)
    for A in NETS:
        with redrawn(0):
            M_ = E33.Model(A)
        xf, _, xt, _, _ = data(M_.n, M_.f16)
        Ast, _, _, _ = stages(M_, A, xf, xt, A in EDIT_NETS); del M_; gc.collect()
        for L in LAYERS:
            src = {k[0]: v for k, v in Ast.items() if k[1] == L}
            for B in NETS:
                if B == A: continue
                ed = A in EDIT_NETS and B in EDIT_NETS
                rng = np.random.default_rng([4946, NETS.index(A), NETS.index(B), LAYERS.index(L)])
                Mp = ridge(src['fit'], ST[B][('fit', L)], rng); o = stitch(src, B, L, Mp, ed)
                m, c = callm(o['lo'], A, B); PAIRS['untrained'][L][(A, B)] = m; SAME['untrained'][L][(A, B)] = (c == CALL[B]).astype(np.float32)
                if ed: SR['untrained'][L][(A, B)] = o['resp']
        log(f'untrained {A}: same_B S ' + ' '.join(f'{PAIRS["untrained"]["S"][(A, B)]["same_B"]:.3f}' for B in NETS if B != A))
        del Ast; gc.collect()

    R['pairs'] = {kind: {L: {f'{A}|{B}': v for (A, B), v in PAIRS[kind][L].items()} for L in LAYERS} for kind in PAIRS}; save()

    # ---- same-call summaries and tests
    rngp = np.random.default_rng(4947); R['samecall'] = {}
    keys = [(A, B) for A in NETS for B in NETS if A != B]; nd = [k for k in keys if k[0] not in DEGEN and k[1] not in DEGEN]
    for L in LAYERS:
        o = {}
        for kind in PAIRS:
            P = PAIRS[kind][L]
            o[kind] = dict(same_B_mean=float(np.mean([P[k]['same_B'] for k in keys])), same_B_mean_nd=float(np.mean([P[k]['same_B'] for k in nd])),
                           same_A_mean_nd=float(np.mean([P[k]['same_A'] for k in nd])),
                           acc_minus_B_nd=float(np.mean([P[k]['acc'] - INFO[k[1]]['acc'] for k in nd])),
                           dis_follow_A_nd=float(np.mean([P[k]['dis_follow_A'] for k in nd])), dis_follow_B_nd=float(np.mean([P[k]['dis_follow_B'] for k in nd])),
                           mcnemar_holm_lt_0p05=int((holm([P[k]['mcnemar_p'] for k in keys]) < 0.05).sum()),
                           map_r2_test_nd=float(np.mean([P[k].get('map_r2_test', np.nan) for k in nd])) if kind != 'untrained' else None)
        o['native_nd'] = float(np.mean([native[k] for k in nd]))
        dd = np.array([PAIRS['stitch'][L][k]['dis_follow_A'] - PAIRS['stitch'][L][k]['dis_follow_B'] for k in nd])
        sg = rngp.choice([-1.0, 1.0], (10000, len(dd))); o['dis_A_minus_B_p_two_sided'] = float((1 + (np.abs(sg @ dd) >= abs(dd.sum())).sum()) / 10001)
        # trained vs untrained / trained vs logit-donor per pair, sign-flip, Holm
        for ctl in ('untrained', 'logit'):
            ps, diffs = [], []
            for k in keys:
                d = SAME['stitch'][L][k] - SAME[ctl][L][k]; obs = float(d.mean()); cnt = 0
                for c0 in range(0, NPERM, 200):
                    s_ = rngp.choice(np.array([-1.0, 1.0], np.float32), (200, len(d))); cnt += int(((s_ @ d) / len(d) >= obs).sum())
                ps.append((1 + cnt) / (1 + NPERM)); diffs.append(obs)
            hp = holm(ps); dn = np.array([diffs[keys.index(k)] for k in nd])
            sg = rngp.choice([-1.0, 1.0], (10000, len(dn)))
            o[f'stitch_minus_{ctl}'] = dict(mean_nd=float(dn.mean()), min_nd=float(dn.min()), max_nd=float(dn.max()),
                                            n_holm_lt_0p05=int((hp < 0.05).sum()), n_pairs=len(keys),
                                            across_pairs_p_one_sided=float((1 + ((sg @ dn) >= dn.sum()).sum()) / 10001),
                                            ci_mean_nd=BT.ci(np.mean([SAME['stitch'][L][k] - SAME[ctl][L][k] for k in nd], 0)))
        R['samecall'][L] = o
        log(f'[{L}] same-call ' + json.dumps(clean(o)))
    save()

    # ---- inheritance index (fresh run): jet bootstrap + network jackknife + const null + regression
    nets7 = [n for n in EDIT_NETS if n not in DEGEN]; R['inheritance'] = {}
    for L in LAYERS:
        R['inheritance'][L] = {}
        for rn in ('T', 'DW', 'DZ', 'N', 'Kabs'):
            Bx = BE[rn]; rows = []
            for A in EDIT_NETS:
                for B in EDIT_NETS:
                    if A == B: continue
                    rA, rB = RESP[A][rn], RESP[B][rn]; se = float(Bx.means(rA - rB).std()); diff = float((rA - rB).mean())
                    rows.append(dict(A=A, B=B, rA=float(rA.mean()), rB=float(rB.mean()), elig=abs(diff) > 2 * se,
                                     acc_matched=PAIRS['stitch'][L][(A, B)]['acc'] >= INFO[B]['acc'] - 0.02,
                                     **{k: float(SR[k][L][(A, B)][rn].mean()) for k in PAIRS}))
            o = analyse_rows(rows, ['stitch', 'logit', 'untrained'], nets7)
            # jet bootstrap CI for trained stitch pooled I
            sel = [r for r in rows if r['elig'] and r['A'] not in DEGEN and r['B'] not in DEGEN]
            if sel:
                for k in ('stitch', 'logit', 'untrained'):
                    bA = np.stack([Bx.means(RESP[r['A']][rn]) for r in sel]); bB = np.stack([Bx.means(RESP[r['B']][rn]) for r in sel])
                    bS = np.stack([Bx.means(SR[k][L][(r['A'], r['B'])][rn]) for r in sel])
                    bs = ((bS - bB) * (bA - bB)).sum(0) / ((bA - bB) ** 2).sum(0)
                    o[k]['jet_boot_ci'] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
            o['acc_matched'] = pooled_from_means([dict(r, elig=r['elig'] and r['acc_matched']) for r in rows], 'stitch')
            o['n_acc_matched'] = sum(r['elig'] and r['acc_matched'] and r['A'] not in DEGEN and r['B'] not in DEGEN for r in rows)
            o['rows'] = rows; R['inheritance'][L][rn] = o
            s_ = o['stitch']
            log(f'fresh [{L}] {rn}: n_elig {s_["n_elig"]}; I stitch {s_["pooled"]["est"]} netJK {s_["pooled"].get("ci")} jet {s_.get("jet_boot_ci")}; '
                f'logit-donor {o["logit"]["pooled"]["est"]}; untrained {o["untrained"]["pooled"]["est"]}; const-null {s_["null_const"]}; '
                f'acc-matched {o["acc_matched"]} (n {o["n_acc_matched"]}); b {s_["b_donor"]["est"]:.2f}±{s_["b_donor"]["se"]}, '
                f'c {s_["c_recipient"]["est"]:.2f}±{s_["c_recipient"]["se"]}; diff vs untrained {o["diff_stitch_minus_untrained"]["est"]} '
                f'{o["diff_stitch_minus_untrained"].get("ci")}; diff vs logit {o["diff_stitch_minus_logit"]["est"]} {o["diff_stitch_minus_logit"].get("ci")}')
    save(); log(f'wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); main(ap.parse_args())
