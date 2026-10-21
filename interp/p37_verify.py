"""P37 verification (loop C, round C9): fresh sample, P37's head ranking frozen, and the control P37 lacks: random head sets matched
in the number of class-attention heads (and per block).

Why. All information about the particles reaches the output only through the 16 class-attention heads (b1h1-8, b2h1-8): the
class token starts as a learned constant, and the class-block MLPs see only the class token. A random 5-head set drawn from all 80
heads holds on average one class head, so 'top-5 vs random 5 of 80' partly compares class heads with particle heads. The matched
control draws the same number of class heads from the 16 (and the same number of particle heads from the 64), or the same number
of heads from each block (block-matched).
Sample: JetClass test_20M <sample>_118.root (P37 used _111), numpy default_rng(3790), random rows (never first-k). Per class:
  EVAL 300 (N 3,000): per-class specificity of the circuit knock-out and literal circuits (SUB = 100 per class of EVAL);
  REF 200 (N 2,000): the head / MLP means used for mean ablation (fresh means);
  TOPX 1,000 extra t->bqq' jets: single-head scan (out-of-sample check of P37's ranking), joint knock-out and the prong-merge edits.
Every jet passes once through part_core.build (eta = jet_eta + sign(jet_eta) deta).
Circuit sets are P37's ranked heads read from p37_top_circuit_<model>.json (no re-ranking on these jets): ParT_full top-5 / top-10,
ParT_kin top-5 / top-10.
Tests per model:
  A  joint knock-out on TOPX: d = change of logit_t - logit_QCD; share = mean d(set) / mean d(all 80); circuit vs NR random sets
     from all 80 (P37's control), class-matched, block-matched; all 16 class heads; all 8 b1 heads.
  B  specificity on EVAL: the circuit knock-out's change of each class's own log-odds lo_k = log p_k/(1-p_k) and of its recall,
     vs class-matched random sets (percentile p per class).
  C  edits a (b-prong merged), b (pT-matched W-prong cluster), c (one W subjet) on TOPX (P37 code), patch share of the circuit vs
     class-matched and block-matched random sets; all 16 class heads patched (should carry about all of it).
  D  literal circuits (variants literal / cls_mlp) at k = 5, 10 on SUB vs class-matched random sets.
  ./sync.sh && ./part_job.sh c-p37-verify 'python -m interp.p37_verify'   -> $AI_RESULTS/part/p37_verify_{full,kin}.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_argv = sys.argv; sys.argv = [_argv[0]]
import interp.p37_top_circuit as P               # noqa: E402  (functions only; its argparse sees no flags)
sys.argv = _argv
SMOKE = '--smoke' in sys.argv
SEED = 3790; FILE_NO = '118'
NEVAL = 20 if SMOKE else 300; NREF = 20 if SMOKE else 200; NTOPX = 60 if SMOKE else 1000; NSUB = 10 if SMOKE else 100
NR = 5 if SMOKE else 200; NRB = 5 if SMOKE else 100; NB = 200 if SMOKE else 1000; NPERM = 2000 if SMOKE else 20000
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'; TAG = '_smoke' if SMOKE else ''
TOP, QCD = P.TOP, P.QCD
T0 = time.time()
CLS = list(range(64, 80))


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def idx_of(lbl):
    return P.HL.index(lbl)


def matched(S, rng, mode):
    """random head set matched to S: 'class' = same number of class heads (from the 16) and particle heads (from the 64);
    'block' = same number of heads in each of the 10 blocks; 'all' = k of 80"""
    S = list(S); k = len(S)
    if mode == 'all': return rng.choice(80, k, replace=False).tolist()
    if mode == 'class':
        nc = sum(s >= 64 for s in S)
        return rng.choice(CLS, nc, replace=False).tolist() + rng.choice(64, k - nc, replace=False).tolist()
    out = []
    for b in range(10):
        nb = sum(s // 8 == b for s in S)
        if nb: out += (8 * b + rng.choice(8, nb, replace=False)).tolist()
    return out


def pemp(x, v, larger=True):
    v = np.asarray(v, float); return float((1 + (v >= x - 1e-12).sum() if larger else 1 + (v <= x + 1e-12).sum()) / (1 + len(v)))


def summ(v, x=None, larger=True):
    v = np.asarray(v, float); e = dict(mean=float(v.mean()), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                                      min=float(v.min()), max=float(v.max()), n=int(len(v)))
    if x is not None: e['p_emp'] = pemp(x, v, larger); e['frac_random_ge'] = float((v >= x).mean())
    return e


def load():
    from interp.part_core import sub, cat, PRE, CL, sample
    rng = np.random.default_rng(SEED); ev, rf, tx = [], [], []
    for c in CL:
        nx = NTOPX if c == 'Tbqq' else 0
        J, rows = P.load_rows(TEST / f'{PRE[c]}_{FILE_NO}.root', NEVAL + NREF + nx, rng); perm = rng.permutation(len(rows))
        ev.append(sub(J, np.sort(perm[:NEVAL]))); rf.append(sub(J, np.sort(perm[NEVAL:NEVAL + NREF])))
        if nx: tx.append(sub(J, np.sort(perm[NEVAL + NREF:])))
        log(f'loaded {c}: {len(rows)} random rows of {PRE[c]}_{FILE_NO}.root')
    B = P.rebuild(cat(ev)); BR = P.rebuild(cat(rf)); BT = P.rebuild(cat(tx)); y = B['label'].astype(int)
    assert (BT['label'].astype(int) == TOP).all()
    sub_idx = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], NSUB, replace=False) for k in range(10)]))
    P1 = P.rebuild(sample(1000, 3000)); return rng, B, BR, BT, y, sub_idx, P1


def main():
    import torch
    from interp.part_core import sub
    OUTD.mkdir(parents=True, exist_ok=True); torch.manual_seed(SEED)
    rng, B, BR, BT, y, sub_idx, P1 = load(); yP1 = P1['label'].astype(int); BSUB = sub(B, sub_idx); ysub = y[sub_idx]
    W_T = P.boot_W(len(BT['mask']), NB, rng)
    # edits on TOPX (deterministic, P37 code)
    info = P.prongs(BT, rng)
    va = np.array([g is not None for g in info['gb']]); vb = np.array([g is not None for g in info['gw']]); vc = np.array([g is not None for g in info['gc']])
    V = np.where(va & vb & vc)[0]; BV = sub(BT, V); Wv = P.boot_W(len(V), NB, rng)
    SFv = np.where(np.random.default_rng(SEED + 2).random((NPERM, len(V))) < 0.5, -1.0, 1.0)
    EDS = {}
    for en, key in (('a', 'gb'), ('b', 'gw'), ('c', 'gc')):
        E_, mp_, _ = P.merge_edit(BV, [info[key][j] for j in V]); EDS[en] = (E_, mp_)
    from interp.part_core import jmass
    echk = dict(n_top=int(len(BT['mask'])), n_valid=int(len(V)),
                dmass_max={e: float(np.abs(jmass(EDS[e][0]) - jmass(BV)).max()) for e in EDS},
                removed={e: float((BV['mask'].sum(1) - EDS[e][0]['mask'].sum(1)).mean()) for e in EDS})
    log('edit checks', json.dumps(echk))
    for key in ([k for k in ('full', 'kin')] if not SMOKE else ['full']):
        name, feat, null = P.MODEL_SPEC[key]; rr = np.random.default_rng(SEED + 10 + (key == 'kin'))
        PJ = json.loads((OUTD / f'p37_top_circuit_{key}.json').read_text())['model']
        rank = [idx_of(h) for h in PJ['scan']['rank']]; old_score = np.array(PJ['scan']['score'])
        res = dict(name=name, p37_rank_top10=[P.hlab(k) for k in rank[:10]]); M = P.HRun(feat)
        res['acc_P1'] = P.acc_block(M.run(P1), yP1, conf=False)['acc10']
        L = M.run(B); res['eval'] = P.acc_block(L, y, conf=False); corr0 = L.argmax(1) == y
        M.run(BR, dict(recmean=True)); Mh, Mf = M.means(); res['recon_max'] = M.recon
        log(f'==== {name}: acc P1 {res["acc_P1"]:.4f}, fresh eval acc10 {res["eval"]["acc10"]:.4f} top recall {res["eval"]["top_recall"]:.3f}')
        # ---- out-of-sample single-head scan (d_QCD, d_W) on TOPX
        L0 = M.run(BT); a0, b0 = P.ctr(L0, TOP, QCD), P.ctr(L0, TOP, P.WQQ)
        dq = np.zeros(80); dw = np.zeros(80)
        for k in range(80):
            Lk = M.run(BT, P.ctl_ablate(M, [k], Mh)); dq[k] = (P.ctr(Lk, TOP, QCD) - a0).mean(); dw[k] = (P.ctr(Lk, TOP, P.WQQ) - b0).mean()
        sc = (dq + dw) / 2; from scipy.stats import spearmanr
        newrank = list(np.argsort(sc))
        res['scan'] = dict(d_QCD={P.hlab(k): float(dq[k]) for k in range(80)}, d_W={P.hlab(k): float(dw[k]) for k in range(80)},
                           spearman_vs_p37=float(spearmanr(sc, old_score).correlation),
                           spearman_vs_p37_class_heads=float(spearmanr(sc[64:], old_score[64:]).correlation),
                           new_top10=[P.hlab(k) for k in newrank[:10]], overlap_top5=len(set(newrank[:5]) & set(rank[:5])),
                           overlap_top10=len(set(newrank[:10]) & set(rank[:10])))
        log(f'  scan: Spearman vs P37 {res["scan"]["spearman_vs_p37"]:.3f}; new top10 {res["scan"]["new_top10"]}; overlap top5 {res["scan"]["overlap_top5"]}')
        # ---- A joint knock-out
        def dmean(S): return float((P.ctr(M.run(BT, P.ctl_ablate(M, S, Mh))) - a0).mean())
        Lall = M.run(BT, P.ctl_ablate(M, range(80), Mh)); dall = P.ctr(Lall) - a0; d80 = float(dall.mean())
        d16 = dmean(CLS); db1 = dmean(range(64, 72)); db2 = dmean(range(72, 80)); dpart = dmean(range(64))
        A = dict(d_all80=d80, share_all16_class=d16 / d80, share_b1=db1 / d80, share_b2=db2 / d80, share_all64_particle=dpart / d80)
        for k in (5, 10):
            S = rank[:k]; Lk = M.run(BT, P.ctl_ablate(M, S, Mh)); dk = P.ctr(Lk) - a0; sh = float(dk.mean() / d80)
            e = dict(heads=[P.hlab(q) for q in S], n_class=int(sum(q >= 64 for q in S)), share=sh, share_ci=P.ci((W_T @ dk) / (W_T @ dall)),
                     dQCD=float(dk.mean()), top_recall=float((Lk.argmax(1) == TOP).mean()), share_of_all16=float(dk.mean() / d16))
            for mode, nr in (('all', NR), ('class', NR), ('block', NRB)):
                v = [dmean(matched(S, rr, mode)) / d80 for _ in range(nr)]; e[f'random_{mode}'] = summ(v, sh)
            A[str(k)] = e
            log(f'  A k={k}: share {sh:.3f} {np.round(e["share_ci"], 3).tolist()} | all {e["random_all"]["mean"]:.3f} (p95 {e["random_all"]["p95"]:.3f}) '
                f'| class-matched {e["random_class"]["mean"]:.3f} (p95 {e["random_class"]["p95"]:.3f}, p {e["random_class"]["p_emp"]:.3f}) '
                f'| block-matched {e["random_block"]["mean"]:.3f} (p95 {e["random_block"]["p95"]:.3f}, p {e["random_block"]["p_emp"]:.3f}); all-16 class {d16 / d80:.3f}')
        res['joint'] = A
        # ---- B specificity on EVAL (circuit top-5 for full and kin, top-10 for kin)
        lo0 = np.stack([P.lodds(L, c) for c in range(10)], 1); cls_idx = [np.where(y == c)[0] for c in range(10)]

        def perclass(S):
            Lk = M.run(B, P.ctl_ablate(M, S, Mh)); ck = Lk.argmax(1) == y
            return (np.array([(P.lodds(Lk[ix], c) - lo0[ix, c]).mean() for c, ix in enumerate(cls_idx)]),
                    np.array([(ck[ix].astype(float) - corr0[ix]).mean() for ix in cls_idx]))
        Bres = {}
        for k in ((5,) if key == 'full' else (5, 10)):
            S = rank[:k]; dlo, drc = perclass(S); rl, rc = [], []
            for _ in range(NRB):
                a_, b_ = perclass(matched(S, rr, 'class')); rl.append(a_); rc.append(b_)
            rl, rc = np.array(rl), np.array(rc)
            Bres[str(k)] = dict(heads=[P.hlab(q) for q in S], dlo_own={P.CLN[c]: float(dlo[c]) for c in range(10)},
                                drecall={P.CLN[c]: float(drc[c]) for c in range(10)},
                                random_class_dlo_mean={P.CLN[c]: float(rl[:, c].mean()) for c in range(10)},
                                p_dlo_more_negative={P.CLN[c]: pemp(dlo[c], rl[:, c], larger=False) for c in range(10)},
                                random_class_drecall_mean={P.CLN[c]: float(rc[:, c].mean()) for c in range(10)},
                                top_rank_among_classes_dlo=int(1 + (dlo < dlo[TOP]).sum()))
            log(f'  B k={k}: own-class dlo ' + ', '.join(f'{P.CLN[c]} {dlo[c]:+.2f} (rand {rl[:, c].mean():+.2f}, p {Bres[str(k)]["p_dlo_more_negative"][P.CLN[c]]:.3f})' for c in range(10)))
            log(f'           drecall ' + ', '.join(f'{P.CLN[c]} {drc[c]:+.3f}' for c in range(10)))
        res['specificity'] = Bres
        # ---- C edits, patch share vs matched random
        Cres = {}
        sets = {'all16': CLS, 'b1': list(range(64, 72))}
        for k in (5, 10):
            S = rank[:k]; sets[f'circ{k}'] = S
            for r_ in range(NRB): sets[f'rc{k}_{r_}'] = matched(S, rr, 'class'); sets[f'rb{k}_{r_}'] = matched(S, rr, 'block')
        for en in ('a', 'b', 'c'):
            E_, mp_ = EDS[en]; L0e, L1e, LP = P.patch_runs(M, BV, E_, mp_, sets)
            dfull = P.lodds(L1e) - P.lodds(L0e); den = dfull.mean(); sh = lambda n: float((P.lodds(LP[n]) - P.lodds(L0e)).mean() / den)
            e = dict(dlo_t=P.mstat(dfull, Wv, SFv), top_before=float((L0e.argmax(1) == TOP).mean()), top_after=float((L1e.argmax(1) == TOP).mean()),
                     share_all16=sh('all16'), share_b1=sh('b1'), _d=dfull)
            for k in (5, 10):
                dp = P.lodds(LP[f'circ{k}']) - P.lodds(L0e); s_ = float(dp.mean() / den)
                e[f'patch{k}'] = dict(share=s_, share_ci=P.ci((Wv @ dp) / (Wv @ dfull)),
                                      random_class=summ([sh(f'rc{k}_{r_}') for r_ in range(NRB)], s_),
                                      random_block=summ([sh(f'rb{k}_{r_}') for r_ in range(NRB)], s_))
            Cres[en] = e
            log(f'  C edit {en}: dlo_t {e["dlo_t"]["v"]:+.3f} {np.round(e["dlo_t"]["ci"], 3).tolist()}; all-16 share {e["share_all16"]:.3f}, b1 {e["share_b1"]:.3f}; '
                + '; '.join(f'circ{k} {e[f"patch{k}"]["share"]:.3f} vs class-matched {e[f"patch{k}"]["random_class"]["mean"]:.3f} (p95 {e[f"patch{k}"]["random_class"]["p95"]:.3f}, '
                            f'p {e[f"patch{k}"]["random_class"]["p_emp"]:.3f}) block-matched {e[f"patch{k}"]["random_block"]["mean"]:.3f} (p {e[f"patch{k}"]["random_block"]["p_emp"]:.3f})' for k in (5, 10)))
        Cres['a_minus_b'] = P.mstat(Cres['a']['_d'] - Cres['b']['_d'], Wv, SFv); Cres['a_minus_c'] = P.mstat(Cres['a']['_d'] - Cres['c']['_d'], Wv, SFv)
        res['edits'] = Cres; res['edit_checks'] = echk
        # ---- D literal circuits vs class-matched random sets on SUB
        Dres = {}
        for variant in ('literal', 'cls_mlp'):
            mlp_idx = P.VARIANTS[variant]
            for k in (5, 10):
                S = rank[:k]; e = P.acc_block(M.run(BSUB, P.ctl_keep(M, S, Mh, Mf, mlp_idx)), ysub, conf=True)
                rv = dict(balanced=[], top_recall=[], auc_tq=[])
                for _ in range(NRB):
                    er = P.acc_block(M.run(BSUB, P.ctl_keep(M, matched(S, rr, 'class'), Mh, Mf, mlp_idx)), ysub, conf=False)
                    for q in rv: rv[q].append(er[q] if er[q] is not None else np.nan)
                e['random_class'] = {q: summ(np.nan_to_num(v, nan=0.5), e[q]) for q, v in rv.items()}
                Dres[f'{variant}_{k}'] = e
                log(f'  D {variant} k={k}: balanced {e["balanced"]:.3f} top recall {e["top_recall"]:.3f} AUC {e["auc_tq"]:.3f} | class-matched balanced '
                    f'{e["random_class"]["balanced"]["mean"]:.3f} (p95 {e["random_class"]["balanced"]["p95"]:.3f}, p {e["random_class"]["balanced"]["p_emp"]:.3f}); '
                    f'AUC {e["random_class"]["auc_tq"]["mean"]:.3f} (p {e["random_class"]["auc_tq"]["p_emp"]:.3f})')
        res['circuits'] = Dres; res['runtime_s'] = time.time() - T0
        fn = OUTD / f'p37_verify_{key}{TAG}.json'
        fn.write_text(json.dumps(P.strip(dict(node='P37v', seed=SEED, file=f'test_20M <sample>_{FILE_NO}.root', NEVAL=NEVAL, NREF=NREF, NTOPX=NTOPX,
                                             NSUB=NSUB, NR=NR, NRB=NRB, model=res)), indent=1)); log(f'  wrote {fn}')
        del M; torch.cuda.empty_cache()
    log('done')


if __name__ == '__main__':
    main()
