"""P34 verify (loop C) — independent rerun of the P34 lepton-circuit result on a fresh file and seed, plus missing controls.

Sample: JetClass test_20M <sample>_117.root (P32 used _115, P32v and P34 _116), numpy default_rng(7341), random rows (never
  first-k): NE rows per class (evaluation) + NREF disjoint rows per class (mean-ablation reference, and threshold calibration).
Reuses the P34 machinery (interp/p34_lepton_circuit.py: Run with per-head mean replacement and fc2 mean replacement; metr).

A  Flag edits (ParT_full, ParT_kinpid, PN_full) on H->lvqq' (Hqql) jets with a lepton candidate:
   e1 lepton flag -> charged hadron; e2n matched charged hadron -> neutral hadron; e1x e<->mu swap.
   New: e1 vs e2n on the tight-match subset (matched hadron pT within x1.5 of the lepton; P34's median match ratio was x3).
   New (sufficiency): in jets of the no-lepton classes (QCD, H->4q, Z->qq, W->qq, t->bqq') without a lepton candidate, the
   leading charged hadron's flag -> muon (s_mu) or electron (s_e), vs the same particle -> neutral hadron (s_n, control);
   share called Hqql before/after, McNemar on the Hqql call.
   Token alignment check: lepton index in rebuild(BL) equals the index found in BL.
B  ParT_full knock-outs (P34's knockouts(): b2h1, class16, particle64, all80 lepton key vs matched hadron key; b2h1 + k
   random vs k random for k = 1, 7, 31, 10 sets each).
C  Literal circuit (ParT_full, ParT_kinpid; all other heads and all 10 MLPs at their reference mean):
   dominant class head alone and P34's top-3: recall, AUC Hqql vs rest, AUC vs t->blv, recall/t->blv with the 2 class-block MLPs kept.
   New controls:
   C1 class-block MLPs kept (particle MLPs at mean) for each of the 16 class heads alone, and for NRC random composition-
      matched 3-head sets (1 class + 2 particle heads): is the H->lvqq' call specific to the lepton heads?
   C2 threshold read-out: the literal circuit's Hqql log-odds with one threshold fixed on the reference set (share called
      Hqql on reference = 0.10, the class prior); recall and t->blv called Hqql on the evaluation sample. Tests whether the
      'no call' of the literal circuit is an offset (the argmax never reaches Hqql) rather than missing information.
   C3 all 80 heads with the 2 class-block MLPs at mean / the 8 particle-block MLPs at mean / all kept: are the class-block MLPs
      needed for the call in the otherwise intact network?
Stats: bootstrap over jets (NB), McNemar exact, empirical p vs random sets.
  ./sync.sh && ./part_job.sh c-p34-verify 'python -m interp.p34v_verify --smoke && python -m interp.p34v_verify'
  -> $AI_RESULTS/part/p34_verify[_smoke].json
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p32_lepton_route as P      # noqa: E402  (parses --smoke)
import interp.p34_lepton_circuit as Q    # noqa: E402

SMOKE = P.SMOKE
SEED = 7341
FILE = '117'
NE = 60 if SMOKE else 1000
NREF = 20 if SMOKE else 300
NB = 100 if SMOKE else 1000
NRC = 4 if SMOKE else 60
HQQL, TBL = 5, 9
NOLEP = (0, 4, 6, 7, 8)
FN = P.OUTD / f'p34_verify{P.TAG}.json'
log = P.log
Q.NRKO = 3 if SMOKE else 10
Q.KS_KO = (1, 7, 31)


def tight(mq): return mq < np.log(1.5)


def lead_charged(J, ok):
    ch = (J['isChargedHadron'] > 0.5) & (J['mask'] > 0)
    return np.where(ch.any(1) & ok, ch.argmax(1), -1)


def main():
    import torch
    P.add_kinpid()
    from interp.part_core import sub, cat, PRE, CL
    P.OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    R = dict(node='P34v', seed=SEED, smoke=SMOKE, files=f'test_20M <sample>_{FILE}.root', NE=NE, NREF=NREF, NB=NB, NRC=NRC)
    pe, pr = [], []
    for c in CL:
        J, rows = P.load_rows(P.TEST / f'{PRE[c]}_{FILE}.root', NE + NREF, rng)
        perm = rng.permutation(len(rows)); pe.append(sub(J, np.sort(perm[:NE]))); pr.append(sub(J, np.sort(perm[NE:])))
    B = P.rebuild(cat(pe)); BREF = P.rebuild(cat(pr)); y = B['label'].astype(int); N = len(y); tok = P.lepton_idx(B)
    lv = np.where((y == HQQL) & (tok >= 0))[0]; mh, mq = P.matched_hadron(sub(B, lv), tok[lv], rng)
    keep = mh >= 0; lv, mh, mq = lv[keep], mh[keep], mq[keep]; tl = tok[lv]; BL = sub(B, lv); r_ = np.arange(len(lv))
    ise = BL['isElectron'][r_, tl] > 0.5; ti = np.where(tight(mq))[0]
    E = dict(base=P.rebuild(BL), e1=P.ed_flag(BL, tl, 'isChargedHadron'), e2n=P.ed_flag(BL, mh, 'isNeutralHadron'), e1x=Q.flavswap(BL, tl, ise))
    # sufficiency sample: no-lepton classes, jets without a lepton candidate, leading charged hadron
    nl = np.where(np.isin(y, NOLEP) & (tok < 0))[0]; BN = sub(B, nl); lc = lead_charged(BN, np.ones(len(nl), bool))
    okc = lc >= 0; nl, lc = nl[okc], lc[okc]; BN = sub(B, nl); yn = y[nl]
    S = dict(base=P.rebuild(BN), s_mu=P.ed_flag(BN, lc, 'isMuon'), s_e=P.ed_flag(BN, lc, 'isElectron'), s_n=P.ed_flag(BN, lc, 'isNeutralHadron'))
    R['sample'] = dict(N=N, n_lv=int(len(lv)), electron_frac=float(ise.mean()), n_tight=int(len(ti)),
                       match_abs_log_pt_ratio_median=float(np.median(mq)), lepton_rank0=float((tl == 0).mean()),
                       token_aligned=float((P.lepton_idx(E['base']) == tl).mean()), n_suff=int(len(nl)),
                       suff_per_class={CL[k]: int((yn == k).sum()) for k in NOLEP},
                       suff_lead_rank0=float((lc == 0).mean()))
    R['edit_checks'] = dict(e1_flag_cols_changed_only=bool(all(np.array_equal(E['e1'][k], E['base'][k]) for k in ('px', 'py', 'pz', 'energy', 'charge', 'd0val', 'dzval'))),
                            s_mu_lepton_added=float((P.lepton_idx(S['s_mu']) >= 0).mean()))
    log('sample', R['sample'], R['edit_checks'])
    Wall = P.boot_W(N, NB, rng); Wl = P.boot_W(len(lv), NB, rng); Wt = P.boot_W(len(ti), NB, rng); Wn = P.boot_W(len(nl), NB, rng)
    R['models'] = {}
    for name, kind, feat in (('ParT_full', 'part', 'full'), ('ParT_kinpid', 'part', 'kinpid'), ('PN_full', 'pn', 'full')):
        log(f'==== {name}'); res = {}; t0 = time.time()
        M = Q.Run(name, feat) if kind == 'part' else P.PNRun(feat)
        L0 = M.run(B); res['sample'] = Q.metr(L0, y, tok, Wall, conf=False)
        LE = {k: M.run(v) for k, v in E.items()}; ed = {}
        for k in ('e1', 'e2n', 'e1x'): ed[k] = P.edit_stats(LE['base'], LE[k], Wl)
        ed['e1_vs_e2n'] = P.mcnemar(ed['e2n']['_ch'], ed['e1']['_ch'])
        tb = {k: P.edit_stats(LE['base'][ti], LE[k][ti], Wt) for k in ('e1', 'e2n')}
        ed['tight'] = dict(n=int(len(ti)), e1=tb['e1'], e2n=tb['e2n'], mcnemar=P.mcnemar(tb['e2n']['_ch'], tb['e1']['_ch']))
        log(f'  acc10 {res["sample"]["acc10"]:.4f}; e1 {ed["e1"]["calls_changed"]["v"]:.4f} {np.round(ed["e1"]["calls_changed"]["ci"], 4).tolist()} '
            f'e2n {ed["e2n"]["calls_changed"]["v"]:.4f} {np.round(ed["e2n"]["calls_changed"]["ci"], 4).tolist()} e1x {ed["e1x"]["calls_changed"]["v"]:.4f}; '
            f'tight (n {len(ti)}): e1 {tb["e1"]["calls_changed"]["v"]:.4f} e2n {tb["e2n"]["calls_changed"]["v"]:.4f} {np.round(tb["e2n"]["calls_changed"]["ci"], 4).tolist()}')
        LS = {k: M.run(v) for k, v in S.items()}; c0 = LS['base'].argmax(1); su = {}
        for k in ('s_mu', 's_e', 's_n'):
            c1 = LS[k].argmax(1)
            su[k] = dict(hqql_before=P.mstat((c0 == HQQL).astype(float), Wn), hqql_after=P.mstat((c1 == HQQL).astype(float), Wn),
                         tbl_after=float((c1 == TBL).mean()), calls_changed=P.mstat((c1 != c0).astype(float), Wn),
                         dlodds=P.mstat(P.lodds(LS[k]) - P.lodds(LS['base']), Wn), mcnemar_hqql=P.mcnemar(c0 == HQQL, c1 == HQQL),
                         hqql_after_by_class={CL[j]: float((c1[yn == j] == HQQL).mean()) for j in NOLEP})
            log(f'  suff {k}: Hqql {su[k]["hqql_before"]["v"]:.4f} -> {su[k]["hqql_after"]["v"]:.4f} {np.round(su[k]["hqql_after"]["ci"], 4).tolist()} '
                f'(by class {np.round(list(su[k]["hqql_after_by_class"].values()), 3).tolist()}) changed {su[k]["calls_changed"]["v"]:.4f} dlodds {su[k]["dlodds"]["v"]:+.2f}')
        su['s_mu_vs_s_n'] = P.mcnemar(LS['s_n'].argmax(1) == HQQL, LS['s_mu'].argmax(1) == HQQL)
        res['edits'] = P.strip(ed); res['sufficiency'] = P.strip(su)
        if kind == 'part':
            if name == 'ParT_full':
                res['knockout'] = P.strip(Q.knockouts(M, name, E['base'], LE['base'], tl, mh, Wl, rng))
            res['circuit'] = circuit(M, name, B, y, tok, Wall, BREF, rng)
            res['head_recon_max_abs'] = M.recon
        res['runtime_s'] = time.time() - t0; R['models'][name] = res
        FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'  wrote {FN} ({res["runtime_s"]:.0f} s)')
        del M; torch.cuda.empty_cache()
    R['runtime_s'] = time.time() - P.T0; FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'done; wrote {FN}')


def circuit(M, name, B, y, tok, Wall, BREF, rng):
    from interp.part_core import sub
    out = {}; yr = BREF['label'].astype(int); tokr = P.lepton_idx(BREF)
    M.run(BREF, lambda idx, Pn: dict(rec=True)); M.fix_means()
    HEADS = P.HEADS; cls16 = [x for x in HEADS if x[0] >= 8]; par64 = [x for x in HEADS if x[0] < 8]

    def L_(keep, J, mlp):
        return M.run(J, lambda idx, Pn: M.circ_ctl(set(keep), mlp))

    def ev(keep, mlp=True, W=Wall):
        return Q.metr(L_(keep, B, mlp), y, tok, W, conf=False)
    short = lambda e: {k: e[k] for k in ('hqql_recall', 'balanced', 'acc10', 'tbl_called_hqql', 'other_called_hqql', 'auc_hqql_vs_rest', 'auc_hqql_vs_tbl', 'n_distinct_calls', 'pred_share') if k in e}
    # C3: all 80 heads, MLP variants
    for lab, mlp in (('all80_all_mlps_kept', False), ('all80_class_mlps_at_mean', Q.CMLP), ('all80_particle_mlps_at_mean', Q.PMLP), ('all80_all_mlps_at_mean', True)):
        e = ev(HEADS, mlp); out[lab] = dict(short(e), hqql_recall_ci=e['hqql_recall_ci'], tbl_called_hqql_ci=e['tbl_called_hqql_ci'], balanced_ci=e['balanced_ci'])
        log(f'  C3 {lab}: recall {e["hqql_recall"]:.3f} {np.round(e["hqql_recall_ci"], 3).tolist()} Tbl->Hqql {e["tbl_called_hqql"]:.3f} balanced {e["balanced"]:.3f}')
    dom = Q.DOM[name]; top3 = [Q.lab2ih(s) for s in Q.RANK[name]['all'][:3]]
    sel = {}
    for lab, hs in (('dominant', [dom]), ('top3', top3)):
        e = ev(hs); ec = ev(hs, Q.PMLP)
        # C2: threshold read-out calibrated on the reference set (share called Hqql = 0.10)
        lr = P.lodds(L_(hs, BREF, True)); thr = float(np.quantile(lr, 0.9)); le = P.lodds(L_(hs, B, True)); call = le > thr
        th = dict(threshold=thr, ref_share_called=float((lr > thr).mean()), hqql_recall=P.mstat(call[y == HQQL].astype(float), P.boot_W(int((y == HQQL).sum()), NB, rng)),
                  tbl_called_hqql=P.mstat(call[y == TBL].astype(float), P.boot_W(int((y == TBL).sum()), NB, rng)),
                  other_nontbl_called_hqql=float(call[(y != HQQL) & (y != TBL)].mean()),
                  other_nontbl_with_lepton_called=float(call[(y != HQQL) & (y != TBL) & (tok >= 0)].mean()),
                  other_nontbl_without_lepton_called=float(call[(y != HQQL) & (y != TBL) & (tok < 0)].mean()),
                  lodds_spread_sd=float(le.std()), ref_lodds_range=[float(lr.min()), float(lr.max())])
        sel[lab] = dict(heads=[P.hlab(x) for x in hs], literal=short(e), class_mlps_kept=dict(short(ec), hqql_recall_ci=ec['hqql_recall_ci'], tbl_called_hqql_ci=ec['tbl_called_hqql_ci']), threshold_readout=th)
        log(f'  {lab} {sel[lab]["heads"]}: literal recall {e["hqql_recall"]:.3f} calls {e["n_distinct_calls"]} AUC {e["auc_hqql_vs_rest"]:.3f} vs Tbl {e["auc_hqql_vs_tbl"]:.3f} | '
            f'class MLPs kept recall {ec["hqql_recall"]:.3f} Tbl {ec["tbl_called_hqql"]:.3f} | threshold read-out recall {th["hqql_recall"]["v"]:.3f} '
            f'Tbl {th["tbl_called_hqql"]["v"]:.3f} other-with-lepton {th["other_nontbl_with_lepton_called"]:.3f} without {th["other_nontbl_without_lepton_called"]:.3f}')
    out['selected'] = sel
    # C1: class-block MLPs kept; each class head alone (16), random composition-matched 3-head sets (NRC)
    singles = []
    for h in cls16:
        e = ev([h], Q.PMLP, None); singles.append(dict(head=P.hlab(h), **short(e)))
    out['cls_single_class_mlps_kept'] = singles
    log('  C1 single class heads + class MLPs: ' + ', '.join(f'{s["head"]} {s["hqql_recall"]:.2f}/{s["tbl_called_hqql"]:.2f}/{s["auc_hqql_vs_rest"]:.2f}' for s in singles))
    rv = []
    for _ in range(NRC):
        s = [cls16[j] for j in rng.choice(16, 1, replace=False)] + [par64[j] for j in rng.choice(64, 2, replace=False)]
        e = ev(s, Q.PMLP, None); e2 = ev(s, True, None)
        rv.append(dict(heads=[P.hlab(x) for x in s], has_dom=bool(dom in s), class_mlps_kept=short(e), literal_auc=e2['auc_hqql_vs_rest']))
    nod = [r for r in rv if not r['has_dom']]
    ss = lambda rs, k: dict(mean=float(np.mean([r['class_mlps_kept'][k] for r in rs])), max=float(np.max([r['class_mlps_kept'][k] for r in rs])),
                            q95=float(np.percentile([r['class_mlps_kept'][k] for r in rs], 95))) if rs else None
    out['random3_class_mlps_kept'] = dict(n=len(rv), n_without_dom=len(nod), sets=rv,
                                          recall=ss(rv, 'hqql_recall'), recall_without_dom=ss(nod, 'hqql_recall'),
                                          tbl=ss(rv, 'tbl_called_hqql'), balanced=ss(rv, 'balanced'),
                                          p_top3_recall=Q.emp_p(sel['top3']['class_mlps_kept']['hqql_recall'], [r['class_mlps_kept']['hqql_recall'] for r in rv]),
                                          literal_auc_mean=float(np.mean([r['literal_auc'] for r in rv])),
                                          p_top3_literal_auc=Q.emp_p(sel['top3']['literal']['auc_hqql_vs_rest'], [r['literal_auc'] for r in rv]))
    o = out['random3_class_mlps_kept']
    log(f'  C1 random 3-head sets + class MLPs ({len(rv)}; {len(nod)} without dominant): recall mean {o["recall"]["mean"]:.3f} max {o["recall"]["max"]:.3f}; '
        f'without dominant mean {o["recall_without_dom"]["mean"] if nod else float("nan"):.3f} max {o["recall_without_dom"]["max"] if nod else float("nan"):.3f}; '
        f'p(top3 recall) {o["p_top3_recall"]:.3f}; literal AUC mean {o["literal_auc_mean"]:.3f} p(top3) {o["p_top3_literal_auc"]:.3f}')
    return out


if __name__ == '__main__':
    main()
