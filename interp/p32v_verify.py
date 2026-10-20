"""P32 verify (loop C) — independent re-run of the lepton-route result on a fresh sample, plus the missing controls.

Sample: JetClass test_20M <sample>_116.root (P32 used _115), numpy default_rng(9157), random rows (never first-k):
  NLV HToWW2Q1L rows + NOTH rows of each other class. Every jet passes through part_core.build; edited and unedited jets are
  rebuilt the same number of times (base = rebuild(BL)).
Re-run (all five models): e1 lepton flag -> charged hadron; e2 matched hadron -> lepton flag; e3/e3c merges; e5mu/e5mu_r.
New controls:
  e2n   matched charged hadron's flag -> neutral hadron (same-size one-hot change on a matched-pT token, no lepton flag involved)
  e1x   lepton flavour swap (electron <-> muon): flag identity vs flag presence
  tight e1 vs e2 restricted to jets whose matched hadron is within a pT factor 1.5 of the lepton (P32's control was loose)
Class-attention capture (own re-computation of the softmax of the last class block, b2 = cls_blocks[1], all 8 heads; checked
  against P32's capture): weight on the lepton, on the leading (highest-pT) particle, max weight and its slot; Hqql jets split by
  lepton pT rank (0 vs > 0), t->blv jets, other classes with and without a lepton candidate.
Knock-out replication (ParT): lepton key masked (logit -1e9) in the dominant class head, the 16 class heads, all 80 heads, vs the
  matched-hadron key.
Circuits (ParT_full, ParT_kinpid): heads not kept replaced by their mean output over a class-balanced pool (MLPs, embeddings kept,
  as in P32): P32's top-3; the dominant class head alone; dominant + 2 random particle heads (NR sets); 1 random non-dominant class
  head + 2 random particle heads (NR sets). Metrics: Hqql recall, other jets called Hqql, t->blv jets called Hqql, non-t->blv
  other jets called Hqql, 10-class accuracy, balanced (per-class mean) accuracy.
Stats: 1,000 bootstrap resamples over jets (95 % percentile intervals); McNemar for paired call changes.
  ./sync.sh && ./part_job.sh c-p32-verify 'python -m interp.p32v_verify'   -> $AI_RESULTS/part/p32_verify[_smoke].json
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p32_lepton_route as P   # noqa: E402  (parses --smoke)

SMOKE = P.SMOKE
SEED = 9157
FILE = '116'
NLV = 150 if SMOKE else 3000
NOTH = 30 if SMOKE else 400
NB = 100 if SMOKE else 1000
NR = 3 if SMOKE else 20
HQQL, TBL, H, C, HD = 5, 9, 8, 128, 16
FN = P.OUTD / f'p32_verify{P.TAG}.json'
log = P.log
DOM = {'ParT_full': (9, 0), 'ParT_kinpid': (9, 3), 'ParT_kin': (9, 3)}          # b2h1 / b2h4 (P32)
TOP3 = {'ParT_full': [(9, 0), (6, 7), (7, 5)], 'ParT_kinpid': [(9, 3), (7, 4), (3, 4)]}   # b2h1 p7h8 p8h6 / b2h4 p8h5 p4h5


def flavswap(B, tok, ise):
    E = {k: (v.copy() if np.ndim(v) else v) for k, v in B.items()}; r = np.arange(len(tok))
    for k in P.FLAGS: E[k][r, tok] = 0
    E['isMuon'][r[ise], tok[ise]] = 1; E['isElectron'][r[~ise], tok[~ise]] = 1
    return P.rebuild(E)


def add_clscap(M):
    """re-computes the class-token attention softmax of cls_blocks[1] (all heads) when ctl['clscap'] is set"""
    t = M.t; attn = M.m.cls_blocks[1].attn; f0 = attn.forward

    def g(query, key, value, key_padding_mask=None, **kw):
        if M.ctl.get('clscap'):
            W, b = attn.in_proj_weight, attn.in_proj_bias; n, S = query.shape[1], key.shape[0]
            q = (query @ W[:C].T + b[:C]).view(1, n, H, HD).permute(1, 2, 0, 3)
            k = (key @ W[C:2 * C].T + b[C:2 * C]).view(S, n, H, HD).permute(1, 2, 0, 3)
            s = (q @ k.transpose(-1, -2)) / math.sqrt(HD)
            s = s.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
            M.out['clsw'] = t.softmax(s.float(), -1)[:, :, 0, :].detach().cpu().numpy()     # (n, H, 1 + P)
        return f0(query, key, value, key_padding_mask=key_padding_mask, **kw)
    attn.forward = g


def cls_attention(M, J, tokA, tokB):
    """per jet and head: weight on token tokA, on tokB (-1 -> nan), on slot 0, max over particle keys, argmax slot"""
    n = len(J['mask']); o = {k: np.full((n, H), np.nan) for k in ('A', 'B', 'lead', 'max')}; o['arg'] = np.zeros((n, H), int)

    def aft(idx, S):
        w = S.out['clsw'][:, :, 1:]; ar = np.arange(len(idx))
        for nm, tk in (('A', tokA[idx]), ('B', tokB[idx])):
            ok = tk >= 0; o[nm][idx[ok]] = w[ar[ok], :, tk[ok]]
        o['lead'][idx] = w[:, :, 0]; o['max'][idx] = w.max(-1); o['arg'][idx] = w.argmax(-1)
    M.run(J, lambda idx, P_: dict(clscap=True), after=aft)
    return o


def bmean(x, W):
    x = np.asarray(x, np.float64); bs = W.astype(np.float64) @ x
    return dict(v=float(x.mean()), ci=P.ci(bs))


def main():
    import torch
    NW = P.add_kinpid()
    from interp.part_core import sub, cat, PRE, CL, sample
    P.OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    R = dict(node='P32v', seed=SEED, files=f'test_20M <sample>_{FILE}.root', NLV=NLV, NOTH=NOTH, NB=NB, NR=NR)
    parts = []
    for c in CL:
        J, rows = P.load_rows(P.TEST / f'{PRE[c]}_{FILE}.root', NLV if c == 'Hqql' else NOTH, rng); parts.append(J)
        log(f'loaded {c}: {len(rows)} rows')
    J0 = cat(parts); B = P.rebuild(J0); y = B['label'].astype(int); N = len(y); tok = P.lepton_idx(B)
    lv = np.where((y == HQQL) & (tok >= 0))[0]; mh, mq = P.matched_hadron(sub(B, lv), tok[lv], rng)
    keep = mh >= 0; lv, mh, mq = lv[keep], mh[keep], mq[keep]; tl = tok[lv]; BL = sub(B, lv); r_ = np.arange(len(lv))
    ise = BL['isElectron'][r_, tl] > 0.5; tight = mq < np.log(1.5)
    R['sample'] = dict(N=N, per_class=np.bincount(y, minlength=10).tolist(), no_lepton_frac=[float((tok[y == k] < 0).mean()) for k in range(10)],
                       n_lv=int(len(lv)), lepton_rank0=float((tl == 0).mean()), tight_match_frac=float(tight.mean()),
                       match_abs_log_pt_ratio_median=float(np.median(mq)))
    log('sample', R['sample'])
    nb = P.nearest_charged(BL, tl); nbc = P.nearest_charged(BL, mh, exclude=tl); sel3 = np.where((nb >= 0) & (nbc >= 0))[0]
    E = dict(base=P.rebuild(BL), e1=P.ed_flag(BL, tl, 'isChargedHadron'), e2n=P.ed_flag(BL, mh, 'isNeutralHadron'), e1x=flavswap(BL, tl, ise))
    e2 = {k: (v.copy() if np.ndim(v) else v) for k, v in BL.items()}
    for k in P.FLAGS: e2[k][r_, mh] = 0
    e2['isElectron'][r_[ise], mh[ise]] = 1; e2['isMuon'][r_[~ise], mh[~ise]] = 1; E['e2'] = P.rebuild(e2)
    E['e3'] = P.ed_merge(sub(BL, sel3), tl[sel3], nb[sel3]); E['e3c'] = P.ed_merge(sub(BL, sel3), mh[sel3], nbc[sel3])
    left = P.lepton_idx(E['e1']) >= 0
    R['edit_checks'] = dict(e1_other_lepton_left=float(left.mean()), e3_n=int(len(sel3)))
    rv = np.where(np.isin(y, [0, 4, 8]))[0]; BR = sub(B, rv); chm = (BR['isChargedHadron'] > 0.5) & (BR['mask'] > 0)
    lead = np.where(chm.any(1), chm.argmax(1), -1); rnd = np.full(len(rv), -1)
    for j in range(len(rv)):
        c_ = np.where(chm[j])[0]
        if len(c_) > 1: rnd[j] = rng.choice(c_[1:])
    rvs = np.where((lead >= 0) & (rnd >= 0))[0]; BRs = sub(BR, rvs)
    ER = dict(base=P.rebuild(BRs), e5mu=P.ed_flag(BRs, lead[rvs], 'isMuon'), e5mu_r=P.ed_flag(BRs, rnd[rvs], 'isMuon'))
    P1s = P.rebuild(sample(1000, 3000)); yP1 = P1s['label'].astype(int)
    Wall = P.boot_W(N, NB, rng); Wl = P.boot_W(len(lv), NB, rng); Wt = P.boot_W(int(tight.sum()), NB, rng); W3 = P.boot_W(len(sel3), NB, rng)
    WR = P.boot_W(len(rvs), NB, rng)
    MODELS = [('ParT_full', 'part', 'full'), ('ParT_kinpid', 'part', 'kinpid'), ('ParT_kin', 'part', 'kin'),
              ('PN_full', 'pn', 'full'), ('PN_kinpid', 'pn', 'kinpid')]
    R['models'] = {}
    for name, kind, feat in MODELS:
        log(f'==== {name}'); res = {}
        M = P.PRun(name, feat) if kind == 'part' else P.PNRun(feat)
        if kind == 'part': add_clscap(M)
        res['acc_P1'] = P.acc_block(M.run(P1s), yP1)['acc10']
        Lall = M.run(B); res['sample'] = P.acc_block(Lall, y, Wall); c_all = Lall.argmax(1)
        nl = (tok < 0) & (y == HQQL)
        res['hqql_recall_no_lepton'] = dict(n=int(nl.sum()), v=float((c_all[nl] == HQQL).mean()) if nl.any() else None)
        oth = y != HQQL
        res['others_called_hqql'] = dict(with_lepton=float((c_all[oth & (tok >= 0)] == HQQL).mean()), without=float((c_all[oth & (tok < 0)] == HQQL).mean()))
        LE = {k: M.run(v) for k, v in E.items()}; ed = {}
        for k in ('e1', 'e2', 'e2n', 'e1x'):
            ed[k] = P.edit_stats(LE['base'], LE[k], Wl)
        ed['e3'] = P.edit_stats(LE['base'][sel3], LE['e3'], W3); ed['e3c'] = P.edit_stats(LE['base'][sel3], LE['e3c'], W3)
        for a, b in (('e1', 'e2'), ('e1', 'e2n'), ('e2', 'e2n')):
            ed[f'{a}_vs_{b}'] = dict(mcnemar_changed=P.mcnemar(ed[b]['_ch'], ed[a]['_ch']), ddlogp=P.mstat(ed[a]['_d'] - ed[b]['_d'], Wl))
        ed['tight'] = {k: dict(n=int(tight.sum()), changed=bmean(ed[k]['_ch'][tight], Wt), dlogp=bmean(ed[k]['_d'][tight], Wt)) for k in ('e1', 'e2', 'e2n')}
        ed['tight']['e1_vs_e2_mcnemar'] = P.mcnemar(ed['e2']['_ch'][tight], ed['e1']['_ch'][tight])
        ed['e1_by_other_lepton_left'] = dict(left=float(ed['e1']['_ch'][left].mean()) if left.any() else None, n_left=int(left.sum()),
                                             none_left=float(ed['e1']['_ch'][~left].mean()))
        for k in ('e1', 'e2', 'e2n', 'e1x', 'e3', 'e3c'):
            log(f'  {k:4s} changed {ed[k]["calls_changed"]["v"]:.3f} {np.round(ed[k]["calls_changed"]["ci"], 3).tolist()} dlogp {ed[k]["dlogp"]["v"]:+.2f} to {ed[k]["changed_to"]}')
        log(f'  tight (n {int(tight.sum())}): ' + ', '.join(f'{k} {ed["tight"][k]["changed"]["v"]:.3f}' for k in ('e1', 'e2', 'e2n')) +
            f'; e1 with another lepton left {ed["e1_by_other_lepton_left"]}; no-lepton Hqql recall {res["hqql_recall_no_lepton"]}; others->Hqql {res["others_called_hqql"]}')
        res['edits'] = ed
        LR = {k: M.run(v) for k, v in ER.items()}; c0 = LR['base'].argmax(1); rvr = {}
        for k in ('e5mu', 'e5mu_r'):
            c1 = LR[k].argmax(1); rvr[k] = dict(to_hqql=P.mstat(((c1 == HQQL) & (c0 != HQQL)).astype(float), WR),
                                                dlogp=P.mstat(P.logsm(LR[k])[:, HQQL] - P.logsm(LR['base'])[:, HQQL], WR))
        rvr['mcnemar'] = P.mcnemar(LR['e5mu_r'].argmax(1) == HQQL, LR['e5mu'].argmax(1) == HQQL)
        log(f'  e5mu to Hqql {rvr["e5mu"]["to_hqql"]["v"]:.4f} vs random hadron {rvr["e5mu_r"]["to_hqql"]["v"]:.4f}')
        res['reverse'] = rvr
        if kind == 'part':
            res['internals'] = internals(M, name, B, y, tok, E, LE, lv, tl, mh, Wl, Wall, rng)
        R['models'][name] = res
        FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'  wrote {FN}')
        del M; torch.cuda.empty_cache()
    R['runtime_s'] = time.time() - P.T0
    FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'done; wrote {FN}')


def internals(M, name, B, y, tok, E, LE, lv, tl, mh, Wl, Wall, rng):
    from interp.part_core import sub
    out = {}; dom = DOM[name]; h = dom[1]
    # ---- class attention (unedited jets): own capture, cross-checked against P32's capture on the Hqql jets
    tokB = np.full(len(y), -1); tokB[lv] = mh
    A = cls_attention(M, B, tok, tokB)
    capl = np.zeros((len(lv), H))

    def aft(idx, S): capl[idx] = S.out['A'][9][0]
    M.run(sub(B, lv), lambda idx, P_: dict(cap=dict(tok=M.t.from_numpy(tl[idx]).to(M.dev).long(), tokc=M.t.from_numpy(mh[idx]).to(M.dev).long())), after=aft)
    out['capture_check_max_abs'] = float(np.nanmax(np.abs(capl - A['A'][lv])))
    hl = np.zeros(len(y), bool); hl[lv] = True; rank0 = np.zeros(len(y), bool); rank0[lv] = tl == 0
    groups = dict(hqql_lepton_rank0=hl & rank0, hqql_lepton_rank_gt0=hl & ~rank0, tbl_with_lepton=(y == TBL) & (tok >= 0),
                  other_with_lepton=(y != HQQL) & (y != TBL) & (tok >= 0), other_without_lepton=(y != HQQL) & (tok < 0))
    att = {}
    for g, m in groups.items():
        att[g] = dict(n=int(m.sum()), per_head={f'b2h{k + 1}': dict(
            lepton=float(np.nanmean(A['A'][m, k])) if (tok[m] >= 0).any() else None, matched=float(np.nanmean(A['B'][m, k])) if (tokB[m] >= 0).any() else None,
            leading=float(A['lead'][m, k].mean()), max=float(A['max'][m, k].mean()), argmax_is_lepton=float((A['arg'][m, k] == tok[m]).mean()),
            argmax_is_leading=float((A['arg'][m, k] == 0).mean())) for k in range(H)})
        a = att[g]['per_head'][f'b2h{h + 1}']
        log(f'  b2h{h + 1} [{g}, n {att[g]["n"]}]: lepton {a["lepton"]} matched {a["matched"]} leading {a["leading"]:.3f} max {a["max"]:.3f} '
            f'argmax=lepton {a["argmax_is_lepton"]:.3f} argmax=leading {a["argmax_is_leading"]:.3f}')
    out['attention_b2'] = att; out['dominant'] = f'b2h{h + 1}'
    # ---- knock-out replication on all lv jets
    BK = E['base']; L0 = LE['base']; lo0 = P.lodds(L0); c0 = L0.argmax(1)
    HEADS = P.HEADS

    def run_ko(heads, toks):
        return M.run(BK, lambda idx, P_: dict(kmask=P.kmask_for(M, len(idx), P_, heads, toks[idx])))
    ko = {}
    for lab, hs in (('dominant', [dom]), ('class16', [x for x in HEADS if x[0] >= 8]), ('all80', HEADS)):
        La, Lb = run_ko(hs, tl), run_ko(hs, mh); a, b = P.lodds(La) - lo0, P.lodds(Lb) - lo0
        ko[lab] = dict(lepton=P.mstat(a, Wl), matched=P.mstat(b, Wl), changed=bmean((La.argmax(1) != c0).astype(float), Wl),
                       changed_matched=bmean((Lb.argmax(1) != c0).astype(float), Wl), mcnemar=P.mcnemar(Lb.argmax(1) != c0, La.argmax(1) != c0))
        log(f'  KO {lab}: lepton {ko[lab]["lepton"]["v"]:+.2f} {np.round(ko[lab]["lepton"]["ci"], 2).tolist()} matched {ko[lab]["matched"]["v"]:+.2f} '
            f'changed {ko[lab]["changed"]["v"]:.3f} / {ko[lab]["changed_matched"]["v"]:.3f}')
    out['ko'] = ko
    if name not in TOP3: return out
    # ---- circuits (all 80 heads in scope; MLPs and embeddings kept, as in P32)
    pool = np.concatenate([np.where(y != HQQL)[0], rng.choice(np.where(y == HQQL)[0], int(np.bincount(y[y != HQQL]).max()), replace=False)])
    M.msum = {i: M.t.zeros(H, C, dtype=M.t.float64, device=M.dev) for i in range(10)}; M.mcnt = {i: 0.0 for i in range(10)}
    M.run(sub(B, pool), lambda idx, P_: dict(recmean=True)); MU = M.means()
    t = M.t

    def circ(keep):
        abl = [x for x in HEADS if x not in keep]

        def mk(idx, P_):
            d = {}
            for (i, hh) in abl:
                if i not in d: d[i] = (t.zeros(H, dtype=t.bool, device=M.dev), MU[i])
                d[i][0][hh] = True
            return dict(mean=d)
        return M.run(B, mk)

    def metr(L, full=False):
        c = L.argmax(1); e = P.acc_block(L, y, Wall if full else None)
        e['hqql_recall'] = e['recall'][HQQL]; e['other_called_hqql'] = float((c[y != HQQL] == HQQL).mean())
        e['tbl_called_hqql'] = float((c[y == TBL] == HQQL).mean())
        e['non_tbl_other_called_hqql'] = float((c[(y != HQQL) & (y != TBL)] == HQQL).mean())
        if full:
            hq = y == HQQL; e['hqql_recall_ci'] = P.ci(Wall[:, hq] @ (c[hq] == HQQL).astype(np.float32) / Wall[:, hq].sum(1))
            e['balanced_ci'] = P.ci(np.stack([Wall[:, y == k] @ (c[y == k] == k).astype(np.float32) / Wall[:, y == k].sum(1) for k in range(10)], 1).mean(1))
        else: e.pop('confusion')
        return e
    CI = {}
    CI['full_model'] = metr(M.run(B), True)
    CI['top3'] = metr(circ(TOP3[name]), True); CI['top3']['heads'] = [P.hlab(x) for x in TOP3[name]]
    CI['dominant_only'] = metr(circ([dom]), True); CI['dominant_only']['heads'] = [P.hlab(dom)]
    cls_other = [x for x in HEADS if x[0] >= 8 and x != dom]
    for lab in ('dominant_plus2', 'random3_no_dominant'):
        rv = []
        for _ in range(NR):
            hs = [HEADS[j] for j in rng.choice(64, 2, replace=False)]
            hs = ([dom] if lab == 'dominant_plus2' else [cls_other[rng.integers(len(cls_other))]]) + hs
            e = metr(circ(hs)); e['heads'] = [P.hlab(x) for x in hs]; rv.append(e)
        CI[lab] = dict(n=NR, sets=rv, **{f'{k}_mean': float(np.mean([r[k] for r in rv])) for k in
                                         ('hqql_recall', 'other_called_hqql', 'tbl_called_hqql', 'non_tbl_other_called_hqql', 'acc10', 'balanced')},
                       **{f'{k}_range': [float(np.min([r[k] for r in rv])), float(np.max([r[k] for r in rv]))] for k in ('hqql_recall', 'tbl_called_hqql', 'balanced')})
    for k in ('full_model', 'top3', 'dominant_only'):
        e = CI[k]
        log(f'  circuit {k} {e.get("heads", "")}: recall {e["hqql_recall"]:.3f} {np.round(e["hqql_recall_ci"], 3).tolist()} others->Hqql {e["other_called_hqql"]:.3f} '
            f'(Tbl {e["tbl_called_hqql"]:.3f}, non-Tbl {e["non_tbl_other_called_hqql"]:.3f}) acc10 {e["acc10"]:.3f} balanced {e["balanced"]:.3f} {np.round(e["balanced_ci"], 3).tolist()}')
    for k in ('dominant_plus2', 'random3_no_dominant'):
        e = CI[k]
        log(f'  circuit {k} ({NR} sets): recall mean {e["hqql_recall_mean"]:.3f} range {np.round(e["hqql_recall_range"], 3).tolist()} Tbl->Hqql {e["tbl_called_hqql_mean"]:.3f} '
            f'non-Tbl->Hqql {e["non_tbl_other_called_hqql_mean"]:.3f} acc10 {e["acc10_mean"]:.3f} balanced {e["balanced_mean"]:.3f}')
    out['circuit'] = CI
    return out


if __name__ == '__main__':
    main()
