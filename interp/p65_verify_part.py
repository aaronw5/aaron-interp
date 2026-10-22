"""P65 verifier (loop C): rerun of the key P65 measurements on a fresh file and seed, plus the controls P65 did not run.

Sample. JetClass test_20M <sample>_118.root (P65 used _117), numpy default_rng(6599), random rows (never first-k):
  H->bb, H->cc, Z->qq, QCD: NE evaluation rows each + NSP rows each for the same-class impact-parameter pool;
  other six classes: NOTH evaluation rows each; all 10 classes: NDON donor rows each (literal-circuit donors, mean-ablation means);
  QCD: NPOOL rows = light-track pool. Every evaluation / donor jet passes once through part_core.build (rebuild).
Frozen from P65 (not re-ranked here): head ranking (resample du_bc on _117 ranking rows) and the 32 class-MLP neurons.
Score s_bc = logit(H->bb) - logit(H->cc); margin u_bc = +s_bc on H->bb, -s_bc on H->cc; du = mean change of u (P65 definitions).

1  Input edits a, b, c, d of P65 (same code) + new control e: the (d0, sigma(d0), dz, sigma(dz)) tuples of a jet's charged tracks
   permuted among its own charged tracks (the jet keeps its set of impact-parameter values; which track carries which value is
   scrambled). Edit d (values drawn from same-class tracks of other jets) also changes which values the jet has; e isolates the
   track-value assignment.
2  Head resample ablation with two donor types, on the H->bb + H->cc evaluation jets:
     other-flavour donor (P65): pT-matched jet of the other class (H->bb <- H->cc and back);
     same-class donor (null, missing in P65): pT-matched different jet of the same class.
   A head that carries flavour gives du(other) << du(same); a swap that only disrupts the network gives du(other) ~ du(same).
   Units: P65 top-8 heads and 8 random other heads, one at a time; joint sets top-1/2/4/8, all 8 heads of class block 1, all 64
   particle-block heads; particle-block MLP outputs (each block alone, all 8 together). Mean ablation of the same single heads.
   Test of other vs same: per-jet difference, bootstrap CI, normal-approximation p, Holm over the tests of each family.
3  Key masking (P65): one head's keys read the edit-a stream (all tracks' impact parameters light-matched); share of the edit-a du.
4  Literal circuit (10-class donors, as P65): frozen top-k heads (k = 8, 32) + 32 class-MLP neurons; with the particle-block MLPs
   kept; and the missing controls: particle MLPs kept with 0 heads, and particle MLPs kept with NRS block-matched random head sets
   (k = 8, 32). 10-class accuracy and confusion for each.

  ./sync.sh && ./part_job.sh c-p65-verify 'python -m interp.p65_verify_part --smoke && python -m interp.p65_verify_part'
  -> $AI_RESULTS/part/p65_verify[_smoke].json, p65_verify[_smoke].png
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p65_flavour_circuit_part as Q   # noqa: E402  (Run, Pair, edits, builders; parses --smoke through p32)

P = Q.P
SMOKE = P.SMOKE
SEED = 6599
FILE = '118'
NE = 60 if SMOKE else 2000
NOTH = 20 if SMOKE else 300
NDON = 20 if SMOKE else 300
NSP = 40 if SMOKE else 1000
NPOOL = 80 if SMOKE else 2000
NB = 100 if SMOKE else 1000
NRS = 3 if SMOKE else 20
NSUB = 30 if SMOKE else 1000
QCD, HBB, HCC, ZQQ = Q.QCD, Q.HBB, Q.HCC, Q.ZQQ
FN = P.OUTD / f'p65_verify{P.TAG}.json'
log = P.log
RANK = ['b1h8', 'b1h2', 'b1h3', 'b1h7', 'p7h5', 'b1h1', 'p8h4', 'b2h1', 'p1h5', 'p6h3', 'p7h6', 'p6h6', 'p3h3', 'p6h5', 'p7h8', 'p5h5',
        'b2h2', 'p2h3', 'p1h3', 'b2h7', 'b2h5', 'p6h4', 'p4h5', 'p5h4', 'b2h3', 'b2h4', 'p3h5', 'p4h7', 'p2h5', 'p1h6', 'p4h6', 'b1h5']
NEUR_L = ['b2n116', 'b2n267', 'b2n22', 'b2n157', 'b2n74', 'b2n375', 'b2n427', 'b2n418', 'b2n303', 'b2n292', 'b1n79', 'b1n105', 'b2n251',
          'b1n366', 'b2n201', 'b1n114', 'b2n284', 'b1n398', 'b2n156', 'b1n92', 'b1n222', 'b2n349', 'b1n185', 'b1n310', 'b1n435', 'b2n144',
          'b1n76', 'b1n99', 'b2n474', 'b1n46', 'b1n106', 'b2n312']


def hparse(s):
    b, h = int(s[1]), int(s[3:]) - 1
    return (b - 1, h) if s[0] == 'p' else (7 + b, h)


def nparse(s): return (int(s[1]) - 1, int(s.split('n')[1]))


def same_donors(pt, ycls, rng, k=10):
    """pT-matched donor of the same class, never the jet itself"""
    D = np.arange(len(pt))
    for c in np.unique(ycls):
        ii = np.where(ycls == c)[0]; n = len(ii)
        if n < 2: continue
        o = np.argsort(pt[ii]); rk = np.empty(n, int); rk[o] = np.arange(n)
        off = rng.integers(1, k // 2 + 1, n) * rng.choice([-1, 1], n); r2 = rk + off
        r2 = np.where((r2 < 0) | (r2 >= n), rk - off, r2); D[ii] = ii[o[np.clip(r2, 0, n - 1)]]
    return D


def other_donors(pt, ycls, pairs, rng):
    D = np.arange(len(pt))
    for ka, kb in pairs:
        ia, ib = np.where(ycls == ka)[0], np.where(ycls == kb)[0]; D[ia] = ib[Q.match_donors(pt[ia], pt[ib], rng)]
    return D


def ed_perm(B, rng):
    E = {k: (v.copy() if np.ndim(v) else v) for k, v in B.items()}; ch = Q.charged(B)
    for j in range(len(ch)):
        idx = np.where(ch[j])[0]
        if len(idx) < 2: continue
        p = idx[rng.permutation(len(idx))]
        for k in Q.IPK: E[k][j, idx] = B[k][j, p]
    return E


def mk_mlp(blocks):
    return lambda idx, Pn, rec: dict(mlp_rep={i: rec['mlp'][i] for i in blocks})


def diff_stat(pr, L0, La, Lb):
    d = (pr.u(La) - pr.u(L0)) - (pr.u(Lb) - pr.u(L0)); return P.mstat(d, pr.W)


def main():
    import torch
    from interp.part_core import sub, cat, PRE, CL, sample, kin
    P.OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    R = dict(node='P65 verify', seed=SEED, smoke=SMOKE, files=f'test_20M <sample>_{FILE}.root', NE=NE, NOTH=NOTH, NDON=NDON, NSP=NSP,
             NPOOL=NPOOL, NB=NB, NRS=NRS, NSUB=NSUB, frozen_ranking=RANK, frozen_neurons=NEUR_L)
    ev, don, spool = [], [], {}
    for k, c in enumerate(CL):
        main4 = k in (QCD, HBB, HCC, ZQQ); ne = NE if main4 else NOTH
        n = ne + NDON + (NSP if main4 else 0) + (NPOOL if k == QCD else 0)
        J, rows = P.load_rows(P.TEST / f'{PRE[c]}_{FILE}.root', n, rng); perm = rng.permutation(len(rows)); o = 0
        ev.append(sub(J, np.sort(perm[o:o + ne]))); o += ne
        don.append(sub(J, np.sort(perm[o:o + NDON]))); o += NDON
        if main4: spool[k] = P.rebuild(sub(J, np.sort(perm[o:o + NSP]))); o += NSP
        if k == QCD: lpool = sub(J, np.sort(perm[o:o + NPOOL]))
        log(f'loaded {c}: {len(rows)} random rows of {PRE[c]}_{FILE}.root')
    EV = P.rebuild(cat(ev)); DON = P.rebuild(cat(don)); y = EV['label'].astype(int); yd = DON['label'].astype(int)
    LP = Q.track_pool(lpool); SP = {k: Q.track_pool(v) for k, v in spool.items()}
    R['sample'] = dict(N_eval=int(len(y)), per_class=np.bincount(y, minlength=10).tolist(), N_donor=int(len(yd)))
    E = dict(a=Q.ed_ip(EV, Q.charged(EV), LP, rng), b=Q.ed_ip(EV, Q.topk_sel(EV, 3, True), LP, rng),
             c=Q.ed_ip(EV, Q.topk_sel(EV, 3, False), LP, rng))
    Ed = {k: (v.copy() if np.ndim(v) else v) for k, v in EV.items()}
    for k in SP:
        r = np.where(y == k)[0]; part = Q.ed_ip(sub(EV, r), Q.charged(sub(EV, r)), SP[k], rng)
        for kk in Ed:
            if np.ndim(Ed[kk]): Ed[kk][r] = part[kk]
    E['d'] = Ed; E['e'] = ed_perm(EV, rng)
    ch = Q.charged(EV)
    R['edit_checks'] = {k: dict(fourmom_identical=bool(all(np.array_equal(v[q], EV[q]) for q in ('px', 'py', 'pz', 'energy', 'charge'))),
                                ip_changed_frac=float((v['d0val'] != EV['d0val'])[ch].mean()),
                                neutral_ip_unchanged=bool(np.array_equal(v['d0val'][~ch], EV['d0val'][~ch])),
                                per_jet_ip_multiset_same=bool(all(np.array_equal(np.sort(v['d0val'][j][ch[j]]), np.sort(EV['d0val'][j][ch[j]])) for j in range(len(y)))))
                        for k, v in E.items()}
    log('edit checks', R['edit_checks'])
    W_bc = P.boot_W(int(((y == HBB) | (y == HCC)).sum()), NB, rng); W_bl = P.boot_W(int(((y == HBB) | (y == ZQQ)).sum()), NB, rng)
    pe_bc, pe_bl = Q.Pair(y, HCC, Q.sbc, W_bc), Q.Pair(y, ZQQ, Q.sbl, W_bl)
    M = Q.Run('full'); out = {}
    P1s = P.rebuild(sample(100 if SMOKE else 1000, 3000)); yP1 = P1s['label'].astype(int)
    out['acc_P1'] = float((M.run(P1s).argmax(1) == yP1).mean())
    M.run(DON, lambda idx, Pn, rec: dict(recmean=True, check=True)); M.fix_means(); out['manual_attention_recon_max'] = M.recon
    L0 = M.run(EV); out['eval'] = dict(Q.acc_block(L0, y), **Q.pair_metrics(L0, y)); out['eval']['acc_bc_ci'] = pe_bc.acc_ci(L0)
    log(f'acc P1 {out["acc_P1"]:.4f}; eval acc10 {out["eval"]["acc10"]:.4f} acc_bc {out["eval"]["acc_bc"]:.4f} {out["eval"]["acc_bc_ci"]}')
    # ---- 1 input edits
    ic, _, LE = Q.input_check(M, EV, E, y, pe_bc, pe_bl, L0); out['input'] = ic
    da = pe_bc.u(LE['a']) - pe_bc.u(L0)
    for x in ('d', 'e'):
        dx = pe_bc.u(LE[x]) - pe_bc.u(L0)
        out[f'a_minus_{x}'] = dict(du_bc=P.mstat(da - dx, pe_bc.W), share_of_a=float(dx.mean() / da.mean()),
                                   share_of_a_ci=P.ci((pe_bc.W.astype(np.float64) @ dx) / (pe_bc.W.astype(np.float64) @ da)),
                                   mcnemar=P.mcnemar(pe_bc.u(LE[x]) > 0, pe_bc.u(LE['a']) > 0))
        log(f'  a - {x}: {out[f"a_minus_{x}"]["du_bc"]["v"]:+.3f} {np.round(out[f"a_minus_{x}"]["du_bc"]["ci"], 3).tolist()}; {x} / a = {out[f"a_minus_{x}"]["share_of_a"]:.3f}')
    # ---- 2 head / MLP swaps, other-flavour vs same-class donor (H->bb + H->cc jets)
    ibc = np.where(np.isin(y, [HBB, HCC]))[0]; Jbc = sub(EV, ibc); ybc = y[ibc]; L0bc = L0[ibc]; ptb = kin(Jbc)[0].sum(1)
    pbc = Q.Pair(ybc, HCC, Q.sbc, P.boot_W(len(ibc), NB, rng))
    Do = other_donors(ptb, ybc, ((HBB, HCC), (HCC, HBB)), rng); Ds = same_donors(ptb, ybc, rng)
    R['donor_checks'] = dict(other_all_other_class=bool((ybc[Do] != ybc).all()), same_all_same_class=bool((ybc[Ds] == ybc).all()),
                             same_never_self=bool((Ds != np.arange(len(ibc))).all()),
                             median_abs_log_pt_ratio_other=float(np.median(np.abs(np.log(ptb[Do] / ptb)))),
                             median_abs_log_pt_ratio_same=float(np.median(np.abs(np.log(ptb[Ds] / ptb)))))
    log('donor checks', R['donor_checks'])
    top8 = [hparse(s) for s in RANK[:8]]
    rest = [h for h in Q.HEADS if h not in top8]; rnd8 = [rest[i] for i in rng.choice(len(rest), 8, replace=False)]
    singles = top8 + rnd8
    joints = [('top1', top8[:1]), ('top2', top8[:2]), ('top4', top8[:4]), ('top8', top8), ('classblock1_all8', [(8, h) for h in range(8)]),
              ('particle64', [h for h in Q.HEADS if h[0] < 8])]
    mlps = [(f'pmlp{i + 1}', [i]) for i in range(8)] + [('pmlp_all8', list(range(8)))]
    units = [(Q.hlab(h), Q.mk_rep(M, [h])) for h in singles] + [(nm, Q.mk_rep(M, g)) for nm, g in joints] + [(nm, mk_mlp(b)) for nm, b in mlps]
    tt = time.time()
    Lo, LDo = M.multi(Jbc, [f for _, f in units] + [Q.mk_mean(M, [h]) for h in singles], Jd=Jbc, D=Do, dctl=Q.REC_ALL)
    Ls, LDs = M.multi(Jbc, [f for _, f in units], Jd=Jbc, D=Ds, dctl=Q.REC_ALL); log(f'  swap runs {time.time() - tt:.0f} s')
    R['donor_checks']['donor_logits_minus_clean_max'] = float(max(np.abs(LDo - L0bc[Do]).max(), np.abs(LDs - L0bc[Ds]).max()))
    sw = {}
    for j, (nm, _) in enumerate(units):
        eo, es = Q.eff(pbc, L0bc, Lo[j]), Q.eff(pbc, L0bc, Ls[j]); dd = diff_stat(pbc, L0bc, Lo[j], Ls[j])
        sw[nm] = dict(other=Q.strip_e(eo), same=Q.strip_e(es), other_minus_same=dd, acc_bc_other=pbc.acc(Lo[j]), acc_bc_same=pbc.acc(Ls[j]),
                      mcnemar_other_vs_same=P.mcnemar(pbc.u(Ls[j]) > 0, pbc.u(Lo[j]) > 0))
        if j < len(singles): sw[nm]['mean_ablation'] = Q.strip_e(Q.eff(pbc, L0bc, Lo[len(units) + j])); sw[nm]['random_head'] = j >= 8
    fam = {'single': [Q.hlab(h) for h in singles], 'joint': [nm for nm, _ in joints], 'mlp': [nm for nm, _ in mlps]}
    for f, names in fam.items():
        ph = P.holm([sw[n]['other_minus_same']['p0'] for n in names])
        for n, p in zip(names, ph): sw[n]['other_minus_same']['p_holm'] = p
    for nm, _ in units:
        e = sw[nm]
        log(f'  {nm}: other {e["other"]["v"]:+.3f} ({e["other"]["share_margin"]:.3f} of margin) same {e["same"]["v"]:+.3f} ({e["same"]["share_margin"]:.3f}); '
            f'diff {e["other_minus_same"]["v"]:+.3f} {np.round(e["other_minus_same"]["ci"], 3).tolist()} p_holm {e["other_minus_same"]["p_holm"]:.2e}; '
            f'acc_bc other {e["acc_bc_other"]:.4f} same {e["acc_bc_same"]:.4f}' + (f'; mean abl {e["mean_ablation"]["v"]:+.3f}' if 'mean_ablation' in e else ''))
    out['swap'] = sw; out['swap_margin'] = float(pbc.u(L0bc).mean())
    # ---- 3 key masking (edit-a keys)
    ref = pbc.u(LE['a'][ibc]) - pbc.u(L0bc); Ea = sub(E['a'], ibc)
    kg = [('b1h8', [hparse('b1h8')]), ('b1h2', [hparse('b1h2')]), ('top8', top8), ('all80', Q.HEADS)]
    Lk, _ = M.multi(Jbc, [Q.mk_key(M, g) for _, g in kg], Jd=Ea, D=np.arange(len(ibc)), dctl=Q.REC_KEY)
    out['key'] = {nm: Q.strip_e(Q.eff(pbc, L0bc, L, ref)) for (nm, _), L in zip(kg, Lk)}
    for nm, _ in kg: log(f'  key {nm}: du {out["key"][nm]["v"]:+.3f} share of input a {out["key"][nm]["share_input"]:.3f} {np.round(out["key"][nm]["share_input_ci"], 3).tolist()}')
    # ---- 4 literal circuit with 10-class donors
    NEUR = [nparse(s) for s in NEUR_L]; sets = {k: [hparse(s) for s in RANK[:k]] for k in (8, 32)}
    D = Q.match_donors(kin(EV)[0].sum(1), kin(DON)[0].sum(1), rng)
    var = [('k8', Q.mk_circ(M, sets[8], NEUR)), ('k32', Q.mk_circ(M, sets[32], NEUR)),
           ('k8_pmlp', Q.mk_circ(M, sets[8], NEUR, pmlp_keep=True)), ('k32_pmlp', Q.mk_circ(M, sets[32], NEUR, pmlp_keep=True)),
           ('k0_pmlp', Q.mk_circ(M, [], NEUR, pmlp_keep=True)), ('k0_pmlp_no_neurons', Q.mk_circ(M, [], [], pmlp_keep=True)),
           ('all80_allneurons_pmlp_ablated', Q.mk_circ(M, Q.HEADS, [], all_neurons=True)),
           ('nothing', Q.mk_circ(M, [], [])), ('everything', Q.mk_circ(M, Q.HEADS, [], pmlp_keep=True, all_neurons=True))]
    Lc, LD = M.multi(EV, [f for _, f in var], Jd=DON, D=D, dctl=Q.REC_ALL)
    circ = {}
    for (nm, _), L in zip(var, Lc):
        e = dict(Q.acc_block(L, y), **Q.pair_metrics(L, y)); e['acc_bc_ci'] = pe_bc.acc_ci(L); circ[nm] = e
        log(f'  circuit {nm}: acc_bc {e["acc_bc"]:.4f} {np.round(e["acc_bc_ci"], 4).tolist()} AUC {e["auc_bc"]:.4f}; acc10 {e["acc10"]:.4f}')
    circ['checks'] = dict(nothing_minus_donor=float(np.abs(Lc[7] - LD).max()), everything_minus_clean=float(np.abs(Lc[8] - L0).max()))
    sub_i = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], min(NSUB, int((y == k).sum())), replace=False) for k in (HBB, HCC)]))
    Es, ys, Dsub = sub(EV, sub_i), y[sub_i], D[sub_i]
    rs = {k: [Q.block_matched(sets[k], rng) for _ in range(NRS)] for k in (8, 32)}
    mks = [Q.mk_circ(M, sets[k], NEUR, pmlp_keep=pk) for k in (8, 32) for pk in (False, True)]
    mks += [Q.mk_circ(M, s, NEUR, pmlp_keep=pk) for k in (8, 32) for pk in (False, True) for s in rs[k]]
    mks += [Q.mk_circ(M, [], NEUR, pmlp_keep=True)]
    Lr, _ = M.multi(Es, mks, Jd=DON, D=Dsub, dctl=Q.REC_ALL)
    rnd = {}; j = 4
    for a, k in enumerate((8, 32)):
        for b, pk in enumerate((False, True)):
            sel = Q.pair_metrics(Lr[2 * a + b], ys)['acc_bc']; rr = [Q.pair_metrics(Lr[j + r], ys)['acc_bc'] for r in range(NRS)]; j += NRS
            nm = f'k{k}' + ('_pmlp' if pk else ''); rnd[nm] = dict(selected=sel, random=Q.summ(rr), p=Q.emp_p(sel, rr))
            log(f'  random {nm}: selected {sel:.4f} random mean {rnd[nm]["random"]["mean"]:.4f} [{rnd[nm]["random"]["min"]:.4f}, {rnd[nm]["random"]["max"]:.4f}] p {rnd[nm]["p"]:.3f}')
    rnd['k0_pmlp_sub'] = Q.pair_metrics(Lr[-1], ys)['acc_bc']; circ['random_sub'] = rnd; out['circuit'] = circ
    R['ParT_full'] = out; R['runtime_s'] = time.time() - P.T0
    FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'done; wrote {FN}')
    plot(R)


def plot(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sw = R['ParT_full']['swap']; names = list(sw.keys())
    fig, ax = plt.subplots(1, 1, figsize=(13, 4.6)); x = np.arange(len(names))
    for off, key, col, lab in ((-0.2, 'other', '#c0392b', 'donor = pT-matched jet of the OTHER flavour (P65 resample ablation)'),
                               (0.2, 'same', '#2c7fb8', 'donor = pT-matched different jet of the SAME class (null)')):
        v = [sw[n][key]['share_margin'] for n in names]; ci = np.array([sw[n][key]['share_margin_ci'] for n in names]).T
        ax.bar(x + off, v, 0.4, color=col, label=lab); ax.errorbar(x + off, v, yerr=[np.array(v) - ci[0], ci[1] - np.array(v)], fmt='none', ecolor='k', lw=0.8)
    ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('change of b-vs-c margin / unedited margin'); ax.legend(fontsize=8, loc='lower left')
    ax.set_title(f'ParT_full, H→bb + H→cc jets ({R["files"]}, seed {R["seed"]}): swapping one unit\'s output from another jet\n'
                 'first 8 = P65 top heads, next 8 = random heads, then joint head sets, then particle-block MLP outputs', fontsize=9)
    fig.tight_layout(); fo = P.OUTD / f'p65_verify{P.TAG}.png'; fig.savefig(fo, dpi=140); print('wrote', fo)


if __name__ == '__main__':
    main()
