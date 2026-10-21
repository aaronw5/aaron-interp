"""P55v (loop C, verify of P55): re-evaluate P55's frozen L-GATr head rankings on a fresh JetClass sample, and test whether
P55's resample ablation is a fair "head removed" baseline for a Lorentz-equivariant network.

Concern. L-GATr reads lab-frame four-momenta (E, px, py, pz). A head output's grade-1 part (and grades 2-4) is built from them, so
it points along the jet. P55's donor is matched only in jet pT: it sits at a random azimuth phi and a different eta. Its head
outputs therefore carry four-vectors pointing elsewhere; pasted into the target jet they may push the residual stream off the
data manifold, which would make any ablated head look necessary (P55: removing b0h0 alone takes W/Z from 82 % to 50 %).

Arms (all on the same jets; frozen rankings read from results/lgatr/p55_ranking.json, greedy order):
  pt       P55's protocol: donor = random among the 5 jets nearest in jet pT (any class, not self).
  aligned  donor = random among the 5 jets nearest in (ln pT, eta_jet) (z-scored Euclidean; any class, not self), then the whole
           donor jet is rotated about the beam axis by phi_target - phi_donor (exact symmetry of L-GATr: the beam and time
           spurions are invariant), so the donor's head outputs point along the target jet's direction.
  mean     P55's secondary: each ablated head's output replaced by its mean over a reference set (global / beam / time / particles).
Per arm and task (W vs Z, t->bqq' vs QCD): unedited, all heads ablated (floor), greedy top-k circuits (k = 4, 8, 16, 24, 32),
20 block-matched and 10 unrestricted random k-sets per k; for pt and aligned also every head removed alone. 10-class accuracy of
each circuit on all 10 classes.

Sample: JetClass test_20M <class>_114.root (not used by P55, not in L-GATr training/validation), default_rng(5599), 500 random
rows per class (N 5,000; 1,000 task jets per task) + 100 other random rows per class for the mean-ablation reference.

  ./sync.sh && ./part_job.sh c-p55-verify '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p55v_verify'
-> $AI_RESULTS/part/p55_verify.json
"""
import sys, json, os, time
from pathlib import Path
import numpy as np

SMOKE = '--smoke' in sys.argv
sys.argv = [sys.argv[0]]                       # P55 parses argv at import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p55_lgatr_head_circuit as P     # noqa: E402

FILE, SEED = 114, 5599
PER = 20 if SMOKE else 500
PER_REF = 10 if SMOKE else 100
N_BM, N_UN = (2, 2) if SMOKE else (20, 10)
KS = P.KS; NHEAD, NH, NBLK = P.NHEAD, P.NH, P.NBLK
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part' / ('p55_verify_smoke.json' if SMOKE else 'p55_verify.json')
RANKF = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr' / 'p55_ranking.json'
log = P.log


def load():
    from interp.part_core import sub, cat, kin, build
    rng = np.random.default_rng(SEED); E, F, rows = [], [], []
    for c in P.CL:
        N = P.nent(c, FILE); perm = rng.permutation(N); re, rf = perm[:PER], perm[PER:PER + PER_REF]
        E.append(P.read_rows(c, FILE, re)); F.append(P.read_rows(c, FILE, rf)); rows.append(np.sort(re))
    Je, Jf = cat(E), cat(F); p = rng.permutation(len(Je['label'])); Je = sub(Je, p)
    b = lambda J: build(J, *kin(J))
    return b(Je), b(Jf)


def lab_axis(J):
    m = J['mask'] > 0; px, py, pz = (np.where(m, J[k], 0).sum(1) for k in ('px', 'py', 'pz'))
    pt = np.hypot(px, py); return np.log(np.maximum(pt, 1e-3)), np.arcsinh(pz / np.maximum(pt, 1e-3)), np.arctan2(py, px)


def aligned_donors(J, seed):
    lpt, eta, phi = lab_axis(J); z = np.stack([(lpt - lpt.mean()) / lpt.std(), (eta - eta.mean()) / eta.std()], 1)
    rng = np.random.default_rng(seed); out = np.empty(len(z), int)
    for s in range(0, len(z), 1000):
        d = ((z[s:s + 1000, None] - z[None]) ** 2).sum(-1); d[np.arange(len(d)), s + np.arange(len(d))] = np.inf
        nn = np.argsort(d, 1)[:, :P.K_DONOR]; out[s:s + len(d)] = nn[np.arange(len(d)), rng.integers(0, P.K_DONOR, len(d))]
    return out


def rotated_pool(J, don):
    """row i = donor don[i] rotated about the beam axis so its lab-frame phi equals target i's"""
    from interp.part_core import sub
    _, _, phi = lab_axis(J); a = (phi - phi[don])[:, None]; D = sub(J, don); c, s = np.cos(a), np.sin(a)
    px, py = D['px'].copy(), D['py'].copy(); D = dict(D)
    D['px'] = (c * px - s * py).astype(np.float32); D['py'] = (s * px + c * py).astype(np.float32)
    return D


def main():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; T0 = time.time()
    rk = json.loads(RANKF.read_text())['rankings']; name2i = {P.hn(i): i for i in range(NHEAD)}
    order = {t: [name2i[h] for h in rk[t]['greedy']] for t in P.TASKS}
    Je, Jf = load(); ye = Je['label'].astype(int); N = len(ye)
    lpt, eta, phi = lab_axis(Je)
    don_pt = P.pick_donors(Je['jet_pt'], Je['jet_pt'], True, 5591); don_al = aligned_donors(Je, 5592); Jrot = rotated_pool(Je, don_al)
    _, _, phr = lab_axis(Jrot)
    dphi = lambda a, b: np.abs(np.angle(np.exp(1j * (a - b))))
    dpt = dict(med_abs_dlogpt=float(np.median(np.abs(lpt - lpt[don_pt]))), med_abs_deta=float(np.median(np.abs(eta - eta[don_pt]))),
               med_abs_dphi=float(np.median(dphi(phi, phi[don_pt]))), same_class=float((ye[don_pt] == ye).mean()))
    dal = dict(med_abs_dlogpt=float(np.median(np.abs(lpt - lpt[don_al]))), med_abs_deta=float(np.median(np.abs(eta - eta[don_al]))),
               med_abs_dphi_before_rotation=float(np.median(dphi(phi, phi[don_al]))), max_abs_dphi_after_rotation=float(dphi(phi, phr).max()),
               same_class=float((ye[don_al] == ye).mean()))
    out = dict(node='P55v', sample=f'test_20M <class>_{FILE}.root, default_rng({SEED}), {PER} random rows per class (N {N}); '
                                  f'reference {PER_REF} other rows per class', ranking_file=str(RANKF), donors=dict(pt=dpt, aligned=dal))
    log('donors', out['donors'])
    lg = P.LGA(P.ARGS.ckpt, dev); log('loaded L-GATr')
    ar = np.arange(N); tix = {t: np.where(P.task_mask(ye, t))[0] for t in P.TASKS}
    B = dict(pt=P.make_batches(lg, Je, ar, don_pt, Je), aligned=P.make_batches(lg, Je, ar, ar, Jrot))
    for t in P.TASKS:
        B[f'pt_{t}'] = P.make_batches(lg, Je, tix[t], don_pt, Je); B[f'aligned_{t}'] = P.make_batches(lg, Je, tix[t], ar, Jrot)
    B['mean'] = B['pt']
    for t in P.TASKS: B[f'mean_{t}'] = B[f'pt_{t}']
    # symmetry check: rigid rotation of the target jets by random angles
    from interp.part_core import sub
    rr = np.random.default_rng(1).uniform(-np.pi, np.pi, N)[:, None]; JR = dict(Je); c, s = np.cos(rr), np.sin(rr)
    JR['px'] = (c * Je['px'] - s * Je['py']).astype(np.float32); JR['py'] = (s * Je['px'] + c * Je['py']).astype(np.float32)
    L0 = P.plain_logits(lg, B['pt'], N); bR = P.make_batches(lg, JR, ar[:512], ar, Je); LR = P.plain_logits(lg, bR, N)
    ok = ~np.isnan(LR[:, 0]); out['rotation_symmetry'] = dict(N=int(ok.sum()), max_abs_dlogit=float(np.abs(LR[ok] - L0[ok]).max()),
                                                             same_call=float((LR[ok].argmax(1) == L0[ok].argmax(1)).mean()))
    out['unedited'] = dict(acc10=P.boot_ci(L0.argmax(1) == ye), **{f'bin_{t}': P.boot_ci(P.bin_correct(L0, ye, t)) for t in P.TASKS})
    log('rotation check', out['rotation_symmetry'], 'unedited', out['unedited'])
    # mean-ablation reference
    Sm = dict(Bf=P.make_batches(lg, Jf, np.arange(len(Jf['label'])), np.arange(len(Jf['label'])), Jf), yf=Jf['label']); P.sec_means(lg, Sm, {})
    rng = np.random.default_rng(5593); res = {}
    for arm in ('pt', 'aligned', 'mean'):
        mode = 'mean' if arm == 'mean' else 'resample'; ra = {}
        # 10-class with each circuit
        cf = [np.ones((NHEAD, 1)), np.zeros((NHEAD, 1))] + [P.keep_cfg(order[t][:k]) for t in P.TASKS for k in KS]
        L = P.eval_configs(lg, B[arm], np.stack(cf), N, mode=mode)
        ra['acc10'] = dict(full=P.boot_ci(L[0].argmax(1) == ye), floor=P.boot_ci(L[1].argmax(1) == ye))
        j = 2
        for t in P.TASKS:
            for k in KS: ra['acc10'][f'{t}|k{k}'] = P.boot_ci(L[j].argmax(1) == ye); j += 1
        ra['floor_confusion'] = P.confusion(ye, L[1].argmax(1)).tolist()
        for t in P.TASKS:
            sets = [order[t][:k] for k in KS]; bm = {k: P.bm_random(order[t][:k], N_BM, rng) for k in KS}
            un = {k: [rng.choice(NHEAD, k, replace=False) for _ in range(N_UN)] for k in KS}
            cf = [np.ones((NHEAD, 1)), np.zeros((NHEAD, 1))] + [P.keep_cfg(s_) for s_ in sets]
            for k in KS: cf += [P.keep_cfg(h) for h in bm[k]] + [P.keep_cfg(h) for h in un[k]]
            nrem0 = len(cf)
            if arm != 'mean':
                for h in range(NHEAD): w = np.ones((NHEAD, 1)); w[h] = 0; cf.append(w)
            L = P.eval_configs(lg, B[f'{arm}_{t}'], np.stack(cf), N, mode=mode); C = P.bin_correct(L, ye, t); acc = C.mean(-1)
            F = (acc - acc[1]) / (acc[0] - acc[1]); r = dict(acc_unedited=float(acc[0]), acc_floor=float(acc[1]), circuits={})
            j = 2 + len(KS); ps = []
            for i, k in enumerate(KS):
                ib = slice(j, j + N_BM); iu = slice(j + N_BM, j + N_BM + N_UN); j += N_BM + N_UN
                p = float((1 + (acc[ib] >= acc[2 + i]).sum()) / (1 + N_BM)); ps.append(p)
                r['circuits'][f'k{k}'] = dict(heads=[P.hn(h) for h in sets[i]], bin_acc=P.boot_ci(C[2 + i]), F=P.faith(C[2 + i], C[1], C[0]),
                                              F_bm_random=[float(F[ib].mean()), float(np.percentile(F[ib], 2.5)), float(np.percentile(F[ib], 97.5)), float(F[ib].max())],
                                              F_unrestricted=[float(F[iu].mean()), float(F[iu].max())], p_rank_vs_bm=p)
                arr_key = f'_C_{arm}_{t}_k{k}'; CTXC[arr_key] = C[2 + i]
            for i, k in enumerate(KS): r['circuits'][f'k{k}']['p_rank_vs_bm_holm'] = float(P.holm(ps)[i])
            CTXC[f'_C_{arm}_{t}_full'], CTXC[f'_C_{arm}_{t}_floor'] = C[0], C[1]
            if arm != 'mean':
                Cr = C[nrem0:]; dr = Cr.mean(-1) - acc[0]
                pm = P.holm([P.mcnemar(C[0], Cr[h])['p'] for h in range(NHEAD)])
                bi = np.random.default_rng(3).integers(0, Cr.shape[1], (P.NB, Cr.shape[1]))
                mdrop = [(Cr[:, b_].mean(-1) - C[0][b_].mean()).mean() for b_ in bi]
                r['removed_alone'] = dict(dacc={P.hn(h): float(dr[h]) for h in range(NHEAD)},
                                          n_holm_sig=int((pm < 0.05).sum()), n_drop_gt_10pts=int((dr < -0.10).sum()),
                                          n_at_floor=int((Cr.mean(-1) < acc[1] + 0.03).sum()),
                                          mean_drop_over_heads=[float(dr.mean()), float(np.percentile(mdrop, 2.5)), float(np.percentile(mdrop, 97.5))],
                                          mean_drop_blocks=[float(dr[b * NH:(b + 1) * NH].mean()) for b in range(NBLK)])
                CTXC[f'_R_{arm}_{t}'] = Cr
            ra[t] = r
            log(f"{arm} {t}: unedited {acc[0]:.4f} floor {acc[1]:.4f}; F " + ' '.join(f"k{k} {r['circuits'][f'k{k}']['F'][0]:.3f} (bm {r['circuits'][f'k{k}']['F_bm_random'][0]:.3f}, p {r['circuits'][f'k{k}']['p_rank_vs_bm']:.2f})" for k in KS)
                + (f"; removed-alone mean drop {r['removed_alone']['mean_drop_over_heads'][0]:.3f}, n<-10pts {r['removed_alone']['n_drop_gt_10pts']}, at floor {r['removed_alone']['n_at_floor']}" if 'removed_alone' in r else ''))
        res[arm] = ra; out['arms'] = res; out['wall_s'] = time.time() - T0; P.jdump(out, OUT)
    # paired differences aligned - pt (same jets): F and removed-alone mean drop
    dif = {}
    for t in P.TASKS:
        for k in KS:
            a, b = CTXC[f'_C_aligned_{t}_k{k}'], CTXC[f'_C_pt_{t}_k{k}']
            fa = lambda ix, arm, kk: ((CTXC[f'_C_{arm}_{t}_k{kk}'][ix].mean(-1) - CTXC[f'_C_{arm}_{t}_floor'][ix].mean(-1))
                                      / (CTXC[f'_C_{arm}_{t}_full'][ix].mean(-1) - CTXC[f'_C_{arm}_{t}_floor'][ix].mean(-1)))
            bi = np.random.default_rng(4).integers(0, len(a), (P.NB, len(a)))
            d0 = float(fa(np.arange(len(a)), 'aligned', k) - fa(np.arange(len(a)), 'pt', k)); db = fa(bi, 'aligned', k) - fa(bi, 'pt', k)
            dif[f'{t}|k{k}'] = dict(dF=[d0, float(np.nanpercentile(db, 2.5)), float(np.nanpercentile(db, 97.5))],
                                   p_boot_two_sided=float(min(1, 2 * min((db <= 0).mean(), (db >= 0).mean()))))
            dif[f'{t}|k{k}']['mcnemar_circuit_calls'] = P.mcnemar(b, a)
        Ra, Rp = CTXC[f'_R_aligned_{t}'], CTXC[f'_R_pt_{t}']; bi = np.random.default_rng(5).integers(0, Ra.shape[1], (P.NB, Ra.shape[1]))
        dd = [Ra[:, b_].mean() - Rp[:, b_].mean() for b_ in bi]
        dif[f'{t}|removed_alone_mean_acc'] = [float(Ra.mean() - Rp.mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
    ph = P.holm([dif[f'{t}|k{k}']['p_boot_two_sided'] for t in P.TASKS for k in KS])
    for i, kk in enumerate([f'{t}|k{k}' for t in P.TASKS for k in KS]): dif[kk]['p_holm'] = float(ph[i])
    out['aligned_minus_pt'] = dif; out['wall_s'] = time.time() - T0; P.jdump(out, OUT); log('done', dif)


CTXC = {}
if __name__ == '__main__':
    main()
