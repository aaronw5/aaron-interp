"""P40v — verification of P40 (class-token MLP neuron atlas of ParT) on a fresh sample, with the missing controls.

Fresh sample: JetClass test_20M <sample>_111.root (P40 used _110), 2,000 random rows per class (default_rng(4141), never
  first-k), random labelling / held-out halves of 10,000 jets each, every jet rebuilt through part_core.build (sign-aware eta).
Re-used code: interp.p40_neuron_atlas (Net hooks on cls_blocks[b].fc1 / .act, Scorer, label_units, clean_stats, haar, Boot).
Checks per model (ParT_full, ParT_kin):
 (1) atlas replication: untrained-net threshold (seed 0), clean / selective fractions for neurons vs 10 Haar rotations
     (raw, standardised); added unthresholded metrics: median family gap (best minus second-best label family, held-out |S|)
     over all 1,024 post-GELU units, and the selective share among clean units; label stability of P40's 300 steered units.
 (2) steering, fresh random 300 clean post-GELU units, +-2 s.d.: neuron, the same-index rotated direction Q0[:, i] at its own
     projection s.d. (P40 control) and at the neuron's s.d. ('equal step': same L2 size of the edit, the missing control).
 (3) pre-registered subset: P40's units with |Delta l| >= 0.05 at +2 s.d. (24 full, 36 kin), with P40's target class and sign,
     re-steered on the fresh held-out half; equal-step rotated control for the same indices. Also the 20 most readable units
     of P40 (top_units, e.g. the lepton-class units b2n153, b2n249, b1n138 of ParT_full), which P40 did not steer.
 (4) calibration (all particle angles x 1.05): joint patch of P40's 18 mass units (fixed list) and of the fresh mass-labelled
     units; controls: 20 random neuron sets with the same per-block counts, 20 random sets of clean non-mass units, 10 random
     rotated direction sets, and an oracle set (the 18 units whose activation shifts most under the dilation, chosen on the
     labelling half).
 (5) rigid phi shift (all particles dphi + 0.06 rad, invariant mass unchanged): true-W -> Z rate before/after; share of the
     W log-odds change on true-W jets carried by P40's mass units, b1n92 + b1n108 (full) / b1n8 (kin), vs random neuron sets.
  ./sync.sh && ./part_job.sh c-p40-verify 'python -m interp.p40v_verify --smoke && python -m interp.p40v_verify'
  -> $AI_RESULTS/part/p40_verify[_smoke].json
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p40_neuron_atlas as A      # noqa: E402
P = A.P
SMOKE = P.SMOKE
SEED = 4141
A.FILE = '111'
A.NPC = 100 if SMOKE else 2000
A.NROT = 2 if SMOKE else 10
NB = 100 if SMOKE else 1000
A.NB = NB
NSTEER = 6 if SMOKE else 300
NRSET = 3 if SMOKE else 20
NRCAL = 2 if SMOKE else 10
DPHI = 0.06
FN = P.OUTD / f'p40_verify{P.TAG}.json'
P40 = P.OUTD / 'p40_neuron_atlas.json'
log = P.log
W_, Z_ = 7, 6


def ul(name):  # 'b1n92' -> (1, 92)
    b, i = name[1:].split('n'); return int(b), int(i)


def main():
    import torch
    from scipy.stats import binomtest, fisher_exact, wilcoxon
    from interp.part_core import CL, sample, kin, build, jmass
    A.CLN = list(CL); CLN = A.CLN
    t = torch; rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    R0 = json.loads(P40.read_text())
    R = dict(node='P40v', seed=SEED, smoke=SMOKE, file='test_20M <sample>_111.root', NPC=A.NPC, NB=NB, NSTEER=NSTEER, DPHI=DPHI)
    JL, TL, JH, TH = A.load(rng); yL, yH = JL['label'].astype(int), JH['label'].astype(int)
    OL, _ = A.observables(JL, TL, dev); OH, _ = A.observables(JH, TH, dev)
    scL, scH = A.Scorer(OL, yL), A.Scorer(OH, yH)
    R['sample'] = dict(n_label=len(yL), n_heldout=len(yH), per_class_heldout=np.bincount(yH, minlength=10).tolist())
    J1 = P.rebuild(sample(per_class=1000)); y1 = J1['label'].astype(int)
    edits = {}
    for nm, J in (('H', JH), ('L', JL)):
        pt, de, dp, ms = kin(J)
        edits['dil' + nm] = build(J, pt, de * A.DIL, dp * A.DIL, ms)
        if nm == 'H': edits['phiH'] = build(J, pt, de, dp + DPHI, ms)
    R['edits'] = dict(dil_dlogm_median=float(np.median(np.log(jmass(edits['dilH']) + 1e-3) - np.log(jmass(JH) + 1e-3))),
                      phi_dlogm_maxabs=float(np.abs(np.log(jmass(edits['phiH']) + 1e-3) - np.log(jmass(JH) + 1e-3)).max()))
    log('edits', R['edits'])
    out = {}
    for feat in ('full', 'kin'):
        t0 = time.time(); res = {}; p40m = R0['models'][f'ParT_{feat}']
        # ---------------- threshold from the untrained net (as P40)
        M = A.Net(feat, 'untrained', 0); _, ALu, _ = M.run(JL); _, AHu, _ = M.run(JH)
        Uu = A.atlas_model(ALu, AHu, scL, scH, rot=False)['neuron']
        thr = {st: float(np.quantile(np.concatenate([Uu[f'b{b}_{st}']['v'] for b in (1, 2)]), 0.95)) for st in ('pre', 'post')}
        res['thresholds'] = thr; res['thresholds_P40'] = R0['thresholds'][feat]
        del M, ALu, AHu; torch.cuda.empty_cache()
        # ---------------- trained
        M = A.Net(feat); L1, _, _ = M.run(J1); res['acc_P1_sample'] = float((L1.argmax(1) == y1).mean())
        LL, AL, _ = M.run(JL); LH, AH, C = M.run(JH, cache=True); N = len(yH)
        res['acc_heldout'] = float((LH.argmax(1) == yH).mean())
        L0 = M.tail(C, N); res['tail_max_abs_diff'] = float(np.abs(L0.cpu().numpy() - LH).max())
        log(f'==== ParT_{feat}: acc P1 {res["acc_P1_sample"]:.4f} held-out {res["acc_heldout"]:.4f} tail {res["tail_max_abs_diff"]:.1e} thr {thr}')
        # (1) atlas
        At = A.atlas_model(AL, AH, scL, scH, rot=True); U = At['neuron']
        cs = {k: A.clean_stats(V, thr) for k, V in At.items()}
        atl = {}
        for st in ('post', 'pre'):
            def met(k):
                s = cs[k][st]['thr_untrained']; gap = np.concatenate([At[k][f'b{b}_{st}']['top1'] - At[k][f'b{b}_{st}']['top2'] for b in (1, 2)])
                return dict(clean=s['frac'], selective=s['frac_selective'], sel_among_clean=s['frac_selective'] / max(s['frac'], 1e-9),
                            gap_median=float(np.median(gap)), gap_mean=float(gap.mean()), v_median=cs[k][st]['v_median'])
            d = dict(neuron=met('neuron'))
            for kind in ('raw', 'std'):
                ms_ = [met(f'rot_{kind}{s}') for s in range(A.NROT)]
                d[f'rot_{kind}'] = {m: dict(mean=float(np.mean([x[m] for x in ms_])), min=float(np.min([x[m] for x in ms_])), max=float(np.max([x[m] for x in ms_])),
                                            p_ge=float((1 + sum(x[m] >= d['neuron'][m] for x in ms_)) / (1 + len(ms_)))) for m in ms_[0]}
            d['neuron']['ci_selective'] = cs['neuron'][st]['thr_untrained']['ci_selective']
            d['neuron']['label_counts'] = cs['neuron'][st]['thr_untrained']['label_counts']
            atl[st] = d
            log(f'  atlas {st}: neuron clean {d["neuron"]["clean"]:.3f} sel {d["neuron"]["selective"]:.3f} sel|clean {d["neuron"]["sel_among_clean"]:.3f} '
                f'gap {d["neuron"]["gap_median"]:.3f} | raw sel {d["rot_raw"]["selective"]["mean"]:.3f} [{d["rot_raw"]["selective"]["min"]:.3f},{d["rot_raw"]["selective"]["max"]:.3f}] '
                f'sel|clean {d["rot_raw"]["sel_among_clean"]["mean"]:.3f} gap {d["rot_raw"]["gap_median"]["mean"]:.3f} | std sel {d["rot_std"]["selective"]["mean"]:.3f} '
                f'[{d["rot_std"]["selective"]["min"]:.3f},{d["rot_std"]["selective"]["max"]:.3f}] sel|clean {d["rot_std"]["sel_among_clean"]["mean"]:.3f} gap {d["rot_std"]["gap_median"]["mean"]:.3f}')
        res['atlas'] = atl
        labn = lambda b, i: A.labname(int(U[f'b{b}_post']['lab'][i]))
        agree = [labn(*ul(r['unit'])) == r['label'] for r in p40m['write_units']]
        res['label_stability_P40_steered'] = dict(n=len(agree), frac_same_label=float(np.mean(agree)))
        log(f'  label stability (P40 steered units): {np.mean(agree):.3f}')
        # ---------------- steering helpers
        l0 = A.lodds_all(L0); B = A.Boot(N, NB, rng, dev, yH)
        sd = {b: AL[('post', b - 1)].std(0) for b in (1, 2)}; mu = {b: AL[('post', b - 1)].mean(0) for b in (1, 2)}
        Q0 = A.haar(512, 0); Qt = t.from_numpy(Q0.astype(np.float32)).to(dev)

        def steer(b, vec, step, c):
            o = {}
            for k in (2, -2):
                Lx = M.tail(C, N, lambda it: {b - 1: [('add', (k * step) * vec)]}); dc = (A.lodds_all(Lx) - l0)[:, c]
                o[str(k)] = B.ms(dc) if k == 2 else dict(v=float(dc.mean()))
            return o

        def ok(st, s):
            e = st['2']; lo, hi = e['ci']
            return bool((s * e['v'] > 0) and ((lo > 0) if s > 0 else (hi < 0)) and (s * st['-2']['v'] < 0))

        def unit_rows(units):   # units: list of (b, i, c, s)
            rows = []
            for b, i, c, s in units:
                e = t.zeros(512, device=dev); e[i] = 1; q = Qt[:, i]; pa = AL[('post', b - 1)] @ Q0[:, i]
                r = dict(unit=f'b{b}n{i}', label=labn(b, i), target=CLN[c], c=c, s=s, sd=float(sd[b][i]), rot_sd=float(pa.std()),
                         neuron=steer(b, e, float(sd[b][i]), c), rot_own=steer(b, q, float(pa.std()), c), rot_eq=steer(b, q, float(sd[b][i]), c))
                for k in ('neuron', 'rot_own', 'rot_eq'): r[k]['ok'] = ok(r[k], s); r[k]['sdl'] = s * r[k]['2']['v']
                rows.append(r)
            return rows

        def summ(rows, tag):
            o = dict(n=len(rows))
            for k in ('neuron', 'rot_own', 'rot_eq'):
                okk = np.array([r[k]['ok'] for r in rows]); sg = np.array([r[k]['sdl'] for r in rows])
                o[k] = dict(frac_ok=float(okk.mean()) if len(okk) else None, n_ok=int(okk.sum()), ci=A.wilson(int(okk.sum()), len(okk)) if len(okk) else None,
                            frac_sign_ok=float((sg > 0).mean()) if len(sg) else None, abs_dl_median=float(np.median(np.abs(sg))) if len(sg) else None,
                            binom_p_vs_half=float(binomtest(int(okk.sum()), len(okk), 0.5).pvalue) if len(okk) else None)
            if rows:
                for k in ('rot_own', 'rot_eq'):
                    a = np.array([r['neuron']['sdl'] for r in rows]); b_ = np.array([r[k]['sdl'] for r in rows])
                    tab = [[o['neuron']['n_ok'], len(rows) - o['neuron']['n_ok']], [o[k]['n_ok'], len(rows) - o[k]['n_ok']]]
                    o[f'neuron_vs_{k}'] = dict(wilcoxon_p=float(wilcoxon(a, b_).pvalue) if len(a) > 1 and np.any(a != b_) else None,
                                               fisher_p=float(fisher_exact(tab)[1]))
            log(f'  {tag}: n {len(rows)}; ok neuron {o["neuron"]["frac_ok"]} rot_own {o["rot_own"]["frac_ok"]} rot_eq {o["rot_eq"]["frac_ok"]}; '
                f'|dl| med neuron {o["neuron"]["abs_dl_median"]} rot_own {o["rot_own"]["abs_dl_median"]} rot_eq {o["rot_eq"]["abs_dl_median"]}')
            return o
        # (2) fresh random clean units
        clean = [(b, i) for b in (1, 2) for i in range(512) if U[f'b{b}_post']['v'][i] >= thr['post'] and U[f'b{b}_post']['q'][i] < 0.05]
        pick = np.sort(rng.choice(len(clean), min(NSTEER, len(clean)), replace=False))
        units = [(b, i, *A.target_of(U, OL, yL, b, i)) for b, i in (clean[k] for k in pick)]
        rows = unit_rows(units); res['steer_random'] = summ(rows, 'random clean units')
        for thr_dl in (0.05,):
            for k in ('neuron', 'rot_own', 'rot_eq'):
                sub_ = [r for r in rows if abs(r[k]['2']['v']) >= thr_dl]
                okk = [r[k]['sdl'] > 0 for r in sub_]
                res['steer_random'][f'{k}_visible_subset'] = dict(n=len(sub_), n_sign_ok=int(sum(okk)),
                                                                  binom_p=float(binomtest(int(sum(okk)), len(okk), 0.5).pvalue) if okk else None)
            log('  visible subset (fresh, rule |dl|>=0.05):', {k: res['steer_random'][f'{k}_visible_subset'] for k in ('neuron', 'rot_own', 'rot_eq')})
        res['steer_random_rows'] = rows
        # (3) pre-registered P40 subset and P40's top-20 readable units
        pre = [r for r in p40m['write_units'] if abs(r['steer']['2']['dl']) >= 0.05]
        rows = unit_rows([(*ul(r['unit']), int(r['c']), float(r['s'])) for r in pre]); res['steer_preregistered'] = summ(rows, 'P40 visible subset (pre-registered)')
        res['steer_preregistered']['frac_sign_ok_neuron'] = float(np.mean([r['neuron']['sdl'] > 0 for r in rows]))
        res['steer_preregistered_rows'] = rows
        tu = []
        for u in p40m['top_units'][:20]:
            b, i = ul(u['unit'])
            lab = u['label']
            if lab.startswith('cls_'): c, s = CLN.index(lab[4:]), float(u['sign'])
            else: c, s = A.target_of(U, OL, yL, b, i)
            tu.append((b, i, c, s))
        rows = unit_rows(tu); res['steer_top20_readable'] = summ(rows, 'P40 top-20 readable units')
        # mean ablation of the top-20 readable units, one by one: class recall change
        c0 = L0.argmax(1); yt = t.from_numpy(yH).to(dev); abl = []
        for (b, i, c, s), r in zip(tu, rows):
            m = t.zeros(512, dtype=t.bool, device=dev); m[i] = True; mu_t = t.from_numpy(mu[b]).to(dev)
            Lx = M.tail(C, N, lambda it: {b - 1: [('mean', m, mu_t)]}); c1 = Lx.argmax(1); cl = yt == c
            dc = (A.lodds_all(Lx) - l0)[:, c][cl]
            r['ablate'] = dict(recall_before=float((c0[cl] == c).float().mean()), recall_after=float((c1[cl] == c).float().mean()),
                               dl_class=B.ms(dc, c), mcnemar=P.mcnemar((c0[cl] == c).cpu().numpy(), (c1[cl] == c).cpu().numpy()))
            abl.append(r['ablate']['mcnemar']['p'])
        for r, q in zip(rows, A.bh(abl)): r['ablate']['bh_q'] = float(q)
        res['steer_top20_rows'] = rows
        for r in rows[:8]: log(f'    {r["unit"]} {r["label"]}->{r["target"]} s {r["s"]:+.0f}: +2sd {r["neuron"]["2"]["v"]:+.4f} {np.round(r["neuron"]["2"]["ci"], 4).tolist()} '
                               f'-2sd {r["neuron"]["-2"]["v"]:+.4f}; rot_eq {r["rot_eq"]["2"]["v"]:+.4f}; ablate recall {r["ablate"]["recall_before"]:.3f}->{r["ablate"]["recall_after"]:.3f} '
                               f'dl {r["ablate"]["dl_class"]["v"]:+.4f}')
        # (4) calibration
        LD, AD, _ = M.run(edits['dilH']); _, ADL, _ = M.run(edits['dilL'])
        ld = A.lodds_all(t.from_numpy(LD).to(dev)); dD = ld - l0
        ADt = {b: t.from_numpy(AD[('post', b - 1)]).to(dev) for b in (1, 2)}

        def patch_set(units, src, cols=None):     # units [(b, i)] or cols {b: (512, k) orthonormal tensor}
            if cols is None:
                cols = {}
                for b in (1, 2):
                    ix = [i for bb, i in units if bb == b]
                    if ix:
                        Ub = t.zeros(512, len(ix), device=dev)
                        for k, i in enumerate(ix): Ub[i, k] = 1
                        cols[b] = Ub
            Lx = M.tail(C, N, lambda it: {b - 1: [('patch', cols[b], src[b][it] @ cols[b])] for b in cols}); return A.lodds_all(Lx) - l0

        cal = dict(dil_dl_W=B.ms(dD[:, W_]), dil_calls_changed=float((ld.argmax(1) != l0.argmax(1)).float().mean()))
        p40mass = [ul(u['unit']) for u in p40m['calibration']['units']]
        fresh_mass = [(b, i) for b, i in clean if labn(b, i) in A.MASSLAB]
        shiftL = {b: (ADL[('post', b - 1)] - AL[('post', b - 1)]).mean(0) / np.maximum(sd[b], 1e-6) for b in (1, 2)}
        allu = sorted([(abs(float(shiftL[b][i])), b, i) for b in (1, 2) for i in range(512)], reverse=True)
        oracle = [(b, i) for _, b, i in allu[:len(p40mass)]]
        sets = dict(p40_mass_units=p40mass, fresh_mass_units=fresh_mass, oracle_top_shift=oracle)
        if feat == 'full': sets.update(b1n92=[(1, 92)], b1n108=[(1, 108)], b1n92_b1n108=[(1, 92), (1, 108)])
        else: sets.update(b1n8=[(1, 8)])
        for k, s_ in sets.items():
            if not s_: cal[k] = None; continue
            dP = patch_set(s_, ADt); cal[k] = dict(units=[f'b{b}n{i}' for b, i in s_], R_W=B.ratio(dP[:, W_], dD[:, W_]))
        nb = {b: sum(1 for bb, _ in p40mass if bb == b) for b in (1, 2)}
        nonmass = [(b, i) for b, i in clean if labn(b, i) not in A.MASSLAB]
        rn, rc, rr = [], [], []
        for _ in range(NRSET):
            s_ = [(b, int(i)) for b in (1, 2) for i in rng.choice(512, nb[b], replace=False)]
            dP = patch_set(s_, ADt); rn.append(float(dP[:, W_].mean() / dD[:, W_].mean()))
            pool = {b: [i for bb, i in nonmass if bb == b] for b in (1, 2)}
            s_ = [(b, int(i)) for b in (1, 2) for i in (rng.choice(pool[b], min(nb[b], len(pool[b])), replace=False) if nb[b] else [])]
            dP = patch_set(s_, ADt); rc.append(float(dP[:, W_].mean() / dD[:, W_].mean()))
        for r_ in range(NRCAL):
            Qr = A.haar(512, 100 + r_)
            cols = {b: t.from_numpy(Qr[:, rng.choice(512, nb[b], replace=False)].astype(np.float32)).to(dev) for b in (1, 2) if nb[b]}
            dP = patch_set(None, ADt, cols); rr.append(float(dP[:, W_].mean() / dD[:, W_].mean()))
        Rp = cal['p40_mass_units']['R_W']['v']
        cal['random_neuron_sets'] = dict(R=rn, mean=float(np.mean(rn)), max=float(np.max(rn)), p_ge=float((1 + sum(x >= Rp for x in rn)) / (1 + len(rn))))
        cal['random_clean_nonmass_sets'] = dict(R=rc, mean=float(np.mean(rc)), max=float(np.max(rc)), p_ge=float((1 + sum(x >= Rp for x in rc)) / (1 + len(rc))))
        cal['random_rotated_sets'] = dict(R=rr, mean=float(np.mean(rr)), max=float(np.max(rr)))
        cal['overlap_p40_oracle'] = len(set(p40mass) & set(oracle)); cal['overlap_p40_fresh'] = len(set(p40mass) & set(fresh_mass))
        res['calibration'] = cal
        log(f'  calibration: dil dl_W {cal["dil_dl_W"]["v"]:+.3f}; R_W p40 mass {Rp:+.3f} {np.round(cal["p40_mass_units"]["R_W"]["ci"], 3).tolist()} '
            f'fresh ({len(fresh_mass)}) {cal["fresh_mass_units"]["R_W"]["v"] if cal["fresh_mass_units"] else None} oracle {cal["oracle_top_shift"]["R_W"]["v"]:+.3f} '
            f'| random neurons {cal["random_neuron_sets"]["mean"]:+.3f} (max {cal["random_neuron_sets"]["max"]:+.3f}) nonmass clean {cal["random_clean_nonmass_sets"]["mean"]:+.3f} '
            f'(max {cal["random_clean_nonmass_sets"]["max"]:+.3f}) rotated {cal["random_rotated_sets"]["mean"]:+.3f}; overlap oracle {cal["overlap_p40_oracle"]} fresh {cal["overlap_p40_fresh"]}')
        for k in sets:
            if k.startswith('b1n') and cal[k]: log(f'    {k}: R_W {cal[k]["R_W"]["v"]:+.3f} {np.round(cal[k]["R_W"]["ci"], 3).tolist()}')
        del AD, ADL, ADt; torch.cuda.empty_cache()
        # (5) phi shift on true-W jets
        LS, AS, _ = M.run(edits['phiH']); ls = A.lodds_all(t.from_numpy(LS).to(dev)); dS = ls - l0
        ASt = {b: t.from_numpy(AS[('post', b - 1)]).to(dev) for b in (1, 2)}; wm = yt == W_
        Wc = t.from_numpy(P.boot_W(int(wm.sum()), NB, rng)).to(dev)

        def ratio_w(x, y):
            x, y = x[wm].float(), y[wm].float(); r = ((Wc @ x) / (Wc @ y)).cpu().numpy()
            return dict(v=float(x.mean() / y.mean()), ci=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))])
        ph = dict(n_trueW=int(wm.sum()), WtoZ_before=float((l0.argmax(1)[wm] == Z_).float().mean()), WtoZ_after=float((ls.argmax(1)[wm] == Z_).float().mean()),
                  dl_W_trueW=float(dS[wm, W_].mean()), mcnemar_Z=P.mcnemar((l0.argmax(1)[wm] == Z_).cpu().numpy(), (ls.argmax(1)[wm] == Z_).cpu().numpy()))
        psets = dict(p40_mass_units=p40mass, oracle_top_shift_dilation=oracle)
        if feat == 'full': psets['b1n92_b1n108'] = [(1, 92), (1, 108)]
        else: psets['b1n8'] = [(1, 8)]
        for k, s_ in psets.items():
            dP = patch_set(s_, ASt); ph[k] = dict(units=[f'b{b}n{i}' for b, i in s_], R_W_trueW=ratio_w(dP[:, W_], dS[:, W_]))
        rn = []
        for _ in range(NRSET):
            s_ = [(b, int(i)) for b in (1, 2) for i in rng.choice(512, nb[b], replace=False)]
            dP = patch_set(s_, ASt); rn.append(float(dP[wm, W_].mean() / dS[wm, W_].mean()))
        ph['random_neuron_sets'] = dict(R=rn, mean=float(np.mean(rn)), max=float(np.max(rn)),
                                        p_ge=float((1 + sum(x >= ph['p40_mass_units']['R_W_trueW']['v'] for x in rn)) / (1 + len(rn))))
        res['phi_shift'] = ph
        log(f'  phi + {DPHI}: true W -> Z {ph["WtoZ_before"]:.3f} -> {ph["WtoZ_after"]:.3f}; dl_W {ph["dl_W_trueW"]:+.3f}; R_W p40 mass {ph["p40_mass_units"]["R_W_trueW"]["v"]:+.3f} '
            f'{np.round(ph["p40_mass_units"]["R_W_trueW"]["ci"], 3).tolist()}; random neurons {ph["random_neuron_sets"]["mean"]:+.3f} (max {ph["random_neuron_sets"]["max"]:+.3f})')
        res['runtime_s'] = time.time() - t0; out[f'ParT_{feat}'] = res; R['models'] = out
        FN.parent.mkdir(parents=True, exist_ok=True); FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'  wrote {FN} ({res["runtime_s"]:.0f} s)')
        del M, C, AS, ASt; torch.cuda.empty_cache()
    R['runtime_s'] = time.time() - P.T0; FN.write_text(json.dumps(P.strip(R), indent=1)); log('done')


if __name__ == '__main__':
    main()
