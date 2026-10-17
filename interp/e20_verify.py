"""E20v (skeptic rerun of E20, claims A1/A3): are the pT-rank windows causal for the angular second-round objects?

Fresh sample: interp.jedi_core.sample(per_class=2000, seed=SEED (default 77)); exact fixed-point Jedi(64) and the E20 float + masks
programs (imported from interp.e20_rank_windows_jedi: masks, edits, program forward pass are E20's own code).
1. Reproduction of E20's headline numbers (A13 / I5 on S43, S52, S63, S1; S13 / S17 with qs1 ones; S17 A12 with the window shifted +3).
2. Edge scan (control not in E20): angle swaps W_k (ranks k <-> k+1, pT order kept) for k = 0..16 and 30, in the network and in the
   programs (trained masks; all four slot masks = 1; all four shifted +3 / -3; single and paired layers set to 1). Window hypothesis:
   |dS_c| peaks at the swap across each unit's window edge, the peak moves by +/-3 with the shifted masks, and the profile flattens
   with the masks removed. H0 (no window): the across swap gives no larger |dS| than the adjacent swaps (k-1, k+1).
   Per-jet paired tests: Wilcoxon on |dS_edge| - |dS_adjacent|, Holm over all (unit, edge, neighbour) tests; bootstrap CIs.
3. Near-rank difference-in-differences for the 0-13 units: [|dS_W13|mask - |dS_W13|ones] - [|dS_W12|mask - |dS_W12|ones]
   (W12 is inside the 0-13 window at the adjacent rank, unlike E20's I5 at rank 5).
4. Network calls changed per swap, with McNemar for W13 vs W12 / W14 / W5.
5. Accuracy controls: permuted qs1 masks (5 draws), all four (3 draws), ones / shifts; sd(S) per program (scale check).
6. Mask structure: zero-bit pattern per slot of every quantizer, including 'add' (not used by the E20 program) and l1.
Writes $AI_RESULTS/jedi/e20_verify.json.     Run: python -m interp.e20_verify [--seed 77]
"""
import json, os, sys, time
import numpy as np

SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 77
PER = 2000; NB = 1000
from interp import e20_rank_windows_jedi as E        # loads Jedi(64), masks, program forward (E20 code)
from interp.jedi_core import sample
from interp import e19_common as C
from scipy import stats

T0 = time.time()
def log(*a): print(f'[v {time.time() - T0:7.1f}s]', *a, flush=True)
OUT = E.OUT
rng = np.random.default_rng(SEED + 1000)
LIVE = E.LIVE_S; M1 = E.M1; MASK0 = E.MASK0; SLOT = E.SLOT
KEY = [43, 52, 63, 0, 1, 13, 17, 19, 20, 51, 56, 62, 29, 30]


def boot_mean(a, B=NB):
    a = np.asarray(a, np.float64); n = len(a); v = []
    for _ in range(B // 100):
        idx = rng.integers(0, n, (100, n)); v.append(a[idx].mean(1))
    v = np.concatenate(v); return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def boot_ratio(a, b, B=NB):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); n = len(a); v = []
    for _ in range(B // 100):
        idx = rng.integers(0, n, (100, n)); v.append(a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-12))
    v = np.concatenate(v); return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def wil(d):
    d = np.asarray(d, np.float64)
    if not (d != 0).any(): return 1.0
    try: return float(stats.wilcoxon(d).pvalue)
    except ValueError: return 1.0


def bits(name):
    q = E.L[name].iq; qs = q.quantizers if hasattr(q, 'quantizers') else [q]; out = []
    for qq in qs:
        qq = qq.quantizer if hasattr(qq, 'quantizer') else qq
        out.append(np.asarray(qq.b) if hasattr(qq, 'b') else None)
    return out


def main():
    R = dict(seed=SEED, per_class=PER, n_boot=NB)
    # ---------------------------------------------------------------- 6. mask structure
    MS = {}
    for name in E.NAMES + ('add',):
        try:
            bl = bits(name)
        except Exception as ex:
            MS[name] = f'error {ex}'; continue
        rec = []
        for b in bl:
            if b is None: rec.append(None); continue
            r = dict(shape=list(b.shape), zero_frac=float((b == 0).mean()))
            if b.ndim == 3 and b.shape[1] == 64:
                r['zero_frac_per_slot'] = (b[0] == 0).mean(1).round(3).tolist()
                r['slot_dependent'] = bool((b[0] != b[0][:1]).any())
            rec.append(r)
        MS[name] = rec
    R['mask_structure'] = MS
    for k, v in MS.items():
        log('mask', k, v if isinstance(v, str) else [None if r is None else (r['shape'], round(r['zero_frac'], 3), r.get('slot_dependent')) for r in v])
    l1 = MASK0['l1'][0]
    R['l1_kept_slots_per_feature'] = {f: E.describe(l1[:, i]) for i, f in enumerate(('pt', 'deta', 'dphi'))} if l1.ndim == 2 and l1.shape[0] == 64 else str(l1.shape)
    log('l1 kept slots per input feature:', R['l1_kept_slots_per_feature'])
    l3 = MASK0['l3'][0]; l2 = MASK0['l2'][0]
    try:
        R['l3_kept_slots_s_inputs'] = {f's{c}': E.describe(l3[:, c]) for c in (43, 60, 53, 32, 52, 14)}
        R['l2_kept_slots_a_inputs'] = {f'a{c}': E.describe(l2[:, c]) for c in (29, 6, 36, 40, 43, 46, 3, 30, 32, 15, 56)}
    except Exception as ex:
        R['l3_kept_slots_s_inputs'] = R['l2_kept_slots_a_inputs'] = f'shape l3 {l3.shape} l2 {l2.shape}: {ex}'
    log('l3 kept slots (s inputs):', R['l3_kept_slots_s_inputs']); log('l2 kept slots (a inputs):', R['l2_kept_slots_a_inputs'])
    R['qs1_windows'] = {c: E.describe(M1[:, c]) for c in LIVE}
    EDGES = {c: [e for e in range(0, 17) if M1[e, c] != M1[e + 1, c]] for c in LIVE}
    R['qs1_edges_0_16'] = EDGES; log('qs1 edges in 0..16 per live S unit:', EDGES)

    # ---------------------------------------------------------------- sample, programs
    x, y, _ = sample(per_class=PER, seed=SEED)
    nreal = (x[..., 0] > 0).sum(1)
    R['N'] = int(len(y)); R['sorted_frac'] = float((np.diff(x[..., 0], axis=1) <= 1e-6).all(1).mean())
    base_net = E.net.run(x, keep=('logits', 'S')); cnet = base_net['logits'].argmax(1)
    R['acc_net'] = float((cnet == y).mean()); log(f'seed {SEED}: N={len(y)} acc net {100 * R["acc_net"]:.2f} %; sorted {R["sorted_frac"]:.4f}')
    def ones(layers):
        M = dict(MASK0); M.update({k: np.ones_like(MASK0[k]) for k in layers}); return M
    PR = {'mask': dict(MASK0), 'ones_all4': ones(SLOT)}
    for d in (+3, -3):
        M = dict(MASK0); M.update({k: E.shift_mask(MASK0[k], d) for k in SLOT}); PR[f'shift{d:+d}_all4'] = M
    for k in SLOT: PR[f'ones_{k}'] = ones((k,))
    PR['ones_l2_l3'] = ones(('l2', 'l3')); PR['ones_l3_qs1'] = ones(('l3', 'qs1')); PR['ones_l2_qs1'] = ones(('l2', 'qs1'))
    base = {'net': base_net}; R['programs'] = {}
    for p, M in PR.items():
        lo, S = E.prog_run(x, M); base[p] = dict(logits=lo, S=S)
        R['programs'][p] = dict(acc=float((lo.argmax(1) == y).mean()), calls_vs_net=float((lo.argmax(1) != cnet).mean()),
                                sd_S={c: float(S[:, c].std()) for c in LIVE})
        log(f'program {p}: acc {100 * R["programs"][p]["acc"]:.2f} %, calls vs net {100 * R["programs"][p]["calls_vs_net"]:.2f} %; '
            f'sd S43/S52/S63 {S[:, 43].std():.3f}/{S[:, 52].std():.3f}/{S[:, 63].std():.3f}')
    R['sd_S_net'] = {c: float(base_net['S'][:, c].std()) for c in LIVE}

    # ---------------------------------------------------------------- 2. swap scan
    KS = list(range(0, 17)) + [30]
    MAIN = ['net', 'mask', 'ones_all4', 'shift+3_all4', 'shift-3_all4']
    # common jet set: jets with >= 32 real particles are not required; use jets with nreal > 17 for the 0..16 scan (all swaps defined)
    Jc = np.flatnonzero(nreal > 17); R['N_scan'] = int(len(Jc)); xs = x[Jc]
    log(f'scan jets (nreal > 17): {len(Jc)}')
    dS = {p: {} for p in PR}; dS['net'] = {}; chg = {'net': {}, 'mask': {}, 'ones_all4': {}}
    scan = {}
    for k in KS:
        J, xe, _ = E.make_edit(xs, 'swap', k)
        if len(J) != len(Jc):   # k = 30 is defined on fewer jets; keep it on its own set
            pass
        r = E.net.run(xe, keep=('logits', 'S'))
        b0 = {kk: v[Jc][J] for kk, v in base_net.items()}
        dS['net'][k] = (r['S'] - b0['S']).astype(np.float32); chg['net'][k] = (J, r['logits'].argmax(1) != b0['logits'].argmax(1))
        for p, M in PR.items():
            lo, S = E.prog_run(xe, M); bS = base[p]['S'][Jc][J]
            dS[p][k] = (S - bS).astype(np.float32)
            if p in chg: chg[p][k] = (J, lo.argmax(1) != base[p]['logits'][Jc][J].argmax(1))
        scan[k] = dict(N=int(len(J)), units={c: {p: float(np.abs(dS[p][k][:, c]).mean()) for p in dS} for c in LIVE},
                       calls_changed={p: float(chg[p][k][1].mean()) for p in chg})
        log(f'W{k}: N={len(J)} calls net {100 * scan[k]["calls_changed"]["net"]:.2f} %; |dS| mask/ones/s+3/s-3 ' +
            ' '.join(f'S{c}:{scan[k]["units"][c]["mask"]:.4f}/{scan[k]["units"][c]["ones_all4"]:.4f}/'
                     f'{scan[k]["units"][c]["shift+3_all4"]:.4f}/{scan[k]["units"][c]["shift-3_all4"]:.4f}' for c in (43, 52, 63, 1, 13, 17)))
    R['scan'] = scan

    # edge tests: for each live unit and each qs1 edge e in 0..15 (program with trained masks and network): |dS_e| vs |dS_{e-1}|, |dS_{e+1}|
    ET, ps, keys = {}, [], []
    for c in LIVE:
        for e in EDGES[c]:
            if e > 15: continue
            for p in ('mask', 'net'):
                for nb in (e - 1, e + 1):
                    if nb < 0 or nb > 16 or nb in EDGES[c]: continue
                    a, b = np.abs(dS[p][e][:, c]).astype(np.float64), np.abs(dS[p][nb][:, c]).astype(np.float64); d = a - b
                    rec = dict(edge=e, neighbour=nb, prog=p, mean_abs_edge=float(a.mean()), mean_abs_nb=float(b.mean()),
                               diff=float(d.mean()), diff_ci=boot_mean(d), wilcoxon_p=wil(d))
                    key = f'S{c}_e{e}_{p}_vs{nb}'; ET[key] = rec; ps.append(rec['wilcoxon_p']); keys.append(key)
    for kk, h in zip(keys, C.holm(ps)): ET[kk]['holm_p'] = h
    R['edge_tests'] = ET; R['edge_tests_m'] = len(ps)
    for kk, v in ET.items():
        log(f'edge {kk}: |dS| edge {v["mean_abs_edge"]:.4f} vs nb {v["mean_abs_nb"]:.4f}, diff {v["diff"]:+.4f} {np.round(v["diff_ci"], 4).tolist()} holm p {v["holm_p"]:.1e}')

    # edge following: peak position of the |dS| profile (k = 0..16) per program for the 0-13 units, S13, S17, S1
    PK = {}
    for c in (43, 52, 63, 0, 13, 17, 1, 51, 20, 56, 62):
        PK[c] = {}
        for p in MAIN + ['ones_qs1', 'ones_l3', 'ones_l2', 'ones_l2_l3', 'ones_l3_qs1', 'ones_l2_qs1', 'ones_qs']:
            prof = [float(np.abs(dS[p][k][:, c]).mean()) for k in range(17)]
            PK[c][p] = dict(profile=[round(v, 5) for v in prof], argmax=int(np.argmax(prof)), max=float(max(prof)),
                            median=float(np.median(prof)))
        log(f'S{c} peak rank (mask / ones_all4 / +3 / -3 / net): {PK[c]["mask"]["argmax"]} / {PK[c]["ones_all4"]["argmax"]} / '
            f'{PK[c]["shift+3_all4"]["argmax"]} / {PK[c]["shift-3_all4"]["argmax"]} / {PK[c]["net"]["argmax"]}; '
            f'max {PK[c]["mask"]["max"]:.4f} / {PK[c]["ones_all4"]["max"]:.4f} / {PK[c]["shift+3_all4"]["max"]:.4f} / {PK[c]["shift-3_all4"]["max"]:.4f}')
    R['profiles'] = PK
    # shifted-window edge tests for 0-13 units: under shift-3, W10 vs W9 / W11; under shift+3, W16 vs W15 (and W13 there)
    SH = {}
    for c in (43, 52, 63):
        for p, e, nbs in (('shift-3_all4', 10, (9, 11)), ('shift+3_all4', 16, (15,)), ('mask', 10, (9, 11)), ('mask', 16, (15,))):
            for nb in nbs:
                a, b = np.abs(dS[p][e][:, c]).astype(np.float64), np.abs(dS[p][nb][:, c]).astype(np.float64); d = a - b
                SH[f'S{c}_{p}_W{e}_vs_W{nb}'] = dict(mean_abs_edge=float(a.mean()), edge_ci=boot_mean(a), mean_abs_nb=float(b.mean()),
                                                      diff=float(d.mean()), diff_ci=boot_mean(d), wilcoxon_p=wil(d))
    hp = C.holm([v['wilcoxon_p'] for v in SH.values()])
    for (kk, v), h in zip(SH.items(), hp): v['holm_p'] = h; log(f'shifted {kk}: {v["mean_abs_edge"]:.4f} vs {v["mean_abs_nb"]:.4f} holm p {h:.1e}')
    R['shifted_edge_tests'] = SH

    # ---------------------------------------------------------------- 1. reproduction of E20 headline (scan jets)
    REP = {}
    for c in (43, 52, 63, 1, 13, 17, 29, 30, 0, 19):
        r = {}
        for e in (13, 5, 12, 0, 30):
            for p in ('net', 'mask', 'ones_all4', 'shift+3_all4', 'shift-3_all4', 'ones_l3', 'ones_qs1'):
                a = np.abs(dS[p][e][:, c]).astype(np.float64); r[f'W{e}_{p}'] = dict(mean_abs=float(a.mean()), ci=boot_mean(a))
        a, b = np.abs(dS['mask'][13][:, c]), np.abs(dS['ones_all4'][13][:, c])
        r['W13_reduction_ones_all4'] = dict(value=float(1 - b.mean() / max(a.mean(), 1e-12)),
                                            ci=[1 - v for v in boot_ratio(b, a)[::-1]] if a.mean() > 0 else None)
        r['W13_net_vs_mask_r2'] = E.r2(dS['net'][13][:, c].astype(np.float64), dS['mask'][13][:, c].astype(np.float64))
        REP[c] = r
        log(f'rep S{c}: W13 net {r["W13_net"]["mean_abs"]:.4f} mask {r["W13_mask"]["mean_abs"]:.4f} ones {r["W13_ones_all4"]["mean_abs"]:.4f} '
            f'+3 {r["W13_shift+3_all4"]["mean_abs"]:.4f} -3 {r["W13_shift-3_all4"]["mean_abs"]:.4f} ones_l3 {r["W13_ones_l3"]["mean_abs"]:.4f} '
            f'ones_qs1 {r["W13_ones_qs1"]["mean_abs"]:.4f}; W5 mask/ones {r["W5_mask"]["mean_abs"]:.4f}/{r["W5_ones_all4"]["mean_abs"]:.4f}; '
            f'W12 +3 {r["W12_shift+3_all4"]["mean_abs"]:.4f}; reduction {r["W13_reduction_ones_all4"]}')
    R['reproduction'] = REP

    # ---------------------------------------------------------------- 3. near-rank difference in differences
    DD = {}
    for c in (43, 52, 63):
        for inside in (12, 5):
            d = (np.abs(dS['mask'][13][:, c]) - np.abs(dS['ones_all4'][13][:, c])) - (np.abs(dS['mask'][inside][:, c]) - np.abs(dS['ones_all4'][inside][:, c]))
            d = d.astype(np.float64)
            DD[f'S{c}_W13_vs_W{inside}'] = dict(did=float(d.mean()), ci=boot_mean(d), wilcoxon_p=wil(d),
                                                inside_mask=float(np.abs(dS['mask'][inside][:, c]).mean()),
                                                inside_ones=float(np.abs(dS['ones_all4'][inside][:, c]).mean()))
        # outside neighbour (W14) in the trained program: both particles outside the window -> 0 expected
        a = np.abs(dS['mask'][14][:, c]).astype(np.float64); DD[f'S{c}_W14_mask'] = dict(mean_abs=float(a.mean()), ci=boot_mean(a))
    for kk, v in DD.items(): log('DiD', kk, {k: (np.round(v_, 5).tolist() if isinstance(v_, list) else round(v_, 5)) for k, v_ in v.items()})
    R['did'] = DD

    # ---------------------------------------------------------------- 4. network calls
    CC = {}
    for a_, b_ in ((13, 12), (13, 14), (13, 5), (0, 5), (13, 30)):
        Ja, ca = chg['net'][a_]; Jb, cb = chg['net'][b_]; com = np.intersect1d(Ja, Jb)
        xa, xb = ca[np.searchsorted(Ja, com)], cb[np.searchsorted(Jb, com)]
        CC[f'W{a_}_vs_W{b_}'] = dict(N=int(len(com)), A=float(xa.mean()), B=float(xb.mean()), A_ci=boot_mean(xa), B_ci=boot_mean(xb),
                                     mcnemar=C.mcnemar(xb, xa))
        log(f'calls W{a_} vs W{b_}: {100 * xa.mean():.2f} vs {100 * xb.mean():.2f} %, McNemar p {CC[f"W{a_}_vs_W{b_}"]["mcnemar"]["p"]:.2e}')
    R['calls'] = CC

    # ---------------------------------------------------------------- 5. accuracy controls (permuted masks)
    AC = {}
    for scope, layers, nd in (('qs1', ('qs1',), 5), ('all4', SLOT, 3)):
        accs = []
        for r_ in range(nd):
            rr = np.random.default_rng(2000 + r_); M = dict(MASK0); M.update({k: E.permute_mask(MASK0[k], rr) for k in layers})
            lo, _ = E.prog_run(x, M); accs.append(float((lo.argmax(1) == y).mean()))
        AC[f'perm_{scope}'] = dict(acc_all=accs, acc_mean=float(np.mean(accs)))
        for d in (+3, -3):
            M = dict(MASK0); M.update({k: E.shift_mask(MASK0[k], d) for k in layers}); lo, _ = E.prog_run(x, M)
            AC[f'shift{d:+d}_{scope}'] = float((lo.argmax(1) == y).mean())
        lo, _ = E.prog_run(x, ones(layers)); AC[f'ones_{scope}'] = float((lo.argmax(1) == y).mean())
        log(f'accuracy controls {scope}:', {k: v for k, v in AC.items() if k.endswith(scope)})
    R['accuracy_controls'] = AC
    R['wall_s'] = time.time() - T0
    json.dump(R, open(OUT / 'e20_verify.json', 'w'), indent=1, default=float); log('wrote', OUT / 'e20_verify.json')


if __name__ == '__main__':
    main()
