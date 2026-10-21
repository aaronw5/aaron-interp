"""B8 verification (fresh test seed + missing baselines).

Re-scores the B8 programs on a fresh test sample and adds two baselines that use the nested program's own physics inputs
without its staged structure (the control B8 lacks):
  DIRECT  MARS from the E11 physics set (Q library + rank-window sums H:..., ~475 inputs) straight to the 5 logits: the
          B4 forward path (results/jedi/b4_direct_terms.json, selected on sample(6000, 0)), prefixes K = 8 .. 100 per logit,
          coefficients refitted by least squares on the B8 fit rows (sample(4000, 808), rng 0, first 2/3). DL by the B8 rule.
  PGBDT   HistGradientBoosting regressors on the same physics set (quantised to 8 bits), leaves 8 / 16, 30 / 100 iterations.
and STD MARS beyond K = 30 (K = 60, 100) so that STD reaches the bit range of the P levels.
Fresh test: sample(2000, 8810) minus every jet of sample(4000, 808), sample(6000, 0) and sample(2000, 809) (B8's test).
Edits: trans (all real particles deta + 0.06 rad), soft (one particle at 0.5 x softest pT, first empty slot), swap (pT of
slots 13/14), D (prong sharing dc = +0.3, E35, on 1,500 random test jets) with control cn. The cf search is not repeated.
Programs: P levels (stored E11/B4 nested programs, as B8), STD MARS (B8 fit rows), EFP MARS (B8 fit-row EFPs from
b8_efp.npz; fresh-test EFPs computed in venv-ef), GBDT on STD u EFP (q 8 grid + q4 L8 T100), DIRECT, PGBDT.
Scores as B8 (same call, Cohen kappa of call-change indicators, margin ratio R), P IIA (P30, P8r10, P2r10) with permuted
alignment, matched-DL paired bootstrap with BH. Reference rules for kappa trans: 'flip iff the network's base call is c'.
Modes (one Job: pool && efp && main):
  python -m interp.b8_verify pool;  /j-jepa-vol/aaron-interp/venv-ef/bin/python -m interp.b8_verify efp;  python -m interp.b8_verify main
Writes $AI_RESULTS/jedi/b8_verify.json (intermediate b8_verify_pool.npz / b8_verify_efp.npz are removed at the end of main).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('KERAS_BACKEND', 'jax')
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
SEED_T = 8810
EDITS = ['trans', 'soft', 'swap', 'D', 'cn']


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def mode_pool(a):
    from interp.jedi_core import Jedi, sample
    from interp.e35_prong_sharing_route_jedi import make_edits
    from interp import b8_fidelity_dl_iia as B
    net = Jedi(64); xT, yT, _ = sample(2000, SEED_T)
    iT = B.indices(2000, SEED_T); excl = np.concatenate([B.indices(4000, 808), B.indices(6000, 0), B.indices(2000, 809)])
    keep = ~np.isin(iT, excl); xT, yT = np.ascontiguousarray(xT[keep]), yT[keep]
    J = len(xT); real = xT[..., 0] > 0; nn = real.sum(1); X, BASE, BSET = {'test': xT}, {}, {}
    info = dict(N_test=int(J), removed=int((~keep).sum()), contiguous_real=bool((np.diff(real.astype(int), axis=1) <= 0).all()))
    tr = xT.copy(); tr[..., 1] += np.float32(0.06) * real; X['trans'] = tr; BASE['trans'] = np.arange(J); BSET['trans'] = 'test'
    js = np.flatnonzero(nn < 64); rng = np.random.default_rng(8094); so = xT[js].copy()
    ptmin = np.where(real[js], xT[js][..., 0], np.inf).min(1); r_ = 0.4 * np.sqrt(rng.uniform(size=len(js))); ph = rng.uniform(0, 2 * np.pi, len(js))
    ar = np.arange(len(js)); so[ar, nn[js], 0] = 0.5 * ptmin; so[ar, nn[js], 1] = r_ * np.cos(ph); so[ar, nn[js], 2] = r_ * np.sin(ph)
    X['soft'] = so.astype(np.float32); BASE['soft'] = js; BSET['soft'] = 'test'
    jw = np.flatnonzero(nn >= 15); sw = xT[jw].copy(); sw[:, [13, 14], 0] = xT[jw][:, [14, 13], 0]; X['swap'] = sw; BASE['swap'] = jw; BSET['swap'] = 'test'
    C = np.sort(np.random.default_rng(8091).choice(J, 1500, replace=False))
    XD, acc, _, _, _, nnC, _ = make_edits(xT[C], 8092, False)
    dc = min([k for k in acc], key=lambda d: abs(d - 0.3)); m = np.flatnonzero(acc[dc] & (nnC >= 3))
    X['C_id'] = XD['id']; X['D'] = XD[(dc, 'D')][m]; X['cn'] = XD[(dc, 'cn')][m]
    BASE['D'] = m; BSET['D'] = 'C_id'; BASE['cn'] = m; BSET['cn'] = 'C_id'; del XD
    info.update(N_soft=int(len(js)), N_swap=int(len(jw)), D=dict(dc=float(dc), N_C=1500, accepted=int(len(m))))
    out = {}
    for s, xx in X.items():
        xx = np.ascontiguousarray(xx, np.float32)
        R = net.run(xx, keep=('S', 'h4', 'h5', 'z', 'logits') if s == 'test' else ('logits',))
        out[f'L|{s}'] = R['logits'].astype(np.float32); out[f'x|{s}'] = xx; out[f'std|{s}'] = B.std_obs(xx)
        if s == 'test':
            for k in ('S', 'h4', 'h5', 'z'): out[f'{k}|test'] = R[k].astype(np.float32)
        if s in BASE: out[f'base|{s}'] = BASE[s]
        log(f'set {s}: {len(xx)} jets')
    out['y|test'] = yT
    info['accuracy_test'] = float((out['L|test'].argmax(1) == yT).mean()); info['base_set'] = BSET
    info['flip'] = {s: float((out[f'L|{s}'].argmax(1) != out[f'L|{BSET[s]}'].argmax(1)[BASE[s]]).mean()) for s in BASE}
    log(json.dumps(info))
    np.savez_compressed(OUT / 'b8_verify_pool.npz', **out); (OUT / 'b8_verify_pooltmp.json').write_text(json.dumps(info))


def mode_efp(a):
    import energyflow as ef
    Z = np.load(OUT / 'b8_verify_pool.npz')
    es = ef.EFPSet('d<=5', 'p==1', measure='hadr', beta=1, normed=True, coords='ptyphim')
    out = {}
    for s in ('test', 'soft', 'swap', 'C_id', 'D', 'cn'):
        t = time.time(); xx = Z[f'x|{s}']
        out[f'efp|{s}'] = np.asarray(es.batch_compute([j[j[:, 0] > 0][:, [0, 1, 2]].astype(np.float64) for j in xx], n_jobs=1), np.float32)
        log(f'EFP {s} {out[f"efp|{s}"].shape} {time.time() - t:.0f} s')
    xx = Z['x|trans'][:500]; e5 = np.asarray(es.batch_compute([j[j[:, 0] > 0][:, [0, 1, 2]].astype(np.float64) for j in xx], n_jobs=1))
    out['trans_check'] = np.array([float(np.max(np.abs(e5 - out['efp|test'][:500]) / (np.abs(out['efp|test'][:500]) + 1e-12)))])
    log('EFP trans invariance max rel diff (500 jets)', out['trans_check'])
    np.savez_compressed(OUT / 'b8_verify_efp.npz', **out)


def mode_main(a):
    import warnings, gc
    warnings.filterwarnings('ignore')
    from sklearn.ensemble import HistGradientBoostingRegressor
    from interp.jedi_core import Jedi
    from interp import e11_mars_all as E
    from interp.e11_verify import Last, chain
    from interp.b6w_symreg import quantiser, hgb_bits_q
    from interp.b6_pareto import term_bits
    from jetdistill import mars
    from interp import b8_fidelity_dl_iia as B
    Z = np.load(OUT / 'b8_verify_pool.npz'); info = json.loads((OUT / 'b8_verify_pooltmp.json').read_text())
    EFv = np.load(OUT / 'b8_verify_efp.npz'); P8 = np.load(OUT / 'b8_pool.npz'); EF8 = np.load(OUT / 'b8_efp.npz')
    BSET = info['base_set']; sets = ['test', 'trans', 'soft', 'swap', 'C_id', 'D', 'cn']
    L = {s: Z[f'L|{s}'].astype(np.float64) for s in sets}; L['fit'] = P8['L|fit'].astype(np.float64); L['dev'] = P8['L|dev'].astype(np.float64)
    call = {s: v.argmax(1) for s, v in L.items()}; base = {s: Z[f'base|{s}'] for s in BSET}
    STD = {s: Z[f'std|{s}'] for s in sets}; STD['fit'] = P8['std|fit']; STD['dev'] = P8['std|dev']
    EFP = {s: EFv[f'efp|{s}'].astype(np.float64) for s in ('test', 'soft', 'swap', 'C_id', 'D', 'cn')}
    EFP['trans'] = EFP['test']                       # EFPs are translation invariant (checked on 500 jets in mode efp)
    EFP['fit'] = EF8['efp|fit'].astype(np.float64); EFP['dev'] = EF8['efp|dev'].astype(np.float64)
    allS = sets + ['fit', 'dev']
    EFX = {s: np.concatenate([EFP[s], STD[s][:, [0, 5, 6]]], 1) for s in allS}
    UN = {s: np.concatenate([STD[s], EFP[s]], 1) for s in allS}
    RES = dict(procedure=__doc__, pool=info, efp_trans_check=float(EFv['trans_check'][0]), programs={})
    Nt = len(call['test']); Bn = a.B; Ct = B.counts(Nt, Bn, 11)
    RES['accuracy_test'] = float((call['test'] == Z['y|test']).mean())
    FAM = {}
    for i, f in enumerate(EDITS):
        bs = BSET[f]; nb = call[bs][base[f]]; ne = call[f]
        FAM[f] = dict(nb=nb, nf=nb != ne, dm=B.margin(L[f], nb) - B.margin(L[bs][base[f]], nb), C=B.counts(len(nb), Bn, 100 + i), bs=bs)
    RES['network'] = {f: dict(N=int(len(v['nf'])), flip=float(v['nf'].mean()), flip_ci=B.ci(v['C'] @ v['nf'].astype(np.float32) / len(v['nf'])), dmargin=float(v['dm'].mean())) for f, v in FAM.items()}
    # reference rules for kappa trans: flip iff base call == c
    ft = FAM['trans']; RES['trans_by_class'] = {B.CL[c]: dict(n=int((ft['nb'] == c).sum()), flip=float(ft['nf'][ft['nb'] == c].mean())) for c in range(5)}
    RES['kappa_trans_class_rule'] = {B.CL[c]: B.kappa(ft['nf'], ft['nb'] == c) for c in range(5)}
    RES['kappa_trans_class_rule']['W or Z'] = B.kappa(ft['nf'], np.isin(ft['nb'], [2, 3]))
    log('network', json.dumps(RES['network']), json.dumps(RES['trans_by_class']), json.dumps(RES['kappa_trans_class_rule']))
    PROG = {}

    def add(nm, fam, bits, preds, extra=None): PROG[nm] = dict(family=fam, bits=float(bits), L=preds, extra=extra or {})
    # ---- physics for P / DIRECT / PGBDT
    xs = {s: Z[f'x|{s}'] for s in sets}; xs['fit'] = P8['x|fit']; xs['dev'] = P8['x|dev']
    PH = {}
    for s in allS: PH[s] = B.physics(xs[s]); log(f'physics {s} {len(xs[s])}')
    # ---- P levels
    stored = {K: B.load_P(K) for K in sorted({lv[1] for lv in B.P_LEVELS})}
    rfm = np.random.default_rng(8098).choice(len(xs['fit']), 3000, replace=False); PHf = {k: v[rfm] for k, v in PH['fit'].items()}
    PV = {}; net = Jedi(64); last = Last(net)
    for name, K, b, dc in B.P_LEVELS:
        means = None
        if dc > 0: Vf = chain(stored[K], PHf); means = {s: Vf[s].mean(0) for s in E.STG}
        fb, bits, dropped = B.level_P(stored[K], b, dc, means); preds = {}
        for s in sets:
            V = chain(fb, PH[s]); preds[s] = last.jax_logits(V['z'])
            if s == 'test': PV[name] = (fb, V)
        add(name, 'P', bits, preds, dict(K=K))
        log(f'{name}: {bits:.0f} bits; same call {(preds["test"].argmax(1) == call["test"]).mean() * 100:.2f} %')
    # ---- DIRECT (B4 terms, lstsq refit on B8 fit rows)
    dterms = json.loads((OUT / 'b4_direct_terms.json').read_text()); keysP = list(PH['fit'].keys()); nkP = len(keysP)
    need = {t['q'] for f in dterms for t in f['terms']} | {t['q2'] for f in dterms for t in f['terms'] if t.get('q2')}
    miss = [k for k in need if k not in PH['fit']]; RES['direct_missing_keys'] = miss; log('direct: missing keys', miss)
    if not miss:
        NF = len(L['fit'])
        for kb in (8, 16, 32, 64, 100):
            parts = []
            for f in dterms:
                tr = f['terms'][:kb]; co = mars._lstsq(tr, PH['fit'], L['fit'][:, f['cls']] + f['shift'], NF); parts.append((tr, co, f['shift']))
            bits = sum(32 + sum(sum(term_bits(t, nkP)) for t in tr) for tr, _, _ in parts)
            pred = lambda ph, parts=parts: np.stack([np.stack([np.ones(len(next(iter(ph.values()))))] + [mars.term_value(t, ph) for t in tr], 1) @ co - sh for tr, co, sh in parts], 1)
            add(f'DIRECT{kb}', 'DIRECT', bits, {s: pred(PH[s]) for s in allS}, dict(K=kb))
            log(f'DIRECT{kb}: {bits:.0f} bits; same call {(PROG[f"DIRECT{kb}"]["L"]["test"].argmax(1) == call["test"]).mean() * 100:.2f} %')
    # ---- PGBDT on physics
    keysG = [k for k in keysP if np.isfinite(PH['fit'][k]).all() and np.ptp(PH['fit'][k]) > 0]
    PX = {s: np.nan_to_num(np.stack([PH[s][k] for k in keysG], 1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) for s in allS}
    del PH, PHf; gc.collect()
    qf = quantiser(PX['fit'], 8)
    for lv, it in ((8, 30), (8, 100), (16, 100)):
        t = time.time(); rs = [HistGradientBoostingRegressor(max_leaf_nodes=lv, max_iter=it, learning_rate=0.5, early_stopping=False, random_state=0).fit(qf(PX['fit']), L['fit'][:, k]) for k in range(5)]
        add(f'PGBDT_L{lv}_T{it}', 'PGBDT', sum(hgb_bits_q(r, len(keysG), 8) for r in rs), {s: np.stack([r.predict(qf(PX[s])) for r in rs], 1) for s in allS})
        log(f'PGBDT_L{lv}_T{it}: {PROG[f"PGBDT_L{lv}_T{it}"]["bits"]:.0f} bits, {time.time() - t:.0f} s; same call {(PROG[f"PGBDT_L{lv}_T{it}"]["L"]["test"].argmax(1) == call["test"]).mean() * 100:.2f} %')
    del PX; gc.collect()
    # ---- STD / EFP MARS
    for fam, XX, keys, Ks in (('STD', STD, B.STD_NAMES, (2, 4, 8, 16, 30, 60, 100)), ('EFP', EFX, [f'EFP{i}' for i in range(EFX['fit'].shape[1] - 3)] + ['m', 'sum pT', 'N'], (2, 4, 8, 16, 30, 50))):
        progs, pred = B.mars_fit(XX['fit'], L['fit'], XX['dev'], L['dev'], keys, Ks, fam, 8000)
        for K, (parts, bits) in progs.items():
            add(f'{fam}{K}', fam, bits, {s: pred(parts, XX[s]) for s in allS}, dict(K=K))
            log(f'{fam}{K}: {bits:.0f} bits; same call {(PROG[f"{fam}{K}"]["L"]["test"].argmax(1) == call["test"]).mean() * 100:.2f} %')
    # ---- GBDT on STD u EFP
    F = UN['fit'].shape[1]
    for q in (8,):
        qg = quantiser(UN['fit'], q)
        for lv in (4, 8, 16):
            for it in (10, 30, 100):
                rs = [HistGradientBoostingRegressor(max_leaf_nodes=lv, max_iter=it, learning_rate=0.5, early_stopping=False, random_state=0).fit(qg(UN['fit']), L['fit'][:, k]) for k in range(5)]
                add(f'GBDT_q{q}_L{lv}_T{it}', 'GBDT', sum(hgb_bits_q(r, F, q) for r in rs), {s: np.stack([r.predict(qg(UN[s])) for r in rs], 1) for s in allS})
    log('GBDT done')
    # ---- scores
    PER = {}
    for nm, p in PROG.items():
        sc = p['L']['test'].argmax(1) == call['test']; p['same_call'] = [float(sc.mean())] + B.ci(Ct @ sc.astype(np.float32) / Nt)
        p['same_call_dev'] = float((p['L']['dev'].argmax(1) == call['dev']).mean()) if 'dev' in p['L'] else None
        PER[nm] = dict(nat=sc); p['edits'] = {}
        for f in EDITS:
            n = FAM[f]; Pb = p['L'][n['bs']][base[f]]; Pe = p['L'][f]; sf = Pb.argmax(1) != Pe.argmax(1)
            sdm = B.margin(Pe, n['nb']) - B.margin(Pb, n['nb']); bk = B.kappa_b(n['C'], n['nf'], sf)
            p['edits'][f] = dict(kappa=[B.kappa(n['nf'], sf)] + B.ci(bk), prog_flip=float(sf.mean()),
                                 ratio=float(sdm.mean() / n['dm'].mean()) if abs(n['dm'].mean()) > 1e-12 else None)
            PER[nm][f] = dict(bk=bk, ag=sf == n['nf'])
    for nm, p in PROG.items():
        log(f"{nm:18s} {p['bits'] / 1e3:7.1f} kbit  same {p['same_call'][0] * 100:5.2f}  " + '  '.join(f"k{f} {p['edits'][f]['kappa'][0]:+.3f}" for f in EDITS) + f"  R {p['edits']['trans']['ratio']}")
    # ---- matched DL
    tests = []
    fams = ['STD', 'EFP', 'GBDT', 'DIRECT', 'PGBDT']
    for name, *_ in B.P_LEVELS:
        pb = PROG[name]['bits']
        for fam in fams:
            mem = [n for n, p in PROG.items() if p['family'] == fam]
            if not mem: continue
            ok = [n for n in mem if PROG[n]['bits'] <= pb]
            m = max(ok, key=lambda n: PROG[n]['same_call_dev']) if ok else min(mem, key=lambda n: PROG[n]['bits'])
            d = PER[name]['nat'].astype(np.float32) - PER[m]['nat']; bd = Ct @ d / Nt
            tests.append(dict(P=name, family=fam, member=m, member_bits=PROG[m]['bits'], above=not ok, score='same call', d=float(d.mean()), ci=B.ci(bd), p=B.bootp(bd)))
            for f in EDITS:
                bk = PER[name][f]['bk'] - PER[m][f]['bk']
                tests.append(dict(P=name, family=fam, member=m, member_bits=PROG[m]['bits'], above=not ok, score=f'kappa {f}',
                                  d=PROG[name]['edits'][f]['kappa'][0] - PROG[m]['edits'][f]['kappa'][0], ci=B.ci(bk), p=B.bootp(bk)))
    q = B.bh([t['p'] for t in tests])
    for t, q_ in zip(tests, q): t['bh'] = float(q_)
    RES['matched_dl'] = tests
    for t in tests:
        if t['P'] in ('P30', 'P8r10', 'P2r10') and t['score'] in ('same call', 'kappa trans', 'kappa D'):
            log(f"{t['P']} vs {t['family']} {t['member']}{' ABOVE' if t['above'] else ''}: {t['score']} {t['d']:+.3f} {[round(x, 3) for x in t['ci']]} q {t['bh']:.3g}")
    # ---- IIA for P (aligned vs permuted)
    Sx = {k: Z[f'{k}|test'].astype(np.float64) for k in ('S', 'h4', 'h5', 'z')}
    T, d0 = B.exact_tail(net, Sx['S'], L['test']); RES['tail_max_diff'] = d0
    rng = np.random.default_rng(8813); NP = 3000; bi = rng.choice(Nt, NP, replace=False); si = rng.choice(Nt, NP, replace=True)
    si[si == bi] = (si[si == bi] + 1) % Nt; nbase = call['test'][bi]; Cp = B.counts(NP, Bn, 12)
    live = {s: [int(u) for u in np.flatnonzero(Sx[s].std(0) > 1e-9)] for s in ('S', 'h4', 'z')}

    def net_swap(st, u):
        if st == 'S': Sp = Sx['S'][bi].copy(); Sp[:, u] = Sx['S'][si][:, u]; return T(Sp).argmax(1)
        mm = np.zeros(B.WID[st]); mm[u] = 1
        return T(Sx['S'][bi], h4=Sx['h4'][si], m4=mm).argmax(1) if st == 'h4' else T(Sx['S'][bi], z=Sx['z'][si], mz=mm).argmax(1)
    rngp = np.random.default_rng(8100); NETSW = {}
    VARS = {k: v for k, v in B.P_VARS.items() if v[1] is not None}
    for vn, (st, u) in VARS.items():
        pool = [x for x in E.UNITS[st] if x not in u] if st != 'z' else [x for x in live['z'] if x not in u]
        NETSW[vn] = dict(al=net_swap(st, u), perm=[net_swap(st, list(rngp.choice(pool, len(u), replace=False))) for _ in range(5)])
    RES['iia'] = {}
    for name in ('P30', 'P8r10', 'P2r10'):
        fb, V = PV[name]; pbase = last.jax_logits(V['z'][bi]).argmax(1); RES['iia'][name] = {}
        for vn, (st, u) in VARS.items():
            cols = [E.UNITS[st].index(x) for x in u]; Vb = {s: V[s][bi] for s in E.STG}; Vb[st] = Vb[st].copy(); Vb[st][:, cols] = V[st][si][:, cols]
            pc = last.jax_logits(B.chain_from(fb, Vb, st)['z']).argmax(1)
            eff = NETSW[vn]['al'] != nbase; ag = pc == NETSW[vn]['al']; ne = Cp @ eff.astype(np.float32)
            be = (Cp @ (ag & eff).astype(np.float32)) / np.where(ne > 0, ne, np.nan)
            pe, pbs = [], []
            for ncp in NETSW[vn]['perm']:
                e2 = ncp != nbase; a2 = pc == ncp; n2 = Cp @ e2.astype(np.float32)
                pe.append(float(a2[e2].mean())); pbs.append((Cp @ (a2 & e2).astype(np.float32)) / np.where(n2 > 0, n2, np.nan))
            dd = be - np.nanmean(np.stack(pbs), 0)
            RES['iia'][name][vn] = dict(n_eff=int(eff.sum()), iia_eff=[float(ag[eff].mean())] + B.ci(be), perm=float(np.mean(pe)),
                                        noswap=float((pbase[eff] == NETSW[vn]['al'][eff]).mean()), d=float(ag[eff].mean() - np.mean(pe)), d_ci=B.ci(dd), p=B.bootp(dd))
        effs = [r['iia_eff'][0] for r in RES['iia'][name].values()]; perms = [r['perm'] for r in RES['iia'][name].values()]
        RES['iia'][name]['_mean'] = dict(eff=float(np.mean(effs)), perm=float(np.mean(perms)))
        log(f'{name} IIA mean eff {np.mean(effs):.3f} perm {np.mean(perms):.3f}; ' + '; '.join(f'{v.split(" (")[0]} {r["iia_eff"][0]:.3f}/{r["perm"]:.3f} n{r["n_eff"]}' for v, r in RES['iia'][name].items() if v != '_mean'))
    RES['programs'] = {nm: {k: v for k, v in p.items() if k != 'L'} for nm, p in PROG.items()}
    (OUT / 'b8_verify.json').write_text(json.dumps(RES, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)))
    for f in ('b8_verify_pool.npz', 'b8_verify_efp.npz', 'b8_verify_pooltmp.json'):
        try: (OUT / f).unlink()
        except Exception: pass  # noqa: E701
    log('wrote b8_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['pool', 'efp', 'main']); ap.add_argument('--B', type=int, default=1000)
    a = ap.parse_args(); log('args', vars(a))
    dict(pool=mode_pool, efp=mode_efp, main=mode_main)[a.mode](a); log('done')
