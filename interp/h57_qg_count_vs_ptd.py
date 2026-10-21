"""H57 (loop D; claims D22 / D7): do the Pythia-trained quark/gluon taggers respond to the particle count N or to the momentum
sharing among the hadrons, pT^D = sqrt(sum pT_i^2) / sum pT_i (the h5/H54 definition; the squared form sum pT_i^2 / (sum pT_i)^2
changes by twice the relative amount), when both are changed separately by the generator?

In H54 (Hz+/Hz-, Lund a/b + StringPT sigma) N and pT^D moved together (dpT^D/pT^D ~ -0.45 dN/N for every string knob), so H54
cannot tell which one the networks read. Here a pilot scan adds knobs that change N and pT^D in different ratios: hadron
species (StringFlav:probStoUD, probQQtoQ, mesonUDvector) and hadron-decay switches (mayDecay = off for rho, omega, K*, eta,
pi0, K_S + hyperons, all light vectors). A decay that is switched off replaces 2-3 decay products by the parent: N drops while
the momentum sharing among the hardest hadrons changes less than under a string-parameter change of the same dN/N.

Settings are solved from the round-1 single-knob responses (linear model) for
  NA+/NA-  N +-9 %, pT^D unchanged, EFN (IRC-safe) unchanged          (count moves alone)
  PA+/PA-  N unchanged, pT^D +-4.4 % (its Hz+ shift), EFN unchanged  (momentum sharing moves alone)
  IR+      N and pT^D unchanged, EFN +0.25                            (IRC-safe content moves alone; control)
measured in round 2, and the best measured setting per target is regenerated with fresh seeds (10,000 q + 10,000 g jets).

Generator: Pythia 8.317 as H54 (interp.h54_qg_generator_knobs: Monash, 14 TeV, Z(nunu)+g / Z(nunu)+uds, MPI on, anti-kT R=0.4,
pT 500-550 GeV, |y| < 1.7), per-event reseeding: every setting starts event i of worker w from the same random state, so the
parton-level event (hard process, shower, MPI) is identical and only hadronisation / decays differ (checked by a parton hash).
Worker ids: pilot 5701/5702 (g/q), pilot second nominal 5703/5704; final 5711-5714, final second nominal V0b 5721-5724.

Stable resonances (decay switched off) are not in the KMT particle vocabulary (11, 13, 22, 130, 211, 321, 2112, 2212). For
scoring they are relabelled by charge: pi0, eta -> 22 (a merged photon pair), other neutral mesons -> 130, charged mesons ->
+-211, neutral baryons -> +-2112, charged baryons -> +-2212. Kinematic-only networks do not read the label.

Steps
  --step pilot  (CPU, env-gen) Herwig availability check; round 1 (21 settings), linear solve, round 2, choice -> h57_pilot.json
  --step gen    (CPU, env-gen) final settings V0, V0b, NAp, NAm, PAp, PAm, IRp, Hzp -> data/qg/gen/h57/final/<s>.npz
  --step score  (GPU, env-part) nets + LR(N), LR(pT^D) + observables on final + pilot settings -> h57_scores.npz
  --step refs   (CPU, venv-ef) GBDT(EFP), GBDT(EFP+N), GBDT(EFP+pT^D), LR(N, pT^D), C2ST -> h57_refs.npz, h57_valid.json
  --step stats  (CPU) paired shifts, kappas, call changes + McNemar, Holm, alpha/beta fit -> h57_run.json
  --step fig    (CPU) h57_settings.png, h57_shifts.png, h57_jet.png
Score convention: s = log p(quark) - log p(gluon).
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
GH = V / 'data' / 'qg' / 'gen' / 'h57'
PIL = GH / 'pilot'
FIN = GH / 'final'
W_PIL, W_PILB, W_FIN, W_FINB = 5701, 5703, 5711, 5721

# continuous knobs: Pythia name, Monash nominal (read back and checked at run time), allowed range
KNOBS = {'a': ('StringZ:aLund', 0.68, (0.2, 1.5)), 'b': ('StringZ:bLund', 0.98, (0.4, 1.8)),
         's': ('StringPT:sigma', 0.335, (0.2, 0.47)), 'sud': ('StringFlav:probStoUD', 0.217, (0.10, 0.35)),
         'qq': ('StringFlav:probQQtoQ', 0.081, (0.02, 0.16)), 'vud': ('StringFlav:mesonUDvector', 0.5, (0.15, 0.9))}
KORDER = list(KNOBS)
DECAYS = {'rho': [113, 213], 'omega': [223], 'Kstar': [313, 323], 'eta': [221, 331], 'KSL': [310, 3122, 3112, 3222, 3312, 3322, 3334],
          'vec': [113, 213, 223, 313, 323, 333], 'pi0': [111]}
# charge of the positive-id particle, baryon flag (for relabelling stable resonances)
CHG = {111: 0, 113: 0, 213: 1, 221: 0, 223: 0, 310: 0, 313: 0, 323: 1, 331: 0, 333: 0, 3122: 0, 3112: -1, 3222: 1, 3312: -1,
       3322: 0, 3334: -1}
VOCAB = {11, 13, 22, 130, 211, 321, 2112, 2212}
ROUND1 = {'nominal': {}, 'nominal_b': {}, 'a+0.5': {'a': 0.5}, 'a-0.4': {'a': -0.4}, 'b-0.4': {'b': -0.4}, 'b+0.6': {'b': 0.6},
          's-0.08': {'s': -0.08}, 's+0.09': {'s': 0.09}, 'sud+0.10': {'sud': 0.10}, 'sud-0.10': {'sud': -0.10},
          'qq+0.07': {'qq': 0.07}, 'qq-0.05': {'qq': -0.05}, 'vud+0.35': {'vud': 0.35}, 'vud-0.30': {'vud': -0.30},
          **{f'D:{d}': {} for d in DECAYS}}
HZP = {'a': 0.11, 'b': -0.10, 's': -0.088}         # H54 Hz+ (0.79, 0.88, 0.247) as deltas
TARGETS = {'NAp': dict(N=9.0, ptd=0.0, efn=0.0), 'NAm': dict(N=-9.0, ptd=0.0, efn=0.0),
           'PAp': dict(N=0.0, ptd=4.4, efn=0.0), 'PAm': dict(N=0.0, ptd=-4.4, efn=0.0), 'IRp': dict(N=0.0, ptd=0.0, efn=0.25)}
WTS = dict(N=1 / 1.0, ptd=1 / 0.25, efn=1 / 0.02)   # inverse tolerances: 1 % in N, 0.25 % in pT^D, 0.02 log-odds in EFN
FINAL = ['V0', 'V0b', 'NAp', 'NAm', 'PAp', 'PAm', 'IRp', 'Hzp']


def G():
    from interp import h54_qg_generator_knobs as g
    return g


def extra_of(spec):
    """spec = dict(knobs={k: delta}, decays=name|None) -> Pythia settings dict"""
    ex = {KNOBS[k][0]: round(KNOBS[k][1] + d, 4) for k, d in spec.get('knobs', {}).items() if abs(d) > 1e-9}
    if spec.get('decays'):
        for dn in spec['decays'].split('+'):
            for pid in DECAYS[dn]: ex[f'{pid}:mayDecay'] = 'off'
    return ex


def relabel(pid):
    """stable resonances -> KMT vocabulary by charge; returns new pid array and the count of relabelled particles"""
    pid = pid.astype(np.int64); ap = np.abs(pid); out = pid.copy(); sg = np.sign(pid); n = 0
    for a, c in CHG.items():
        m = ap == a
        if not m.any(): continue
        n += int(m.sum()); q = c * sg[m]; baryon = a > 1000
        if a in (111, 221): out[m] = 22
        elif baryon: out[m] = np.where(q > 0, 2212, np.where(q < 0, -2212, 2112 * sg[m]))
        else: out[m] = np.where(q > 0, 211, np.where(q < 0, -211, 130))
    return out, n


def load_npz(path, rel=True):
    Z = np.load(path); X, n = Z['X'], Z['n']
    L = [X[j, :n[j]].astype(np.float64) for j in range(len(n))]; nrel = 0
    if rel:
        for a in L:
            a[:, 3], k = relabel(a[:, 3]); nrel += k
    key = Z['widx'].astype(np.int64) * 100_000 + Z['event'].astype(np.int64)
    return L, Z['y'].astype(int), key, Z, nrel


# ----------------------------------------------------------------------------------------------------------------- pilot
def _gen(specs, wbase_of, n_jets, outdir, cpu, max_events=40_000, hash_events=100):
    from multiprocessing import Pool
    g = G(); outdir.mkdir(parents=True, exist_ok=True); tasks = []
    for name, spec in specs.items():
        for cls in (0, 1):
            if (outdir / f'{name}_{cls}.npz').exists(): continue
            tasks.append(dict(cls=cls, widx=wbase_of(name) + cls, extra=extra_of(spec), n_jets=n_jets, max_events=max_events,
                              out=str(outdir / f'{name}_{cls}.npz'), hash_events=hash_events, name=name))
    if tasks:
        with Pool(cpu) as pool: summ = pool.map(g.gen_worker, tasks)
    else: summ = []
    for t, s in zip(tasks, summ): s['name'] = t['name']
    return summ


def _measure(names, ref='nominal'):
    """paired (same event) shifts vs the nominal pilot run: dN/N %, dpT^D/pT^D %, EFN, girth/SD, (m/pT)/SD, C1/SD"""
    g = G(); data = {}
    for nm in names + [ref]:
        for cls in (0, 1):
            L, y, key, Z, nrel = load_npz(PIL / f'{nm}_{cls}.npz')
            data[(nm, cls)] = dict(L=L, key=key, hashes=Z['hashes'], nrel=nrel, npart=sum(len(a) for a in L))
    allL = [x for k in data for x in data[k]['L']]; S = g.efn_scores(allL); O = g.jet_obs(allL); o = 0
    for k in data:
        m = len(data[k]['L']); data[k]['efn'] = S[o:o + m]; data[k]['obs'] = {q: v[o:o + m] for q, v in O.items()}; o += m
    rows = {}
    for nm in names:
        row = {}
        for cls, cn in ((0, 'g'), (1, 'q')):
            d0, d1 = data[(ref, cls)], data[(nm, cls)]
            h0 = {int(h[0]): (h[1], h[2]) for h in d0['hashes']}; h1 = {int(h[0]): (h[1], h[2]) for h in d1['hashes']}
            ch = sorted(set(h0) & set(h1))
            pos = {int(k): j for j, k in enumerate(d0['key'])}
            pr = [(pos[int(k)], j) for j, k in enumerate(d1['key']) if int(k) in pos]
            i0 = np.array([p[0] for p in pr], int); i1 = np.array([p[1] for p in pr], int)
            get = lambda d, q: d['efn'] if q == 'efn' else d['obs'][q]
            r = dict(n=len(d1['L']), n_paired=len(pr), parton_identical=float(np.mean([h0[i] == h1[i] for i in ch])) if ch else None,
                     relabelled_frac=d1['nrel'] / max(d1['npart'], 1))
            if len(pr) < 20:          # unpaired (different seeds, e.g. nominal_b)
                x0, x1 = get(d0, 'N'), get(d1, 'N'); r['paired'] = False
                r['N'] = 100 * (x1.mean() / x0.mean() - 1); r['ptd'] = 100 * (get(d1, 'ptd').mean() / get(d0, 'ptd').mean() - 1)
                r['efn'] = float(get(d1, 'efn').mean() - get(d0, 'efn').mean())
                r['efn_se'] = float(np.sqrt(get(d1, 'efn').var() / len(x1) + get(d0, 'efn').var() / len(x0)))
                for q in ('girth', 'm_over_pt', 'C1_1.0'): r[q] = float((get(d1, q).mean() - get(d0, q).mean()) / get(d0, q).std())
            else:
                r['paired'] = True
                dN = get(d1, 'N')[i1] - get(d0, 'N')[i0]; dp = get(d1, 'ptd')[i1] - get(d0, 'ptd')[i0]; de = get(d1, 'efn')[i1] - get(d0, 'efn')[i0]
                r['N'] = float(100 * dN.mean() / get(d0, 'N')[i0].mean()); r['ptd'] = float(100 * dp.mean() / get(d0, 'ptd')[i0].mean())
                r['ptd_se'] = float(100 * dp.std() / np.sqrt(len(dp)) / get(d0, 'ptd')[i0].mean())
                r['efn'] = float(de.mean()); r['efn_se'] = float(de.std() / np.sqrt(len(de)))
                for q in ('girth', 'm_over_pt', 'C1_1.0'):
                    dd = get(d1, q)[i1] - get(d0, q)[i0]; r[q] = float(dd.mean() / get(d0, q).std())
            row[cn] = r
        rows[nm] = row
        log(f'{nm:24s}', ' | '.join(f"{cn}: dN {row[cn]['N']:+5.1f}% dpTD {row[cn]['ptd']:+5.2f}% dEFN {row[cn]['efn']:+.3f}"
                                   f" girth/sd {row[cn]['girth']:+.3f} m/sd {row[cn]['m_over_pt']:+.3f} parton-same {row[cn]['parton_identical']}"
                                   for cn in ('q', 'g')))
    return rows


def _rvec(r):
    return np.array([r['q']['N'], r['g']['N'], r['q']['ptd'], r['g']['ptd'], r['q']['efn'], r['g']['efn']])


def _tvec(t):
    return np.array([t['N'], t['N'], t['ptd'], t['ptd'], t['efn'], t['efn']])


WV = np.array([WTS['N'], WTS['N'], WTS['ptd'], WTS['ptd'], WTS['efn'], WTS['efn']])


def _objective(r, t):
    return float(((WV * (r - t)) ** 2).sum())


def _bls(A, b, lo, hi, sweeps=3000):
    """box-constrained least squares min ||A x - b||^2, lo <= x <= hi, by cyclic coordinate descent (convex, converges)"""
    Q = A.T @ A; c = A.T @ b; x = np.clip(np.zeros(A.shape[1]), lo, hi)
    for _ in range(sweeps):
        x_old = x.copy()
        for i in range(len(x)):
            if Q[i, i] <= 0: continue
            x[i] = np.clip(x[i] + (c[i] - Q[i] @ x) / Q[i, i], lo[i], hi[i])
        if np.abs(x - x_old).max() < 1e-9: break
    return x


def step_pilot(a):
    g = G(); RES.mkdir(parents=True, exist_ok=True)
    # Herwig availability
    hw = dict(which_Herwig=shutil.which('Herwig'), which_herwig_config=shutil.which('herwig-config'))
    try:
        import herwig  # noqa
        hw['python_module'] = True
    except Exception as e: hw['python_module'] = repr(e)
    hw['volume_dirs'] = sorted(str(p) for p in Path('/j-jepa-vol').glob('*') if 'herwig' in p.name.lower())
    log('Herwig check', hw)
    p = g.make_pythia(0, {}, 123)
    nominal = {k: float(p.settings.parm(v[0])) for k, v in KNOBS.items()}; log('nominal (read back)', nominal)
    for k, v in nominal.items():
        if abs(v - KNOBS[k][1]) > 1e-6: log('nominal differs from table, using read-back value', k, v, KNOBS[k][1])
        KNOBS[k] = (KNOBS[k][0], v, KNOBS[k][2])
    for d, pids in DECAYS.items():
        try: log('decay', d, {pid: bool(p.particleData.mayDecay(pid)) for pid in pids})
        except Exception as e: log('mayDecay read failed', repr(e))
    wb = lambda nm: W_PILB if nm == 'nominal_b' else W_PIL
    specs1 = {nm: dict(knobs=kn, decays=nm[2:] if nm.startswith('D:') else None) for nm, kn in ROUND1.items()}
    summ1 = _gen(specs1, wb, a.pilot_jets, PIL, a.cpu)
    rows = _measure([k for k in specs1 if k != 'nominal'])
    # linear model: slope per unit knob from the two single-knob settings (least squares through the origin)
    J = np.zeros((6, len(KORDER)))
    for i, k in enumerate(KORDER):
        nm = [n for n, kn in ROUND1.items() if list(kn) == [k]]
        dx = np.array([ROUND1[n][k] for n in nm]); R_ = np.stack([_rvec(rows[n]) for n in nm])
        J[:, i] = (R_ * dx[:, None]).sum(0) / (dx ** 2).sum()
    offs = {None: np.zeros(6), **{d: _rvec(rows[f'D:{d}']) for d in DECAYS}}
    import itertools
    small = ['pi0', 'eta', 'KSL', 'omega', 'Kstar']            # combinations assumed additive (checked in round 2)
    for k_ in (2, 3):
        for c_ in itertools.combinations(small, k_): offs['+'.join(c_)] = sum(offs[d] for d in c_)
    # co-movement ratio of string knobs: (dpT^D/pT^D) / (dN/N)
    co = [np.mean([rows[n][c]['ptd'] / rows[n][c]['N'] for c in 'qg']) for n in ('a+0.5', 'a-0.4', 'b-0.4', 'b+0.6', 's-0.08', 's+0.09')]
    comove = float(np.median(co)); log('string co-movement ratio dpTD%/dN%', comove, co)
    rng_ = np.array([KNOBS[k][2][1] - KNOBS[k][2][0] for k in KORDER])
    lo = np.array([KNOBS[k][2][0] - KNOBS[k][1] for k in KORDER]); hi = np.array([KNOBS[k][2][1] - KNOBS[k][1] for k in KORDER])
    lam = a.ridge
    cands = {}
    for tn, t in TARGETS.items():
        tv = _tvec(t); sol = []
        for d, off in offs.items():
            for keep in (KORDER, ['a', 'b', 's'], ['a', 's', 'sud', 'qq', 'vud']):
                ix = [KORDER.index(k) for k in keep]
                A = np.r_[WV[:, None] * J[:, ix], np.sqrt(lam) * np.diag(1 / rng_[ix])]
                bb = np.r_[WV * (tv - off), np.zeros(len(ix))]
                x = np.zeros(len(KORDER)); x[ix] = _bls(A, bb, lo[ix], hi[ix])   # bounded least squares
                pred = J @ x + off
                cost = _objective(pred, tv) + lam * float(((x / rng_) ** 2).sum())
                sol.append((cost, d, tuple(keep), x, pred))
        sol.sort(key=lambda s: s[0]); seen = set(); k = 0
        for cost, d, keep, x, pred in sol:
            sig = (d, tuple(np.round(x, 3)))
            if sig in seen: continue
            seen.add(sig); k += 1
            cands[f'R2:{tn}#{k}'] = dict(knobs={kk: float(round(v, 4)) for kk, v in zip(KORDER, x) if abs(v) > 1e-4}, decays=d,
                                      predicted=pred.tolist(), cost=cost, target=tn)
            log('candidate', f'R2:{tn}#{k}', cands[f'R2:{tn}#{k}']['knobs'], d, 'pred', np.round(pred, 3).tolist(), 'cost', round(cost, 2))
            if k >= a.per_target: break
    specs2 = {nm: dict(knobs=c['knobs'], decays=c['decays']) for nm, c in cands.items()}
    summ2 = _gen(specs2, lambda nm: W_PIL, a.pilot_jets, PIL, a.cpu)
    rows2 = _measure(list(specs2)); rows.update(rows2)
    for nm in cands: cands[nm]['measured'] = _rvec(rows[nm]).tolist()
    allspec = {**specs1, **specs2}
    chosen = {}
    for tn, t in TARGETS.items():
        tv = _tvec(t); best = None
        for nm in rows:
            if nm in ('nominal_b',) or not rows[nm]['q']['paired'] or (nm not in specs1 and nm not in specs2): continue
            c = _objective(_rvec(rows[nm]), tv)
            if best is None or c < best[0]: best = (c, nm)
        r = rows[best[1]]; mN = 0.5 * (r['q']['N'] + r['g']['N']); mP = 0.5 * (r['q']['ptd'] + r['g']['ptd'])
        crit = dict(efn_within_0p03=bool(max(abs(r['q']['efn']), abs(r['g']['efn'])) <= 0.03))
        if tn.startswith('NA'):
            crit.update(N_in_8_10=bool(8 <= abs(mN) <= 10.5), ptd_below_tenth_comove=bool(abs(mP) < 0.1 * abs(comove * mN)))
        elif tn.startswith('PA'):
            crit.update(N_below_1=bool(abs(mN) < 1), ptd_within_1p5_of_target=bool(abs(mP - t['ptd']) < 1.5))
        else:
            crit.update(N_below_1=bool(abs(mN) < 1), ptd_below_0p5=bool(abs(mP) < 0.5))
        e_ = max(abs(r['q']['efn']), abs(r['g']['efn']))
        if tn.startswith('NA'): feas = np.sign(mN) == np.sign(t['N']) and abs(mN) >= 4.5 and abs(mP) < 0.25 * abs(comove * mN) and e_ <= 0.06
        elif tn.startswith('PA'): feas = np.sign(mP) == np.sign(t['ptd']) and abs(mP) >= 2.2 and abs(mN) < 2 and e_ <= 0.06
        else: feas = abs(mN) < 2 and abs(mP) < 0.5 and min(r['q']['efn'], r['g']['efn']) > 0.12
        chosen[tn] = dict(name=best[1], spec=allspec[best[1]], objective=best[0], measured=_rvec(r).tolist(), mean_dN=mN, mean_dptd=mP,
                          criteria=crit, all_met=all(crit.values()), feasible=bool(feas),
                          feasible_rule='NA: dN/N in the target direction >= 4.5 %, |dpT^D| < 0.25 x co-movement x |dN|, |dEFN| <= 0.06; '
                                        'PA: dpT^D in the target direction >= 2.2 %, |dN/N| < 2 %, |dEFN| <= 0.06; IR: |dN| < 2 %, |dpT^D| < 0.5 %, dEFN > 0.12')
        log('chosen', tn, chosen[tn])
    out = dict(herwig=hw, nominal=nominal, round1=list(specs1), round2=cands, rows=rows, J=J.tolist(), knob_order=KORDER,
               decay_offsets={str(k): v.tolist() for k, v in offs.items()}, comove=comove, comove_each=co, chosen=chosen,
               specs={k: v for k, v in allspec.items()}, summaries=summ1 + summ2, jets_per_class=a.pilot_jets, ridge=lam,
               rule=('per target: minimise sum over (dN/N q, g; dpT^D/pT^D q, g; dEFN q, g) of ((measured - target) / tolerance)^2 '
                     'with tolerances 1 %, 0.25 %, 0.02 log-odds, over all paired pilot settings (round 1 + round 2)'))
    (RES / 'h57_pilot.json').write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    log('wrote h57_pilot.json')


# ------------------------------------------------------------------------------------------------------------------- gen
def final_specs():
    P = json.loads((RES / 'h57_pilot.json').read_text())
    for k, v in P['nominal'].items(): KNOBS[k] = (KNOBS[k][0], v, KNOBS[k][2])
    sp = {'V0': dict(knobs={}, decays=None), 'V0b': dict(knobs={}, decays=None), 'Hzp': dict(knobs=HZP, decays=None)}
    for tn in TARGETS:
        if P['chosen'][tn]['feasible']: sp[tn] = P['chosen'][tn]['spec']
    return sp


def final_names():
    sp = final_specs(); return [n for n in FINAL if n in sp]


def step_gen(a):
    from multiprocessing import Pool
    g = G(); FIN.mkdir(parents=True, exist_ok=True); tmp = FIN / 'tmp'; tmp.mkdir(exist_ok=True)
    if a.pick:                                    # e.g. NAm=R2:NAm#3,PAm=R2:PAm#2: choice by hand after round 2 (reason in --pick_note)
        P = json.loads((RES / 'h57_pilot.json').read_text())
        for kv in a.pick.split(','):
            tn, nm = kv.split('=')
            r = P['rows'][nm]; P['chosen'][tn].update(name=nm, spec=P['specs'][nm], measured=_rvec(r).tolist(), feasible=True,
                                                       mean_dN=0.5 * (r['q']['N'] + r['g']['N']), mean_dptd=0.5 * (r['q']['ptd'] + r['g']['ptd']),
                                                       picked_by_hand=a.pick_note)
            log('picked', tn, nm, P['chosen'][tn])
        (RES / 'h57_pilot.json').write_text(json.dumps(P, indent=1))
    sp = final_specs(); per = a.jets // 2; tasks = []
    names = [n for n in final_names() if not a.only or n in a.only.split(',')]
    for nm in names:
        if (FIN / f'{nm}.npz').exists(): continue
        base = W_FINB if nm == 'V0b' else W_FIN
        for cls in (0, 1):
            for w in range(2):
                tasks.append(dict(cls=cls, widx=base + 2 * cls + w, extra=extra_of(sp[nm]), n_jets=per, max_events=48_000,
                                  out=str(tmp / f'{nm}_{cls}_{w}.npz'), hash_events=100, name=nm))
    log('final settings', {k: extra_of(v) for k, v in sp.items()})
    with Pool(a.cpu) as pool: summ = pool.map(g.gen_worker, tasks)
    info = {}
    for nm in names:
        tt = [t for t in tasks if t['name'] == nm]
        if not tt: continue
        Z = [np.load(t['out']) for t in tt]; Lmax = max(z['X'].shape[1] for z in Z)
        X = np.concatenate([np.pad(z['X'], ((0, 0), (0, Lmax - z['X'].shape[1]), (0, 0))) for z in Z])
        np.savez_compressed(FIN / f'{nm}.npz', X=X, n=np.concatenate([z['n'] for z in Z]), y=np.concatenate([z['y'] for z in Z]),
                            event=np.concatenate([z['event'] for z in Z]), widx=np.concatenate([z['widx'] for z in Z]),
                            jet=np.concatenate([z['jet'] for z in Z]), hashes=np.concatenate([z['hashes'] for z in Z]),
                            hash_widx=np.concatenate([np.full(len(z['hashes']), t['widx']) for z, t in zip(Z, tt)]))
        info[nm] = dict(spec=sp[nm], extra=extra_of(sp[nm]), jets=int(len(X)), workers=[s for s, t in zip(summ, tasks) if t['name'] == nm])
        for t in tt: os.remove(t['out'])
        log('wrote', nm, len(X))
    tag = '_'.join(names)
    if not (FIN / 'V0.npz').exists():
        (RES / f'h57_gen_{tag}.json').write_text(json.dumps(info, indent=1, default=str)); return
    Z0 = np.load(FIN / 'V0.npz'); h0 = {(int(w), int(h[0])): (h[1], h[2]) for w, h in zip(Z0['hash_widx'], Z0['hashes'])}
    for nm in info:
        Z1 = np.load(FIN / f'{nm}.npz'); h1 = {(int(w), int(h[0])): (h[1], h[2]) for w, h in zip(Z1['hash_widx'], Z1['hashes'])}
        com = sorted(set(h0) & set(h1)); info[nm]['parton_identical_frac'] = float(np.mean([h0[k] == h1[k] for k in com])) if com else None
        info[nm]['parton_hash_n'] = len(com); log(nm, 'parton level identical to V0 in', info[nm]['parton_identical_frac'], 'of', len(com))
    (RES / f'h57_gen_{tag}.json').write_text(json.dumps(info, indent=1, default=str))


# ----------------------------------------------------------------------------------------------------------------- score
MODELS = ['part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_k', 'pfn_t', 'efn_k', 'efn_t', 'hw_part_kin',
          'hw_pnet_kin', 'perm0', 'perm1']


def samples():
    D = {}
    for nm in final_names(): D[f'F|{nm}'] = [FIN / f'{nm}.npz']
    P = json.loads((RES / 'h57_pilot.json').read_text())
    for nm in P['rows']:
        D[f'P|{nm}'] = [PIL / f'{nm}_0.npz', PIL / f'{nm}_1.npz']
    D['P|nominal'] = [PIL / 'nominal_0.npz', PIL / 'nominal_1.npz']
    return D


def load_sample(paths, rel=True):
    L, y, key, nrel = [], [], [], 0
    for p in paths:
        l, yy, k, Z, n = load_npz(p, rel); L += l; y.append(yy); key.append(k); nrel += n
    return L, np.concatenate(y), np.concatenate(key), nrel


def step_score(a):
    import torch
    import copy
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    g = G(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    M = H.Models(dev)
    for name, arch, pf in (('part_kin', 'part', False), ('part_kinpid', 'part', True), ('pnet_kin', 'pn', False), ('pnet_kinpid', 'pn', True)):
        M.add_weaver(name, arch, pf)
    M.add_lnet()
    for kind in ('efn', 'pfn'):
        H14.add_keras_ef(M, kind)
        net = H.make_ef(kind); net.load_state_dict(torch.load(V / 'models' / 'qg' / f'h5_{kind}_torch.pt', map_location='cpu'))
        M.add_torch(f'{kind}_t', net, json.loads((V / 'models' / 'qg' / f'h5_{kind}_torch.json').read_text()))
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        H14.add_weaver_dir(M, 'hw_' + name, arch, False, V / 'models' / 'qg_herwig' / name)
    for k in range(2):
        kind, arch, pf, n0 = M.m['part_kin']; gg = torch.Generator().manual_seed(k); n2 = copy.deepcopy(n0).cpu()
        with torch.no_grad():
            for prm in n2.parameters():
                fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=gg)])
        M.m[f'perm{k}'] = (kind, arch, pf, n2.eval().to(dev)); M.info[f'perm{k}'] = dict(note=f'part_kin weights permuted, seed {k}')
    T, _ = H14.lrn_tables(); lrn = -T['py']
    Lt, yt, _ = H14.efp_train_lists('py', n_per_class=30000, seed=g.SEED + 2)
    ptd_t = np.array([np.sqrt(((x[:, 0] / x[:, 0].sum()) ** 2).sum()) for x in Lt])
    bp = np.quantile(ptd_t, np.linspace(0, 1, 41)); bp[0], bp[-1] = 0, 1.01; hp = g._lr_hist(ptd_t, yt, bp)
    out = {}; D = samples(); relinfo = {}
    for sname, paths in D.items():
        L, y, key, nrel = load_sample(paths); relinfo[sname] = nrel / max(sum(len(x) for x in L), 1)
        bad = set(np.unique(np.abs(np.concatenate([x[:, 3] for x in L])).astype(int)).tolist()) - VOCAB
        O = g.jet_obs(L); out[f'y|{sname}'] = y.astype(np.int8); out[f'key|{sname}'] = key
        for k in ('N', 'Nch', 'ptd', 'girth', 'm_over_pt', 'C1_1.0'): out[f'obs|{sname}|{k}'] = O[k].astype(np.float32)
        out[f'{sname}|lr_n'] = lrn[np.minimum(O['N'].astype(int), 299)].astype(np.float32)
        out[f'{sname}|lr_ptd'] = hp[np.clip(np.searchsorted(bp, O['ptd'], side='right') - 1, 0, 39)].astype(np.float32)
        for nm in [m for m in MODELS if m in M.m]:
            v = []
            for s in range(0, len(L), 10000):
                Ls = L[s:s + 10000]; v.append(-M.logit(nm, H.from_list(Ls, Lpad=max(len(x) for x in Ls))))
            out[f'{sname}|{nm}'] = np.concatenate(v).astype(np.float32)
        log(sname, len(L), f'jets; relabelled {100 * relinfo[sname]:.2f} % of particles; pids outside vocabulary {sorted(bad)}; acc ' +
            ' '.join(f'{m} {np.mean((out[f"{sname}|{m}"] > 0) == (y == 1)):.4f}' for m in MODELS + ['lr_n', 'lr_ptd'] if f'{sname}|{m}' in out))
    np.savez_compressed(RES / 'h57_scores.npz', **out)
    (RES / 'h57_models.json').write_text(json.dumps(dict(models=M.info, relabelled_frac=relinfo), indent=1, default=str))
    log('wrote h57_scores.npz')


# ------------------------------------------------------------------------------------------------------------------ refs
def step_refs(a):
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h14_qg_generator as H14
    g = G(); pool = Pool(a.cpu)
    Z0 = np.load(FIN / 'V0.npz'); h0 = {(int(w), int(h[0])): (h[1], h[2]) for w, h in zip(Z0['hash_widx'], Z0['hashes'])}; pc = {}
    for nm in final_names():
        Z1 = np.load(FIN / f'{nm}.npz'); h1 = {(int(w), int(h[0])): (h[1], h[2]) for w, h in zip(Z1['hash_widx'], Z1['hashes'])}
        com = sorted(set(h0) & set(h1)); pc[nm] = dict(n=len(com), identical=float(np.mean([h0[k] == h1[k] for k in com])) if com else None,
                                                     jets=int(len(Z1['n'])), mean_N=float(Z1['n'].mean()))
    log('parton-level identity vs V0', pc); (RES / 'h57_gen_parton.json').write_text(json.dumps(pc, indent=1))
    Lt, yt, Nt = H14.efp_train_lists('py', n_per_class=30000, seed=g.SEED + 2)
    ptd_t = np.array([np.sqrt(((x[:, 0] / x[:, 0].sum()) ** 2).sum()) for x in Lt])
    Ft = H14.efps(Lt, pool); q = (yt == 1).astype(int)
    lf = lambda F: np.log(np.maximum(F, 1e-12))
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15, random_state=0)
    clf = dict(gbdt_efp=gb().fit(lf(Ft), q), gbdt_efpn=gb().fit(np.c_[lf(Ft), Nt], q), gbdt_efpptd=gb().fit(np.c_[lf(Ft), ptd_t], q),
               lr_n_ptd=gb().fit(np.c_[Nt, ptd_t], q))
    log('references trained on', len(yt), 'KMT train jets')
    out = {}; feats = {}
    FN = final_names(); S = {f'F|{nm}': load_sample([FIN / f'{nm}.npz'])[:2] for nm in FN}
    S['KMT'] = g.kmt_sample(a.kmt_per_class)
    for sn, (L, y) in S.items():
        O = g.jet_obs(L); F = H14.efps([x[:, :3] for x in L], pool); N = O['N']; p = O['ptd']
        out[f'{sn}|gbdt_efp'] = clf['gbdt_efp'].decision_function(lf(F)).astype(np.float32)
        out[f'{sn}|gbdt_efpn'] = clf['gbdt_efpn'].decision_function(np.c_[lf(F), N]).astype(np.float32)
        out[f'{sn}|gbdt_efpptd'] = clf['gbdt_efpptd'].decision_function(np.c_[lf(F), p]).astype(np.float32)
        out[f'{sn}|lr_n_ptd'] = clf['lr_n_ptd'].decision_function(np.c_[N, p]).astype(np.float32)
        out[f'y|{sn}'] = y.astype(np.int8); feats[sn] = g.c2st_feats(L, O)
        log(sn, 'acc ' + ' '.join(f'{r} {np.mean((out[f"{sn}|{r}"] > 0) == (y == 1)):.4f}' for r in ('gbdt_efp', 'gbdt_efpn', 'gbdt_efpptd', 'lr_n_ptd')))
    np.savez_compressed(RES / 'h57_refs.npz', **out)
    val = dict(c2st={})
    for cls, cn in ((1, 'q'), (0, 'g')):
        f0 = feats['F|V0'][out['y|F|V0'] == cls]
        val['c2st'][f'V0_vs_KMT_{cn}'] = g.c2st(feats['KMT'][out['y|KMT'] == cls], f0)
        for nm in FN[1:]:
            val['c2st'][f'{nm}_vs_V0_{cn}'] = g.c2st(f0, feats[f'F|{nm}'][out[f'y|F|{nm}'] == cls])
            val['c2st'][f'{nm}_vs_KMT_{cn}'] = g.c2st(feats['KMT'][out['y|KMT'] == cls], feats[f'F|{nm}'][out[f'y|F|{nm}'] == cls])
        log('C2ST', cn, {k: round(v['auc'], 3) for k, v in val['c2st'].items() if k.endswith(cn)})
    (RES / 'h57_valid.json').write_text(json.dumps(val, indent=1, default=float)); pool.close(); log('wrote h57_refs.npz, h57_valid.json')


# ----------------------------------------------------------------------------------------------------------------- stats
NETS = ['part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_k', 'pfn_t', 'hw_part_kin', 'hw_pnet_kin']
NULLS = ['perm0', 'perm1']
REFS_S = ['lr_n', 'lr_ptd']
REFS_R = ['gbdt_efp', 'gbdt_efpn', 'gbdt_efpptd', 'lr_n_ptd']
TEST_REFS = ['efn_k', 'lr_n', 'lr_ptd', 'gbdt_efp']


def _p(d, se):
    from scipy.stats import norm
    return float(2 * norm.sf(abs(d) / se)) if se > 0 else 1.0


def _mcnemar(b, c):
    from scipy.stats import binom
    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))


def step_stats(a):
    from interp.h5_qg_counting import holm
    Zs = dict(np.load(RES / 'h57_scores.npz')); Zr = dict(np.load(RES / 'h57_refs.npz'))
    snames = sorted({k[2:] for k in Zs if k.startswith('y|')})
    S = {}
    for sn in snames:
        S[sn] = {m: Zs[f'{sn}|{m}'].astype(np.float64) for m in MODELS + REFS_S if f'{sn}|{m}' in Zs}
        if sn.startswith('F|'):
            assert (Zr[f'y|{sn}'] == Zs[f'y|{sn}']).all(); S[sn].update({m: Zr[f'{sn}|{m}'].astype(np.float64) for m in REFS_R})
    y = {sn: Zs[f'y|{sn}'].astype(int) for sn in snames}; key = {sn: Zs[f'key|{sn}'] for sn in snames}
    obs = {sn: {k: Zs[f'obs|{sn}|{k}'].astype(np.float64) for k in ('N', 'ptd', 'girth', 'm_over_pt', 'C1_1.0')} for sn in snames}
    fm = [m for m in MODELS + REFS_S + REFS_R if m in S['F|V0']]
    R = dict(B=a.boot, models=fm, accuracy={}, thresholds={}, final={}, tests=[], fit={}, pilot_rows={})
    from sklearn.metrics import roc_auc_score
    for sn in [s for s in snames if s.startswith('F|')]:
        R['accuracy'][sn] = {m: dict(acc=float(np.mean((S[sn][m] > 0) == (y[sn] == 1))), auc=float(roc_auc_score(y[sn], S[sn][m])))
                             for m in fm if m in S[sn]}
    # 50 % quark-efficiency threshold per model (median score of V0 quark jets)
    thr = {m: float(np.median(S['F|V0'][m][y['F|V0'] == 1])) for m in fm}; R['thresholds'] = thr
    rng = np.random.default_rng(5757)

    def pairs(s0, s1, cls):
        pos = {int(k): j for j, k in enumerate(key[s0]) if y[s0][j] == cls}
        pr = [(pos[int(k)], j) for j, k in enumerate(key[s1]) if y[s1][j] == cls and int(k) in pos]
        return np.array([p[0] for p in pr], int), np.array([p[1] for p in pr], int)

    def block(s0, s1, cls, models, B, calls=True):
        i0, i1 = pairs(s0, s1, cls); paired = len(i0) > 100
        if paired:
            Dm = np.stack([S[s1][m][i1] - S[s0][m][i0] for m in models], 1); J = len(i0)
            bm = np.empty((B, len(models)))
            for b in range(B):
                c = np.bincount(rng.integers(0, J, J), minlength=J).astype(np.float64); bm[b] = c @ Dm / J
            d = Dm.mean(0)
            orel = {k: float(obs[s1][k][i1].mean() / obs[s0][k][i0].mean() - 1) for k in obs[s0]}
        else:                                   # unpaired (V0b: different seeds): independent bootstrap of both samples
            a0 = np.flatnonzero(y[s0] == cls); a1 = np.flatnonzero(y[s1] == cls); J = min(len(a0), len(a1))
            M0 = np.stack([S[s0][m][a0] for m in models], 1); M1 = np.stack([S[s1][m][a1] for m in models], 1)
            d = M1.mean(0) - M0.mean(0); bm = np.empty((B, len(models)))
            for b in range(B):
                bm[b] = M1[rng.integers(0, len(a1), len(a1))].mean(0) - M0[rng.integers(0, len(a0), len(a0))].mean(0)
            orel = {k: float(obs[s1][k][a1].mean() / obs[s0][k][a0].mean() - 1) for k in obs[s0]}
        r = dict(n_pairs=int(len(i0)) if paired else 0, n=int(J), paired=paired, obs_rel=orel)
        mi = {m: i for i, m in enumerate(models)}
        for m in models:
            i = mi[m]; r[m] = dict(d=float(d[i]), ci=np.percentile(bm[:, i], [2.5, 97.5]).tolist(), se=float(bm[:, i].std()))
            if calls and paired:
                c0 = S[s0][m][i0] > thr[m]; c1 = S[s1][m][i1] > thr[m]
                bq = int((c0 & ~c1).sum()); cq = int((~c0 & c1).sum())
                r[m]['calls'] = dict(changed=float((c0 != c1).mean()), q_to_g=bq, g_to_q=cq, mcnemar_p=_mcnemar(bq, cq))
        for m in models:
            if m in ('efn_k',) + tuple(REFS_S) + tuple(REFS_R): continue
            i = mi[m]
            for ref in TEST_REFS:
                if ref not in mi: continue
                j = mi[ref]; dd = d[i] - d[j]; bd = bm[:, i] - bm[:, j]
                r[m][f'minus_{ref}'] = dict(d=float(dd), ci=np.percentile(bd, [2.5, 97.5]).tolist(), p=_p(dd, bd.std()))
            e = mi['efn_k']
            for ref, nmk in (('lr_n', 'kappa_N'), ('lr_ptd', 'kappa_pTD')):
                j = mi[ref]; den = d[j] - d[e]; bk = (bm[:, i] - bm[:, e]) / (bm[:, j] - bm[:, e])
                r[m][nmk] = dict(k=float((d[i] - d[e]) / den), ci=np.percentile(bk, [2.5, 97.5]).tolist(), den=float(den))
        return r

    FN = final_names()
    for nm in FN[1:]:
        for cls, cn in ((1, 'q'), (0, 'g')):
            r = block('F|V0', f'F|{nm}', cls, fm, a.boot); R['final'].setdefault(nm, {})[cn] = r
            if nm != 'V0b':
                for m in NETS:
                    for ref in TEST_REFS:
                        if f'minus_{ref}' in r[m]:
                            R['tests'].append(dict(setting=nm, cls=cn, model=m, ref=ref, d=r[m][f'minus_{ref}']['d'], p=r[m][f'minus_{ref}']['p']))
            log(nm, cn, f"dN {100 * r['obs_rel']['N']:+.1f}% dpTD {100 * r['obs_rel']['ptd']:+.2f}% girth {100 * r['obs_rel']['girth']:+.2f}% |",
                ' '.join(f"{m} {r[m]['d']:+.3f}" for m in fm))
    adj = holm([t['p'] for t in R['tests']])
    for t, h in zip(R['tests'], adj): t['holm'] = float(h)
    R['holm'] = dict(n_tests=len(R['tests']), n_pass_005=int((adj < 0.05).sum()))
    # alpha / beta fit over all paired settings (pilot vs pilot nominal; final vs V0)
    pm = [m for m in MODELS + REFS_S if m in S['P|nominal']]
    rows = []
    for sn in snames:
        if sn.startswith('P|') and sn not in ('P|nominal', 'P|nominal_b'):
            rr = {cn: block('P|nominal', sn, cls, pm, 300, calls=False) for cls, cn in ((1, 'q'), (0, 'g'))}
            R['pilot_rows'][sn[2:]] = {cn: dict(obs_rel=rr[cn]['obs_rel'], **{m: dict(d=rr[cn][m]['d'], se=rr[cn][m]['se']) for m in pm}) for cn in rr}
            rows.append((sn, rr))
    for nm in FN[2:]:
        rows.append((f'F|{nm}', R['final'][nm]))
    R['fit_rows'] = [s for s, _ in rows]
    for cn in ('q', 'g'):
        XN = np.array([r[cn]['lr_n']['d'] for _, r in rows]); XP = np.array([r[cn]['lr_ptd']['d'] for _, r in rows])
        XE = np.array([r[cn]['efn_k']['d'] for _, r in rows])
        R['fit'].setdefault(cn, {})['regressor_corr'] = dict(lrn_lrptd=float(np.corrcoef(XN, XP)[0, 1]), lrn_efn=float(np.corrcoef(XN, XE)[0, 1]),
                                                             lrptd_efn=float(np.corrcoef(XP, XE)[0, 1]), n_settings=len(rows))
        for m in NETS + NULLS + ['efn_k', 'efn_t']:
            yv = np.array([r[cn][m]['d'] for _, r in rows]); se = np.array([r[cn][m]['se'] for _, r in rows]); w = 1 / np.maximum(se, 1e-3) ** 2
            res = {}
            for fn, X in (('N_ptd', np.c_[XN, XP]), ('efn_N_ptd', np.c_[XE, XN, XP])):
                def fit(ix):
                    sw = np.sqrt(w[ix]); return np.linalg.lstsq(X[ix] * sw[:, None], yv[ix] * sw, rcond=None)[0]
                c = fit(np.arange(len(yv))); r2 = np.random.default_rng(7)
                bs = np.array([fit(r2.integers(0, len(yv), len(yv))) for _ in range(2000)])
                pred = X @ c; chi2 = float((w * (yv - pred) ** 2).sum())
                res[fn] = dict(coef=c.tolist(), ci=[np.percentile(bs[:, j], [2.5, 97.5]).tolist() for j in range(X.shape[1])],
                               chi2=chi2, ndf=len(yv) - X.shape[1], R2=float(1 - chi2 / (w * (yv - np.average(yv, weights=w)) ** 2).sum()))
            R['fit'][cn][m] = res
            log('fit', cn, m, {k: [round(x, 3) for x in v['coef']] for k, v in res.items()})
    (RES / 'h57_run.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h57_run.json')


# ------------------------------------------------------------------------------------------------------------------- fig
LAB = dict(part_kin='ParT_kin', part_kinpid='ParT_kinpid', pnet_kin='ParticleNet_kin', pnet_kinpid='ParticleNet_kinpid',
           lorentznet='LorentzNet', pfn_k='PFN', pfn_t='PFN (2nd run)', efn_k='EFN (IRC-safe)', efn_t='EFN (2nd run)',
           hw_part_kin='ParT_kin [Herwig-trained]', hw_pnet_kin='ParticleNet_kin [Herwig-trained]', perm0='permuted ParT_kin',
           lr_n='LR(N)', lr_ptd='LR($p_T^D$)', lr_n_ptd='LR(N, $p_T^D$)', gbdt_efp='GBDT(EFP)', gbdt_efpn='GBDT(EFP+N)',
           gbdt_efpptd='GBDT(EFP+$p_T^D$)')
SLAB = dict(NAp='NA+ (N up, $p_T^D$ fixed)', NAm='NA− (N down, $p_T^D$ unchanged)', PAp='PA+ ($p_T^D$ up, N fixed)',
            PAm='PA− ($p_T^D$ down while N also falls)', IRp='IR+ (IRC-safe only)', Hzp='Hz+ (H54: N and $p_T^D$ together)', V0b='V0b (2nd nominal)')


def step_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((RES / 'h57_run.json').read_text()); P = json.loads((RES / 'h57_pilot.json').read_text())
    # 1: settings map dN/N vs dpT^D/pT^D (pilot, average of q and g), chosen marked; and final obs shifts
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6))
    for nm, r in P['rows'].items():
        if not r['q']['paired']: continue
        x = 0.5 * (r['q']['N'] + r['g']['N']); yv = 0.5 * (r['q']['ptd'] + r['g']['ptd']); e = 0.5 * (r['q']['efn'] + r['g']['efn'])
        kind = 'decay' if nm.startswith('D:') else ('solved' if '#' in nm else 'single')
        ax.scatter(x, yv, c=[e], cmap='coolwarm', vmin=-0.4, vmax=0.4, s=40, marker=dict(decay='s', solved='^', single='o')[kind], edgecolors='k', lw=0.4)
    for tn, c in P['chosen'].items():
        if not c.get('feasible'): continue
        nm = c['name']; ax.annotate(tn, (c['mean_dN'], c['mean_dptd']), xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')
    xs = np.linspace(-15, 15, 3); ax.plot(xs, P['comove'] * xs, 'k--', lw=0.8, label=f'string-knob co-movement ($\\Delta p_T^D$ = {P["comove"]:.2f} ΔN)')
    ax.axhline(0, color='0.7', lw=0.5); ax.axvline(0, color='0.7', lw=0.5)
    ax.set_xlabel('ΔN / N (%), paired with nominal, mean of quark and gluon jets'); ax.set_ylabel('Δ$p_T^D$ / $p_T^D$ (%)')
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(-0.4, 0.4)); fig.colorbar(sm, ax=ax, label='ΔEFN quark log-odds')
    ax.scatter([], [], marker='o', c='w', edgecolors='k', label='one string/flavour knob'); ax.scatter([], [], marker='s', c='w', edgecolors='k', label='one decay switched off')
    ax.scatter([], [], marker='^', c='w', edgecolors='k', label='solved combination'); ax.legend(fontsize=7.5, loc='upper right')
    ax.set_title('H57 pilot: generator settings separate particle count from momentum sharing\n(2,000 q + 2,000 g jets per setting, same parton-level events)', fontsize=9.5)
    fig.tight_layout(); fig.savefig(RES / 'h57_settings.png', dpi=130); plt.close(fig)
    # 2: shifts
    rows = ['part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_k', 'hw_part_kin', 'hw_pnet_kin', 'perm0', 'efn_k',
            'gbdt_efp', 'lr_n', 'lr_ptd', 'gbdt_efpn', 'gbdt_efpptd']
    sets = [v for v in ['NAm', 'NAp', 'PAp', 'PAm', 'IRp', 'Hzp', 'V0b'] if v in R['final']]
    fig, axs = plt.subplots(1, len(sets), figsize=(2.6 * len(sets) + 1.5, 6.2), sharey=True)
    for k, v in enumerate(sets):
        ax = axs[k]
        for o, (cn, c) in enumerate((('q', '#1f77b4'), ('g', '#d62728'))):
            for i, m in enumerate(rows):
                r = R['final'][v][cn][m]
                ax.errorbar(r['d'], i + (o - 0.5) * 0.3, xerr=[[r['d'] - r['ci'][0]], [r['ci'][1] - r['d']]], fmt='o', ms=3, color=c,
                            label=('quark jets' if cn == 'q' else 'gluon jets') if i == 0 else None)
        ax.axvline(0, color='k', lw=0.6); ax.axhline(8.5, color='0.6', lw=0.6)
        orq, org = R['final'][v]['q']['obs_rel'], R['final'][v]['g']['obs_rel']
        ax.set_title(f"{SLAB[v]}\nΔN/N {100 * orq['N']:+.1f} / {100 * org['N']:+.1f} %\nΔ$p_T^D$ {100 * orq['ptd']:+.1f} / {100 * org['ptd']:+.1f} % (q / g)", fontsize=7.5)
        ax.set_xlabel('Δs (quark log-odds)', fontsize=8)
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([LAB[m] for m in rows], fontsize=8); axs[0].invert_yaxis(); axs[0].legend(fontsize=7)
    fig.suptitle('H57: mean paired score shift per tagger (95 % CI, 10,000 q + 10,000 g jets per setting); networks above the grey line', fontsize=10)
    fig.tight_layout(); fig.savefig(RES / 'h57_shifts.png', dpi=120); plt.close(fig)
    # 3: one real gluon jet under V0, NA-, PA+
    Zs = dict(np.load(RES / 'h57_scores.npz'))
    show = ['V0'] + [v for v in ['NAm', 'NAp', 'PAp', 'PAm'] if v in R['final']]
    Z = {v: dict(np.load(FIN / f'{v}.npz')) for v in show}            # load arrays once (NpzFile re-reads on every access)
    keys = {v: Z[v]['widx'].astype(np.int64) * 100_000 + Z[v]['event'] for v in show}
    com = set(keys['V0'][Z['V0']['y'] == 0].tolist())
    for v in show[1:]: com &= set(keys[v].tolist())
    com = sorted(com)
    # example: the gluon jet whose N and pT^D changes are closest to the class-mean changes of each setting
    def nptd(v, k):
        j = pos[v][k]; x = Z[v]['X'][j, :Z[v]['n'][j]]; return len(x), np.sqrt((x[:, 0] ** 2).sum()) / x[:, 0].sum()
    pos = {v: {int(k): j for j, k in enumerate(keys[v])} for v in show}
    tgt = {v: (R['final'][v]['g']['obs_rel']['N'], R['final'][v]['g']['obs_rel']['ptd']) for v in show[1:]}
    best = None
    for k in com[:3000]:
        n0, p0 = nptd('V0', k)
        if not 40 <= n0 <= 70: continue
        c = 0.0
        for v in show[1:]:
            n1, p1 = nptd(v, k); c += ((n1 / n0 - 1 - tgt[v][0]) / 0.02) ** 2 + ((p1 / p0 - 1 - tgt[v][1]) / 0.01) ** 2
        if best is None or c < best[0]: best = (c, k)
    kx = best[1]
    fig, axs = plt.subplots(1, len(show), figsize=(3.6 * len(show), 4.4))
    for c, v in enumerate(show):
        ax = axs[c]; j = int(np.flatnonzero(keys[v] == kx)[0]); x = Z[v]['X'][j, :Z[v]['n'][j]]; jy, jphi = Z[v]['jet'][j, 1], Z[v]['jet'][j, 2]
        z = x[:, 0] / x[:, 0].sum(); ptd = np.sqrt((x[:, 0] ** 2).sum()) / x[:, 0].sum()
        ax.scatter(x[:, 1] - jy, x[:, 2] - jphi, s=3 + 500 * z, c='C0' if v == 'V0' else 'C3', alpha=0.6, edgecolors='none')
        ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.45, 0.45); ax.set_aspect('equal')
        sc = Zs[f'F|{v}|part_kin'][j]; e = Zs[f'F|{v}|efn_k'][j]
        ax.set_title(f"{'V0 nominal' if v == 'V0' else SLAB[v]}\nN = {len(x)},  $p_T^D$ = {ptd:.3f}\nParT_kin s {sc:+.2f}, EFN s {e:+.2f}", fontsize=8)
        ax.set_xlabel('Δy to jet axis', fontsize=8)
        if c == 0: ax.set_ylabel('Δφ to jet axis', fontsize=8)
    fig.suptitle('H57: one gluon-jet parton-level event, hadronised under each setting\n(marker area ∝ particle pT; event whose N and $p_T^D$ changes are near the setting means)', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(RES / 'h57_jet.png', dpi=120); plt.close(fig); log('wrote figures')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['pilot', 'gen', 'score', 'refs', 'stats', 'fig'])
    ap.add_argument('--pilot_jets', type=int, default=2000); ap.add_argument('--jets', type=int, default=10000)
    ap.add_argument('--cpu', type=int, default=16); ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--ridge', type=float, default=0.5); ap.add_argument('--per_target', type=int, default=3)
    ap.add_argument('--kmt_per_class', type=int, default=10000); ap.add_argument('--only', default='')
    ap.add_argument('--pick', default=''); ap.add_argument('--pick_note', default='')
    a = ap.parse_args()
    dict(pilot=step_pilot, gen=step_gen, score=step_score, refs=step_refs, stats=step_stats, fig=step_fig)[a.step](a)
