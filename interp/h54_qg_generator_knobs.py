"""H54 (loop D; claim D7 restated by the reviewer): do quark/gluon taggers respond to constituent multiplicity N beyond what
IRC-safe energy flow predicts, under an on-manifold, generator-level change of the jets?

Generator-level intervention (Pythia 8, PyPI pythia8mc, in the venv /j-jepa-vol/aaron-interp/env-gen over the image's conda
python). Settings reproduce the Komiske-Metodiev-Thaler (KMT) QG sample (energyflow docs: Pythia 8.226, default tune and shower
parameters = Monash 2013, pp at sqrt(s) = 14 TeV, q qbar -> Z(nu nu) + g and q g -> Z(nu nu) + q(uds), hadronisation and MPI
on, all final-state non-neutrino particles clustered into anti-kT R = 0.4 jets, jet pT in [500, 550] GeV, |y| < 1.7, particles
(pt, y, phi, pid) with phi within pi of the jet). Choices not stated by KMT, made here: PhaseSpace:pTHatMin 450, pTHatMax 700
GeV; only the hardest jet of an event is kept; quark sample PDFinProcess:nQuarkIn = 3 plus a check that the outgoing quark is
u, d or s; clustering with Pythia's SlowJet in fjcore mode (the FastJet core code), visible final-state particles, real masses.

Variants (same per-event random seeds in every variant: before event i of worker w the generator is reseeded with
seed(w, i), so every variant starts each event from the same random state; a hadronisation-only change therefore leaves the
hard process, the parton shower and MPI identical event by event, and changes only how the partons turn into hadrons)
  V0   nominal Monash
  Hz+  hadronisation only (StringZ:aLund, StringZ:bLund, StringPT:sigma), chosen in the pilot so that <N> rises by ~10 %
       with the smallest change of the IRC-safe EFN output
  Hz-  the same, <N> lowered by ~10 %
  S+   TimeShower:alphaSvalue = nominal + 0.01 (final-state shower emits more: N and IRC-safe shapes both move)
  S-   nominal - 0.01

Steps
  --step check   (CPU, env-gen) import check, API check (rndm.init, SlowJet), 20-event timing, reseed reproducibility
  --step pilot   (CPU 8, env-gen) candidate hadronisation settings, 1000 q + 1000 g jets each, paired by event with V0:
                 Delta<N>/<N>, paired shifts of IRC-safe shapes and of the KMT-trained EFN logit -> physics/h54_pilot.json
  --step gen     (CPU 8, env-gen) --variant V0|Hzp|Hzm|Sp|Sm: 30,000 q + 30,000 g jets -> data/qg/gen/<variant>.npz
  --step score   (GPU, env-part) every model on the five variants and on 30,000 + 30,000 KMT test jets -> physics/h54_scores.npz
  --step refs    (CPU 8, venv-ef) references trained on KMT train jets, observables, C2ST -> physics/h54_refs.npz, h54_valid.json
  --step stats   (CPU) shifts, CIs, tests, reweighting check -> physics/h54_run.json
  --step fig     (CPU) physics/h54_variants.png, physics/h54_shifts.png
Score convention: s = log p(quark) - log p(gluon) (quark log-odds; positive = toward quark).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
GEN = V / 'data' / 'qg' / 'gen'
SEED = 5401
PILOT_SEED = 5501
WORKERS = 4                    # per class in the main generation
VARIANTS = ['V0', 'Hzp', 'Hzm', 'Sp', 'Sm']
VLAB = dict(V0='V0 nominal', Hzp='Hz+ (hadronisation, N up)', Hzm='Hz- (hadronisation, N down)',
            Sp='S+ (FSR alpha_s +0.01)', Sm='S- (FSR alpha_s -0.01)', KMT='KMT test')
NOMINAL = {'StringZ:aLund': 0.68, 'StringZ:bLund': 0.98, 'StringPT:sigma': 0.335}   # Monash 2013 (checked at run time)

# pilot candidates: (aLund, bLund, sigma); a up / b down / sigma down -> more hadrons
CANDIDATES = {
    'a+0.3': (0.98, 0.98, 0.335), 'a+0.6': (1.28, 0.98, 0.335), 'b-0.25': (0.68, 0.73, 0.335), 'b-0.45': (0.68, 0.53, 0.335),
    'ab+0.4': (1.08, 1.18, 0.335) , 'ab-0.3': (0.38, 0.68, 0.335), 's-0.06': (0.68, 0.98, 0.275),
    'a+0.3,b-0.2,s-0.04': (0.98, 0.78, 0.295),
    'a-0.3': (0.38, 0.98, 0.335), 'a-0.5': (0.18, 0.98, 0.335), 'b+0.4': (0.68, 1.38, 0.335), 'b+0.8': (0.68, 1.78, 0.335),
    's+0.07': (0.68, 0.98, 0.405), 'a-0.3,b+0.3,s+0.04': (0.38, 1.28, 0.375),
}
# round 2 (after round 1: a/b changes move the EFN output opposite to sigma changes; combinations solved linearly from the
# round-1 table so that the EFN shift cancels at |dN/N| ~ 0.10)
CANDIDATES2 = {
    'a+0.11,s-0.088': (0.79, 0.98, 0.247), 'a+0.15,s-0.10': (0.83, 0.98, 0.235), 'a+0.08,s-0.08': (0.76, 0.98, 0.255),
    's-0.10': (0.68, 0.98, 0.235), 'a+0.11,b-0.1,s-0.088': (0.79, 0.88, 0.247),
    'a-0.2,s+0.103': (0.48, 0.98, 0.438), 'a-0.25,s+0.095': (0.43, 0.98, 0.43), 'a-0.15,s+0.115': (0.53, 0.98, 0.45),
    's+0.115': (0.68, 0.98, 0.45), 'a-0.2,b+0.15,s+0.085': (0.48, 1.13, 0.42),
}


# ------------------------------------------------------------------------------------------------------------- generator
def _event_seed(widx, i):
    return 1 + (SEED + widx) * 50_000 + i if widx < 1000 else 1 + widx * 50_000 + i


def _py8():
    try:
        import pythia8mc as m
    except ImportError:
        import pythia8 as m
    return m


def make_pythia(cls, extra, seed):
    pythia8 = _py8()
    p = pythia8.Pythia('', False)
    base = ['Beams:eCM = 14000.', 'Tune:pp = 14', 'PartonLevel:MPI = on', 'PartonLevel:ISR = on', 'PartonLevel:FSR = on',
            'HadronLevel:all = on', 'WeakZ0:gmZmode = 2', '23:onMode = off', '23:onIfAny = 12 14 16',
            'PhaseSpace:pTHatMin = 450.', 'PhaseSpace:pTHatMax = 700.', 'Next:numberCount = 0', 'Next:numberShowEvent = 0',
            'Next:numberShowProcess = 0', 'Next:numberShowInfo = 0', 'Init:showChangedSettings = off',
            'Init:showProcesses = off', 'Init:showChangedParticleData = off', 'Random:setSeed = on', f'Random:seed = {seed}']
    base += ['WeakBosonAndParton:qqbar2gmZg = on'] if cls == 0 else ['WeakBosonAndParton:qg2gmZq = on', 'PDFinProcess:nQuarkIn = 3']
    for s in base: assert p.readString(s), s
    for k, v in extra.items(): assert p.readString(f'{k} = {v}'), k
    assert p.init()
    return p


def variant_settings(variant, pilot=None):
    if variant == 'V0': return {}
    if variant in ('Sp', 'Sm'):
        return {'TimeShower:alphaSvalue': round(0.1365 + (0.01 if variant == 'Sp' else -0.01), 4)}
    pilot = pilot or json.loads((RES / 'h54_pilot.json').read_text())
    a, b, s = pilot['chosen'][variant]['setting']
    return {'StringZ:aLund': a, 'StringZ:bLund': b, 'StringPT:sigma': s}


def _jet_from_event(p, sj):
    """hardest anti-kT R=0.4 jet of the visible final state if it passes 500 < pT < 550, |y| < 1.7 -> (n,4) (pt, y, phi, pid)"""
    ev = p.event
    if not sj.analyze(ev) or sj.sizeJet() == 0: return None, None
    pt, y, phi = sj.pT(0), sj.y(0), sj.phi(0)
    if not (500 <= pt <= 550 and abs(y) < 1.7): return None, (pt, y)
    rows = []
    for i in sj.constituents(0):
        q = ev[i]; dphi = (q.phi() - phi + np.pi) % (2 * np.pi) - np.pi
        rows.append((q.pT(), q.y(), phi + dphi, q.id()))
    a = np.array(rows, np.float64); a = a[np.argsort(-a[:, 0], kind='stable')]
    return a, (pt, y, phi, sj.m(0))


def _parton_hash(p):
    """sum of |px| + 2|pz| over all shower / MPI / beam-remnant entries (|status| 21-79 excluding hadronisation 81+)"""
    ev = p.event; h = 0.0; n = 0
    for i in range(ev.size()):
        st = abs(ev[i].status())
        if 21 <= st <= 69: h += abs(ev[i].px()) + 2 * abs(ev[i].pz()); n += 1
    return round(h, 6), n


def gen_worker(task):
    """task: dict(cls, widx, extra, n_jets, max_events, out, hash_events) -> writes npz, returns summary"""
    pythia8 = _py8()
    cls, widx, extra = task['cls'], task['widx'], task['extra']
    p = make_pythia(cls, extra, _event_seed(widx, 0))
    sj = pythia8.SlowJet(-1, 0.4, 400.0, 20.0, 2, 2)
    reseed = hasattr(p, 'rndm') and hasattr(p.rndm, 'init')
    L, meta, evi, hashes = [], [], [], []
    nfail = nflav = nev = 0; t0 = time.time()
    for i in range(task['max_events']):
        if len(L) >= task['n_jets']: break
        if reseed: p.rndm.init(_event_seed(widx, i))
        nev += 1
        if not p.next(): nfail += 1; continue
        if cls == 1:
            pr = p.process; out_q = [pr[j].idAbs() for j in range(pr.size()) if pr[j].status() == 23 and pr[j].idAbs() != 23]
            if not out_q or out_q[0] > 3: nflav += 1; continue
        if i < task.get('hash_events', 0): hashes.append((i,) + _parton_hash(p))
        a, m = _jet_from_event(p, sj)
        if a is None: continue
        L.append(a); meta.append(m); evi.append(i)
    n = np.array([len(a) for a in L]); Lmax = int(n.max()) if len(n) else 1
    X = np.zeros((len(L), Lmax, 4), np.float32)
    for j, a in enumerate(L): X[j, :len(a)] = a
    dt = time.time() - t0
    np.savez_compressed(task['out'], X=X, n=n, y=np.full(len(L), cls, np.int8), event=np.array(evi, np.int64),
                        widx=np.full(len(L), widx, np.int32), jet=np.array(meta, np.float64).reshape(-1, 4),
                        hashes=np.array(hashes, np.float64).reshape(-1, 3))
    s = dict(cls=cls, widx=widx, jets=len(L), events=nev, failed=nfail, flavour_rejected=nflav, reseed_per_event=reseed,
             eff=len(L) / max(nev, 1), sec=dt, ms_per_event=1e3 * dt / max(nev, 1), mean_N=float(n.mean()) if len(n) else None,
             settings=extra)
    print(f'[worker {widx} cls {cls}] {s}', flush=True)
    return s


def load_gen(variant, root=GEN):
    """-> list of (n,4) (pt, y, phi, pid) arrays, labels (1 = quark), event index, worker index"""
    Z = np.load(root / f'{variant}.npz')
    X, n = Z['X'], Z['n']
    return [X[j, :n[j]].astype(np.float64) for j in range(len(n))], Z['y'].astype(int), Z['event'], Z['widx']


# ----------------------------------------------------------------------------------------------------------------- check
def step_check(a):
    pythia8 = _py8()
    log('pythia8 module', pythia8.__file__)
    try:
        import importlib.metadata as md
        log('pythia8mc version', md.version('pythia8mc'))
        try: log('fastjet version', md.version('fastjet'))
        except Exception as e: log('fastjet not installed', e)
    except Exception as e: log('metadata', e)
    p = make_pythia(0, {}, 123)
    sett = {k: p.settings.parm(k) for k in ('StringZ:aLund', 'StringZ:bLund', 'StringPT:sigma', 'TimeShower:alphaSvalue',
                                            'SpaceShower:alphaSvalue', 'MultipartonInteractions:pT0Ref')}
    log('nominal settings', sett, 'Tune:pp', p.settings.mode('Tune:pp'))
    log('has rndm', hasattr(p, 'rndm'), 'rndm.init', hasattr(getattr(p, 'rndm', None), 'init'))
    sj = pythia8.SlowJet(-1, 0.4, 400.0, 20.0, 2, 2); log('SlowJet ok')
    t = time.time(); got = 0
    for i in range(40):
        p.rndm.init(1000 + i); p.next(); jet, m = _jet_from_event(p, sj); got += jet is not None
    log(f'40 events {1e3 * (time.time() - t) / 40:.1f} ms/event, {got} jets kept')
    # reproducibility: same seed, two instances with different hadronisation -> same parton hash
    pA = make_pythia(0, {}, 7); pB = make_pythia(0, {'StringZ:bLund': 0.6}, 7); same = 0
    for i in range(10):
        pA.rndm.init(500 + i); pB.rndm.init(500 + i); pA.next(); pB.next()
        hA, hB = _parton_hash(pA), _parton_hash(pB); same += hA == hB
        nA = sum(1 for k in range(pA.event.size()) if pA.event[k].isFinal()); nB = sum(1 for k in range(pB.event.size()) if pB.event[k].isFinal())
        log(f'event {i}: parton hash equal {hA == hB} ({hA[1]} vs {hB[1]} partons), final particles {nA} vs {nB}')
    log(f'parton level identical in {same}/10 events with bLund 0.98 vs 0.6')
    try:
        import fastjet
        # cross-check SlowJet(fjcore) vs fastjet on 5 events
        jd = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
        for i in range(5):
            p.rndm.init(900 + i); p.next(); ev = p.event
            pj = [fastjet.PseudoJet(ev[k].px(), ev[k].py(), ev[k].pz(), ev[k].e()) for k in range(ev.size()) if ev[k].isFinal() and ev[k].isVisible()]
            cs = fastjet.ClusterSequence(pj, jd); js = sorted(cs.inclusive_jets(400.0), key=lambda j: -j.pt())
            sj.analyze(ev)
            log(f'fastjet vs SlowJet leading pT {js[0].pt() if js else None} vs {sj.pT(0) if sj.sizeJet() else None}')
    except Exception as e:
        log('fastjet cross-check failed', repr(e))
    try:
        import torch; log('torch', torch.__version__)
    except Exception as e: log('torch missing', e)


# ----------------------------------------------------------------------------------------------------------------- pilot
def jet_obs(L):
    """per jet: N, N_ch, pT^D, girth, mass/pT, C1^(0.5, 1, 2) (E-scheme axis, massless constituents as the dataset conversion)"""
    from interp import h5_qg_counting as H
    out = {}
    for s in range(0, len(L), 5000):
        Ls = L[s:s + 5000]; Jt = H.from_list(Ls, Lpad=max(len(x) for x in Ls)); O = H.shapes(Jt)
        jpt = Jt['pt'].sum(1)
        O['m_over_pt'] = O['mass'] / jpt; O['jpt'] = jpt
        for k, v in O.items(): out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def efn_scores(L, dev='cpu'):
    """KMT-trained EFN (H2 energyflow, ported) quark log-odds"""
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    M = H.Models(dev); H14.add_keras_ef(M, 'efn')
    out = []
    for s in range(0, len(L), 5000):
        Ls = L[s:s + 5000]; out.append(-M.logit('efn_k', H.from_list(Ls, Lpad=max(len(x) for x in Ls))))
    return np.concatenate(out)


def step_pilot(a):
    from multiprocessing import Pool
    RES.mkdir(parents=True, exist_ok=True); tmp = GEN / 'pilot'; tmp.mkdir(parents=True, exist_ok=True)
    cands = CANDIDATES if a.round == 1 else CANDIDATES2
    sets = {'nominal': (0.68, 0.98, 0.335), **cands}
    tasks = []
    for name, (aa, bb, ss) in sets.items():
        extra = {} if name == 'nominal' else {'StringZ:aLund': aa, 'StringZ:bLund': bb, 'StringPT:sigma': ss}
        for cls in (0, 1):
            tasks.append(dict(cls=cls, widx=PILOT_SEED + cls, extra=extra, n_jets=a.pilot_jets, max_events=50_000,
                              out=str(tmp / f'{name}_{cls}.npz'), hash_events=200, name=name))
    with Pool(a.cpu) as pool: summ = pool.map(gen_worker, tasks)
    for t, s in zip(tasks, summ): s['name'] = t['name']
    R = dict(settings=sets, summaries=summ, jets_per_class=a.pilot_jets, rows={})
    prev = json.loads((RES / 'h54_pilot.json').read_text()) if a.round == 2 else None
    data = {}
    for name in sets:
        for cls in (0, 1):
            Z = np.load(tmp / f'{name}_{cls}.npz'); X, n = Z['X'], Z['n']
            L = [X[j, :n[j]].astype(np.float64) for j in range(len(n))]
            data[(name, cls)] = dict(L=L, event=Z['event'], hashes=Z['hashes'])
    allL = [x for k in data for x in data[k]['L']]; S = efn_scores(allL); O = jet_obs(allL); o = 0
    for k in data:
        m = len(data[k]['L']); data[k]['efn'] = S[o:o + m]; data[k]['obs'] = {q: v[o:o + m] for q, v in O.items()}; o += m
    keys = ['N', 'ptd', 'girth', 'm_over_pt', 'C1_0.5', 'C1_1.0', 'C1_2.0', 'efn']
    for name in sets:
        row = {}
        for cls, cn in ((0, 'g'), (1, 'q')):
            d0, d1 = data[('nominal', cls)], data[(name, cls)]
            h0 = {int(h[0]): (h[1], h[2]) for h in d0['hashes']}; h1 = {int(h[0]): (h[1], h[2]) for h in d1['hashes']}
            common_h = sorted(set(h0) & set(h1)); same = np.mean([h0[i] == h1[i] for i in common_h]) if common_h else None
            ev0 = {int(e): j for j, e in enumerate(d0['event'])}; pairs = [(ev0[int(e)], j) for j, e in enumerate(d1['event']) if int(e) in ev0]
            i0 = np.array([p[0] for p in pairs]); i1 = np.array([p[1] for p in pairs])
            get = lambda d, q: d['efn'] if q == 'efn' else d['obs'][q]
            r = dict(n=len(d1['L']), n_paired=len(pairs), parton_level_identical_frac=same,
                     N_mean=float(get(d1, 'N').mean()), dN_rel_unpaired=float(get(d1, 'N').mean() / get(d0, 'N').mean() - 1))
            for q in keys:
                x0, x1 = get(d0, q), get(d1, q); sd = float(x0.std())
                dd = x1[i1] - x0[i0] if len(pairs) else np.array([np.nan])
                r[q] = dict(mean=float(x1.mean()), d_unpaired=float(x1.mean() - x0.mean()), d_paired=float(dd.mean()),
                            d_paired_se=float(dd.std() / np.sqrt(max(len(dd), 1))), d_paired_over_sd=float(dd.mean() / sd) if sd > 0 else None)
            r['dN_rel_paired'] = float(r['N']['d_paired'] / get(d0, 'N')[i0].mean()) if len(pairs) else None
            row[cn] = r
        R['rows'][name] = row
        log(name, ' | '.join(f"{cn}: N {row[cn]['N_mean']:.2f} dN/N {row[cn]['dN_rel_paired']:+.3f} d_efn {row[cn]['efn']['d_paired']:+.4f} "
                             f"d_girth/sd {row[cn]['girth']['d_paired_over_sd']:+.3f} d_m/sd {row[cn]['m_over_pt']['d_paired_over_sd']:+.3f} "
                             f"parton-same {row[cn]['parton_level_identical_frac']}" for cn in ('q', 'g')))
    # choice: |dN/N| (mean of the two classes) within 0.07-0.14 in the wanted direction, minimise max over classes of
    # |paired EFN shift| / |dN/N| (EFN shift per unit relative multiplicity change); fall back to the closest dN/N
    if prev is not None:                       # same seeds: round-1 nominal is identical; keep its rows and summaries
        R['round1'] = dict(chosen=prev['chosen'], summaries=prev['summaries'])
        for k, v in prev['rows'].items():
            if k != 'nominal': R['rows'][k] = v
        for k, v in prev['settings'].items(): R['settings'][k] = v
        sets = R['settings']
    chosen = {}
    for vname, sgn in (('Hzp', +1), ('Hzm', -1)):
        cand = []
        for name in [k for k in R['rows'] if k != 'nominal']:
            rr = R['rows'][name]; dn = 0.5 * (rr['q']['dN_rel_paired'] + rr['g']['dN_rel_paired'])
            if sgn * dn <= 0: continue
            cost = max(abs(rr['q']['efn']['d_paired']), abs(rr['g']['efn']['d_paired'])) / abs(dn)
            cand.append((0 if 0.07 <= sgn * dn <= 0.14 else 1, cost if 0.07 <= sgn * dn <= 0.14 else abs(sgn * dn - 0.10), name, dn))
        cand.sort(); c = cand[0]
        chosen[vname] = dict(name=c[2], setting=list(sets[c[2]]), dN_rel=c[3], efn_shift_per_dNrel=c[1] if c[0] == 0 else None,
                             in_window=c[0] == 0, ranking=[x[2] for x in cand])
        log('chosen', vname, chosen[vname])
    R['chosen'] = chosen
    R['rule'] = ('among candidates whose mean relative multiplicity change (paired with V0 event by event, average of quark and '
                 'gluon) lies in 0.07-0.14 in the wanted direction, the one with the smallest max-over-class |paired EFN '
                 'quark-log-odds shift| / |dN/N|')
    (RES / 'h54_pilot.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h54_pilot.json')


# ------------------------------------------------------------------------------------------------------------------- gen
def step_gen(a):
    from multiprocessing import Pool
    GEN.mkdir(parents=True, exist_ok=True); tmp = GEN / f'tmp_{a.variant}'; tmp.mkdir(exist_ok=True)
    extra = variant_settings(a.variant); log(a.variant, 'settings', extra)
    per = int(np.ceil(a.jets / WORKERS)); tasks = []
    for cls in (0, 1):
        for w in range(WORKERS):
            tasks.append(dict(cls=cls, widx=cls * WORKERS + w, extra=extra, n_jets=per, max_events=49_999,
                              out=str(tmp / f'{cls}_{w}.npz'), hash_events=0))
    with Pool(a.cpu) as pool: summ = pool.map(gen_worker, tasks)
    Xs, ns, ys, evs, ws, jets = [], [], [], [], [], []
    for t in tasks:
        Z = np.load(t['out']); Xs.append(Z['X']); ns.append(Z['n']); ys.append(Z['y']); evs.append(Z['event']); ws.append(Z['widx']); jets.append(Z['jet'])
    Lmax = max(x.shape[1] for x in Xs)
    X = np.concatenate([np.pad(x, ((0, 0), (0, Lmax - x.shape[1]), (0, 0))) for x in Xs])
    np.savez_compressed(GEN / f'{a.variant}.npz', X=X, n=np.concatenate(ns), y=np.concatenate(ys), event=np.concatenate(evs),
                        widx=np.concatenate(ws), jet=np.concatenate(jets))
    info = dict(variant=a.variant, settings=extra, workers=summ, jets=int(len(X)), Lmax=int(Lmax),
                pythia_fixed=dict(note='see module docstring'))
    (GEN / f'{a.variant}.json').write_text(json.dumps(info, indent=1, default=float))
    for t in tasks: os.remove(t['out'])
    log('wrote', GEN / f'{a.variant}.npz', len(X), 'jets', 'Lmax', Lmax)


# ----------------------------------------------------------------------------------------------------------------- score
def kmt_sample(per_class, seed=SEED + 1):
    from interp import h14_qg_generator as H14
    L, y, info = H14.load_sample('py', per_class, seed)
    return L, y


def all_samples(per_class_kmt=30000):
    S = {}
    for v in VARIANTS:
        L, y, ev, w = load_gen(v); S[v] = (L, y)
    S['KMT'] = kmt_sample(per_class_kmt)
    return S


def step_score(a):
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
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
    import copy
    for k in range(6):                                     # six weight-permuted ParT_kin nulls (seeds 0-5)
        kind, arch, pf, net = M.m['part_kin']; g = torch.Generator().manual_seed(k); net2 = copy.deepcopy(net).cpu()
        with torch.no_grad():
            for prm in net2.parameters():
                fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=g)])
        M.m[f'perm{k}'] = (kind, arch, pf, net2.eval().to(dev)); M.info[f'perm{k}'] = dict(note=f'part_kin with weights permuted, seed {k}')
    names = list(M.m); log('models', names)
    S = all_samples(a.kmt_per_class); out = {}
    for sname, (L, y) in S.items():
        out[f'y|{sname}'] = y.astype(np.int8); nfull = np.array([len(x) for x in L]); out[f'n|{sname}'] = nfull
        log(sname, len(L), 'jets; N>128', int((nfull > 128).sum()), 'N>148', int((nfull > 148).sum()))
        for nm in names:
            v = []
            for s in range(0, len(L), 10000):
                Ls = L[s:s + 10000]; v.append(-M.logit(nm, H.from_list(Ls, Lpad=max(len(x) for x in Ls))))
            out[f'{sname}|{nm}'] = np.concatenate(v).astype(np.float32)
        log(sname, '; '.join(f'{nm} acc {np.mean((out[f"{sname}|{nm}"] > 0) == (y == 1)):.4f} '
                             f'<s>q {out[f"{sname}|{nm}"][y == 1].mean():+.3f} g {out[f"{sname}|{nm}"][y == 0].mean():+.3f}' for nm in names))
    np.savez_compressed(RES / 'h54_scores.npz', **out)
    (RES / 'h54_models.json').write_text(json.dumps(M.info, indent=1, default=str)); log('wrote h54_scores.npz')


# ------------------------------------------------------------------------------------------------------------------ refs
def _lr_hist(x, y, bins):
    hq = np.histogram(x[y == 1], bins)[0] + 0.5; hg = np.histogram(x[y == 0], bins)[0] + 0.5
    return np.log(hq / hq.sum()) - np.log(hg / hg.sum())          # quark log-odds


def c2st_feats(L, O):
    z = [np.pad(a[:5, 0] / a[:, 0].sum(), (0, max(0, 5 - len(a)))) for a in L]
    return np.c_[O['N'], O['Nch'], O['ptd'], np.log(O['girth'] + 1e-6), O['m_over_pt'], np.log(O['C1_0.5'] + 1e-9),
                 np.log(O['C1_1.0'] + 1e-9), np.log(O['C1_2.0'] + 1e-9), O['jpt'], np.array(z)]


def c2st(F0, F1, seed=0, B=300):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); n = min(len(F0), len(F1))
    F0 = F0[rng.permutation(len(F0))[:n]]; F1 = F1[rng.permutation(len(F1))[:n]]
    X = np.r_[F0, F1]; y = np.r_[np.zeros(n), np.ones(n)]; p = rng.permutation(2 * n); tr, te = p[:n], p[n:]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr]); s = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], s); bs = []
    for _ in range(B):
        ix = rng.integers(0, len(te), len(te)); bs.append(roc_auc_score(y[te][ix], s[ix]))
    return dict(auc=float(auc), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_per_side=int(n))


def step_refs(a):
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from scipy.stats import ks_2samp
    from interp import h14_qg_generator as H14
    RES.mkdir(parents=True, exist_ok=True); pool = Pool(a.cpu)
    T, lrinfo = H14.lrn_tables(); lrn = -T['py']                     # quark log-odds from NB fits to KMT train multiplicity
    Lt, yt, Nt = H14.efp_train_lists('py', n_per_class=a.train_per_class, seed=SEED + 2)
    Lt4 = [np.c_[x, np.zeros(len(x))] for x in Lt]
    ptd_t = np.array([np.sqrt(((x[:, 0] / x[:, 0].sum()) ** 2).sum()) for x in Lt])
    bp = np.quantile(ptd_t, np.linspace(0, 1, 41)); bp[0], bp[-1] = 0, 1.01; hp = _lr_hist(ptd_t, yt, bp)
    Ft = H14.efps(Lt, pool); q = (yt == 1).astype(int)
    feat = lambda F, N, wn: np.c_[np.log(np.maximum(F, 1e-12)), N] if wn else np.log(np.maximum(F, 1e-12))
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15, random_state=0)
    clf = dict(gbdt_efpn=gb().fit(feat(Ft, Nt, True), q), gbdt_efp=gb().fit(feat(Ft, Nt, False), q),
               lr_n_ptd=gb().fit(np.c_[Nt, ptd_t], q))
    log(f'references trained on {len(yt)} KMT train jets ({Ft.shape[1]} EFPs)')
    S = all_samples(a.kmt_per_class); out = {}; feats = {}
    for sname, (L, y) in S.items():
        O = jet_obs(L); F = H14.efps([x[:, :3] for x in L], pool); N = O['N']
        out[f'{sname}|lr_n'] = lrn[np.minimum(N.astype(int), 299)].astype(np.float32)
        out[f'{sname}|lr_ptd'] = hp[np.clip(np.searchsorted(bp, O['ptd'], side='right') - 1, 0, 39)].astype(np.float32)
        out[f'{sname}|lr_n_ptd'] = clf['lr_n_ptd'].decision_function(np.c_[N, O['ptd']]).astype(np.float32)
        out[f'{sname}|gbdt_efpn'] = clf['gbdt_efpn'].decision_function(feat(F, N, True)).astype(np.float32)
        out[f'{sname}|gbdt_efp'] = clf['gbdt_efp'].decision_function(feat(F, N, False)).astype(np.float32)
        for k in ('N', 'Nch', 'ptd', 'girth', 'm_over_pt', 'mass', 'C1_1.0', 'jpt'): out[f'obs|{sname}|{k}'] = O[k].astype(np.float32)
        out[f'y|{sname}'] = y.astype(np.int8); feats[sname] = c2st_feats(L, O)
        log(sname, 'refs: ' + '; '.join(f'{r} acc {np.mean((out[f"{sname}|{r}"] > 0) == (y == 1)):.4f}'
                                       for r in ('lr_n', 'lr_ptd', 'lr_n_ptd', 'gbdt_efp', 'gbdt_efpn')))
    np.savez_compressed(RES / 'h54_refs.npz', **out)
    val = dict(lrn=lrinfo, train_jets=int(len(yt)), c2st={}, dists={})
    for cls, cn in ((1, 'q'), (0, 'g')):
        yk = out['y|KMT'] == cls; y0 = out['y|V0'] == cls
        val['c2st'][f'V0_vs_KMT_{cn}'] = c2st(feats['KMT'][yk], feats['V0'][y0]); log('C2ST V0 vs KMT', cn, val['c2st'][f'V0_vs_KMT_{cn}'])
        for v in VARIANTS[1:]:
            yv = out[f'y|{v}'] == cls; val['c2st'][f'{v}_vs_V0_{cn}'] = c2st(feats['V0'][y0], feats[v][yv]); log('C2ST', v, cn, val['c2st'][f'{v}_vs_V0_{cn}'])
        for k in ('N', 'ptd', 'girth', 'mass'):
            for sname in S:
                ys = out[f'y|{sname}'] == cls; x = out[f'obs|{sname}|{k}'][ys]
                val['dists'].setdefault(cn, {}).setdefault(k, {})[sname] = dict(mean=float(x.mean()), sd=float(x.std()),
                    ks_vs_KMT=float(ks_2samp(x, out[f'obs|KMT|{k}'][yk]).statistic) if sname != 'KMT' else 0.0)
    (RES / 'h54_valid.json').write_text(json.dumps(val, indent=1, default=float)); pool.close(); log('wrote h54_refs.npz, h54_valid.json')


# ----------------------------------------------------------------------------------------------------------------- stats
NETS = ['part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_k', 'pfn_t', 'efn_k', 'efn_t',
        'hw_part_kin', 'hw_pnet_kin'] + [f'perm{k}' for k in range(6)]
REFS = ['lr_n', 'lr_ptd', 'lr_n_ptd', 'gbdt_efp', 'gbdt_efpn']
H0_REFS = ['efn_k', 'gbdt_efp']; H1_REFS = ['lr_n', 'gbdt_efpn']


def _zp(d, se):
    from scipy.stats import norm
    return float(2 * norm.sf(abs(d) / se)) if se > 0 else (1.0 if abs(d) < 1e-12 else 0.0)


def step_stats(a):
    from sklearn.metrics import roc_auc_score
    from interp.h5_qg_counting import holm
    Zs = np.load(RES / 'h54_scores.npz'); Zr = np.load(RES / 'h54_refs.npz')
    samples = VARIANTS + ['KMT']; S, y, obs = {}, {}, {}
    for sn in samples:
        y[sn] = Zs[f'y|{sn}'].astype(int); assert (Zr[f'y|{sn}'] == Zs[f'y|{sn}']).all(), sn
        S[sn] = {m: Zs[f'{sn}|{m}'].astype(np.float64) for m in NETS if f'{sn}|{m}' in Zs.files}
        S[sn].update({m: Zr[f'{sn}|{m}'].astype(np.float64) for m in REFS})
        obs[sn] = {k: Zr[f'obs|{sn}|{k}'].astype(np.float64) for k in ('N', 'Nch', 'ptd', 'girth', 'm_over_pt', 'mass', 'C1_1.0')}
    models = [m for m in NETS if m in S['V0']] + REFS
    B = a.boot; rng = np.random.default_rng(SEED + 9)
    R = dict(B=B, models=models, accuracy={}, means={}, shifts={}, tests=[], ratios={}, reweight={}, paired={}, obs_shift={})
    for sn in samples:
        R['accuracy'][sn] = {m: dict(acc=float(np.mean((S[sn][m] > 0) == (y[sn] == 1))), auc=float(roc_auc_score(y[sn], S[sn][m])))
                             for m in models}
    # bootstrap means per (sample, class): matrix (B, n_models)
    boot = {}
    for sn in samples:
        for cls, cn in ((1, 'q'), (0, 'g')):
            ix = np.flatnonzero(y[sn] == cls); M = np.stack([S[sn][m][ix] for m in models], 1)
            Ob = np.stack([obs[sn][k][ix] for k in obs[sn]], 1); J = len(ix); bm = np.empty((B, M.shape[1])); bo = np.empty((B, Ob.shape[1]))
            for b in range(B):
                c = np.bincount(rng.integers(0, J, J), minlength=J).astype(np.float64); bm[b] = c @ M / J; bo[b] = c @ Ob / J
            boot[(sn, cn)] = (M.mean(0), bm, Ob.mean(0), bo)
            R['means'].setdefault(sn, {})[cn] = dict(n=int(J), **{m: float(v) for m, v in zip(models, M.mean(0))},
                                                     obs={k: float(v) for k, v in zip(obs[sn], Ob.mean(0))})
    mi = {m: i for i, m in enumerate(models)}; ok_ = list(obs['V0'])
    for v in VARIANTS[1:] + ['KMT']:
        for cn in ('q', 'g'):
            m0, b0, o0, bo0 = boot[('V0', cn)]; m1, b1, o1, bo1 = boot[(v, cn)]
            d = m1 - m0; bd = b1 - b0; od = o1 - o0; bod = bo1 - bo0
            R['obs_shift'].setdefault(v, {})[cn] = {k: dict(d=float(od[i]), ci=np.percentile(bod[:, i], [2.5, 97.5]).tolist(),
                                                            rel=float(od[i] / o0[i])) for i, k in enumerate(ok_)}
            sh = {}
            for m in models:
                i = mi[m]; se = bd[:, i].std()
                sh[m] = dict(d=float(d[i]), ci=np.percentile(bd[:, i], [2.5, 97.5]).tolist(), se=float(se), p0=_zp(d[i], se))
                for r in H0_REFS + H1_REFS:
                    if r == m or m in REFS: continue
                    j = mi[r]; dd = d[i] - d[j]; bdd = bd[:, i] - bd[:, j]
                    sh[m][f'minus_{r}'] = dict(d=float(dd), ci=np.percentile(bdd, [2.5, 97.5]).tolist(), p=_zp(dd, bdd.std()))
                    if v != 'KMT' and not m.startswith('perm'):
                        R['tests'].append(dict(variant=v, cls=cn, model=m, ref=r, d=float(dd), p=sh[m][f'minus_{r}']['p']))
                if m not in REFS:
                    # kappa: share of the N increment (GBDT(EFP+N) - GBDT(EFP)) that the network follows
                    e, n_ = mi['gbdt_efp'], mi['gbdt_efpn']; den = d[n_] - d[e]; bden = bd[:, n_] - bd[:, e]
                    k = (d[i] - d[e]) / den; bk = (bd[:, i] - bd[:, e]) / bden
                    sh[m]['kappa_efp'] = dict(k=float(k), ci=np.percentile(bk, [2.5, 97.5]).tolist(), den=float(den))
                    for r in ('efn_k', 'lr_n', 'gbdt_efpn'):
                        j = mi[r]
                        if r == m: continue
                        sh[m][f'ratio_{r}'] = dict(r=float(d[i] / d[j]), ci=np.percentile(bd[:, i] / bd[:, j], [2.5, 97.5]).tolist())
            R['shifts'].setdefault(v, {})[cn] = sh
            log(v, cn, f"dN {od[ok_.index('N')]:+.2f} ({100 * od[ok_.index('N')] / o0[ok_.index('N')]:+.1f} %) | " +
                ' '.join(f"{m} {sh[m]['d']:+.3f}" for m in models))
    adj = holm([t['p'] for t in R['tests']])
    for t, h in zip(R['tests'], adj): t['holm'] = float(h)
    R['holm'] = dict(n_tests=len(R['tests']), n_pass_005=int((adj < 0.05).sum()))
    # paired (event-matched) shifts, hadronisation and shower variants vs V0
    Zg = {v: np.load(GEN / f'{v}.npz') for v in VARIANTS}
    key = lambda v: Zg[v]['widx'].astype(np.int64) * 100_000 + Zg[v]['event'].astype(np.int64)
    k0 = key('V0'); pos0 = {int(k): j for j, k in enumerate(k0)}
    for v in VARIANTS[1:]:
        kv = key(v); pr = [(pos0[int(k)], j) for j, k in enumerate(kv) if int(k) in pos0]
        i0 = np.array([p[0] for p in pr]); i1 = np.array([p[1] for p in pr]); R['paired'][v] = dict(n_pairs=len(pr))
        for cls, cn in ((1, 'q'), (0, 'g')):
            sel = y['V0'][i0] == cls; a0, a1 = i0[sel], i1[sel]; r = dict(n=int(sel.sum()))
            r['dN'] = float((obs[v]['N'][a1] - obs['V0']['N'][a0]).mean())
            for m in models:
                dd = S[v][m][a1] - S['V0'][m][a0]; se = dd.std() / np.sqrt(len(dd))
                r[m] = dict(d=float(dd.mean()), ci=[float(dd.mean() - 1.96 * se), float(dd.mean() + 1.96 * se)], sd_per_jet=float(dd.std()))
            R['paired'][v][cn] = r
    # reweighting check: V0 jets reweighted to the variant's N distribution within deciles of the EFN output
    for v in VARIANTS[1:]:
        for cls, cn in ((1, 'q'), (0, 'g')):
            s0 = y['V0'] == cls; s1 = y[v] == cls
            e0, e1 = S['V0']['efn_k'][s0], S[v]['efn_k'][s1]; n0, n1 = obs['V0']['N'][s0], obs[v]['N'][s1]
            eb = np.quantile(e0, np.linspace(0, 1, 11)); eb[0], eb[-1] = -np.inf, np.inf
            nb = np.unique(np.quantile(np.r_[n0, n1], np.linspace(0, 1, 16))); nb[0], nb[-1] = -np.inf, np.inf

            def weights(i0, i1):
                c0 = np.histogram2d(e0[i0], n0[i0], [eb, nb])[0] + 0.5; c1 = np.histogram2d(e1[i1], n1[i1], [eb, nb])[0] + 0.5
                p0 = c0 / c0.sum(1, keepdims=True); p1 = c1 / c1.sum(1, keepdims=True)
                be = np.clip(np.searchsorted(eb, e0[i0], side='right') - 1, 0, 9); bn = np.clip(np.searchsorted(nb, n0[i0], side='right') - 1, 0, len(nb) - 2)
                return (p1 / p0)[be, bn]
            ia, ib = np.arange(s0.sum()), np.arange(s1.sum()); w = weights(ia, ib); rr = dict(w_max=float(w.max()), n_eff=float(w.sum() ** 2 / (w ** 2).sum()))
            bs = {m: [] for m in models}; bsN = []
            r2 = np.random.default_rng(SEED + 11)
            for b in range(200):
                ja = r2.integers(0, len(ia), len(ia)); jb = r2.integers(0, len(ib), len(ib)); wb = weights(ja, jb)
                bsN.append(np.average(n0[ja], weights=wb) - n0[ja].mean())
                for m in models:
                    x = S['V0'][m][s0][ja]; bs[m].append(np.average(x, weights=wb) - x.mean())
            rr['dN_reweighted'] = float(np.average(n0, weights=w) - n0.mean()); rr['dN_target'] = float(n1.mean() - n0.mean())
            for m in models:
                x = S['V0'][m][s0]; dv = np.average(x, weights=w) - x.mean()
                rr[m] = dict(d=float(dv), ci=np.percentile(bs[m], [2.5, 97.5]).tolist())
            R['reweight'].setdefault(v, {})[cn] = rr
            log('reweight', v, cn, f"dN {rr['dN_reweighted']:+.2f} (target {rr['dN_target']:+.2f})", ' '.join(f"{m} {rr[m]['d']:+.3f}" for m in models))
    (RES / 'h54_run.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h54_run.json')


# ------------------------------------------------------------------------------------------------------------------- fig
LAB = dict(part_kin='ParT_kin', part_kinpid='ParT_kinpid', pnet_kin='ParticleNet_kin', pnet_kinpid='ParticleNet_kinpid',
           lorentznet='LorentzNet', pfn_k='PFN', pfn_t='PFN (2nd run)', efn_k='EFN (IRC-safe)', efn_t='EFN (2nd run)',
           hw_part_kin='ParT_kin [Herwig-trained]', hw_pnet_kin='ParticleNet_kin [Herwig-trained]', lr_n='LR(N)',
           lr_ptd='LR($p_T^D$)', lr_n_ptd='LR(N, $p_T^D$)', gbdt_efp='GBDT(EFP)', gbdt_efpn='GBDT(EFP + N)', perm='permuted ParT_kin (6)')


def step_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((RES / 'h54_run.json').read_text()); Zs = np.load(RES / 'h54_scores.npz'); Zr = np.load(RES / 'h54_refs.npz')
    col = dict(V0='k', Hzp='#1f77b4', Hzm='#17becf', Sp='#d62728', Sm='#ff7f0e', KMT='0.6')
    # figure 1: N and EFN distributions per variant + one event in every variant
    fig = plt.figure(figsize=(14, 8.2)); gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05])
    for k, (cls, cn) in enumerate(((1, 'quark'), (0, 'gluon'))):
        ax = fig.add_subplot(gs[0, k]); bx = fig.add_subplot(gs[0, 2 + k])
        for v in ['KMT'] + VARIANTS:
            s = Zr[f'y|{v}'] == cls; N = Zr[f'obs|{v}|N'][s]; e = Zs[f'{v}|efn_k'][s]
            ax.hist(N, np.arange(0, 141, 3), histtype='step', color=col[v], lw=1.6 if v != 'KMT' else 3, alpha=1 if v != 'KMT' else 0.5,
                    density=True, label=f'{VLAB[v]}: <N> {N.mean():.1f}')
            bx.hist(e, np.linspace(-6, 6, 61), histtype='step', color=col[v], lw=1.6 if v != 'KMT' else 3, alpha=1 if v != 'KMT' else 0.5,
                    density=True, label=f'{VLAB[v]}: <s> {e.mean():+.2f}')
        ax.set_xlabel('constituent multiplicity N'); ax.set_title(f'{cn} jets: multiplicity', fontsize=10); ax.legend(fontsize=6.5)
        bx.set_xlabel('EFN output s = log p(q)/p(g)'); bx.set_title(f'{cn} jets: IRC-safe EFN output', fontsize=10); bx.legend(fontsize=6.5)
    ax = fig.add_subplot(gs[0, 4]); ax.axis('off')
    Pj = json.loads((RES / 'h54_pilot.json').read_text())['chosen']
    txt = 'Generator settings\nV0: Pythia 8 Monash (aLund 0.68,\n  bLund 0.98, sigma 0.335, FSR a_s 0.1365)\n'
    for v in ('Hzp', 'Hzm'): txt += f'{VLAB[v]}:\n  aLund {Pj[v]["setting"][0]}, bLund {Pj[v]["setting"][1]}, sigma {Pj[v]["setting"][2]}\n'
    txt += 'S+ / S-: TimeShower:alphaSvalue\n  0.1465 / 0.1265\nSame per-event seeds in all variants;\nhadronisation variants share the\nparton-level event exactly.'
    ax.text(0, 1, txt, va='top', fontsize=8, family='monospace')
    # example: one gluon event present in all variants
    Zg = {v: np.load(GEN / f'{v}.npz') for v in VARIANTS}
    keys = {v: set((Zg[v]['widx'].astype(np.int64) * 100_000 + Zg[v]['event']).tolist()) for v in VARIANTS}
    common = sorted(set.intersection(*keys.values())); sel = [k for k in common if k // 100_000 < WORKERS]
    kx = sel[len(sel) // 3] if sel else common[0]
    for c, v in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[1, c]); Z = Zg[v]; j = int(np.flatnonzero(Zg[v]['widx'].astype(np.int64) * 100_000 + Zg[v]['event'] == kx)[0])
        x = Z['X'][j, :Z['n'][j]]; jy, jphi = Z['jet'][j, 1], Z['jet'][j, 2]
        ax.scatter(x[:, 1] - jy, x[:, 2] - jphi, s=4 + 600 * x[:, 0] / x[:, 0].sum(), c=col[v], alpha=0.6, edgecolors='none')
        ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.45, 0.45); ax.set_aspect('equal')
        s = Zs[f'{v}|part_kin'][j]; e = Zs[f'{v}|efn_k'][j]
        ax.set_title(f'{VLAB[v]}\nN = {len(x)}, ParT_kin s {s:+.2f}, EFN s {e:+.2f}', fontsize=8)
        ax.set_xlabel('Δy to jet axis'); ax.set_ylabel('Δφ to jet axis') if c == 0 else None
    fig.suptitle('H54: the same gluon-jet hard event generated in five Pythia 8 variants (marker area ∝ pT); top: multiplicity '
                 'and EFN output per variant (30,000 jets per class)', fontsize=10.5)
    fig.tight_layout(); fig.savefig(RES / 'h54_variants.png', dpi=120); plt.close(fig)
    # figure 2: shifts per model and variant
    rows = ['part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_k', 'efn_k', 'hw_part_kin', 'hw_pnet_kin', 'perm',
            'gbdt_efp', 'gbdt_efpn', 'lr_n', 'lr_n_ptd', 'lr_ptd']
    fig, axs = plt.subplots(1, 4, figsize=(15, 6.5), sharey=True)
    for k, v in enumerate(VARIANTS[1:]):
        ax = axs[k]
        for o, (cn, c) in enumerate((('q', '#1f77b4'), ('g', '#d62728'))):
            for i, m in enumerate(rows):
                if m == 'perm':
                    ds = [R['shifts'][v][cn][f'perm{j}']['d'] for j in range(6)]
                    ax.plot(ds, [i + (o - 0.5) * 0.3] * 6, '.', color=c, alpha=0.5, ms=4); continue
                r = R['shifts'][v][cn][m]
                ax.errorbar(r['d'], i + (o - 0.5) * 0.3, xerr=[[r['d'] - r['ci'][0]], [r['ci'][1] - r['d']]], fmt='o', ms=3.5, color=c,
                            label=('quark jets' if cn == 'q' else 'gluon jets') if i == 0 else None)
        ax.axvline(0, color='k', lw=0.6); ax.axhline(9.5, color='0.7', lw=0.6)
        dn = R['obs_shift'][v]['q']['N']['rel'], R['obs_shift'][v]['g']['N']['rel']
        ax.set_title(f'{VLAB[v]}\nΔN/N: quark {100 * dn[0]:+.1f} %, gluon {100 * dn[1]:+.1f} %', fontsize=9)
        ax.set_xlabel('Δ<s> = <s>_variant − <s>_V0\n(quark log-odds, 95 % CI)', fontsize=8.5)
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([LAB[m] for m in rows], fontsize=8.5); axs[0].invert_yaxis(); axs[0].legend(fontsize=8)
    fig.suptitle('H54: mean quark log-odds shift of each tagger when Pythia 8 hadronisation (Hz) or final-state shower α_s (S) is changed; '
                 'networks above the grey line, references trained on KMT jets below', fontsize=10)
    fig.tight_layout(); fig.savefig(RES / 'h54_shifts.png', dpi=120); log('wrote figures')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['check', 'pilot', 'gen', 'score', 'refs', 'stats', 'fig'])
    ap.add_argument('--variant', default='V0'); ap.add_argument('--jets', type=int, default=30000)
    ap.add_argument('--pilot_jets', type=int, default=1000); ap.add_argument('--cpu', type=int, default=8)
    ap.add_argument('--kmt_per_class', type=int, default=30000); ap.add_argument('--train_per_class', type=int, default=30000)
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--round', type=int, default=1)
    a = ap.parse_args()
    dict(check=step_check, pilot=step_pilot, gen=step_gen, score=step_score, refs=step_refs, stats=step_stats, fig=step_fig)[a.step](a)
