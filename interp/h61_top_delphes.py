"""H61 (loop D, claim D24): detector-level, MPI-off rerun of the H60 top hadronisation test, matched to the Top Landscape
training sample (Pythia 8 default tune at 14 TeV, no multiple parton interactions, Delphes ATLAS card, anti-kT R = 0.8 on Delphes
energy-flow objects, 550 < pT < 650 GeV, |eta| < 2, tops truth-matched as in Kasieczka et al.).

Question: at detector level, do the Top Landscape / JetClass top taggers still lower their top score on top jets when only Pythia's
hadronisation settings raise the particle count at fixed jet mass, tau32 and W-pair mass, by more than the IRC-safe EFN does?

Generator: Pythia 8.317 (pythia8mc, venv env-gen), pp 14 TeV, Monash (Tune:pp 14), PartonLevel:MPI = off, ISR / FSR / hadronisation on;
tops: Top:gg2ttbar + qqbar2ttbar with W -> quarks; QCD: HardQCD:all; pTHat 450-800 GeV.  Every event is reseeded before generation
with seed(w, cls, i) = 1 + (2 w + cls) * 20,000 + i for worker seed w in 6101-6110 (V0b: 6111-6120), so a hadronisation-only change
leaves the hard process and shower identical event by event.  Events with a particle-level anti-kT 0.8 jet of pT > 450 GeV and
|eta| < 2.5 (loose prefilter) are written as HepMC2 (stable particles only) and run through Delphes 3.5.1 (conda-forge, prefix
env-delphes) with the stock delphes_card_ATLAS.tcl plus `set RandomSeed` per (worker, class, chunk).  Detector jets: anti-kT 0.8
(Pythia SlowJet / fjcore) on EFlowTrack + EFlowPhoton + EFlowNeutralHadron, massless; jets with 550 < pT < 650 GeV, |eta| < 2 kept;
top: the hadronic top and its b, q, q' all within Delta R < 0.8 of the jet axis; QCD: nearest outgoing hard parton within 0.8.
Pair key = (worker, class, event, slot).

Settings (StringZ:aLund, StringZ:bLund, StringPT:sigma [GeV]); Monash (0.68, 0.98, 0.335):
  V0    Monash                         V0r   V0 particles, Delphes RandomSeed shifted (detector-noise null, paired)
  V0b   Monash, seeds 6111-6120 (independent events: generator-noise band, unpaired)
  Hzp   (0.79, 0.88, 0.247)  N up      Hzm   (0.48, 0.98, 0.438)  N down       (H54 / H60 settings)
  sigup (0.68, 0.98, 0.235)  sigma only, N up        aup (1.28, 0.98, 0.335)  aLund only, N up
Count-only edits on the Delphes jets (step edits): split = V0 jet whose k = max(N_Hzp - N_V0, 0) softest constituents are each
replaced by two pT/2 halves 0.01 apart in (eta, phi) (count up by the paired Hz+ amount, IRC-safe content fixed to O(0.01^2));
merge = Hz+ jet whose k nearest-in-Delta R pairs among its softer half are combined (three-momenta added; count back to V0).

Steps (all cluster jobs):
  check                      (env-gen + env-delphes) Delphes run on a few events, branch list, timing
  gen --setting S            (env-gen, 1 CPU / 2 GiB) 10 workers x 500 top + 500 QCD jets -> data/top/gen_h61/<S>.npz
  efp                        (venv-ef) builds split / merge, EFPs; LR(N), GBDT(EFP), GBDT(EFP+N) fitted on Top Landscape train
  score                      (GPU, env-part) all networks, LNet batch 1, permuted ParT, observables -> h61_scores.npz
  stats                      (CPU) paired shifts, CIs, kappa, working points, McNemar, C2ST, Holm -> h61_run.json, figures
Score convention: s = top log-odds (Top Landscape nets: log p(top) - log p(QCD); JetClass nets: logit(Tbqq) - logit(QCD)).
"""
import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from interp import h60_top_hadronisation as H60

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
GEN = V / 'data' / 'top' / 'gen_h61'
DEL = Path(os.environ.get('H61_DELPHES', '/tmp/env-delphes'))   # pod-local conda prefix (install_delphes.sh; CephFS linking ran at ~4 packages/min)
NOM = (0.68, 0.98, 0.335)
SETTINGS = {'V0': NOM, 'V0r': NOM, 'V0b': NOM, 'Hzp': (0.79, 0.88, 0.247), 'Hzm': (0.48, 0.98, 0.438),
            'sigup': (0.68, 0.98, 0.235), 'aup': (1.28, 0.98, 0.335)}
WSEEDS = lambda s: list(range(6111, 6121)) if s == 'V0b' else list(range(6101, 6111))
LMAX = 200
R_JET = 0.8
CHUNK = 1000


# ======================================================================================================== generator + Delphes
def ev_seed(w, cls, i):
    return 1 + (2 * w + cls) * 20_000 + i


def make_pythia(cls, extra, seed):
    p = H60._py8().Pythia('', False)
    base = ['Beams:eCM = 14000.', 'Tune:pp = 14', 'PartonLevel:MPI = off', 'PartonLevel:ISR = on', 'PartonLevel:FSR = on',
            'HadronLevel:all = on', 'PhaseSpace:pTHatMin = 450.', 'PhaseSpace:pTHatMax = 800.', 'Next:numberCount = 0',
            'Next:numberShowEvent = 0', 'Next:numberShowProcess = 0', 'Next:numberShowInfo = 0', 'Init:showChangedSettings = off',
            'Init:showProcesses = off', 'Init:showChangedParticleData = off', 'Random:setSeed = on', f'Random:seed = {seed}']
    base += (['Top:gg2ttbar = on', 'Top:qqbar2ttbar = on', '24:onMode = off', '24:onIfAny = 1 2 3 4 5'] if cls == 1
             else ['HardQCD:all = on'])
    for s in base: assert p.readString(s), s
    for k, v in extra.items(): assert p.readString(f'{k} = {v}'), k
    assert p.init()
    return p


def settings_dict(name):
    a, b, s = SETTINGS[name]
    return {} if SETTINGS[name] == NOM else {'StringZ:aLund': a, 'StringZ:bLund': b, 'StringPT:sigma': s}


def final_particles(p):
    """(n, 6) id, px, py, pz, e, m of all final-state particles"""
    ev = p.event; rows = []
    for i in range(ev.size()):
        q = ev[i]
        if q.isFinal(): rows.append((q.id(), q.px(), q.py(), q.pz(), q.e(), q.m()))
    return np.array(rows, np.float64)


def write_hepmc(path, events):
    """HepMC2 IO_GenEvent ascii; one vertex at the origin with every stable particle outgoing (all Delphes needs)"""
    with open(path, 'w') as f:
        f.write('\nHepMC::Version 2.06.09\nHepMC::IO_GenEvent-START_EVENT_LISTING\n')
        for evnum, P in events:
            f.write(f'E {evnum} -1 -1.0 -1.0 -1.0 0 -1 1 0 0 0 0\nU GEV MM\nV -1 0 0 0 0 0 0 {len(P)} 0\n')
            f.write(''.join('P %d %d %.10e %.10e %.10e %.10e %.10e 1 0 0 0 0\n' % (10001 + k, int(r[0]), r[1], r[2], r[3], r[4], r[5])
                            for k, r in enumerate(P)))
        f.write('HepMC::IO_GenEvent-END_EVENT_LISTING\n')


def find_card():
    c = sorted(glob.glob(str(DEL / '**' / 'delphes_card_ATLAS.tcl'), recursive=True))
    assert c, 'no ATLAS card under env-delphes'
    return c[0]


def run_delphes(hepmc, out, seed):
    card = Path(out).with_suffix('.tcl')
    card.write_text(f'set RandomSeed {int(seed)}\n' + Path(find_card()).read_text())
    env = os.environ.copy(); env.pop('LD_PRELOAD', None); env['PATH'] = f'{DEL}/bin:' + env.get('PATH', '')
    if os.path.exists(out): os.remove(out)
    for attempt in range(3):                  # d-h61-g-hzp: one run exited with 129 (SIGHUP) mid-job; same seed -> same output
        if os.path.exists(out): os.remove(out)
        r = subprocess.run([str(DEL / 'bin' / 'DelphesHepMC2'), str(card), str(out), str(hepmc)], capture_output=True, text=True, env=env)
        if r.returncode == 0: return r
        log('Delphes failed', attempt, r.returncode, r.stderr[-1500:])
    raise RuntimeError('Delphes')


def _branch(t, name):
    ks = [k for k in t.keys() if k.endswith(name) and k.split('/')[-1] == name]
    assert ks, (name, [k for k in t.keys() if name.split('.')[0] in k][:10])
    return t[ks[0]].array(library='np')


def read_delphes(out):
    """-> list per event of (m, 4) arrays (pt, eta, phi, code) with code 211*charge (EFlowTrack), 22 (EFlowPhoton), 130 (neutral h.)"""
    import uproot
    t = uproot.open(out)['Delphes']
    num = _branch(t, 'Event.Number')
    tr = [_branch(t, f'EFlowTrack.{k}') for k in ('PT', 'Eta', 'Phi', 'Charge')]
    ph = [_branch(t, f'EFlowPhoton.{k}') for k in ('ET', 'Eta', 'Phi')]
    nh = [_branch(t, f'EFlowNeutralHadron.{k}') for k in ('ET', 'Eta', 'Phi')]
    res = []
    for e in range(len(num)):
        a = np.c_[tr[0][e], tr[1][e], tr[2][e], 211.0 * tr[3][e]]
        b = np.c_[ph[0][e], ph[1][e], ph[2][e], np.full(len(ph[0][e]), 22.0)]
        c = np.c_[nh[0][e], nh[1][e], nh[2][e], np.full(len(nh[0][e]), 130.0)]
        res.append(np.r_[a, b, c].astype(np.float64))
    return [int(np.asarray(x).ravel()[0]) for x in num], res


def det_jets(p, sj, objs, tru, cls):
    """anti-kT 0.8 on the Delphes objects (massless) via SlowJet on a rebuilt Pythia event -> list of (slot, (n,5) pt eta phi 0 code,
    jet (pt, eta, phi, m), truth)"""
    ev = p.event; ev.reset(); off = ev.size()          # reset() leaves the system entry (index 0) in place
    pt, eta, phi = objs[:, 0], objs[:, 1], objs[:, 2]
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta); e = pt * np.cosh(eta)
    for k in range(len(objs)): ev.append(211, 1, 0, 0, px[k], py[k], pz[k], e[k], 0.)
    if not sj.analyze(ev) or sj.sizeJet() == 0: return []
    res, used = [], set()
    for j in range(sj.sizeJet()):
        if sj.pT(j) < 540: break
        idx = np.array([i - off for i in sj.constituents(j)], np.int64)
        a = np.c_[objs[idx, 0], objs[idx, 1], objs[idx, 2], np.zeros(len(idx)), objs[idx, 3]]
        a = a[np.argsort(-a[:, 0], kind='stable')]
        jx, jy, jz, jE = px[idx].sum(), py[idx].sum(), pz[idx].sum(), e[idx].sum()
        jpt = np.hypot(jx, jy); jeta = np.arcsinh(jz / jpt); jphi = np.arctan2(jy, jx); jm = np.sqrt(max(jE ** 2 - jx ** 2 - jy ** 2 - jz ** 2, 0))
        if not (550 <= jpt <= 650 and abs(jeta) < 2): continue
        if cls == 1:
            ok = [(s, P) for s, P in tru if all(H60._dr(jeta, jphi, a_, b_) < R_JET for a_, b_ in P)]
            if not ok: continue
            slot, P = ok[0]
        else:
            if not tru: continue
            d = [H60._dr(jeta, jphi, P[0][0], P[0][1]) for _, P in tru]; k = int(np.argmin(d))
            if d[k] > R_JET: continue
            slot, P = tru[k]
        if slot in used: continue
        used.add(slot); res.append((slot, a, (jpt, jeta, jphi, jm), np.array(P, np.float64)))
    return res


def n_particle_level(P, jet):
    """visible final-state particles (no neutrinos) within Delta R < 0.8 of the detector jet axis"""
    vis = ~np.isin(np.abs(P[:, 0]), (12, 14, 16))
    pt = np.hypot(P[:, 1], P[:, 2]); ok = vis & (pt > 0)
    eta = np.arcsinh(P[ok, 3] / pt[ok]); phi = np.arctan2(P[ok, 2], P[ok, 1])
    return int((H60._dr(jet[1], jet[2], eta, phi) < R_JET).sum())


def gen_worker(w, cls, extra, n_jets, dseed_off, tmp, max_events=19_999):
    pythia8 = H60._py8()
    p = make_pythia(cls, extra, ev_seed(w, cls, 0))
    sjp = pythia8.SlowJet(-1, R_JET, 450.0, 5.0, 2, 2)          # particle-level prefilter (visible particles)
    sjd = pythia8.SlowJet(-1, R_JET, 500.0, 5.0, 1, 2)          # detector-level, on rebuilt event of Delphes objects
    sjh = pythia8.SlowJet(-1, R_JET, 500.0, 5.0, 2, 2); npl_jets = 0   # H60 particle-level selection, yield bookkeeping only
    X, J, T, K, NPL = [], [], [], [], []; nev = npass = 0; t0 = time.time(); tdel = 0.0; i = 0; ch = 0
    while len(X) < n_jets and i < max_events:
        evs, tru, parts = [], {}, {}
        for i in range(i, min(i + CHUNK, max_events)):
            p.rndm.init(ev_seed(w, cls, i)); nev += 1
            if not p.next(): continue
            npl_jets += len(H60.jets_from_event(p, sjh, cls))
            if not sjp.analyze(p.event): continue
            if not any(sjp.pT(j) > 450 and abs(sjp.y(j)) < 2.5 for j in range(sjp.sizeJet())): continue
            tr = H60.truth_partons(p, cls)
            if not tr: continue
            P = final_particles(p); evs.append((i, P)); tru[i] = tr; parts[i] = P; npass += 1
        i += 1
        if not evs: continue
        hm = f'{tmp}/w{w}_c{cls}.hepmc'; ro = f'{tmp}/w{w}_c{cls}.root'; write_hepmc(hm, evs)
        td = time.time(); run_delphes(hm, ro, 1 + dseed_off + (2 * w + cls) * 100 + ch); tdel += time.time() - td; ch += 1
        nums, objs = read_delphes(ro); os.remove(hm); os.remove(ro)
        assert nums == [e for e, _ in evs], ('event order', nums[:5], [e for e, _ in evs][:5])
        for evn, ob in zip(nums, objs):
            for slot, a, jet, Tr in det_jets(p, sjd, ob, tru[evn], cls):
                X.append(a[:LMAX]); J.append(jet + (len(a),)); T.append(np.pad(Tr, ((0, 4 - len(Tr)), (0, 3 - Tr.shape[1]))))
                K.append(((w * 2 + cls) * 100_000 + evn) * 10 + slot); NPL.append(n_particle_level(parts[evn], jet))
        log(f'[w{w} c{cls}] {nev} events, {npass} to Delphes, {len(X)} jets; {1e3 * (time.time() - t0) / nev:.1f} ms/event '
            f'(Delphes {1e3 * tdel / max(npass, 1):.1f} ms per Delphes event)')
    X, J, T, K, NPL = X[:n_jets], J[:n_jets], T[:n_jets], K[:n_jets], NPL[:n_jets]
    return dict(X=X, J=J, T=T, K=K, NPL=NPL, y=[cls] * len(X),
                info=dict(w=w, cls=cls, events=nev, particle_level_jets_H60_selection=npl_jets, delphes_events=npass, jets=len(X), sec=time.time() - t0,
                          mean_N=float(np.mean([j[4] for j in J])) if J else 0, mean_N_pl=float(np.mean(NPL)) if NPL else 0))


def step_check(a):
    import uproot
    log('card', find_card())
    tmp = '/tmp/h61chk'; os.makedirs(tmp, exist_ok=True)
    for cls in (1, 0):
        r = gen_worker(6101, cls, {}, 40, 0, tmp, max_events=300)
        log('check', r['info'])
        if r['X']:
            x = r['X'][0]; log('first jet', r['J'][0], 'codes', {int(c): int((x[:, 4] == c).sum()) for c in np.unique(x[:, 4])},
                               'pt min', float(x[:, 0].min()), 'N_pl', r['NPL'][0])
    # detector vs particle-level leading-jet pT (jet response) on 150 QCD events
    pythia8 = H60._py8(); p = make_pythia(0, {}, 77); sjp = pythia8.SlowJet(-1, R_JET, 300.0, 5.0, 2, 2); sjd = pythia8.SlowJet(-1, R_JET, 300.0, 5.0, 1, 2)
    evs, lead = [], []
    for i in range(150):
        p.rndm.init(900 + i); p.next(); sjp.analyze(p.event)
        lead.append([(sjp.pT(j), sjp.y(j), sjp.phi(j)) for j in range(min(2, sjp.sizeJet()))]); evs.append((i, final_particles(p)))
    write_hepmc(f'{tmp}/r.hepmc', evs); run_delphes(f'{tmp}/r.hepmc', f'{tmp}/r.root', 3); _, objs = read_delphes(f'{tmp}/r.root'); rat = []
    for L, ob in zip(lead, objs):
        ev = p.event; ev.reset(); ev.append(90, -11, 0, 0, 0., 0., 0., 0., 0.)
        for o in ob: ev.append(211, 1, 0, 0, o[0] * np.cos(o[2]), o[0] * np.sin(o[2]), o[0] * np.sinh(o[1]), o[0] * np.cosh(o[1]), 0.)
        sjd.analyze(ev)
        for pt, yy, ph in L:
            if abs(yy) > 2: continue
            d = [(H60._dr(yy, ph, sjd.y(k), sjd.phi(k)), sjd.pT(k)) for k in range(sjd.sizeJet())]
            if d: dd = min(d); rat.append(dd[1] / pt if dd[0] < 0.3 else np.nan)
    rat = np.array(rat); log('det/particle leading-jet pT ratio: median', np.nanmedian(rat), 'quantiles', np.nanpercentile(rat, [10, 25, 75, 90]), 'unmatched', np.mean(np.isnan(rat)))
    log('object pT sums per event (mean): track', np.mean([o[np.abs(o[:, 3]) == 211, 0].sum() for o in objs]), 'photon', np.mean([o[o[:, 3] == 22, 0].sum() for o in objs]),
        'nh', np.mean([o[o[:, 3] == 130, 0].sum() for o in objs]))
    # branch list from a tiny run
    p = make_pythia(1, {}, 5); evs = []
    for i in range(3): p.rndm.init(10 + i); p.next(); evs.append((i, final_particles(p)))
    write_hepmc(f'{tmp}/x.hepmc', evs); rr = run_delphes(f'{tmp}/x.hepmc', f'{tmp}/x.root', 1)
    log('Delphes stdout tail', rr.stdout[-600:])
    t = uproot.open(f'{tmp}/x.root')['Delphes']; log('branches (EFlow / Event)', [k for k in t.keys() if 'EFlow' in k or k.startswith('Event')][:40])


def step_gen(a):
    GEN.mkdir(parents=True, exist_ok=True); extra = settings_dict(a.setting); log(a.setting, extra)
    tmp = f'/tmp/h61_{a.setting}'; os.makedirs(tmp, exist_ok=True)
    off = 500_000 if a.setting == 'V0r' else 0
    parts, info = [], []
    for w in WSEEDS(a.setting):
        for cls in (1, 0):
            r = gen_worker(w, cls, extra, a.jets // 10, off, tmp); parts.append(r); info.append(r['info']); log(r['info'])
    X = [x for r in parts for x in r['X']]
    n = np.array([len(x) for x in X]); Xa = np.zeros((len(X), int(n.max()), 5), np.float32)
    for j, x in enumerate(X): Xa[j, :len(x)] = x
    cat = lambda k, dt: np.array([v for r in parts for v in r[k]], dt)
    np.savez_compressed(GEN / f'{a.setting}.npz', X=Xa, n=n, jet=cat('J', np.float64), truth=cat('T', np.float64), key=cat('K', np.int64),
                        y=cat('y', np.int8), n_pl=cat('NPL', np.int64))
    (GEN / f'{a.setting}.json').write_text(json.dumps(dict(setting=a.setting, values=SETTINGS[a.setting], delphes_seed_offset=off,
                                                           card=find_card(), workers=info), indent=1))
    log('wrote', GEN / f'{a.setting}.npz', len(n))


# ======================================================================================================== count-only edits
def load_gen(name):
    Z = np.load(GEN / f'{name}.npz'); return {k: Z[k] for k in Z.files}


def _rows(G, j):
    return G['X'][j, :G['n'][j]].astype(np.float64)[:, [0, 1, 2, 4]]          # pt, eta, phi, code


def split_soft(x, k, rng, delta=0.01):
    """the k softest constituents each replaced by two pT/2 halves at +-delta/2 in (eta, phi), random direction"""
    n = len(x); k = int(min(k, n - 1))
    if k <= 0: return x.copy()
    ch = np.argsort(x[:, 0], kind='stable')[:k]; th = rng.uniform(0, 2 * np.pi, k)
    a = x[ch].copy(); a[:, 0] /= 2; b = a.copy()
    a[:, 1] += 0.5 * delta * np.cos(th); a[:, 2] += 0.5 * delta * np.sin(th)
    b[:, 1] -= 0.5 * delta * np.cos(th); b[:, 2] -= 0.5 * delta * np.sin(th)
    y = np.r_[np.delete(x, ch, 0), a, b]; return y[np.argsort(-y[:, 0], kind='stable')]


def merge_soft(x, k):
    """k nearest-in-Delta R pairs among the softer half combined (three-momenta added, massless; code of the harder one)"""
    n = len(x); k = int(k)
    if k <= 0: return x.copy()
    hard = x[: n // 2]; s = x[n // 2:].copy(); k = min(k, len(s) - 1)
    p = np.c_[s[:, 0] * np.cos(s[:, 2]), s[:, 0] * np.sin(s[:, 2]), s[:, 0] * np.sinh(s[:, 1])]; code = s[:, 3].copy()
    for _ in range(k):
        pt = np.hypot(p[:, 0], p[:, 1]); eta = np.arcsinh(p[:, 2] / pt); phi = np.arctan2(p[:, 1], p[:, 0])
        d = np.hypot(eta[:, None] - eta[None], (phi[:, None] - phi[None] + np.pi) % (2 * np.pi) - np.pi)
        np.fill_diagonal(d, np.inf); i, j = np.unravel_index(np.argmin(d), d.shape)
        if pt[j] > pt[i]: code[i] = code[j]
        p[i] = p[i] + p[j]; p = np.delete(p, j, 0); code = np.delete(code, j)
    pt = np.hypot(p[:, 0], p[:, 1]); s2 = np.c_[pt, np.arcsinh(p[:, 2] / pt), np.arctan2(p[:, 1], p[:, 0]), code]
    y = np.r_[hard, s2]; return y[np.argsort(-y[:, 0], kind='stable')]


def _pack(L, G, rows):
    n = np.array([min(len(x), LMAX) for x in L]); X = np.zeros((len(L), int(n.max()), 5), np.float32); jets = []
    for j, x in enumerate(L):
        x = x[:LMAX]; X[j, :len(x), :3] = x[:, :3]; X[j, :len(x), 4] = x[:, 3]
        px, py, pz = (x[:, 0] * np.cos(x[:, 2])).sum(), (x[:, 0] * np.sin(x[:, 2])).sum(), (x[:, 0] * np.sinh(x[:, 1])).sum()
        E = (x[:, 0] * np.cosh(x[:, 1])).sum(); pt = np.hypot(px, py)
        jets.append((pt, np.arcsinh(pz / pt), np.arctan2(py, px), np.sqrt(max(E * E - px * px - py * py - pz * pz, 0)), len(x)))
    return dict(X=X, n=n, jet=np.array(jets), truth=G['truth'][rows], key=G['key'][rows], y=G['y'][rows], n_pl=G['n_pl'][rows])


def build_edits():
    G0, G1 = load_gen('V0'), load_gen('Hzp')
    pos1 = {int(k): j for j, k in enumerate(G1['key'])}
    pr = np.array([(j, pos1[int(k)]) for j, k in enumerate(G0['key']) if int(k) in pos1])
    k = np.maximum(G1['n'][pr[:, 1]] - G0['n'][pr[:, 0]], 0); rng = np.random.default_rng(6161)
    Ls = [split_soft(_rows(G0, i0), kk, rng) for (i0, _), kk in zip(pr, k)]
    Lm = [merge_soft(_rows(G1, i1), kk) for (_, i1), kk in zip(pr, k)]
    np.savez_compressed(GEN / 'split.npz', **_pack(Ls, G0, pr[:, 0])); np.savez_compressed(GEN / 'merge.npz', **_pack(Lm, G1, pr[:, 1]))
    info = dict(pairs=int(len(pr)), mean_k_top=float(k[G0['y'][pr[:, 0]] == 1].mean()), mean_k_qcd=float(k[G0['y'][pr[:, 0]] == 0].mean()),
                frac_k0=float(np.mean(k == 0)))
    (GEN / 'edits.json').write_text(json.dumps(info)); log('edits', info)


ALL = list(SETTINGS) + ['split', 'merge']


# ======================================================================================================== EFP references (venv-ef)
def tl_train_lists(per_class=25000, seed=6162):
    """Top Landscape train rows (random, 25k per class), (pT, deta, dphi) from kin3 (128 hardest, float16) read sequentially"""
    from interp import data_top as dt
    d = dt.DST / 'train'; y = np.load(d / 'label.npy'); rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in (0, 1)]))
    k3m = np.load(d / 'kin3.npy', mmap_mode='r'); out = np.zeros((len(idx), k3m.shape[1], 3), np.float32); B = 100_000
    for s in range(0, k3m.shape[0], B):
        sel = (idx >= s) & (idx < s + B)
        if sel.any(): blk = np.asarray(k3m[s:s + B]); out[sel] = blk[idx[sel] - s]
    N = np.load(d / 'jet.npy')[idx, 5].astype(float)
    L = [k[k[:, 0] > 0].astype(np.float64) for k in out]
    return L, y[idx].astype(int), N


def step_efp(a):
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from interp import data_top as dt
    from interp import h14_qg_generator as H14
    if not (GEN / 'split.npz').exists() or a.rebuild_edits: build_edits()
    pool = Pool(a.cpu); out = {}; t = time.time()
    Lr, yr, Nr = tl_train_lists(); log(f'TL train rows {len(yr)} read {time.time() - t:.0f}s, <N> top {Nr[yr == 1].mean():.1f} qcd {Nr[yr == 0].mean():.1f}')
    Fr = H14.efps(Lr, pool); log('EFPs TL train', Fr.shape)
    it, yt = H60.tl_test_rows(); ft = np.load(dt.DST / 'test' / 'full.npy'); Pt = np.asarray(ft[it][..., :4], np.float64); del ft
    nt = np.minimum(np.load(dt.DST / 'test' / 'jet.npy')[it, 5].astype(np.int64), Pt.shape[1])
    samples = {'TL': ([H60._ptetaphi(Pt[j], nt[j]) for j in range(len(it))], nt.astype(float), yt[it].astype(int))}; del Pt
    for s in ALL:
        G = load_gen(s); P, n = H60.massless_P4(G)
        samples[s] = ([H60._ptetaphi(P[j].astype(np.float64), n[j]) for j in range(len(n))], n.astype(float), G['y'].astype(int))
    F = {s: H14.efps(v[0], pool) for s, v in samples.items()}; log('EFPs done', {s: f.shape for s, f in F.items()})
    feat = lambda F_, N, wn: np.c_[np.log(np.maximum(F_, 1e-12)), N] if wn else np.log(np.maximum(F_, 1e-12))
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15, random_state=0)
    clf = {'gbdt_efp': gb().fit(feat(Fr, Nr, False), yr), 'gbdt_efpn': gb().fit(feat(Fr, Nr, True), yr)}
    lrN = LogisticRegression().fit(Nr[:, None], yr)
    for s, (L, N, y) in samples.items():
        for k, c in clf.items(): out[f'{s}|{k}'] = c.decision_function(feat(F[s], N, k == 'gbdt_efpn')).astype(np.float32)
        out[f'{s}|lrN'] = lrN.decision_function(N[:, None]).astype(np.float32); out[f'y|{s}'] = y.astype(np.int8)
        log(s, ' '.join(f'{k} acc {np.mean((out[f"{s}|{k}"] > 0) == (y == 1)):.4f}' for k in ('gbdt_efp', 'gbdt_efpn', 'lrN')))
    out['lrN_coef'] = np.array([lrN.coef_[0, 0], lrN.intercept_[0]]); out['TL_train_N'] = np.array([Nr[yr == 1].mean(), Nr[yr == 0].mean()])
    RES.mkdir(parents=True, exist_ok=True); np.savez_compressed(RES / 'h61_efp.npz', **out); pool.close(); log('wrote h61_efp.npz')


# ======================================================================================================== scoring (GPU)
def pt_feats(P, n):
    """pT^D = sqrt(sum pT^2) / sum pT and the constituent pT spectrum: pT fraction of the constituents ranked 1, 2, 3, 5, 10, 20, 40
    and the number of constituents with pT < 1, 1-2, 2-5 GeV"""
    pt = np.hypot(P[..., 0], P[..., 1]).astype(np.float64); s = pt.sum(1)
    out = dict(pTD=np.sqrt((pt ** 2).sum(1)) / s)
    for r in (1, 2, 3, 5, 10, 20, 40): out[f'f{r}'] = pt[:, r - 1] / s if pt.shape[1] >= r else np.zeros(len(s))
    real = pt > 0
    out['n_lt1'] = (real & (pt < 1)).sum(1).astype(float); out['n_1_2'] = (real & (pt >= 1) & (pt < 2)).sum(1).astype(float)
    out['n_2_5'] = (real & (pt >= 2) & (pt < 5)).sum(1).astype(float)
    return out


def step_score(a):
    import torch
    from sklearn.metrics import roc_auc_score
    from interp import data_top as dt
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    RES.mkdir(parents=True, exist_ok=True)
    M = X.Models(); M.massless = True
    perm = H.WeaverNet('ParT', X.RES / 'h4_ckpt_part_kin.pt', M.dev); g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for prm in perm.model.parameters():
            fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=g).to(fl.device)])
    samples = {}
    it, yt = H60.tl_test_rows(); t = time.time(); full = np.load(dt.DST / 'test' / 'full.npy')
    P = np.asarray(full[it][..., :4], np.float32); del full; log(f'TL test rows read {time.time() - t:.0f}s')
    nt = np.minimum(np.load(dt.DST / 'test' / 'jet.npy')[it, 5].astype(np.int64), P.shape[1])
    samples['TL'] = (H._identity(P, nt).astype(np.float32), nt, yt[it].astype(int), np.random.default_rng(605).permutation(len(nt)))
    for s in ALL:
        G = load_gen(s); P, n = H60.massless_P4(G)
        h = (G['key'] * 2654435761) % 1_000_003                        # same pseudo-random batch order for paired jets
        samples[s] = (P, n, G['y'].astype(int), np.argsort(h, kind='stable'))
    out = {}
    for s, (P, n, y, order) in samples.items():
        O = H._obs(P, n); O.update(pt_feats(P, n))
        for k, v in O.items(): out[f'obs|{s}|{k}'] = np.asarray(v, np.float32)
        out[f'y|{s}'] = y.astype(np.int8); out[f'N|{s}'] = n
        t = time.time(); sc = M.score('top', P, n, X.top_base(P, n), order)
        sc['perm0'] = perm.score(P, n)
        sc['LNet_b1'] = M.top['LNet_top'].score(P, n, np.arange(len(n)), batch=1)
        for k, v in sc.items(): out[f'{s}|{k}'] = np.asarray(v, np.float32)
        log(f'{s}: {len(n)} jets, <N> top {n[y == 1].mean():.1f} QCD {n[y == 0].mean():.1f}, N>128 {np.mean(n > 128):.3f}, '
            f'{time.time() - t:.0f}s | ' + '; '.join(f'{k} acc {np.mean((v > 0) == (y == 1)):.4f} auc {roc_auc_score(y, v):.4f}' for k, v in sc.items()))
    np.savez_compressed(RES / 'h61_scores.npz', **out); log('wrote h61_scores.npz')


# ======================================================================================================== statistics
NETS = ['ParT_top', 'PN_top', 'LNet_top', 'LNet_b1', 'PFN_top', 'EFN_top', 'ParT_kin', 'PN_kin', 'perm0']
TESTNETS = ['ParT_top', 'PN_top', 'LNet_top', 'LNet_b1', 'PFN_top', 'ParT_kin', 'PN_kin']
REFS = ['lrN', 'gbdt_efp', 'gbdt_efpn']
OBS = ['N', 'm', 'mSD', 'tau32', 'tau21', 'mW', 'pTD']
C2F = ['N', 'm', 'tau32', 'pTD', 'f1', 'f2', 'f3', 'f5', 'f10', 'f20', 'f40', 'n_lt1', 'n_1_2', 'n_2_5']
COMPS = ['Hzp', 'Hzm', 'sigup', 'aup', 'split', 'merge', 'V0r']
HOLM_COMPS = ['Hzp', 'Hzm', 'sigup', 'aup', 'split', 'merge']
LAB = dict(ParT_top='ParT (Top)', PN_top='ParticleNet (Top)', LNet_top='LorentzNet (Top, batch 32)', LNet_b1='LorentzNet (batch 1)',
           PFN_top='PFN (Top)', EFN_top='EFN (Top, IRC-safe)', ParT_kin='ParT_kin (JetClass)', PN_kin='ParticleNet_kin (JetClass)',
           perm0='permuted ParT', lrN='LR(N) [count only]', gbdt_efp='GBDT(EFP) [IRC-safe]', gbdt_efpn='GBDT(EFP+N) [count at fixed EFPs]')


def step_stats(a):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    Zs = np.load(RES / 'h61_scores.npz'); Ze = np.load(RES / 'h61_efp.npz')
    samples = ['TL'] + ALL; S, y, O, key, npl = {}, {}, {}, {}, {}
    for s in samples:
        y[s] = Zs[f'y|{s}'].astype(int); assert (Ze[f'y|{s}'] == y[s]).all(), s
        S[s] = {m: Zs[f'{s}|{m}'].astype(np.float64) for m in NETS if f'{s}|{m}' in Zs.files}
        S[s].update({m: Ze[f'{s}|{m}'].astype(np.float64) for m in REFS})
        O[s] = {k: Zs[f'obs|{s}|{k}'].astype(np.float64) for k in OBS + C2F if k != 'N'}; O[s]['N'] = Zs[f'N|{s}'].astype(np.float64)
        if s != 'TL': G = load_gen(s); key[s] = G['key']; npl[s] = G['n_pl'].astype(float)
    models = [m for m in NETS + REFS if m in S['V0']]
    R = dict(node='H61', models=models, accuracy={}, n={}, paired={}, noise={}, wp={}, tests=[], share={}, c2st={},
             gen={s: json.loads((GEN / f'{s}.json').read_text()) for s in SETTINGS}, edits=json.loads((GEN / 'edits.json').read_text()),
             lrN=dict(coef=float(Ze['lrN_coef'][0]), intercept=float(Ze['lrN_coef'][1]), TL_train_N=Ze['TL_train_N'].tolist()))
    for s in samples:
        R['accuracy'][s] = {m: dict(acc=float(np.mean((S[s][m] > 0) == (y[s] == 1))), auc=float(roc_auc_score(y[s], S[s][m]))) for m in models}
        R['n'][s] = dict(top=int((y[s] == 1).sum()), qcd=int((y[s] == 0).sum()),
                         N_top=float(O[s]['N'][y[s] == 1].mean()), N_qcd=float(O[s]['N'][y[s] == 0].mean()))
        if s != 'TL': R['n'][s].update(N_pl_top=float(npl[s][y[s] == 1].mean()), N_pl_qcd=float(npl[s][y[s] == 0].mean()))
    rng = np.random.default_rng(6163); B = a.boot; mi = {m: i for i, m in enumerate(models)}
    pos0 = {int(k): j for j, k in enumerate(key['V0'])}; PAIRS = {}
    for v in COMPS:
        pr = np.array([(pos0[int(k)], j) for j, k in enumerate(key[v]) if int(k) in pos0]); PAIRS[v] = pr; R['paired'][v] = {}
        for cls, cn in ((1, 'top'), (0, 'qcd')):
            sel = y['V0'][pr[:, 0]] == cls; a0, a1 = pr[sel, 0], pr[sel, 1]; J = len(a0)
            D = np.stack([S[v][m][a1] - S['V0'][m][a0] for m in models], 1)
            Do = np.stack([O[v][k][a1] - O['V0'][k][a0] for k in OBS] + [npl[v][a1] - npl['V0'][a0]], 1)
            idx = rng.integers(0, J, (B, J)); bD = np.stack([D[ix].mean(0) for ix in idx]); bO = np.stack([Do[ix].mean(0) for ix in idx])
            d = D.mean(0)
            r = dict(n_pairs=int(J), obs={k: dict(d=float(Do[:, i].mean()), ci=np.percentile(bO[:, i], [2.5, 97.5]).tolist(),
                                                  rel=float(Do[:, i].mean() / np.mean((O['V0'][k] if k != 'N_pl' else npl['V0'])[a0])))
                                          for i, k in enumerate(OBS + ['N_pl'])}, models={})
            for m in models:
                i = mi[m]; rr = dict(d=float(d[i]), ci=np.percentile(bD[:, i], [2.5, 97.5]).tolist(), p0=H60._zp(d[i], bD[:, i].std()))
                if m in NETS and m != 'EFN_top':
                    j = mi['EFN_top']; dd = d[i] - d[j]; bdd = bD[:, i] - bD[:, j]
                    rr['minus_EFN'] = dict(d=float(dd), ci=np.percentile(bdd, [2.5, 97.5]).tolist(), p=H60._zp(dd, bdd.std()))
                    if m in TESTNETS and v in HOLM_COMPS:
                        R['tests'].append(dict(variant=v, cls=cn, model=m, d=float(dd), p=rr['minus_EFN']['p']))
                    l, e = mi['gbdt_efpn'], mi['gbdt_efp']
                    rr['kappa'] = dict(k=float(d[i] / d[l]), ci=np.percentile(bD[:, i] / bD[:, l], [2.5, 97.5]).tolist(), den=float(d[l]))
                    den = d[l] - d[e]; bden = bD[:, l] - bD[:, e]
                    rr['kappa_efpn_minus_efp'] = dict(k=float((d[i] - d[e]) / den), ci=np.percentile((bD[:, i] - bD[:, e]) / bden, [2.5, 97.5]).tolist(), den=float(den))
                r['models'][m] = rr
            R['paired'][v][cn] = r
            log(v, cn, f"pairs {J} dN {r['obs']['N']['d']:+.2f} ({100 * r['obs']['N']['rel']:+.1f} %) N_pl {r['obs']['N_pl']['d']:+.2f} dm {r['obs']['m']['d']:+.2f} "
                f"dtau32 {r['obs']['tau32']['d']:+.4f} dmW {r['obs']['mW']['d']:+.2f} | " + ' '.join(f"{m} {r['models'][m]['d']:+.3f}" for m in models))
            # working point: threshold at the V0 50 % top efficiency (median of the paired V0 tops); McNemar on paired calls
            wp = {}
            t0s = {m: S['V0'][m][pr[y['V0'][pr[:, 0]] == 1, 0]] for m in models}
            for m in models:
                th = np.quantile(t0s[m], 0.5); c0 = S['V0'][m][a0] > th; c1 = S[v][m][a1] > th; n01, n10, p = H60._mcnemar(c0, c1)
                wp[m] = dict(th=float(th), eff_v0=float(c0.mean()), eff_var=float(c1.mean()), calls_gained=n01, calls_lost=n10, p_mcnemar=p)
            R['wp'].setdefault(v, {})[cn] = wp
    adj = H60._holm([t['p'] for t in R['tests']])
    for t, h in zip(R['tests'], adj): t['holm'] = float(h)
    R['holm'] = dict(family=f'network - EFN paired shift, {len(TESTNETS)} networks x {len(HOLM_COMPS)} comparisons x 2 classes, bootstrap-z p',
                     n_tests=len(R['tests']), n_pass_005=int((adj < 0.05).sum()))
    # split share of the Hz+ shift, share removed by merge (same V0 keys)
    for cls, cn in ((1, 'top'), (0, 'qcd')):
        ph = PAIRS['Hzp']; ps = {int(key['split'][j]): j for j in range(len(key['split']))}; pm = {int(key['merge'][j]): j for j in range(len(key['merge']))}
        rows = [(i0, i1, ps[int(key['V0'][i0])], pm[int(key['V0'][i0])]) for i0, i1 in ph
                if y['V0'][i0] == cls and int(key['V0'][i0]) in ps and int(key['V0'][i0]) in pm]
        rows = np.array(rows); J = len(rows); bi = rng.integers(0, J, (B, J)); R['share'][cn] = {}
        for m in models:
            dh = S['Hzp'][m][rows[:, 1]] - S['V0'][m][rows[:, 0]]; ds = S['split'][m][rows[:, 2]] - S['V0'][m][rows[:, 0]]
            dm = S['merge'][m][rows[:, 3]] - S['V0'][m][rows[:, 0]]
            bh = np.array([dh[b].mean() for b in bi]); bs = np.array([ds[b].mean() for b in bi]); bm = np.array([dm[b].mean() for b in bi])
            R['share'][cn][m] = dict(n=int(J), split=float(ds.mean() / dh.mean()), split_ci=np.percentile(bs / bh, [2.5, 97.5]).tolist(),
                                     merge_removed=float(1 - dm.mean() / dh.mean()), merge_removed_ci=np.percentile(1 - bm / bh, [2.5, 97.5]).tolist())
        log('share', cn, {m: (round(v['split'], 2), round(v['merge_removed'], 2)) for m, v in R['share'][cn].items()})
    # generator-noise band: V0b - V0 (independent events, unpaired)
    for cls, cn in ((1, 'top'), (0, 'qcd')):
        r = {}
        for m in models + ['N']:
            x0 = (O['V0'][m] if m == 'N' else S['V0'][m])[y['V0'] == cls]; x1 = (O['V0b'][m] if m == 'N' else S['V0b'][m])[y['V0b'] == cls]
            bs = [x1[rng.integers(0, len(x1), len(x1))].mean() - x0[rng.integers(0, len(x0), len(x0))].mean() for _ in range(400)]
            r[m] = dict(d=float(x1.mean() - x0.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist())
        R['noise'][cn] = r
    # C2ST V0 (Delphes) vs Top Landscape test jets, per class
    for cls, cn in ((1, 'top'), (0, 'qcd')):
        for feats, tag in ((C2F, 'all'), ([k for k in C2F if k not in ('N', 'n_lt1', 'n_1_2', 'n_2_5')], 'no_counts'), (['N'], 'N_only')):
            F0 = np.stack([O['TL'][k][y['TL'] == cls] for k in feats], 1); F1 = np.stack([O['V0'][k][y['V0'] == cls] for k in feats], 1)
            Xc = np.r_[F0, F1]; yc = np.r_[np.zeros(len(F0)), np.ones(len(F1))]
            pc = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), Xc, yc, cv=2, method='predict_proba')[:, 1]
            R['c2st'][f'{cn}_{tag}'] = float(roc_auc_score(yc, pc))
        R['c2st'][f'{cn}_means'] = {k: dict(TL=float(O['TL'][k][y['TL'] == cls].mean()), V0=float(O['V0'][k][y['V0'] == cls].mean())) for k in C2F + ['mW', 'mSD']}
    log('C2ST', {k: v for k, v in R['c2st'].items() if not k.endswith('means')})
    (RES / 'h61_run.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h61_run.json')
    fig_shifts(R); fig_jet()


def fig_shifts(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    show = ['ParT_top', 'PN_top', 'LNet_top', 'LNet_b1', 'PFN_top', 'ParT_kin', 'PN_kin', 'perm0', 'EFN_top', 'gbdt_efp', 'lrN', 'gbdt_efpn']
    vs = [('Hzp', 'Hz+ (hadronisation, N up)', '#1f77b4'), ('Hzm', 'Hz- (hadronisation, N down)', '#17becf'),
          ('split', 'split softest (count only, N up)', '#d62728'), ('merge', 'Hz+ then merge back to V0 count', '#ff7f0e'),
          ('V0r', 'V0, Delphes reseeded (null)', '#7f7f7f')]
    fig, axs = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, cn, title in ((axs[0], 'top', 'top jets'), (axs[1], 'qcd', 'QCD jets')):
        for k, (v, lab, c) in enumerate(vs):
            r = R['paired'][v][cn]; yy = np.arange(len(show)) + 0.15 * (k - 2)
            d = np.array([r['models'][m]['d'] for m in show]); ci = np.array([r['models'][m]['ci'] for m in show])
            ax.errorbar(d, yy, xerr=[d - ci[:, 0], ci[:, 1] - d], fmt='o', color=c, ms=3.5,
                        label=f"{lab}: N {100 * r['obs']['N']['rel']:+.1f} %, m {r['obs']['m']['d']:+.2f} GeV")
        ax.axvline(0, color='k', lw=0.8); ax.axhline(7.5, color='0.6', lw=0.6, ls='--')
        ax.set_title(f'{title}: paired change in top log-odds (variant - V0)'); ax.set_xlabel('mean change in top log-odds')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, [x.split(':')[0] for x in l], loc='lower center', ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    axs[0].set_yticks(range(len(show))); axs[0].set_yticklabels([LAB[m] for m in show]); axs[0].invert_yaxis()
    fig.suptitle('Delphes ATLAS card, MPI off: taggers vs IRC-safe and count references (95 % bootstrap CI)')
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(RES / 'h61_shifts.png', dpi=130); log('wrote h61_shifts.png')


def fig_jet():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    G0, G1 = load_gen('V0'), load_gen('Hzp'); Zs = np.load(RES / 'h61_scores.npz')
    pos1 = {int(k): j for j, k in enumerate(G1['key'])}
    # truth check: share of V0 top jets where each of b, q, q' has a constituent with pT > 5 GeV within Delta R < 0.2
    hit = []
    for j in np.flatnonzero(G0['y'] == 1):
        X = G0['X'][j, :G0['n'][j]]; X = X[X[:, 0] > 5]
        hit.append([bool((H60._dr(X[:, 1], X[:, 2], e, f) < 0.2).any()) for e, f in G0['truth'][j, 1:4, :2]])
    log('truth check: b, q, q\' with a >5 GeV constituent within 0.2:', np.mean(hit, 0).round(3).tolist(), 'all three', float(np.mean(np.all(hit, 1))))
    cand = [j for j in np.flatnonzero(G0['y'] == 1) if int(G0['key'][j]) in pos1 and G1['n'][pos1[int(G0['key'][j])]] - G0['n'][j] >= 4
            and 160 < G0['jet'][j, 3] < 190 and 160 < G1['jet'][pos1[int(G0['key'][j])], 3] < 190]
    j0 = cand[len(cand) // 3]; j1 = pos1[int(G0['key'][j0])]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6), sharex=True, sharey=True)
    sty = {211: ('#1f77b4', 'track'), 22: ('#2ca02c', 'photon (ECAL)'), 130: ('#ff7f0e', 'neutral hadron (HCAL)')}
    for ax, G, j, s, nm in ((axs[0], G0, j0, 'V0', 'V0: Monash hadronisation'), (axs[1], G1, j1, 'Hzp', 'Hz+: aLund 0.79, bLund 0.88, sigma 0.247 GeV')):
        X = G['X'][j, :G['n'][j]]; jet = G['jet'][j]; T = G['truth'][j]
        de = X[:, 1] - jet[1]; dp = (X[:, 2] - jet[2] + np.pi) % (2 * np.pi) - np.pi; code = np.abs(X[:, 4])
        for c, (col, lab) in sty.items():
            mk = code == c
            ax.scatter(de[mk], dp[mk], s=4 + 600 * X[mk, 0] / X[:, 0].sum(), c=col, alpha=0.65, edgecolors='none', label=f'{lab} ({mk.sum()})')
        for (e, f), lb, mk in zip(T[1:4, :2], ('b', 'q', "q'"), ('s', '^', 'v')):
            ax.scatter(e - jet[1], (f - jet[2] + np.pi) % (2 * np.pi) - np.pi, marker=mk, s=110, facecolors='none', edgecolors='#d62728', lw=1.5)
            ax.annotate(lb, (e - jet[1], (f - jet[2] + np.pi) % (2 * np.pi) - np.pi), xytext=(6, 6), textcoords='offset points', color='#d62728')
        ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, ls='--', color='0.5'))
        ax.set_title(f'{nm}\nN = {G["n"][j]} Delphes objects, m = {jet[3]:.0f} GeV, ParT top log-odds {Zs[f"{s}|ParT_top"][j]:+.2f}', fontsize=9.5)
        ax.set_xlabel('eta - eta_jet'); ax.set_aspect('equal'); ax.legend(fontsize=7.5, loc='lower left')
    axs[0].set_ylabel('phi - phi_jet [rad]'); axs[0].set_xlim(-0.9, 0.9); axs[0].set_ylim(-0.9, 0.9)
    fig.suptitle("One Delphes top jet, same event, only Pythia hadronisation changed (area ~ pT; red: b, q, q' partons)")
    fig.tight_layout(); fig.savefig(RES / 'h61_jet.png', dpi=130); log('wrote h61_jet.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['check', 'gen', 'edits', 'efp', 'score', 'stats', 'figs'])
    ap.add_argument('--setting', default='V0'); ap.add_argument('--jets', type=int, default=5000)
    ap.add_argument('--cpu', type=int, default=4); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--rebuild_edits', type=int, default=0)
    a = ap.parse_args()
    dict(check=step_check, gen=step_gen, edits=lambda a: build_edits(), figs=lambda a: (fig_shifts(json.loads((RES / 'h61_run.json').read_text())), fig_jet()), efp=step_efp, score=step_score, stats=step_stats)[a.step](a)
