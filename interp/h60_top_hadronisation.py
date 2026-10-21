"""H60 (loop D; candidate claim extending D22 to top tagging): when Pythia's hadronisation settings alone change, so that top and
QCD jets carry more or fewer particles at nearly fixed IRC-safe content (jet mass, tau32, W-pair mass, EFN output), do top taggers
move like the IRC-safe EFN (about zero) or like a particle-count classifier LR(N)?

Generator (Pythia 8.317, PyPI pythia8mc, venv /j-jepa-vol/aaron-interp/env-gen): pp at 14 TeV, Monash 2013 (Tune:pp 14), MPI,
ISR, FSR, hadronisation on.  Top: Top:gg2ttbar + Top:qqbar2ttbar, W -> quarks only (all-hadronic).  QCD: HardQCD:all.
PhaseSpace:pTHatMin 450, pTHatMax 800 GeV.  Jets: Pythia SlowJet (fjcore anti-kT) R = 0.8 on all visible final-state particles
(|eta| < 5, real masses); every jet with 550 < pT < 650 GeV and |eta| < 2 is kept.  Top jets: truth-matched as in the Top
Tagging Landscape (the hadronic top and its b, q, q' from the process record all within Delta R < 0.8 of the jet axis).  QCD
jets: the nearest outgoing hard parton within Delta R < 0.8 defines the pairing slot.  Particle level, no detector simulation.
Constituents are passed to every network as massless four-vectors (E = |p|), as the Top Landscape Delphes constituents are.

Pairing: before event i of worker w the generator is reseeded with seed(w, i) (as H54), so every setting starts each event from
the same random state; a hadronisation-only change leaves the hard process, shower and MPI identical event by event.  Jet pair
key = (worker, event, slot) with slot = which top (t / tbar) or which outgoing hard parton the jet belongs to.

Settings (StringZ:aLund, StringZ:bLund, StringPT:sigma [GeV]); Monash = (0.68, 0.98, 0.335)
  V0   nominal                                   V0b  nominal, other seeds (generator-noise band, also the training sample
                                                      for the particle-level references)
  Hzp  (0.79, 0.88, 0.247)  H54 Hz+ (QG: N +11 %)  Hzm  (0.48, 0.98, 0.438)  H54 Hz- (QG: N -11 %)
  sigup (0.68, 0.98, 0.235) sigma only, N up       sigdn (0.68, 0.98, 0.45)   sigma only, N down   (H54v single-knob controls)
  aup  (1.28, 0.98, 0.335)  aLund only, N up       adn  (0.18, 0.98, 0.335)   aLund only, N down

Steps
  --step check            (env-gen) API, timing, truth-match yield, parton-level identity under a hadronisation change, Delphes
  --step gen --setting S  (env-gen, 1 CPU) 5,000 top + 5,000 QCD jets -> data/top/gen_h60/<S>.npz
  --step efp              (venv-ef, 8 CPU) EFPs (d <= 5) of Top Landscape val (reference training), Top Landscape test sample,
                          V0b and all settings; GBDT(EFP), GBDT(EFP+N) trained on Top Landscape val and on V0b -> h60_efp.npz
  --step score            (GPU, env-part) all networks, permuted ParT, observables, LR / MLP references -> h60_scores.npz
  --step stats            (CPU) paired shifts, CIs, kappa, working points, McNemar, C2ST, Holm -> h60_run.json, figures
Score convention: s = top log-odds (log p(top) - log p(QCD); JetClass nets: logit(Tbqq) - logit(QCD)); positive = toward top.
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
GEN = V / 'data' / 'top' / 'gen_h60'
SEED = 6060
NOM = (0.68, 0.98, 0.335)
SETTINGS = {'V0': NOM, 'V0b': NOM, 'Hzp': (0.79, 0.88, 0.247), 'Hzm': (0.48, 0.98, 0.438),
            'sigup': (0.68, 0.98, 0.235), 'sigdn': (0.68, 0.98, 0.45), 'aup': (1.28, 0.98, 0.335), 'adn': (0.18, 0.98, 0.335)}
VARIANTS = ['Hzp', 'Hzm', 'sigup', 'sigdn', 'aup', 'adn']
WIDX = lambda setting, cls: (10 if setting == 'V0b' else 0) + cls        # worker id: cls 1 = top, 0 = QCD
LMAX = 200
R_JET = 0.8


# ======================================================================================================== generator
def _event_seed(widx, i):
    return 1 + (SEED + widx) * 50_000 + i


def _py8():
    try:
        import pythia8mc as m
    except ImportError:
        import pythia8 as m
    return m


def make_pythia(cls, extra, seed):
    p = _py8().Pythia('', False)
    base = ['Beams:eCM = 14000.', 'Tune:pp = 14', 'PartonLevel:MPI = on', 'PartonLevel:ISR = on', 'PartonLevel:FSR = on',
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


def _dr(e1, p1, e2, p2):
    return np.hypot(e1 - e2, (p1 - p2 + np.pi) % (2 * np.pi) - np.pi)


def _daus(pr, i):
    d1, d2 = pr[i].daughter1(), pr[i].daughter2()
    if d1 <= 0: return []
    return list(range(d1, d2 + 1)) if d2 >= d1 else [d1]


def _follow(pr, i):
    """bottom copy of entry i (single daughter with the same id)"""
    for _ in range(50):
        d = _daus(pr, i)
        if len(d) == 1 and pr[d[0]].id() == pr[i].id(): i = d[0]
        else: break
    return i


def truth_partons(p, cls):
    """top: list of (slot, [(eta, phi) of t, b, q, q']) per hadronic top; QCD: list of (slot, [(eta, phi, id)]) outgoing partons"""
    pr = p.process; out = []
    if cls == 1:
        for i in range(pr.size()):
            if pr[i].idAbs() != 6 or not (pr[i].status() in (-22, 22, -62, 62)): continue
            if pr[pr[i].mother1()].idAbs() == 6: continue           # a copy
            t = _follow(pr, i); d = _daus(pr, t); b = [k for k in d if pr[k].idAbs() == 5]; w = [k for k in d if pr[k].idAbs() == 24]
            if not b or not w: continue
            wq = [k for k in _daus(pr, _follow(pr, w[0])) if pr[k].idAbs() <= 5]
            if len(wq) != 2: continue
            out.append((0 if pr[i].id() > 0 else 1, [(pr[k].eta(), pr[k].phi()) for k in (t, b[0], wq[0], wq[1])]))
    else:
        k = 0
        for i in range(pr.size()):
            if pr[i].status() == 23:
                out.append((k, [(pr[i].eta(), pr[i].phi(), pr[i].id())])); k += 1
    return out


def jets_from_event(p, sj, cls):
    """-> list of (slot, particles (n,5) (pt, eta, phi, m, pid) pT-sorted, jet (pt, eta, phi, m), truth (4,2) or (1,3))"""
    ev = p.event
    if not sj.analyze(ev) or sj.sizeJet() == 0: return []
    tru = truth_partons(p, cls); res = []; used = set()
    for j in range(sj.sizeJet()):
        if sj.pT(j) < 540: break
        rows = []
        for i in sj.constituents(j):
            q = ev[i]; rows.append((q.pT(), q.eta(), q.phi(), q.m(), q.id()))
        a = np.array(rows, np.float64); a = a[np.argsort(-a[:, 0], kind='stable')]
        px = (a[:, 0] * np.cos(a[:, 2])).sum(); py = (a[:, 0] * np.sin(a[:, 2])).sum(); pz = (a[:, 0] * np.sinh(a[:, 1])).sum()
        E = np.sqrt((a[:, 0] * np.cosh(a[:, 1])) ** 2 + a[:, 3] ** 2).sum()
        pt = np.hypot(px, py); eta = np.arcsinh(pz / pt); phi = np.arctan2(py, px); m = np.sqrt(max(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))
        if not (550 <= pt <= 650 and abs(eta) < 2): continue
        if cls == 1:
            ok = [(s, P) for s, P in tru if all(_dr(eta, phi, e, f) < R_JET for e, f in P)]
            if not ok: continue
            slot, P = ok[0]; T = np.array(P, np.float64)
        else:
            if not tru: continue
            d = [_dr(eta, phi, P[0][0], P[0][1]) for _, P in tru]; k = int(np.argmin(d))
            if d[k] > R_JET: continue
            slot, P = tru[k]; T = np.array(P, np.float64)
        if slot in used: continue
        used.add(slot); res.append((slot, a, (pt, eta, phi, m), T))
    return res


def _parton_hash(p):
    ev = p.event; h = 0.0; n = 0
    for i in range(ev.size()):
        st = abs(ev[i].status())
        if 21 <= st <= 69: h += abs(ev[i].px()) + 2 * abs(ev[i].pz()); n += 1
    return round(h, 6), n


def gen_worker(task):
    pythia8 = _py8()
    cls, widx, extra = task['cls'], task['widx'], task['extra']
    p = make_pythia(cls, extra, _event_seed(widx, 0))
    sj = pythia8.SlowJet(-1, R_JET, 500.0, 5.0, 2, 2)
    X, J, T, K, ev_ok = [], [], [], [], 0; nev = nfail = 0; t0 = time.time()
    for i in range(task['max_events']):
        if len(X) >= task['n_jets']: break
        p.rndm.init(_event_seed(widx, i)); nev += 1
        if not p.next(): nfail += 1; continue
        got = jets_from_event(p, sj, cls); ev_ok += bool(got)
        for slot, a, jet, tr in got:
            X.append(a[:LMAX]); J.append(jet + (len(a),)); T.append(np.pad(tr, ((0, 4 - len(tr)), (0, 3 - tr.shape[1]))))
            K.append(widx * 10_000_000 + i * 10 + slot)
        if nev % 2000 == 0: log(f'[w{widx}] {nev} events, {len(X)} jets, {1e3 * (time.time() - t0) / nev:.1f} ms/event')
    n = np.array([len(a) for a in X]); Xa = np.zeros((len(X), int(n.max()), 5), np.float32)
    for j, a in enumerate(X): Xa[j, :len(a)] = a
    np.savez_compressed(task['out'], X=Xa, n=n, jet=np.array(J), truth=np.array(T), key=np.array(K, np.int64),
                        y=np.full(len(X), cls, np.int8))
    s = dict(cls=cls, widx=widx, jets=len(X), events=nev, failed=nfail, events_with_jet=ev_ok, sec=time.time() - t0,
             ms_per_event=1e3 * (time.time() - t0) / max(nev, 1), mean_N=float(np.mean([j[4] for j in J])), settings=extra)
    log(f'[worker {widx}] {s}')
    return s


def step_check(a):
    pythia8 = _py8()
    import importlib.metadata as md
    log('pythia8mc', md.version('pythia8mc'))
    try:
        import urllib.request             # 'conda search' was OOM-killed at 2 GiB (d-h60-check, 2026-10-21): query the API instead
        d = json.loads(urllib.request.urlopen('https://api.anaconda.org/package/conda-forge/delphes', timeout=30).read())
        log('conda-forge delphes:', d.get('versions', [])[-5:], 'latest', d.get('latest_version'))
    except Exception as e: log('delphes check failed', repr(e))
    for cls in (1, 0):
        p = make_pythia(cls, {}, 123); sj = pythia8.SlowJet(-1, R_JET, 500.0, 5.0, 2, 2); t = time.time(); nj = 0; nn = []
        for i in range(200):
            p.rndm.init(7000 + i); p.next(); g = jets_from_event(p, sj, cls); nj += len(g); nn += [len(x[1]) for x in g]
            if i < 2: log(f'cls {cls} ev {i}: truth {truth_partons(p, cls)}; jets kept {[(s, round(j[0], 1), len(x)) for s, x, j, _ in g]}')
        log(f'cls {cls}: 200 events {1e3 * (time.time() - t) / 200:.1f} ms/event, {nj} jets kept, <N> {np.mean(nn) if nn else 0:.1f}, '
            f'N>128 {np.mean(np.array(nn) > 128) if nn else 0:.3f}')
        pA = make_pythia(cls, {}, 7); pB = make_pythia(cls, settings_dict('Hzp'), 7); same = 0
        for i in range(20):
            pA.rndm.init(500 + i); pB.rndm.init(500 + i); pA.next(); pB.next(); same += _parton_hash(pA) == _parton_hash(pB)
        log(f'cls {cls}: parton level identical in {same}/20 events, V0 vs Hzp')


def step_gen(a):
    GEN.mkdir(parents=True, exist_ok=True); extra = settings_dict(a.setting); log(a.setting, extra)
    tasks = [dict(cls=c, widx=WIDX(a.setting, c), extra=extra, n_jets=a.jets, max_events=999_999 // 10,
                  out=str(GEN / f'tmp_{a.setting}_{c}.npz')) for c in (1, 0)]
    summ = [gen_worker(t) for t in tasks]                  # sequential: 1 CPU / 2 GiB job (NRP-exempt size; peak ~0.2 GiB)
    Z = [np.load(t['out']) for t in tasks]; L = max(z['X'].shape[1] for z in Z)
    out = {k: np.concatenate([z[k] for z in Z]) for k in ('n', 'jet', 'truth', 'key', 'y')}
    out['X'] = np.concatenate([np.pad(z['X'], ((0, 0), (0, L - z['X'].shape[1]), (0, 0))) for z in Z])
    np.savez_compressed(GEN / f'{a.setting}.npz', **out)
    (GEN / f'{a.setting}.json').write_text(json.dumps(dict(setting=a.setting, values=SETTINGS[a.setting], workers=summ), indent=1))
    for t in tasks: os.remove(t['out'])
    log('wrote', GEN / f'{a.setting}.npz', len(out['n']))


# ======================================================================================================== data helpers
def load_gen(name):
    Z = np.load(GEN / f'{name}.npz'); return {k: Z[k] for k in Z.files}


def massless_P4(G):
    """(J, L, 4) px, py, pz, E = |p| from stored (pt, eta, phi), pT-sorted, and multiplicities"""
    X = G['X'].astype(np.float64); pt, eta, phi = X[..., 0], X[..., 1], X[..., 2]
    P = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1) * (pt > 0)[..., None]
    return P.astype(np.float32), G['n'].astype(np.int64)


def tl_test_rows(per_class=5000, seed=6061):
    from interp import data_top as dt
    y = np.load(dt.DST / 'test' / 'label.npy'); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in (0, 1)])), y


def tl_val_rows(per_class=25000, seed=4194):
    """identical rows to h19_wpair_cross.ref_train_top (rng 4194, 25k per class)"""
    from interp import data_top as dt
    y = np.load(dt.DST / 'val' / 'label.npy'); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in (0, 1)])), y


def _ptetaphi(P, n):
    pt = np.hypot(P[:n, 0], P[:n, 1]); eta = np.arcsinh(P[:n, 2] / np.maximum(pt, 1e-12)); phi = np.arctan2(P[:n, 1], P[:n, 0])
    o = np.argsort(-pt, kind='stable'); return np.c_[pt, eta, phi][o]


# ======================================================================================================== EFP references
def step_efp(a):
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import data_top as dt
    from interp import h14_qg_generator as H14
    pool = Pool(a.cpu); out = {}
    iv, yv = tl_val_rows(); t = time.time(); fv = np.load(dt.DST / 'val' / 'full.npy')   # whole file, sequential (random memmap rows are slow on CephFS)
    Pv = np.asarray(fv[iv][..., :4], np.float64); del fv
    nv = np.minimum(np.load(dt.DST / 'val' / 'jet.npy')[iv, 5].astype(np.int64), Pv.shape[1]); log(f'TL val rows {len(iv)} read {time.time() - t:.0f}s')
    Lv = [_ptetaphi(Pv[j], nv[j]) for j in range(len(iv))]; del Pv
    Fv = H14.efps(Lv, pool); log('EFPs TL val', Fv.shape)
    it, yt = tl_test_rows(); ft = np.load(dt.DST / 'test' / 'full.npy'); Pt = np.asarray(ft[it][..., :4], np.float64); del ft
    nt = np.minimum(np.load(dt.DST / 'test' / 'jet.npy')[it, 5].astype(np.int64), Pt.shape[1])
    Lt = [_ptetaphi(Pt[j], nt[j]) for j in range(len(it))]; del Pt
    samples = {'TL': (Lt, nt.astype(float), yt[it].astype(int))}
    for s in SETTINGS:
        G = load_gen(s); P, n = massless_P4(G)
        samples[s] = ([_ptetaphi(P[j].astype(np.float64), n[j]) for j in range(len(n))], n.astype(float), G['y'].astype(int))
    F = {s: H14.efps(v[0], pool) for s, v in samples.items()}; log('EFPs done', {s: f.shape for s, f in F.items()})
    feat = lambda F_, N, wn: np.c_[np.log(np.maximum(F_, 1e-12)), N] if wn else np.log(np.maximum(F_, 1e-12))
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15, random_state=0)
    yvv = yv[iv].astype(int)
    clf = {'gbdt_efp': gb().fit(feat(Fv, nv, False), yvv), 'gbdt_efpn': gb().fit(feat(Fv, nv, True), yvv)}
    Nb, yb = samples['V0b'][1], samples['V0b'][2]
    clf['gbdt_efp_pl'] = gb().fit(feat(F['V0b'], Nb, False), yb); clf['gbdt_efpn_pl'] = gb().fit(feat(F['V0b'], Nb, True), yb)
    for s, (L, N, y) in samples.items():
        for k, c in clf.items():
            out[f'{s}|{k}'] = c.decision_function(feat(F[s], N, k.startswith('gbdt_efpn'))).astype(np.float32)
        out[f'y|{s}'] = y.astype(np.int8)
        log(s, ' '.join(f'{k} acc {np.mean((out[f"{s}|{k}"] > 0) == (y == 1)):.4f}' for k in clf))
    np.savez_compressed(RES / 'h60_efp.npz', **out); pool.close(); log('wrote h60_efp.npz')


# ======================================================================================================== scoring
def step_score(a):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from interp import data_top as dt
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    RES.mkdir(parents=True, exist_ok=True)
    M = X.Models(); M.massless = True
    perms = {}
    for k in range(3):                                   # fresh copies of the Top ParT checkpoint, every tensor permuted (seed k)
        w = H.WeaverNet('ParT', X.RES / 'h4_ckpt_part_kin.pt', M.dev); g = torch.Generator().manual_seed(k)
        with torch.no_grad():
            for prm in w.model.parameters():
                fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=g).to(fl.device)])
        perms[f'perm{k}'] = w
    samples = {}
    it, yt = tl_test_rows(); t = time.time(); full = np.load(dt.DST / 'test' / 'full.npy')
    P = np.asarray(full[it][..., :4], np.float32); del full; log(f'TL test rows read {time.time() - t:.0f}s')
    nt = np.minimum(np.load(dt.DST / 'test' / 'jet.npy')[it, 5].astype(np.int64), P.shape[1])
    samples['TL'] = (H._identity(P, nt).astype(np.float32), nt, yt[it].astype(int), np.random.default_rng(605).permutation(len(nt)))
    for s in SETTINGS:
        G = load_gen(s); P, n = massless_P4(G)
        h = (G['key'] * 2654435761) % 1_000_003                        # same pseudo-random order for paired jets in every setting
        samples[s] = (P, n, G['y'].astype(int), np.argsort(h, kind='stable'))
    # references fitted on Top Landscape val (25k per class, rng 4194, = h19 ref rows) and on the particle-level V0b sample
    Xr, Or, yr = X.ref_train_top()
    E3 = [np.linspace(0, 300, 16), np.linspace(0, 1, 21), np.linspace(0, 200, 11)]
    lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    out = {}; obs = {}
    for s, (P, n, y, order) in samples.items():
        O = H._obs(P, n); obs[s] = O
        for k in ('m', 'mSD', 'tau32', 'tau21', 'mW', 'n'): out[f'obs|{s}|{k}'] = O[k].astype(np.float32)
        out[f'y|{s}'] = y.astype(np.int8); out[f'N|{s}'] = n
    def fit_refs(O, y, N):
        R = {'lr3': H._lr_fit(np.stack([O['mSD'], O['tau32'], O['mW']], 1), y, E3)}
        Z = np.stack([O['m'], O['tau32'], O['mW']], 1); sc = StandardScaler().fit(Z)
        R['mlp3'] = (MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=0, max_iter=200).fit(sc.transform(Z), y), sc)
        R['lrN'] = LogisticRegression().fit(N[:, None].astype(float), y)
        return R
    def eval_refs(R, O, N):
        r = dict(lr3=H._lr_eval(R['lr3'], np.stack([O['mSD'], O['tau32'], O['mW']], 1), E3))
        mlp, sc = R['mlp3']; r['mlp3'] = lo(mlp.predict_proba(sc.transform(np.stack([O['m'], O['tau32'], O['mW']], 1)))[:, 1])
        r['lrN'] = R['lrN'].decision_function(N[:, None].astype(float)); return r
    REF = {'': fit_refs(Or, yr, Or['n']), '_pl': fit_refs(obs['V0b'], samples['V0b'][2], samples['V0b'][1].astype(float))}
    info = {f'lrN{k}': dict(coef=float(R['lrN'].coef_[0, 0]), intercept=float(R['lrN'].intercept_[0])) for k, R in REF.items()}
    info['tl_val_N_mean'] = dict(top=float(Or['n'][yr == 1].mean()), qcd=float(Or['n'][yr == 0].mean()))
    log('refs', info)
    for s, (P, n, y, order) in samples.items():
        for k, R in REF.items():
            for nm, v in eval_refs(R, obs[s], n).items(): out[f'{s}|{nm}{k}'] = np.asarray(v, np.float32)
        t = time.time(); sc = {}
        r = M.score('top', P, n, X.top_base(P, n), order); sc.update(r)
        for k, w in perms.items(): sc[k] = w.score(P, n)
        sc['LNet_b1'] = M.top['LNet_top'].score(P, n, np.arange(len(n)), batch=1) if a.lnet_b1 else np.zeros(len(n))
        for k, v in sc.items(): out[f'{s}|{k}'] = np.asarray(v, np.float32)
        from sklearn.metrics import roc_auc_score
        log(f'{s}: {len(n)} jets, <N> top {n[y == 1].mean():.1f} QCD {n[y == 0].mean():.1f}, N>128 {np.mean(n > 128):.3f}, '
            f'{time.time() - t:.0f}s | ' + '; '.join(f'{k} acc {np.mean((v > 0) == (y == 1)):.4f} auc {roc_auc_score(y, v):.4f}' for k, v in sc.items()))
    np.savez_compressed(RES / 'h60_scores.npz', **out); (RES / 'h60_refs_info.json').write_text(json.dumps(info, indent=1))
    log('wrote h60_scores.npz')


# ======================================================================================================== statistics
NETS = ['ParT_top', 'PN_top', 'LNet_top', 'LNet_b1', 'PFN_top', 'EFN_top', 'ParT_kin', 'PN_kin', 'perm0', 'perm1', 'perm2']
REFS = ['lrN', 'lr3', 'mlp3', 'gbdt_efp', 'gbdt_efpn', 'lrN_pl', 'lr3_pl', 'mlp3_pl', 'gbdt_efp_pl', 'gbdt_efpn_pl']
OBS = ['N', 'm', 'mSD', 'tau32', 'tau21', 'mW']
LAB = dict(ParT_top='ParT (Top)', PN_top='ParticleNet (Top)', LNet_top='LorentzNet (Top)', LNet_b1='LorentzNet, batch 1',
           PFN_top='PFN (Top)', EFN_top='EFN (Top, IRC-safe)', ParT_kin='ParT_kin (JetClass)', PN_kin='ParticleNet_kin (JetClass)',
           perm='permuted ParT (3)', lrN='LR(N)', lr3='LR(m_SD, tau32, mW)', mlp3='MLP3 (m, tau32, mW)', gbdt_efp='GBDT(EFP)',
           gbdt_efpn='GBDT(EFP+N)')


def _zp(d, se):
    from scipy.stats import norm
    return float(2 * norm.sf(abs(d) / se)) if se > 0 else (1.0 if abs(d) < 1e-12 else 0.0)


def _mcnemar(a, b):
    from scipy.stats import binomtest
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    return n01, n10, (float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0)


def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def step_stats(a):
    from sklearn.metrics import roc_auc_score
    Zs = np.load(RES / 'h60_scores.npz'); Ze = np.load(RES / 'h60_efp.npz')
    samples = ['TL'] + list(SETTINGS); S, y, O, key = {}, {}, {}, {}
    for s in samples:
        y[s] = Zs[f'y|{s}'].astype(int); assert (Ze[f'y|{s}'] == y[s]).all(), s
        S[s] = {m: Zs[f'{s}|{m}'].astype(np.float64) for m in NETS + REFS if f'{s}|{m}' in Zs.files}
        S[s].update({m: Ze[f'{s}|{m}'].astype(np.float64) for m in REFS if f'{s}|{m}' in Ze.files})
        O[s] = {k: Zs[f'obs|{s}|{k if k != "N" else "n"}'].astype(np.float64) for k in OBS}
        O[s]['N'] = Zs[f'N|{s}'].astype(np.float64)
        if s != 'TL': key[s] = load_gen(s)['key']
    models = [m for m in NETS + REFS if m in S['V0']]
    R = dict(models=models, accuracy={}, paired={}, noise={}, wp={}, tests=[], tests_efpn=[], c2st={}, n={})
    for s in samples:
        R['accuracy'][s] = {m: dict(acc=float(np.mean((S[s][m] > 0) == (y[s] == 1))), auc=float(roc_auc_score(y[s], S[s][m])))
                            for m in models if m in S[s]}
        R['n'][s] = dict(top=int((y[s] == 1).sum()), qcd=int((y[s] == 0).sum()))
    rng = np.random.default_rng(SEED + 9); B = a.boot
    # paired shifts per variant and class
    pos0 = {int(k): j for j, k in enumerate(key['V0'])}
    for v in VARIANTS:
        pr = [(pos0[int(k)], j) for j, k in enumerate(key[v]) if int(k) in pos0]
        i0 = np.array([p[0] for p in pr]); i1 = np.array([p[1] for p in pr]); R['paired'][v] = {}
        for cls, cn in ((1, 'top'), (0, 'qcd')):
            sel = y['V0'][i0] == cls; a0, a1 = i0[sel], i1[sel]; J = len(a0)
            D = np.stack([S[v][m][a1] - S['V0'][m][a0] for m in models], 1)
            Do = np.stack([O[v][k][a1] - O['V0'][k][a0] for k in OBS], 1)
            idx = rng.integers(0, J, (B, J)); bD = np.stack([D[ix].mean(0) for ix in idx]); bO = np.stack([Do[ix].mean(0) for ix in idx])
            d = D.mean(0); mi = {m: i for i, m in enumerate(models)}
            r = dict(n_pairs=int(J), obs={k: dict(d=float(Do[:, i].mean()), ci=np.percentile(bO[:, i], [2.5, 97.5]).tolist(),
                                                  rel=float(Do[:, i].mean() / O['V0'][k][a0].mean()), sd_v0=float(O['V0'][k][a0].std()))
                                          for i, k in enumerate(OBS)}, models={})
            for m in models:
                i = mi[m]; rr = dict(d=float(d[i]), ci=np.percentile(bD[:, i], [2.5, 97.5]).tolist(), p0=_zp(d[i], bD[:, i].std()))
                if m in NETS:
                    for ref in ('EFN_top', 'gbdt_efp', 'gbdt_efp_pl'):
                        if ref == m: continue
                        j = mi[ref]; dd = d[i] - d[j]; bdd = bD[:, i] - bD[:, j]
                        rr[f'minus_{ref}'] = dict(d=float(dd), ci=np.percentile(bdd, [2.5, 97.5]).tolist(), p=_zp(dd, bdd.std()))
                        if ref == 'EFN_top' and not m.startswith('perm'):
                            R['tests'].append(dict(variant=v, cls=cn, model=m, ref=ref, d=float(dd), p=rr[f'minus_{ref}']['p']))
                    for ref in ('gbdt_efpn', 'gbdt_efpn_pl'):
                        j = mi[ref]; dd = d[i] - d[j]; bdd = bD[:, i] - bD[:, j]
                        rr[f'minus_{ref}'] = dict(d=float(dd), ci=np.percentile(bdd, [2.5, 97.5]).tolist(), p=_zp(dd, bdd.std()))
                        if ref == 'gbdt_efpn_pl' and not m.startswith('perm'):
                            R['tests_efpn'].append(dict(variant=v, cls=cn, model=m, ref=ref, d=float(dd), p=rr[f'minus_{ref}']['p']))
                    # kappa_EFP+N: share of the conditional count response (GBDT(EFP+N) - GBDT(EFP), same training set) followed
                    for e_, n_ in (('gbdt_efp', 'gbdt_efpn'), ('gbdt_efp_pl', 'gbdt_efpn_pl')):
                        e, l = mi[e_], mi[n_]; den = d[l] - d[e]; bden = bD[:, l] - bD[:, e]
                        k_ = (d[i] - d[e]) / den; bk = (bD[:, i] - bD[:, e]) / bden
                        rr[f'kappa_{n_}'] = dict(k=float(k_), ci=np.percentile(bk, [2.5, 97.5]).tolist(), den=float(den))
                    for lr in ('lrN', 'lrN_pl'):
                        e, l = mi['EFN_top'], mi[lr]; den = d[l] - d[e]; bden = bD[:, l] - bD[:, e]
                        k_ = (d[i] - d[e]) / den; bk = (bD[:, i] - bD[:, e]) / bden
                        rr[f'kappa_{lr}'] = dict(k=float(k_), ci=np.percentile(bk, [2.5, 97.5]).tolist(), den=float(den))
                r['models'][m] = rr
            R['paired'][v][cn] = r
            log(v, cn, f"pairs {J} dN {r['obs']['N']['d']:+.2f} ({100 * r['obs']['N']['rel']:+.1f} %) dm {r['obs']['m']['d']:+.2f} "
                f"dtau32 {r['obs']['tau32']['d']:+.4f} dmW {r['obs']['mW']['d']:+.2f} | " + ' '.join(f"{m} {r['models'][m]['d']:+.3f}" for m in models))
            # working points: thresholds from the paired V0 tops; QCD calls McNemar (paired), top efficiency change
            wp = {}
            for m in models:
                t0 = S['V0'][m][i0[y['V0'][i0] == 1]]; t1 = S[v][m][i1[y['V0'][i0] == 1]]
                q0 = S['V0'][m][i0[y['V0'][i0] == 0]]; q1 = S[v][m][i1[y['V0'][i0] == 0]]; w_ = {}
                for eps in (0.3, 0.5):
                    th = np.quantile(t0, 1 - eps); c0, c1 = q0 > th, q1 > th
                    n01, n10, p = _mcnemar(c0, c1)
                    w_[str(eps)] = dict(rej_v0=float(1 / max(c0.mean(), 1 / len(c0))), rej_var=float(1 / max(c1.mean(), 1 / len(c1))),
                                        epsS_var=float((t1 > th).mean()), qcd_calls_up=n01, qcd_calls_down=n10, p_mcnemar_qcd=p)
                wp[m] = w_
            R['wp'].setdefault(v, {})[cn] = wp if cn == 'top' else None
    adj2 = _holm([t['p'] for t in R['tests_efpn']])
    for t, h in zip(R['tests_efpn'], adj2): t['holm'] = float(h)
    R['holm_efpn'] = dict(family='network - GBDT(EFP+N) [V0b-trained] paired shift, 8 networks x 6 settings x 2 classes',
                          n_tests=len(R['tests_efpn']), n_pass_005=int((adj2 < 0.05).sum()))
    adj = _holm([t['p'] for t in R['tests']])
    for t, h in zip(R['tests'], adj): t['holm'] = float(h)
    R['holm'] = dict(family='network - EFN paired shift, networks (8, permuted excluded) x 6 settings x 2 classes, bootstrap z p',
                     n_tests=len(R['tests']), n_pass_005=int((adj < 0.05).sum()))
    # generator-noise band: V0b - V0 (independent events)
    for cls, cn in ((1, 'top'), (0, 'qcd')):
        r = {}
        for m in models + OBS:
            x0 = (O['V0'][m] if m in OBS else S['V0'][m])[y['V0'] == cls]; x1 = (O['V0b'][m] if m in OBS else S['V0b'][m])[y['V0b'] == cls]
            bs = [x1[rng.integers(0, len(x1), len(x1))].mean() - x0[rng.integers(0, len(x0), len(x0))].mean() for _ in range(400)]
            r[m] = dict(d=float(x1.mean() - x0.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist(), in_sample_ref=m.endswith('_pl'))
        R['noise'][cn] = r
    # C2ST V0 vs Top Landscape test jets, per class (HGB on N, m, mSD, tau21, tau32, mW; 2-fold AUC)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    for cls, cn in ((1, 'top'), (0, 'qcd')):
        for feats, tag in ((OBS, 'all'), ([k for k in OBS if k != 'N'], 'no_N')):
            F0 = np.stack([O['TL'][k][y['TL'] == cls] for k in feats], 1); F1 = np.stack([O['V0'][k][y['V0'] == cls] for k in feats], 1)
            Xc = np.r_[F0, F1]; yc = np.r_[np.zeros(len(F0)), np.ones(len(F1))]
            pc = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), Xc, yc, cv=2, method='predict_proba')[:, 1]
            R['c2st'][f'{cn}_{tag}'] = float(roc_auc_score(yc, pc))
        R['c2st'][f'{cn}_means'] = {k: dict(TL=float(O['TL'][k][y['TL'] == cls].mean()), V0=float(O['V0'][k][y['V0'] == cls].mean())) for k in OBS}
    log('C2ST', R['c2st'])
    (RES / 'h60_run.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h60_run.json')
    fig_shifts(R); fig_jet()


def fig_shifts(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    show = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'ParT_kin', 'PN_kin', 'perm0', 'EFN_top', 'gbdt_efp_pl', 'mlp3', 'lrN', 'gbdt_efpn_pl']
    vs = [('Hzp', 'Hz+ (N up)', '#1f77b4'), ('Hzm', 'Hz- (N down)', '#17becf'), ('sigup', 'sigma only, N up', '#d62728'),
          ('aup', 'aLund only, N up', '#ff7f0e')]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    for ax, cn, title in ((axs[0], 'top', 'top jets'), (axs[1], 'qcd', 'QCD jets')):
        for k, (v, lab, c) in enumerate(vs):
            r = R['paired'][v][cn]
            yy = np.arange(len(show)) + 0.18 * (k - 1.5)
            d = np.array([r['models'][m]['d'] for m in show]); ci = np.array([r['models'][m]['ci'] for m in show])
            ax.errorbar(d, yy, xerr=[d - ci[:, 0], ci[:, 1] - d], fmt='o', color=c, ms=4,
                        label=f"{lab}: N {100 * r['obs']['N']['rel']:+.0f} %, m {r['obs']['m']['d']:+.1f} GeV")
        ax.axvline(0, color='k', lw=0.8); ax.axhline(6.5, color='0.6', lw=0.6, ls='--')
        ax.set_title(f'{title}: paired score shift (variant - nominal)'); ax.set_xlabel('mean change in top log-odds')
        ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2)
    lab = dict(LAB, perm0='permuted ParT (Top)', gbdt_efp_pl='GBDT(EFP) [ref, IRC-safe]', mlp3='MLP(m, tau32, mW) [ref]',
               lrN='LR(N) alone [ref]', gbdt_efpn_pl='GBDT(EFP+N) [ref, N at fixed EFPs]', EFN_top='EFN (Top, IRC-safe) [ref]')
    axs[0].set_yticks(range(len(show))); axs[0].set_yticklabels([lab[m] for m in show]); axs[0].invert_yaxis()
    fig.suptitle('Pythia hadronisation-only changes: Top Landscape / JetClass taggers vs IRC-safe and count references (95 % CI)')
    fig.tight_layout(); fig.savefig(RES / 'h60_shifts.png', dpi=130); log('wrote h60_shifts.png')


def fig_jet():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    G0, G1 = load_gen('V0'), load_gen('Hzp'); Zs = np.load(RES / 'h60_scores.npz')
    pos1 = {int(k): j for j, k in enumerate(G1['key'])}
    cand = [j for j in np.flatnonzero(G0['y'] == 1) if int(G0['key'][j]) in pos1]
    j0 = cand[len(cand) // 3]; j1 = pos1[int(G0['key'][j0])]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, G, j, s, nm in ((axs[0], G0, j0, 'V0', 'nominal (Monash)'), (axs[1], G1, j1, 'Hzp', 'Hz+ (aLund 0.79, bLund 0.88, sigma 0.247 GeV)')):
        X = G['X'][j, :G['n'][j]]; jet = G['jet'][j]; T = G['truth'][j]
        de = X[:, 1] - jet[1]; dp = (X[:, 2] - jet[2] + np.pi) % (2 * np.pi) - np.pi
        ax.scatter(de, dp, s=4 + 400 * X[:, 0] / X[:, 0].sum(), c='#1f77b4', alpha=0.6, edgecolors='none')
        for (e, f), lb, mk in zip(T[1:4, :2], ('b', 'q', "q'"), ('s', '^', 'v')):
            ax.scatter(e - jet[1], (f - jet[2] + np.pi) % (2 * np.pi) - np.pi, marker=mk, s=110, facecolors='none', edgecolors='#d62728', lw=1.6)
            ax.annotate(lb, (e - jet[1], (f - jet[2] + np.pi) % (2 * np.pi) - np.pi), xytext=(6, 6), textcoords='offset points', color='#d62728')
        ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, ls='--', color='0.5'))
        sc = Zs[f'{s}|ParT_top'][j]
        ax.set_title(f'{nm}\nN = {G["n"][j]} particles, m = {jet[3]:.0f} GeV, ParT top log-odds {sc:+.2f}', fontsize=9.5)
        ax.set_xlabel('eta - eta_jet'); ax.set_aspect('equal')
    axs[0].set_ylabel('phi - phi_jet'); axs[0].set_xlim(-0.9, 0.9); axs[0].set_ylim(-0.9, 0.9)
    fig.suptitle('One top jet, same event, only hadronisation settings changed (marker area ~ particle pT; red = b, q, q\' partons)')
    fig.tight_layout(); fig.savefig(RES / 'h60_jet.png', dpi=130); log('wrote h60_jet.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['check', 'gen', 'efp', 'score', 'stats'])
    ap.add_argument('--setting', default='V0'); ap.add_argument('--jets', type=int, default=5000)
    ap.add_argument('--cpu', type=int, default=8); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--lnet_b1', type=int, default=1)
    a = ap.parse_args()
    dict(check=step_check, gen=step_gen, efp=step_efp, score=step_score, stats=step_stats)[a.step](a)
