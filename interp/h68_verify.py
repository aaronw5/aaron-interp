"""H68 verifier (loop D): rerun of H68's key relabel arms on a fresh JetClass test file, plus controls H68 did not have.

Sample: JetClass test_20M/<sample>_101.root (H68 used _100), all 10 classes, 1,000 random rows per class (default_rng(6811)),
  N = 10,000. References trained on train_100M/<sample>_001.root (H68: _000), 2,000 random rows per class (default_rng(6812)).
Models: ParT_full, PN_full, ParT_kin, PN_kin, three weight-permuted ParT_full copies (torch seeds 7, 8, 9).
Repeated edits (as in H68, functions imported from interp.h68_particle_type_channel): fr05/fr10/fr20, frg10, ph2nh, nh2ph,
  ch2nh2, ch2g2, csame_fr10, ckin_fr10.
New controls:
  swap10   fraction-preserving label swap: the fr10 charged-hadron set -> neutral hadron (as fr10) AND a set of real neutral
           hadrons of the same jet (random order, prefix whose summed energy is closest to the moved set's energy) -> charged
           hadron (charge +-1 at random, track fields d0/dz and errors copied from a random real charged hadron of the sample in
           the same pT decile). f_ch and f_nh change only by the energy mismatch of the two sets. A composition reader (H1)
           predicts Delta s ~ slope x Delta f_ch (near 0); a reader of which particles carry which label predicts Delta s != 0.
  swapg10  the same with photons: fr10 set -> photon (as frg10) and an energy-matched set of real photons -> charged hadron.
  fr10m    fr10 with the relabelled particles made massless (E set to |p|, as real Delphes neutral hadrons are; jet_energy
           updated). Separates the label from the leftover pion mass of a relabelled track. Kin models see the mass change.
  ckin     per-jet RMS of Delta s reported next to the mean (a zero-mean jitter has ~0 mean response to first order).
  perm     Delta s divided by the s.d. over jets of each model's unedited score (logit-scale normalisation of the null).
  ghosts   rho = 10 composition ghosts (H68 arm 2) on the axis (gstack10) with mix / all-neutral-hadron / all-photon labels, and
           uniform with the same three labels: stacked share = Delta s(stack) / Delta s(uniform) per label.
Run: ./part_job.sh d-h68-verify 'python -m interp.h68_verify'  -> $AI_RESULTS/physics/h68_verify.json ; --smoke N
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h68_particle_type_channel as H  # noqa: E402
from interp.part_core import PRE  # noqa: E402

SEED, SEED_REF = 6811, 6812
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def load(per, seed, files):
    import uproot
    from interp.part_core import cat
    rng = np.random.default_rng(seed); Js = []
    for c in H.CLS:
        f = files(c); n = uproot.open(f)['tree'].num_entries; idx = np.sort(rng.choice(n, per, replace=False))
        Js.append(H.read_rows(f, idx)); log(f'{c}: {per} random rows of {n} ({f.name})')
    return cat(Js)


def matched_prefix(cand, E, target, rng):
    """cand particles in random order; the prefix whose summed energy is closest to target (per jet)"""
    key = np.where(cand, rng.random(cand.shape), np.inf); o = np.argsort(key, 1)
    Eo = np.where(np.take_along_axis(cand, o, 1), np.take_along_axis(E, o, 1), 0.0)
    cum = np.concatenate([np.zeros((len(E), 1)), np.cumsum(Eo, 1)], 1); k = np.abs(cum - target[:, None]).argmin(1)
    rank = np.empty_like(o); np.put_along_axis(rank, o, np.arange(o.shape[1])[None].repeat(len(o), 0), 1)
    return cand & (rank < k[:, None])


def to_charged(U, J, T, rng):
    """relabel set T as charged hadrons: charge +-1, track fields from real charged hadrons in the same pT decile"""
    m = J['mask'] > 0; pt = np.hypot(J['px'], J['py']); ch = m & (J['isChargedHadron'] > 0)
    edges = np.quantile(pt[ch], np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
    bch = np.clip(np.digitize(pt[ch], edges) - 1, 0, 9); pool = np.stack([J[k][ch] for k in H.TRK], 1)
    ti = np.nonzero(T); bt = np.clip(np.digitize(pt[ti], edges) - 1, 0, 9); trk = np.zeros((len(bt), 4))
    for b in range(10):
        s = bt == b; cand = np.flatnonzero(bch == b)
        if s.any(): trk[s] = pool[rng.choice(cand, int(s.sum()))]
    for k in ('isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon'): U[k][ti] = 0.0
    U['isChargedHadron'][ti] = 1.0; U['charge'][ti] = np.where(rng.random(len(bt)) < 0.5, -1.0, 1.0)
    for i, k in enumerate(H.TRK): U[k][ti] = trk[:, i]
    return U


def massless(U, sel):
    V = H.copyJ(U); p = np.sqrt(U['px'] ** 2 + U['py'] ** 2 + U['pz'] ** 2); s = sel & (U['mask'] > 0)
    V['energy'] = np.where(s, p, U['energy']).astype(np.float32)
    V['jet_energy'] = (U['jet_energy'] + ((V['energy'] - U['energy']) * (U['mask'] > 0)).sum(1)).astype(np.float32)
    return V


def main(smoke):
    from interp.h1_irc_panel import holm, binom_two_sided
    H.OUT.mkdir(parents=True, exist_ok=True); per = smoke or 1000; B = H.B
    J = load(per, SEED, lambda c: TEST / f'{PRE[c]}_101.root'); y = J['label'].astype(int)
    assert np.bincount(y, minlength=10).tolist() == [per] * 10
    m = J['mask'] > 0; F0 = H.fractions(J); E = J['energy'] * m; pt = np.hypot(J['px'], J['py'])
    Jr = load(smoke or 2000, SEED_REF, lambda c: Path(H.TRAIN) / f'{PRE[c]}_001.root'); yr = Jr['label'].astype(int)
    refs, rmodels = H.references(Jr, None, yr); del Jr
    sets = {f: H.frac_set(J, f / 100, np.random.default_rng([SEED, 2, f])) for f in (5, 10, 20)}
    sets['ch2'] = (J['isChargedHadron'] > 0) & m & (pt < 2.0)
    E10 = (E * sets[10]).sum(1)
    Tnh = matched_prefix(m & (J['isNeutralHadron'] > 0), E, E10, np.random.default_rng([SEED, 7]))
    Tph = matched_prefix(m & (J['isPhoton'] > 0), E, E10, np.random.default_rng([SEED, 8]))
    C = {'base': J, 'fr05': H.relabel(J, sets[5], 'nh'), 'fr10': H.relabel(J, sets[10], 'nh'), 'fr20': H.relabel(J, sets[20], 'nh'),
         'frg10': H.relabel(J, sets[10], 'g'), 'ph2nh': H.relabel(J, J['isPhoton'] > 0, 'nh'), 'nh2ph': H.relabel(J, J['isNeutralHadron'] > 0, 'g'),
         'ch2nh2': H.relabel(J, sets['ch2'], 'nh'), 'ch2g2': H.relabel(J, sets['ch2'], 'g')}
    C['csame_fr10'], fl = H.csame(J, sets[10], np.random.default_rng([SEED, 4]))
    C['ckin_fr10'] = H.ckin(J, sets[10], np.random.default_rng([SEED, 5]))
    C['swap10'] = to_charged(H.relabel(J, sets[10], 'nh'), J, Tnh, np.random.default_rng([SEED, 9]))
    C['swapg10'] = to_charged(H.relabel(J, sets[10], 'g'), J, Tph, np.random.default_rng([SEED, 10]))
    C['fr10m'] = massless(C['fr10'], sets[10])
    C['m10only'] = massless(J, sets[10])            # kinematic part of fr10m alone (labels unchanged)
    rel = [c for c in C if c != 'base']
    dF = {c: {k: H.fractions(C[c])[k] - F0[k] for k in F0} for c in rel}
    log('mean Delta f_ch / f_nh / f_g per edit', {c: [round(float(dF[c][k].mean()), 4) for k in ('f_ch', 'f_nh', 'f_g')] for c in rel})
    log('swap sets: mean n moved fr10', float(sets[10].sum(1).mean()), 'nh->ch', float(Tnh.sum(1).mean()), 'ph->ch', float(Tph.sum(1).mean()),
        '; jets with any neutral hadron', float((Tnh.sum(1) > 0).mean()))
    # ghosts (H68 arm 2, rho = 10)
    P = H.soft_pool(J); G = H.ghost_master(len(y), P, SEED); GH = {}
    for nm, lay, lab in (('gbase', 'uni', 'mix'), ('gmix10', 'uni', 'mix'), ('gnh10', 'uni', 'nh'), ('gph10', 'uni', 'ph'),
                         ('gstack10', 'st', 'mix'), ('gstacknh10', 'st', 'nh'), ('gstackph10', 'st', 'ph'), ('gclus10', 'cl', 'mix')):
        U, ng, gq = H.with_ghosts(J, G, 0 if nm == 'gbase' else 10, lay, lab); C[nm] = U; GH[nm] = dict(n=float(ng.mean()), pt=float(gq.mean()))
    log('soft composition', P['prop'], 'ghosts', GH)
    R0, _, _ = H.ref_scores(rmodels, refs, J, y); REF = {}
    for c in rel:
        R1, _, _ = H.ref_scores(rmodels, refs, C[c], y)
        REF[c] = {H.CLS[k]: {r: float((R1[r] - R0[r])[y == k].mean()) for r in R1} for k in range(1, 10)}
    # models
    nets = H.jc_models(0)
    import torch
    from interp.part_core import ParTNetwork, inputs
    from interp.p1v_verify_part import fwd
    for s in ((7, 8, 9) if not smoke else (7,)):
        netp = ParTNetwork('full'); g = torch.Generator().manual_seed(s)
        with torch.no_grad():
            for p in netp.model.parameters():
                if p.numel() > 1:
                    f = p.data.flatten().cpu(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p).to(p.device))
        nets[f'ParT_full_perm{s}'] = (lambda J_, net=netp: fwd(net, *inputs(J_, 'full'))[1])
    S = {}; acc = {}
    for mn, f in nets.items():
        t = time.time()
        for c in C:
            L = np.asarray(f(C[c]), np.float64); S[(mn, c)] = (H.scores(L, y), L.argmax(1))
        acc[mn] = float((S[(mn, 'base')][1] == y).mean()); acc[mn + '|gbase'] = float((S[(mn, 'gbase')][1] == y).mean())
        log(f'{mn}: acc base {acc[mn]:.4f} gbase {acc[mn + "|gbase"]:.4f} (H68 known {H.KNOWN.get(mn, "-")}) ({time.time() - t:.0f} s)')
    idx = {k: np.random.default_rng([SEED, 99, k]).integers(0, per, (B, per)) for k in range(10)}
    out = dict(sample=f'JetClass test_20M <sample>_101.root, {per} random rows per class (default_rng({SEED}))', n=len(y),
               reference_sample=f'train_100M <sample>_001.root, {smoke or 2000} per class (default_rng({SEED_REF}))', accuracy=acc,
               mean_fractions={k: float(v.mean()) for k, v in F0.items()}, ghosts=GH, soft_composition=P['prop'],
               dF={c: {k: {H.CLS[j]: float(dF[c][k][y == j].mean()) for j in range(10)} for k in dF[c]} for c in rel},
               n_moved=dict(fr10=float(sets[10].sum(1).mean()), swap_nh=float(Tnh.sum(1).mean()), swap_ph=float(Tph.sum(1).mean())),
               ref_delta=REF, cells={}, slopes={}, tests={}, base_sd={})
    fam = []
    for (mn, c), (s, a) in S.items():
        if c in ('base', 'gbase'): continue
        b0 = 'gbase' if c.startswith('g') else 'base'; s0, a0 = S[(mn, b0)]; d = s - s0
        for k in range(10):
            r = np.flatnonzero(y == k); dk = d[r]; bs = dk[idx[k]].mean(1); sd0 = float(s0[r].std())
            n01 = int(((a0[r] == k) & (a[r] != k)).sum()); n10 = int(((a0[r] != k) & (a[r] == k)).sum())
            cell = dict(ds=float(dk.mean()), ci=H.ci(bs), p=H.p_z(float(dk.mean()), bs), rms=float(np.sqrt((dk ** 2).mean())),
                        ds_over_sd=float(dk.mean() / max(sd0, 1e-9)), right_to_wrong=n01, wrong_to_right=n10, p_mcnemar=binom_two_sided(n01, n01 + n10))
            out['cells'][f'{mn}|{c}|{H.CLS[k]}'] = cell; out['base_sd'][f'{mn}|{H.CLS[k]}'] = sd0
            if mn in ('ParT_full', 'PN_full'): fam.append((f'{mn}|{c}|{H.CLS[k]}', cell['p']))
    h = holm([p for _, p in fam]); out['holm_ds'] = {k: float(v) for (k, _), v in zip(fam, h)}
    log(f'Holm: {len(fam)} tests, {sum(v < 0.05 for v in h)} with p_Holm < 0.05')
    for mn in [x for x in nets if not x.endswith(('8', '9'))]:
        for k in range(10):
            r = np.flatnonzero(y == k)
            X = np.stack([dF[c]['f_ch'][r] for c in ('fr05', 'fr10', 'fr20')], 1); Y = np.stack([(S[(mn, c)][0] - S[(mn, 'base')][0])[r] for c in ('fr05', 'fr10', 'fr20')], 1)
            sl = lambda ii: (X[ii] * Y[ii]).sum((-1, -2)) / np.maximum((X[ii] ** 2).sum((-1, -2)), 1e-12)
            est = float(sl(np.arange(per)[None])[0]); bs = np.array([sl(ii[None])[0] for ii in idx[k]])
            out['slopes'][f'{mn}|{H.CLS[k]}'] = dict(slope=est, ci=H.ci(bs), p=H.p_z(est, bs))
            # swap prediction under H1: slope x Delta f_ch per jet
            for sw in ('swap10', 'swapg10'):
                pred = est * dF[sw]['f_ch'][r]; dd = (S[(mn, sw)][0] - S[(mn, 'base')][0])[r] - pred; bsd = dd[idx[k]].mean(1)
                out['tests'][f'{mn}|{sw} - slope*dfch|{H.CLS[k]}'] = dict(diff=float(dd.mean()), pred=float(pred.mean()), ci=H.ci(bsd), p=H.p_z(float(dd.mean()), bsd))
            for a_, b_ in (('fr10', 'frg10'), ('fr10m', 'fr10'), ('fr10', 'swap10'), ('frg10', 'swapg10'), ('gnh10', 'gph10'),
                           ('gstack10', 'gmix10'), ('gstacknh10', 'gnh10'), ('gstackph10', 'gph10')):
                dd = (S[(mn, a_)][0] - S[(mn, b_)][0])[r]; bsd = dd[idx[k]].mean(1)
                out['tests'][f'{mn}|{a_} - {b_}|{H.CLS[k]}'] = dict(diff=float(dd.mean()), ci=H.ci(bsd), p=H.p_z(float(dd.mean()), bsd))
            for lab, st, un in (('mix', 'gstack10', 'gmix10'), ('nh', 'gstacknh10', 'gnh10'), ('ph', 'gstackph10', 'gph10')):
                a_ = (S[(mn, st)][0] - S[(mn, 'gbase')][0])[r]; b_ = (S[(mn, un)][0] - S[(mn, 'gbase')][0])[r]
                bsr = a_[idx[k]].mean(1) / b_[idx[k]].mean(1)
                out['tests'][f'{mn}|stack share {lab}|{H.CLS[k]}'] = dict(share=float(a_.mean() / b_.mean()), ci=H.ci(bsr))
    agree = {}
    for mn in ('ParT_full', 'PN_full'):
        for c in rel:
            agree[f'{mn}|{c}'] = int(sum(np.sign(out['cells'][f'{mn}|{c}|{H.CLS[k]}']['ds']) == np.sign(REF[c][H.CLS[k]]['gbdt']) for k in range(1, 10)))
            agree[f'{mn}|{c}|lr_fch'] = int(sum(np.sign(out['cells'][f'{mn}|{c}|{H.CLS[k]}']['ds']) == np.sign(REF[c][H.CLS[k]]['lr_fch']) for k in range(1, 10)))
    out['sign_agree'] = agree
    for mn in nets:
        log(f'{mn} Ds per class [' + ' '.join(H.CLS) + ']')
        for c in [x for x in C if x not in ('base', 'gbase')]:
            log(f'   {c:12s} ' + ' '.join(f'{out["cells"][f"{mn}|{c}|{H.CLS[k]}"]["ds"]:+.2f}' for k in range(10))
                + '   rms ' + ' '.join(f'{out["cells"][f"{mn}|{c}|{H.CLS[k]}"]["rms"]:.2f}' for k in range(10)))
    for c in rel: log(f'GBDT ref {c:12s} ' + ' '.join(f'{REF[c][H.CLS[k]]["gbdt"]:+.2f}' for k in range(1, 10))
                      + ' | LR fch ' + ' '.join(f'{REF[c][H.CLS[k]]["lr_fch"]:+.2f}' for k in range(1, 10)))
    for k_, v in out['tests'].items():
        if k_.startswith(('ParT_full|', 'PN_full|', 'ParT_kin|')) and ('swap' in k_ or 'share' in k_ or 'fr10m' in k_): log('test', k_, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    log('sign agreement', agree)
    s = '_smoke' if smoke else ''
    (H.OUT / f'h68_verify{s}.json').write_text(json.dumps(out, indent=1, default=float)); log(f'wrote h68_verify{s}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); main(ap.parse_args().smoke)
