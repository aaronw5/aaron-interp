"""H1 verification (loop D). Fresh samples, the key H1 edits, and controls the H1 run did not have.
Uses the edit functions of interp.h1_irc_panel unchanged.

Samples (fresh seeds): hls4ml hls_sample(2000, seed 411) = 10,000 jets, 64 slots; JetClass jc_sample(1000, seed 412) = 10,000 jets.
New checks
  N0 null floors: (a) the same input evaluated twice; (b) the unedited table passed through finalize() (re-sort + cut) only;
     (c) JetClass: the same input evaluated with batch size 97 instead of 256 (other batch composition and padding length).
  S  sortedness of the stored particle order (rank_slot / soft() assume slot 0 is the hardest particle).
  E1 soft-emission eps sweep 1e-6 .. 1e-3 (H1 stopped at 1e-4), each with its recoil-only control.
  E2 soft position: core (r < 0.05 about the axis) vs edge (0.4 < r < 0.8) at eps 1e-4 and 1e-6. A pure counter responds the same.
  E3 count restore: drop the softest real particle, then add a soft particle (eps 1e-4 / 1e-6, uniform disc R < 0.8), vs the drop
     alone. A model that counts recovers part of its drop response (the count is restored); an IRC-safe model does not
     (the added particle carries ~no momentum). Measured as % calls changed vs unedited and as the fraction of drop-changed
     calls that return to the unedited call.
  Key H1 repeats: split_r0_z0.05_dR0 vs reshuffle_r0_f0.05, split_r0_z0.5_dR0 vs reshuffle_r0_f0.5, multi_k8.
Run
  AI_PY=conda ./job.sh d-h1-verify-s 1 3 'python -m interp.h1_verify --part small'
  ./job.sh d-h1-verify-j 'python -m interp.h1_verify --part jedi'
  ./part_job.sh d-h1-verify 'python -m interp.h1_verify --part jetclass'
  -> $AI_RESULTS/physics/h1_verify_{part}.json
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.h1_irc_panel as H  # noqa: E402

T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def soft_ring(T, eps, rmin, rmax, rng):
    """soft particle pT = eps * pT_jet uniform in the annulus rmin < r < rmax about the axis; recoil from slot 0"""
    J = len(T['pt']); rows = np.arange(J)
    px, py, pz = H.p3(T['pt'], T['eta'], T['phi']); real = T['pt'] > 0
    ptj = np.hypot((px * real).sum(1), (py * real).sum(1))
    rr = np.sqrt(rmin ** 2 + rng.random(J) * (rmax ** 2 - rmin ** 2)); th = rng.random(J) * 2 * np.pi
    eta_s = T['ax_eta'] + T['sgn'] * rr * np.cos(th); phi_s = T['ax_phi'] + rr * np.sin(th)
    qx, qy, qz = H.p3(eps * ptj, eta_s, phi_s)
    U = H.copyT(T, 1)
    H.setp(U, rows, 0, *H.from3(px[:, 0] - qx, py[:, 0] - qy, pz[:, 0] - qz), T['m'][:, 0])
    c = U['pt'].shape[1] - 1
    H.setp(U, rows, c, eps * ptj, eta_s, phi_s, np.zeros(J))
    if U['ext']:
        for key in U['ext']: U['ext'][key][:, c] = 0
        U['ext']['isNeutralHadron'][:, c] = 1; U['ext']['mask'][:, c] = 1
    return U


def conds():
    C = [dict(name='identity', kind='identity')]
    C += [dict(name='split_r0_z0.05_dR0', kind='h1', control='reshuffle_r0_f0.05'), dict(name='reshuffle_r0_f0.05', kind='h1'),
          dict(name='split_r0_z0.5_dR0', kind='h1', control='reshuffle_r0_f0.5'), dict(name='reshuffle_r0_f0.5', kind='h1'),
          dict(name='multi_k8', kind='h1'), dict(name='drop_softest', kind='h1')]
    for e in (1e-6, 1e-5, 1e-4, 1e-3):
        C += [dict(name=f'soft_eps{e:g}', kind='ring', eps=e, rmin=0.0, rmax=0.8, control=f'recoil_eps{e:g}'),
              dict(name=f'recoil_eps{e:g}', kind='recoil', eps=e)]
    for e in (1e-6, 1e-4):
        C += [dict(name=f'softcore_eps{e:g}', kind='ring', eps=e, rmin=0.0, rmax=0.05, control=f'softedge_eps{e:g}'),
              dict(name=f'softedge_eps{e:g}', kind='ring', eps=e, rmin=0.4, rmax=0.8),
              dict(name=f'drop_then_soft_eps{e:g}', kind='dropsoft', eps=e, control='drop_softest')]
    return C


def apply_v(T0, c, cap, seed):
    if c['kind'] == 'h1':
        spec = next(d for d in H.condition_list() if d['name'] == c['name'])
        U, applied, info = H.apply(T0, spec, cap, 0.8, seed)
        return U, applied, info['overflow'], info
    rng = np.random.default_rng(seed + zlib.crc32(c['name'].encode()) % 100000); J = len(T0['pt']); applied = np.ones(J, bool)
    if c['kind'] == 'identity':
        U = T0
    elif c['kind'] == 'ring':
        U = soft_ring(T0, c['eps'], c['rmin'], c['rmax'], rng)
    elif c['kind'] == 'recoil':
        U = H.soft(T0, c['eps'], 0.8, rng, add=False)
    elif c['kind'] == 'dropsoft':
        U = soft_ring(H.drop_softest(T0), c['eps'], 0.0, 0.8, rng)
    U, over = H.finalize(U, cap)
    p4a, p4b = H.jet4(T0), H.jet4(U)
    info = dict(dp3_rel=np.abs(p4b[:, :3] - p4a[:, :3]).max(1) / np.maximum(np.hypot(p4a[:, 0], p4a[:, 1]), 1e-9),
                dN=(U['pt'] > 0).sum(1) - (T0['pt'] > 0).sum(1))
    return U, applied, over, info


def sortedness(T):
    pt = T['pt']; real = pt > 0
    slot0_hardest = float((pt.argmax(1) == 0).mean())
    d = np.diff(pt, axis=1); nonincr = float(((d <= 0) | ~real[:, 1:]).all(1).mean())
    return dict(slot0_is_hardest=slot0_hardest, nonincreasing=nonincr)


def run(name, T0, y, models, prep, classes, cap, seed, extra_null=None):
    rng = np.random.default_rng(seed + 7)
    res = dict(part=name, N=int(len(y)), seed=seed, classes=classes, labels=np.bincount(y, minlength=len(classes)).tolist(),
               sortedness=sortedness(T0), frac_at_cap=float(((T0['pt'] > 0).sum(1) >= cap).mean()), models={}, conditions={})
    log(name, 'sortedness', res['sortedness'], 'frac_at_cap', res['frac_at_cap'])
    X0 = prep(T0); L0 = {}; CH = {}
    for mn, f in models.items():
        L0[mn] = f(X0); Lb = f(X0); c0 = L0[mn].argmax(1)
        res['models'][mn] = dict(acc=float((c0 == y).mean()), acc_ci=H.boot_binary(c0 == y, rng),
                                 null_rerun=float((Lb.argmax(1) != c0).mean()), null_rerun_maxlogit=float(np.abs(Lb - L0[mn]).max()), results={})
        if extra_null is not None and mn in extra_null:
            Lc = extra_null[mn](X0); res['models'][mn]['null_rebatch'] = float((Lc.argmax(1) != c0).mean())
            res['models'][mn]['null_rebatch_maxlogit'] = float(np.abs(Lc - L0[mn]).max())
        log(f'{name} {mn}: acc {res["models"][mn]["acc"]:.4f} rerun {res["models"][mn]["null_rerun"]:.4f}'
            + (f' rebatch {res["models"][mn].get("null_rebatch")}' if extra_null else ''))
    for c in conds():
        t = time.time(); U, applied, over, info = apply_v(T0, c, cap, seed)
        res['conditions'][c['name']] = dict(n_applied=int(applied.sum()), overflow=float(over[applied].mean()),
                                           dN_mean=float(info['dN'][applied].mean()), dp3_rel_max=float(np.nanmax(info['dp3_rel'][applied])))
        XU = prep(U)
        for mn, f in models.items():
            L1 = f(XU); c0, c1 = L0[mn].argmax(1), L1.argmax(1); ch = c0 != c1
            st = dict(all=dict(n=int(applied.sum()), changed=float(ch[applied].mean()), ci=H.boot_binary(ch[applied], rng)),
                      no_overflow=dict(n=int((applied & ~over).sum()), changed=float(ch[applied & ~over].mean()) if (applied & ~over).any() else None,
                                       ci=H.boot_binary(ch[applied & ~over], rng)))
            F = np.zeros((len(classes),) * 2, int); np.add.at(F, (c0[applied], c1[applied]), 1)
            st['net_flow'] = {classes[k]: int(F[:, k].sum() - F[k, :].sum()) for k in range(len(classes))}
            st['calls'] = c1
            res['models'][mn]['results'][c['name']] = st; CH[(mn, c['name'])] = (ch, applied, c1)
        log(f'{name} {c["name"]}: dN {res["conditions"][c["name"]]["dN_mean"]:.2f} over {res["conditions"][c["name"]]["overflow"]:.3f} | '
            + ' '.join(f'{mn} {100 * res["models"][mn]["results"][c["name"]]["all"]["changed"]:.2f}' for mn in models) + f' ({time.time() - t:.0f} s)')
    tests = []
    for mn in models:
        c0 = L0[mn].argmax(1)
        for c in conds():
            if 'control' not in c: continue
            a, ap, _ = CH[(mn, c['name'])]; b, bp, _ = CH[(mn, c['control'])]
            P = H.paired(a, b, ap & bp, rng); res['models'][mn]['results'][c['name']]['vs_control'] = P
            tests.append((mn, c['name'], P.get('p_mcnemar', 1.0)))
        # count restore: of the jets whose call the drop changed, the fraction whose call returns to unedited after drop + soft
        _, _, cd = CH[(mn, 'drop_softest')]; dch = cd != c0
        for e in ('1e-06', '0.0001'):
            _, _, cds = CH[(mn, f'drop_then_soft_eps{e}')]
            back = (cds == c0)[dch]
            res['models'][mn]['results'][f'drop_then_soft_eps{e}']['restored_of_drop_changed'] = dict(
                n=int(dch.sum()), frac=float(back.mean()) if dch.any() else None, ci=H.boot_binary(back, rng))
        # same, for the soft add alone: of the jets whose call the drop changed, fraction where the eps-1e-6 soft add (no drop) changes
    hp = H.holm([t[2] for t in tests])
    for (mn, cn, _), h in zip(tests, hp): res['models'][mn]['results'][cn]['vs_control']['holm_p'] = h
    res['n_tests'] = len(tests)
    for mn in models:
        for cn in res['models'][mn]['results']: res['models'][mn]['results'][cn].pop('calls', None)
    out = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'physics'; out.mkdir(parents=True, exist_ok=True)
    p = out / f'h1_verify_{name}.json'; p.write_text(json.dumps(res, indent=1, default=float)); log('wrote', p)


def part_jedi():
    from interp.jedi_core import Jedi
    T0, y = H.hls_sample(2000, seed=411)
    net = Jedi(64, '3-feature')
    run('jedi', T0, y, dict(jedi=lambda x: net.run(x)['logits']), H.hls_x, H.HLS_CL, 64, 411)


def part_small():
    import torch
    torch.set_num_threads(1)
    T0, y = H.hls_sample(2000, seed=411); models = {}
    for tag, r in (('dsets', 'dsets-d64-n64'), ('linf', 'linf-2b-d32-k8-n64'), ('tf_d64', 'tf-2b-d64-n64-e100')):
        net, sh, sc = H.load_small(H.SMALL / r / 'seed0'); models[tag] = (lambda x, net=net, sh=sh, sc=sc: H.torch_logits(net, x, sh, sc))
    efn = H.efn_class()(); efn.load_state_dict(torch.load(H.OUT / 'h1_efn' / 'best.pt', map_location='cpu')); efn.eval()
    models['efn'] = (lambda x: H.torch_logits(efn, x, np.zeros(3, np.float32), np.ones(3, np.float32)))
    run('small', T0, y, models, H.hls_x, H.HLS_CL, 64, 411)


def part_jetclass():
    from interp.part_core import ParTNetwork, CL, inputs
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu')
    J0 = H.jc_sample(1000, seed=412); y = J0['label'].astype(int); T0 = H.jc_table(J0)
    models, rebatch = {}, {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n)
        models[f'ParT_{n}'] = (lambda J, net=net, n=n: fwd(net, *inputs(J, n))[1])
        rebatch[f'ParT_{n}'] = (lambda J, net=net, n=n: fwd(net, *inputs(J, n), batch=97)[1])
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        models[f'PN_{n}'] = (lambda J, pn=pn, n=n: H.pn_logits(pn, J, n))
        rebatch[f'PN_{n}'] = (lambda J, pn=pn, n=n: H.pn_logits(pn, J, n, batch=97))
    run('jetclass', T0, y, models, lambda T: H.jc_jets(J0, T), list(CL), 128, 412, extra_null=rebatch)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['jedi', 'small', 'jetclass']); a = ap.parse_args()
    log('H1 verify', a.part); dict(jedi=part_jedi, small=part_small, jetclass=part_jetclass)[a.part](); log('done')
