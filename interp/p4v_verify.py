"""P4v — verifier rerun of P4 (ParticleNet axis-relative mass response) on fresh rows, plus axis-preserving controls.

Sample: rows 8,000–9,000 of each class's first test file (10,000 jets; P4 used 3,000–4,000 and 6,000–7,000), rebuilt with
part_core.build (sign-corrected η). Reuses P4's code (prep, forward split, statistics, route variants) unchanged.
Re-run: accuracy of ParticleNet_full/kin and ParT_full/kin; rigid φ + 0.06 (W→Z, calls changed, Δ(lW − lZ) on W, McNemar);
  calibrated read-out ratio (P4's ratio_tests); phi0.06 route shares (graphs fixed, deltaR only/removed, (Δη, Δφ) only, layer-1
  centre / edge / shortcut terms).
New controls (edits that keep each particle's ΔR to the axis, hence ½Δlog(M0·M2) = 0, while moving (Δη, Δφ)):
  refl  Δφ → −Δφ (reflection in φ about the axis; invariant mass exactly unchanged);
  rot0.3, rot90  rotation of (Δη, Δφ) about the axis by 0.3 rad / 90° (invariant mass unchanged to small-angle order).
  H1 (response is carried by the axis distance ΔR / the axis moment): these leave W→Z at its unedited value.
  H2 (any change of particle positions of this size moves W→Z): W→Z rises as under the 0.06 shift.
  ./part_job.sh c-p4-verify 'python -m interp.p4v_verify'   -> $AI_RESULTS/part/p4_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import p4_particlenet as P

log = P.log
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
KEEP = ('knnall_fixed', 'dR_only', 'dR_removed', 'dedp_only', 'ctr_only', 'edge_only', 'sc_only', 'sc_removed')
CTRL = ('refl', 'rot0.3', 'rot90')


def main():
    P._imports(); torch, pc, p1v, p2v, pn_net, FEATURES = (P.torch, P.pc, P.p1v, P.p2v, P.pn_net, P.FEATURES)
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 1000; START = 8000
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    S = P.prep(PER, START); y = S['y']; N = len(y); J0, pt, de, dp, ms, real = S['J0'], S['pt'], S['de'], S['dp'], S['ms'], S['real']
    rot = lambda a: pc.build(J0, pt, (de * np.cos(a) - dp * np.sin(a)) * real, (de * np.sin(a) + dp * np.cos(a)) * real, ms)
    CE = {'refl': pc.build(J0, pt, de, -dp, ms), 'rot0.3': rot(0.3), 'rot90': rot(np.pi / 2)}
    chk = {}
    for k, Je in list(CE.items()) + [('phi0.06', S['JE']['phi0.06'])]:
        pte, dee, dpe, _ = pc.kin(Je); w = pt * real
        # displacement of each real particle (pT-weighted mean over particles, mean over jets): computed before the pT re-sort
        if k == 'refl': d = 2 * np.abs(dp)
        elif k.startswith('rot'): a = 0.3 if k == 'rot0.3' else np.pi / 2; d = 2 * np.hypot(de, dp) * abs(np.sin(a / 2))
        else: d = 0.06 * real
        chk[k] = dict(dlogm_true_mean=float((np.log(np.maximum(pc.jmass(Je), 1.0)) - S['logm0']).mean()),
                      dlogm_true_absmax=float(np.abs(np.log(np.maximum(pc.jmass(Je), 1.0)) - S['logm0']).max()),
                      pred_half_dlogM0M2_mean=float((0.5 * np.log(pc.moments(pte, dee, dpe)['M0M2'] / S['M']['M0M2'])).mean()),
                      mean_ptweighted_displacement=float(((w * d).sum(1) / w.sum(1)).mean()),
                      mean_ptweighted_displacement_W=float(((w * d).sum(1) / w.sum(1))[y == P.iW].mean()),
                      mask_equal=bool(np.array_equal(Je['mask'], S['Jb']['mask'])))
    log(f'rows {START}-{START + PER}: {N} jets; calibration {S["cal"]}; control edit checks {chk}')
    res = dict(rows=[START, START + PER], N=N, boot=dict(n=P.NB, seed=2), edit_checks=S['edit_checks'], control_checks=chk, calibration=S['cal'], models={})
    tests = []
    for n in ('full', 'kin'):
        net = pc.ParTNetwork(n); L0 = net.run(S['Jb'])['logits']; cn = net.run(S['Jb'], batch=128)['logits'].argmax(1)
        row = dict(acc=float((L0.argmax(1) == y).mean()), noise_changed=float((cn != L0.argmax(1)).mean()), edits={})
        for e, Je in [('phi0.06', S['JE']['phi0.06'])] + list(CE.items()):
            row['edits'][e] = P.call_stats(S, L0, net.run(Je)['logits'], cn)[0]
            tests.append(((f'ParT_{n}', e), row['edits'][e]['mcnemar_W_to_Z']['p']))
        res['models'][f'ParT_{n}'] = row
        log(f'ParT_{n}: acc {100 * row["acc"]:.2f} %, noise {row["noise_changed"]:.4f} | ' + '; '.join(
            f'{e}: changed {100 * v["changed"]["est"]:.2f} %, W→Z {v["W_to_Z_base"]["est"]:.3f} → {v["W_to_Z"]["est"]:.3f} {np.round(v["W_to_Z"]["ci"], 3).tolist()} '
            f'McNemar {v["mcnemar_W_to_Z"]}' for e, v in row['edits'].items()))
        del net
    for n in ('full', 'kin'):
        model = pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', len(FEATURES[n]), dev); key = f'ParticleNet_{n}'
        trf, trv, tr, te = S['split']; Xb = P.pn_inputs(S['Jb'], n)
        O0 = P.fwd(model, Xb, capture=True); L0, h0, K0 = O0['L'], O0['h'], O0['knn']
        Lnat = P.fwd(model, Xb, native=True)['L']; cn = P.fwd(model, Xb, batch=97)['L'].argmax(1)
        ridge = p1v.Ridge(h0, S['logm0'], trf, trv, tr)
        row = dict(acc=float((L0.argmax(1) == y).mean()), acc_ci=P.pq(S, L0.argmax(1) == y, np.ones(N, bool))[0]['ci'],
                   split_vs_native_max_dlogit=float(np.abs(Lnat - L0).max()), noise_batch97_changed=float((cn != L0.argmax(1)).mean()),
                   readout_r2=p1v.r2(ridge(h0[te]), S['logm0'][te]), edits={}, routes={})
        H = {}
        for e, Je in [('phi0.06', S['JE']['phi0.06']), ('cal_phi0.06', S['JE']['cal_phi0.06'])] + list(CE.items()):
            Xe = P.pn_inputs(Je, n); assert np.array_equal(Xe['m'], Xb['m'])
            O = P.fwd(model, Xe, kref=K0); H[e] = O['h']; st = P.call_stats(S, L0, O['L'], cn)[0]
            dv, sel = P.readout_vec(S, ridge, h0, O['h']); st['dlogm_hat'] = P.pq(S, dv, sel)[0]; st['knn_overlap'] = O['ov'].mean(0).tolist()
            if e != 'cal_phi0.06': tests.append(((key, e), st['mcnemar_W_to_Z']['p']))
            row['edits'][e] = st
            log(f'{key} {e:11s}: acc {100 * st["acc"]["est"]:.2f}, changed {100 * st["changed"]["est"]:.2f} % {np.round(np.array(st["changed"]["ci"]) * 100, 2).tolist()}, '
                f'W→Z {st["W_to_Z_base"]["est"]:.3f} → {st["W_to_Z"]["est"]:.3f} {np.round(st["W_to_Z"]["ci"], 3).tolist()} McNemar {st["mcnemar_W_to_Z"]}, '
                f'Δ(lW−lZ)|W {st["dlogit_WZ_on_W"]["est"]:+.3f} {np.round(st["dlogit_WZ_on_W"]["ci"], 3).tolist()}, Δlog m̂ {st["dlogm_hat"]["est"]:+.4f} '
                f'{np.round(st["dlogm_hat"]["ci"], 4).tolist()}, kNN overlap {np.round(st["knn_overlap"], 4).tolist()}')
        rt = P.ratio_tests(S, ridge, h0, H, 'phi0.06'); row['ratio_phi0.06'] = rt
        log(f'{key} ratio phi0.06: obs {rt["obs"]["est"]:+.4f}, pred {rt["pred"]["est"]:+.4f}, gain c {rt["c"]:.3f}, raw {rt["raw_ratio"]:.3f}, '
            f'calibrated {rt["calibrated_ratio"]:.3f} {np.round(rt["calibrated_ratio_ci"], 3).tolist()}; p_inv {rt["p_inv"]:.4f}/{rt["p_inv_norm"]:.1e}, p_M2 {rt["p_M2"]:.4f}/{rt["p_M2_norm"]:.1e}')
        Xe = P.pn_inputs(S['JE']['phi0.06'], n); V = P.variants(Xb, Xe, n); R = {}; Ba = None
        for vn in ('a_full',) + KEEP:
            O = P.run_variant(model, Xe, V[vn], K0); st, Bst = P.call_stats(S, L0, O['L'], cn)
            dv, sel = P.readout_vec(S, ridge, h0, O['h']); m_est, m_bs = P.pq(S, dv, sel)
            Bq = {'dlogit_WZ_on_W': (st['dlogit_WZ_on_W']['est'], Bst['dlogit_WZ_on_W']), 'dW_to_Z': (st['dW_to_Z']['est'], Bst['dW_to_Z']),
                  'dlogm_hat': (m_est['est'], m_bs)}
            if vn == 'a_full': Ba = Bq
            R[vn] = dict(W_to_Z=st['W_to_Z'], dlogit_WZ_on_W=st['dlogit_WZ_on_W'], share=P.share_stats(Ba, Bq), knn_overlap=O['ov'].mean(0).tolist())
        row['routes']['phi0.06'] = R
        log(f'{key} routes phi0.06: ' + '; '.join(f'{vn} {R[vn]["share"]["dlogit_WZ_on_W"]["share"]:.3f} '
                                                  f'{np.round(R[vn]["share"]["dlogit_WZ_on_W"]["share_ci"], 3).tolist()} (m̂ {R[vn]["share"]["dlogm_hat"]["share"]:.3f})' for vn in KEEP))
        res['models'][key] = row; del model
        json.dump(res, open(OUT / 'p4_verify.json', 'w'), indent=1)
    adj = p2v.holm([p for _, p in tests]); res['holm_mcnemar_W_to_Z'] = [dict(test='/'.join(k), p=p, p_holm=a) for (k, p), a in zip(tests, adj)]
    log('Holm (McNemar W→Z, edit vs unedited): ' + '; '.join(f'{d["test"]} {d["p"]:.2e} → {d["p_holm"]:.2e}' for d in res['holm_mcnemar_W_to_Z']))
    json.dump(res, open(OUT / 'p4_verify.json', 'w'), indent=1); log(f'wrote {OUT / "p4_verify.json"}; P4v DONE ({time.time() - t0:.0f} s)')


def charge_check():
    """follow-up: does the full models' response to the Δφ reflection come from the charge / impact-parameter sign pattern?
    edits refl (Δφ → −Δφ), refl_q (also charge → −charge), refl_qd0 (also d0 → −d0), q_only (charge → −charge alone);
    ParT_full and ParticleNet_full, rows 8,000–9,000.  -> $AI_RESULTS/part/p4_verify_charge.json"""
    P._imports(); torch, pc, pn_net, FEATURES = P.torch, P.pc, P.pn_net, P.FEATURES; t0 = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    S = P.prep(1000, 8000); y = S['y']; J0, pt, de, dp, ms = S['J0'], S['pt'], S['de'], S['dp'], S['ms']
    def ext(**flip):
        E = {k: J0[k] for k in pc.PK}; E.update({k: -J0[k] for k in flip}); return E
    CE = {'refl': pc.build(J0, pt, de, -dp, ms), 'refl_q': pc.build(J0, pt, de, -dp, ms, extra=ext(charge=1)),
          'refl_qd0': pc.build(J0, pt, de, -dp, ms, extra=ext(charge=1, d0val=1)), 'q_only': pc.build(J0, pt, de, dp, ms, extra=ext(charge=1)),
          'phi0.06': S['JE']['phi0.06']}
    res = dict(rows=[8000, 9000], N=len(y), models={})
    net = pc.ParTNetwork('full'); L0 = net.run(S['Jb'])['logits']; cn = net.run(S['Jb'], batch=128)['logits'].argmax(1)
    res['models']['ParT_full'] = dict(acc=float((L0.argmax(1) == y).mean()), edits={e: P.call_stats(S, L0, net.run(Je)['logits'], cn)[0] for e, Je in CE.items()})
    del net
    model = pn_net.load_particlenet(PNDIR / 'ParticleNet_full.pt', len(FEATURES['full']), dev); Xb = P.pn_inputs(S['Jb'], 'full')
    L0 = P.fwd(model, Xb)['L']; cn = P.fwd(model, Xb, batch=97)['L'].argmax(1)
    res['models']['ParticleNet_full'] = dict(acc=float((L0.argmax(1) == y).mean()),
                                             edits={e: P.call_stats(S, L0, P.fwd(model, P.pn_inputs(Je, 'full'))['L'], cn)[0] for e, Je in CE.items()})
    for m, r in res['models'].items():
        log(f'{m}: acc {100 * r["acc"]:.2f} % | ' + '; '.join(f'{e}: acc {100 * v["acc"]["est"]:.2f}, changed {100 * v["changed"]["est"]:.2f} % {np.round(np.array(v["changed"]["ci"]) * 100, 2).tolist()}, '
                                                           f'W→Z {v["W_to_Z_base"]["est"]:.3f} → {v["W_to_Z"]["est"]:.3f} {np.round(v["W_to_Z"]["ci"], 3).tolist()} McNemar {v["mcnemar_W_to_Z"]}, '
                                                           f'Δ(lW−lZ)|W {v["dlogit_WZ_on_W"]["est"]:+.3f} {np.round(v["dlogit_WZ_on_W"]["ci"], 3).tolist()}' for e, v in r['edits'].items()))
    json.dump(res, open(OUT / 'p4_verify_charge.json', 'w'), indent=1); log(f'wrote p4_verify_charge.json ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    if sys.argv[1:] == ['charge']: charge_check()
    else: main()
