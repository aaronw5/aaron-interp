"""P37 verification, part B with more random sets: is the joint knock-out of P37's top-ranked head set specific to the top classes?
Same sample and means as interp/p37_verify.py (test_20M <sample>_118.root, default_rng(3790); EVAL 300 jets per class, REF 200 per
class). Per class k: change of the own-class log-odds lo_k = log p_k/(1-p_k) and of the recall when P37's top-5 (ParT_full, ParT_kin)
or top-10 (ParT_kin) heads are mean-ablated together, against NB_B random head sets with the same number of class-attention heads
(drawn from the 16) and particle heads (from the 64). p per class = (1 + #random sets with a change at least as negative) / (1 + NB_B),
Holm over the 10 classes; z = (circuit - random mean) / random s.d.
  ./sync.sh && ./part_job.sh c-p37-verify2 'python -m interp.p37_verify_b'   -> $AI_RESULTS/part/p37_verify_B_{full,kin}.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_argv = sys.argv; sys.argv = [_argv[0]] + (['--smoke'] if '--smoke' in _argv else [])
import interp.p37_verify as V                    # noqa: E402
sys.argv = _argv
P = V.P; NB_B = 10 if V.SMOKE else 500


def main():
    import torch
    rng, B, BR, BT, y, sub_idx, P1 = V.load(); del BT, P1
    cls_idx = [np.where(y == c)[0] for c in range(10)]
    for key in ('full', 'kin'):
        name, feat, _ = P.MODEL_SPEC[key]; rr = np.random.default_rng(V.SEED + 20 + (key == 'kin'))
        rank = [V.idx_of(h) for h in json.loads((V.OUTD / f'p37_top_circuit_{key}.json').read_text())['model']['scan']['rank']]
        M = P.HRun(feat); L = M.run(B); corr0 = L.argmax(1) == y; lo0 = np.stack([P.lodds(L, c) for c in range(10)], 1)
        M.run(BR, dict(recmean=True)); Mh, _ = M.means(); res = dict(name=name, acc_eval=float(corr0.mean()))

        def perclass(S):
            Lk = M.run(B, P.ctl_ablate(M, S, Mh)); ck = Lk.argmax(1) == y
            return (np.array([(P.lodds(Lk[ix], c) - lo0[ix, c]).mean() for c, ix in enumerate(cls_idx)]),
                    np.array([(ck[ix].astype(float) - corr0[ix]).mean() for ix in cls_idx]))
        for k in ((5,) if key == 'full' else (5, 10)):
            S = rank[:k]; dlo, drc = perclass(S); rl, rc = [], []
            for _ in range(NB_B):
                a_, b_ = perclass(V.matched(S, rr, 'class')); rl.append(a_); rc.append(b_)
            rl, rc = np.array(rl), np.array(rc)
            p_lo = np.array([(1 + (rl[:, c] <= dlo[c]).sum()) / (1 + NB_B) for c in range(10)])
            p_rc = np.array([(1 + (rc[:, c] <= drc[c]).sum()) / (1 + NB_B) for c in range(10)])
            hl, hr = P.holm(p_lo), P.holm(p_rc)
            res[str(k)] = dict(heads=[P.hlab(q) for q in S], n_random=NB_B, classes=P.CLN,
                               dlo=dlo, dlo_random_mean=rl.mean(0), dlo_random_sd=rl.std(0), dlo_random_min=rl.min(0), dlo_z=(dlo - rl.mean(0)) / rl.std(0),
                               p_dlo=p_lo, p_dlo_holm=hl, drecall=drc, drecall_random_mean=rc.mean(0), drecall_random_sd=rc.std(0),
                               p_drecall=p_rc, p_drecall_holm=hr)
            V.log(f'{name} k={k}: ' + '; '.join(f'{P.CLN[c]} dlo {dlo[c]:+.2f} (rand {rl[:, c].mean():+.2f}+-{rl[:, c].std():.2f}, z {res[str(k)]["dlo_z"][c]:+.1f}, '
                                              f'Holm {hl[c]:.3f}) drec {drc[c]:+.3f} (rand {rc[:, c].mean():+.3f}, Holm {hr[c]:.3f})' for c in range(10)))
        fn = V.OUTD / f'p37_verify_B_{key}{V.TAG}.json'
        fn.write_text(json.dumps(P.strip(dict(node='P37v-B', seed=V.SEED, model=res)), indent=1)); V.log('wrote', fn)
        del M; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
