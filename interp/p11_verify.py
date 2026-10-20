"""P11v — verifier rerun of P11 (ParT seed replication, pre-registered S1-S4) on fresh rows, plus an epoch-trajectory control.

Fresh sample: the W/Z sample is drawn from the second test file of each class (WToQQ_101 / ZToQQ_101, never used by P11,
whose W/Z jets came from *_100), 10,000 random rows per class with numpy default_rng(21) (not first-k; P11 used rng 11 on
*_100). The S3 subset is 2,000 W + 2,000 Z random jets of this new sample (default_rng(12) on the new sample, so different
jets). The ridge read-out (S1) is fitted on the P1 sample as in P11 (rows 3,000-4,000 of *_100), which is disjoint from the
new W/Z jets. Every statistic is computed by the unchanged P11 code (interp.p11_seed_tests.analyse): W->Z calls under
phi + 0.06 with McNemar, R (phi / matched dilation), S1 read-out ratio, S2 route shares, S3 head-set patching shares
(class block 1 / 2, single heads, random sets, particle heads), S4 switch scan.
Models: released ParT_<feat>, retrained seeds s0-s2 (P10 best epoch), untrained (weaver init, torch seed 0).

Epoch-trajectory control (not in P11): the same statistics (S3 reduced to class heads + two particle heads, 10 random class
sets, S4 with 500 permutations) on seed 0's checkpoints at epochs 0, 3, 9, 19, 29, 49 and best. Question: does the
class-block-1 share / the switch amplitude grow with training toward the released values (which would point to training
amount rather than the seed as the cause of the released-vs-seed gap)?

  ./part_job.sh c-p11-verify 'python -m interp.p11_verify --feat full && python -m interp.p11_verify --feat kin'
  -> $AI_RESULTS/part/p11_verify_<feat>.json
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import p11_seed_tests as P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feat', default='full'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-epochs', action='store_true')
    a = ap.parse_args()
    a.seed = 11; a.nwz = 300 if a.smoke else 10000; a.n3 = 60 if a.smoke else 2000; a.nb = 100 if a.smoke else 1000
    a.nb4 = 20 if a.smoke else 200; a.nperm = 20 if a.smoke else 4000; a.nrand_c = 4 if a.smoke else 30; a.nrand_p = 3 if a.smoke else 20
    a.s3batch = 128; a.plot = False
    import torch
    P.log(f'P11v feat {a.feat}; torch {torch.__version__}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda"}')
    F = P.class_files()
    F101 = {}
    for c in ('Wqq', 'Zqq'):
        p = Path(str(F[c]).replace('_100.root', '_101.root')); assert p.exists() and p != Path(str(F[c])), p; F101[c] = p
    orig_wz = P.wz_sample
    P.wz_sample = lambda per, seed, F_: orig_wz(per, 21, F101)        # fresh rows: *_101 files, rng 21
    D, info = P.prepare(a.feat, a, F)
    P.wz_sample = orig_wz
    info['wz_files'] = {c: str(p) for c, p in F101.items()}; info['wz_rng'] = 21
    out = P.OUTD / f'p11_verify_{a.feat}{"_smoke" if a.smoke else ""}.json'
    S = P.SEEDS_DIR
    models = [('released', None)] + [(f's{s}', f'{S}/{a.feat}_s{s}/net_best_epoch_state.pt') for s in (0, 1, 2)] + [('untrained', 'init0')]
    res = dict(feat=a.feat, sample=info, models={}, epochs={})
    for name, path in models:
        rng = np.random.default_rng(a.seed + 200)
        net = P.make_net(a.feat, name, path); P.log(f'== {a.feat} {name}: {path}')
        res['models'][name] = P.analyse(name, net, a.feat, D, a, rng); del net
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        json.dump(P.strip(res), open(out, 'w'), indent=1)
    if not a.no_epochs:
        b = copy.copy(a); b.smoke = True; b.nb = 300; b.nb4 = 50; b.nperm = 500; b.nrand_c = 10; b.nrand_p = 3
        eps = [0, 3] if a.smoke else [0, 3, 9, 19, 29, 49]
        for e in eps + ['best']:
            path = f'{S}/{a.feat}_s0/net_{"best_epoch" if e == "best" else f"epoch-{e}"}_state.pt'
            if not os.path.exists(path): P.log(f'missing {path}'); continue
            rng = np.random.default_rng(a.seed + 300)
            net = P.make_net(a.feat, f's0_e{e}', path); P.log(f'== {a.feat} s0 epoch {e}: {path}')
            res['epochs'][str(e)] = P.analyse(f's0_e{e}', net, a.feat, D, b, rng); del net
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            json.dump(P.strip(res), open(out, 'w'), indent=1)
    json.dump(P.strip(res), open(out, 'w'), indent=1); P.log(f'wrote {out}'); P.log('P11v DONE')


if __name__ == '__main__':
    main()
