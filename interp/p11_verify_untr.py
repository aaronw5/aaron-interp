"""P11v (second verifier pass) — untrained-network control over initialisation seeds.

P11v found that one untrained ParT_kin (weaver init, torch seed 0) gives a class-block-1 removed share of 0.868 in S3,
above the retrained seeds (0.63-0.67), and a +4.0-point W->Z change under phi + 0.06 with R -0.23. This control asks whether
those untrained values are stable across initialisations: the same unchanged P11 analysis (interp.p11_seed_tests.analyse)
on untrained ParT_full / ParT_kin with torch seeds 1, 2, 3, on the same fresh sample as P11v (WToQQ_101 / ZToQQ_101,
10,000 random rows each, numpy default_rng(21); S3 subset 2,000 W + 2,000 Z, default_rng(12)).

  ./part_job.sh c-p11-verify2 'python -m interp.p11_verify_untr --feat full && python -m interp.p11_verify_untr --feat kin'
  -> $AI_RESULTS/part/p11_verify_untrained_<feat>.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import p11_seed_tests as P


def make_untrained(feat, s):
    net = P.make_net(feat, 'released', None)
    torch = net.torch
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import CFG, FEATURES
    torch.manual_seed(s); fresh = ParticleTransformer(input_dim=len(FEATURES[feat]), **CFG)
    net.model.load_state_dict(fresh.state_dict(), strict=True); net.model.to(net.device).eval()
    for p in net.model.parameters(): p.requires_grad_(False)
    W = net.model.fc[0].weight.detach().cpu().double().numpy(); b = net.model.fc[0].bias.detach().cpu().double().numpy()
    net.last = (W.T.copy(), b.copy(), None, None); net.K = W.shape[0]; net.path = f'init{s}'
    return net


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feat', default='full'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--inits', default='1,2,3')
    a = ap.parse_args()
    a.seed = 11; a.nwz = 300 if a.smoke else 10000; a.n3 = 60 if a.smoke else 2000; a.nb = 100 if a.smoke else 1000
    a.nb4 = 20 if a.smoke else 200; a.nperm = 20 if a.smoke else 4000; a.nrand_c = 4 if a.smoke else 30; a.nrand_p = 3 if a.smoke else 20
    a.s3batch = 128; a.plot = False
    import torch
    F = P.class_files(); F101 = {}
    for c in ('Wqq', 'Zqq'):
        p = Path(str(F[c]).replace('_100.root', '_101.root')); assert p.exists() and p != Path(str(F[c])), p; F101[c] = p
    orig_wz = P.wz_sample
    P.wz_sample = lambda per, seed, F_: orig_wz(per, 21, F101)
    D, info = P.prepare(a.feat, a, F)
    P.wz_sample = orig_wz
    info['wz_files'] = {c: str(p) for c, p in F101.items()}; info['wz_rng'] = 21
    out = P.OUTD / f'p11_verify_untrained_{a.feat}{"_smoke" if a.smoke else ""}.json'
    res = dict(feat=a.feat, sample=info, models={})
    for s in [int(x) for x in a.inits.split(',')]:
        rng = np.random.default_rng(a.seed + 200)
        net = make_untrained(a.feat, s); P.log(f'== {a.feat} untrained init seed {s}')
        res['models'][f'init{s}'] = P.analyse(f'init{s}', net, a.feat, D, a, rng); del net
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        json.dump(P.strip(res), open(out, 'w'), indent=1)
    P.log(f'wrote {out}'); P.log('P11v-untr DONE')


if __name__ == '__main__':
    main()
