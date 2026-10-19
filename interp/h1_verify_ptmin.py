"""H1 verify: the smallest real-particle pT in the H1-verify samples, against the soft-particle pT eps * pT_jet (eps 1e-6 .. 1e-3).
AI_PY=conda ./job.sh d-h1-verify-p 1 3 'python -m interp.h1_verify_ptmin' -> $AI_RESULTS/physics/h1_verify_ptmin.json"""
import json, os
from pathlib import Path
import numpy as np
import interp.h1_irc_panel as H
out = {}
import sys
for name in sys.argv[1:] or ('hls4ml', 'jetclass'):
    T = H.hls_sample(2000, seed=411)[0] if name == 'hls4ml' else H.jc_table(H.jc_sample(1000, seed=412))
    pt = T['pt']; real = pt > 0
    ptmin = np.where(real, pt, np.inf).min(1); allpt = pt[real]
    px, py, _ = H.p3(pt, T['eta'], T['phi']); ptj = np.hypot((px * real).sum(1), (py * real).sum(1))
    r = np.hypot(T['eta'] - T['ax_eta'][:, None], H.wrap(T['phi'] - T['ax_phi'][:, None]))[real]
    d = dict(ptj_median=float(np.median(ptj)), ptmin_jet_quantiles={q: float(np.quantile(ptmin, q)) for q in (0.0, 0.01, 0.5)},
             allparticle_pt_quantiles={q: float(np.quantile(allpt, q)) for q in (0.0, 0.001, 0.01, 0.5)},
             frac_particles_r_gt_0p4=float((r > 0.4).mean()), frac_particles_r_lt_0p05=float((r < 0.05).mean()))
    for e in (1e-6, 1e-5, 1e-4, 1e-3):
        s = e * ptj; d[f'eps{e:g}'] = dict(soft_pt_median=float(np.median(s)), frac_below_global_min_pt=float((s < allpt.min()).mean()),
                                           frac_below_jet_min_pt=float((s < ptmin).mean()))
    out[name] = d; print(name, json.dumps(d), flush=True)
p = Path(os.environ['AI_RESULTS']) / 'physics' / f'h1_verify_ptmin_{"_".join(out)}.json'; p.write_text(json.dumps(out, indent=1)); print('wrote', p)
