# Loop C, round C1 (planner, 2026-10-17)

| node | title | script | exploratory |
|---|---|---|---|
| P3 | ParT_full W/Z mass circuit at full statistics with the corrected build: b2h8 threshold, far-side heads, neuron n98, and the literal circuit kept alone (accuracy + 10-class) | interp/p3_wz_circuit_part.py | no |
| P4 | ParticleNet_full/kin: accuracy on the P1 sample, translation / WTA-axis tests, input-route and EdgeConv kNN-graph decomposition | interp/p4_particlenet.py | yes |
| P5 | L-GATr: check for released weights, then launch training (authors' jctagging config, JetClass 2M subset) with smoke check; pre-registered symmetry predictions | interp/p5_lgatr_launch.py | yes |
