# Experiment tree

Stages: 1 = initial implementation, 2 = baselines and controls, 3 = research agenda, 4 = ablations and causal tests.

| node | parent | stage | paper/claim | plan | script | status | metric | figure | note |
|---|---|---|---|---|---|---|---|---|---|
| E1 | – | 1 | A1/B | units per layer; mean ablation per unit | interp/e1_e3_jedi.py | done | dead/live units, % calls changed | – | 13 live l1 units |
| E2 | E1 | 1 | A1 | l1 cuts in GeV and rad | interp/e1_e3_jedi.py | done | cut coefficients | – | two cut kinds: pT ramps, angular half-planes |
| E3 | E1 | 1 | A1/B | count+sum identity on P | interp/e1_e3_jedi.py | done | max error | – | exact in float; fixed point differs |
| E3q | E3 | 2 | A1/B | rounding removed per quantizer | scratch e3q3.py | done | % calls changed | – | l1+qs+qs1 float: 3.83 % calls |
| E4 | E3 | 3 | A2 | context round g | interp/e4_e5_jedi.py | running (agent) | | | |
| E5 | E3 | 3 | A3 | second-round objects named + causal | interp/e4_e5_jedi.py | running (agent) | | | |
| E6 | E3 | 3 | A4 | mass from moments; where the product forms | interp/e6_mass_jedi.py | running (agent) | | | |
| E9 | E3 | 2/4 | A7 | 6 official models (n8/32/64/128, perminv n64, 16-feature n64) + untrained, cluster | interp/e1_e3_sweep.py | running (agent) | | | |
| E10 | E3 | 3 | A9 | variables over layers vs algebraic depth | interp/e10_layers_jedi.py | running (agent) | | | |
| E7 | E5,E6 | 3/4 | A5 | neurons on own objects; stage substitution | – | todo | | | |
| E8 | E7 | 4 | A6 | classes as combinations, imposed | – | todo | | | |
| E3w | E3q | 2 | A1 | which quantizer the all-float network lacks | interp/e3w_quantizers.py | done | acc per leave-one-in | – | l3/qs1, not d7 |
| E3m | E3w | 2 | A1/A3/B | float + zero-bit masks | interp/e3m_masks.py | done | 80.68 % vs 81.29 % | – | objects = cut ∧ rank window |
| E6v | E6 | 4 | A4 | skeptic rerun of the translation result (fresh seed, dose–response, direction, input-range and P-unit controls) | interp/e6_verify.py | running (agent) | | results/jedi/e6_verify_translation.png | |
| P0 | – | 1 | A8/A10 | ParT package + helpers in aaron-interp; smoke test ParT_full 86.2 %, ParT_kin 74.4 % (local, before the cluster-only rule); cluster env-part built | interp/part_core.py, interp/p0_smoke_part.py | done | acc | – | found: G192 build mirrored η for jet_eta < 0 (fixed here); G-series neuron labels are index + 1 |
| P1 | P0,E6 | 4 | A4/A8, C3 | translation test on ParT_full / ParT_kin (mirror of E6); cross-model mass row | interp/p1_translation_part.py, p1_plot.py | done | W→Z, Δlog m̂ vs prediction | results/part/p1_translation.png | ParT not invariant: 0.7–0.8 (full), 1.1–1.4 (kin) of M2 prediction; C3 v1 refuted |
| P1v | P1 | 4 | C3, A8, D4 | verify P1: edit hygiene; deltaR vs dphi; fresh rows; WTA axis | interp/p1v_verify_part.py | done | | results/part/p1v_verify.png | verified 0.9; full via deltaR only; WTA 11.7 % / 34.1 % of calls |
| E17 | P1v, A4 | 4 | D4, A4 | JEDI axis test: inputs about the WTA / hardest-particle / recomputed E-scheme axis | interp/e17_axis_jedi.py | done | calls changed, acc | results/jedi/e17_axis.png | WTA: 30.3 % changed, 80.6 → 59.3 %; ratio 0.77–0.82 of M2 prediction |
| E18 | E17, A4 | 3 | A4, D4 | train Deep Sets (and a JEDI-like model) on inputs about the hardest-particle axis (M1 ≠ 0 in training): does the network learn the recentred mass? shift test about its own axis | train/ + interp/ | queued (training agent) | | | |
| E11 | E5,E6 | 3/4 | A5, B (user idea 2026-10-16) | MARS-fit the ENTIRE trigger model: every live unit of P, g, S, h4, h5, z as hinge formulas on (a) physics observables, (b) the previous stage's formulas; stage substitution + full-chain accuracy; per-unit "what it looks at" table; feature-evolution figure | interp/e11_mars_all.py | running (10 cluster jobs: per stage, compositional, shuffled, untrained) | same-call %, acc, R² per stage | results/jedi/e11_evolution.png | |
| E5v | E5 | 4 | A2/A3 | rotation / reflection / fresh-sample verification of E4/E5 | interp/e5_verify.py | script ready; job launch pending (agent's launch denied by its permission check) | | | code review: slot-window bit count (k+i+f vs i+f), sample overlap, octant means vary 2–2.5× |
| E16 | E1,E5 | 3 | A11 (user: Goodfire-style neurons) | neuron atlas: jet images per activation bin, observables, main job vs offshoots, negative pole, upstream/downstream, steering; per-stage tally | interp/e16_atlas.py | running (agent, cluster) | one-job / two-job / unexplained counts | results/jedi/e16/atlas_summary.png, card_*.png | |
