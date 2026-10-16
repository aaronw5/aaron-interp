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
| P0 | – | 1 | A8/A10 | ParT package + helpers copied into aaron-interp; smoke test ParT_full 86 %, ParT_kin 74 % | interp/part_core.py | running (agent) | | | |
| P1 | P0,E6 | 4 | A4/A8 | translation test on ParT_full / ParT_kin (mirror of E6); cross-model mass row | interp/p1_translation_part.py | running (agent) | | results/part/p1_translation.png | |

