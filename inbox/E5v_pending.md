# E5 verification (skeptic agent, 2026-10-16) — script written (interp/e5_verify.py, `python -m interp.e5_verify`), NOT yet run: the agent's
job launch (`./job.sh e5v 4 8 'python -m interp.e5_verify'`) was denied by its permission classifier ("shared cluster mutation").
Code-review findings (no run needed):
(b) The E5 causal sample (seed 1) overlaps the fit sample (seed 0) in 815 of 10,000 jets; the hinge parameters were fit on 80k particles
    from all jets (3 parameters; minor). Seed-7 overlaps seed-0 by 767 jets.
(d) e4_e5_jedi.py's `bits()` counts k+i+f, while HGQ's bit count is b = i+f (sign bit excluded); they disagree where k = 1 and i+f = 0,
    so a zero-magnitude quantizer was counted as kept. Likely source of the isolated slots in the windows (S50: 42, 44, 47; S40: 19–21, 29,
    44; S51: 14). Those slots output 0, so the R² effect is small, but the window descriptions need correcting.
(e) The inbox sentence "other 11 … hinge … R² 0.88–0.996" matches the 3-term hand-built fits (0.865–0.999), not the hinge sums (0.74–0.996)
    or the per-particle hinge R² (S13 0.61, S63 0.64, S50 0.71, S52 0.75, S40 0.76).
(f) The azimuth claim rests on an 8-octant cell R² (+≤ 0.017), yet octant means at r 0.1–0.3 vary by 2.5× for S1 (0.20–0.51) and 2× for
    S52 (0.88–1.78): the rotation test decides it.
Provisional confidence (unrun): R1 radial objects 0.3; R2 S29 naming 0.6; R3 context through four objects 0.6 (the 14.4 % counts changed
calls, not agreement with g → mean's calls; the script measures both).
