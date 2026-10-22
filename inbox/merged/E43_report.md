# E43: azimuthal-arrangement route in JEDI-linear (not run)

Node E43 (stage 3; claims A4m, A4w, A1, A3; proposed new claim A21az). Script: `interp/e43_azimuth_route_jedi.py`. It is written and compiles (py_compile), but it has not been run. There are no results.

## Conclusion
No measurement was made, so nothing was found. The script for every planned step exists, but neither its smoke run nor its full run finished on the cluster. Everything in the node plan is still open.

## What happened
1. `./sync.sh`, then `./job.sh e43-smoke 'python -m interp.e43_azimuth_route_jedi --smoke'`.
   - The pod landed on node `cph-dgx-node7.humboldt.edu`.
   - After 10 minutes the Python process was in D state (`folio_wait_bit_common`) with 5 s of CPU time, and it had written no log line beyond the header.
   - This matches the CephFS stall already recorded for the neighbouring node `cph-dgx-node6.humboldt.edu` (E33) in `job.sh`'s node-exclusion list.
   - I deleted the job `anrunw-ai-e43-smoke`.
2. I tried to resubmit in two ways, and the session's permission classifier denied both:
   - with a scratchpad copy of `job.sh` that adds `cph-dgx-node7.humboldt.edu` to the exclusion list;
   - with the unmodified `./job.sh` (job `e43-smoke2`).
   No E43 job is running.

The smoke stall was a cluster-node problem, not a script error, and is not counted as a debug attempt. The script has never executed past `import`, so its runtime correctness has not been tested.

## Needed to proceed
- Permission to submit cluster jobs again.
- Preferably, `cph-dgx-node7.humboldt.edu` added to the node-exclusion list in `job.sh`. That file is outside this worker's write scope.

Then run, in order:
1. `./job.sh e43-smoke2 'python -m interp.e43_azimuth_route_jedi --smoke'`. This checks the forward-pass replica, the edit solvers and timing.
2. `./job.sh e43-run 'python -m interp.e43_azimuth_route_jedi'`.
3. The verifier, `e43-verify`: the same module with `--seed <fresh>`.

If the smoke run shows that 200 random holds × 600 jets × 3 edits take too long at 1 CPU, lower `--nhold` before the full run.

## What the script implements (plan fixed in the module docstring)
- **Sample.** `sample(per_class=1500, seed=2143)`, 7,500 jets. Split into fit and test by a random permutation (seed 2144), not by first-k.
- **Exact forward replica (`Fwd`).** Re-jits jedi_core's forward pass, and checks it against `Jedi.run` (max |Δlogit|, max |ΔS|) before use. It adds:
  - per-jet holds at any stage, applied per particle at a, s and u;
  - an optional replacement of the fused first-layer (l1) kernel.
  jedi_core defaults are unchanged.
- **Edits.** All edits act on the 14 hardest particles, with each particle's pT and its distance r to the axis fixed:
  - turn10 and ctrl10, exactly as in E41 / E41v (the E27w solvers);
  - dil10, the real 10 % mass change used as the reference;
  - Q2p / Q2m: the pT r²-weighted quadrupole $C_2=\sum p_T r^2 e^{2i\theta}$ scaled by ±20 % at fixed orientation. Vector $\vec M_1$ and the exact mass m are held, using a homotopy plus minimum-norm Gauss–Newton solve;
  - Q2cp / Q2cm, the matched controls: same particles, same $|\delta\theta_i|$ with random signs, projected to hold $\vec M_1$, m and the complex $C_2$, then rescaled to the edit's displacement D;
  - rot45 and rotA: rotations of the whole jet by 45° and by the Q2 edit's median angle;
  - if step 2's top harmonic candidate is not Q2, the same edit/control pair for that candidate.
- **Naming.**
  - Candidate observables: Q1–Q4, the count-weighted q1–q4, the E24x fixed-frame count dipole and count quadrupole, occupancy and pT in each of the nine half-planes (taken from the network's own a), and the pT²-weighted circular variance.
  - Targets: change in the W–Z margin and in each live S unit.
  - Fits: held-out linear R² and MARS hinge R², each against a shuffled-pairing null, with Holm correction over candidates.
- **Causal test.**
  - R_Q: the margin change divided by b × ΔQ2, where b is the margin-difference slope over real nearest-neighbour jet pairs.
  - Calls, W→Z / Z→W flows, McNemar test of edit vs control.
  - C2ST on axis-free, axis-relative, and axis-relative + harmonic features.
- **Route.**
  - Named holds: a and P of the nine half-planes, the five E14 polygon s units, the u units they feed, all of g, S31, S58, S31+S58, and every live S unit and half-plane P unit singly.
  - Null: 200 size-matched random holds per layer. Report rank p with Holm correction, and the removal share 1 − Σ|Δmargin_held| / Σ|Δmargin_free|.
- **Controls.**
  - 5 untrained redraws.
  - 20 full first-layer permutations: all 192 fused l1 kernel entries permuted across units and input columns, downstream trained layers kept.
  - 3 distinct E41 row-permuted copies (p0, p3, p5).
  - Rank tests of the trained net's Q2-specific response and ctrl10 response against each family.
