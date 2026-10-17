# E6 — how JEDI-linear computes mass (agent report, 2026-10-16; verification pending → see inbox/E6_verify.md when present)

Sample: 3,000 test jets per class (15,000), seed 6; exact fixed-point network; accuracy 0.813.
Readout: ridge, held-out R², fixed 50/50 split; linear basis and linear + pairwise products of the top-8 PCA components.
Edits: angles × k (0.8–1.25); pT × c; translation r → r + δ (δ = 0.03, 0.06 rad along η or φ; m exactly unchanged);
iso-mass (pT × c, angles × 1/c: m unchanged, M0 × c, M2 ÷ c); dense k and c scans 0.5–2.0.
Controls: g frozen at each jet's unedited value; S0 frozen; same metrics on unedited jets.
Metric per unit: slopes e_k = dU/dlog k, e_c = dU/dlog c, ρ = e_k/e_c (m-like: 1; M2-like: 2; M0-like: 0).

Approximation: m² ≈ M0·M2 − |M1|² (NO factor 2; PAPERS.md A4 had an extra 2): corr with m 0.999986, median ratio 0.99998
(5–95 %: 0.997–1.003). |M1|²/(M0M2) median 3.5e-4 (inputs are axis-centred). M0 spread/mean 0.11, so log M2 alone gives
R² 0.990 for log m.

Readout R² (linear | with products):
| stage | M0 | |M1| | M2 | M0·M2 | m² | m | log m |
| P | .998|.997 | .44|.48 | .73|.82 | .68|.81 | .68|.81 | .73|.78 | .63|.68 |
| g | .86|.87 | .46|.48 | .77|.80 | .72|.77 | .72|.77 | .76|.78 | .64|.72 |
| S | .945|.951 | .59|.59 | .995|.996 | .965|.990 | .965|.990 | .977|.991 | .899|.962 |
| h4 | .943|.947 | .59|.59 | .995|.995 | .979|.988 | .980|.989 | .991|.994 | .973|.987 |
| h5 | .84|.92 | .59|.59 | .984|.989 | .972|.983 | .972|.983 | .985|.990 | .972|.985 |
| z | .75|.86 | .57|.57 | .972|.981 | .955|.978 | .956|.978 | .967|.985 | .943|.983 |

Key units (response in units of the unit's spread at k or c = 1.25):
- P angle units (9): +0.34…+0.36 ∝ k; no pT response (angle only, degree 1). P15: 0 | +2.20 ∝ c (M0).
- S0: 0 | +2.36 (+0.78 with g frozen): M0, mostly via g. S50: +0.54 | +0.25, ρ 1.96 (M2). S56, S1: +1.40, +1.20 | ≈0 (angle only,
  steeper than k²). S13, S43: ρ 1.30, 0.75, unchanged by g-freeze.
- h4 18, h5 0, h5 28: ρ 1.18, 1.10, 1.30 → with S0 frozen 2.67, 2.06, 2.98. z9, z11, z12: ρ 0.84, 0.90, 1.33 → 2.43, 4.79, 3.22.
- Units with ρ ≈ 1 fitting "m" better than "M2": S 2, h4 9, h5 8, z 3. M2-followers: S 1, h4 3, h5 1.
- Window shapes on the dense angle scan: S 0/17 responsive units; h4 5/50; h5 13/32; z 10/16.
- Linear (M0, M1, M2, count) model: S50 fit 0.94, predicted 0.61 vs observed 0.53; S56 fit 0.07, predicted 0 vs observed 1.37.
- log m̂ readouts under pT × 1.25 (true Δlog m 0.223): h4 0.203, h5 0.224, z 0.240; S0 frozen: 0.066, 0.050, 0.069; g frozen:
  0.111, 0.097, 0.129; angle response unchanged by either freeze.
- Translation 0.06 rad (m unchanged): log m̂ rises 0.24–0.29 at S, h4, h5, z (√(M0M2) predicts 0.306); 76 % of W jets called Z.
- Iso-mass: h4 log m̂ moves ≤ 0.057 over c = 0.67–1.5; P up to +0.16. Iso-mass c = 0.8: 78 % of W jets → t. pT × 0.8: 63 % of Z → t;
  with S0 frozen 1 %.

Mass-window knots (3-knot fit with per-jet offset; angle scan | pT scan; median per-jet peak mass):
z0 80, 110 (falls −0.19/GeV) | 75, 117.5; peak 74.7 / 76.6. z7 62.5, 92.5, 110 | 65, 92.5, 152.5; peak 94.0 / 96.0.
z12 65, 97.5, 177.5 | 62.5, 95, 207.5. z14 57.5, 97.5, 255 | 42.5, 100, 210; peak 96.2 / 98.3 (ρ 5.2: not m alone).
z10 87.5, 160 | 80, 137.5; peak 148 / 135. W → Z calls switch between 80 and 90 GeV of edited mass in both scans.

Reading (scoped): M0 and M2 are each linearly readable at S; units responding to angle and pT as m does sit mostly in h4/h5
and depend on S0 (two-thirds of its pT response comes through g); their output follows √(M0·M2) without the −|M1|² term.
Files: interp/e6_mass_jedi.py, results/jedi/e6_mass.json, logs/e6_mass.log; jedi_core.py gained run_jetwise (additive).
