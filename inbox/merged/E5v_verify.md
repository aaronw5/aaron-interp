# E5v — verification of the E4/E5 readings R1–R3 (verifier report, 2026-10-16)

Script interp/e5_verify.py; results/jedi/e5_verify.json; logs/e5_verify.log (cluster job e5v, 926 s). Fresh sample: 2,000 jets/class,
seed 7 (767 of 10,000 are also in the E5 fit sample, seed 0; 815 of the E5-causal seed-1 jets were in the fit sample too). Fixed-point
accuracy 80.66 %; masked float 80.25 % (4.03 % calls differ). Nothing refit: the stored e4_e5.json coefficients are applied.

## Checks 1–2: rigid rotations (r of every particle unchanged, float max |Δr| ≤ 1e-7) and reflections
Calls changed: 45° 3.81 %, 90° 4.08 %, 180° 3.87 %, Δφ → −Δφ 3.73 %, Δη → −Δη 4.08 %; per-particle azimuth scramble 6.85 %.
Masked-float network: 3.13 / 3.90 / 3.17 / 3.04 / 3.62 / 6.37 %. Per class at 45°: g 3.9, q 3.8, W 3.5, Z 4.0, t 4.0 %. Accuracy 80.7–81.0 %.
First round, for contrast: 9 of 13 live P units move 0.7–1.2 SD (P3, 6, 29, 30, 36, 40, 43, 46, 57); P15, P21, P32, P56 move 0.
g5, g29, g36, g39, g44, g47, g55, g63 move 1.0–1.6 SD; g4 0.1–0.25; g49, g32 0. Reflections: g5, g29 respond to the η flip only,
g39, g55 to the φ flip only.
Second round, rms ΔS in units of SD(S) at 180° (grid-free: the quantized r changes by 0.0002), fixed point / masked float:
S0 0/0, S19 0/0, S50 0.06/0.03, S40 0.07/0.06, S29 0.08/0.06, S43 0.08/0.07, S51 0.08/0.07, S30 0.09/0.05, S60 0.09/0.08, S20 0.09/0.09,
S13 0.12/0.12, S63 0.16/0.14, S17 0.18/0.17, S1 0.21/0.16, S62 0.25/0.09, S52 0.29/0.28, S56 0.30/0.16, S58 0.41/0.18, S31 0.44/0.43.
45°, 90° and both reflections give the same ordering (45°: S58 0.68, S31 0.44, S56 0.34, S52 0.33, S62 0.26, S1 0.22; the 45° grid
effect in slots 48–63, quantized r moving 0.026, raises the fixed-point values of S1, S56, S62 above the masked-float ones).
Scramble: S58 1.52 (mean shift +1.03 SD), S31 1.25 (−0.69 SD); all other units as under rotation.
Carriers of the call change (one rotated S_c patched into the unrotated network, 180°): S1 2.27 %, S29 2.11 %, S56 1.58 %, S31 1.09 %,
S17 1.05 %, S30 0.83 %, S13 0.70 %; all live S patched 3.87 %.

## Check 3: S29 naming on the fresh sample
Stored formulas, R² vs fixed-point S29: 3-term 0.976 (0.976 on the 9,233 jets outside the fit sample), ΣpT·r² alone 0.880, hinge sum
0.937 (rescaled 0.971). All 17 particle units reproduce their stored R² within 0.01.
Edit: r × 1.1 for particles with r > 0.15 (14.1 particles/jet; 9,997 in-range jets; 8.78 % calls changed). S29 measured ΔS +1.749 ± 0.016
(0.20 SD; identical with g frozen). 3-term formula +1.979 (ratio 0.88, corr 0.935, slope 0.825, R²(Δ) 0.82); hinge sum +1.799 (ratio 0.97,
corr 0.948, slope 0.987, R²(Δ) 0.90); ΣpT·r² alone +1.145. The analytic prediction 0.21·(0.463 Σ_{r>0.15} pT r² + 7.30 Σ_{r>0.15} r²)
equals the formula's to 3e-6. Same edit on other units: S1 3-term slope 1.05, corr 0.93; S50 0.99, 0.92; S30 measured −0.301 but −0.048
with g frozen and 3-term corr 0.33 (the change comes through g, not through its Σr term); S63 measured +0.003, hinge predicts +0.069.

## Check 4: context
g → mean (fresh-sample mean; the E4-sample mean gives the same): 80.66 → 74.24 %, 14.61 % calls changed (g 21.3, q 14.4, W 16.4, Z 7.5,
t 13.4 %). g → 0: 51.65 %.
S_c → its mean-context value alone: S0 9.91, S30 4.22, S31 3.32, S58 1.75, S17 0.82, S13 0.31, S43 0.17, S20 0.04 %.
Four together (S0, S30, S31, S58): 13.91 %, acc 74.81 %; same call as g → mean on 98.07 % of jets and on 90.3 % of the 1,461 jets that
g → mean changes; 0.51 % changed where g → mean did not. All 19 live S together: 14.61 %, agreement 100 %.
Controls, 5 draws of 4 from the 19 live units: 1.78, 10.26 (contains S0), 4.96 (S30, S58), 0.04, 5.57 % (S30, S31). 5 draws from the
15 other units: 0.00, 0.82, 0.42, 0.17, 0.88 %.

## Check 5: code
Zero-bit masks from k+i+f > 0 vs HGQ's own b > 0: 0 disagreeing (slot, channel) entries for l3, qs1, qs and both inputs of add; KEEP
equals the stored pooled_slots; windows are read from qs1; the isolated slots (S50: 42, 44, 47; S40: 19–21, 29, 44; S51: 14) are real
nonzero-bit entries, so no window description changes. Pool scale 1/16. Pads are excluded from the hinge fits. The hinge parameters were
fit on particles from both jet halves (3 parameters; R² is unchanged on fresh jets). Masked-float vs fixed-point per-particle response
R²: S30 0.75, S62 0.72, S56 0.86, S1 0.89, others ≥ 0.95. The inbox line "Other 11 … hinge … R² 0.88–0.996" refers to the 3-term
hand-built fits (0.86–0.999); the hinge sums are 0.74–0.996 and the per-particle hinge R² is 0.61–0.76 for S13, S63, S50, S52, S40.

## Verdicts
R1 — reproduced: partly; refuted: yes as stated; confidence 0.85. Rigid rotations and reflections change 3.7–4.1 % of calls (masked
float 3.0–3.9 %), above the 3 % threshold; S1, S29 and S56 carry the change. Ten of the 17 particle units move ≤ 0.1 SD under every
r-preserving edit (S0, S19, S20, S29, S30, S40, S43, S50, S51, S60); S13, S17, S63, S1 move 0.12–0.22 SD; S52, S56, S62 move
0.25–0.43 SD (S52 0.28–0.42 in the masked-float network as well); the g-only objects S31, S58 move 0.4–0.7 SD under rotation and
1.3–1.5 SD under scramble. Corrected reading: round two is radial to ≤ 0.1 SD for the ten units listed (among them the five costliest
particle units), radial to 0.1–0.2 SD for four, and directional for S52, S56, S62 and for S31/S58, which are built from the directional
g units (g5, g29: η cuts; g39, g55: φ cuts). The fit-based statement "azimuth adds ≤ 0.017 R²" does not transfer to the network's calls.

R2 — reproduced: yes; refuted: no; confidence 0.9. R² 0.976 holds on fresh jets; the edit moves S29 with corr 0.94–0.95 and slope 0.99
(hinge) / 0.83 (3-term formula, which over-predicts by 13 % because its Σr² term counts soft particles at r 0.15–0.23 that lie below the
hinge threshold). Corrected reading: the causal object is the hinge sum Σ ReLU(4.44 ln pT + 93.8 r − 21.2)/16; ΣpT·r² + Σr² +
N(pT<2, r>0.1) is a correlational proxy for it.

R3 — reproduced: yes; refuted: no; confidence 0.9. 14.61 % (E4: 14.90 %) and 13.91 % (E4: 14.44 %); the four objects reproduce
g → mean's calls on 90 % of the affected jets; four random units from the other 15 change ≤ 0.9 %. Qualification: two of the four
(S31, S58) are directional objects, and S30's own term is not Σr (its edit response comes through g).
