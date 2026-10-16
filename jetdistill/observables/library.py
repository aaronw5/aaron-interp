"""The observables the formulas may use: one table, keyed by a short id.

Every observable is computed from the N particles the network sees, each (pT [GeV], Δη, Δφ) relative to the jet axis,
hardest first, empty slots (pT = 0) at the end. Each entry has
    label  the name shown on the pages,
    expr   a plain-Python expression, evaluated inside `quantities()` of an exported file (helpers: python_code.py),
    desc   a one-line description (units in brackets).
The numpy implementation (compute.py) follows the same algorithms and tie-breaking; tests/test_observables.py checks
that both give the same values. Every observable with units of mass has 'mass' in its id (the setups without mass
observables remove them by that rule, see config.py)."""
import itertools
from typing import NamedTuple

KH, KS = 15, 10            # per-particle observables: the KH hardest; the KS softest real particles beyond them (N > KH)
TOPK = (2, 3, 5, 10, 15, 20, 30, 40, 50)
RINGS = ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, None))


class Observable(NamedTuple):
    label: str
    expr: str
    desc: str


def ring_tag(lo, hi):
    return (f'{lo:g}_{hi:g}' if hi else f'{lo:g}_up').replace('.', 'p')


def library(n):
    """{id: Observable} for a network with n particles"""
    L = {}

    def add(id_, label, expr, desc):
        assert id_ not in L, id_
        L[id_] = Observable(label, expr, desc)

    # ---- whole jet ----
    add('sum_pt', 'Σᵢ pTᵢ', 'tot', 'total pT of the particles [GeV]')
    add('log_sum_pt', 'log Σᵢ pTᵢ', 'math.log(tot)', 'natural log of the total pT')
    add('mass', 'mass', 'mass_of(n)', 'invariant mass of all particles (massless four-vectors) [GeV]')
    add('mass_over_sum_pt', 'm / Σᵢ pTᵢ', 'mass_of(n) / tot', 'jet mass / total pT')
    add('mass_over_sum_pt_sq', '(m / Σᵢ pTᵢ)²', '(mass_of(n) / tot) ** 2', '(jet mass / total pT) squared')
    add('max_dr', 'maxᵢ ΔRᵢ', 'max(dr[i] for i in real)', 'largest distance ΔR of a particle from the jet axis')
    add('pt_dispersion', '√(Σᵢ pTᵢ²) / Σᵢ pTᵢ', 'math.sqrt(sum(x * x for x in z))', '√(Σ pTᵢ²) / Σ pTᵢ')
    add('z_1st', 'pT₀ / Σᵢ pTᵢ', 'zs[0]', 'largest pT share')
    add('z_2nd', 'pT₁ / Σᵢ pTᵢ', 'zs[1]', '2nd-largest pT share')
    add('z_3rd', 'pT₂ / Σᵢ pTᵢ', 'zs[2]', '3rd-largest pT share')
    add('z_top5', '(pT₀+pT₁+pT₂+pT₃+pT₄) / Σᵢ pTᵢ', 'sum(zs[:5])', 'pT share of the 5 largest')
    add('girth', 'Σᵢ pTᵢ·ΔRᵢ / Σᵢ pTᵢ', 'sum(z[i] * dr[i] for i in P)', 'pT-weighted mean ΔR')
    add('girth2', 'Σᵢ pTᵢ·ΔRᵢ² / Σᵢ pTᵢ', 'sum(z[i] * dr[i] ** 2 for i in P)', 'pT-weighted mean ΔR²')
    add('mean_eta', 'Σᵢ pTᵢ·Δηᵢ / Σᵢ pTᵢ', 'sum(z[i] * eta[i] for i in P)', 'pT-weighted mean Δη')
    add('mean_eta2', 'Σᵢ pTᵢ·Δηᵢ² / Σᵢ pTᵢ', 'sum(z[i] * eta[i] ** 2 for i in P)', 'pT-weighted mean Δη²')
    add('mean_phi', 'Σᵢ pTᵢ·Δφᵢ / Σᵢ pTᵢ', 'sum(z[i] * phi[i] for i in P)', 'pT-weighted mean Δφ')
    add('mean_phi2', 'Σᵢ pTᵢ·Δφᵢ² / Σᵢ pTᵢ', 'sum(z[i] * phi[i] ** 2 for i in P)', 'pT-weighted mean Δφ²')
    add('n_particles', 'number of particles', 'len(real)', 'number of real particles (pT > 0)')
    for c in (1, 5, 10, 50):
        add(f'n_pt_above_{c}', f'Σᵢ [pTᵢ > {c} GeV]', f'sum(1 for x in pt if x > {c})', f'number of particles with pT > {c} GeV')
    add('tau21', 'τ21', 'tau(2) / max(tau(1), 1e-12)', 'N-subjettiness τ2/τ1 (axes from pT-weighted k-means)')
    add('tau32', 'τ32', 'tau(3) / max(tau(2), 1e-12)', 'N-subjettiness τ3/τ2')
    add('e2', 'Σᵢ<ⱼ pTᵢpTⱼ·ΔRᵢⱼ / (Σᵢ pTᵢ)²', 'e2', 'energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ')
    add('C2', 'C2', 'e3 / max(e2 ** 2, 1e-12)', 'energy correlation ratio e3/e2²')
    add('D2', 'D2', 'e3 / max(e2 ** 3, 1e-12)', 'energy correlation ratio e3/e2³')
    add('e2_sq', 'Σᵢ<ⱼ zᵢzⱼΔRᵢⱼ²', 'sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j)', 'Σ_{i<j} zᵢzⱼΔRᵢⱼ²')
    add('LHA', 'LHA = Σᵢ zᵢ(ΔRᵢ/0.8)^½', 'sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P)', 'Les Houches angularity')
    add('centroid_offset', '√((Σᵢ zᵢΔηᵢ)² + (Σᵢ zᵢΔφᵢ)²)', 'math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P))', 'distance of the pT centroid from the jet axis')
    add('planar_flow', 'planar flow 4λ₁λ₂/(λ₁+λ₂)²', '4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12)', 'planar flow of the pT-weighted (Δη, Δφ) tensor')
    add('eccentricity', '1 − λ₂/λ₁', '1 - lam2 / max(lam1, 1e-12)', '1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor')
    add('width', 'λ₁ + λ₂', 'ta + tc', 'λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor')
    add('lam1', 'λ₁', 'lam1', 'larger eigenvalue of the pT-weighted (Δη, Δφ) tensor')
    add('lam2', 'λ₂', 'lam2', 'smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor')
    add('orientation_deg', '½·atan2(2c, a − b) [deg]', 'math.degrees(0.5 * math.atan2(2 * tb, ta - tc))', 'direction of the major axis [degrees]')
    # ---- the two / three hardest particles ----
    add('dr01', 'ΔR₀₁', 'math.sqrt(dist2(0, 1))', 'ΔR between particles 0 and 1')
    add('pt1_over_pt0', 'pT₁ / pT₀', 'pt[1] / max(pt[0], 1e-9)', 'pT1 / pT0')
    add('pt_balance01', 'min(pT₀, pT₁)/(pT₀ + pT₁)', 'min(pt[0], pt[1]) / max(pt[0] + pt[1], 1e-9)', 'min(pT0, pT1) / (pT0 + pT1)')
    add('pt1_dr01', 'pT₁·ΔR₀₁', 'pt[1] * math.sqrt(dist2(0, 1))', 'pT1 · ΔR01')
    add('m01', 'm₀₁', 'pair_mass(0, 1)', 'mass of particles 0 and 1 [GeV]')
    add('m012', 'm₀₁₂', 'math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2)', 'mass of particles 0, 1 and 2 [GeV]')
    add('min_pair_mass', 'min(m₀₁, m₀₂, m₁₂)', 'min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2))', 'smallest pair mass among particles 0, 1, 2 [GeV]')
    add('max_pair_mass', 'max(m₀₁, m₀₂, m₁₂)', 'max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2))', 'largest pair mass among particles 0, 1, 2 [GeV]')
    if n >= 3:
        add('dr02', 'ΔR₀₂', 'math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0', 'ΔR between particles 0 and 2')
        add('dr12', 'ΔR₁₂', 'math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0', 'ΔR between particles 1 and 2')
        d3 = 'math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))'
        add('dr_min_012', 'min(ΔR₀₁, ΔR₀₂, ΔR₁₂)', f'min({d3}) if pt[2] > 0 else math.sqrt(dist2(0, 1))', 'smallest distance among the 3 hardest')
        add('dr_max_012', 'max(ΔR₀₁, ΔR₀₂, ΔR₁₂)', f'max({d3}) if pt[2] > 0 else math.sqrt(dist2(0, 1))', 'largest distance among the 3 hardest')
        add('pt2_over_pt0', 'pT₂ / pT₀', 'pt[2] / max(pt[0], 1e-9)', 'pT2 / pT0')
        m012 = 'max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9)'
        add('mratio_min_012', 'min(m₀₁, m₀₂, m₁₂) / m₀₁₂', f'min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / {m012}', 'smallest pair mass / mass of the 3 hardest (dimensionless)')
        add('mratio_max_012', 'max(m₀₁, m₀₂, m₁₂) / m₀₁₂', f'max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / {m012}', 'largest pair mass / mass of the 3 hardest (dimensionless)')
    # ---- rings, integrated jet shape, counting ----
    for lo, hi in RINGS:
        rng, t = (f'{lo:g} ≤ ΔRᵢ < {hi:g}' if hi else f'ΔRᵢ ≥ {lo:g}'), ring_tag(lo, hi)
        cond = f'{lo:g} <= dr[i] < {hi:g}' if hi else f'{lo:g} <= dr[i] < 10'
        add(f'n_dr_{t}', f'Σᵢ [{rng}]', f'sum(1 for i in real if {cond})', f'number of particles with {rng.replace("ᵢ", "")}')
        add(f'z_dr_{t}', f'Σᵢ pTᵢ·[{rng}] / Σᵢ pTᵢ', f'sum(z[i] for i in real if {cond})', f'pT share of the particles with {rng.replace("ᵢ", "")}')
    for r in (0.1, 0.2, 0.3):
        add(f'psi_{r:g}'.replace('.', 'p'), f'Ψ({r:g}) = Σᵢ zᵢ [ΔRᵢ < {r:g}]', f'sum(z[i] for i in real if dr[i] < {r:g})', f'pT share within ΔR < {r:g} of the jet axis')
    for f in (0.5, 0.9):
        add(f'n_for_{int(100 * f)}pct', f'number of particles for {int(100 * f)}% of ΣpT', f'ncum({f:g})', f'number of hardest particles that carry {int(100 * f)}% of the jet pT')
    add('pt_entropy', 'pT entropy −Σᵢ zᵢ ln zᵢ', '-sum(z[i] * math.log(z[i]) for i in real)', 'pT entropy −Σ zᵢ ln zᵢ')
    # ---- each of the hardest particles ----
    for i in range(min(n, KH)):
        add(f'pt_{i}', f'pT of particle {i}', f'pt[{i}]', f'pT of particle {i} [GeV]')
        add(f'eta_{i}', f'Δη of particle {i}', f'eta[{i}]', f'Δη of particle {i}')
        add(f'phi_{i}', f'Δφ of particle {i}', f'phi[{i}]', f'Δφ of particle {i}')
        add(f'dr_{i}', f'ΔR of particle {i}', f'dr[{i}] if pt[{i}] > 0 else 0.0', f'ΔR of particle {i} from the jet axis')
        add(f'z_{i}', f'pT share of particle {i}', f'z[{i}]', f'pT of particle {i} / total pT')
        add(f'abseta_{i}', f'|Δη| of particle {i}', f'abs(eta[{i}])', f'|Δη| of particle {i}')
        add(f'absphi_{i}', f'|Δφ| of particle {i}', f'abs(phi[{i}])', f'|Δφ| of particle {i}')
        add(f'zdr_{i}', f'pT share × ΔR of particle {i}', f'z[{i}] * dr[{i}]', f'pT share × ΔR of particle {i} (its term in ΣzΔR)')
        if i >= 2:
            add(f'dr0_{i}', f'ΔR between particles 0 and {i}', f'math.sqrt(dist2(0, {i})) if pt[{i}] > 0 else 0.0', f'ΔR between particle {i} and the hardest particle')
            add(f'dr1_{i}', f'ΔR between particles 1 and {i}', f'math.sqrt(dist2(1, {i})) if pt[{i}] > 0 else 0.0', f'ΔR between particle {i} and the 2nd-hardest particle')
            add(f'ptdr0_{i}', f'pT of particle {i} × ΔR to particle 0', f'pt[{i}] * math.sqrt(dist2(0, {i})) if pt[{i}] > 0 else 0.0', f'pT{i} · ΔR(0, {i}) [GeV]')
            add(f'pair_mass_0_{i}', f'mass of particles 0 and {i}', f'pair_mass(0, {i})', f'mass of particles 0 and {i} [GeV]')
    # ---- the softest real particles beyond the hardest KH (a particle among the KH hardest is named by its hardest index) ----
    if n > KH:
        for s in range(1, KS + 1):
            for nm, w, d in (('pT', 'pt', 'pT [GeV]'), ('pT share', 'z', 'pT share'), ('|Δη|', 'abseta', '|Δη|'), ('|Δφ|', 'absphi', '|Δφ|'), ('ΔR', 'dr', 'ΔR from the jet axis')):
                add(f'soft{s}_{w}', f'{nm} of softest particle {s}', f'softp({s}, {w!r})', f'{d} of the {s}. softest real particle (0 if it is among the {KH} hardest)')
            add(f'soft{s}_dr0', f'ΔR between particle 0 and softest particle {s}', f"softp({s}, 'dr0')", f'ΔR between the hardest and the {s}. softest real particle (0 if among the {KH} hardest)')
    # ---- the k hardest particles ----
    for k in TOPK:
        if k >= n: break
        add(f'mass_top{k}', f'mass of the {k} hardest particles', f'mass_of({k})', f'mass of the {k} hardest particles [GeV]')
        add(f'sum_pt_top{k}', f'Σ pT of the {k} hardest particles', f'sum(pt[:{k}])', f'total pT of the {k} hardest particles [GeV]')
        add(f'z_top{k}_slots', f'pT share of the {k} hardest particles', f'sum(pt[:{k}]) / tot', f'pT share of the {k} hardest particles')
        add(f'girth2_top{k}', f'Σ zΔR² of the {k} hardest particles', f'sum(pt[i] * dr[i] ** 2 for i in range({k})) / max(sum(pt[:{k}]), 1e-9)', f'pT-weighted mean ΔR² of the {k} hardest particles')
        add(f'n_real_top{k}', f'number of real particles among the {k} hardest', f'sum(1 for x in pt[:{k}] if x > 0)', f'number of real particles among the {k} hardest')
    # ---- energy correlation functions (β = 1 unless noted) ----
    E = lambda s: f"ecf('{s}')"; mx = lambda s: f'max({s}, 1e-30)'
    add('e3', 'e₃ = Σᵢ<ⱼ<ₖ zᵢzⱼzₖ ΔRᵢⱼΔRᵢₖΔRⱼₖ', E('e3'), 'energy correlation e3 (β=1, 24 hardest)')
    add('e4', 'e₄ (hardest 12)', E('e4'), 'energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)')
    add('C3', 'C₃ = e₄e₂/e₃²', f"{E('e4')} * {E('e2')} / {mx(E('e3') + ' ** 2')}", 'energy correlation ratio e4·e2/e3² (small = three-prong)')
    add('D3', 'D₃ = e₄e₂³/e₃³', f"{E('e4')} * {E('e2')} ** 3 / {mx(E('e3') + ' ** 3')}", 'energy correlation ratio e4·e2³/e3³ (small = three-prong)')
    add('N2', 'N₂ = ₂e₃/(₁e₂)²', f"{E('g32')} / {mx(E('e2') + ' ** 2')}", 'generalized ECF ratio N2 (two smallest angles; small = two-prong)')
    add('N3', 'N₃ = ₂e₄/(₁e₃)²', f"{E('g42')} / {mx(E('g31') + ' ** 2')}", 'generalized ECF ratio N3 (small = three-prong)')
    add('M2', 'M₂ = ₁e₃/₁e₂', f"{E('g31')} / {mx(E('e2'))}", 'generalized ECF ratio M2 (smallest angle)')
    add('M3', 'M₃ = ₁e₄/₁e₃', f"{E('g41')} / {mx(E('g31'))}", 'generalized ECF ratio M3')
    add('C2_b2', 'C₂ (β=2) = e₃/e₂²', f"{E('e3b2')} / {mx(E('e2b2') + ' ** 2')}", 'energy correlation ratio e3/e2² with β = 2')
    add('D2_b2', 'D₂ (β=2) = e₃/e₂³', f"{E('e3b2')} / {mx(E('e2b2') + ' ** 3')}", 'energy correlation ratio e3/e2³ with β = 2')
    # ---- N-subjettiness, subjets ----
    for k in (1, 2, 3, 4): add(f'tau{k}', f'τ{k}', f'tau_n({k})', f'N-subjettiness τ{k} (β=1)')
    add('tau43', 'τ43', 'tau_n(4) / max(tau_n(3), 1e-12)', 'N-subjettiness τ4/τ3')
    add('tau21_b2', 'τ21 (β=2)', 'tau_n(2, 2) / max(tau_n(1, 2), 1e-12)', 'N-subjettiness τ2/τ1 with β = 2 (same axes)')
    add('sj2_zsoft', 'pT share of the softer of 2 subjets', 'subjets(2)["z"][1]', 'pT share of the softer of the 2 N-subjettiness subjets')
    add('sj2_dr', 'ΔR between 2 subjet axes', 'subjets(2)["dr"][0]', 'distance between the 2 subjet axes')
    add('sj2_mass1', 'mass of the harder of 2 subjets', 'subjets(2)["mass"][0]', 'mass of the harder of 2 subjets [GeV]')
    add('sj2_mass2', 'mass of the softer of 2 subjets', 'subjets(2)["mass"][1]', 'mass of the softer of 2 subjets [GeV]')
    for p in range(3):
        add(f'sj3_z{p + 1}', f'pT share of subjet {p + 1} of 3', f'subjets(3)["z"][{p}]', f'pT share of subjet {p + 1} of 3 (by pT)')
        add(f'sj3_mass{p + 1}', f'mass of subjet {p + 1} of 3', f'subjets(3)["mass"][{p}]', f'mass of subjet {p + 1} of 3 [GeV]')
    for c, (p, q) in enumerate(itertools.combinations(range(3), 2)):
        add(f'sj3_dr{p + 1}{q + 1}', f'ΔR between subjets {p + 1} and {q + 1} of 3', f'subjets(3)["dr"][{c}]', f'distance between subjet axes {p + 1} and {q + 1} (of 3)')
    add('sj3_dr_min', 'smallest ΔR among 3 subjets', 'min(subjets(3)["dr"])', 'smallest distance among the 3 subjet axes')
    add('sj3_dr_max', 'largest ΔR among 3 subjets', 'max(subjets(3)["dr"])', 'largest distance among the 3 subjet axes')
    add('sj3_pair_mass_min', 'smallest pair mass among 3 subjets', 'min(subjets(3)["mpair"])', 'smallest mass of two of the 3 subjets [GeV]')
    add('sj3_pair_mass_max', 'largest pair mass among 3 subjets', 'max(subjets(3)["mpair"])', 'largest mass of two of the 3 subjets [GeV]')
    add('sj3_pairmin_over_m', 'smallest subjet-pair m / jet m (3 subjets)', 'min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9)', 'smallest subjet-pair mass / jet mass (dimensionless)')
    add('sj3_pairmax_over_m', 'largest subjet-pair m / jet m (3 subjets)', 'max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9)', 'largest subjet-pair mass / jet mass (dimensionless)')
    # ---- soft drop ----
    add('sd_mass', 'soft-drop mass (β=0, z_cut=0.1)', 'softdrop("mass")', 'soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]')
    add('sd_zg', 'soft-drop z_g', 'softdrop("zg")', 'momentum sharing of the soft-drop splitting')
    add('sd_rg', 'soft-drop R_g', 'softdrop("rg")', 'angle of the soft-drop splitting')
    add('sd_nremoved', 'soft-drop: number of removed branches', 'softdrop("removed")', 'number of branches removed by soft drop')
    return L


def mass_ids(n):
    """observables with units of mass, and the dimensionless ratios built from masses (removed without mass observables)"""
    return {k for k, o in library(n).items() if 'mass' in k or k in ('m01', 'm012')}
