"""Display symbols for the jet quantities (what readers see on the pages and slides), and descriptive code names for the few
quantities whose ids are jargon (girth, width, e2_sq; used in the exported Python files).  zᵢ = pTᵢ / ΣpT; indices count particles from the hardest (0).
symbol(qid) -> str;  CODE_RENAME: {old code name: new code name};  relabel(obj): the same JSON object with every quantity
name replaced by its symbol (dict keys and exact values; inside text, code names containing '_' or digits and the jargon names)."""
import re

SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
FIXED = {
    'sum_pt': 'ΣpT', 'log_sum_pt': 'ln ΣpT', 'mass': 'mass', 'mass_over_sum_pt': 'm/ΣpT', 'mass_over_sum_pt_sq': '(m/ΣpT)²',
    'max_dr': 'max ΔRᵢ', 'pt_dispersion': '√Σzᵢ²', 'z_1st': 'z(1st largest)', 'z_2nd': 'z(2nd largest)', 'z_3rd': 'z(3rd largest)',
    'z_top5': 'z(5 largest)', 'girth': 'ΣzΔR', 'girth2': 'ΣzΔR²', 'mean_eta': 'ΣzΔη', 'mean_eta2': 'ΣzΔη²', 'mean_phi': 'ΣzΔφ',
    'mean_phi2': 'ΣzΔφ²', 'n_particles': 'n', 'tau21': 'τ₂₁', 'tau32': 'τ₃₂', 'e2': 'e₂', 'C2': 'C₂', 'D2': 'D₂', 'e2_sq': 'ΣzᵢzⱼΔRᵢⱼ²',
    'LHA': 'Σz(ΔR/0.8)^½', 'centroid_offset': '|Σz(Δη,Δφ)|', 'planar_flow': 'planar flow', 'eccentricity': '1−λ₂/λ₁',
    'width': 'λ₁+λ₂', 'lam1': 'λ₁', 'lam2': 'λ₂', 'orientation_deg': 'axis angle', 'dr01': 'ΔR(0,1)', 'pt1_over_pt0': 'pT₁/pT₀',
    'pt_balance01': 'min(pT₀,pT₁)/(pT₀+pT₁)', 'pt1_dr01': 'pT₁ΔR(0,1)', 'm01': 'm(0,1)', 'm012': 'm(0,1,2)', 'min_pair_mass': 'min pair mass(0,1,2)',
    'max_pair_mass': 'max pair mass(0,1,2)', 'dr02': 'ΔR(0,2)', 'dr12': 'ΔR(1,2)', 'dr_min_012': 'min ΔR(0,1,2)',
    'dr_max_012': 'max ΔR(0,1,2)', 'pt2_over_pt0': 'pT₂/pT₀', 'mratio_min_012': 'min pair mass(0,1,2)/m(0,1,2)',
    'mratio_max_012': 'max pair mass(0,1,2)/m(0,1,2)', 'pt_entropy': '−Σz ln z', 'e3': 'e₃', 'e4': 'e₄', 'C3': 'C₃', 'D3': 'D₃', 'N2': 'N₂',
    'N3': 'N₃', 'M2': 'M₂', 'M3': 'M₃', 'C2_b2': 'C₂(β=2)', 'D2_b2': 'D₂(β=2)', 'tau1': 'τ₁', 'tau2': 'τ₂', 'tau3': 'τ₃', 'tau4': 'τ₄',
    'tau43': 'τ₄₃', 'tau21_b2': 'τ₂₁(β=2)', 'sj2_zsoft': 'z(softer of 2 subjets)', 'sj2_dr': 'ΔR(2 subjets)',
    'sj2_mass1': 'm(harder of 2 subjets)', 'sj2_mass2': 'm(softer of 2 subjets)', 'sj3_dr_min': 'min ΔR(3 subjets)',
    'sj3_dr_max': 'max ΔR(3 subjets)', 'sj3_pair_mass_min': 'min m_pair(3 subjets)', 'sj3_pair_mass_max': 'max m_pair(3 subjets)',
    'sj3_pairmin_over_m': 'min m_pair(3 subjets)/m', 'sj3_pairmax_over_m': 'max m_pair(3 subjets)/m', 'sd_mass': 'm_SD', 'sd_zg': 'z_g',
    'sd_rg': 'R_g', 'sd_nremoved': 'n(soft-drop removed)'}
CODE_RENAME = {'girth': 'sum_z_dr', 'girth2': 'sum_z_dr2', 'width': 'lam1_plus_lam2', 'e2_sq': 'sum_zz_dr2'}
CODE_RENAME.update({f'girth2_top{k}': f'sum_z_dr2_top{k}' for k in (2, 3, 5, 10, 15, 20, 30, 40, 50)})


def symbol(q):
    if q in FIXED: return FIXED[q]
    for pat, f in ((r'girth2_top(\d+)', lambda k: f'ΣzΔR²({k} hardest)'), (r'mass_top(\d+)', lambda k: f'm({k} hardest)'),
                   (r'sum_pt_top(\d+)', lambda k: f'ΣpT({k} hardest)'), (r'z_top(\d+)_slots', lambda k: f'z({k} hardest)'),
                   (r'n_real_top(\d+)', lambda k: f'n({k} hardest)'), (r'n_pt_above_(\d+)', lambda c: f'n(pT > {c} GeV)'),
                   (r'n_for_(\d+)pct', lambda p: f'n({p}% of ΣpT)'), (r'psi_0p(\d)', lambda r: f'Ψ(0.{r})')):
        m = re.fullmatch(pat, q)
        if m: return f(m.group(1))
    m = re.fullmatch(r'(pt|eta|phi|dr|z|abseta|absphi|zdr)_(\d+)', q)
    if m:
        i = m.group(2).translate(SUB)
        return {'pt': f'pT{i}', 'eta': f'Δη{i}', 'phi': f'Δφ{i}', 'dr': f'ΔR{i}', 'z': f'z{i}', 'abseta': f'|Δη{i}|', 'absphi': f'|Δφ{i}|',
                'zdr': f'z{i}ΔR{i}'}[m.group(1)]
    m = re.fullmatch(r'(dr0|dr1|ptdr0)_(\d+)', q)
    if m:
        i = m.group(2).translate(SUB)
        j = m.group(2); return {'dr0': f'ΔR(0,{j})', 'dr1': f'ΔR(1,{j})', 'ptdr0': f'pT{i}ΔR(0,{j})'}[m.group(1)]
    m = re.fullmatch(r'pair_mass_0_(\d+)', q)
    if m: return f'm(0,{m.group(1)})'
    m = re.fullmatch(r'(n_dr|z_dr)_(\d+p?\d*)_(\d+p?\d*|up)', q)
    if m:
        lo, hi = m.group(2).replace('p', '.'), m.group(3)
        rng = f'ΔR ≥ {lo}' if hi == 'up' else f'{lo} ≤ ΔR < {hi.replace("p", ".")}'
        return ('n' if m.group(1) == 'n_dr' else 'z') + f'({rng})'
    m = re.fullmatch(r'soft(\d+)_(pt|z|abseta|absphi|dr|dr0)', q)
    if m:
        s = m.group(1)
        if m.group(2) == 'dr0': return f'ΔR(0, softest {s})'
        return {'pt': 'pT', 'z': 'z', 'abseta': '|Δη|', 'absphi': '|Δφ|', 'dr': 'ΔR'}[m.group(2)] + f'(softest {s})'
    m = re.fullmatch(r'sj3_(z|mass)(\d)', q)
    if m: return ('z' if m.group(1) == 'z' else 'm') + f'(subjet {m.group(2)} of 3)'
    m = re.fullmatch(r'sj3_dr(\d)(\d)', q)
    if m: return f'ΔR(subjets {m.group(1)},{m.group(2)} of 3)'
    return q


def _token_re(ids):
    ids = sorted((q for q in ids if q != 'mass' and (('_' in q) or re.search(r'\d', q) or q in FIXED)), key=len, reverse=True)
    return re.compile(r'(?<![\w.])(' + '|'.join(re.escape(q) for q in ids) + r')(?![\w])')


def relabel(obj, ids):
    """replace quantity code names by symbols everywhere in a JSON object (keys, exact values, and inside text)"""
    ids = set(ids); rx = _token_re(ids)
    greek = re.compile(r'([λτ])(\d+)(?![\d.])')
    def s(x):
        if x in ids: return symbol(x)
        x = greek.sub(lambda m: m.group(1) + m.group(2).translate(SUB), rx.sub(lambda m: symbol(m.group(1)), x))
        return re.sub(r'(overall |average )?jet (width|λ₁\+λ₂)', 'λ₁+λ₂', x)
    def walk(o):
        if isinstance(o, dict): return {(s(k) if isinstance(k, str) else k): walk(v) for k, v in o.items()}
        if isinstance(o, list): return [walk(v) for v in o]
        if isinstance(o, str): return s(o)
        return o
    return walk(obj)


_CODE_RX = re.compile(r'(?<![\w])(' + '|'.join(sorted(map(re.escape, CODE_RENAME), key=len, reverse=True)) + r')(?![\w])')


def code_names(text):
    """the exported file text with the jargon code names replaced by descriptive ones (girth2 -> sum_z_dr2, ...)"""
    return _CODE_RX.sub(lambda m: CODE_RENAME[m.group(1)], text)
