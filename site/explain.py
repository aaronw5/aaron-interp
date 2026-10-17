"""Plain-language layer of the results page: background, glossary, and for every claim a short plain summary, the
background needed to read it, and why it is tested. Merged into site/content.py by site/gen.py (keyed by claim id).
Writing rule: say what is meant in literal terms; no metaphor, no verdict words, no editorial language."""

BACKGROUND = [
    ('What this project does',
     'We take trained jet-tagging networks and try to write down, as explicit formulas, what each part of the network computes, '
     'from the input particles to the final class. Each formula is then tested by changing the input jets or the network\'s '
     'internal values and checking that the network responds the way the formula says it should.'),
    ('Jets and classes',
     'A jet is a spray of particles from one quark, gluon or decaying heavy particle. Each particle has a transverse momentum pT '
     '(GeV) and a direction (η, φ). Taggers decide what produced the jet: in the hls4ml dataset the classes are gluon (g), light '
     'quark (q), W boson (W, mass 80.4 GeV), Z boson (Z, 91.2 GeV) and top quark (t, 173 GeV). JetClass has 10 classes (QCD, Higgs '
     'decays, W, Z, top decays).'),
    ('Coordinates relative to the jet axis',
     'Every particle is given to the networks as an offset from a reference direction called the jet axis: Δη and Δφ, and their '
     'combination $r = \\Delta R = \\sqrt{\\Delta\\eta^2 + \\Delta\\phi^2}$. In both datasets the axis is the direction of the summed momentum of the jet, which is '
     'also the pT-weighted centre of the particles.'),
    ('The models',
     'JEDI-linear: a small network built for the CMS hardware trigger (an FPGA). It sees the 64 highest-pT particles, each as (pT, Δη, Δφ), '
     'and computes in stages: P (sums over particles of per-particle cuts), g (jet-level context), S (a second round of per-particle '
     'cuts, summed), h4 and h5 (dense layers), z (16 class neurons) and the class scores. It uses fixed-point numbers. '
     'ParT (Particle Transformer): a 2-million-parameter transformer trained on JetClass; it sees up to 128 particles with up to 17 '
     'features each and also pairwise features between particles. ParT_full uses all features, ParT_kin only the kinematic ones. '
     'Small taggers: plain transformers, Linformers and Deep Sets we trained on the same hls4ml inputs as JEDI-linear.'),
    ('How claims are tested',
     'Mean ablation: replace one unit\'s value by its average over many jets and count how many of the network\'s decisions change. '
     'Input edits: change the jets in a controlled way (shift, rotate, rescale particles) and compare the network\'s response with what '
     'a formula predicts. Read-outs (probes): fit a linear map from a layer\'s values to a physical quantity and measure how well it '
     'predicts on held-out jets (R²). Substitution: replace a layer by its formula and let the rest of the network run. A skeptic agent '
     'reruns each result on fresh jets and tries to refute it before a claim is marked verified.'),
]

GLOSSARY = [
    ('calls changed', 'the fraction of jets whose predicted class changes after an edit or intervention'),
    ('same call', 'the fraction of jets where a formula (or edited network) predicts the same class as the original network'),
    ('mean ablation', 'replacing a unit\'s value by its average over jets; the change in calls measures how much the decision uses the unit'),
    ('statement', 'a cut on one particle, e.g. "pT < 11 GeV" or "the particle lies on the +η side of the axis"; in the network, one ReLU unit applied to a particle'),
    ('object', 'the set of particles that pass one statement, inside a fixed range of pT ranks, summarised by how many there are and their summed pT, Δη, Δφ'),
    ('pT rank / rank window', 'particles are sorted by pT; rank 0 is the hardest. A rank window such as "ranks 0–13" means only the 14 hardest particles are counted'),
    ('M0, M1, M2', '$M_0 = \\sum_i p_{T,i}$ (total transverse momentum); $\\vec M_1 = \\sum_i p_{T,i}\\,\\vec r_i$ (pT-weighted offset of the particles from the axis; zero when the axis is the pT-weighted centre); $M_2 = \\sum_i p_{T,i}\\, r_i^2$ (pT-weighted spread of the particles around the axis)'),
    ('invariant mass m', 'the mass of the particle that produced the jet. For small angles $m^2 \\approx \\sum_{i<j} p_{T,i}\\, p_{T,j}\\, \\Delta R_{ij}^2 = M_0 M_2 - |\\vec M_1|^2$, which depends only on the angles between particles, not on where the axis is'),
    ('R² (held-out)', 'fraction of the variance of a target that a fitted formula predicts on jets not used for the fit (1 = perfect, 0 = no information)'),
    ('SD', 'standard deviation of a unit over jets; "moves 0.1 SD" means the change is one tenth of the unit\'s normal spread'),
    ('W → Z', 'fraction of true W jets that the network calls Z'),
    ('WTA axis', 'winner-take-all axis: the direction found by reclustering the jet and following the harder branch at each step; it points along the hardest prong rather than the centre'),
    ('hinge term / if-statement', '$\\max(0, Q - t)$ or $\\max(0, t - Q)$: zero on one side of a threshold $t$, linear on the other ("if $Q > t$, add $a\\,(Q - t)$"); the formulas are sums of such terms'),
    ('fixed point, zero-bit entry', 'JEDI-linear stores numbers with a learned number of bits per (particle slot, channel); entries with 0 bits are always zero'),
]

# per claim: plain summary, background, motivation (why it is tested / why it matters)
CLAIMS = {
'A1': dict(
    plain='JEDI-linear\'s first layer is 13 simple cuts on single particles. Summing a cut over the particles gives an "object": how many particles pass the cut and their summed momenta and positions. Which particles are counted is also fixed by pT rank.',
    background='The first layer applies the same 64 ReLU units to every particle and then adds them up over particles. A ReLU unit is zero below a threshold and linear above it, so summing it over particles gives exactly $\\sum_i \\mathrm{ReLU}(w\\cdot x_i + b) = b\\,|A| + w\\cdot\\sum_{i\\in A} x_i$, where $A$ is the set of particles with $w\\cdot x_i + b > 0$: the number of passing particles times the offset, plus the weights times the summed inputs of the passing particles. Because JEDI-linear runs in fixed point, each (particle slot, channel) entry has its own number of bits; entries with zero bits are always zero.',
    motivation='This is the first step of writing the network as formulas: if the first layer is exactly a set of cuts and counts, everything after it can be expressed in terms of those counts and sums. It also tests whether the fixed-point format is part of the computation or only an approximation.'),
'A2': dict(
    plain='After the first round, the network computes a few jet-level numbers (mainly the total pT) and adds them to every particle before the second round of cuts. Removing this context changes 15 % of the decisions, almost all through four objects.',
    background='JEDI-linear\'s "g" layer is computed from the first-round objects and added to each particle\'s input to the second round, so the second-round cuts can depend on the jet as a whole.',
    motivation='To know whether the second-round objects are fixed particle cuts or cuts whose thresholds move with the jet; this decides how the full formula must be written.'),
'A3': dict(
    plain='The second-round objects are again cuts on single particles, now on pT and on the distance r from the axis, summed over fixed pT ranks. Most of them depend only on r, not on direction; a few still depend on direction.',
    background='The first-round angular cuts are half-planes (e.g. "Δη > 0"). Physical jet quantities such as jet mass or girth depend on r, the distance from the axis, not on the direction. If the second round combines half-planes into functions of r, rotating a jet around its axis should not change these objects.',
    motivation='Checks whether the network builds rotation-symmetric quantities, as physics formulas are, and gives each object a name in physical terms (e.g. Σ pT·r²). The rotation test is the direct check: it leaves every particle\'s r unchanged.'),
'A4': dict(
    plain='The network does not compute the invariant mass. It computes $M_0 M_2 = \\left(\\sum_i p_{T,i}\\right)\\left(\\sum_i p_{T,i}\\, r_i^2\\right)$: the total momentum times the momentum-weighted spread of the particles around the given axis. On real jets the two are the same number, because the axis is the centre of the jet. If the jet is shifted away from its axis, the network\'s "mass" goes up while the true mass stays the same, and 77 % of W jets become Z calls.',
    background='For small angles the jet mass satisfies $m^2 \\approx M_0 M_2 - |\\vec M_1|^2$ with $M_0 = \\sum_i p_{T,i}$, $\\vec M_1 = \\sum_i p_{T,i}\\vec r_i$ (offset of the jet\'s centre from the axis) and $M_2 = \\sum_i p_{T,i} r_i^2$ (spread around the axis). The $-|\\vec M_1|^2$ term corrects for the axis not being at the centre ("recentring"). In the datasets the axis is the pT-weighted centre, so $\\vec M_1 \\approx 0$ on every training jet (median $|\\vec M_1|^2/(M_0 M_2) = 3.5\\times10^{-4}$) and both formulas give the same number. A rigid shift of all particles by $\\delta$ leaves every angle between particles unchanged, so the true mass is unchanged, but $M_2$ grows: $\\Delta M_2 = 2\\delta\\cdot\\vec M_1 + M_0\\delta^2$.',
    motivation='Mass is the main variable for W/Z/top tagging. Knowing which formula the network computes tells us how it will behave on jets that differ from the training data (e.g. a mis-measured or differently defined axis), and gives a worked example of how a high-level variable is built over the layers (sums at S, product at h4/h5, windows at z).'),
'A5': dict(
    plain='Test whether each of the 16 class neurons can be written as a formula of the network\'s own objects, and whether replacing the neurons by those formulas keeps the network\'s decisions.',
    background='Earlier work wrote the 16 neurons as formulas of hand-chosen jet observables (mass, τ21, ...). Here the inputs are the network\'s own S objects instead.',
    motivation='This closes the chain from particles to classes in the network\'s own terms, which is what "front to back" means.'),
'A6': dict(
    plain='Test whether each class corresponds to a combination of neuron conditions (e.g. "W-mass window on AND two-prong on AND top gate off"), and whether forcing that combination makes the network pick the class.',
    background='Earlier, on the fitted formula, W = n12 ∧ n4 ∧ n0 described 48 % of W calls with 97 % purity and, when imposed, turned 26 % of other jets into W.',
    motivation='The last step of the thesis: classes as combinations of objects.'),
'A9': dict(
    plain='Simple quantities (sums over particles) can be read from early layers; quantities that need products or ratios (mass, girth) appear later; quantities that need clustering (τ21, soft-drop mass) are read best at h4 and mostly not well.',
    background='"Algebraic depth" is how many steps beyond plain sums a quantity needs: 0 = a sum (ΣpT, particle count), 1 = a product or ratio of sums (mass, girth), 2 = ratios of those (D2, C2), 3 = needs reclustering or grooming (τ21, soft-drop mass).',
    motivation='Tests the user\'s idea that high-level variables are built over the layers, with a measurable prediction: the layer where a variable first becomes readable should rise with its depth.'),
'A11': dict(
    plain='Look at every neuron directly: which jets make it fire strongly, weakly, or (for the class neurons) negatively; describe the common pattern in physical terms; and test the description by setting the neuron high or low.',
    background='Cammarata (Goodfire, "Neural Networks Are Already Clean") argues that most neurons in vision and language models have one main job once the tools are good enough, with weak activations that are related to the main job.',
    motivation='Checks whether the same holds for a physics network and gives a per-neuron reference (cards) for the paper.'),
'A8/A10': dict(
    plain='Check whether the same stages (cuts, objects, products, windows) appear in other architectures: ParT, ParticleNet, and small transformers trained on the same data as JEDI.',
    background='A result on one 16-neuron trigger model needs comparison with larger and different models to be general.',
    motivation='Reviewers will ask whether the findings are specific to JEDI-linear.'),
'B1': dict(plain='The exact "cut → count + sum" identity holds for JEDI\'s pooled layers.', background='See A1.', motivation='The part of the method that needs no fitting.'),
'B2': dict(
    plain='The whole network, written as about 6,000 if-statements fitted layer by layer from physical quantities, reproduces 94 % of its decisions at the same accuracy.',
    background='Each unit is fitted with MARS-style hinge terms (if Q > t, add a·(Q − t)). Each layer is fitted on the previous layer\'s fitted values, so the final program never uses the network\'s own activations.',
    motivation='This is the "turn the model into if-statements" method applied to every unit, and tests whether such a program can stand in for the network.'),
'B3': dict(plain='A formula can make the same decisions as the network for different reasons.', background='Earlier ParT results: the formula kept 89 % of accuracy but 1–48 % of background rejection, and agreed on only half of the key particles.', motivation='Defines what "faithful" means for the method.'),
'B4': dict(plain='Compare the method with other ways of explaining networks.', background='Decision-tree extraction, symbolic regression, linear probes.', motivation='Needed for a method paper.'),
'C1': dict(plain='Most of ParT\'s attention pattern is set by a fixed formula of particle-pair geometry, not by the particles\' learned features.', background='Each ParT attention score is a learned content term plus a "pair bias" computed from pair features (ln kT, ln z, ln ΔR, ln m²).', motivation='Basis for describing ParT\'s routing as formulas.'),
'C2': dict(plain='In ParT the jet mass enters through the attention pattern; one head switches on or off around 85 GeV.', background='Measured by decomposing a neuron\'s change under a mass edit into attention and per-particle parts.', motivation='First step of the ParT mass circuit.'),
'C3': dict(
    plain='ParT has the same behaviour as JEDI: shifting the jet away from its axis raises its "mass" reading and turns W jets into Z, even though the true mass is unchanged. In ParT_full this comes entirely through one input, each particle\'s distance ΔR to the axis.',
    background='ParT\'s pair features depend only on differences between particles, so they do not change under the shift. Its per-particle inputs include Δη, Δφ and ΔR measured from the stored axis; those do change.',
    motivation='Tests whether the un-recentred mass is specific to JEDI or appears in a large transformer that also has shift-invariant pair inputs.'),
'C4': dict(plain='Run the "variables over layers" test on ParT\'s blocks.', background='See A9.', motivation='Cross-architecture version of A9.'),
'D1': dict(plain='JEDI tags W/Z/top mainly with mass windows plus total pT; it does not check that a top jet contains a W.', background='Physics edits from the earlier study.', motivation='Physics audit.'),
'D2': dict(plain='ParT identifies b-jets by counting displaced tracks rather than by their displacement; it uses lepton isolation; it does not use π⁰ → γγ pairs or the dead cone.', background='Physics edits from the earlier study (400 jets per class).', motivation='Physics audit.'),
'D4': dict(
    plain='Because the taggers measure spread around the given axis, they give different answers if the particles\' coordinates are measured from a different, equally valid axis definition. With the winner-take-all axis, ParT_full changes 12 % of its decisions and ParT_kin 34 %.',
    background='Experiments define the jet axis in different ways. The standard (E-scheme) axis is the summed momentum direction. The winner-take-all (WTA) axis follows the hardest branch when the jet is reclustered; it is less sensitive to soft radiation and is used for some substructure observables. For a two-prong W jet the WTA axis points along the harder prong, a median 0.075 rad from the centre.',
    motivation='A practical consequence of A4/C3: the networks\' mass sense depends on a convention that the true mass does not depend on. The size of the change is predicted by the $M_0 M_2$ formula (ParT_full read-out $\\Delta\\log\\hat m = +0.133$ vs predicted $+0.138$). The networks were trained with one convention, so part of any change is a distribution shift; the formula predicts its direction and size.'),
'D3': dict(plain='Rerun all physics edits with more jets and error bars.', background='Earlier runs used 400 jets per class.', motivation='Needed before publication.'),
}

PAPER_MOTIVATION = {
    'a': 'Most interpretability work explains parts of a network. This paper aims to write the whole decision of a deployed tagger as formulas in physical terms, check each step by intervention, and compare with larger architectures.',
    'b': 'A method paper: how to turn a trained tagger into if-statements that can be checked, and how to tell when such a rewrite reproduces the decisions but not the reasons.',
    'c': 'How a large transformer (ParT) computes jet mass and related variables, traced through its attention heads and neurons.',
    'd': 'Which known physics each tagger uses, measured by editing jets in physically meaningful ways and comparing with classical taggers.',
}

# Main results explained (overview tab): title, explanation, figure
KEY_RESULTS_HOW = (
    'How we found it, step by step',
    'There is no single "mass neuron". The evidence comes in four steps. '
    '(1) Read-out: a linear fit from a layer\'s values to the jet mass computed from the particles predicts it on held-out jets with $R^2$ = 0.73 at the first pooled layer P, '
    '0.977 at S and 0.991 at h4. This says the mass can be read from S onward; it does not say the network uses it. '
    '(2) Which units behave like a mass: we edit jets in two ways with known effects. Scaling all angles by $k$ multiplies the mass by $k$, $M_2$ by $k^2$ and leaves $M_0$; '
    'scaling all pT by $c$ multiplies the mass by $c$, $M_0$ by $c$ and $M_2$ by $c$. For each unit we take the ratio $\\rho$ of its response to the two edits (in log): '
    'a function of the mass has $\\rho = 1$, a function of $M_2$ alone $\\rho = 2$, of $M_0$ alone $\\rho = 0$. Units with $\\rho \\approx 1$: 2 in S, 9 in h4, 8 in h5, 3 in z. '
    'The pieces: S29 and S50 carry $M_2$ (sums of $p_T r^2$ over rank windows), S0 carries $M_0$ (summed pT of the 14 hardest particles, fed through the context unit g49). '
    'Freezing S0 halves the pT response of the h4/h5 mass units, so the product is formed in h4/h5. '
    '(3) The decision uses it: the W neuron z0 peaks at 75–77 GeV of input mass, the Z neurons z7/z14 at 94–98 GeV, the top gate z10 at 135–148 GeV, and W jets switch to Z calls '
    'between 80 and 90 GeV of edited mass, under both the angle and the pT edit. '
    '(4) Which mass: on ordinary jets the invariant mass $m$ and $\\sqrt{M_0 M_2}$ about the axis agree to a correlation of 0.99999, so no ordinary jet can tell them apart. '
    'The shift edit separates them ($m$ fixed, $\\sqrt{M_0 M_2}$ grows). The network\'s mass read-out grows at 0.74–0.84 of the $\\sqrt{M_0 M_2}$ prediction; other candidates '
    '($\\sum p_T r$, $\\sqrt{\\sum r^2}$) predict 1.7× and 2.1× too much. Not settled: the response is 0.74–0.84 rather than 1, a direction-dependent part of 12–20 % remains, '
    'and the network also uses the total pT at fixed mass (pT × 0.8 at fixed mass sends 78 % of W jets to top). So the internal variable is "mass about the given axis" to within these numbers, '
    'not an exact copy of any one formula.')

KEY_RESULTS = [
    ('The networks\' mass is not the invariant mass',
     'Summary (full step-by-step explanation in the Explained tab): JEDI-linear and ParT compute the jet mass as the momentum-weighted spread of the particles about the given axis, '
     '$\\sqrt{M_0 M_2}$, not as the invariant mass $m$. On every training jet the two are equal; when all particles are shifted together (true mass unchanged), the networks\' mass rises '
     'and W jets are called Z (JEDI-linear 77 %, ParT_full 31 %).',
     'results/site/X1_mass_shift.png'),
    ('Why this matters: the jet axis is a convention',
     'The axis from which particle positions are measured is a choice. The standard axis (E-scheme) is the summed momentum direction; the winner-take-all (WTA) axis follows the '
     'harder branch when the jet is reclustered and is used for some substructure measurements because soft radiation does not move it; a trigger may use a seed particle. '
     'The invariant mass is the same for every choice. The networks\' mass variable is not: for the W jet in the figure it reads 81 GeV about the standard axis, 106 GeV about the WTA axis '
     'and 125 GeV about the hardest particle. On 10,000 JetClass jets, measuring the inputs about the WTA axis changes 11.7 % of ParT_full\'s decisions (accuracy 85.9 → 80.9 %) and 34.1 % of '
     'ParT_kin\'s (74.5 → 56.0 %); the size of the change in ParT_full\'s internal mass (+0.133 in log) matches the $M_0 M_2$ prediction (+0.138). Part of any axis change is a shift away from the '
     'training data; the formula predicts its direction and size. JEDI-linear under the same test: running (E17).',
     'results/site/X2_axis.png'),
    ('The whole trigger network as if-statements',
     'Every one of JEDI-linear\'s 144 active units was fitted with hinge terms ("if $Q > t$, add $a\\,(Q - t)$"), layer by layer, each layer from the previous layer\'s fitted values and '
     'starting from physical quantities computed from the particles. The resulting program of 6,064 if-statements makes the same decision as the network on 94.0 % of held-out jets '
     '(94.6 % with half-plane sums as inputs) with 80.72 % accuracy against the network\'s 80.42 %. Read from these fits, mass-type terms carry 0 % of the formula weight in the first '
     'layers and 13 %, 21 %, 28 % in h4, h5 and the class neurons: the mass is formed in the dense layers. Verification is running.',
     'results/site/T5_e11_program.png'),
]


# A dedicated, ordered explanation of the mass result (rendered as its own tab). (heading, text, figure or None)
MASS_EXPLAINER = [
 ('1. Two quantities that are both called "mass"',
  'The invariant mass of a jet is the mass of the particle that produced it (W: 80.4 GeV, Z: 91.2 GeV, top: 173 GeV). It is computed from the particles\' momenta. '
  'For small angles it depends only on the angles between pairs of particles:'
  '$$m^2 \\approx \\sum_{i<j} p_{T,i}\\, p_{T,j}\\, \\Delta R_{ij}^2 .$$'
  'The networks do not receive pair angles. They receive each particle\'s position $\\vec r_i = (\\Delta\\eta_i, \\Delta\\phi_i)$ measured from the jet axis. Written with those positions, the same sum is exactly'
  '$$m^2 = M_0 M_2 - |\\vec M_1|^2, \\qquad M_0 = \\sum_i p_{T,i},\\quad M_2 = \\sum_i p_{T,i}\\, r_i^2,\\quad \\vec M_1 = \\sum_i p_{T,i}\\, \\vec r_i .$$'
  '$M_0$ is the total momentum, $M_2$ the momentum-weighted spread of the particles around the axis, and $\\vec M_1$ measures how far the jet\'s momentum-weighted centre is from the axis. '
  'If the last term is dropped, the result is a second quantity, the "mass about the axis" $\\sqrt{M_0 M_2}$. The two are equal whenever the axis is at the centre ($\\vec M_1 = 0$).', None),
 ('2. Why the difference never shows in the data',
  'In both datasets the jet axis is defined as the summed momentum direction, which is the momentum-weighted centre of the particles. So $\\vec M_1 \\approx 0$ on every jet '
  '(median $|\\vec M_1|^2/(M_0 M_2) = 3.5\\times10^{-4}$ on hls4ml jets) and the two masses agree with correlation 0.99999. A network trained on these jets gets the same answer from either formula. '
  '$\\sqrt{M_0 M_2}$ is simpler for these networks: it is two sums over particles and one product, which is the kind of operation JEDI-linear\'s sum-then-dense structure computes directly.', None),
 ('3. Question',
  'Does the network compute a mass at all, where, and which of the two?', None),
 ('4. Can the mass be read from the network?',
  'We fit a linear map from each layer\'s values to the jet mass (computed from the particles) on half of 15,000 jets and test it on the other half. Held-out $R^2$: '
  'P 0.73, g 0.76, S 0.977, h4 0.991, h5 0.985, z 0.967. From S onward the mass is almost fully readable. This shows the information is present; it does not show the network uses it.', None),
 ('5. Which units behave like a mass?',
  'We change the jets in two ways whose effect on each quantity is known. Scaling all angles by $k$ multiplies $m$ by $k$, $M_2$ by $k^2$ and leaves $M_0$ unchanged. '
  'Scaling all pT by $c$ multiplies $m$ by $c$, $M_0$ by $c$ and $M_2$ by $c$. For each unit we measure its response to both edits and take the ratio '
  '$\\rho = \\dfrac{dU/d\\log k}{dU/d\\log c}$. A unit that depends on the jet only through $m$ has $\\rho = 1$; through $M_2$ alone $\\rho = 2$; through $M_0$ alone $\\rho = 0$. '
  'Units with $\\rho \\approx 1$: none in P, 2 in S, 9 in h4, 8 in h5, 3 in z. In S the units follow $M_2$ (S29, S50; $\\rho \\approx 2$) or $M_0$ (S0; $\\rho = 0$) separately.', None),
 ('6. Where the pieces come from and where they are combined',
  '$M_0$: the first-layer unit a15 is linear in pT for every particle, so its sum P15 is $\\sum p_T$; the context unit g49 takes 72 % of its value from P15 and feeds the second-round object S0, '
  'the summed pT of the 14 hardest particles. $M_2$: the second-round objects S29 and S50 are sums of a hinge in $(\\ln p_T, r)$ that track $\\sum p_T r^2$ (held-out $R^2$ 0.98, 0.94); '
  'moving particles outward changes S29 by the amount the hinge predicts (slope 0.99). The product: when S0 is frozen at each jet\'s own unedited value, the pT response of the h4/h5 mass units halves '
  'and their $\\rho$ rises from about 1 to 2–5 ($M_2$-like). So $M_0$ and $M_2$ are formed separately at S and multiplied in h4/h5. In the if-statement fit of the whole network (E11), the h4 unit with the '
  'largest effect (h4:49) is written mainly as $\\sqrt{M_0 M_2} < 100.6$ GeV ($R^2$ 0.991).', 'results/site/X3_mass_path.png'),
 ('7. How the decision uses it',
  'The class neurons respond to the mass variable with windows: the W neuron z0 peaks at 75–77 GeV of input mass, the Z neurons z7 and z14 at 94–98 GeV, the top gate z10 at 135–148 GeV. '
  'Changing only the mass of W jets (by the angle or the pT edit) switches their calls to Z between 80 and 90 GeV of edited mass.', 'results/site/T2_mass_and_windows.png'),
 ('8. Which mass: the shift test',
  'To separate the two masses we move every particle of a jet by the same small angle $\\delta$ and leave the axis where it is. Angles between particles do not change, so $m$ does not change; '
  '$\\vec M_1$ becomes $M_0\\vec\\delta$ and $M_2$ grows by $M_0\\delta^2$ (plus $2\\vec\\delta\\cdot\\vec M_1 \\approx 0$). Predictions for the network\'s read-out mass: '
  'invariant mass, no change; $\\sqrt{M_0 M_2}$, an increase of $\\tfrac12\\Delta\\log M_2$. Measured on 10,000 fresh jets at $\\delta = 0.06$ rad: the read-out rises by 0.24–0.29 in log against 0.30 predicted '
  '(0.74–0.84 of the prediction at every $\\delta \\ge 0.02$, growing as $\\delta^2$), and 77 % of W jets are called Z (3 % unshifted). Two other candidates are excluded: $\\sum p_T r$ and $\\sqrt{\\sum r^2}$ '
  'predict 1.7× and 2.1× larger changes with a different dependence on $\\delta$. ParT, a different architecture, gives the same picture (0.66–0.77 of the prediction; true-W → Z 12 % → 31 %).', 'results/site/X1_mass_shift.png'),
 ('9. What is not settled',
  'The response is 0.74–0.84 of the $\\sqrt{M_0 M_2}$ prediction rather than 1. A part that depends on the direction of the shift remains (12–20 % of the response), consistent with the network building $r$ '
  'from nine directional cuts. The network also uses the total pT at fixed mass: lowering pT by 20 % while keeping $m$ fixed sends 78 % of W jets to top. So the internal variable is "mass about the given axis" '
  'to within these numbers, plus a separate dependence on total pT; it is not an exact copy of one formula.', None),
 ('10. Why it matters',
  'For interpretability: the network computes a different formula from the physics one, and the two can only be told apart away from the training data; the formula predicts how the network fails there. '
  'For physics: the jet axis is a convention (summed momentum, winner-take-all, or a seed particle in a trigger). The invariant mass does not depend on it; the networks\' mass variable does. '
  'With the winner-take-all axis, ParT_full changes 11.7 % of its decisions and ParT_kin 34.1 %; JEDI-linear under the same test is running (E17).', 'results/site/X2_axis.png'),
]
