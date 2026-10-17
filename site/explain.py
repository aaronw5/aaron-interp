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
    ('mass read-out (of a network)', 'the network has no mass output. We fit a linear map from the 64 values of layer h4 to log(invariant mass) on ordinary jets (held-out R² 0.97); applying the same map to an edited jet gives the network\'s mass reading in GeV. The GeV values of $\\sqrt{M_0 M_2}$ in the figures are computed from the particles with the formula, not by the network'),
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
     'Summary (step-by-step explanation in the Paper A tab, "Main result explained"): JEDI-linear and ParT compute the jet mass as the momentum-weighted spread of the particles about the given axis, '
     '$\\sqrt{M_0 M_2}$, not as the invariant mass $m$. On every training jet the two are equal; when all particles are shifted together (true mass unchanged), the networks\' mass rises '
     'and W jets are called Z (JEDI-linear 77 %, ParT_full 31 %).',
     'results/site/X1_mass_shift.png'),
    ('Why this matters: the jet axis is a convention',
     'The axis from which particle positions are measured is a choice. The standard axis (E-scheme) is the summed momentum direction; the winner-take-all (WTA) axis follows the '
     'harder branch when the jet is reclustered and is used for some substructure measurements because soft radiation does not move it; a trigger may use a seed particle. '
     'The invariant mass is the same for every choice. The networks\' mass variable is not: for the W jet in the figure the formula $\\sqrt{M_0 M_2}$ gives 81 GeV about the standard axis, 106 GeV about the WTA axis '
     'and 125 GeV about the hardest particle; JEDI-linear\'s own mass read-out (see the glossary) gives 90, 110 and 109 GeV, and its call changes from W to q. On 10,000 JetClass jets, measuring the inputs about the WTA axis changes 11.7 % of ParT_full\'s decisions (accuracy 85.9 → 80.9 %) and 34.1 % of '
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
  'predict 1.7× and 2.1× larger changes with a different dependence on $\\delta$. ParT, a different architecture, gives the same picture (0.66–0.77 of the prediction; true-W → Z 12 % → 31 %). '
  'In the figure, the dashed curve is JEDI-linear\'s own mass read-out for one W jet (a linear map from h4 to log mass, fitted on ordinary jets): it reads 90 GeV unshifted (true 81 GeV; single-jet read-outs scatter by about 10 %), '
  'rises with the shift to 101 GeV at $\\delta = 0.06$, and the network\'s call changes from W to Z at $\\delta \\approx 0.045$ and to q above 0.07. The other two curves are computed from the particles with the formulas.', 'results/site/X1_mass_shift.png'),
 ('9. What is not settled',
  'The response is 0.74–0.84 of the $\\sqrt{M_0 M_2}$ prediction rather than 1. A part that depends on the direction of the shift remains (12–20 % of the response), consistent with the network building $r$ '
  'from nine directional cuts. The network also uses the total pT at fixed mass: lowering pT by 20 % while keeping $m$ fixed sends 78 % of W jets to top. So the internal variable is "mass about the given axis" '
  'to within these numbers, plus a separate dependence on total pT; it is not an exact copy of one formula.', None),
 ('10. Why it matters',
  'For interpretability: the network computes a different formula from the physics one, and the two can only be told apart away from the training data; the formula predicts how the network fails there. '
  'For physics: the jet axis is a convention (summed momentum, winner-take-all, or a seed particle in a trigger). The invariant mass does not depend on it; the networks\' mass variable does. '
  'With the winner-take-all axis, ParT_full changes 11.7 % of its decisions and ParT_kin 34.1 %; JEDI-linear under the same test is running (E17).', 'results/site/X2_axis.png'),
]



ARCH_EXPLAINER = [
 ('What goes in',
  'Each jet is given as its 64 highest-pT particles, sorted by pT (slot 0 is the hardest). Each particle is three numbers: its transverse momentum $p_T$ (GeV) and its position '
  '$(\\Delta\\eta, \\Delta\\phi)$ relative to the jet axis (radians). Jets with fewer than 64 particles have empty slots with $p_T = 0$. The network outputs a score for each class: g, q, W, Z, t.',
  'results/site/X0_architecture.png'),
 ('The letters',
  'The letters are the names of the network\'s layers in our code. Per particle (the same weights are applied to every particle): '
  '$a$ = the first layer, $a_i = \\mathrm{ReLU}(W_1 x_i + b_1)$, 64 units per particle; $s$ = a second per-particle layer, $s_i = \\mathrm{ReLU}(W_2 a_i + b_2)$; '
  '$u$ = the second-round per-particle layer, $u_i = \\mathrm{ReLU}(W_3 (s_i + g) + b_3)$. Per jet: '
  '$P = \\tfrac{1}{64}\\sum_i a_i$, the first layer summed over particles ("P" for pooled; we call its entries first-round objects); '
  '$g = \\mathrm{ReLU}(W_g P + b_g)$, 64 jet-level numbers ("g" for global context) that are added to every particle before the second round; '
  '$S = \\tfrac{1}{16}\\sum_i u_i$, the second round summed over particles ("S" for summed; second-round objects); '
  '$h_4, h_5$ = two dense layers on $S$ (64 and 32 units; "h" for hidden, numbered by layer); '
  '$z = W_6 h_5 + b_6$ = the 16 class neurons before their ReLU; the class scores are $W_7\\,\\mathrm{ReLU}(z) + b_7$.', None),
 ('Which units matter',
  'Many units never switch on. We count a unit as "live" if replacing it by its average over jets changes at least 0.1 % of the network\'s decisions. '
  'Units changing ≥ 1 % of decisions: 13 of 64 in $a$, 11 in $P$, 7 in $g$, 9 in $s$, 19 in $u$/$S$, 42 in $h_4$, 30 in $h_5$, 14 of 16 in $z$. '
  'The paper writes these units as formulas.', None),
 ('Fixed-point numbers',
  'JEDI-linear runs on an FPGA in the CMS trigger, so it stores numbers in fixed point with a learned number of bits for each (particle slot, channel). 80–84 % of the entries '
  'in the second round have zero bits and are always zero. As a result each second-round object only sums over a fixed range of pT ranks (for example "the 14 hardest particles"). '
  'With floating-point arithmetic and these zero entries kept, the network makes the same decision as the fixed-point network on 95.5 % of jets (80.68 % vs 81.29 % accuracy); '
  'without the zero entries it drops to 43.7 %. So the zero entries are part of what the network computes, and the rounding is a small perturbation.', None),
]

A1_EXPLAINER = [
 ('Question', 'What does the first layer compute, and can it be written exactly?', None),
 ('One unit on one particle is a cut',
  'A first-layer unit computes $\\mathrm{ReLU}(w\\cdot x + b)$ on one particle $x = (p_T, \\Delta\\eta, \\Delta\\phi)$. It is zero on one side of the plane $w\\cdot x + b = 0$ and grows linearly on the other. '
  'So each unit is a cut on a single particle ("a statement"). Converting the weights back to GeV and radians gives the cuts in physical units.', None),
 ('The 13 cuts',
  'Four cuts depend only on $p_T$: a32 is on for particles with $p_T < 11$ GeV, a56 for $p_T < 31$ GeV, a21 for $p_T < 93$ GeV, and a15 is always on and linear in $p_T$. '
  'Nine cuts depend only on direction: each is a line through the jet axis (within 0.01 rad), on for particles on one side of it. None mixes $p_T$ and direction. '
  'The angular cuts carry no $p_T$ weight, so they count particles regardless of their momentum.', 'results/site/X4_first_layer_cuts.png'),
 ('Summing a cut over particles gives an object',
  'The network sums each unit over particles. For a ReLU unit this sum is exactly '
  '$$\\sum_i \\mathrm{ReLU}(w\\cdot x_i + b) = b\\,|A| + w\\cdot\\sum_{i\\in A} x_i ,$$ where $A$ is the set of particles that pass the cut. So each pooled number is: how many particles pass the cut, '
  'and the sum of their $p_T$, $\\Delta\\eta$, $\\Delta\\phi$. Examples: a15 sums to $\\sum p_T$ (variance share 1.00); a32 sums to "number of particles below 11 GeV" and their summed $p_T$ (0.56 and 0.44); '
  'each angular cut sums to "number of particles on one side" and their summed offset on that side.', None),
 ('Checks',
  'The identity holds to 0.0 error on the network\'s own rounded inputs (10,000 jets) and on 260,000 jets on the cluster. The remaining difference to the network\'s stored values is fixed-point rounding '
  'per slot (7.8 % relative, the same as in an untrained network with the same number formats).', None),
 ('Why it matters',
  'The first half of the network is now exact formulas: lists of particles passing simple cuts, counted and summed. Everything after this is built from these counts and sums, which is what the '
  'thesis of the paper requires. The angular cuts weighting every particle equally (not by momentum) is a difference from the physics observables, which weight by $p_T$.', None),
]

A23_EXPLAINER = [
 ('Question', 'What does the second round compute, and does the jet as a whole influence it?', None),
 ('The context',
  'Before the second round the network computes 64 jet-level numbers $g$ from the first-round objects and adds them to every particle. The largest one, g49, takes 72 % of its value from '
  '$\\sum p_T$. Replacing $g$ by its average changes 14.6 % of decisions on fresh jets; four second-round objects (S0, S30, S31, S58) carry 13.9 % of that, and four random other objects carry 0–0.9 %.', None),
 ('The second-round objects',
  'For each second-round object we measured how much one particle contributes as a function of its $p_T$, its distance $r$ from the axis, and its direction. Direction adds almost nothing; '
  'the contribution is a cut in $(p_T, r)$, summed over a fixed range of pT ranks. Five examples with the decision fraction each one carries: S29 (40.9 %): particles at large $r$, more so at high $p_T$, '
  'tracking $\\sum p_T r^2$; S1 (20.2 %): particles beyond $r \\approx 0.25$; S0 (12.0 %): $\\sum p_T$ of the 14 hardest; S19 (11.1 %): number of particles below 2 GeV; S30 (8.0 %): number of particles within $r < 0.165$.',
  'results/site/X5_second_round_objects.png'),
 ('Test: rotate the whole jet',
  'If an object depends only on $r$, rotating the jet around its axis (which keeps every $r$) should not change it. On 10,000 fresh jets, rotations by 45°, 90°, 180° and the two mirror reflections '
  'move 10 of 17 objects by at most 0.1 of their normal spread (S0, S19, S20, S29, S30, S40, S43, S50, S51, S60); four move 0.1–0.2; three (S52, S56, S62) and the two context-only objects (S31, S58) change '
  'with direction. The rotations change 3.7–4.1 % of the network\'s decisions.', 'results/site/T3_rotation_test.png'),
 ('Test: move particles outward',
  'Moving every particle with $r > 0.15$ outward by 10 % changes S29 by +1.75; the fitted cut predicts +1.80 (slope 0.99). The three-term name "$\\sum p_T r^2 + \\sum r^2$ + soft count" predicts +1.98, '
  'so the cut itself is the accurate description and the name is an approximation.', 'results/site/T4_e5v_naming_context.png'),
 ('Why it matters',
  'Round two turns the nine directional half-planes of round one into functions of the distance from the axis, which is the kind of variable physics uses (girth, mass). The objects can be written as '
  'explicit cuts with a rank range. The few directional objects and the 4 % rotation sensitivity are differences from physics observables, which are rotation invariant.', None),
]

A9_EXPLAINER = [
 ('Question', 'Are complex jet variables computed later in the network than simple ones?', None),
 ('How complexity is defined',
  '"Algebraic depth" counts how many steps beyond plain sums over particles a variable needs. Depth 0: sums such as $\\sum p_T$ or the particle count. Depth 1: products or ratios of sums, such as the mass '
  '($\\sqrt{M_0 M_2}$) or girth ($\\sum p_T r / \\sum p_T$). Depth 2: ratios of those or three-particle correlations, such as $D_2 = e_3/e_2^3$. Depth 3: variables that need reclustering or grooming, such as $\\tau_{21}$ or the soft-drop mass. '
  '45 variables were labelled this way.', None),
 ('Measurement',
  'For each layer and variable we fit a linear map from the layer\'s live units to the variable on half of 20,000 jets and measure $R^2$ on the other half. We record the first layer where $R^2 \\ge 0.9$.', None),
 ('Result',
  'Depth 0–1 variables first reach 0.9 at P or S; most depth 2–3 variables never reach 0.9 and are best read at h4 (median $R^2$ 0.80 and 0.75). The correlation between depth and the first layer is 0.51 (Spearman, $p = 4\\times10^{-4}$). '
  'An untrained network with the same structure gives $R^2 \\le 0.006$ from S onward. The if-statement fit of the whole network (Paper B) agrees: terms using the mass appear first in h4.', 'results/jedi/F2_variables_over_layers.png'),
 ('Causal checks',
  'A read-out shows information is present, not that it is used. Changing the particle count while keeping everything else fixed moves the S read-out with slope 0.52; changing girth with $\\sum p_T$ fixed moves it with slope 0.91. '
  'Changing $D_2$ at fixed mass moves the h4 read-out with slope only 0.06: $D_2$ can be read from the network but the network does not respond to it.', None),
 ('Why it matters',
  'Gives a measurable version of "high-level variables are built over the layers": sums first, products next, three-particle and clustering variables only weakly and late. It also shows which classical '
  'substructure variables ($D_2$, $\\tau_{21}$) the network does not use, which matters for the physics paper.', None),
]

B2_EXPLAINER = [
 ('Question', 'Can the whole network be replaced by if-statements that a person can read, and does that program make the same decisions?', None),
 ('What "fit" means',
  'A unit of the network takes a value on every jet. To fit the unit means to choose a formula, here a sum of if-statements on input quantities, and adjust its thresholds and coefficients so that the formula\'s '
  'value is as close as possible to the unit\'s value over 20,000 jets (least squares). How well it fits is measured on other jets that were not used for the fit: $R^2 = 1$ means the formula reproduces the unit exactly, '
  '$R^2 = 0$ means it predicts nothing beyond the unit\'s average.', None),
 ('Which units are fitted',
  'Every layer that sits between two sums or after the second sum is fitted at the jet level: $P$, $g$, $S$, $h_4$, $h_5$, $z$. The per-particle layers ($a$, $s$, $u$) are not fitted one by one; their effect enters through the '
  'jet-level sums $P$ and $S$, which are fitted directly. Within a layer, only "live" units are fitted: units that change at least 0.1 % of the network\'s decisions when replaced by their average. '
  'Fitted / total: $P$ 12 / 64, $g$ 11 / 64, $S$ 19 / 64, $h_4$ 54 / 64, $h_5$ 32 / 32, $z$ 16 / 16 (144 units). Most of the others never switch on (for example 50 of 64 in $P$). '
  'Open point under verification: in the substitution test the units that were not fitted keep the network\'s own values; a skeptic is evaluating the program with those units set to constants, so that no network value enters.', None),
 ('Method (the MARS fit)',
  'Each unit is written as a sum of hinge terms $\\max(0, Q - t)$, $\\max(0, t - Q)$ and products of two, where $Q$ is an input quantity and $t$ a threshold. A hinge is an if-statement: "if $Q > t$, add $a\\,(Q - t)$". '
  'Terms are added one at a time, each time the one that reduces the error most, until held-out accuracy stops improving (at most 30 per unit).', None),
 ('Fitting layer by layer',
  'All 144 live units of JEDI-linear are fitted. The first layer uses physical quantities computed from the particles (sums in rank ranges, $\\sum p_T$, $\\sum p_T r^2$, counts, half-plane sums, and 383 standard jet observables). '
  'Each later layer is fitted on the previous layer\'s fitted values, not on the network\'s activations. The result is one program from particles to class scores; the network\'s own last layer turns the 16 fitted class neurons into scores.', None),
 ('Result',
  'On 8,848 jets not used for fitting, the program makes the same decision as the network on 94.0 % of jets (94.6 % with half-plane sums as inputs), with 80.72 % accuracy against the network\'s 80.42 %. '
  'Per class the accuracies agree within 1.6 points. Size: 3,824 terms, 6,064 if-statements, 211 physical inputs. Fits on an untrained network reach $R^2$ 0.08–0.18 from S onward, so the fit quality is a property of the trained weights.', 'results/site/T5_e11_program.png'),
 ('What the fits say about the layers',
  'Counting which kinds of quantity each layer\'s formulas use: radial moments dominate the second-round objects (70 % of formula weight); mass terms appear in h4 (13 %) and grow to 28 % at the class neurons; '
  'N-subjettiness, energy correlators and subjet variables stay at 4 % or less everywhere.', 'results/site/F4_e11_evolution.png'),
 ('Checks by a second agent (E11v)',
  'No network value enters the program: with every unfitted unit set to a constant it makes exactly the same decisions (94.02 %). On 17,011 fresh jets it agrees with the network on 94.2 % and its accuracy is 0.3 points lower (80.44 % vs 80.74 %); '
  'the small accuracy lead seen first does not hold. Background rejection for W jets at 50 % efficiency is 0.70× the network\'s; for the other classes it is within the uncertainty. The 6 % of disagreements are jets where the network itself is close to a tie.', None),
 ('The important test: edited jets',
  'Agreement on ordinary jets does not show that the program computes what the network computes. Under the rigid shift of all particles (invariant mass unchanged), the network calls 74.5 % of W jets Z. The nested program calls 77.6 % of them Z, because it builds the mass the same way '
  '(sums of $p_T r^2$ at S, then a product). A formula for the 16 class neurons fitted directly on physics observables, which agrees with the network on 92.5 % of ordinary jets, calls 0.1 % of shifted W jets Z: it uses the invariant mass. '
  'So the nested program agrees with the network in mechanism, the direct formula only in outcome on ordinary jets.', 'results/site/T11_e11_faithfulness.png'),
 ('Why it matters', 'A trigger network can be stated as a readable program that makes the same decisions on 94 % of jets and fails in the same way on edited jets. The comparison with the direct formula gives the method paper its test for when a rewrite reproduces the reasons and not only the outcome.', None),
]

C3_EXPLAINER = [
 ('Question', 'Does a large transformer compute the jet mass the same way as JEDI-linear (about the given axis), or as the invariant mass?', None),
 ('Why it could differ',
  'ParT receives pair features between particles (their $\\ln k_T$, $\\ln z$, $\\ln \\Delta R$, $\\ln m^2$). These depend only on the two particles, not on the axis, so the invariant mass could be computed from them alone. '
  'ParT also receives per-particle inputs measured from the axis ($\\Delta\\eta$, $\\Delta\\phi$, $\\Delta R$).', None),
 ('Test', 'The same shift as for JEDI-linear: move every particle by $\\delta$ in $\\phi$ with the axis fixed (true mass unchanged), 10,000 JetClass jets, two ParT models (all features; kinematics only).', None),
 ('Result',
  'Both respond. At $\\delta = 0.06$ rad ParT_full changes 8.1 % of decisions and true-W → Z rises from 12 % to 31 %; its internal mass rises by 0.66–0.77 of the $\\sqrt{M_0 M_2}$ prediction (JEDI-linear: 0.74–0.84). '
  'Feeding the shifted jets but keeping the per-particle $\\Delta R$ at its unshifted value removes 92–100 % of ParT_full\'s response: the response comes through that one input, not through the pair features.',
  'results/site/F3_part_translation.png'),
 ('Where it enters', 'The figure shows the same test with individual inputs held fixed.', 'results/site/F5_part_input_location.png'),
 ('What this does and does not show',
  'ParT can compute the invariant mass from its pair features: the pairwise masses $m_{ij}^2$ sum to the jet mass. The shift leaves every pair feature unchanged, so any response to the shift must come through the axis-relative per-particle inputs; the response shows that ParT uses that route. '
  'It does not show that ParT ignores the pair route. Both routes give the same answer on ordinary jets; under the shift they disagree, and ParT follows the axis-relative one by 0.66–0.77 of its full size (ParT_full). '
  'The next test (P2) changes the true mass and feeds the change to only one route at a time, to measure how much of ParT\'s ordinary mass response goes through each.', None),
 ('Why it matters', 'JEDI-linear has only the axis-relative route. ParT has both and its decisions still move when only the axis-relative route changes. The share of each route (P2) decides whether ParT\'s mass is mainly the invariant mass with an axis-dependent part, or mainly the moment mass. The full explanation of the mass variable is in the Paper A tab.', None),
]

D4_EXPLAINER = [
 ('Question', 'Does the networks\' way of computing the mass make them depend on how the jet axis is defined?', None),
 ('Background',
  'The jet axis is a convention. The standard axis (E-scheme) is the summed momentum direction. The winner-take-all (WTA) axis follows the harder branch when the jet is reclustered; '
  'it is insensitive to soft radiation and used for some substructure measurements. A trigger may use a seed particle. The invariant mass is the same for every choice; '
  '$\\sqrt{M_0 M_2}$ about the axis is not.', 'results/site/X2_axis.png'),
 ('Test', 'Recompute every particle\'s $\\Delta\\eta$, $\\Delta\\phi$, $\\Delta R$ about the WTA axis, the hardest particle, or the E-scheme axis recomputed from the stored particles, keeping the particles\' momenta unchanged; 10,000 JetClass jets.', None),
 ('Result',
  'WTA axis (a median 0.095 rad from the stored axis): ParT_full changes 11.7 % of decisions (accuracy 85.9 → 80.9 %), ParT_kin 34.1 % (74.5 → 56.0 %). Hardest particle: 14.0 % and 38.0 %. '
  'Recomputed E-scheme axis: 0.08 % and 0.18 %. ParT_full\'s internal mass changes by +0.133 in log against +0.138 predicted by $\\sqrt{M_0 M_2}$.', 'results/site/T6_axis_convention.png'),
 ('Result for JEDI-linear (E17)',
  'Same test on 20,000 hls4ml jets. On these jets the WTA axis is a median 0.040 rad from the stored axis (it coincides with the hardest particle in 66 % of jets). With inputs about the WTA axis JEDI-linear changes 30.3 % of its decisions and its accuracy drops from 80.6 % to 59.3 %; '
  'with the hardest-particle axis 31.5 % and 58.4 %. Most of the change is W jets called Z (W → Z rises from 2.1 % to 56 %) and Z jets called q. The internal mass rises by 0.77–0.82 of the $\\sqrt{M_0 M_2}$ prediction. '
  'Recomputing the centroid from the 64 stored particles (a 0.0015 rad change) changes 1.4 % of decisions. For W and Z jets the fraction of changed decisions grows with the axis displacement: 31 % at 0.02–0.04 rad, 71 % at 0.04–0.07 rad, 96 % at 0.07–0.10 rad.',
  'results/site/T10_axis_all.png'),
 ('Dependence on the size of the axis change', 'Left: fraction of changed decisions by how far the new axis is from the stored one, for W/Z jets and for g/q/t jets. Right: the network\'s internal mass change against the $\\sqrt{M_0 M_2}$ prediction.', 'results/site/F6_axis_bins_jedi.png'),
 ('Caveat', 'The networks were trained with one convention, so any other axis is also a change in the input distribution. The formula predicts the direction and size of the mass change.', None),
 ('Why it matters', 'A tagger deployed with a different axis definition than in training changes its decisions by an amount the mass formula predicts; for JEDI-linear a WTA or seed-particle axis changes about 30 % of decisions. Which axis the deployed trigger inputs use is an open question for the user. A follow-up trains a model on inputs about the hardest-particle axis, where the axis is never at the centre, to test whether a network then learns the recentred (invariant) mass (E18).', None),
]

EARLIER_C = [
 ('C1: ParT\'s attention is mostly a fixed formula of pair geometry',
  'Each attention score in ParT is a learned content term plus a "pair bias" computed from the pair features. The pair bias alone explains a median 79 % of the variance of the scores. A fitted formula of four pair features '
  'replaces the pair bias with 85.0 % accuracy against 85.4 %. Attention from the pair bias alone keeps 69.4 %, from the content term alone 40.2 %.', None),
 ('C2: the mass enters ParT through the attention',
  'Under a 10 % angle scaling (a 10 % mass increase) the W/Z neuron changes by −0.478 of its spread; −0.458 of that comes from changes in where the heads attend, −0.019 from the per-particle features. '
  'One head (b2h8) attends to the class token 87 % of the time at 60–65 GeV and 7 % above 105 GeV, switching between the W and Z masses.', None),
]
EARLIER_D = [
 ('D1: JEDI-linear and known physics (earlier edits)',
  'Changing only the mass (by scaling angles) turns W jets into Z at 91 GeV (93 %); lowering the jet momentum by 10 % turns W jets into top (87 → 28 % W) because the network also uses $\\sum p_T$; removing the W inside a top jet barely changes the top call.', None),
 ('D2: ParT and known physics (earlier edits, 400 jets per class)',
  'Making all tracks prompt (no displacement) drops H→bb̄ calls from 92 % to 10 %; shrinking the displacement by a factor 0.35 keeps 86 %, but keeping only two displaced tracks drops to 32 %: ParT counts displaced tracks rather than measuring how far they travel. '
  'Removing lepton isolation drops t→bℓν from 99 % to 71 %. No response to π⁰ → γγ pairing or to the heavy-quark dead cone.', None),
]

SMALL_EXPLAINER = [
 ('Question', 'Do other small architectures, trained on exactly JEDI-linear\'s inputs, reach its accuracy? Those that do are the comparison models for the cross-architecture part of the paper.', None),
 ('The models',
  'All take the 32 or 64 highest-pT particles as (pT, Δη, Δφ), normalised as for JEDI-linear, and are trained from scratch (30 epochs, 2 seeds, test on the 260,000 hls4ml test jets). '
  'Plain transformer (tf-$B$b-d$D$): each particle becomes a vector of size $D$; in each of $B$ attention blocks every particle reads a weighted mix of the others; the vectors are averaged and a linear layer gives the class scores. '
  'Linformer (linf): like the transformer, but each particle attends to $k$ learned mixtures of the particle slots instead of to every particle; because slots are ordered by pT, these mixtures are fixed weightings over pT ranks, '
  'comparable to JEDI-linear\'s rank windows. Deep Sets (dsets): a per-particle network, a sum over particles, then a dense network; this is JEDI-linear without the context round, so the cut-and-count identity of claim A1 applies to it exactly.', None),
 ('Result',
  'At 64 particles the 2-block Linformer reaches 80.91 ± 0.15 % and Deep Sets 80.73 ± 0.09 %, against JEDI-linear\'s 80.92 %. The plain transformers reach 69–76 %; '
  'their best epoch is the last one, so longer training is running (100 epochs, width 64). At 32 particles the ranking is the same (Linformer 78.37 %, Deep Sets 77.79 %, JEDI 78.00 %).',
  'results/site/T7_small_models.png'),
 ('Why it matters',
  'Two architectures with JEDI-linear\'s accuracy are available: one built on sums over particles (Deep Sets) and one on attention over pT-rank mixtures (Linformer). Running the same tests on them '
  '(first-layer cuts and objects, the mass shift test, variables over layers) shows which findings belong to JEDI-linear and which to models of this kind in general. All runs saved a checkpoint after every epoch, '
  'so the formation of each unit during training can be followed.', None),
]

# per-paper explainers, rendered inside the paper's tab (title, steps)
PAPER_EXPLAINERS = {
    'a': [('How to read this paper: JEDI-linear and what the letters (a, P, g, s, u, S, h4, h5, z) mean', ARCH_EXPLAINER, True),
          ('Result 1 explained: the first layer is 13 cuts, and summing them gives objects (claim A1)', A1_EXPLAINER, False),
          ('Result 2 explained: the context round and the second-round objects (claims A2, A3)', A23_EXPLAINER, False),
          ('Result 3 explained: the network\'s mass variable (claim A4; same test on ParT in Paper C, claim C3; axis consequence in Paper D, claim D4)', MASS_EXPLAINER, False),
          ('Result 4 explained: variables appear over the layers in order of complexity (claim A9)', A9_EXPLAINER, False),
          ('Result 5 explained: small comparison models trained on the same inputs (claims A8/A10)', SMALL_EXPLAINER, False)],
    'b': [('Result explained: the whole network as if-statements (claim B2)', B2_EXPLAINER, True)],
    'c': [('Result explained: ParT\'s decisions depend on the axis-relative inputs (claim C3)', C3_EXPLAINER, True),
          ('Earlier results in brief (claims C1, C2)', EARLIER_C, False)],
    'd': [('Result explained: the taggers depend on the jet-axis convention (claim D4)', D4_EXPLAINER, True),
          ('Earlier results in brief (claims D1, D2)', EARLIER_D, False)],
}
