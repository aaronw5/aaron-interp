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
KEY_RESULTS = [
    ('The networks\' mass is not the invariant mass',
     'Physics defines the jet mass from the angles between pairs of particles: for small angles '
     '$m^2 \\approx \\sum_{i<j} p_{T,i}\\, p_{T,j}\\, \\Delta R_{ij}^2$. The same number can be written with positions measured from any reference point: '
     '$m^2 = M_0 M_2 - |\\vec M_1|^2$, where $M_0 = \\sum_i p_{T,i}$ is the total momentum, $M_2 = \\sum_i p_{T,i} r_i^2$ the momentum-weighted spread around the reference, '
     'and $\\vec M_1 = \\sum_i p_{T,i}\\vec r_i$ the offset of the jet\'s centre from the reference. The networks receive each particle\'s position relative to the jet axis, '
     'and in the training data the axis is always the jet\'s pT-weighted centre, so $\\vec M_1 = 0$ and $m^2 = M_0 M_2$ on every training jet. A network can therefore '
     'compute the mass as $M_0 M_2$ (two sums and one product, which a sum-pooling network does easily) and never needs the correction term. '
     'To find out which formula a network uses, we shift all particles of a jet together by $\\delta$ while keeping the axis fixed. The angles between particles do not change, '
     'so the invariant mass does not change; $M_2$ grows. JEDI-linear\'s internal mass grows by 0.74–0.84 of the $M_0 M_2$ prediction and 77 % of W jets (80 GeV) are called Z (91 GeV) at '
     '$\\delta = 0.06$ rad; ParT_full\'s grows by 0.66–0.77 of the prediction and true-W → Z rises from 12 % to 31 %. Both networks compute the momentum-weighted spread about the axis, not the invariant mass.',
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
