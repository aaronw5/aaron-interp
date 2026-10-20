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
    plain='For each class, a rule of 2–3 conditions on the class neurons (for W: "W-mass neuron z0 on, z1 low, z4 on") picks out the jets the network calls that class with 96–98 % precision. Setting those neurons to typical values of the class moves 26–62 % of other jets into it; just entering the windows moves far fewer, because the last layer adds the neurons\' contributions.',
    background='Earlier, on the fitted formula, W = n12 ∧ n4 ∧ n0 described 48 % of W calls with 97 % purity and, when imposed, turned 26 % of other jets into W. The class scores of JEDI-linear are one linear layer on ReLU(z), so each neuron adds its own term to each class score.',
    motivation='The last step of the thesis: classes as combinations of objects. The test separates two readings: the class as a logical AND of windows, or the class as a sum of neuron contributions that a rule only approximates.'),
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
'A4n': dict(plain='On ordinary (unedited) jets whose axis is slightly off the pT centre, moving the particles so the jet is centred keeps the invariant mass the same but lowers the network\'s mass reading, as the axis-relative formula predicts in sign.', background='The rigid-shift test of A4 creates off-centre jets that do not occur in data; on most data the two mass definitions coincide. This test uses the naturally off-centre tail of real jets and moves them toward the centred configuration.', motivation='Review weakness 1 (off-manifold edits).'),
'B5c': dict(plain='A short formula (at most 50 hinge terms) on quantities measured about the jet axis, tested on whether it changes its calls when the jet is shifted or re-axised the way the network does.', background='B4 showed the 3,824-term program is not more compact than standard surrogates; the proposed thesis v3 needs a short formula that carries the axis dependence.', motivation='Review weakness 3: the compact form of the mass/axis account.'),
'D4m': dict(plain='When the jet axis convention changes, do the calls move the way each model\'s own mass formula predicts, or would any shift of the inputs of the same size do the same?', background='D4 measured that the axis convention changes 11.7–34.1 % of calls. A reviewer can read that as covariate shift: any model that is not translation invariant changes calls when its inputs move.', motivation='Review weakness 2: separates a formula-specific response from generic sensitivity to shifted inputs.'),
'B3': dict(plain='A formula can make the same decisions as the network for different reasons.', background='Earlier ParT results: the formula kept 89 % of accuracy but 1–48 % of background rejection, and agreed on only half of the key particles.', motivation='Defines what "faithful" means for the method.'),
'B4': dict(plain='Compare the method with other ways of explaining networks.', background='Decision-tree extraction, symbolic regression, linear probes.', motivation='Needed for a method paper.'),
'C1': dict(plain='Most of ParT\'s attention pattern is set by a fixed formula of particle-pair geometry, not by the particles\' learned features.', background='Each ParT attention score is a learned content term plus a "pair bias" computed from pair features (ln kT, ln z, ln ΔR, ln m²).', motivation='Basis for describing ParT\'s routing as formulas.'),
'C2': dict(plain='In ParT the jet mass enters through the attention pattern; one head switches on or off around 85 GeV.', background='Measured by decomposing a neuron\'s change under a mass edit into attention and per-particle parts.', motivation='First step of the ParT mass circuit.'),
'C3': dict(
    plain='ParT has the same behaviour as JEDI: shifting the jet away from its axis raises its "mass" reading and turns W jets into Z, even though the true mass is unchanged. In ParT_full this comes entirely through one input, each particle\'s distance ΔR to the axis.',
    background='ParT\'s pair features depend only on differences between particles, so they do not change under the shift. Its per-particle inputs include Δη, Δφ and ΔR measured from the stored axis; those do change.',
    motivation='Tests whether the un-recentred mass is specific to JEDI or appears in a large transformer that also has shift-invariant pair inputs.'),
'C4': dict(plain='Run the "variables over layers" test on ParT\'s blocks.', background='See A9.', motivation='Cross-architecture version of A9.'),
'C2a': dict(plain='One attention head near the end of ParT stops attending to the class token once the jet mass passes 87 GeV, between the W (80 GeV) and Z (91 GeV) masses; no other head of that stage does this.', background='ParT ends with two "class-attention" blocks in which a learned class token collects information from the particles through 16 heads; each head also gives part of its attention to the class token itself.', motivation='Step 2 of the Paper C thesis, re-measured at 10× the earlier statistics with the corrected input build.'),
'C2b': dict(plain='The mass information that reaches ParT\'s W/Z output unit goes through the first class-attention block, about half through one head; the 87 GeV switch head carries only about 5 % of it.', background='Measured by holding one head\'s output at its unedited value while the jet mass is changed, and counting how much of the output unit\'s change disappears.', motivation='Step 3 of the Paper C thesis: which units carry the mass into the decision.'),
'C5': dict(plain='ParticleNet, a graph network with no attention, also changes its W/Z calls when all particles are shifted relative to the jet axis at fixed mass, through the same input as ParT (each particle\'s distance to the axis).', background='ParticleNet groups each particle with its nearest neighbours in (Δη, Δφ) and reads the differences between them, which a rigid shift does not change; it also reads each particle\'s own features.', motivation='Tests whether the axis-relative mass response of C3 is specific to ParT\'s attention.'),
'C6': dict(plain='A network built to respect relativity (L-GATr) reads four-momenta, not offsets from a stored axis; the plan is to test whether it responds to mass-preserving changes of the axis spread.', background='No trained L-GATr is published for JetClass, so one is being trained with the authors\' code and settings.', motivation='The contrast model that decides whether the input format causes the axis response (C3, C5).'),
'C7': dict(plain='Inside ParT, one direction of the averaged particle representation follows the axis-moment mass (total pT times the pT-weighted squared distance from the axis, square-rooted) from the input embedding onward, and carries about two thirds of the response to a shift at fixed invariant mass; no single layer builds it.', background='Patching each layer, a learned one-dimensional direction, and copying its value between real jets that differ in the axis-moment mass.', motivation='Locates where the axis-relative mass of C3 is formed (the ParT-specific mechanism the reviewers ask for).'),
'C8': dict(plain='No circuit found with up to a tenth of ParT\'s connections reproduces its W-versus-Z calls when the rest comes from a jet of the other class; circuits are reported as accuracy against size.', background='Whole-network search over 21,504 connections with attribution ranking, greedy pruning and a trained mask.', motivation='Tests whether Paper C can name a literal circuit under the user\'s definition.'),
'C3s': dict(plain='Whether ParT retrained with other random seeds, on the 2M jets used for L-GATr, shows the same axis response; six trainings are running.', background='Interim numbers at epoch 3 of 50; pre-registered statements S1–S4 for the final checkpoints.', motivation='Removes the single-checkpoint and data-scale objections to C3 and the head-level claims.'),
'D1': dict(plain='JEDI tags W/Z/top mainly with mass windows plus total pT; it does not check that a top jet contains a W.', background='Physics edits from the earlier study.', motivation='Physics audit.'),
'D2': dict(plain='ParT identifies b-jets by counting displaced tracks rather than by their displacement; it uses lepton isolation; it does not use π⁰ → γγ pairs or the dead cone.', background='Physics edits from the earlier study (400 jets per class).', motivation='Physics audit.'),
'D4': dict(
    plain='Because the taggers measure spread around the given axis, they give different answers if the particles\' coordinates are measured from a different, equally valid axis definition. With the winner-take-all axis, ParT_full changes 12 % of its decisions and ParT_kin 34 %.',
    background='Experiments define the jet axis in different ways. The standard (E-scheme) axis is the summed momentum direction. The winner-take-all (WTA) axis follows the hardest branch when the jet is reclustered; it is less sensitive to soft radiation and is used for some substructure observables. For a two-prong W jet the WTA axis points along the harder prong, a median 0.075 rad from the centre.',
    motivation='A practical consequence of A4/C3: the networks\' mass sense depends on a convention that the true mass does not depend on. The size of the change is predicted by the $M_0 M_2$ formula (ParT_full read-out $\\Delta\\log\\hat m = +0.133$ vs predicted $+0.138$). The networks were trained with one convention, so part of any change is a distribution shift; the formula predicts its direction and size.'),
'D3': dict(plain='Rerun all physics edits with more jets and error bars.', background='Earlier runs used 400 jets per class.', motivation='Needed before publication.'),
'A11b': dict(
    plain='JEDI-linear\'s pooled units are individually readable as short physics formulas far more often than random mixings of the same units, and a sparse autoencoder does not make them more readable.',
    background='Any layer can be read in a rotated basis $v = Qh$ ($Q$ orthogonal) without losing information. If single units are special, they should be easier to describe than rotated directions. A sparse autoencoder (SAE) learns an overcomplete dictionary of sparse directions and is the usual alternative basis in interpretability work.',
    motivation='Review weakness 4: the framing "the units are already clean" needs a test against rotations and SAE features, with a rule fixed before the data are seen.'),
'A4c': dict(
    plain='Total pT and the pT-weighted spread about the axis each sit in one or two internal directions of the object layer; swapping those directions between jets changes the decision as editing the jet would. Their log-product is not one internal direction.',
    background='Distributed alignment search (DAS) learns an orthonormal $Q$ so that $h\' = h_b + (h_s - h_b)QQ^\\top$ reproduces the call of an input edit that sets a variable to the source jet\'s value. Effective IIA is the fraction of call-changing pairs it reproduces.',
    motivation='X9 found the program\'s named formulas no better than random rotations (review weakness 3). A learned subspace separates "no causal variable" from "wrong basis for the variable", and tests where $\\tfrac12\\log(M_0M_2)$ forms.'),
'A1r': dict(
    plain='Retraining JEDI-linear from scratch reproduces its axis-dependent mass reading, but not the strong rank-window effect of the official network; the "pure" first-layer cuts follow from sparsity.',
    background='JEDI-linear is trained with HGQ, which learns bit widths per weight and per particle slot; zero-bit weights are zeros and zero-bit slots are masks that fix pT-rank windows. The official network trained for 2,868 epochs.',
    motivation='Review weaknesses 2, 6, 7: are the program elements a property of the architecture and training, or of one checkpoint, and which need quantization-aware training? The float arm uses the same script with quantizers off.'),
'B2m': dict(
    plain='Are the program\'s variables also variables inside the network? Swapping a variable\'s value between two jets inside the network should change the decision as an input edit would.',
    background='Interchange intervention / causal abstraction (Geiger et al.): replace part of a layer for a base jet by its value on a source jet and compare the call with the call after an input edit that sets the variable to the source value.',
    motivation='Matching the network\'s calls (B2) does not show that the program computes the same intermediate quantities.'),
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
    'Freezing S0 halves the pT response of the h4/h5 mass units. Where the product forms is open: a later test (E15) could not separate a product from a sum followed by ReLU units (E15 verification). '
    '(3) The decision uses it: the W neuron z0 peaks at 75–77 GeV of input mass, the Z neurons z7/z14 at 94–98 GeV, the top gate z10 at 135–148 GeV, and W jets switch to Z calls '
    'between 80 and 90 GeV of edited mass, under both the angle and the pT edit. '
    '(4) Which mass: on ordinary jets the invariant mass $m$ and $\\sqrt{M_0 M_2}$ about the axis agree to a correlation of 0.99999, so no ordinary jet can tell them apart. '
    'The shift edit separates them ($m$ fixed, $\\sqrt{M_0 M_2}$ grows). The network\'s mass read-out grows at 0.74–0.84 of the $\\sqrt{M_0 M_2}$ prediction; other candidates '
    '($\\sum p_T r$, $\\sqrt{\\sum r^2}$) predict 1.7× and 2.1× too much. Not settled: the response is 0.74–0.84 rather than 1, a direction-dependent part of 12–20 % remains, '
    'and the network also uses the total pT at fixed mass (pT × 0.8 at fixed mass sends 78 % of W jets to top). So the internal variable is "mass about the given axis" to within these numbers, '
    'not an exact copy of any one formula.')

KEY_RESULTS = [
    ('Statistics for the claims (R1, R2, P2v)',
     'Every claim now carries a statistics line in its drop-down: effect with a 95 % bootstrap interval on fresh jets, a null or control, the hypotheses tested with Holm-corrected p-values, and the corrections that followed. '
     'The table summarises the claims of Papers A and B; the mass and axis statistics are in the Paper A and Paper C tabs.',
     'results/site/T14_claim_stats.png'),
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
     'training data; the formula predicts its direction and size. JEDI-linear under the same test (E17, 20,000 jets): the WTA axis changes 30.3 % of its calls (accuracy 80.6 → 59.3 %); the hardest-particle axis changes 31.5 % (E17) and 31.0–31.7 % on two fresh samples (X7/X7v).',
     'results/site/X2_axis.png'),
    ('The whole trigger network as if-statements',
     'Every one of JEDI-linear\'s 144 active units was fitted with hinge terms ("if $Q > t$, add $a\\,(Q - t)$"), layer by layer, each layer from the previous layer\'s fitted values and '
     'starting from physical quantities computed from the particles. The resulting program of 6,064 if-statements makes the same decision as the network on 94.0 % of held-out jets '
     '(94.6 % with half-plane sums as inputs). Its accuracy was 80.72 % against the network\'s 80.42 % on the first test set; on 17,011 fresh jets it is 0.31 ± 0.17 points below the network (E11v), and R2 measures −0.04 [−0.35, 0.26] points. Read from these fits, mass-type terms carry 0 % of the formula weight in the first '
     'layers and 13 %, 21 %, 28 % in h4, h5 and the class neurons: the mass is formed in the dense layers. Verified (E11v, R2): no fit/test overlap, the standalone program gives the same calls (94.02 %), and fresh jets give 94.10 % [93.76, 94.42] same call.',
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
 ('4. Can the mass be read from the network? (the "read-out")',
  'The network has no mass output. For every jet it computes internal numbers, for example the 64 values of layer $h_4$. '
  'Step 1: take 10,000 ordinary jets; for each we know its true mass $m$ (computed from its particles) and its 64 $h_4$ values. '
  'Step 2 ("fitting"): find 64 weights $w_k$ and an offset $b$ such that $\\log m \\approx b + \\sum_{k=1}^{64} w_k\\, h_{4,k}$ for these jets, as closely as possible (least squares with a small penalty, called ridge regression). '
  'Step 3: check on 10,000 other jets that were not used in step 2: the same weighted sum predicts $\\log m$ with $R^2 = 0.97$, i.e. it reproduces 97 % of how the mass varies from jet to jet. '
  'So the mass is present in $h_4$, and this weighted sum is our gauge for it. '
  'Step 4: for an edited jet, run the network, take its 64 $h_4$ values, apply the same weights, and convert $\\log m$ back to GeV. This is what we call the network\'s mass reading: the mass that the network\'s internal state corresponds to, judged by how that state relates to mass on ordinary jets. '
  'Limits: it is a gauge we built, not an output of the network; on a single jet it is off by about 10 %; and a fitted gauge reacts less than fully to changes (it shrinks toward the average), so its changes are compared with its response to a real mass change of the same size (calibration). '
  'The same procedure on each layer gives $R^2$ = 0.73 at P, 0.76 at g, 0.977 at S, 0.991 at $h_4$ (15,000 jets), 0.985 at $h_5$, 0.967 at $z$. The class decisions (for example W called Z) are the network\'s own output and do not depend on this gauge.', None),
 ('5. Which units behave like a mass?',
  'We change the jets in two ways whose effect on each quantity is known. Scaling all angles by $k$ multiplies $m$ by $k$, $M_2$ by $k^2$ and leaves $M_0$ unchanged. '
  'Scaling all pT by $c$ multiplies $m$ by $c$, $M_0$ by $c$ and $M_2$ by $c$. For each unit we measure its response to both edits and take the ratio '
  '$\\rho = \\dfrac{dU/d\\log k}{dU/d\\log c}$. A unit that depends on the jet only through $m$ has $\\rho = 1$; through $M_2$ alone $\\rho = 2$; through $M_0$ alone $\\rho = 0$. '
  'Units with $\\rho \\approx 1$: none in P, 2 in S, 9 in h4, 8 in h5, 3 in z. In S the units follow $M_2$ (S29, S50; $\\rho \\approx 2$) or $M_0$ (S0; $\\rho = 0$) separately.', None),
 ('6. Where the pieces come from and where they are combined',
  '$M_0$: the first-layer unit a15 is linear in pT for every particle, so its sum P15 is $\\sum p_T$; the context unit g49 takes 72 % of its value from P15 and feeds the second-round object S0, '
  'the summed pT of the 14 hardest particles. $M_2$: the second-round objects S29 and S50 are sums of a hinge in $(\\ln p_T, r)$ that track $\\sum p_T r^2$ (held-out $R^2$ 0.98, 0.94); '
  'moving particles outward changes S29 by the amount the hinge predicts (slope 0.99). The product: when S0 is frozen at each jet\'s own unedited value, the pT response of the h4/h5 mass units halves '
  'and their $\\rho$ rises from about 1 to 2–5 ($M_2$-like). So $M_0$ and $M_2$ are formed separately at S. A test of whether they are combined as a product in h4/h5 (E15) was refuted by its verification: the read-out statistic used gives product-like values also for a sum followed by ReLU units and for random ReLU layers on S, so the location of the product is open. In the if-statement fit of the whole network (E11), the h4 unit with the '
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
 ('8b. Statistics (R1)',
  'One read-out per model, calibrated by its response to a real mass change of the same predicted size; 95 % bootstrap intervals over 20,000 fresh jets; Holm correction over 165 tests. '
  'The invariant-mass hypothesis (no change) is rejected for every shift and axis change in all three networks. For JEDI-linear the calibrated ratio to $\\sqrt{M_0 M_2}$ is 0.56, 0.70, 0.90, 1.02 [1.00, 1.03] for shifts of 0.02, 0.03, 0.045, 0.06 rad, '
  '0.92 [0.90, 0.94] for the WTA axis and 0.97 [0.95, 0.99] for the hardest-particle axis; a 90° rotation (no change of either mass) moves the read-out by −0.0001 [−0.0009, +0.0007]. '
  'So $\\sqrt{M_0 M_2}$ describes large changes; small shifts move the reading by only 56–74 % of it. One more result: at 0.06 rad the read-out moves as much as under a real mass change of the same size, but 77 % of W jets are called Z against 12 % under the real change, '
  'so the decision depends on more than this one mass variable.', 'results/site/T13_mass_axis_stats.png'),
 ('9. What is not settled',
  'The response is 0.74–0.84 of the $\\sqrt{M_0 M_2}$ prediction rather than 1. A part that depends on the direction of the shift remains (12–20 % of the response); E14 showed that it is the sum of the second-round per-particle responses, which are polygons built from 3–9 of the directional cuts (0.96–1.05 of the measured direction term). '
  ' The network also uses the total pT at fixed mass: lowering pT by 20 % while keeping $m$ fixed sends 78 % of W jets to top. So the internal variable is "mass about the given axis" '
  'to within these numbers, plus a separate dependence on total pT; it is not an exact copy of one formula.', None),
 ('10. Why it matters',
  'For interpretability: the network computes a different formula from the physics one, and the two can only be told apart away from the training data; the formula predicts how the network fails there. '
  'For physics: the jet axis is a convention (summed momentum, winner-take-all, or a seed particle in a trigger). The invariant mass does not depend on it; the networks\' mass variable does. '
  'With the winner-take-all axis, ParT_full changes 11.7 % of its decisions, ParT_kin 34.1 % and JEDI-linear 30.3 % (E17); about the hardest particle JEDI-linear changes 31–32 % (E17, X7/X7v).', 'results/site/X2_axis.png'),
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
  'move 5 of 17 objects by significantly less than 0.1 of their normal spread under every one of these edits (S0, S19, S20, S29, S51; R2, 20,000 jets, 95 % intervals); 10 objects (adding S30, S40, S43, S50, S60) stay below 0.1 only under the 180° rotation; four move 0.1–0.2; three (S52, S56, S62) and the two context-only objects (S31, S58) change '
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
  'The units that were not fitted keep constant values in the program, so no network value enters; the standalone program makes the same decisions as the substitution test (94.02 %, E11v).', None),
 ('Method (the MARS fit)',
  'Each unit is written as a sum of hinge terms $\\max(0, Q - t)$, $\\max(0, t - Q)$ and products of two, where $Q$ is an input quantity and $t$ a threshold. A hinge is an if-statement: "if $Q > t$, add $a\\,(Q - t)$". '
  'Terms are added one at a time, each time the one that reduces the error most, until held-out accuracy stops improving (at most 30 per unit).', None),
 ('Fitting layer by layer',
  'All 144 live units of JEDI-linear are fitted. The first layer uses physical quantities computed from the particles (sums in rank ranges, $\\sum p_T$, $\\sum p_T r^2$, counts, half-plane sums, and 383 standard jet observables). '
  'Each later layer is fitted on the previous layer\'s fitted values, not on the network\'s activations. The result is one program from particles to class scores; the network\'s own last layer turns the 16 fitted class neurons into scores.', None),
 ('Result',
  'On 8,848 jets not used for fitting, the program makes the same decision as the network on 94.0 % of jets (94.6 % with half-plane sums as inputs), with 80.72 % accuracy against the network\'s 80.42 %; on fresh jets this accuracy lead does not hold (E11v: 0.31 ± 0.17 points below the network; R2: −0.04 [−0.35, 0.26] points). '
  'Per class the accuracies agree within 1.6 points. Size: 3,824 terms, 6,064 if-statements, 268 physical inputs (E11v; E11 reported 211). Fits on an untrained network reach $R^2$ 0.08–0.18 from S onward, so the fit quality is a property of the trained weights.', 'results/site/T5_e11_program.png'),
 ('What the fits say about the layers',
  'Counting which kinds of quantity each layer\'s formulas use: radial moments dominate the second-round objects (70 % of formula weight); mass terms appear in h4 (13 %) and grow to 28 % at the class neurons; '
  'N-subjettiness, energy correlators and subjet variables stay at 4 % or less everywhere.', 'results/site/F4_e11_evolution.png'),
 ('Checks by a second agent (E11v)',
  'No network value enters the program: with every unfitted unit set to a constant it makes exactly the same decisions (94.02 %). On 17,011 fresh jets it agrees with the network on 94.2 % and its accuracy is 0.3 points lower (80.44 % vs 80.74 %); '
  'the small accuracy lead seen first does not hold. Background rejection for W jets at 50 % efficiency was 0.70× [0.50, 0.95] the network\'s in E11v and 1.13× [0.96, 1.42] in R2 on 20,000 other fresh jets; only 60 background jets pass the cut, so the W rejection ratio is not resolved. For the other classes it is within the uncertainty. The 6 % of disagreements are jets where the network itself is close to a tie.', None),
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
 ('Test 2 (P2): which input route carries a real mass change',
  'We make a real mass change (all angles × $k$, or all pT × $c$) and feed it to ParT through one route at a time: (a) both routes; (b) pair features from the edited jet with every per-particle input from the unedited jet; '
  '(c) per-particle inputs from the edited jet with pair features from the unedited jet. For angles × 1.1 (true mass +9.4 %), ParT_full\'s mass read-out rises by +0.076 in (a), +0.001 in (b) and +0.075 in (c); '
  'W → Z goes from 0.12 to 0.37 in (a), 0.14 in (b) and 0.33 in (c). Over all edits the per-particle route carries 93–101 % of the read-out change and the pair route 1–6 % (ParT_kin 5–6 %); the pair route alone moves 10–21 % of the W → Z change for mass increases and 26–42 % for decreases, '
  'with almost no change in the read-out. For mass increases the two routes add up: (b) + (c) matches (a) to within 0.008 in the read-out. For mass decreases they do not (P2v: the pair route alone gives 0.24–0.58 of the logit change and (b) + (c) gives 1.20–1.71 of (a)). One row is unusual: ParT_kin with pT × 0.8 through the pair route alone sends 54 % of W jets to H → bb̄. P2v excluded an input-range effect (no edited pair feature leaves the unedited range; the effect is the same for jets that stay in the 500–1000 GeV pT range); the response goes through the ln m² pair feature: shifting ln m² alone by −0.446 sends 53 % of W jets to H → bb̄, shifting ln kT alone 0.4 %.', 'results/site/T12_part_routes.png'),
 ('What this does and does not show',
  'ParT can compute the invariant mass from its pair features: the pairwise masses $m_{ij}^2$ sum to the jet mass. The shift leaves every pair feature unchanged, so any response to the shift must come through the axis-relative per-particle inputs; the response shows that ParT uses that route. '
  'Test 2 answers the remaining question: ParT does use the pair route, but for a small part of its response (about a tenth to a fifth of the W → Z change and a few percent of the mass read-out). Its mass sense comes mostly from the per-particle inputs measured from the axis, as in JEDI-linear, which has only that route. P2v (fresh rows, 2,000 stratified bootstraps): for mass increases the pair route alone carries 0.10–0.20 of the W − Z logit change and the per-particle route 0.82–0.97; both "share 0" and "share 1" are rejected (Holm p ≤ 0.03).', None),
 ('Why it matters', 'JEDI-linear has only the axis-relative route. ParT has both and its decisions still move when only the axis-relative route changes. The share of each route (P2) decides whether ParT\'s mass is mainly the invariant mass with an axis-dependent part, or mainly the moment mass. The full explanation of the mass variable is in the Paper A tab.', None),
]

P3_EXPLAINER = [
 ('Question', 'Does ParT_full\'s class-attention head b2h8 act as a W/Z mass switch, and does it carry the jet mass into the W/Z coordinate $z_{97}$ ("n98") of the class token? What does the circuit {b2h8, b1h3, n98} score when it is kept alone?', None),
 ('Background',
  'ParT ends with two class-attention blocks (b1, b2), each with 8 heads. A learned class token reads the particle embeddings only through the 16 head context vectors $c_h=\\sum_p a_{hp} v_{hp}$, where $a_{hp}$ is head $h$\'s attention to particle $p$ (the class token itself is $p=0$) and $v_{hp}$ the value vector. Its final value $z$ gives the class scores through $\\ell = zW + b$. '
  'Earlier fits (G21–G22, 400 jets per class, an η build that mirrored jets with $\\eta<0$) suggested that b2h8 turns its attention away from the class token as the mass passes 85 GeV and that this lowers $z_{97}$.', None),
 ('Method',
  'Samples: 2,000 jets per class plus 10,000 W and 10,000 Z jets, random rows (seed 31), corrected build. '
  'Switch: b2h8\'s class-token share $a_{h0}$ in 5 GeV mass bins, fitted with $s(m)=s_{lo}+\\frac{s_{hi}-s_{lo}}{1+e^{(m-m_{50})/w}}$; span compared over all 16 heads (Holm). '
  'Edits: all angles $\\times k$ (mass $\\times k$) and a rigid $\\phi$ shift of 0.06 rad (mass unchanged). '
  'Route: for a head set $G$, the removal share $R_G = 1-\\overline{\\Delta z_{97}[c_G \\leftarrow c_G^{\\rm unedited}]}/\\overline{\\Delta z_{97}}$ ($R_{\\rm all\\,16}=1$ by construction); exact split $\\Delta z_{97}=\\Delta_{\\rm attn}+\\Delta_{\\rm value}+\\Delta_{\\rm int}$. '
  'Literal circuit: keep only the named heads and neurons; every other class-stage unit and the other 127 class-token coordinates come from one random donor jet; particle blocks intact; compared with 100 random circuits of the same size.', None),
 ('Result',
  'Switch: $m_{50}=87.1$ GeV [86.8, 87.4], width 4.3 GeV, share 0.86 → 0.08; span 82 points against at most 16 for every other head (fresh rerun: 86.6 GeV). The share also drops by 0.17 under the $\\phi$ shift that keeps the mass fixed. '
  'Route (verifier, angles $\\times 1.1$): $R_{\\rm b2h8}=0.046$, $R_{\\rm b1h3}=0.47$, all eight block-1 heads 0.99, block 2 0.06. Attention re-weighting is 0.34 of $\\Delta z_{97}$ and the value change 0.68 (G21b had found attention 0.96). '
  'Circuit: {b2h8, b1h3, n98} gives 50–54 % W/Z accuracy (random circuits 49.7 %, full model 83 %, ParT_full 85.9 % 10-class on the P1 sample). Under the same strict definition all 16 heads + n98 give 63 %; n98 read from an intact class stage gives 72 %.', 'results/part/p3_wz_circuit.png'),
 ('Checks',
  'The re-implemented class stage reproduces the logits to $|\\Delta\\ell|<10^{-5}$. Fresh files and seeds (verifier) reproduce the switch and the shares. P3\'s first route metric (a resample slope between different jets) gave all 16 heads only 0.12–0.15 although they carry the whole input, so it is diluted and was replaced by $R_G$. Particle-block heads 4 and 8 remove 0.26 on W jets against 0.13 for random block-stratified sets ($p=0.016$), but are not separated on Z jets or under the $\\phi$ shift.', None),
 ('Why it matters', 'The switch exists but the decision does not go through it: the mass reaches the W/Z coordinate through the first class-attention block. Thesis step 3 is rewritten (claims C2a, C2b), and literal-circuit scores are reported together with the ceiling the strict definition imposes (63 %).', None),
]

P4_EXPLAINER = [
 ('Question', 'ParT\'s sense of jet mass moves when every particle is shifted relative to the jet axis at fixed invariant mass (C3). Does ParticleNet, a graph network with no attention trained on the same inputs, do the same, and through which input?', None),
 ('Background',
  'JetClass gives each particle its offsets from the jet axis, $\\Delta\\eta_i$, $\\Delta\\phi_i$, and $\\Delta R_i=\\sqrt{\\Delta\\eta_i^2+\\Delta\\phi_i^2}$. A rigid shift $\\Delta\\phi_i\\to\\Delta\\phi_i+\\delta$ leaves $m^2=(\\sum_i p_i)^2$ unchanged but changes the axis moment $M_2=\\sum_i p_{T,i}\\Delta R_i^2$; a network that estimates mass as $\\sqrt{M_0M_2}$ ($M_0=\\sum_i p_{T,i}$) predicts $\\Delta\\log\\hat m=\\tfrac12\\Delta\\log(M_0M_2)\\approx0.073$ at $\\delta=0.06$. '
  'ParticleNet mixes particles by EdgeConv on a $k$-nearest-neighbour graph: $x_i\'=\\mathrm{ReLU}\\big(W_{sc}x_i+\\tfrac1k\\sum_{j\\in\\mathcal N(i)}h(h_\\Phi(x_i)+h_\\Theta(x_j-x_i))\\big)$. The first graph is built on $(\\Delta\\eta,\\Delta\\phi)$, so it and $x_j-x_i$ of the shifted coordinates do not change under a rigid shift.', None),
 ('Method',
  'Official ParticleNet_full and _kin weights; ParT on the same 10,000 jets (1,000 per class; three samples). Edits: $\\delta=0.06$ in $\\phi$, a random direction of the same size, angles $\\times1.1$ (real mass +9.9 %). Metrics: true-W → Z rate, $\\Delta(\\ell_W-\\ell_Z)$, and a ridge read-out of $\\log m$ calibrated against a matched real mass change. '
  'Routes: feed the edit through one input group or one layer-1 term at a time (or keep it unedited), and fix the kNN graphs. Controls: ΔR-keeping rotations $(\\Delta\\eta,\\Delta\\phi)\\to R_\\alpha(\\Delta\\eta,\\Delta\\phi)$ and the mirror $\\Delta\\phi\\to-\\Delta\\phi$; re-runs for float noise.', None),
 ('Result',
  'True W jets called Z: ParticleNet_full 12 → 28 % (fresh rows 12 → 26 %), ParT_full 12 → 31 % (33 %); ParticleNet_kin 18 → 27–29 %, ParT_kin 16 → 36–37 %. The calibrated read-out moves by 0.62–0.71 of the $\\sqrt{M_0M_2}$ prediction (0 and 1 both excluded). Fed alone, the edited $\\Delta R$ gives 0.94–1.41 of the response; fed unedited it leaves at most 0.06. Fixing all graphs keeps 0.96–1.01. In layer 1 the per-particle shortcut $W_{sc}x_i$ alone gives 0.85–1.05, the centre term 0.15–0.30, the edge term $x_j-x_i$ at most 0.10.', 'results/particlenet/p4_wz_shift.png'),
 ('Where it enters', 'Share of the response carried by each input group and layer-1 term.', 'results/particlenet/p4_routes.png'),
 ('Checks',
  'The split forward pass equals weaver\'s own forward exactly; re-running unchanged inputs changes no call; accuracy 84.6 % matches the released 84.4 %. A rotation by 0.3 rad that keeps every $\\Delta R_i$ changes W → Z by −0.5 to +0.8 points in all four models (Holm $p=1$). The mirror needs charge and $d_0$ flipped with $\\Delta\\phi$ in the full models (then +1.7 / +1.9 points). The layer-1 graph is unchanged by construction; the measured graph result is for layers 2–3.', None),
 ('Why it matters', 'The non-invariant, axis-relative mass response is not specific to attention: a graph network on the same inputs shows about two-thirds (full) to one-half (kin) of ParT\'s W → Z increase through the same input (claim C5). Whether the input format causes it, or both networks learned a similar solution, needs a model without axis-relative inputs (L-GATr, C6).', None),
]

P5_EXPLAINER = [
 ('Question', 'Can a network built to respect the symmetries of special relativity (L-GATr) serve as the contrast model for the axis-relative mass response (C3, C5)?', None),
 ('Background',
  'L-GATr represents each particle\'s four-momentum $p_i=(E,p_x,p_y,p_z)$ as a multivector and uses only Lorentz-equivariant layers; reference tokens (the beam directions and the time direction $(1,0,0,0)$) reduce the symmetry to rotations about the beam. A rigid rotation $\\phi_i\\to\\phi_i+\\delta$ is then an exact symmetry, $f(R_\\delta p)=f(p)$, and leaves $m^2$ unchanged; with the stored axis held fixed it is the same edit that raises $M_2$ for ParT.', None),
 ('Method',
  'No published JetClass weights (checked 2026-10-19), so L-GATr is trained with the authors\' code and settings (12 blocks, 1.08 M parameters, 200k iterations) on 2M jets from randomly drawn files. On an early checkpoint, 10,000 test jets: a $\\phi$ rotation of 0.06, an $\\eta$ shift and a longitudinal boost of 0.06, and a real mass change (angles $\\times1.1$, $\\Delta\\log m=+0.094$); noise floor from reordering jets within batches.', None),
 ('Result',
  'Checkpoint at 2.5 % of training (accuracy 79.2 %): the rotation changes 0.01 % of calls (ParT: true-W → Z 12 → 31 %); $\\eta$ shift and boost 0.5 %; the real mass change moves W → Z from 17.9 to 42.6 %.', None),
 ('Checks', 'Random-init network: rotation 0.00 %; accuracy on this sample equals the run\'s own validation accuracy (79.2 %).', None),
 ('Why it matters', 'The rotation result holds by construction, so it does not show how L-GATr estimates mass. The pre-registered test on the final checkpoint uses edits that keep the invariant mass but change $M_2$ and are not symmetries: an invariant-mass reader predicts a read-out ratio near 0, an axis-moment reader 0.7–1.4 as in ParT.', None),
]

P8_EXPLAINER = [
 ('Question', 'ParT\'s W/Z decision reacts to a $\\phi$ shift that leaves the invariant mass unchanged (C3), and the reaction enters through each particle\'s distance $\\Delta R_i$ to the jet axis. Where between that input and the class token is the mass-like quantity formed, and which quantity is it?', None),
 ('Background',
  'ParT has 8 particle blocks; each adds an attention output and an MLP output to every particle\'s 128-dimensional residual vector $h_p$. Two class-attention blocks then read the particles into a class token whose coordinate $z_{97}$ ("n98") separates W from Z. '
  'Candidate quantities, with $S_0=\\sum_i p_{T,i}$ and $S_2=\\sum_i p_{T,i}\\Delta R_i^2$: the invariant mass $m$; the normalised axis moment $M_2=S_2/S_0$; the axis-moment mass $\\mathrm{mom}=\\sqrt{S_0 S_2}$, which equals $m$ for natural jets about the $p_T$ centroid at small angles. '
  'A rigid shift $\\Delta\\phi_i\\to\\Delta\\phi_i+0.06$ raises $M_2$ and mom at fixed $m$; a $p_T$ rescaling by 1.25 raises $m$ and mom together at fixed $M_2$.', None),
 ('Method',
  '(1) Patching: run the shifted jet, put back the unshifted output of one sublayer (or of all sublayers in blocks $1..l$), and measure the restored share $1-\\overline{\\Delta z_{97}^{\\rm patched}}/\\overline{\\Delta z_{97}^{\\rm edit}}$; control: the same patch from a different jet of the same class and mass bin. '
  '(2) Distributed alignment search: learn a unit vector $q$ such that writing the shifted jet\'s value of $q^\\top\\bar h$ ($\\bar h$ = jet mean of $h_p$) into the unshifted jet, $h_p\\leftarrow h_p+(\\bar h\'-\\bar h)qq^\\top$, reproduces the shifted logits; compare with random $q$ and an untrained network. '
  '(3) Substitution: copy $q^\\top\\bar h$ from a real donor jet and compare the slope $\\mathrm d\\Delta z_{97}/\\mathrm d\\Delta\\log X$ with the shift\'s own slope, for $X=M_2$ (P8: donors of equal $m$) and $X=$ mom (verifier: donors that differ in mom). '
  'Samples: 10,000 W + 10,000 Z random test jets (seed 2026); verifier fresh files, 5,000 + 5,000 (seed 7). Unedited ParT_full 85.93 % on the P1 sample.', None),
 ('Result',
  'Under $\\phi+0.06$ (true-W called Z 10 → 31 %), no single sublayer restores more than 0.06 of $\\Delta z_{97}$, and these single shares do not exceed the donor control. All 16 sublayers together restore 0.42 [0.41, 0.43]; the other 0.58 reaches the class stage through the residual path of the shifted embedding. '
  'One direction $q$ reproduces 0.58–0.69 of the logit change at every depth from the embedding to block 8 (random directions $\\le 0.06$). '
  'With donors of equal $m$ (which also have nearly equal mom, $|\\Delta\\log\\mathrm{mom}|=0.022$) substitution gives no response (ratio 0.004). With donors that differ in mom, $q^\\top\\bar h$ follows $\\Delta\\log\\mathrm{mom}$ ($r=0.87$) and substitution moves $z_{97}$ with the shift\'s sign at 0.42 [0.38, 0.46] of its slope; random directions give about 0. Across natural jets the projection depends on $\\log S_0$ and $\\log(S_2/S_0)$ in the ratio $\\beta_{S_0}/(2\\beta_{M_2})=1.02$ [1.00, 1.05] at block 1 (mom predicts 1, $M_2$ predicts 0).', 'results/part/p8_mass_formation.png'),
 ('Checks',
  'Patching and alignment numbers replicate within 0.03 on fresh files; ParT_kin shows the same pattern (all sublayers 0.38, direction 0.69–0.81). Hooks, padding masks, random rows and the corrected $\\eta$ build were reviewed. The direction is not an exact copy of $\\log$ mom: per unit $\\Delta\\log$ mom a $p_T$ rescaling moves it 0.71 as much as the $\\phi$ shift at block 1 (n98 itself 0.74). Under the shift, block-1 class heads carry 0.99 of $\\Delta z_{97}$ (b1h3 0.53).', None),
 ('Why it matters', 'ParT does not form its axis-relative mass at a particular block: a direction that follows the axis-moment mass is already present in the input embedding and is read by the first class-attention block (claims C7, C2b). Nonlinear and per-particle forms of the quantity are not tested yet.', None),
]

P9_EXPLAINER = [
 ('Question', 'Is there a small circuit inside ParT_full that reproduces its W-versus-Z decision when everything else is replaced by values from another jet?', None),
 ('Background',
  'User definition: a circuit consists literally of its kept units; every other unit is resample-ablated. P3 found that {b2h8, b1h3, n98} kept alone separates W from Z at about 53 % (model 83 %), searching only the class-attention stage. Here the whole network is searched.', None),
 ('Method',
  'Write ParT_full as a graph of 433 nodes (embedding, 64 pair biases, 64 particle heads, 128 particle-MLP groups of 32 units, 16 class heads, 32 class-MLP groups, 128 class-token coordinates) and 21,504 edges, each with a mask $m_e$. A masked edge carries the value from a jet of the other class with matched $p_T$. '
  'Faithfulness $F=\\frac{\\sum_j s_j(d^{\\rm circ}_j-d^{\\rm corr}_j)}{\\sum_j s_j(d_j-d^{\\rm corr}_j)}$ with $d=\\ell_W-\\ell_Z$, $s_j=\\pm1$ by class; accuracy = W/Z calls correct. '
  'Rank edges by integrated gradients over the masks, $\\mathrm{score}_e=\\int_0^1 \\partial D/\\partial m_e\\,\\mathrm d\\alpha$ (5 steps), then prune greedily (ACDC). Verifier: train a soft top-$K$ mask $m_e=\\sigma((\\theta_e-\\tau_K)/T)$ to maximise $F$ on separate files. 10,000 discovery and 10,000 evaluation jets (random rows, seeds 2026/2027).', None),
 ('Result',
  'Ranked circuits: 5 % of edges give $F=0.51$ and W/Z 55 %; $F$ reaches 0.8 near 29 % of edges. Literal ACDC circuits of 10, 27 and 89 nodes give 29, 36 and 59 % W/Z (random circuits 17 %; model 83 %). A trained mask with 10 % of edges reaches $F=0.98$ but only 63 % W/Z, the same as the ranked circuit of that size: $F$ is a sum over jets and grows by enlarging $d$ on jets already called correctly.', 'results/part/p9_eap_wz.png'),
 ('Checks',
  'All masks on reproduces the model to $4\\times10^{-6}$; integrated-gradient completeness 1.02; attributions agree in sign with single-edge resampling for 50/50 top edges and are stable on fresh files (Spearman 0.91). With the read-out kept (embedding, class stage, class-token coordinates), random particle-stage nodes score $F=0.38$ against 0.56 for the 89-node circuit and 0.25 for the read-out alone. The P3 circuits reproduce with uniform donors (52 %, 64 %).', 'results/part/p9_verify2.png'),
 ('Why it matters', 'Under the literal definition, no circuit found with up to 10 % of the connections reproduces ParT_full\'s W/Z calls; most of what the found circuits add over random sits at the read-out (claim C8). Paper C reports accuracy and faithfulness against circuit size rather than one named circuit. A different search, coarser units or another corruption could change the size.', None),
]

P10_EXPLAINER = [
 ('Question', 'Is ParT\'s axis-relative mass response a property of the one released checkpoint (trained on 100M jets), or does every ParT trained on JetClass learn it, also at L-GATr\'s 2M-jet scale?', None),
 ('Background', 'Rotating all particles by 0.06 rad about the stored jet axis leaves the invariant mass $m$ unchanged but raises the axis moment $M_2=\\sum_i p_{T,i}\\Delta R_i^2/\\sum_i p_{T,i}$; the released ParT then calls more true W jets Z (C3).', None),
 ('Method',
  'Train ParT from scratch with the authors\' script on the 2M-jet L-GATr subset, seeds 0/1/2, full and kinematic inputs (samples per epoch 1/10 of the authors\'). On 10,000 random test jets, measure the W → Z change under the shift, under a real 10 % mass change (angles $\\times1.1$) and under a whole-jet rotation (null), and the ratio $R=\\overline{\\Delta(\\ell_W-\\ell_Z)}_{\\phi}/\\overline{\\Delta(\\ell_W-\\ell_Z)}_{\\times1.1}$, compared with the axis-moment prediction 0.97 and the invariant-mass prediction 0.', None),
 ('Result',
  'Released ParT_full: +22.0 points, $R=0.76$; ParT_kin +19.0, $R=0.69$. At epoch 3 of 50: full-input seed 0 +22.6 [19.7, 25.4], $R=0.94$ [0.89, 0.99]; kinematic seeds +4 to +9, $R=0.17$–0.35.', 'results/part/p10_battery_e3.png'),
 ('Checks', 'The released models reproduce their accuracies (86.0, 74.2 %) and P1 numbers on these rows; the whole-jet rotation stays within 0.5 points; an E-scheme re-axis changes at most 0.3 % of calls.', None),
 ('Why it matters', 'The final checkpoints (about 2026-10-20) test the pre-registered statements S1–S4 per seed and give a ParT matched to L-GATr in data size (claim C3s). The epoch-3 numbers are interim.', None),
]

P13_EXPLAINER = [
 ('Question', 'P9 found no small circuit of attention heads and MLP neuron groups that reproduces ParT_full\'s W-versus-Z calls. Heads and neurons may be the wrong units. With learned sparse features as units, is there a small feature circuit, and does a single feature carry the axis-moment mass $m_A$?', None),
 ('Background',
  'A transcoder replaces an MLP by a wide, sparse layer trained to reproduce its output: $\\hat y = W_d\\,\\mathrm{TopK}(W_e x + b_e) + W_s x + b$, where $x$ is the MLP input, at most $k$ latents ("features") are non-zero per token, and $W_s x$ is a linear bypass ("skip map"). A sparse autoencoder (SAE) does the same for the residual stream itself ($\\hat x = W_d\\,\\mathrm{TopK}(W_e x)$). '
  'The difference between the real activation and its reconstruction is kept as an explicit error term, so the network with dictionaries spliced in is exact. '
  'The axis-moment mass is $m_A=\\sqrt{\\sum_i p_{T,i}\\cdot\\sum_i p_{T,i}\\Delta R_i^2}$; a rigid shift $\\phi\\to\\phi+0.06$ of all particles raises it at fixed invariant mass.', None),
 ('Method',
  'Dictionaries on 1,000,000 random training jets: transcoders ($k=32$, 1,024 latents) on all 8 particle-block MLPs and 2 class-block MLPs; SAEs (2,048 latents, $k=32$) on the class-token residual after each class block. '
  'Attribution: integrated gradients of $D=\\sum_j s_j(\\ell_W-\\ell_Z)_j$ ($s_j=\\pm1$ by class) over masks on every feature and error term, interpolating from a W↔Z donor jet with matched $p_T$ to the clean jet. '
  'Circuit of size $K$: keep the top-$K$ features; every other feature and every error term takes the donor value. Main variant: attention layers and skip maps computed on the real jet; strict variant: those also take the donor value. '
  'Controls: 100 random equal-size feature sets; P9 head/MLP circuits. Steering: move the feature most correlated with $\\log m_A$ by the amount a 5 % $m_A$ change produces in natural jets, and set it to its mean under the $\\phi$ shift. '
  'Samples: N = 20,000 (2,000/class) for 10-class accuracy, 3,000 W/Z jets for attribution, 10,000 for evaluation (random rows, seed 13); verifier on four fresh files.', None),
 ('Result',
  'Splicing all dictionaries changes 10-class accuracy by 0.03 pt (86.16 → 86.13 %). Kept alone, 50 features give 50 % W/Z accuracy (model 84 %), 1,000 give 74 %, 2,000 give 79 %; random sets give 16–22 %. In the strict variant accuracy stays below 43 % even with every feature kept. '
  'The feature most correlated with $\\log m_A$ ($r=-0.76$) does not change the calls when steered at the calibrated step or set to its mean: W → Z flow $-0.02$ [$-0.10$, $+0.06$] points against $+21.1$ for the $\\phi$ shift.', 'results/part/p13_feature_circuit.png'),
 ('Checks',
  'Fresh files reproduce the curve (50 features 51.4 % [50.4, 52.4]; top-50 overlap 0.96). The attention and skip paths alone, with all last-layer residual features kept and everything upstream from the donor, give 17 % (= donor). '
  'Steering all features of one class-token residual SAE together along their natural $m_A$ slopes reproduces the natural slope of the read-out ($dZW$ 0.15–0.16 vs prediction 0.152; permuted control ≈ 0), which is 14 % of the $\\phi$ shift\'s logit change (1.08). '
  'At 4–8 s.d. the single feature moves calls toward W ($-8$ % at 8 s.d.). The figure\'s x-axis counts only the kept features\' parameters, not the live attention and skip parameters.', None),
 ('Why it matters',
  'Under the literal circuit definition, changing units from heads and neurons to learned features does not produce a small W/Z circuit (claim C8 extended). The axis-moment-mass information is spread over many features, and natural $m_A$ variation accounts for only about a seventh of the $\\phi$-shift response, which leaves open what else the shift changes (claim C7). Trained rather than ranked feature circuits are untested.', None),
]

P14_EXPLAINER = [
 ('Question', 'Do our tools find a mass variable and route when the answer is planted, and does a ParT trained on an exact invariant-mass label still respond to a shift that keeps the invariant mass fixed?', None),
 ('Background',
  'The released ParT responds to a rigid $\\phi$ shift at fixed invariant mass $m$ (claim C3). Two explanations predict different things for a network trained on a known label: if the response is learned from the 10-class data, a network trained to find $|m-85|<10$ GeV should ignore the shift; if ParT\'s axis-relative per-particle inputs bias it toward the axis-moment mass $m_A=\\sqrt{\\sum p_T\\cdot\\sum p_T\\Delta R^2}$, it should respond anyway.', None),
 ('Method',
  'ParT (kinematic inputs, 2 outputs, authors\' optimiser, 10 epochs) trained on 1.5M W/Z/QCD jets with four labels: (a) $\\mathbb 1[|m-85|<10]$; (b) $\\mathbb 1[|m_A-84.6|<10]$ with the same positive rate; (c) as (a) but with the per-particle position inputs zeroed, so geometry enters only through the pair features; (s) shuffled (a). '
  'Test: $R=$ per-jet slope of $\\Delta(\\ell_W-\\ell_Z)$ under $\\phi+0.06$ against a matched real dilation with the same $\\Delta\\log m_A$. Predictions: learned choice, $R_a<0.2$ and $R_b\\approx1$; architecture prior, $R_a\\ge0.5$. '
  'Control this round: the released ParT_full and ParT_kin on 20,000 random held-out W/Z/QCD jets (seed 14).', None),
 ('Result',
  'Released control: $R=0.79$ [0.78, 0.80] (full) and 0.72 [0.71, 0.73] (kin), rotation null ≈ 0; a matched real mass change passes 83 % through the per-particle inputs and 15–17 % through the pair features; under the $\\phi$ shift the pair-only route moves the read-out no more than the rotation null. '
  'Training: arm c reaches 97 % held-out label accuracy by epoch 2. The arm analysis has not run yet.', 'results/part/p14_planted_released.png'),
 ('Checks', 'The released models reproduce their P1-sample accuracies (85.93, 74.47 %). Labels a and b agree on 97.5 % of jets (correlation of $m$ and $m_A$ 0.999), so only the $\\phi$ edit separates the two arms. Pair features are not exactly rotation invariant in float32 (0.03–0.8 % of calls), a noise floor for arm c.', None),
 ('Why it matters', 'Arm c shows ParT can compute the invariant-mass window from pair features alone. Arm a decides whether the axis response is driven by the training data or by ParT\'s inputs and architecture (claim C10, pre-registered); the analysis runs next round.', None),
]

P15_EXPLAINER = [
 ('Question', 'ParticleNet shares ParT\'s response to a mass-fixed $\\phi$ shift (claim C5). Inside ParticleNet, is there one direction that carries the response and follows the axis-moment mass, do the rebuilt neighbour graphs carry it, does a small set of channels reproduce the calls, and how do standard attribution and probe baselines compare?', None),
 ('Background',
  'ParticleNet applies three EdgeConv blocks (ec1–ec3): each particle is updated from its $k$ nearest neighbours, with the neighbour graph rebuilt from the learned features in blocks 2 and 3; then global pooling and a dense head (fc). '
  'Distributed alignment search (DAS) fits one direction $u$ in a layer and swaps the jet\'s component along $u$ with that of the shifted jet; the share is the fraction of the shift\'s change in $d=\\ell_W-\\ell_Z$ that the swap reproduces. An axis-moment-mass ($m_A$) direction would respond equally, per unit $\\Delta\\log m_A$, to a $\\phi$ shift, to scaling all angles and to scaling all $p_T$.', None),
 ('Method',
  'Official ParticleNet_full and ParticleNet_kin; 20,000 random validation jets (2,000/class, seed 15). One-direction DAS at ec1, ec2, ec3 (pooled) and fc; controls: random directions, the same search on a weight-shuffled network. kNN edges changed and graphs frozen per layer. Channel circuits: top-$K$ pooled channels by integrated gradients kept, others taken from a $p_T$-matched opposite-class jet. LRP-ε relevance; linear probes of $d$ from 35 energy-flow polynomials (EFPs) and primary Lund-plane bins. '
  'Verifier: fresh file (seed 45); the mirror shift $\\phi-0.06$, which changes $d$ by the same amount; directions fitted on both signs; atlas with standardised rotations.', None),
 ('Result',
  'Accuracy 84.5 % (full), 70.1 % (kin). The ec1 direction reproduces 1.16 of the $+0.06$ response but $-0.29$ of the $-0.06$ response: it records the sign of the shift, which the output ignores. The ec2 direction reproduces 0.75–0.80 of each sign. Weight-shuffled networks reach similar one-direction shares. Per unit $\\Delta\\log m_A$, angle scaling moves the ec1–ec2 directions 0.11–0.39 as much as the shift (an $m_A$ copy predicts 1). '
  'Freezing the rebuilt graphs removes 0.7–4.0 % of the response. 32 pooled channels kept alone give 69–71 % W/Z accuracy against the model\'s 71–72 %, but 20–45 % 10-class accuracy against 83–85 %.', 'results/particlenet/p15_das_layers.png'),
 ('Checks',
  'Hooked forward equals weaver\'s ($|\\Delta L|=0$); edits keep $|\\Delta\\log m|\\le1.3\\times10^{-7}$. The post-pooling share varies by sample (0.11–0.57). With centred rotations channels align less with single observables than random directions, with standardised rotations more, so no basis statement is made. EFP and Lund probes cannot respond to a $\\phi$ shift by construction; gradient and LRP predict the per-jet response at $r$ 0.31–0.45.', 'results/particlenet/p15_circuits.png'),
 ('Why it matters', 'ParticleNet\'s response to the shift does not pass through its rebuilt graphs (claim C5) and is not carried by a trained axis-moment-mass direction; a one-direction match on its own does not show a trained variable, since untrained networks give one too (claim C9; the same control is proposed for ParT\'s C7). Small channel circuits reproduce the binary W/Z decision but not the 10-class decision.', None),
]

D4_EXPLAINER = [
 ('Question', 'Does the networks\' way of computing the mass make them depend on how the jet axis is defined?', None),
 ('Background',
  'The jet axis is a convention. The standard axis (E-scheme) is the summed momentum direction. The winner-take-all (WTA) axis follows the harder branch when the jet is reclustered; '
  'it is insensitive to soft radiation and used for some substructure measurements. A trigger may use a seed particle. The invariant mass is the same for every choice; '
  '$\\sqrt{M_0 M_2}$ about the axis is not.', 'results/site/X2_axis.png'),
 ('Why the axis choice changes the networks\' mass: a two-prong jet',
  'A W or Z jet has two prongs carrying momentum fractions $z$ and $1-z$, separated by an angle $\\theta$. Its invariant mass is $m^2 \\approx z(1-z)\\,p_T^2\\,\\theta^2$. '
  'About the centre the networks\' variable equals the mass: $M_0 M_2 = m^2$. About an axis placed on one prong (fraction $z$), that prong sits at $r = 0$ and the other at $r = \\theta$, so '
  '$M_0 M_2 = (1-z)\\,p_T^2\\theta^2 = m^2/z$ and the networks\' mass becomes $m/\\sqrt{z}$. The WTA axis sits on the harder prong ($z \\ge 0.5$), so the reading grows by a factor between 1 and $\\sqrt2 = 1.41$: '
  'an 80 GeV W jet reads 80–113 GeV, which crosses the Z window. The hardest single particle can sit in the softer prong ($z < 0.5$), which gives larger factors. For the example jet: $106/81.3 = 1.30$ ($z = 0.59$, WTA on the harder prong) '
  'and $125/81.3 = 1.53$ ($z = 0.43$, hardest particle in the softer prong). One-prong jets (light quarks, gluons) have their WTA axis close to the centre (median 0.030 rad for QCD jets on JetClass), so their reading changes little; '
  'this is why the changed decisions are mostly W → Z and Z → q.', None),
 ('Three-prong jets (top): the reading rises, the call mostly stays (E19)',
  'The same algebra applies to a top jet with three prongs: about an axis on one prong, $M_0 M_2$ includes the full squared distances to the other two prongs. For a symmetric three-prong jet '
  '$m^2 = \\tfrac13 p_T^2\\theta^2$ while $M_0 M_2$ about one prong is $\\tfrac23 p_T^2\\theta^2$, a factor $\\sqrt2$ in the reading; on real top jets with three resolved subjets the factor has median 1.18. '
  'Measured (E19, WTA axis, fresh jets): JEDI-linear\'s mass reading for top jets rises from a median 146 to 175 GeV; the calibrated ratio to the $\\sqrt{M_0 M_2}$ prediction is 1.02 [0.98, 1.06] (p for "no change" $10^{-296}$). '
  'Yet only 12.2 % [11.2, 13.2] of top calls change and top accuracy goes 84.0 → 83.2 % (McNemar p = 0.095, not significant). '
  'The reason is in the three class neurons that push the top score: z10 (+), z8 (+), z13 (−, counts against top). On unedited jets z13 switches off above about 143–170 GeV and z8 is on from about 110 GeV, both one-sided; '
  'z10 is on between about 100 and 220 GeV and off above, two-sided. A heavier reading turns z13 off (on-fraction 0.52 → 0.28), keeps z8 on (0.95 → 0.98) and turns some z10 off (0.97 → 0.66): the top score drops by 1.21 but the margin over the next class stays positive (2.33 → 1.79). '
  'For W and Z the windows are narrow and next to each other, so the same kind of increase crosses into the neighbouring class (68 % and 45 % of calls change). '
  'ParT_full keeps its top calls (95.5 → 92.1 %); ParT_kin loses 42 % of them, mostly to t → bℓν, while its top mass reading rises only 0.45 of the prediction, so that change does not follow the mass reading. '
  'Caveat: ParT_full\'s read-out fits top jets poorly (per-class $R^2$ 0.37).', 'results/site/F9_top_axis.png'),
 ('Per-class numbers', 'Read-out change vs prediction, calibrated ratio, calls changed and accuracy per class for the three models (E19).', 'results/site/T16_axis_by_class.png'),
 ('Does training with an off-centre axis fix it? (E18, X4)',
  'Deep Sets and the Linformer were retrained with every particle measured from the hardest particle, so the axis is never at the centre during training (accuracy 80.15 % and 80.90 %, close to the originals). '
  'When the axis is moved to the other prong of a W or Z jet (the invariant mass changes by 0.3 %), these models\' mass reading moves only 0.13–0.41 of a real mass change of the predicted size, and their calls change in 6–9 % of jets, against 58–64 % for a real mass change of that size. '
  'They are neither pure invariant-mass nor pure axis-moment computers (both rejected, p ≤ $10^{-27}$). The 0.13–0.41 figure holds for this axis-jump edit only (X7v): evaluated on stored-axis inputs, the same models move Z → W as $\\tfrac12\\Delta\\log(M_0 M_2)$ about their own axis predicts, with a read-out change of 0.16–0.62 of a matched real change, and a translation with that quantity held fixed changes 23–31 % of their calls, so they also read a spread that does not depend on the axis position. '
  'A residual remains: on natural W jets whose hardest particle sits in a soft prong ($z < 0.3$), they call Z 7–22 % of the time, against 1–4 % for the original models.', 'results/site/F11_axis_jump.png'),
 ('Where these axes are used',
  'The standard (E-scheme) axis is the direction of the summed four-momentum and is how anti-$k_T$ jets are usually defined. The winner-take-all recombination scheme was proposed so that the axis is not moved by soft radiation '
  '(recoil-free); it is used, for example, for the axes of N-subjettiness, and the angle between the standard and WTA axes has itself been studied as an observable of soft radiation. Trigger-level jet finding can build jets around a seed particle. '
  'A tagger trained with one convention and given inputs in another sees its mass reading change by the factor above.', None),
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
 ('Why it matters', 'A tagger deployed with a different axis definition than in training changes its decisions by an amount the mass formula predicts; for JEDI-linear a WTA or seed-particle axis changes about 30 % of decisions. Which axis the deployed trigger inputs use is an open question for the user. Models trained on inputs about the hardest-particle axis, where the axis is never at the centre, are described above (E18, X4, X7v).', None),
]

EARLIER_C = [
 ('C1: ParT\'s attention is mostly a fixed formula of pair geometry',
  'Each attention score in ParT is a learned content term plus a "pair bias" computed from the pair features. The pair bias alone explains a median 79 % of the variance of the scores. A fitted formula of four pair features '
  'replaces the pair bias with 85.0 % accuracy against 85.4 %. Attention from the pair bias alone keeps 69.4 %, from the content term alone 40.2 %.', None),
 ('C2 (v1, 2026-10-16): the mass enters ParT through the attention',
  'Under a 10 % angle scaling (a 10 % mass increase) the W/Z neuron changes by −0.478 of its spread; −0.458 of that comes from changes in where the heads attend, −0.019 from the per-particle features. '
  'One head (b2h8) attends to the class token 87 % of the time at 60–65 GeV and 7 % above 105 GeV, switching between the W and Z masses. '
  'Re-measured in P3/P3v (2026-10-19, 10× statistics, corrected build): the switch reproduces (87 GeV; claim C2a), but the attention re-weighting is 0.34 of the neuron change and the value change 0.68, and b2h8 carries 3–5 % of it (claim C2b).', None),
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
  'their best epoch is the last one, so a transformer was also trained for 100 epochs (tf-2b-d32: 80.08 %; it is among the X2 models). At 32 particles the ranking is the same (Linformer 78.37 %, Deep Sets 77.79 %, JEDI 78.00 %).',
  'results/site/T7_small_models.png'),
 ('Result of the same tests (X2)',
  'Deep Sets has the exact cut-and-count identity, but its first layer is not sparse: 63–64 of 64 units matter and most mix pT and angle, while JEDI-linear has 13 cuts that are either pT-only or angle-only. '
  'JEDI-linear\'s clean first layer is therefore a property of JEDI-linear (possibly of its fixed-point training, which prunes inputs), not of sum-pooling networks in general. '
  'Every model\'s mass sense is axis-relative, but how strongly it follows $\\sqrt{M_0 M_2}$ varies from 0.07 to 2.0 of the prediction between architectures and even between two seeds of the same model. '
  'Variables appear over the layers in order of complexity in every model (Spearman 0.85–0.90), partly because sums over particles are readable from the sum layers even without training.', 'results/site/T8_small_mass_shift.png'),
 ('Why it matters',
  'Two architectures with JEDI-linear\'s accuracy are available: one built on sums over particles (Deep Sets) and one on attention over pT-rank mixtures (Linformer). Running the same tests on them '
  '(first-layer cuts and objects, the mass shift test, variables over layers) shows which findings belong to JEDI-linear and which to models of this kind in general. All runs saved a checkpoint after every epoch, '
  'so the formation of each unit during training can be followed.', None),
]

ATLAS_EXPLAINER = [
 ('Question', 'Following Cammarata (Goodfire, "Neural Networks Are Already Clean"): does each unit of JEDI-linear have one main job that can be stated as a physical pattern?', None),
 ('Method',
  'For each of the 144 live units (20,000 test jets): the distribution of its value per class; average jet images (pT-weighted, in the network\'s frame and rotated to the jet\'s principal axis) for the jets where it is strongest (top 1 %), '
  'moderately on (90–99 %), weakly on (50–90 %), and for class neurons the most negative 1 %; the median of 10 physical quantities in each of these bins; the best one- and two-quantity description out of 539 candidates (held-out $R^2$); '
  'a clustering of the jets where it is on (main job vs other jobs); which earlier units feed it and which class scores it moves; and steering: set the unit to its 99th percentile on jets where it is off, compared with doing the same to a random unit of the same layer.', None),
 ('Example: z0, the W neuron', 'Strongest on jets with $\\sum p_T r^2$ around 6.2 GeV·rad² (≈ 80 GeV of mass): 97 % of its top 1 % are W jets, two prongs, jet mass 79 GeV. '
  'Its negative side is top jets (82 % of its lowest 1 %; jet mass 170 GeV, three prongs). It pushes the W score up (+0.75 per spread) and the Z score down (−1.13). Steering it on turns Z calls into W calls.', 'results/site/C_card_z0.png'),
 ('Example: S29, the largest second-round object', 'Described by $\\sum p_T r^2$ over pT ranks 8–63 above 8.8 GeV·rad² ($R^2$ 0.94). Its two clusters (lower and higher mass) are described by the same quantity, so it counts as "related jobs". Steering it on sends 98.6 % of the affected jets to top.', 'results/site/C_card_S29.png'),
 ('Tally',
  'One main job: 31 units (21.5 %); related jobs (the same quantity over two jet populations, mostly lower and higher mass): 82 (56.9 %); unrelated jobs: 24 (16.7 %); no description with $R^2 \\ge 0.5$: 7 (4.9 %). '
  'Weighted by how much each unit affects the decisions: 13.6 / 72.5 / 12.5 / 1.4 %. So 78 % of units (86 % by decision weight) have one quantity that describes them; unrelated or unexplained units are concentrated in h4, h5 and z. '
  'For comparison, Goodfire reports about 90 % one main job, 7 % two related jobs and 2–4 % unexplained for language-model neurons, with different definitions.', 'results/site/T17_atlas_tally.png'),
 ('Limits', 'Steering is not specific except for class neurons: setting a unit to its 99th percentile changes a median 9.9 % of calls, the same as for a random unit (9.8 %). The categories depend on thresholds (R² 0.5, 80 %, ρ 0.7), stated in the table.', None),
 ('Corrections to earlier claims',
  'The first layer has 13 live cuts: 4 pT ramps and 9 half-planes (an earlier count of 6 pT ramps included two units that are constant on data). '
  'z10, previously called the "top gate", is on in 74 % of jets and is best described by $\\sum p_T r$ > 38.9 GeV·rad ($R^2$ 0.70); its strongest 1 % are 95 % top jets at a median 160 GeV. '
  'z9 is on for $\\sqrt{M_0 M_2}$ < 77.9 GeV or 120–163 GeV, with its strongest jets mostly g and q; an earlier one-term fit ("$\\sqrt{M_0 M_2}$ > 143 GeV") was too simple.', 'results/site/C_card_z10.png'),
 ('Why it matters', 'Most units of a trained trigger network can be described by one physical quantity, which supports writing the network as named formulas; the units that cannot are listed and are where the formula program loses accuracy.', None),
]

E8_EXPLAINER = [
 ('Question', 'Is each class a combination of conditions on the 16 class neurons $z$, and does putting the neurons into that combination make the network choose the class?', None),
 ('Background',
  'The class scores are one linear layer on the class neurons: $\\mathrm{score}_c = \\sum_k W_{7,ck}\\,\\mathrm{ReLU}(z_k) + b_{7,c}$, and the network calls the class with the largest score. '
  'A "window" is a condition such as $z_0 \\ge 1.35$; a rule is two or three windows joined by AND. The neuron $z_0$ is the W-mass neuron of the mass result (on for jets of 78–81 GeV inside this window), $z_7$ the Z-mass neuron (90–92 GeV).', None),
 ('Method',
  'Step 1 (description): on 20,000 jets a decision tree of depth 3 predicts "the network calls class $c$" from the 16 values of $z$; the best leaf is the rule. Precision = share of jets inside the rule that the network calls $c$; coverage = share of the network\'s $c$ calls inside the rule; both on 10,000 other jets. '
  'Step 2 (intervention): on jets the network calls something else, set only the rule\'s neurons to the median values of jets called $c$ inside the rule and let the network\'s last layer decide; count how many are now called $c$. '
  'Step 3 (variants): move the neurons only just inside the windows; move them part of the way (dose); move other sets of three neurons to the same kind of values (all 560 triples).', None),
 ('Result',
  'The rules describe the calls: precision 96–98 % and coverage 70–82 % for all five classes, on fresh jets as well. Setting the rule\'s neurons to typical class values moves 54 % (g), 32 % (q), 41 % (W), 61 % (Z) and 26 % (t) of other-called jets into the class; setting all 16 neurons moves 100 %. '
  'Moving the neurons only just inside the windows moves 3–25 %, although 96–98 % of ordinary jets inside the windows are called the class. The change in the class scores from moving three neurons equals the sum of the changes from moving each alone, to numerical precision.', 'results/site/T18_e8_conjunctions.png'),
 ('Checks (E8v, fresh jets)',
  'Among all 560 triples of neurons set to values of the same kind, the rule\'s triple ranks 6th (g, Z), 10th (W), 39th (q) and 78th (t); the best triple for t, $\\{z_8, z_{10}, z_{13}\\}$, moves 77 % and contains $z_{10}$, the top neuron of the mass result, which the tree did not pick. '
  'Triples without any rule neuron move 4–8 % on average. Rules of similar precision (87–99 %) are found after randomly rotating the 16 neuron axes, so a rule of this kind is not specific to individual neurons. '
  'The random-neuron comparison in the original run used different values from the rule (class medians instead of in-rule medians), and the untrained network calls every jet top, so neither control tested the rule; the exhaustive triple comparison replaces them.', 'results/site/F7_e8_conjunctions.png'),
 ('What this shows',
  'The class decision can be summarised by window rules on the class neurons, which name the mass windows for W and Z. The causal effect is set by how far the neurons move along the additive score difference, not by whether a window condition is met: the network adds the neurons\' contributions, and a rule is an approximate description of where that sum favours a class.', None),
 ('Why it matters',
  'This is the last stage of the thesis (classes as combinations of neuron statements). It changes the claim from "a logical AND of windows" to "a sum of neuron contributions, summarised by window rules", which is the form the paper can state with the network\'s own last layer.', None),
]

E14_EXPLAINER = [
 ('Question', 'The first layer only has directional cuts (half-planes through the axis). How does the second round turn them into objects that depend almost only on the distance $r$ from the axis, and does this explain the direction-dependent part of the mass response?', None),
 ('Background',
  'Each first-layer angular unit is $a_j = \\mathrm{ReLU}(\\vec n_j\\cdot(\\Delta\\eta, \\Delta\\phi) + \\beta_j)$: on for particles on one side of a line through (or within 0.01 rad of) the axis. There are nine, with normals at 0°, 51°, 131°, 144°, 161°, 204°, 248°, 270° and 290°. '
  'A sum of such half-planes with weights $w_j$, $\\sum_j w_j\\,\\mathrm{ReLU}(r\\,\\vec n_j\\cdot\\hat x + \\beta_j)$, grows with $r$ in every direction that is covered; the set of points where it reaches a level $L$ is a polygon around the axis.', None),
 ('Method',
  'Write each second-round unit $u_c$ exactly as a sum of terms from each half-plane, the context $g$ and biases (largest error $3\\times10^{-5}$); find which intermediate $s$ units carry the half-planes. '
  'Map each unit\'s response to one particle over $(p_T, r, \\theta)$ and find its on-contour $r_c(\\theta)$. Compare the contour with the polygon predicted from the weights, with a circle, and with polygons in random directions. '
  'Shift whole jets by $\\delta$ in six directions and compare the network\'s mass read-out with the sum over particles of the per-particle maps, and with the same maps made symmetric.', None),
 ('Result: paths and shapes',
  '15 second-round units take half-plane input through five $s$ units (for example $s_{43}$ combines $a_{29}, a_6, a_{36}, a_{40}, a_{43}$ and feeds $u_1$ and $u_{29}$). Eleven use one $s$ unit, so their on-contour is exactly a level set of one weighted sum of 3–5 half-planes; four combine two or three. '
  'The contours are polygons with a ratio of largest to smallest radius of 1.14–1.47, within 3–9 % of a circle. Only $u_{40}$ uses all nine directions.', 'results/site/F8_e14_halfplanes.png'),
 ('Result: the direction term of the mass',
  'With the context held fixed, summing the per-particle maps over the particles of the shifted jet gives 0.96–1.05 of the measured direction-dependent part of the mass read-out (95 % intervals within [0.93, 1.08]). Making the maps point-symmetric removes that part ($0.0000$, interval $[-0.0005, +0.0005]$) and leaves the direction-independent part unchanged. '
  '15–24 % of the direction term passes through the context $g$ instead.', 'results/site/T19_e14_halfplanes.png'),
 ('Checks (E14v, fresh jets)',
  'All structure numbers reproduce within 4 points. Polygons in random directions with fitted weights fit the contours to 5–8 %, about as well as a circle, so the contour shape alone does not identify the directions; the identification rests on the exact weight decomposition. '
  'On the point-reflected sample the term from the sample\'s own asymmetry is $-0.0005$ to $+0.0003$ ($p \\ge 0.20$), so the direction term is the network\'s per-particle response. E14\'s first counterfactual added half of a reflected jet\'s change to the original, which produced negative inputs that the fixed-point layer cannot represent; the corrected construction symmetrizes both the base and the edited jet.', None),
 ('Why it matters',
  'It completes the step from directional cuts to near-radial objects with an explicit formula per unit, and it explains the part of the mass response that a function of $r$ alone cannot: the objects are polygons, not circles, and a shifted jet moves particles across their uneven edges.', None),
]

E20_EXPLAINER = [
 ('Question', 'Are the pT-rank windows of the second-round objects causal, or would a plain sum over all particles (as in sum pooling or energy-flow networks) respond the same way?', None),
 ('Background',
  'Each second-round object is $S_c = \\tfrac1{16}\\sum_i m_{ic}\\,u_c(x_i)$, where $u_c$ is the unit\'s response to one particle and $m_{ic}\\in\\{0,1\\}$ is fixed per slot $i$ (slots ordered by $p_T$) by the zero-bit entries of the fixed-point format. '
  'For example, $S_{43}$, $S_{52}$ and $S_{63}$ use only the 14 hardest particles (ranks 0–13). A sum-pooling network has $m_{ic}=1$ everywhere. '
  'Swapping only the angles $(\\Delta\\eta,\\Delta\\phi)$ of the particles at ranks $k$ and $k+1$ keeps the $p_T$ order, so the jet stays a valid input, and changes $S_c$ by $\\tfrac1{16}[m_{kc}(u_c(x_k\')-u_c(x_k)) + m_{k+1,c}(u_c(x_{k+1}\')-u_c(x_{k+1}))]$. '
  'Inside a window the two terms partly cancel; at an edge ($m_{kc}\\neq m_{k+1,c}$) one particle\'s full change survives.', None),
 ('Method',
  'On 10,000 jets (seed 41) and again on 9,861 fresh jets (seed 77): swap the angles of ranks $k, k+1$ for $k = 0,\\dots,16$ and record the mean $|\\Delta S_c|$ in the network and in the float program with the trained masks, with the masks shifted by $\\pm3$ slots, and with the masks removed. '
  'Controls: the adjacent swap inside the window (12/13) as the matched edit; masks permuted across slots at the same density; a random-weight network with the trained masks.', None),
 ('Result',
  'For $S_{43}$ the response is 0.084 per jet at the 13/14 edge, 0.004 at 12/13 and 0 at 14/15. With the masks shifted by $-3$ or $+3$ the peak moves to 10/11 or 16/17 (0.088). With the masks removed it falls by 95–98 %. $S_{52}$ and $S_{63}$ give the same pattern (0.056, 0.084), and $S_{13}$, $S_{17}$, $S_{62}$ show it at their own edges. '
  'Permuting the masks across slots lowers accuracy from 80.7 % to 32–36 %.', 'results/jedi/e20_verify_profiles.png'),
 ('Checks',
  'The network\'s $\\Delta S$ equals the window sum of its own per-particle responses ($R^2$ 0.99–1.00 for $S_{43}$, $S_{63}$). The 0–13 window is written three times in the network (the add-layer mask on $s$, the l3 input mask and qs1); any one of them reproduces the peak, which is why removing one layer\'s masks costs at most 1.1 points. '
  'Swaps inside a window still change hinge objects for up to 27 % of jets, because the hinge in $(p_T, r)$ is not a sum of a $p_T$ part and an $r$ part, and through fixed-point rounding. '
  'Not covered: $S_0$ and $S_{19}$ depend on $p_T$ only, so angle swaps do not test their windows; $S_1$ (edge 0/1), $S_{20}$, $S_{51}$, $S_{56}$ show no edge peak.', 'results/jedi/e20_rank_windows.png'),
 ('Why it matters', 'It tests with an input edit and a matched control what distinguishes the program\'s objects from sum pooling: whether a particle counts depends on its $p_T$ rank as well as on its cut, and the response follows the mask edge wherever it is placed.', None),
]

E7_EXPLAINER = [
 ('Question', 'Are the 16 class neurons $z_k$ functions of the network\'s own second-round objects $S$, and does a formula on $S$ respond to input edits the way the network does, where a formula on hand-chosen jet observables does not?', None),
 ('Background',
  'A formula can match a neuron on ordinary jets because it computes what the neuron computes, or because its inputs are correlated with the neuron on the data. '
  'A rigid shift of all particles by $\\delta$ leaves the invariant mass unchanged but raises $M_2=\\sum_i p_{T,i} r_i^2$ about the axis; the network responds (77 % of W jets become Z). Such an edit separates the two cases.', None),
 ('Method',
  'Each neuron is fitted on 20,000 jets as $\\hat z_k = c_0+\\sum_{m=1}^{K} c_m\\,h_m(X)$ with hinge terms $h_m$ on $X$ = the 19 $S$ units or 87 jet observables. The network\'s $z$ is replaced by $\\max(0,\\hat z)$ and its own last layer makes the call on 13,000 other jets. '
  'Then the jets are edited (shift 0.06 rad, rotation 45°, one particle 10 → 12 GeV across the $a_{32}$ cut) without refitting, and the formula\'s change $\\Delta\\max(0,\\hat z_k)$ is regressed on the network\'s change $\\Delta\\max(0,z_k)$ (slope 1 = same size). Controls: random hinge terms on $S$, and $S$ of a random-weight network.', None),
 ('Result',
  'At $K=32$ the $S$ formula changes 9.2–10.4 % of calls and the observable formula 7.4–7.5 %. Under the shift the $S$ formula\'s slope is 0.57–0.58 and the observable formula\'s 0.12–0.16 (it sends 2–4 % of W jets to Z, the network 77 %). Under the rotation the slopes are 0.42 and $\\le 0.011$.', 'results/jedi/e7_z_objects.png'),
 ('Checks (E7v, fresh seeds)',
  'Substituting the network\'s own $z$ changes 0 calls. The $S$ slope is the same for jets inside and outside the fit range (0.572 / 0.567). Random hinge terms on $S$ also send 72–86 % of shifted W jets to Z, so that number reflects the $S$ inputs; the slope separates the fitted formula (0.57) from random terms (0.25). '
  'The observable formula puts 37 % of its weight on the invariant mass; without the mass observables it changes 9.05 % of calls, sends 75.7 % of shifted W jets to Z and has slope 0.37.', 'results/jedi/e7_verify_z_objects.png'),
 ('Why it matters', 'The class neurons are approximately, not exactly, formulas on the objects: the $S$ formula keeps the sign and ordering of the network\'s responses at about half their size. The same-call rate on ordinary jets does not tell which formula follows the network; an input edit does.', None),
]

E21_EXPLAINER = [
 ('Question', 'Does the network combine the $M_0$ carrier $S_0$ and the $M_2$ carrier $S_{50}$ as a product $M_0 M_2$ (as the invariant mass needs), and at which layer?', None),
 ('Background',
  'An earlier test with a read-out probe was withdrawn because the probe gives similar values for sums and products. Here the second-round objects are set directly, with no input edit. '
  'Along the curve $S_0 S_{50}=$ const a unit that depends only on the product does not change; along the line where a standardised sum is constant a unit that depends only on that sum does not change. '
  'A second test uses the gradient: for $f(S_0 S_{50})$ the ratio of derivatives is $\\partial f/\\partial S_{50} \\,/\\, \\partial f/\\partial S_0 = S_0/S_{50}$, so regressing $\\log(g_{50}/g_0)$ on $\\log(S_0/S_{50})$ over jets gives slope 1 for a product and 0 for a fixed-weight sum.', None),
 ('Method',
  'On 10,000 jets (seed 47; repeated on seed 91), move $(S_0, S_{50})$ by 0.15–0.5 sd along each curve and record the ratio $R = \\sum|\\Delta U|_{\\text{product}}/\\sum|\\Delta U|_{\\text{sum}}$ for every unit of h4, h5, z and for the W − Z score difference. '
  'Compute the per-jet gradient slope for z0, z7, z10 and the W − Z difference. Controls: the same steps on 20 random pairs or triples of $S$ units, random ReLU layers on $S$, and synthetic product and sum units.', None),
 ('Result',
  'The ratio $R$ is 0.43–0.44 for the W − Z difference, but random $S$ pairs give 0.36–1.38: the two step directions are only about 32° apart, so $R$ mostly measures how much a read-out uses $S_{50}$ relative to $S_0$. Per-unit $R$ is the same at h4, h5 and z and the same as for random ReLU layers, so this test does not locate a layer. '
  'The gradient slope is 0.64 [0.59, 0.68] for the W − Z difference, 0.94 for z7 and 0.80 for z0, against $-0.59$ to $0.66$ for random pairs and $\\le 0.01$ for random ReLU layers; per jet the scatter is large (Spearman 0.21–0.43).', 'results/jedi/e21_isoproduct.png'),
 ('Checks', 'Setting the objects to their own values reproduces the network exactly. The patch steps are 6–40 times the input quantizer step of the next layer. The untrained network has only one live second-round unit, so it cannot serve as a control here. The seed-91 numbers come from a job that completed after the verifier\'s session and were not reviewed by the verifier.', None),
 ('Why it matters', 'The narrative step "the dense layers multiply the moments" is replaced by a measured statement: the class neurons depend on $S_0$ and $S_{50}$ approximately as on their product, with large per-jet scatter; the layer where this forms is the next test.', None),
]

X7_EXPLAINER = [
 ('Question', 'When a tagger is given inputs measured about a different jet axis than in training, do its calls change in the direction and by the amount its own axis-relative mass formula predicts, or does any network change calls under any shift of its inputs (covariate shift)?', None),
 ('Background',
  'The networks compute a mass-like variable $\\sqrt{M_0 M_2}$ from the particles\' offsets about the input axis, with $M_0 = \\sum_i p_{T,i}$ and $M_2 = \\sum_i p_{T,i}\\, r_i^2$. '
  'Moving the axis by a vector $\\mathbf{a}$ from the $p_T$ centroid $\\mathbf{c}$ gives $M_2(\\mathbf{a}) = M_2(\\mathbf{c}) + M_0\\,|\\mathbf{a}-\\mathbf{c}|^2$, so the predicted change of the log mass reading is '
  '$\\Delta = \\tfrac12\\,\\Delta\\log(M_0 M_2) = \\tfrac12 \\log\\big(M_2^{\\text{new axis}}/M_2^{\\text{training axis}}\\big)$. '
  'For a model trained on the stored (centroid) axis and evaluated about the hardest particle, $\\Delta > 0$ for every jet (mean $+0.14$). For a model trained about the hardest particle (E18) and evaluated on the stored axis, $\\Delta < 0$ for every jet (mean $-0.14$). '
  'A covariate-shift account predicts call changes in both directions without a preferred W/Z direction, and similar changes for any input shift of the same size.', None),
 ('Method',
  'Models: JEDI-linear (20,000 jets, sample seed 107) and Deep Sets / Linformer trained on the stored axis or about the hardest particle, two seeds each (20,000 jets of the val archive, X4 rows). '
  'Edits: inputs recomputed about the hardest particle or the WTA axis (stored-trained models), or on the stored axis or WTA axis (hardest-trained models). '
  'Reference for the formula: a real change of $\\tfrac12\\log(M_0M_2)$ of exactly $\\Delta$ per jet (offsets about the training axis scaled by $e^{\\Delta}$). '
  'Control: a rotation about the training axis whose per-jet angle $\\theta = 2\\arcsin(\\bar d / 2\\bar r)$ gives the same mean particle displacement $\\bar d$ as the axis change, with $M_0$ and $M_2$ unchanged. '
  'Invariant controls: A, JEDI-linear applied after recentring each jet on its $p_T$ centroid; B, an L1 logistic regression on 32 energy-flow polynomials (pairwise $\\Delta R$). '
  'Read-out: ridge from the last hidden layer to $\\tfrac12\\log(M_0M_2)$ about the training axis, change under the edit divided by its change under the matched real change.', None),
 ('Result',
  'Stored-trained models evaluated about the hardest particle change 31.0 % (JEDI-linear), 41.8–42.8 % (Deep Sets) and 32.4–34.6 % (Linformer) of calls; the matched real mass change changes 32.3 %, 35.0–35.2 % and 33.9–34.5 %; the matched rotation 3.2 %, 2.5 % and 0.9–1.0 %. '
  'Among calls that switch between W and Z, 1,740–2,230 go W → Z and 1–45 go Z → W. '
  'Hardest-trained models evaluated on the stored axis change 10.1–11.1 % (Deep Sets) and 17.6–20.5 % (Linformer) of calls, against 28.8–32.6 % for the matched real change and 0.8–2.9 % for the rotation; here 1–233 calls go W → Z and 694–1,997 go Z → W '
  '(true Z called W: 5 → 19 % Deep Sets, 5 → 45–50 % Linformer). Read-out response relative to the matched real change: forward 0.47–1.09 (JEDI-linear 0.80 [0.78, 0.82]), reverse 0.16–0.62; forward minus reverse 0.085–0.53, every 95 % interval above 0. '
  'Invariant controls A and B change 0 of 20,000 calls under both axis changes (their inputs are identical bit for bit).', 'results/jedi/x7_axis_reverse.png'),
 ('Checks',
  'JEDI-linear reproduces 80.75 % on its sample; the small models reproduce their stored test probabilities to $10^{-5}$. The rotation leaves $\\tfrac12\\log(M_0M_2)$ unchanged to $4\\times10^{-8}$ and matches the edit\'s mean displacement (median ratio 1.000). '
  'Rotation calls show no W/Z direction (JEDI-linear W → Z 2.4 → 2.2 %, McNemar p = 0.40). Per jet, the read-out change correlates with $\\Delta$ (Spearman 0.26–0.74, all intervals above 0). '
  'The per-jet change of the Z − W logit margin correlates with $\\Delta$ with opposite signs for W jets (forward +0.46 to +0.74) and Z jets (−0.59 to −0.76), so the pooled W+Z correlation is near 0. 266 tests, Holm-corrected.', None),
 ('Verification (X7v, fresh samples, confidence 0.4)',
  'Rerun on JEDI sample(3000, 211) and 10,000 new val rows: edit call changes within 1 point of X7 (JEDI 31.66 % [30.89, 32.35]). '
  'Added control m2t: the same translation $\\vec t$, then the offsets rescaled so that $\\tfrac12\\log(M_0M_2)$ about the training axis is unchanged ($\\Delta = 0$). Stored-trained models: 9.8–18.0 % of calls change (JEDI 11.4 %), 0.30–0.42 of the edit, 3.4–12× the rotation, with no W → Z flow (JEDI 95 W → Z vs 329 Z → W). '
  'Hardest-trained models: 23–31 % of calls change, 1.1–3.0× the edit, with W → Z 886–1,207 vs Z → W 1–68 (true W called Z 2.5–2.8 → 42.6–56.5 %). '
  'Added control rnd (JEDI): a translation of the same length in a random direction has the same $\\Delta$ (+0.141) and changes 35.4 % of calls with 1,594 W → Z vs 9 Z → W. '
  'Given the translation length $|\\vec t|$, the per-jet margin correlation with $\\Delta$ falls to −0.23 to +0.17. The invariant controls change 0 calls because their inputs are bit-identical by construction (a code check).', None),
 ('Why it matters',
  'For models trained on the stored axis, the W → Z direction of the call changes and about 2/3 of their number require the change of $\\Delta$; the remaining third also changes under a translation with $\\Delta = 0$. '
  'A rotation about the axis is not an adequate covariate-shift control for these near-radial models. Models trained about the hardest particle read $\\tfrac12\\log(M_0M_2)$ about their own axis together with a translation-invariant spread; their response on the stored axis (10.6–21.6 % of calls; read-out 0.16–0.59 of a matched real change) is the net of the two, and it is larger than the 0.13–0.41 measured in X4 with the axis-jump edit.', None),
]

X6_EXPLAINER = [
 ('Question', 'Are the program elements found in JEDI-linear (64 particles) properties of one trained checkpoint, or do they appear in other, separately trained official JEDI-linear models?', None),
 ('Background',
  'Three other official checkpoints read the same (pT, Δη, Δφ) inputs: 32 particles (78.0 %), 128 particles (80.9 %), and a permutation-invariant variant (81.8 %) whose fixed-point number formats are the same for every particle slot, so it cannot keep or drop particles by pT rank. '
  'In the 64-particle model, zero-bit number formats in the two sum layers keep only certain pT ranks for each object ("rank windows"). The permutation-invariant model tests whether the objects need such windows.', None),
 ('Method',
  'The tests used on the reference model, with thresholds fixed before the run, on 20,000 test jets (seed 0), 9,223 disjoint jets (seed 77) and, in the verification, 6,950 fresh jets (seed 4242): mean ablation of every unit; classification of each live first-layer unit as '
  '$\\alpha_{p_T} p_T + \\beta > 0$ (pT-only), $\\alpha_\\eta \\Delta\\eta + \\alpha_\\phi \\Delta\\phi + \\beta > 0$ (angle-only) or mixed, with a null that permutes each input\'s nonzero kernel entries over the 64 units; the kept pT ranks of each second-round object; '
  'swapping $(\\Delta\\eta, \\Delta\\phi)$ of two adjacent ranks across a window edge vs inside a window, ratio $R = \\overline{|\\Delta S|}_{\\rm across}/\\overline{|\\Delta S|}_{\\rm inside}$; the radial share of each object\'s response to one particle within each pT, against random combinations of the model\'s own angle cuts; and a rigid shift of every particle by 0.06 rad, '
  'which leaves the invariant mass unchanged, against a rotation about the axis with the same mean displacement.', None),
 ('Result',
  'All 57 live first-layer units in the four models have exactly zero weight on either pT or angle; random placement of the nonzero weights gives this in 16–48 % of draws per model, so the pattern is not separated from the sparsity. '
  'Within-pT radial share is 0.977–0.999 for 11 of 12 position-dependent objects among the five most used (above the 97.5 % point of the combination null, 0.92–0.955); one 128-particle object has 0.941. '
  'The 0.06 rad shift turns 45–76 % of W jets into Z calls, against 2–3 % for the matched rotation. In the three sorted-input models a hard-particle window (10, 14, 20 ranks) gives $R$ = 2.4, 5.4 and 4.6 (the 128-particle edge chosen after the run), 0.71–0.82 with the masks removed; at five other tested edges $R$ is 0.27–1.00 or about 0. '
  'The permutation-invariant model has no rank windows and is 0.90 [0.58, 1.22] points more accurate than the 64-particle model on the same 20,000 jets (paired; 0.49 [−0.04, 1.02] on the fresh jets).',
  'results/jedi/x6_models.png'),
 ('Checks', 'Unedited accuracy matches the official values within 0.7 points. Removing the slot masks only at real-particle slots costs 2.7 / 18.4 / 20.7 points (32 / 64 / 128 particles); letting padded slots into the sums as well raises this to 3.1 / 37.7 / 52.9. The untrained redraws give one class to every jet, so they cannot change calls under any edit.', None),
 ('Why it matters', 'The near-radial objects and the shift-based mass response are present in four separately trained checkpoints, including one without rank windows, so in this model family the objects do not require rank windows. Where windows exist, the hard-particle window acts causally, with a strength that differs between checkpoints.', None),
]

E22_EXPLAINER = [
 ('Question', 'When one unit of the network is set to its value on another jet, does the if-statement program, with the same variable swapped, make the same new call?', None),
 ('Background',
  'The program of E11 reproduces 94 % of the network\'s calls on ordinary jets. Agreement on ordinary jets can come from correlations between the program\'s variables and the network\'s units. '
  'An interchange intervention tests the variables one at a time: unit $k$ of the network gets its value on a source jet $s$, the program\'s formula $\\hat V_k$ gets its value on $s$, and the two calls are compared. '
  'On pairs where the swap does not change the network\'s call, agreement is mostly the 94 % base rate, so the measure is taken on the pairs $E(k)$ where it does (effective IIA).', None),
 ('Method',
  'Three samples (seeds 11, 91, 137; 8,149–8,848 held-out jets, 5,000 random base/source pairs each). For each live unit of $S$, $h_4$, $h_5$, $z$: the network with $V_k \\leftarrow V_k(s)$ and the program with $\\hat V_k \\leftarrow \\hat V_k(s)$ and its later formulas recomputed. '
  'Controls: the program\'s variable aligned to another unit (200 derangements); 1000 random directions per stage; a linear probe on 380 observables in place of the formula, fitted to the same target; a swap-free predictor (the program\'s runner-up class on the base jet). Intervals from a bootstrap over jets (pair weight $w_b w_s$); Holm over 118 units and over 4 stages.', None),
 ('Result',
  'Effective IIA per unit is 0.19–0.88 (stage means 0.52–0.66), against 0.22–0.35 for the permuted alignment. $S_0$ 0.76 [0.71, 0.80], $S_{29}$ 0.55–0.57, $S_1$ 0.38–0.41, $z_0$ 0.75–0.77, $z_7$ 0.63–0.72; none reaches 0.8. '
  'Against the swap-free predictor (0.51–0.54) the program is higher at $z$ by 0.15 [0.07, 0.23]; at $S$ by 0.03 [−0.04, 0.10], at $h_4$ and $h_5$ by about 0. Per unit, 17 of 118 are higher after Holm. Random swaps of a quarter of all units give 0.78–0.83 (0.944 without a swap; 0.54–0.62 misaligned).', 'results/jedi/e22_verify.png'),
 ('Checks',
  'The network tail used for the swaps reproduces the network exactly. With the $S$ formulas and the network\'s own layers after $S$, $S_{29}$ rises to 0.83 and $S_1$ to 0.74, so most of the shortfall is in the downstream formulas\' response to swapped values. '
  'At $z$ the unit basis scores 0.66 against 0.43 for random directions ($p$ 0.019; 0.076 after Holm over 4 stages). The probe gives lower IIA than the formula for 6–8 of 16 $z$ units, but it also fits post-ReLU $z$ less well ($R^2$ 0.36–0.89 vs 0.83–0.99) and the gap follows the fit gap (Spearman 0.43). The untrained network calls one class for every jet and gives no effective pairs.', None),
 ('Why it matters', 'The 94 % same-call figure measures agreement on ordinary jets. Under single-unit swaps the program gives the network\'s new call for about half to two thirds of the changed calls; the swap-specific part is measured at $z$, and the stage-level difference from a swap-free predictor at $S$, $h_4$, $h_5$ is not.', None),
]

B4_EXPLAINER = [
 ('Question', 'Does the nested if-statement program (E11) describe JEDI-linear better than standard compact surrogates of the same size, both in reproducing the calls and in reproducing how the calls change when the input jet is edited?', None),
 ('Background',
  'Description length (DL) counts the fitted pieces of a surrogate: hinge terms for MARS, internal nodes for a tree, leaves for boosted trees, nonzero weights for linear models. '
  'Edit agreement on a jet is 1 if the surrogate changes its call exactly when the network does, to the same new class: '
  '$A = \\mathbb{1}[f_s(x\') \\ne f_s(x) \\Leftrightarrow f_n(x\') \\ne f_n(x)]\\cdot\\mathbb{1}[f_n(x\') \\ne f_n(x) \\Rightarrow f_s(x\') = f_n(x\')]$. '
  'The control for the 0.06 rad shift moves each particle 0.06 rad in its own random direction; its expected change of $\\sum_i p_{T,i} r_i^2$ is $|d|^2\\sum_i p_{T,i}$, the same as the shift\'s.', None),
 ('Method',
  'Fit on 20,000 jets (seed 0), select on 10,000, test on 8,848 (seed 11), confirm on 8,519 (seed 91) and on 8,218 fresh jets (seed 23). Surrogates: the nested program with 2–30 terms per unit, MARS from 475 physics inputs straight to the 5 logits, '
  'decision trees, gradient-boosted trees (on the call and on the logits), L1 logistic and Lasso, energy-flow polynomials up to degree 3. '
  'Edits: 0.06 rad shift, 45° rotation, swap of the angles of pT ranks 13 and 14, one particle from 10 to 12 GeV, each with a control. For a fixed baseline $b$ the lead difference per jet is $[a_n - a_b]_{\\rm shift} - [a_n - a_b]_{\\rm random}$.', None),
 ('Result',
  'Direct MARS and boosted trees reach the same call as the 2–16-term programs with 4.6–19 times fewer pieces; at 16 and 30 terms the program is within 1 point (94.06 % at 30 terms on fresh jets). '
  'Under the shift the 16- and 30-term programs agree with the network 2.8 and 3.3 points more often than the strongest boosted-tree model (recall 61 % and 69 % vs 52 %); under the random displacement they lead the same model by 2.7 and 5.2 points, so the lead difference is $+0.13$ [$-1.29$, $1.52$] and $-1.95$ [$-3.41$, $-0.41$]. '
  'MARS fitted straight to the logits reproduces 18–23 % of the shift\'s call changes with no W → Z changes, and 56–57 % of the random displacement\'s.', 'results/jedi/b4_fidelity_dl.png'),
 ('Checks',
  'Every model reproduces its stored same call exactly on the test jets; recomputed physics inputs match the stored ones to $3\\times10^{-9}$; a Δφ shift gives the same leads as the Δη shift. Under rotation, rank swap and the 2 GeV crossing the network changes 1–4 % of calls and no surrogate reproduces more than 12 % of them. PySR and energyflow are not installed in the cluster env.', None),
 ('Why it matters',
  'The staged program is not the most compact way to reproduce the calls, and its advantage over boosted trees holds for mass-raising displacements in general, not for the coherent shift only. The edit that separates surrogates by mechanism here is the shift against the direct physics-to-logit MARS fits, which use the invariant mass.', None),
]

B5_EXPLAINER = [
 ('Question', 'Can a short hinge formula (at most 50 terms) on observables measured about the jet axis reproduce how JEDI-linear\'s calls change when the jet is shifted or re-axised, and does a formula of equal length on axis-free (translation-invariant) observables fail to?', None),
 ('Background',
  'The network\'s mass variable is formed about the given axis, close to $\\sqrt{M_0 M_2}$ with $M_0 = \\sum_i p_{T,i}$ and $M_2 = \\sum_i p_{T,i} r_i^2$, rather than the invariant mass $m$ (A4). A shift of every particle by 0.06 rad in $\\Delta\\eta$ leaves $m$ unchanged and turns 75 % of W calls into Z. '
  'The nested 3,824-term program (E11) reproduces 94 % of calls, but B4 showed it is not more compact than standard surrogates. $m$ and the energy-flow polynomials (EFPs) depend only on pairwise angles $\\Delta R_{ij}$, so they do not change under a shift or a re-axis.', None),
 ('Method',
  'Each of the 5 logits is a sum of hinge terms $(x-t)_+$, $(t-x)_+$ and products of two hinges, added one at a time by forward-stepwise MARS; one budget of $T \\in \\{5, 10, 20, 30, 50\\}$ terms is shared over the 5 logits (each new term goes to the logit with the largest dev-MSE gain), then refitted by least squares. '
  'Pools: P_axis (347 inputs: pT sums, observables about the stored axis, $M_0M_2$, $|\\vec M_1|^2$, rank-window sums), P_inv (158: pT sums, $m$, $m/M_0$, 32 EFPs), P_all (both). Splits as B4: fit 20,000, dev 10,000 (seed 0), test 8,848 (seed 11), confirmation 8,519 (seed 91). '
  'Edits on test jets with the physics inputs recomputed: +0.06 $\\Delta\\eta$ shift (trans) and a random per-particle 0.06 rad displacement (trans_c); 45° rotation; WTA and hardest-particle re-axis with random-direction re-axis of equal length (wta_c, hardest_c); the X4 axis jump with its null; the rank-13/14 swap. '
  '"Reproduces" = the program\'s 95 % interval of calls changed overlaps the network\'s and the paired difference interval contains 0 (jet bootstrap, 1,000; McNemar; Holm over 253 cells).', None),
 ('Result',
  'Network: trans 55.8 % [54.8, 56.8] of calls changed, wta 30.5 %, hardest 32.0 %, axis jump 4 of 212 vs null 7 of 212 (p 0.375). '
  'P_axis: same call 83.2 % at $T$ = 20 and 89.5 % at $T$ = 50; trans 51.2 / 68.1 %, wta 26.8 / 32.5 %; W → Z under trans 22.5 / 40.9 % (network 75.4 %). '
  'P_inv: same call 88.8 % at $T$ = 50 and 0.0 % calls changed under trans, wta and hardest (by construction). P_all $T$ = 50: trans 44.6 %, W → Z 78.0 %. mars30 (3,824 terms): trans 59.7 % (+3.8 [3.0, 4.6] points), wta 29.8 % (within interval), W → Z 77.5 %. '
  'No program of 50 terms or fewer meets the criterion on trans, wta and the jump together; mars30 also fails it on trans. '
  'The 20-term P_axis program, with $\\langle\\Delta R^2\\rangle_{[w]} = \\sum_{i\\in w} p_{T,i}\\Delta R_i^2 / \\sum_{i\\in w} p_{T,i}$ over pT ranks $w$: '
  '$\\ell_q = 0.469 + 0.000915\\,(6361 - M_0M_2)_+ - 0.354\\,(7.371 - \\Sigma r)_+$, '
  '$\\ell_W = -2.992 + 0.727\\,(10.26 - \\Sigma r)_+ - 0.0154\\,(1008 - \\Sigma p_{T[0\\text{-}49]})_+ - 621.8\\,(0.003486 - \\langle\\Delta R^2\\rangle_{[0\\text{-}49]})_+$, '
  '$\\ell_t = 4.175 - 254.1\\,(0.02557 - \\langle\\Delta R^2\\rangle)_+ + 0.620\\,(0.02557 - \\langle\\Delta R^2\\rangle)_+(1017 - M_0)_+$; $\\ell_g$ (7 terms) and $\\ell_Z$ (6 terms, including $(120.4 - \\sqrt{M_0M_2})_+$) are listed in inbox/merged/B5_report.md.', 'results/jedi/b5_summary.png'),
 ('Checks',
  'The recomputed physics matches the stored values to $5.7\\times10^{-8}$; splits have no overlap; network test accuracy 80.42 % [79.63, 81.22]. A 40-leaf boosted-tree model matches the network\'s trans rate (55.5 %) with W → Z 1.7 %. '
  'Verification (B5v, seed 23, 8,218 jets, programs not refit): rates reproduce within 0.6 points. The network\'s response depends on the sign and direction of the shift: W → Z 77.9 / 43.3 / 62.7 % for $+\\Delta\\eta$ / $-\\Delta\\eta$ / random direction, calls changed 55.5 / 62.1 / 59.1 % ($p = 5\\times10^{-54}$ for + vs −), and 3.7 % of calls change under the 45° rotation; '
  'every program (mars30 included) gives the same response for every direction (within 0.5 points) and 0–1.8 % under the rotation (P_axis 0.0 %). mars30 under $-\\Delta\\eta$: W → Z 73.1 % (+29.8 points). '
  'A random re-axis matched per jet to the WTA re-axis in $\\Delta M_2$ changes 33.6 % of network calls against 29.8 % for WTA ($p = 1.3\\times10^{-27}$), so that gap depends on the re-axis direction, not on $M_2$. Per-jet agreement with the network\'s changed calls is 3.3–3.6 points below the never-flip reference at $T$ = 20.', None),
 ('Why it matters',
  'Same call on unedited jets does not separate a program built on axis-relative observables from one built on invariant observables (≤ 1.2 points apart); the shift and re-axis edits separate them (51–68 % vs 0 %). '
  'A compact program that reproduces the shift response was not found at 50 terms with this fitter, and the network carries a direction-dependent term (sign of the shift, rotation) that none of the fitted radial programs has.', None),
]

E23_EXPLAINER = [
 ('Question',
  'On unedited jets whose stored axis is off the pT centroid, does JEDI-linear\'s mass read-out follow $\\sqrt{M_0 M_2}$ about the stored axis or the invariant mass $m$? '
  'This answers the objection to the rigid-shift test (E6, E6v): that edit creates off-centre jets, while on most data $m$ and $\\sqrt{M_0M_2}$ coincide.', None),
 ('Background',
  'About the stored axis, with $M_0 = \\sum_i p_{T,i}$, $\\vec M_1 = \\sum_i p_{T,i}(\\Delta\\eta_i, \\Delta\\phi_i)$, $M_2 = \\sum_i p_{T,i} r_i^2$, the small-angle mass is '
  '$m^2 \\approx M_0 M_2 - |\\vec M_1|^2$. Define $q = |\\vec M_1|^2/(M_0 M_2)$ and the centroid offset $d = |\\vec M_1|/M_0$; then $\\log\\sqrt{M_0M_2} - \\log m \\approx -\\tfrac12\\log(1-q)$. '
  'Moving every particle by $-\\vec M_1/M_0$ (recentring) sets $\\vec M_1 = 0$, leaves $m$ unchanged (a longitudinal boost plus an azimuthal rotation), and lowers $M_2$ to $M_2 - M_0 d^2$. '
  'The prediction for the read-out change is $\\tfrac12\\Delta\\log(M_0M_2) = \\tfrac12\\log(1-q)$ if the network uses $\\sqrt{M_0M_2}$ about the stored axis (H1), and 0 if it uses $m$ (H0). '
  'On the 100,000-jet sample the median $q$ is 0.00035 and the median $d$ 1.5 mrad; the top 5 % by $q$ have $q \\ge 0.0032$ ($d$ median 11 mrad), and 72 % of them are top jets, 0.4 % W or Z.', None),
 ('Method',
  '100,000 test jets (sample(20000, 101)); ridge read-out of $\\log m$ on $z$ fitted on 6,177 disjoint jets (seed 103; held-out $R^2$ 0.942). Tail sets: top 1/2/5 % of $q$, of $d$, and of $q$ within each class; bulk: $q$ below the median. '
  '(A) Each tail jet of class W, Z or t matched to a bulk jet of the same class with $|\\Delta\\log m|$, $|\\Delta\\log M_0| \\le 0.02$ (H1 predicts a read-out difference, H0 none), and separately on $\\log\\sqrt{M_0M_2}$, $\\log M_0$. '
  '(B) Recentring of every tail jet; matched real change: all offsets scaled by $k = e^{\\rm prediction}$, which changes $m$ and $\\sqrt{M_0M_2}$ by the same amount. '
  'Controls: recentring of 5,000 bulk jets (c1); the reflected move $+\\vec M_1/M_0$, which raises $M_2$ by $3M_0d^2$ (c2); a rotation about the axis with the same mean particle displacement, $M_2$ unchanged (c3); recentring with the nine angle-only P units frozen (c4). '
  '(C) 20,000 bulk jets shifted in a random direction by the median, 90th and 99th percentile of $d$ (1.5, 5.8, 13.1 mrad), plus a zero-size identity run. 95 % bootstrap intervals over jets (1,000), sign-flip permutation tests, exact McNemar tests for paired call changes, Holm over 37 set-by-edit tests.', None),
 ('Result',
  'Top 5 % by $q$ (5,000 jets): recentring changes the read-out by $-4.2\\times10^{-3}$ [$-5.1$, $-3.4$] against a prediction of $-2.4\\times10^{-3}$ and a matched real change of $-2.7\\times10^{-3}$; '
  'ratio to the matched real change 1.56 [1.23, 1.91] (1.50–2.01 over the nine tail sets; H0 ratio 0 rejected, Holm-adjusted normal $p \\le 10^{-10.4}$). The reflected move gives $+7.9\\times10^{-3}$ (prediction $+7.3$; ratio to its matched change 1.06 [0.94, 1.18]). '
  'Rotation gives $-0.4\\times10^{-3}$ [$-1.1$, $0.4$]; bulk recentring $0.02\\times10^{-3}$ [$-0.25$, $0.30$]; freezing the angle-only P units keeps 57–80 % of the recentring response. '
  'Calls: recentring changes 1.6–3.2 % of tail calls, rotation 0.9–2.5 %, the matched real change 0.2–0.5 %; Z → W vs W → Z counts are 10 vs 5 on the 2,000 tail W/Z jets (McNemar $p$ 0.30). '
  'Random shifts of natural size change 1.4 % (1.5 mrad), 2.2 % (5.8 mrad) and 3.8 % (13.1 mrad) of bulk calls, with W → Z 171 vs Z → W 22 at 13.1 mrad ($p = 9\\times10^{-30}$); the identity run changes none.', 'results/jedi/e23_tail.png'),
 ('Checks',
  'Unedited accuracy 80.92 %. Recentring: $|\\vec M_1|/M_0 \\le 7\\times10^{-9}$ after the edit, $|\\Delta\\log m| \\le 4\\times10^{-8}$; pT and padded slots unchanged; rotation leaves $M_2$ unchanged to $7\\times10^{-8}$. '
  'The observational match (A) gives tail − bulk read-out differences of $-5$ to $+10\\times10^{-3}$ with intervals of about $\\pm 6$–$13\\times10^{-3}$, against predicted differences of 2–4$\\times10^{-3}$: it does not separate H0 from H1 at this sample size. '
  'The particles beyond the training 99 % angle range change from 2.46 % to 2.53 % of particles under recentring (top 5 % $q$).', None),
 ('Verification (E23v, fresh sample, confidence 0.6)',
  'Rerun on sample(20000, 211), 100,000 jets (accuracy 80.86 %; read-out held-out $R^2$ 0.944). Recentring over the matched real change: Tq1 2.50 [1.91, 3.27], Tq5 2.08 [1.67, 2.49], Td5 2.03, Tq5c 2.26; over both samples 1.5–2.5. '
  'Added control: a translation by $\\vec t \\perp \\vec M_1$ with the same per-particle displacement, which raises $M_2$ by $M_0|\\vec t|^2$ and leaves $\\vec c\\cdot\\vec M_1 = 0$. It raises the read-out by $+1.5$ to $+2.2\\times10^{-3}$, 0.46–0.80 of its matched real change, so a generic translation does not lower the read-out. '
  'For a shift $\\vec c$, $\\Delta M_2 = 2\\vec c\\cdot\\vec M_1 + M_0|\\vec c|^2$. From recentre ($\\vec c = -\\vec M_1/M_0$) and reflect ($+\\vec M_1/M_0$) per jet, the part quadratic in $\\vec c$ is 0.62–0.71 of its prediction (equal to the perpendicular response within the intervals) and the part linear in $\\vec c\\cdot\\vec M_1$ is 1.34–1.82 (intervals exclude 1). '
  'The rotation (c3) leaves every radial quantity unchanged, so its null is expected for radial objects whatever the mechanism; bulk recentring (c1) moves particles by 0.65 mrad against 8–15 mrad on the tails. '
  'Calls: recentring and the perpendicular translation change the same share of calls (Tq5 −0.11 points [−0.49, +0.24]), although one lowers and one raises the predicted mass. Arm C W → Z excess reproduced (p90 76 vs 37, p99 175 vs 22).', None),
 ('Why it matters',
  'Under an edit that keeps $m$ fixed and moves unedited off-centre jets toward the centred configuration, the read-out changes with the sign of $\\tfrac12\\Delta\\log(M_0M_2)$ about the stored axis, and it changes with that sign for the reflected move and the perpendicular translation as well, so the axis dependence is not limited to the off-centre jets made by the rigid shift. '
  'The read-out is not a single function of $\\sqrt{M_0M_2}$: the part quadratic in the shift enters at 0.6–0.8 of the prediction and the part linear in $\\vec M_1$ at 1.3–1.8. '
  'Natural centroid offsets are 1–13 mrad, so the predicted change for tail W and Z jets is below $10^{-3}$ in $\\log m$ and the call changes on natural jets at this size are mostly not mass-specific (rotation changes a similar fraction).', None),
]

# per-paper explainers, rendered inside the paper's tab (title, steps)
H1_EXPLAINER = [
 ('Question', 'Do jet taggers change their decision when one particle is split into two exactly collinear pieces, or when a very soft particle is added? Neither change alters any IRC-safe property of the jet.', None),
 ('Physics background: IRC safety',
  'An observable $O$ is collinear safe if $O(p_1,\\dots,p_i,\\dots,p_n)=O(p_1,\\dots,z\\,p_i,(1-z)\\,p_i,\\dots,p_n)$ and infrared safe if $\\lim_{\\epsilon\\to 0} O(p_1,\\dots,p_n,\\epsilon q)=O(p_1,\\dots,p_n)$ for every direction $q$. '
  'Only such observables have finite perturbative QCD predictions and do not depend on how hadronisation or the detector splits energy. Mass, soft-drop mass, $\\tau_{21}=\\tau_2/\\tau_1$, $\\tau_{32}$ and $D_2=e_3/e_2^3$ are IRC safe. '
  'The multiplicity $N=\\sum_i 1$ and $p_T^D=\\sqrt{\\sum_i p_{T,i}^2}/\\sum_i p_{T,i}$ are not. An energy flow network $L=\\sum_i z_i\\,\\Phi(\\Delta\\eta_i,\\Delta\\phi_i)$ with $z_i=p_{T,i}/\\sum_j p_{T,j}$ is IRC safe by construction and serves as the zero control.', None),
 ('Method',
  'Split: replace particle $p$ by $p_1=z\\,p+k\\hat u$ and $p_2=(1-z)\\,p-k\\hat u$, with $\\hat u\\perp p$ and $k=\\Delta R\\,z(1-z)\\,p_T$; the 3-momentum is conserved exactly, and at $\\Delta R=0$ the four-momentum too. '
  'Soft emission: add a particle with $p_T=\\epsilon\\,p_{T,\\rm jet}$ in $\\Delta R<0.8$, its momentum taken from the hardest particle (control: that recoil alone). '
  'Count-fixed control: move the same fraction $z$ of the particle\'s momentum to its nearest neighbour, so $N$ stays fixed. '
  'Samples: 10,000 hls4ml jets (seed 401) and 10,000 JetClass jets (1,000 random rows per class, seed 402); nine models including an EFN trained here. '
  'Test: $H_0$: calls changed by the split = calls changed by the control (paired McNemar), Holm over 521 tests.', None),
 ('Result',
  'Splitting the hardest particle into 5 % and 95 % pieces at $\\Delta R=0$ changes 1.1–5.7 % of the calls of the eight trained taggers (JEDI-linear 2.60 [2.31, 2.93] % vs 1.22 % for the control; ParT_full 2.11 vs 0.72 %), against 0 % for the EFN on jets without truncation. '
  'Splitting more particles changes more calls (8 splits: JEDI 11.6 %, ParT_full 17.9 %, ParticleNet_full 21.6 %); hls4ml jets move into the gluon class, JetClass jets out of $W\\to qq$ into QCD.', 'results/physics/h1_panel.png'),
 ('Is the soft response a count? (verification H1v)',
  'A pure count $\\sum_i 1$ gives a response independent of $\\epsilon$ and of where the particle sits; a term $\\sum_i g(\\Delta\\eta_i,\\Delta\\phi_i)$ gives a position-dependent one. '
  'Down to $\\epsilon=10^{-6}$ the EFN goes to 0.00 %, the hls4ml taggers stay at 2–7 % (2.4–19× larger at the jet edge than in the core), ParT_full levels off at about 0.55 %, and ParT_kin rises to 17 %. '
  'Every added particle at $\\epsilon\\le 10^{-4}$ is softer than any particle in the data (0.10 vs 0.25 GeV in hls4ml, 0.059 vs 0.12 GeV in JetClass), so this is a mathematical IR probe on out-of-range inputs. '
  'The in-distribution evidence for counting: removing the softest real particle moves calls opposite to the splits, and adding an $\\epsilon=10^{-6}$ particle afterwards returns 38–67 % of those changed calls in hls4ml taggers and 14–19 % in the full JetClass models, against 1.3 % for the EFN.', None),
 ('Checks', 'IRC-safe observables change by at most $10^{-13}$ (relative) under the $\\Delta R=0$ split. Re-evaluation, re-sorting and rebatching change 0 calls. Slot 0 holds the hardest particle in all jets. Each model\'s unedited accuracy is within 1 point of its reference. Reproduced on fresh samples (seeds 411, 412).', None),
 ('Physical significance', 'The taggers read particle multiplicity, which is IRC unsafe and depends on hadronisation and detector granularity; this bears on transfer to data and to other simulators. The form of the IR unsafety differs by model: position-weighted counting (hls4ml taggers), a small constant rate (ParT_full), a growing response to sub-threshold inputs (kinematic-only JetClass models).', None),
]
H2_EXPLAINER = [
 ('Question', 'Before editing top and quark–gluon taggers, what do classical observables reach on the same test jets? Does the W mass inside a top jet carry information beyond total mass and three-prong structure? Does constituent multiplicity follow Casimir scaling?', None),
 ('Physics background',
  'A hadronic top decays as $t\\to bW\\to bq\\bar q\'$: three hard prongs, total mass $m\\approx m_t$, one prong pair with $m_W\\approx 80$ GeV. '
  'N-subjettiness $\\tau_N=\\frac{1}{R_0\\sum_i p_{T,i}}\\sum_i p_{T,i}\\min_k \\Delta R_{i,k}$ measures how well $N$ axes describe the radiation, so three-prong jets have small $\\tau_{32}=\\tau_3/\\tau_2$. '
  'Soft drop declusters until $\\min(p_{T1},p_{T2})/(p_{T1}+p_{T2})>z_{\\rm cut}$. '
  'Gluons radiate with colour factor $C_A=3$, quarks with $C_F=4/3$; at leading (double-log) order $\\langle N_g\\rangle/\\langle N_q\\rangle\\to C_A/C_F=9/4$, lowered by energy conservation and the running coupling.', None),
 ('Method', 'All test jets (403,999 Top Landscape, 199,998 quark–gluon). Observables from the constituents, cross-checked against fastjet on 300 jets. Rejection $R=1/\\varepsilon_B$ at $\\varepsilon_S=0.3$ and $0.5$ with bootstrap intervals; likelihood ratios of ($m_{SD}$, $\\tau_{32}$) with and without $m_W$ filled on separate validation jets; paired bootstrap of the two.', None),
 ('Result',
  'At $\\varepsilon_S=0.3$: $R=158$ for the $m_{SD}$ window + $\\tau_{32}$ cut, 243 for LR($m_{SD}$, $\\tau_{32}$), 305 for LR($m_{SD}$, $\\tau_{32}$, $m_W$), against 1602 for published ParT and 2193 for the released LorentzNet scores. Adding $m_W$ multiplies $R_{30}$ by 1.28 (log ratio +0.244 [+0.172, +0.335]). '
  'Quark–gluon: $N$ alone gives AUC 0.840; $\\langle N_g\\rangle/\\langle N_q\\rangle=1.589\\,[1.584, 1.594]$, below 9/4 ($p<0.001$).', 'results/physics/h2_fig_top_rejection.png'),
 ('Observables', 'Signal and background distributions of the classical observables used as references.', 'results/physics/h2_fig_observables.png'),
 ('Status', 'Released weights exist only for LorentzNet (top and QG). ParT, ParticleNet, EFN and PFN are being trained with the authors\' code; the edits (W-pair mass inside top; collinear split and particle removal on quark–gluon jets) follow on the trained models.', None),
 ('Physical significance', 'These are the reference points for the edits: a network that uses only $m$, $m_{SD}$ and $\\tau_{32}$ would sit near the classical LR2 rejection. The $m_W$ information exists, so the edit can tell whether a network uses it. Multiplicity carries most of the classical quark–gluon separation, and the measured Casimir ratio (1.59) sets the expected size of a multiplicity-driven response.', None),
]
H3_EXPLAINER = [
 ('Question', 'Do the physics statements D1 (JEDI-linear: mass windows plus a $\\sum p_T$ shortcut, no W-mass constraint inside top) and D2 (ParT: displaced-track multiplicity rather than lifetime, lepton isolation used) hold with 10× more jets, random JetClass rows, the corrected $\\eta$ build and a matched control for every edit?', None),
 ('Physics background',
  'A longitudinal boost along the jet axis rescales momenta by $k$ and angles by $1/k$, leaving $m^2\\approx\\sum_{i<j}p_{T,i}p_{T,j}\\Delta R_{ij}^2$ and N-subjettiness ratios unchanged; only $\\sum p_T$ changes. '
  'Top taggers can require a subjet pair with $m_{jj}\\approx m_W=80.4$ GeV. A b hadron travels $c\\tau\\approx 450\\,\\mu$m and yields about 5 charged tracks; a track\'s impact-parameter significance is $S=d_0/\\sigma_{d_0}$. '
  'A prompt lepton from $W\\to\\ell\\nu$ is isolated.', None),
 ('Method',
  'For each edit $E$ and control $C$ (same size of change, tested quantity held fixed), on jets of class $c$: $\\Delta_E=\\Pr(\\text{call}=c\\mid E)-\\Pr(\\text{call}=c)$ with a 95 % bootstrap interval; $H_0:\\Delta_E=\\Delta_C$ by exact McNemar on paired calls, Holm-corrected. '
  'Samples: JEDI-linear 20,000 hls4ml jets (seed 501); ParT 40,000 JetClass jets (4,000 random rows per class). Verification (H3v) on fresh samples with two new controls: a W-pair edit control that holds the W pair at its mass, and an H→bb edit that caps every track at $S=3.5$, which keeps the number of displaced tracks.', None),
 ('Result',
  'JEDI-linear W jets under $p_T\\times 0.9$: W call 87.5 → 27.0 % (rotation control 87.2 %); fresh sample 87.4 → 26.9 %. ParT_full H→bb: capping $S$ at 3.5 costs 3.1 [2.2, 3.9] points, keeping only the 2 most displaced tracks costs 61 points, so the call follows the number of tracks with $S\\gtrsim 2$–3, not how far they travel. '
  'W-pair mass inside top (60–120 GeV at fixed jet mass) against the W-fixed control: no model differs after Holm (JEDI-linear, ParT_full, ParT_kin; all $p\\ge 0.13$). ParT\'s top call drops by a similar amount (up to 16 points at 120 GeV) whichever subjet pair is moved.', 'results/physics/h3_effects.png'),
 ('Checks and corrections', 'Unedited accuracies: JEDI 81.07 %, ParT_full 86.38 %, ParT_kin 74.55 %. The corrected rebuild changes 0.025 % of ParT_full calls (mirrored build 1.14 %). H3\'s first W-pair control also moved the W-pair mass by 10–28 GeV the other way; its ParT_full ±2.7-point asymmetry did not replicate. "Lifetime × 0.35" lowers the $S>2$ count by 26 %, so it is not a fixed-count edit. Not re-verified: isolation, lepton charge flip, $\\pi^0$, dead cone.', None),
 ('Physical significance', 'The earlier physics edits hold at 10× statistics. JEDI-linear\'s W/Z/top calls depend on $\\sum p_T$ through a boost that leaves every invariant shape unchanged; ParT\'s b tagging counts significantly displaced tracks; no tested network checks that one subjet pair has the W mass inside a top jet.', None),
]

X8_EXPLAINER = [
 ('Question', 'Do the headline edit responses of JEDI-linear (translation W → Z, WTA re-axis, rank-window swap, radial response) require its trained weights, or do networks with the same architecture, inputs and quantizer masks but no trained function give the same responses? Do the unit costs found by mean ablation hold when a unit is instead given its value on another jet?', None),
 ('Background',
  'An edit response can come from the geometry of the inputs and layers rather than from learned function: for example, any sum of ReLU units over particles grows when particles move outward. Functional nulls remove the learned function while keeping the architecture. '
  'Shuffled-label training uses the same data with the labels permuted once; weight permutation keeps every weight value and rearranges them within a matrix. For JEDI-linear three permutations were used: (a) nonzero kernel entries permuted, zero-bit masks in place; (c) as (a), plus the per-slot number formats permuted per channel; (b) all entries and all quantizer tables permuted. '
  'Mean ablation sets a unit to $\\bar h_c$, a value no jet may have; resample ablation sets it to $h_c(\\pi(j))$, its value on another jet $\\pi(j)$, with $\\pi$ a random derangement.', None),
 ('Method',
  'Sample: sample(per_class=2000, seed=307), $N = 10{,}000$ jets, $N_W = 2{,}000$. Models: trained JEDI-linear and 15 permuted copies (5 per variant); Deep Sets (dsets-d64) and a float JEDI-style model with a context round (jedis-d64), each with trained seeds 0 and 1, shuffled-label seeds 0 and 1 and 5 permuted copies. '
  'Edits, each with a matched null edit: T, translation by 0.06 rad along $\\Delta\\eta$ (null: random direction); W, inputs about the WTA axis (null: translation of the same per-jet length in a random direction); K, angles $\\times 1.1$ at fixed $p_T$ (null: rotation with equal displacement); angle swap of ranks 13|14 (across the window edge) vs 5|6 (inside). '
  'Statistics: H1 $\\Delta_{W\\to Z} = \\frac{1}{N_W}\\sum_{j\\in W}\\big([c_T(j)=Z]-[c_0(j)=Z]\\big)$; H2 share of calls changed under W; H3 across/inside ratio of $\\overline{|\\Delta S_c|}/\\mathrm{sd}(S_c)$ over the live S units with a mask edge at 13|14; H4 response under K, in s.d., of the S unit with the largest $|\\rho|$ with $\\sum_i p_{T,i} r_i^2$. '
  '$H_0$: $\\theta_{\\rm trained}\\le\\theta_{\\rm null}$ for some null; $p_{\\rm IUT}=\\max_i p_i$ over nulls from a paired jet bootstrap (1,000 resamples), Holm over 12 tests; exact McNemar for paired call changes against each matched null edit.', None),
 ('Result',
  'Trained against null range. H1 W → Z flow: JEDI-linear 0.752 [0.730, 0.771] vs −0.107 to 0.135; Deep Sets 0.676 / 0.724 vs ≤ 0.016; JEDI-style float 0.660 / 0.734 vs ≤ 0.0005 (Holm $p = 0.012$, the floor, for all three). '
  'H2 calls changed under W: JEDI-linear 0.298 vs 0 to 0.627 (Holm $p = 1$); the accuracy drop under W is 0.219 vs ≤ 0.020. '
  'H3 ratio: 5.22 [4.99, 5.48] vs 0.52–9.27 (variant a) and 3.2–11.1 plus two $\\infty$ (variant c); all variant-b copies give a constant call. '
  'H4: S50 (selected by the frozen rule, $\\rho = 0.93$) moves 0.200 [0.196, 0.205] s.d. vs ≤ 0.129. '
  'Ablation: mean and resample costs rank the 43 live units the same way (Spearman 0.965 [0.954, 0.973]); the context round g changes 14.6 % [13.9, 15.3] of calls under mean ablation and 27.8 % [27.3, 28.3] under resample ablation (McNemar $p = 7\\times10^{-174}$).', 'results/jedi/x8_nulls.png'),
 ('Checks',
  'The script\'s forward pass equals Jedi.run exactly before and after the permutations; small-model forwards equal the torch modules; shuffled-label models are at chance on true labels; each matched null edit gives ≈ 0 radial response (Kr 0.0006). '
  'The frozen radial rule chose S50, not S29 (S29: 0.261 s.d.). Many permuted copies give a constant call, so for them "calls changed" is 0 by construction.', None),
 ('Verification (X8v, fresh sample seed 7307, confidence 0.65)',
  'All JEDI-linear numbers return within their intervals: H1 0.744 [0.724, 0.765] vs ≤ 0.109 (now also against three untrained redraws that keep the trained mean and s.d.); H3 5.19; H4 0.205; g 14.85 % → 27.88 %; Spearman 0.965 over 44 units. Added controls: '
  '(1) on the trained network, live S units without a mask edge at 13|14 give an across/inside ratio of 1.45, edge units 5.19, and non-degenerate mask-keeping permuted copies 2.07–11.34, so the ratio follows the mask edge with any weights; '
  '(2) dividing the radial response by $|\\rho|\\,E_Q$ ($E_Q$ = expected change of $\\sum p_T r^2$ in s.d.) gives 0.98 for the trained unit and 0.95–1.13 for constant-output nulls, so the gap comes from how closely the trained unit tracks $\\sum p_T r^2$ ($\\rho$ 0.93 vs ≤ 0.70); '
  '(3) the random-direction translation of the WTA length lowers accuracy by 0.250, more than the WTA re-axis (0.218), and changes more calls (McNemar 323 vs 641, $p = 7\\times10^{-25}$). The small models were not rerun.', None),
 ('Why it matters',
  'The translation response that defines the un-recentred mass variable (claim A4) needs the trained weights in three architectures. The rank-window asymmetry measured by the angle-swap edit (claims A1, A3) is produced by the zero-bit mask layout, so it describes the masks fixed during HGQ training rather than a use of the window by the trained weights beyond the masks. The WTA accuracy loss (claim D4) follows the size of the displacement. Unit costs, and the role of the context round, hold under resample ablation, where g\'s share of calls is about twice the mean-ablation figure.', None),
]

X5_EXPLAINER = [
 ('Question', 'Do the elements found in JEDI-linear (pure pT-only or angle-only first-layer cuts, pT-rank windows, count-weighted radial objects, the response to a rigid shift) come from its fixed-point (HGQ) training, or from the task and inputs?', None),
 ('Background',
  'JEDI-linear is a Deep-Sets-like network trained with HGQ, which learns a bit width per weight and per input slot; zero bits remove a weight or a slot. Its first layer keeps 20 of 192 weights, and every live unit depends on $p_T$ only or on $(\\Delta\\eta,\\Delta\\phi)$ only. '
  'A permutation null (X6v) showed that random placement of 20 nonzero weights gives this purity often ($p$ 0.16–0.48). Training the same Deep Sets without quantization, or with an L1 penalty that only produces sparsity, separates the sources.', None),
 ('Method',
  'Deep Sets $S_c=\\sum_i \\phi_2(\\phi_1(x_i))_c$ followed by $\\rho(S)$, trained on the same 3-feature, 64-particle inputs, 3 seeds each: float; L1 on $\\phi_1$ with $\\lambda=3\\times10^{-2}$ (0.89–0.91 zero weights; JEDI-linear 0.896); HGQ for 10 epochs at $\\beta=3\\times10^{-4}$ (largest zero fraction reached, ≤ 0.094) and $\\beta=10^{-8}$ (accuracy within 1 point). '
  'On 10,000 test jets (seed 311): (a) share of live first-layer units with ≥ 90 % of their input variance from $p_T$ alone or from the angles alone, against weights permuted across units (200 draws); (b) swap the angles of ranks $k, k{+}1$ for $k = 0\\ldots16$ and look for an edge (contrast ≥ 1.5); (c) single-particle scans of each pooled unit for a radial, count-weighted profile; (d) translate every particle by 0.06 rad and compare the log-mass read-out change with $\\tfrac12\\Delta\\log(M_0M_2)$, calibrated by a real angular scaling, and with a rotation of equal displacement. '
  'An element is present in a regime when it passes Holm $\\alpha = 0.05$ in ≥ 2 of 3 seeds; regimes compared by an exact permutation test over 1,680 relabellings.', None),
 ('Result',
  'The shift response appears in every regime: W → Z 0.02–0.08 → 0.49–0.87 (rotation 0.02–0.09), calibrated read-out 0.72–1.26 of the $\\sqrt{M_0M_2}$ prediction (float 0.72–0.88, L1 0.82–0.95, HGQ 0.92–1.26). '
  'Pure cuts: float 0.31–0.39 ($p$ 0.47–0.96); L1 1.00 with a null of 0.86–1.0 ($p$ 0.55–0.74); HGQ 0.30–0.50, above the null in seeds 0 and 2. No regime has rank windows (edge contrast 1.03–1.05) or more than 2 count-weighted radial objects. Regime differences after Holm: $p \\ge 0.095$.', 'results/jedi/x5_dsets_sparsity.png'),
 ('Checks',
  'Each network\'s forward reproduces its stored test probabilities (argmax agreement 1.000). HGQ at 10 epochs set ≤ 1 % of pooling-input slots to zero bits, so it formed no slot masks; whether longer HGQ training forms windows is not tested. HGQ $\\beta = 3\\times10^{-4}$ models are 3–5 points below float (flagged).', None),
 ('Verification (X5v, fresh sample seed 4242, confidence 0.72)',
  'With a 2,000-draw null, HGQ purity $p$ is 0.0005 / 0.175 / 0.0015 ($\\beta = 3\\times10^{-4}$) and 0.0010 / 0.062 / 0.0005 ($\\beta = 10^{-8}$); L1 0.56–0.72; float 0.45–0.96. At a 0.95 threshold the $\\beta = 10^{-8}$ arm holds in 1 of 3 seeds. The two HGQ arms have near-identical first layers at equal seed (per-unit $p_T$-share correlation 0.976–0.987), so they are one result. '
  'The HGQ training script also differs from the float one in optimizer (Adam vs AdamW), batch-norm folding, epochs, learning rate, batch size and initialisation; no float model trained with the HGQ script exists, so the purity is not attributed to quantization. '
  'Translations along $\\pm\\Delta\\eta$: W → Z 0.43–0.89 vs rotation 0.022–0.075; read-out ratio 0.71–1.29 (rotation −0.005 to 0.007). The calibration edit (angles $\\times 1.35$) overflows a later HGQ quantizer in 19–28 % of jets.', None),
 ('Why it matters', 'The un-recentred shift response of claim A4 is present in this architecture without fixed-point training (claim A10). The purity of a very sparse first layer is what sparsity alone gives (L1 equals its null); above-null purity appears only with the HGQ training script and is not separated from its other training differences. Rank windows did not form in any regime tested, so their origin in HGQ training remains open (X5w).', None),
]

H8_EXPLAINER = [
 ('Question', 'If we describe each tagger by the best combination of standard IRC-safe jet shapes, does that description also predict how the tagger responds to the axis, translation, boost and collinear-split edits?', None),
 ('Physics background: EFPs and ADO',
  'Energy-flow polynomials (EFPs) form a linear basis of IRC-safe observables: for a multigraph $G$ with $N$ vertices and edges $(k,l)$, $\\mathrm{EFP}_G=\\sum_{i_1\\dots i_N} z_{i_1}\\cdots z_{i_N}\\prod_{(k,l)\\in G}\\theta_{i_k i_l}$ with $z_i=p_{T,i}/\\sum_j p_{T,j}$ and $\\theta_{ij}=\\Delta R_{ij}^{\\beta}$. '
  'The single edge at $\\beta=2$ is $\\approx m^2/p_T^2$; the triangle is $e_3$, so $C_2=e_3/e_2^2$ and $D_2=e_3/e_2^3$. All EFPs depend only on pairwise angles, so they do not change when the jet axis is redefined or the particles are translated together. '
  'Average decision ordering between two scores $f,g$: $\\mathrm{ADO}=\\Pr\\big[\\mathrm{sign}(f_s-f_b)=\\mathrm{sign}(g_s-g_b)\\big]$ over signal–background pairs $(s,b)$; ADO = 1 means the two rank every pair the same way (Faucett et al. 2021).', None),
 ('Method',
  'Networks: JEDI-linear (10,000 hls4ml jets, seed 801), ParT_full and ParT_kin (4,000 JetClass QCD/W/Z/top jets, 1,000 random rows per class, rng 802); binary score $s=\\log\\sum_{\\rm sig}p-\\log\\sum_{\\rm bkg}p$. '
  'Pool: 303 EFPs ($d\\le5$, $\\beta\\in\\{0.5,1,2\\}$), $m$, $m_{SD}$, $\\tau_{21}$, $\\tau_{32}$, $C_2$, $D_2$. Guided search on one half of the jets: start with the observable of largest ADO with $s$; then repeatedly fit a boosted-tree surrogate, find the pairs it orders differently from $s$, and add the observable with the largest ADO on those pairs; stop when ADO gains $<0.002$ or at 8 observables. '
  'On the other half, apply the WTA re-axis, a 0.06 rad translation, $p_T\\times0.9$ with angles $/0.9$, and the 5 %/95 % collinear split of the hardest particle, and compare % binary calls changed for network and surrogate (paired McNemar, BH over 36 tests). Null: the same networks with redrawn weights.', None),
 ('Result',
  'Natural jets: ADO 0.90–0.97 for W vs QCD and top vs QCD in all three networks, 0.977 for JEDI Z vs W ($m$, $C_2$), 0.66 and 0.75 for ParT_full and ParT_kin Z vs W. The first observables chosen are $\\tau_{21}$ (W vs QCD), $e_3$ with $\\tau_{32}$ (top vs QCD, ParT) and $m$ (Z vs W). '
  'Edits: under the WTA re-axis the networks change 1.7–37.3 % of binary calls and the surrogates 0.0 %; under translation 0.9–52.8 % vs 0.0 %; under the split 0.1–3.5 % vs 0.0 % (outside 64-slot truncation). 33 of 36 network–surrogate differences have BH $q<0.05$. '
  'Under the boost the surrogates change too (EFPs scale as $\\theta^\\beta$): JEDI Z vs W 17.5 % vs 0.8 %, ParT_full W vs QCD 1.9 % vs 6.6 %.', 'results/physics/h8_fig.png'),
 ('Checks', 'Unedited accuracies: JEDI-linear 80.86 % (ref. 80.6 %), ParT_full 86.05 % (85.9 %), ParT_kin 73.71 % (74.5 %); JEDI WTA changes 30.5 % of 5-class calls (D4 30.3 %). '
  'Observables change by more than $10^{-2}$ (relative) in at most 0.01 % of jets under WTA and translation; under the split only in hls4ml jets that fill all 64 slots. '
  'A boosted-tree and an MLP surrogate give ADO within 0.02 and the same edit pattern. Redrawn ParT_full: ADO $\\le0.58$, yet 7–33 % of its calls change under the edits; redrawn JEDI gives a constant call.', None),
 ('Verification (H8v): a positive control with the axis-moment mass',
  'The surrogate\'s 0 % under WTA and translation holds by construction (the pool is axis-free), so those tests only show that the networks are not axis invariant (claim D4). '
  'H8v replaced $m$ by the axis-moment mass $m_A=\\sqrt{M_0M_2}$ with $M_0=\\sum_i p_{T,i}$ and $M_2=\\sum_i p_{T,i}\\,\\Delta R_i^2$ about the axis the network is given (claim A4). On natural jets $m_A\\approx m$ (correlation of the logs 0.99999 on hls4ml), and the ADO is unchanged or higher in all 9 rows. '
  'On fresh jets (seeds 811, 812) this surrogate changes 37.7 % of JEDI-linear\'s Z vs W calls under WTA against the network\'s 37.6 % (difference CI [−1.7, 1.5] points, per-jet $r=0.88$), and 22.5 vs 17.1 % for W vs QCD. For ParT it gives responses of the same order but larger (per-jet $r$ 0.3–0.6). '
  'It does not reproduce the collinear-split response (0.0–0.4 % vs 0.1–3.4 %) or JEDI-linear\'s response to $p_T\\times0.9$. The redrawn ParT predicts one class for 99.95 % of jets, so its binary-call changes reflect a near-zero binary score.', None),
 ('Physical significance', 'Standard IRC-safe shapes (mass, $\\tau_{21}$, $\\tau_{32}$, $e_3$) reproduce most of these taggers\' rankings on natural jets. '
  'They give no response to the jet-axis convention, but natural jets cannot distinguish the invariant mass from the mass about the given axis, and the latter reproduces JEDI-linear\'s axis response; the axis dependence is therefore mostly one known variable. '
  'Not described by either surrogate: the response to collinear splitting, JEDI-linear\'s response to the total $p_T$, and ParT\'s Z vs W ranking (ADO 0.66–0.76).', None),
]

H4_EXPLAINER = [
 ('Question', 'Inside a top jet, do trained top taggers use the mass of the W-candidate subjet pair beyond the total jet mass and the three-prong measure $\\tau_{32}$ (claim D6)?', None),
 ('Physics background',
  '$t\\to bW\\to bq\\bar q\'$ gives three prongs with $m\\approx m_t$ and one pair with $m_{jj}\\approx m_W=80.4$ GeV. N-subjettiness $\\tau_N=\\frac{1}{R_0\\sum_i p_{T,i}}\\sum_i p_{T,i}\\min_k\\Delta R_{ik}$, and $\\tau_{32}=\\tau_3/\\tau_2$ is small for three-prong jets. '
  'For three massive subjets $m^2=m_{12}^2+m_{13}^2+m_{23}^2-\\sum_k m_k^2$, so one pair mass can change at fixed $m$ if the other pairs compensate. The earlier result D1 found no W-pair-mass use in JEDI-linear and ParT trained on hls4ml / JetClass.', None),
 ('Method',
  'Sample: 5,000 top + 5,000 QCD Top Landscape test jets (seed 601, random rows) + 100k QCD. Three exclusive-$k_T$ subjets; W pair = pair with mass closest to 80.4 GeV. '
  'Edit $W$: rescale the W pair\'s opening about its centroid until $m_W=60$ or 100 GeV (top) or 80.4 GeV (QCD); move the third subjet to restore $m$ ($|\\Delta m|<0.5$ GeV); rescale the subjet sizes by a common factor $h$ to restore $\\tau_{32}$ ($|\\Delta\\tau_{32}|<0.02$). '
  'Control $N$: the same relative change on a non-W pair ($|\\Delta m_W|<2$ GeV). Score $s$ = log-odds of top; statistic $\\Delta_W-\\Delta_N$ with $\\Delta_E=\\langle s(E(j))-s(j)\\rangle$, paired bootstrap, Holm. '
  'Positive controls: LR and MLP on $(m_{SD},\\tau_{32},m_W)$; negative: LR on $(m_{SD},\\tau_{32})$. Verification adds MLPx, an MLP on the edit\'s side-effect variables (subjet masses, $\\tau_{21}$, $\\tau_{1\\text{–}3}$, $N$, subjet $p_T$ fractions) without any pair mass, and MLPxW = MLPx + $m_W$.', None),
 ('Result',
  'Top jets, 80 → 60 GeV, $\\Delta_W-\\Delta_N$ (log-odds): ParT −1.73 [−1.91, −1.55], ParticleNet −1.80, LorentzNet −2.31, PFN −0.83, EFN −0.68, MLP($m_W$) −1.52, LR($m_{SD},\\tau_{32}$) +0.03 [−0.02, +0.07]. '
  'At $\\varepsilon_S=0.5$: ParT pass rate 69.1 → 32.4 % (control 64.9 %; McNemar 204/23, $p=2\\times10^{-37}$). QCD jets with the pair moved onto 80.4 GeV: ParT +0.24 [+0.07, +0.42].', 'results/physics/h4_fig_wmass.png'),
 ('Verification (H4v): side effects of the edit',
  'Fresh sample (seed 611, H4 rows excluded): ParT −1.79 [−1.98, −1.60] at 60 GeV, −1.14 at 100 GeV. MLPx responds −0.63 [−0.71, −0.55] and −0.54, MLPxW −1.89 and −1.11. '
  'ParT, ParticleNet and LorentzNet respond 2.1–4.1 times MLPx and as much as MLPxW; PFN (−0.77, −0.61) and EFN (−0.63, −0.66) equal MLPx within the intervals. '
  'The edit shrinks the subjets ($h$ median 0.78 vs 0.97 for the control) and lowers $\\tau_{21}$; the 60 GeV jets are separable from natural ones (C2ST AUC 0.98), and on the natural-looking half of the 100 GeV edits ParT still responds (−0.66 [−0.91, −0.41]).', None),
 ('Checks', 'Unedited AUC: ParT 0.9860 (R30 1493 vs published 1602), ParticleNet 0.9862, LorentzNet 0.9871 (released scores 0.9870), PFN 0.9806, EFN 0.9756. A rotation of equal displacement changes scores by at most 0.09.', None),
 ('Physical significance', 'Top taggers trained on Top Landscape (ParT, ParticleNet, LorentzNet) check the W-pair mass inside the top, not only the total mass and three-prong shape; JEDI-linear and ParT trained on the multi-class hls4ml / JetClass tasks did not (D1). For PFN and EFN this edit cannot separate W-mass use from the side effects.', None),
]

H5_EXPLAINER = [
 ('Question', 'Gluon jets have more particles than quark jets. Do trained quark–gluon taggers decide by counting particles, by the amount Casimir scaling with Poisson statistics predicts (claim D7)?', None),
 ('Physics background',
  'Gluons radiate with $C_A=3$, quarks with $C_F=4/3$; in this sample $\\langle N_g\\rangle/\\langle N_q\\rangle=1.589$ (H2). If $N_q\\sim\\mathrm{Poisson}(\\lambda_q)$ and $N_g\\sim\\mathrm{Poisson}(\\lambda_g)$, then $\\log\\mathrm{LR}(N)=N\\log(\\lambda_g/\\lambda_q)-(\\lambda_g-\\lambda_q)$, so one more particle adds $\\log1.589=0.463$. '
  'A collinear split $p\\to(zp,(1-z)p)$ leaves IRC-safe observables (girth $\\sum_i z_i\\Delta R_i$, $C_1^{(\\beta)}=\\tfrac12\\sum_{ij}z_iz_j\\Delta R_{ij}^\\beta$) unchanged, adds one particle, and changes $p_T^D=\\sqrt{\\sum_i p_{T,i}^2}/\\sum_i p_{T,i}$.', None),
 ('Method',
  '10,000 Pythia test jets (5,000 per class, seed 701). Scorers: ParT and ParticleNet with and without particle ID, LorentzNet (released), PFN, EFN, the count-only LR($N$) from training histograms, a weight-permuted ParT. '
  'Edits: split particle of rank 0 / 1 / 5 / softest ($\\Delta R=5\\times10^{-4}$); $k=1..5$ random splits; remove the softest; count-fixed control (merge the two softest and split the hardest, $\\Delta N=0$). Metric: mean $\\Delta\\,\\mathrm{logit}=\\Delta[\\log p_g-\\log p_q]$ with bootstrap CI; slope in $k$.', None),
 ('Result',
  'Slope per random split: networks 0.030–0.061, EFN 0.000, LR($N$) 0.095, Poisson 0.463. Split hardest: +0.10 to +0.20; count-fixed control: the same (restore fraction −0.11 to +0.14). Remove softest: −0.011 to +0.003 (LR($N$) −0.099). Split softest: toward quark (−0.01 to −0.05).', 'results/physics/h5_counting.png'),
 ('Verification (H5v): an on-manifold count test',
  'Fresh sample (seed 9137). Adding one soft particle at a realistic position (a classifier cannot tell these jets from real ones, C2ST AUC 0.50) moves the networks −0.024 to −0.004 (McNemar $p\\ge0.085$), against +0.097 for LR($N$) and +0.057 for the PFN; adding three moves LR($N$) +0.29 and the networks −0.003 to −0.071. '
  'The hardest-split response is stable from $\\Delta R=5\\times10^{-4}$ to $5\\times10^{-3}$. Poisson overpredicts the LR($N$) slope by 4.8× because the measured multiplicity widths (13.3 for quarks, 15.7 for gluons) are about twice the Poisson widths (5.8, 7.3). '
  'A $p_T^D$-matched control without a split also changes IRC-safe shapes (EFN +0.50), so the function of the hard-particle energy sharing is not identified.', None),
 ('Checks', 'Unedited AUC: ParT kinpid 0.914 (full test 0.9121 = published), LorentzNet 0.919 (released 0.9161), PFN 0.888, EFN 0.872, LR($N$) 0.846. Every ParT / ParticleNet checkpoint gives the same pattern. On Herwig 7.1 jets the AUC drops to 0.80–0.81 and the split response is unchanged within 0.005. Split jets are separable from real jets (C2ST ≈ 1).', None),
 ('Physical significance', 'The QG taggers do not count particles: soft particles barely matter to them. Their collinear unsafety enters through how the hardest particles share energy, which also depends on hadronisation. Casimir-meets-Poisson overstates the information in one extra particle by about 5× in this sample.', None),
]

X9_EXPLAINER = [
 ('Question', 'Are the elements found in JEDI-linear (pT-weighted radial objects over pT-rank windows; a mass variable that is not recentred on the jet axis) a property of JEDI-linear, or do the standard set architectures of the field, trained on the same inputs, show them too?', None),
 ('Background',
  'An Energy Flow Network computes $L_k=\\sum_i z_i\\,\\Phi_k(\\Delta\\eta_i,\\Delta\\phi_i)$ with $z_i=p_{T,i}/\\sum_j p_{T,j}$ (IRC safe); a Particle Flow Network computes $L_k=\\sum_i \\Phi_k(p_{T,i},\\Delta\\eta_i,\\Delta\\phi_i)$. Both are permutation invariant, so they cannot use a particle\'s pT rank. Both see the angles about the stored jet axis, as JEDI-linear does. '
  'The un-recentred mass variable is $\\tfrac12\\log(M_0M_2)$ with $M_0=\\sum_i p_{T,i}$, $M_2=\\sum_i p_{T,i}r_i^2$; the invariant mass satisfies $m^2\\approx M_0M_2-|M_1|^2$ with $M_1=\\sum_i p_{T,i}(\\Delta\\eta_i,\\Delta\\phi_i)$. A rigid translation by $\\vec\\delta$ leaves $m$ unchanged and raises $M_2$ by $2\\,\\vec\\delta\\cdot\\vec M_1+\\delta^2 M_0$.', None),
 ('Method',
  'EFN ($\\Phi$ 100-100-128) and PFN ($\\Phi$ 64-64-32), $F$ 100-100-100, trained with energyflow on the same 595k jets (64 particles, seeds 0 and 1); $\\Phi$ size chosen on dev accuracy within 1 point of 80.9 % (the EFN reached 74 % at best). An exact numpy forward pass next to JEDI-linear makes every comparison paired on the same jets. '
  'On 10,000 test jets (seed 313): (1) first-layer cuts typed against a shuffled-kernel null; (2) each live latent unit fitted with MARS (≤ 15 terms) on the E11 observable library; (3) 0.06 rad translation vs a rotation of equal displacement, mass read-out calibrated against a real angular scaling of the same predicted size, WTA re-axis, angle swap of ranks 13|14 vs 11|12; (4) interchange interventions per named unit vs permuted alignments and random rotations.', None),
 ('Result',
  'Translation W → Z: JEDI-linear 61–77 %, PFN 51–82 %, EFN 37–77 %; rotation 1.5–2.9 %. Calibrated read-out ratio 0.97–1.30 in all three (0 would be an invariant-mass read-out). '
  'Rank-14 swap ratio: JEDI-linear 5.25 (X6\'s edge units), PFN/EFN 0.79–0.84 (expected from the pT-share gaps: 0.72). PFN pure first-layer fraction 0.11, equal to its null; JEDI-linear 13/13. Latent naming: median MARS $R^2$ JEDI-linear 0.98, PFN 0.85–0.87, EFN 0.73–0.82. '
  'Mean effective IIA: JEDI-linear 0.70, PFN 0.53–0.58, EFN 0.52–0.58, each above permuted alignment ($p = 0.005$) and none above random rotations ($p_{\\rm rot}$ 0.44–0.95).', 'results/jedi/x9_compare.png'),
 ('Checks', 'numpy forward reproduces TensorFlow ($|\\Delta p| < 10^{-5}$); two seeds per model; edits leave padding and $p_T$ untouched; paired bootstrap and McNemar tests, Holm within model; EFN accuracy gap (74 % vs 80.9 %) stated; an EFN with standardised angles reaches the same 74.4 %.', None),
 ('Verification (X9v, fresh sample seed 2027, confidence 0.7)',
  'Added control: translate by 0.06 rad in a random direction, then scale every angle by $s=\\sqrt{M_2/M_2\'}$, so $M_0$ and $M_2$ about the stored axis are unchanged while the jet stays displaced ($\\log m$ falls by 0.305). W → Z: plain translation 39–81 %, $M_2$-held translation 4.0 % (JEDI-linear), 11.1 / 6.9 % (PFN), 9.2 / 5.0 % (EFN), rotation 1.8–2.9 %. Read-out ratio under the $M_2$-held edit: 0.00, −0.18 / −0.13, 0.01 / 0.02 (an invariant-mass reader gives −1). Calls changed: 17–31 % vs 4–9 % under rotation. '
  'Rank swap with JEDI-linear\'s units chosen as the flow models\' (top 10 by cost): 2.96 [2.88, 3.05] vs PFN/EFN 0.84–0.87; at 0|1 vs 3|4 JEDI-linear 1.93, PFN 1.97–2.20. The pure-cut comparison confounds architecture with HGQ quantization; the absence of windows in PFN/EFN holds by construction.', None),
 ('Why it matters',
  'The axis-dependent mass response (claims A4, D4) is measured in three set architectures trained on the same stored-axis inputs and runs mainly through $M_2$ about that axis in all of them (claim A10m); whether it comes from the input representation in general (recentred or augmented training) is not tested. The rank-window response, the pure cut types and the near-exactly radial objects are measured only in JEDI-linear. Interchange agreement of the S formulas is not above that of random rotations of the aligned subspace (claim B2m).', None),
]

X10_EXPLAINER = [
 ('Question', 'Does JEDI-linear hold the mass ingredients $M_0=\\sum_i p_{T,i}$ and $M_2=\\sum_i p_{T,i}r_i^2$ (about the stored axis), their combination $\\Delta=\\tfrac12\\log(M_0M_2)$, and the W−Z score in a few linear directions of a layer? If it does, where does each first appear?', None),
 ('Background',
  'Interchange intervention: take a base and a source jet, copy part of a layer from the source into the base, and run the rest of the network. If that part is the variable $V$, the base jet should then be called as if an input edit had set its $V$ to the source value. Distributed alignment search (DAS; Geiger et al. 2023) learns which directions to copy. X9 had swapped the program\'s named formulas and found them no better than random rotations.', None),
 ('Method',
  'For a layer $h\\in\\mathbb{R}^d$ ($S$, $h_4$, $h_5$) and orthonormal $Q\\in\\mathbb{R}^{d\\times k}$, the interchanged state is $$h\'=\\Pi\\!\\left(h_b+(h_s-h_b)\\,QQ^\\top\\right),$$ with $\\Pi$ the clip to the next layer\'s fixed-point input range (the unsigned quantizers wrap otherwise). $Q$ minimises $$\\mathcal{L}(Q)=-\\sum_{(b,s)}w_{bs}\\,\\log\\operatorname{softmax}\\!\\big(f_{>L}(h\')\\big)_{y^H_{bs}},$$ where $f_{>L}$ is the exact downstream network and $y^H_{bs}$ the network\'s call on the base jet after the input edit do($V=V_s$): $p_T\\to c\\,p_T$ with angles $\\times c^{-1/2}$ for $M_0$; angles scaled for $M_2$; $p_T$ scaled for $\\Delta$. Score: effective IIA, the fraction of call-changing test pairs $E_H$ with $\\arg\\max f_{>L}(h\')=y^H_{bs}$. Samples: fit sample(4000, 0), test sample(4000, 17) minus fit jets (18,410), 5,000 test pairs. Controls: 1,000 random $Q$, permuted-label DAS, untrained networks (seeds 0–2), PFN latent.', None),
 ('Result',
  'Effective IIA (DAS vs random mean): $M_0$ 0.87 at $S$ with $k=1$ (0.10); $M_2$ 0.75 at $S$, 0.78 at $h_4$ (≤ 0.07); W−Z 0.43 at $S$, 0.82 at $h_5$; $\\Delta$ 0.51 at $S$, 0.62 at $h_4$. All 36 cells above random (Holm 0.036). Patching the 2-direction $\\Delta$ subspace at $h_4$ from 0.06 rad translated jets carries 0.73 of the translation\'s mass read-out shift and sends 82 % of W to Z; the $M_0$ subspace carries 0–2 % (a translation does not change $M_0$).', 'results/jedi/x10_das.png'),
 ('Checks', 'Exact fixed-point network throughout (80.79 % on the test jets); the training tail reproduces the exact tail on 100 % of calls; two pipeline faults (float tail, wrap) found and fixed before the reported run; bootstrap over pairs.', None),
 ('Verification (X10v, two passes, confidence 0.45)',
  'Fresh disjoint samples (sample(2500, 41) / sample(2500, 43), 4,000 pairs): all 36 cells within 0.06. Full-swap baseline $\\Pr[y^H=\\hat y_s\\mid y^H\\ne\\hat y_b]$: $M_0$ 0.35, $M_2$ 0.61, $\\Delta$ 0.33, W−Z 0.51; DAS exceeds it by about +0.5 ($M_0$), +0.05 to +0.17 ($M_2$), +0.15 to +0.30 (W−Z at $h_4$/$h_5$). Angle-only $\\Delta$ edit $(\\Delta\\eta,\\Delta\\phi)\\to c\\,(\\Delta\\eta,\\Delta\\phi)$ ($M_0$ fixed, $M_2\\times c^2$): the $\\Delta$ subspace reproduces it on 0.19–0.43 of pairs, below its full swap (0.61) in 9/9 cells; the $M_2$ subspace reproduces it (0.65–0.73), and a subspace trained on it overlaps $Q_{M_2}$ by 0.68–0.99 ($\\|Q_1^\\top Q_2\\|_F^2/k$). Untrained networks beat their random subspaces in 41/72 cells but stay below the full swap in 36/36 $M_0$/$M_2$ cells.', None),
 ('Why it matters', '$M_0$ and $M_2$ about the stored axis are causal variables in low-dimensional subspaces from $S$ on (claim A4c), so X9\'s null came from the formula basis (claim B2m). $\\Delta$ is not one internal variable: its subspace encodes the response to a pT rescale. For causal-abstraction tests on small networks, the full-swap baseline, not random subspaces, separates trained from untrained networks.', None),
]

X11_EXPLAINER = [
 ('Question', 'Are JEDI-linear\'s hidden units individually readable as physics formulas, or would any mixing of the same activations (a random rotation) or a sparse-autoencoder dictionary be as readable or more?', None),
 ('Background',
  'A layer with $d$ live units $h\\in\\mathbb{R}^d$ can be read in any basis $v=Qh$, $Q$ orthogonal, without losing information. A sparse autoencoder learns $f=\\mathrm{ReLU}(W_e(x-b_d)+b_e)$, $\\hat x=W_df+b_d$, minimising $\\lVert x-\\hat x\\rVert^2/d+\\lambda\\sum_k f_k$. If units are privileged, single units should be describable by short formulas more often than rotated directions.', None),
 ('Method',
  'Preregistered rule. For each direction $v_j$ fit $\\hat v_j=c_0+\\sum_{t\\le 8}c_tB_t(\\mathcal O)$, with $B_t$ linear terms, hinges $\\max(0,\\pm(\\mathcal O_k-\\tau))$ and products, on 97 rank-window observables such as $\\sum_{i\\in w}p_{T,i}r_i^2$ and $\\sum_{i\\in w}(r_i-0.2)_+$; $v_j$ is named if $R^2=1-\\mathbb E[(v_j-\\hat v_j)^2]/\\mathrm{Var}(v_j)\\ge 0.9$ on 22,551 held-out jets (fit on 16,666). Edit test: scale $p_T$ or angles by 1.1 in the top term\'s window; pass if the slope of $\\Delta v_j$ on $\\Delta\\hat v_j$ has its 95 % CI in $[0.8,1.2]$. Bases: native, 20 random rotations, PCA, SAE 2×/4×; layers: JEDI-linear $S$ and $h_4$, Deep Sets pooled layer, PFN latent.', None),
 ('Result',
  'Fraction named: JEDI $S$ 0.79 (15/19; rotations 0.32–0.63, SAE 0.16–0.21); $h_4$ 0.55 (rotations 0.20–0.36); Deep Sets 0.29 (rotations 0.00–0.12); PFN 0.10 (rotations 0.07–0.20). SAE − native: −0.58 to −0.63 for JEDI $S$ (Holm 0.028). Edit test: 7 of 19 JEDI $S$ units pass; mismatched formula 0/15.', 'results/jedi/x11_atlas.png'),
 ('Checks', 'Rule frozen before fitting and checked unchanged on the cluster; accuracy 80.87 %; fit and test jets disjoint; azimuthal-rotation control leaves the library unchanged (≤ 2.2e-6 s.d.).', None),
 ('Verification (X11v, two passes, confidence 0.85)',
  'Two fresh samples: JEDI $S$ 0.789 with the same 15 units; 0 of 200 and 0 of 50 new rotations reach it ($p=0.005$, $0.020$; $p=(1+\\#\\{f_{\\rm rot}\\ge f_{\\rm nat}\\})/(n+1)$); $h_4$ 0.564 vs 40 rotations (max 0.364); a generic 380-observable library gives 0.74 vs ≤ 0.58. Among named directions, native and rotated pass the edit test equally (0.47 vs 0.50), so the edit-test excess follows from naming. Untrained networks (weights from $\\mathcal N(\\mu_{\\rm layer},\\sigma_{\\rm layer})$): 0/56, 1/57 (JEDI architecture), 0/46 (Deep Sets) named.', None),
 ('Why it matters', 'Supports naming JEDI-linear\'s object units one by one (claims A11, A11b, A3) against the two standard alternatives. Not unique to JEDI-linear (Deep Sets has a weaker preference); whether the preference is learned or comes from the per-particle ReLU-sum architecture is not decided, because the untrained control is at floor.', None),
]

X12_EXPLAINER = [
 ('Question', 'Are the parts of the JEDI-linear program (pT-only / angle-only first-layer cuts, rank-window masks, the un-recentred mass response) a property of the architecture and its training, or of the one official checkpoint? Which need quantization-aware training?', None),
 ('Background',
  'JEDI-linear is trained with HGQ, which learns a bit width for every weight and per-particle activation; a penalty $\\beta\\cdot\\mathrm{EBOPs}$ pushes bits to zero. A zero-bit weight is a zero; a zero-bit slot is a mask, so in $S_c=\\sum_i u_{ic}$ particles at masked pT ranks drop out (a rank window). The official network has 20 nonzero first-layer weights of 192 and responds to a rigid shift through $M_2=\\sum_i p_{T,i}r_i^2$.', None),
 ('Method',
  'Retrain the same architecture: HGQ-150 seeds 0–2 (300 epochs, primary), HGQ-50 seeds 0–2 (200 epochs), float seeds 0–1 (quantizers = identity, same script). On sample(5000, 0) (25,000 jets): purity of live first-layer units $a_c=\\alpha_{pT}p_T+\\alpha_\\eta\\Delta\\eta+\\alpha_\\phi\\Delta\\phi+\\beta$ (pure if ≥ 90 % of the spread comes from $p_T$ alone or the angles alone) vs a column-permutation null and a sparsity-matched null; zero-bit masks; an angle swap of adjacent pT ranks across a mask edge vs inside, against weight-permuted copies that keep the masks and classify above chance; a 0.06 rad translation vs a random-direction translation and an $M_2$-held translation; single-particle maps of $S$.', None),
 ('Result',
  'Every quantized first layer is 100 % pure, as are its sparsity nulls. Across/inside ratio at the top edge: official 5.28 (13|14); retrained 0.00–1.48 at their own edges (ranks 24–45). Translation W→Z: official 0.76, usable HGQ 0.61–0.78, float 0.21–0.35; $M_2$-held 0.03–0.19 in all.', 'results/jedi/x12_final.png'),
 ('Checks', 'Battery forward equals Jedi.run and the Keras model exactly; parameter count and quantizer-table shapes match the official network; mask-keeping nulls have non-constant output (accuracy 0.30–0.75); Holm over 15. One primary seed collapsed to 57 % at its last epoch and is kept as preregistered.', None),
 ('Verification (X12v, seeds 1 and 2, confidence 0.55)',
  'A shift by $d=0.06$ rad leaves the invariant mass unchanged and raises $M_2$ by about $M_0d^2$; a real widening (angles $\\times s_T$, $s_T=\\sqrt{M_2(T)/M_2}$, mean 1.32) gives the same $(M_0,M_2)$. Ratio $R=\\Delta_{W\\to Z}(T)/\\Delta_{W\\to Z}(\\mathrm{dil}_{M_2})$: official 1.22, HGQ 1.02–1.38, float 0.69–1.30 (network-level $p=0.24$); a reader of $(M_0,M_2)$ only gives $R=1$, a recentred-mass reader $R\\approx0$. Under a real W→Z widening by $m_Z/m_W$, float 0.85 vs HGQ 0.80–0.84 W→Z. At the official 13|14 edge the retrained networks give 0.20–0.90 (official 5.18). Radial-hinge excess with quantization: network-level permutation $p=0.048$ (all usable) / 0.17 (primary).', None),
 ('Why it matters', 'The uncentred mass reading is a property of this architecture and its inputs in both training arms (claims A10m, A1r); the float networks differ only in where the Z class ends in mass. The 13|14 rank-window effect is specific to the official checkpoint or its long training (claim A1). "Pure cuts" reduce to "HGQ makes the first layer sparse" (A1q reworded).', None),
]


H12_EXPLAINER = [
 ('Question', 'Does a tagger read the jet mass as the Lorentz-invariant $m$, or as a combination of $p_T$ and opening angles tied to the $p_T$ range it was trained on? A boost along the jet axis separates the two: it changes $p_T$ and the angles but keeps $m$ fixed.', None),
 ('Physics background',
  'For a boosted jet $m^2\\simeq\\sum_{i<j}p_{T,i}\\,p_{T,j}\\,\\Delta R_{ij}^2$. A boost $B_\\lambda$ along the jet 3-momentum with rapidity $\\zeta=\\operatorname{arsinh}(\\lambda|\\vec P|/m)-\\operatorname{arsinh}(|\\vec P|/m)$ sends $|\\vec P|\\to\\lambda|\\vec P|$, keeps $m$ exactly and scales the opening angles by about $1/\\lambda$. '
  'Two comparison edits: a $p_T$-only scaling $P_\\lambda$ (all four-vectors $\\times\\lambda$, so $m\\to\\lambda m$) and an angle-only scaling $A_\\lambda$ ($\\Delta\\eta,\\Delta\\phi\\to/\\lambda$ at fixed $p_T$, so $m\\to m/\\lambda$). To first order $B_\\lambda=A_\\lambda\\circ P_\\lambda$. '
  'An invariant-mass reader does not respond to $B$; a reader of $p_T$ at fixed angles responds like $P$; a reader of angles at fixed $p_T$ responds like $A$. The axis-moment mass $m_A=\\sqrt{M_0M_2}$ (claim A4) is also unchanged by $B$ (mean change $\\le10^{-4}$).', None),
 ('Method',
  'Samples (seed 301): 20,000 hls4ml jets, 40,000 JetClass jets (random rows), 40,000 Top Landscape and 40,000 quark–gluon test jets. More than 20 models: JEDI-linear and 10 other hls4ml-trained networks (including JEDI-style retrains on $p_T$-normalised inputs and with flat-$p_T$ weights), ParT and ParticleNet on JetClass, ParT, ParticleNet, LorentzNet, PFN and EFN on Top, ParT, ParticleNet and EFN on quark–gluon. '
  '$\\lambda\\in\\{0.8,0.9,1.1,1.25\\}$ (Top and QG also 0.95, 1.05). Controls: a rotation about the jet axis and a random translation, each with the same per-jet displacement as $B$; a weight-permuted JEDI-linear. '
  'Metrics: % calls changed with 95 % bootstrap intervals, exact McNemar tests of $B$ against the rotation with Holm correction, class flows, $\\Delta$log-odds and the fit $\\Delta_B=a\\Delta_P+b\\Delta_A$.', None),
 ('Result',
  'At $\\lambda=0.9$ the hls4ml-trained models change 21–38 % of calls under $B$, against 0.3–4.2 % under the rotation (JEDI-linear 32.5 % [31.8, 33.1] vs 2.6 %; Holm $p\\le3\\times10^{-3}$ for every model and $\\lambda$). '
  'JetClass ParT_full changes 4.5 % under $B$, 6.9 % under $P$ and 14.4 % under $A$; ParticleNet_full 4.3 / 8.1 / 14.5 %. Top ParT at $\\lambda=0.95$: 0.6 / 1.8 / 1.9 %. '
  'JEDI-linear\'s W call falls from 87.3 % to 26.8 % under $B$, to 47.9 % under $P$, to 14.4 % under $A$; 62 % of the calls $B$ changes are W or Z jets becoming top. The classical mass windows and $\\tau_{32}$ cut do not change under $B$.', 'results/physics/h12_boost.png'),
 ('Checks',
  'Unedited accuracies reproduce the references within 0.6 points. An independent implementation of the boost (verifier) gives identical calls. A fresh sample (seed 917) gives JEDI-linear 32.9 % [32.2, 33.7] vs rotation 2.4 %, ParT_full 4.36 / 6.89 / 14.5 %, Top ParT at $\\lambda=0.9$ 0.94 / 3.34 / 4.05 %. '
  'Limits: boosted Top and QG jets are mostly outside the training $p_T$ window (in-window fraction 0.32 and 0 at $\\lambda=0.9$); a classifier separates boosted from natural jets (C2ST AUC 0.69–0.93); absolute rates are not comparable across datasets (the same EFN changes 36.8 % of hls4ml calls and 4.4 % of Top calls).', None),
 ('Verification (H12v) and corrections',
  'Because $B$ keeps $m_A$ fixed as well as $m$, the test cannot tell an invariant-mass reading from an $m_A$ reading. JEDI-linear\'s W/Z → t flow also appears under the $p_T$-only edit (the $\\Sigma p_T$ shortcut of claim D1); at the W/Z boundary its response adds up, $\\Delta_B\\approx\\Delta_P+\\Delta_A$ (+1.45 = −1.36 + 2.81). '
  'The $p_T$-normalised and EFN models can only form $m/p_T$-like quantities, so their $B=A$ follows from the inputs. The Top models were trained in a 550–650 GeV window, as narrow as hls4ml, so the training $p_T$ width does not explain why they respond less.', None),
 ('Physical significance',
  'The hls4ml-trained taggers, JEDI-linear included, call the same jet differently when it is seen at a different $p_T$ with its mass unchanged; for JEDI-linear most of this is the known total-$p_T$ shortcut toward top. The JetClass and Top taggers are much closer to boost invariant. What makes the two groups differ is not yet known.', None),
]

H13_EXPLAINER = [
 ('Question', 'How do jet taggers respond to diffuse, pileup-like soft contamination, compared with ungroomed and groomed (soft-drop) mass taggers? Does the small jet-axis shift it causes matter? Do the networks react to the mass the soft particles add or to how many there are?', None),
 ('Physics background',
  'Pileup adds soft particles roughly uniformly in $\\eta$–$\\phi$ with transverse-momentum density $\\rho$ per unit area. For a jet of transverse momentum $p_{T,J}$ and radius $R$ the small-angle mass shift is $\\Delta m^2\\approx p_{T,J}\\sum_i q_i\\Delta R_i^2=p_{T,J}\\,\\rho\\,\\pi R^4/2$, so the ungroomed mass rises quickly with $\\rho$. '
  'Soft drop (C/A declustering, $z_{\\rm cut}=0.1$, $\\beta=0$) removes soft wide-angle branches, so $m_{SD}$ changes little. A soft particle near the axis ($r\\to0$) adds almost no mass but still adds one constituent. The E-scheme axis moves by about $\\sum_i q_i\\vec r_i/p_{T,J}$.', None),
 ('Method',
  'Edit: add $\\mathrm{Poisson}(\\rho\\pi R^2/\\langle q\\rangle)$ massless particles uniformly in the $R=0.8$ disc, $\\rho\\in\\{5,10,20,40\\}$ GeV per unit area, with $q$ drawn from the softest-quartile constituents; recompute the axis and all inputs. Samples: 10,000 jets each from hls4ml, JetClass and Top Landscape (+100k QCD for rejection). '
  'Controls: a SoftKiller-style cut; the same $\\sum q$ on the axis; the original jet translated by the induced axis offset; the edit about the stored axis; nulls; LorentzNet and an IRC-safe EFN. Verifier content controls: (c4) the same particles at $r/20$ (same count and $\\sum p_T$, $\\Delta m\\approx0$); (c5) one particle with the same kept $\\sum p_T$ and invariant mass; (c2k) that particle on the axis. '
  'Metrics: signal efficiency, % calls changed, $1/\\varepsilon_B$ at $\\varepsilon_S=0.3,0.5$; McNemar tests with Holm correction (64 tests in H13, 240 in H13v).', None),
 ('Result',
  'hls4ml true-W efficiency at $\\rho=5$: JEDI-linear 87 → 42 %, ungroomed window $W(m)$ 93 → 36 %, soft-drop window $W(m_{SD})$ 77 → 78 %. JetClass at $\\rho=40$: ParT_full 80 → 26 %, $W(m)$ 75 → 0 %, $W(m_{SD})$ 72 → 61 %. Top ParT $1/\\varepsilon_B$ at $\\varepsilon_S=0.3$ falls by ×0.41 [0.31, 0.52] at $\\rho=40$. '
  'The induced axis offset is 2.5–13 mrad (median), against 40–95 mrad between WTA and E-scheme axes; translating the jet by it changes 0–3.3 % of calls. '
  'Content controls at $\\rho=5$ (edit / c4 / c5): JEDI-linear W efficiency 43.3 / 87.5 / 39.0 %; ParT_full 72.0 / 64.2 / 76.8 %; Top ParT retention 0.71 / 0.71 / 0.92.', 'results/physics/h13_panel.png'),
 ('Checks',
  'Unedited accuracies reproduce earlier values (JEDI-linear 0.808 / 0.811, ParT_full 0.858 / 0.856, Top ParT AUC 0.987) on two samples. Nulls change ≤ 0.3 % of calls. Recomputing vs freezing the axis changes the call rate by ≤ 0.8 points. The c5 mass match is exact to $10^{-12}$ GeV. '
  'Limits: toy pileup (uniform disc, massless particles), no network trained with pileup; the $r/20$ control also moves soft particles relative to the hard prongs; at a matched Top working point ParT and EFN keep more tags than $\\mathrm{top}(m,\\tau_{32})$ at $\\rho=40$.', None),
 ('Physical significance',
  'No network tested behaves like a groomed observable under soft contamination. JEDI-linear loses efficiency through the mass the soft particles add, like an ungroomed mass window. The particle-ID JetClass networks and the Top Landscape ParT, LorentzNet and PFN lose it mainly through the number of added particles, which no mass-based tagger does. The axis-convention dependence of claim D4 involves offsets about ten times larger than pileup produces.', None),
]

H14_EXPLAINER = [
 ('Question', 'Do quark–gluon taggers trained on one event generator (Pythia) count particles when tested on jets from another (Herwig 7.1)? Which property of the hardest particles makes them respond to a collinear split?', None),
 ('Physics background',
  'Gluons radiate more than quarks (colour factors $C_A=3$ vs $C_F=4/3$), so gluon jets have more particles and share momentum more evenly. Multiplicity $N$ is IRC unsafe and its modelling differs between generators ($\\langle N\\rangle_g$ 53.1 in Pythia vs 48.7 in Herwig). '
  '$p_T^D=\\sqrt{\\sum_i z_i^2}$ with $z_i=p_{T,i}/\\sum_j p_{T,j}$ is also IRC unsafe and large for quark jets; $z_1$ is the leading-particle fraction.', None),
 ('Method',
  '40,000 Pythia and 40,000 Herwig test jets (seed 9311). Logit $\\ell=\\log p(g)-\\log p(q)$. Count-only reference $\\mathrm{LR}(N)=\\log\\mathrm{NB}_g(N)-\\log\\mathrm{NB}_q(N)$ from negative-binomial fits per generator; an IRC-safe-plus-count reference (gradient-boosted trees on EFPs with $d\\le5$ and $N$). '
  'Edits: add one soft particle at a realistic position (C2ST 0.52), remove the softest, split the hardest collinearly, the same split with a merge that keeps $N$ ($\\Delta N=0$), $k$ random splits. Hard-core arm: change $p_T^D$ at fixed $z_1$, $z_1$ at fixed $p_T^D$, local $p_T^D$ (particles 3–4 only), and the leading-pair $\\Delta R$, all at fixed $N$ and positions; coefficient $\\beta=\\sum_j(\\Delta\\ell^{\\uparrow}_j-\\Delta\\ell^{\\downarrow}_j)/\\sum_j(\\Delta x^{\\uparrow}_j-\\Delta x^{\\downarrow}_j)$. Holm over 766 tests.', None),
 ('Result',
  'AUC Pythia → Herwig: ParT_kin 0.901 → 0.808, LorentzNet 0.918 → 0.800, $\\mathrm{LR}(N)$ 0.840 → 0.786. Adding one soft particle: networks −0.016 to −0.002 on Herwig (−0.024 to −0.004 on Pythia), $\\mathrm{LR}(N)$ +0.089 (+0.107). '
  'Hardest split minus $\\Delta N=0$ control: networks −0.019 to +0.011, EFP+$N$ trees +0.058 to +0.084. $z_1$ at fixed $p_T^D$: ≤ 0.006 per +0.01. Local $p_T^D$ predicts a split response of +0.04 to +0.15 against a measured +0.08 to +0.22, but the IRC-safe EFN responds to the same edit (−0.082 per +0.01).', 'results/physics/h14_generator.png'),
 ('Checks',
  'Each unedited model reproduces its test accuracy (ParT_kin 0.827). The ported EFN matches its own test metrics. A permuted-weight ParT is the null. The split and pair-$\\Delta R\\to0$ edits are off-manifold (C2ST 1.0); the local $p_T^D$ arm has 262 jets. Not yet checked by a verifier; Herwig-trained ParT and ParticleNet are still training.', 'results/physics/h14_hardcore.png'),
 ('Physical significance',
  'The networks\' insensitivity to an extra soft particle does not depend on the test generator, so a generator-dependent multiplicity is not what they use. Their response to a collinear split comes from how momentum is shared among the hardest particles; the leading fraction alone is not it, and $p_T^D$ is not separated from the angular energy flow. Whether a model trained on Herwig counts particles is not yet measured.', None),
]

PAPER_EXPLAINERS = {
    'a': [('How to read this paper: JEDI-linear and what the letters (a, P, g, s, u, S, h4, h5, z) mean', ARCH_EXPLAINER, True),
          ('Result 1 explained: the first layer is 13 cuts, and summing them gives objects (claim A1)', A1_EXPLAINER, False),
          ('Result 2 explained: the context round and the second-round objects (claims A2, A3)', A23_EXPLAINER, False),
          ('Result 3 explained: the network\'s mass variable (claim A4; same test on ParT in Paper C, claim C3; axis consequence in Paper D, claim D4)', MASS_EXPLAINER, False),
          ('Result 4 explained: variables appear over the layers in order of complexity (claim A9)', A9_EXPLAINER, False),
          ('Result 5 explained: small comparison models trained on the same inputs (claims A8/A10)', SMALL_EXPLAINER, False),
          ('Result 6 explained: the neuron atlas, one card per unit (claim A11)', ATLAS_EXPLAINER, False),
          ('Result 7 explained: how round two turns half-planes into near-radial objects (claims A3, A4; E14)', E14_EXPLAINER, False),
          ('Result 8 explained: classes as window rules on the class neurons, imposed (claim A6; E8)', E8_EXPLAINER, False),
          ('Result 9 explained: the pT-rank windows act causally (claims A1, A3; E20)', E20_EXPLAINER, False),
          ('Result 10 explained: the class neurons as formulas on the objects (claim A5; E7; cross-reference Paper B, B3)', E7_EXPLAINER, False),
          ('Result 11 explained: is the mass a product of the moments? (claims A4, A9; E21)', E21_EXPLAINER, False),
          ('Result 12 explained: the same tests on other official JEDI-linear checkpoints (claims A1, A3, A4; X6)', X6_EXPLAINER, False),
          ('Result 13 explained: interchange interventions on the if-statement program (claim A5; E22; cross-reference Paper B)', E22_EXPLAINER, False),
          ('Result 14 explained: naturally off-centre jets and recentring with the mass fixed (claims A4, D4; E23)', E23_EXPLAINER, False),
          ('Result 15 explained: do the edit responses need the trained weights? Functional nulls and resample ablation (claims A1, A2, A3, A4, D4; X8)', X8_EXPLAINER, False),
          ('Result 16 explained: Deep Sets trained float, with L1 and with HGQ (claims A1, A10; X5)', X5_EXPLAINER, False),
          ('Result 17 explained: EFN and PFN on the same inputs (claims A4, A10, B2m; X9)', X9_EXPLAINER, False),
          ('Result 18 explained: learned directions for the mass moments inside the network (claims A4c, A4, B2m; X10, X10v)', X10_EXPLAINER, False),
          ('Result 19 explained: are the units a preferred basis? Rotations and sparse autoencoders (claims A11b, A11, A3; X11, X11v)', X11_EXPLAINER, False),
          ('Result 20 explained: JEDI-linear retrained with and without quantization (claims A1r, A1, A10; X12, X12v)', X12_EXPLAINER, False)],
    'b': [('Result explained: the whole network as if-statements (claim B2)', B2_EXPLAINER, True),
          ('Result explained: a fit on hand observables vs a fit on the network\'s objects under edits (claim B3; E7, explained in Paper A result 10)', E7_EXPLAINER, False),
          ('Result explained: interchange interventions, per variable (claim B2; E22)', E22_EXPLAINER, False),
          ('Result explained: learned subspaces instead of named formulas (claim B2m; X10, explained in Paper A result 18)', X10_EXPLAINER, False),
          ('Result explained: the if-statement program vs standard surrogates of the same size (claims B3, B4; B4)', B4_EXPLAINER, False),
          ('Result explained: a 50-term formula on axis-relative vs invariant observables under shift and re-axis edits (claims B3, B5c; B5)', B5_EXPLAINER, False)],
    'c': [('Result explained: ParT\'s mass comes mostly through the axis-relative inputs, not the pair features (claim C3)', C3_EXPLAINER, True),
          ('Result explained: the 87 GeV switch head and the route of the mass into the W/Z neuron; the literal circuit kept alone (claims C2a, C2b; P3, P3v)', P3_EXPLAINER, False),
          ('Result explained: ParticleNet shares the axis-relative mass response through ΔR (claim C5; P4, P4v)', P4_EXPLAINER, False),
          ('Result explained: L-GATr as the contrast model, training and symmetry checks (claim C6; P5)', P5_EXPLAINER, False),
          ('Result explained: where ParT forms its axis-relative mass; a direction that follows the axis-moment mass from the embedding onward (claim C7; P8, P8v)', P8_EXPLAINER, False),
          ('Result explained: circuit size for ParT_full W vs Z, whole-network search (claim C8; P9, P9v)', P9_EXPLAINER, False),
          ('Result explained: ParT retrained with three seeds on 2M jets, interim axis battery (claim C3s; P10)', P10_EXPLAINER, False),
          ('Result explained: learned sparse features (transcoders, SAEs) as circuit units in ParT_full; feature circuits kept alone and m_A-feature steering (claims C8, C7; P13, P13v)', P13_EXPLAINER, False),
          ('Result explained: ParT trained on planted mass labels; released-model control (claims C10, C3; P14, setup)', P14_EXPLAINER, False),
          ('Result explained: inside ParticleNet: one-direction search per layer with a mirror-sign test, neighbour graphs, channel circuits, LRP and probe baselines (claims C9, C5; P15, P15v)', P15_EXPLAINER, False),
          ('Earlier results in brief (claims C1, C2)', EARLIER_C, False)],
    'd': [('Result explained: the taggers depend on the jet-axis convention (claim D4)', D4_EXPLAINER, True),
          ('Result explained: the axis-convention response in both directions, with invariant and rotation controls (claims D4, A12; X7)', X7_EXPLAINER, False),
          ('Result explained: taggers respond to collinear splits and soft particles, which IRC-safe observables do not (claim D5; H1, H1v)', H1_EXPLAINER, False),
          ('Result explained: the earlier physics edits at 10× statistics with matched controls (claims D1, D2, D3; H3, H3v)', H3_EXPLAINER, False),
          ('Result explained: classical references for top tagging and quark–gluon (claims D6, D7; H2)', H2_EXPLAINER, False),
          ('Result explained: EFP/ADO-chosen IRC-safe surrogates vs the networks under axis, translation, boost and split edits (claims D8, D4, D4m, D5; H8, H8v)', H8_EXPLAINER, False),
          ('Result explained: top taggers and the W-pair mass inside top, with positive and side-effect controls (claims D6, D1; H4, H4v)', H4_EXPLAINER, False),
          ('Result explained: quark–gluon taggers do not count particles (claims D7, D5; H5, H5v)', H5_EXPLAINER, False),
          ('Result explained: a boost along the jet axis at fixed invariant mass vs pT-only and angle-only scaling (claims D9, D1; H12, H12v)', H12_EXPLAINER, False),
          ('Result explained: pileup-like soft contamination vs ungroomed and soft-drop mass; mass or particle count (claims D10, D5, D4; H13, H13v)', H13_EXPLAINER, False),
          ('Result explained: quark–gluon taggers on Herwig jets and the hard-particle function under a split (claims D7, D5; H14)', H14_EXPLAINER, False),
          ('Earlier results in brief (claims D1, D2)', EARLIER_D, False)],
}
