"""Plain-language conclusions for every result on the results page (read by site/gen.py).

CLAIM: claim id in content.py -> conclusion.  EXPLAINER: variable name in explain.py -> conclusion.
KEY_RESULT: title in explain.KEY_RESULTS -> conclusion.  Where an older statement and a later verifier
correction disagree, the conclusion follows the later correction (PAPERS.md, EXPERIMENTS.md, 2026-10-17).
"""

CLAIM = {
    'A1': (
        "Every first-layer unit of JEDI-linear that affects decisions is a cut on one particle that uses either its pT "
        "or its direction, never both, and the layer's sum over particles equals exactly the number of passing particles "
        "times an offset plus the summed inputs of those particles. Each sum also covers only a fixed range of pT ranks "
        "(a rank window, set by entries that the fixed-point number format stores with zero bits): swapping the directions "
        "of two neighbouring particles across a window edge changes the downstream sums, the same swap inside the window "
        "barely does, and shifting the window moves the response with it. So the first half of this network is exactly a "
        "set of cuts, counts and sums over pT-ranked particles. Not shown in general: in other official checkpoints the "
        "pT-or-direction split is about what random placement of the few nonzero weights gives, the window effect appears "
        "only at the hardest-particle edge, and a variant without rank windows is as accurate. The window-edge response "
        "also appears when the trained weights are rearranged but the zero-bit masks are kept, so it shows where the "
        "masks put the window, not that the trained weights use it beyond the masks."
    ),
    'A2': (
        "Before the second round of cuts the network computes jet-level numbers from the first-round sums, mainly the "
        "total pT, and adds them to every particle's input. Replacing these numbers by their average changes about 15 % "
        "of the decisions, and almost all of that effect passes through four second-round sums (one is the summed pT of "
        "the hardest particles), while random sets of other sums of the same size carry almost none. So the second-round "
        "cuts depend on the jet as a whole, mainly through its total pT. The set is not unique: one of its members can be "
        "replaced by any of several other sums with the same effect."
    ),
    'A3': (
        "Each second-round sum adds up, over a fixed range of pT ranks, a function of a particle's pT and its distance r "
        "from the jet axis; adding the particle's direction improves the fits by almost nothing. Under every rotation and "
        "mirror reflection of the jet about its axis, which keeps every r fixed, only five of these sums stay within a "
        "tenth of their normal spread; the others move more, and a few depend on direction by construction. The largest, "
        "S29 (a sum of a hinge in ln pT and r that tracks the pT-weighted squared distance from the axis), changes by the "
        "amount its fitted hinge predicts when particles are moved outward, while its shorter physics name is only a "
        "correlated approximation. So round two turns the first round's directional cuts into near-radial quantities, not "
        "exactly radial ones: their on-regions are polygons around the axis, not circles."
    ),
    'A4': (
        "JEDI-linear's internal mass behaves like the total pT times the pT-weighted squared spread of the particles about "
        "the given axis, without the term that corrects for the axis being off the jet's centre. Shifting every particle "
        "by the same small angle keeps the invariant mass fixed but raises this quantity; the network's mass read-out (a "
        "linear map from its internal units to log mass, fitted on ordinary jets) rises by about the predicted amount for "
        "large shifts and by less for small ones, and 77 % of W jets become Z calls, in four official checkpoints. On "
        "training jets the axis is the centre and the two definitions agree, so the difference appears only when the axis "
        "is moved. Not established: part of the response depends on the shift direction, and whether, and in which layer, "
        "the total pT and the spread are multiplied is open; a gradient test shows a product-like dependence at the class "
        "neurons only."
    ),
    'A4n': (
        "On real, unedited jets whose axis is slightly off the pT-weighted centre, moving the particles so that the jet is "
        "centred keeps the invariant mass fixed but lowers JEDI-linear's mass read-out, and moving them the opposite way or "
        "sideways raises it, as the axis-relative mass predicts in sign. So the axis dependence is not limited to the "
        "artificial off-centre jets made by the large shift. The size does not match one formula: the part that grows with "
        "the square of the shift enters below the prediction and the part linear in the centre offset above it. These "
        "natural offsets are a few milliradians, and at that size the changes in the network's decisions follow how far "
        "the particles move, not the predicted mass change."
    ),
    'A5': (
        "This is partly supported. The class neurons (the units just before the final linear layer) can be approximated by "
        "hinge formulas (sums of terms of the form 'if X > t, add a(X − t)') on the second-round sums: substituting all of "
        "them changes about 10 % of decisions, and under input edits the formulas move in the same direction as the "
        "network's neurons but by roughly half as much. In interchange interventions (setting one internal value to its "
        "value on another jet, in both the network and the formulas) the formulas give the network's new decision in most "
        "cases, but do better than a guess that ignores the swap only at the class neurons. So the class neurons are "
        "approximately, not exactly, functions of the second-round sums."
    ),
    'A6': (
        "For each class, a rule of a few conditions on the class neurons (for W: the W-mass neuron on, plus conditions on "
        "two others) picks out the jets the network calls that class with high precision. Setting the rule's neurons to "
        "typical in-class values moves 26–62 % of other jets into the class, but moving them only just inside the "
        "conditions moves far fewer, and rules of similar quality also exist after randomly rotating the neuron axes. So "
        "the rules summarise the decisions, while the cause is how far the neurons move the class score, because the final "
        "layer adds each neuron's contribution; the class is not a logical AND of windows. For some classes, another set of "
        "neurons has a larger effect than the rule's."
    ),
    'A9': (
        "Linear read-outs (a weighted sum of a layer's units fitted to a quantity) show that sums over particles become "
        "readable at the pooling layers, products and ratios of sums such as mass and girth at the same or slightly later "
        "stages, and three-particle or clustering quantities such as D2 and τ21 mostly never reach a good fit and are best "
        "read in the first dense layer. The rank correlation between a quantity's algebraic depth and the first layer where "
        "it becomes readable is moderate (0.51). This shows where quantities can be read, not that the network uses them: "
        "edits of particle count and girth move the read-outs as expected, but an edit of D2 at fixed mass does not, and the "
        "layer where the mass product forms has not been located."
    ),
    'A11': (
        "Inspecting each of JEDI-linear's active units one by one, 78 % are described by one physical quantity (one main "
        "job, or the same quantity over two jet populations, usually lower and higher mass); the rest have unrelated jobs "
        "or no description that fits well, and these sit mostly in the dense layers and the class neurons. This supports "
        "naming most units with physical formulas. Steering (setting a unit to a high value on jets where it is off) changes "
        "as many decisions for a random unit as for the described one, except for the class neurons, so for most units the "
        "description is not causally tested. The tally depends on chosen thresholds and has not been rerun independently."
    ),
    'A12': (
        "Deep Sets and Linformer models retrained with every particle measured from the hardest particle (so the axis is "
        "never at the centre in training) do not follow the axis-relative mass as the original networks do: when the axis "
        "jumps to the other prong, their mass reading moves only a fraction of a matched real mass change, with a sign that "
        "differs between seeds. A later reverse test shows they read the axis-relative mass about their own axis together "
        "with a second spread that does not change under a shift, and their decisions still move under axis changes. So "
        "the mass formula a network learns depends on the axis convention of its training data, but training off-centre "
        "does not produce an invariant-mass network. Confidence is low: two seeds per model with opposite signs, and the "
        "jump test uses a narrow set of jets."
    ),
    'A8/A10': (
        "Small taggers trained on the same inputs (Deep Sets, Linformer, plain transformers) all have an axis-relative mass: "
        "a shift of the jet that keeps its invariant mass changes their mass read-out and their decisions, and the "
        "invariant-mass hypothesis is rejected in all but one case. Variables also become readable in order of algebraic "
        "complexity in every model, but an untrained Deep Sets already reads many of them at its sum layers, so part of "
        "the ordering comes from the architecture. Not general: the size of the mass response varies widely between "
        "architectures and even between two seeds of one model, and JEDI-linear's sparse first layer of pure pT or pure "
        "direction cuts does not appear in Deep Sets, whose first-layer units mostly mix the two. ParT has been compared "
        "only on the mass result. Later tests (2026-10-19) add Deep Sets trained with L1 or fixed-point training, a "
        "Particle Flow Network and an Energy Flow Network: all respond to the shift, through the pT-weighted spread about "
        "the axis, while networks without trained function do not. Rank windows, pure cuts above chance and nearly exact "
        "radial objects were found only in JEDI-linear."
    ),
    'B1': (
        "The identity that a ReLU unit summed over particles equals the number of passing particles times its bias plus its "
        "weights times their summed inputs holds with zero error in floating point, checked on all 260,000 test jets. The "
        "only difference from the network's stored values is the fixed-point rounding applied per particle slot, which an "
        "untrained network with the same number formats shows equally. So this part of the if-statement method needs no "
        "fitting: it is exact for any layer that applies ReLU units per particle and then sums them."
    ),
    'B2': (
        "The whole network, written as a nested program of hinge if-statements ('if Q > t, add a(Q − t)'; each layer "
        "fitted on the previous layer's fitted values, starting from physical quantities computed from the particles), "
        "makes the same decision as the network on 94 % of fresh jets, its accuracy is not distinguishable from the "
        "network's, and no network value enters it. Under the shift that keeps the invariant mass fixed it also sends W "
        "jets to Z as the network does, but it reproduces only part of the network's changed decisions, and interchange "
        "interventions (swapping one internal value between two jets in both the network and the program) show its "
        "variables match the network's causal variables only partly. So it reproduces the network's decisions; it is not "
        "shown to reach them through the same internal variables. An accuracy lead reported first did not hold on fresh "
        "jets, and the background-rejection comparison is not resolved with the available counts."
    ),
    'B3': (
        "Agreeing with the network on ordinary jets does not show that a surrogate computes the same thing. A formula "
        "fitted directly from physics observables to the class neurons agrees on most ordinary jets but sends almost no "
        "shifted W jets to Z, because it uses the invariant mass, while the nested program follows the network; programs "
        "of equal size on axis-relative and on shift-invariant observables differ by about one point on ordinary jets and "
        "by more than half of all decisions under the shift. The direct formula's failure depends on its inputs: without "
        "the invariant mass among them it does follow the shift, and no fitted program reproduces the network's dependence "
        "on the shift direction. So the agreement rate on ordinary jets cannot separate mechanisms; input edits that "
        "separate candidate mechanisms can."
    ),
    'B5c': (
        "This was not supported. No hinge program of up to 50 terms on quantities measured about the jet axis reproduces "
        "how JEDI-linear's decisions change under the jet shift: programs of 20–50 terms bracket the network's overall "
        "rate of changed decisions but send too few W jets to Z, and none depends on the direction of the shift as the "
        "network does. Programs of equal size on shift-invariant quantities match ordinary decisions about as well and "
        "change no decision under the shift, so the edits separate the two kinds of program. A compact formula for the "
        "network's mass and axis behaviour was not found with this fitter; whether direction-dependent inputs would help "
        "is open."
    ),
    'B4': (
        "This was not supported as a compactness claim. Standard surrogates fitted on the same jets, MARS (forward-selected "
        "hinge terms) fitted directly from physics inputs to the class scores and gradient-boosted trees, reach the "
        "agreement of the small nested programs with many fewer fitted pieces, and the largest nested programs are within "
        "about one point of them. Under the jet shift the nested programs agree with the network somewhat more often than "
        "the best boosted trees, but they lead by a similar margin under a random displacement that raises the axis spread "
        "by the same amount, so the lead is not specific to the shift mechanism. The surrogate the shift separates is the "
        "direct physics-to-score MARS fit, which uses the invariant mass and misses most of the network's changed decisions; "
        "symbolic regression was not run."
    ),
    'C1': (
        "In ParT, the fixed pair bias (a learned function of each particle pair's geometry: ln kT, ln z, ln ΔR, ln m²) "
        "explains most of the variance of the attention scores. ParT run with attention from the pair bias alone keeps "
        "much more accuracy than with attention from the learned content term alone, and replacing the bias by a fitted "
        "additive hinge formula costs under half a point of accuracy. So most of ParT's routing of information between "
        "particles is set by a fixed formula of pair geometry. These are earlier results without an independent rerun or "
        "intervals."
    ),
    'C2': (
        "Under a 10 % scaling of all angles (a 10 % mass increase), nearly all of the change of ParT's W/Z class-token "
        "neuron comes through changes in where the attention heads look, and almost none through the per-particle feature "
        "path. One attention head attends to the class token for lighter jets and rarely for heavier ones, switching "
        "between the W and Z mass ranges. This is an earlier result without a rerun. The split here is inside the network "
        "(attention versus per-particle feature updates), which is not the same as the input-route split of C3, where the "
        "per-particle inputs measured from the axis carry most of the mass response; how the two fit together has not been "
        "tested."
    ),
    'C3': (
        "ParT, which also receives shift-invariant pair features from which the invariant mass could be computed, still "
        "responds to a shift of all particles that keeps the invariant mass fixed: more of its W jets become Z calls, and "
        "its internal mass read-out rises by somewhat less than the axis-relative formula predicts. In ParT_full (ParT "
        "with all input features), feeding the unshifted per-particle distance to the axis removes almost all of this "
        "response, and under real mass changes the per-particle inputs carry most of the change while the pair features "
        "carry about a tenth to a fifth of the W-to-Z change. So ParT's mass sense comes mainly from particle positions "
        "measured from the axis, as in JEDI-linear, with a small nonzero contribution from the pair features. For mass "
        "decreases the two routes do not add, so a share is not defined there."
    ),
    'C4': (
        "This is open: the test of whether quantities become readable in ParT's blocks in order of algebraic depth has not "
        "been run."
    ),
    'D1': (
        "In earlier physics edits on JEDI-linear, changing only the mass of W jets (by scaling angles) to the Z mass turns "
        "almost all of them into Z calls, lowering the jet momentum by 10 % turns most W jets into top calls, and removing "
        "the W inside a top jet barely changes the top call. So JEDI-linear tags W, Z and top mainly by mass windows plus "
        "the total pT, and does not check that a top jet contains a W. The momentum-scaling result was reproduced at 10 "
        "times the statistics on two fresh samples (W call 87 % → 27 %, rotation control unchanged). With a control that "
        "holds the W pair fixed, neither JEDI-linear nor ParT responds to the W-pair mass inside top, though ParT's top "
        "call drops when the subjet-pair masses are rearranged in general."
    ),
    'D2': (
        "In earlier physics edits on ParT_full (400 jets per class), making all tracks prompt removes most H → bb̄ calls "
        "and shrinking all displacements keeps most of them, but keeping only two displaced tracks removes many, so ParT "
        "responds to the number of displaced tracks more than to how far they travel. Removing lepton isolation lowers the "
        "leptonic-top calls, and edits of π⁰ → γγ pairing and of the heavy-quark dead cone have no measurable effect. At 10 "
        "times the statistics the same signs return; a sharper test caps every track's displacement significance just "
        "above threshold, which keeps the number of displaced tracks and costs 3 points, while keeping only two displaced "
        "tracks costs 61. Small new effects (lepton charge in H → ℓνqq′, charm dead cone) were not rerun."
    ),
    'D4': (
        "Measuring particle positions from another standard jet axis (the winner-take-all axis, which follows the harder "
        "branch when the jet is reclustered, or the hardest particle) instead of the summed-momentum axis changes about 30 % "
        "of JEDI-linear's decisions and lowers its accuracy; ParT with all input features changes fewer, ParT with "
        "kinematic inputs only about as many. Most changed decisions are W jets called Z, and the size of the internal mass "
        "change matches the axis-relative formula. So the taggers depend on the axis convention, which the invariant mass "
        "does not. Because they were trained with one axis, any other axis is also a change of input distribution: the "
        "formula predicts the direction of the change, and some changes also occur under a shift of equal size that keeps "
        "the formula fixed."
    ),
    'D4m': (
        "When the axis convention changes, the W and Z decisions move in the direction each model's own axis-relative mass "
        "predicts: models trained on the centred axis turn W into Z, and models trained about the hardest particle, "
        "evaluated on the centred axis, turn Z into W. A control translation of the same size that keeps the formula's "
        "value fixed still changes about a third as many decisions in centred-trained models, but with no W-to-Z flow; so "
        "the direction and most of the number of changes follow the formula, and the rest is general sensitivity to moved "
        "inputs. For hardest-trained models the same control moves many decisions toward Z, so they also read a second "
        "spread that does not change under a shift. Confidence is limited, and a rotation about the axis turned out not to "
        "be a valid control for these near-radial models."
    ),
    'D3': (
        "With 10 times more jets, random JetClass rows, the corrected η build and a matched control for each edit, the "
        "earlier physics-edit results return with the same sign and mostly within the old intervals: for example, "
        "JEDI-linear's W call falls from 87 % to 27 % when the jet momentum is scaled by 0.9, while a rotation of equal "
        "size does nothing. A new control showed that one of the earlier controls (for the W mass inside top) also moved "
        "the quantity it was meant to hold fixed. Several smaller effects (lepton charge, charm dead cone, π⁰) were not "
        "rerun by the verifier."
    ),
    'D5': (
        "Every trained tagger tested (JEDI-linear, Deep Sets, Linformer, a transformer, ParT, ParticleNet) changes 1–5 % "
        "of its decisions when its hardest particle is replaced by two exactly collinear pieces, while an energy-flow "
        "network, which is infrared and collinear (IRC) safe by construction, changes none on jets that keep all their "
        "particles. Splitting more particles, or removing the softest particle, moves the decisions along the particle "
        "count, so the taggers use multiplicity, which depends on hadronisation and detector granularity. A very soft added "
        "particle also changes decisions, but that particle is softer than any in the data and its effect depends on where "
        "it is placed, so that test alone does not show counting."
    ),
    'D6': (
        "This is set up but not yet tested on a network: for a classical top tagger, adding the mass of the best W-candidate "
        "subjet pair to the jet mass and the three-prong ratio τ32 raises the QCD rejection by about 28 %, so the W mass "
        "carries usable information. Whether ParT, ParticleNet and the energy-flow networks use it will be tested once "
        "their training on the Top Tagging Landscape finishes."
    ),
    'D7': (
        "This is set up but not yet tested on a network: in the quark–gluon sample, gluon jets carry 1.59 times as many "
        "particles as quark jets (the leading-order colour-factor expectation is 9/4), and the particle count alone "
        "separates them with AUC 0.84. Whether trained quark–gluon taggers count particles will be tested once their "
        "training finishes."
    ),
}

EXPLAINER = {
    'H1_EXPLAINER': (
        "Replacing one particle by two exactly collinear pieces leaves every IRC-safe jet shape unchanged and leaves an "
        "IRC-safe energy-flow network's decisions unchanged, but changes 1–5 % of the decisions of every trained tagger "
        "tested, more than a matched edit that keeps the particle count. Removing a real soft particle and adding back an "
        "extremely soft one restores many of the changed decisions, so the taggers count particles. The extremely soft "
        "particles used in the infrared test are below anything in the data, and their effect depends on position."
    ),
    'H2_EXPLAINER': (
        "On the full Top Tagging Landscape test set, the best classical tagger built here (soft-drop mass, τ32 and the "
        "W-candidate pair mass) rejects QCD about 5 times less well than published ParT, and the W-pair mass adds about 28 % "
        "to the rejection of mass and τ32. In quark–gluon jets the particle count is the strongest single observable, and "
        "gluons carry 1.59 times as many particles as quarks. The trained networks needed for the edits are still training."
    ),
    'H3_EXPLAINER': (
        "The earlier physics edits return with the same sign at 10 times the statistics: JEDI-linear's W call collapses "
        "under a 10 % momentum scaling that leaves every jet shape unchanged, and ParT's H → bb call follows the number of "
        "significantly displaced tracks, not how far they travel. With a control that keeps the W pair fixed, no network "
        "responds to the W-pair mass inside top; whether ParT reads the three subjet-pair masses as a set is open."
    ),
    'A1_EXPLAINER': (
        "JEDI-linear's first layer reduces to cuts on single particles: a few select soft particles by pT, the rest select "
        "particles on one side of a line through the jet axis, and none mixes the two. Summed over particles, each cut "
        "gives exactly the count of passing particles and their summed pT and positions, with zero error before the "
        "fixed-point rounding. So the input half of the network can be written as exact formulas. The directional cuts "
        "count particles without pT weighting, unlike standard physics observables, which weight by pT."
    ),
    'A23_EXPLAINER': (
        "The network adds jet-level context, mostly the total pT, to every particle before the second round, and the "
        "decision needs it: averaging the context changes about 15 % of decisions, almost all through a few second-round "
        "sums. Each second-round sum adds up a cut in a particle's pT and distance from the axis over a fixed pT-rank "
        "range, and moving particles outward changes the largest one (S29, which tracks the pT-weighted squared distance "
        "from the axis) by the amount its fitted cut predicts. Rotating the jet about its axis changes a few percent of "
        "decisions and moves some sums, so these quantities are close to, not exactly, functions of distance alone: five "
        "sums stay below a tenth of their spread under every rotation and reflection, ten under the 180° rotation only."
    ),
    'A9_EXPLAINER': (
        "Sums over particles can be read linearly from the network's pooling layers, mass and girth from the same or "
        "slightly later layers, and three-particle or clustering quantities mostly not at all, best in the first dense "
        "layer. A quantity's algebraic depth and the layer where it first becomes readable are moderately correlated, and "
        "an untrained copy reads almost nothing from the second round on. Readable does not mean used: the read-outs follow "
        "edits of particle count and girth, but not an edit of D2 (a three-particle correlation ratio) at fixed mass."
    ),
    'ATLAS_EXPLAINER': (
        "Unit-by-unit inspection describes most of JEDI-linear's active units with one physical quantity: about four in "
        "five, counting units that track the same quantity over two jet populations. Examples are the W neuron, which "
        "fires on two-prong jets near the W mass and is negative on top jets, and the largest second-round sum, which "
        "tracks the pT-weighted squared distance from the axis. Units without a single description sit mostly in the dense "
        "layers and the class neurons. Steering a unit (setting it high where it is off) changes as many decisions as "
        "steering a random unit, except for class neurons, so for most units the description is a correlation, not a "
        "tested cause."
    ),
    'B2_EXPLAINER': (
        "Writing every active unit of JEDI-linear as a sum of hinge if-statements, each layer fitted on the previous "
        "layer's fitted values, gives a program that makes the same decision as the network on 94 % of held-out jets, with "
        "no network value inside it. Its accuracy is not distinguishable from the network's: the small lead reported first "
        "did not hold on fresh jets, and the background-rejection difference is not resolved with the available counts. "
        "Under the shift that keeps the invariant mass fixed, the program sends W jets to Z as the network does, while a "
        "formula fitted directly on physics observables does not, because it uses the invariant mass. The program "
        "reproduces decisions; swapping single internal values between jets (E22) shows its variables match the network's "
        "only partly."
    ),
    'B4_EXPLAINER': (
        "The nested if-statement program is not the most compact way to reproduce JEDI-linear's decisions: MARS fitted "
        "directly to the class scores and gradient-boosted trees reach the agreement of the small programs with many times "
        "fewer fitted pieces. Under the jet shift the program agrees with the network a few points more often than the "
        "best boosted trees, but it leads by a similar amount under a random displacement that raises the axis spread "
        "equally, so this lead does not single out the shift mechanism. The surrogate the shift separates is MARS fitted "
        "straight from physics inputs to the scores: it uses the invariant mass and misses most of the network's changed "
        "decisions. Smaller edits (rotation, rank swap, a small pT crossing) change too few decisions to separate any "
        "surrogates."
    ),
    'B5_EXPLAINER': (
        "Short hinge programs (up to 50 terms) on quantities measured about the jet axis bracket the network's overall rate "
        "of changed decisions under the jet shift and the re-axis, but none meets the reproduction criterion, and they "
        "send too few W jets to Z. Equal-size programs on shift-invariant quantities match ordinary decisions almost as "
        "well and change no decision under these edits, so the edits separate the two kinds while ordinary jets do not. "
        "The network's response also depends on the direction of the shift and changes some decisions under a rotation, "
        "which no fitted program reproduces. A compact formula for the network's axis behaviour was not found with this "
        "fitter."
    ),
    'C3_EXPLAINER': (
        "ParT responds to a shift of all particles that keeps the invariant mass fixed, although its pair features would "
        "allow it to compute the invariant mass: ParT_full (ParT with all input features) turns more W jets into Z calls "
        "and its internal mass rises by somewhat less than the axis-relative formula predicts. Holding each particle's "
        "distance to the axis at its unshifted value removes almost all of this response, and under real mass changes the "
        "per-particle inputs carry most of the change in the internal mass while the pair features carry a small, nonzero "
        "part of the decision change. So ParT's mass sense comes mainly from positions measured from the axis, as in "
        "JEDI-linear. The statistics (P2v) show both route shares differ from zero and from one, the routes do not add for "
        "mass decreases, and an unusual ParT_kin row (pT × 0.8) responds through the pair-mass feature."
    ),
    'D4_EXPLAINER': (
        "Because the networks measure spread about the given axis, recomputing particle positions about another standard "
        "axis (winner-take-all, or the hardest particle) changes their decisions while the invariant mass stays the same: "
        "about 30 % of JEDI-linear's decisions change, mostly W jets called Z, and its internal mass rises by close to the "
        "predicted amount. For a two-prong jet, an axis on one prong raises the reading by up to a factor √2, enough to "
        "cross from the W window into the Z window, while top jets mostly keep their call because the top-score neurons "
        "cover a wide mass range. Retraining with the axis on the hardest particle does not remove the effect: later tests "
        "(X7) show those models follow the same formula about their own axis together with a second, shift-invariant "
        "spread. Part of any axis change is "
        "also a shift of the input distribution."
    ),
    'E14_EXPLAINER': (
        "The second round builds its near-radial quantities by adding several first-round half-plane cuts (each on for "
        "particles on one side of a line through the axis), so each unit's on-region is a polygon around the axis, close "
        "to but not a circle. Summing these per-particle polygon responses over a shifted jet reproduces the "
        "direction-dependent part of the network's mass response, and making the responses point-symmetric removes that "
        "part and leaves the rest unchanged. So the direction term of the mass response comes from the uneven polygon "
        "edges. The contour shape alone cannot identify which directions are used, because polygons in random directions "
        "fit about as well; the identification rests on the exact decomposition of the weights."
    ),
    'E20_EXPLAINER': (
        "Swapping only the directions of two neighbouring pT-ranked particles changes a second-round sum when the swap "
        "crosses the edge of that sum's rank window and barely changes it inside the window; moving the window edge moves "
        "the response, and removing the window masks removes almost all of it. So whether a particle counts depends on its "
        "pT rank as well as on its cut, which a plain sum over all particles (as in sum-pooling or energy-flow networks) "
        "does not have. The test covers only sums that depend on direction: the windows of the pT-only sums are untested, "
        "and several other sums show no edge response."
    ),
    'E21_EXPLAINER': (
        "Setting the second-round sums that carry the total pT and the axis spread directly, the first test (steps along a "
        "constant product versus steps along a constant sum) cannot tell a product from a sum, because the two step "
        "directions are too close together. A gradient test shows that the W-versus-Z score and the Z-mass class neuron "
        "depend on the two sums approximately as on their product, more so than for random pairs of sums or random layers, "
        "but with large scatter from jet to jet. The layer that forms the product is not located, and the repeat on a "
        "second sample was not reviewed by the verifier."
    ),
    'E22_EXPLAINER': (
        "Interchange interventions (setting one internal value of the network to its value on another jet, and doing the "
        "same to the if-statement program's matching formula) show that the program gives the network's new decision for "
        "about half to two thirds of the decisions that change. This is above a misaligned control at every layer, but "
        "above a guess that ignores the swap only at the class neurons. Most of the shortfall lies in the program's later "
        "formulas: the second-round formulas followed by the network's own later layers agree more often. So the program's "
        "94 % agreement on ordinary jets does not mean each of its variables is the network's causal variable."
    ),
    'E23_EXPLAINER': (
        "On real jets whose axis is off the pT-weighted centre, centring the jet (which keeps the invariant mass) lowers "
        "JEDI-linear's mass read-out by more than a real mass change of the predicted size, and the opposite move and a "
        "sideways translation raise it, so the read-out follows the sign of the axis-relative mass on unedited data, not "
        "only on artificially shifted jets. A rotation of equal displacement leaves the read-out unchanged, but that is "
        "expected for near-radial sums whatever the mass formula; the sideways translation is the control that tests the "
        "mechanism. The read-out has two different gains, below the prediction for the part quadratic in the shift and "
        "above it for the part linear in the centre offset, so it is not a single function of the axis-relative mass. At "
        "natural offsets the changes in decisions are small and follow how far the particles move, not the predicted mass "
        "change."
    ),
    'E7_EXPLAINER': (
        "Fitted hinge formulas on the second-round sums replace the class neurons at a cost of about one decision in ten, "
        "and under the jet shift and a rotation they move with the network's neurons in sign but by roughly half the size. "
        "A formula on standard jet observables changes fewer decisions on ordinary jets but does not follow the shift when "
        "the invariant mass is among its inputs; without it, it follows partly. So the class neurons are approximately "
        "functions of the second-round sums, and the input edit, not the agreement rate on ordinary jets, shows which "
        "formula follows the network. Random hinge terms on the same sums also send shifted W jets to Z, so that number "
        "alone does not support the fitted formula; the size of the response does."
    ),
    'E8_EXPLAINER': (
        "A rule of a few conditions on the class neurons describes each class's decisions with high precision, and "
        "setting the rule's neurons to typical class values moves a large share of other jets into the class. Moving them "
        "only just inside the conditions moves far fewer, and the score change from moving several neurons equals the sum "
        "of the single moves, because the final layer adds each neuron's contribution. So the rules summarise where that "
        "sum favours a class rather than acting as a logical AND. Rules of similar precision appear after randomly rotating "
        "the neuron axes, and for top jets another set of neurons has a larger effect than the rule's."
    ),
    'EARLIER_C': (
        "Earlier ParT results: the fixed pair-geometry bias explains most of the attention scores and can be replaced by "
        "a fitted formula with little accuracy loss, and under a mass increase most of the change of the W/Z class-token "
        "neuron comes through changes in attention, with one head switching between the W and Z mass ranges. These have "
        "no independent rerun or intervals. Later tests (C3) show that ParT's mass response enters mainly through "
        "per-particle positions measured from the axis; how that input route relates to the attention result has not been "
        "tested."
    ),
    'EARLIER_D': (
        "Earlier edits: JEDI-linear tags W, Z and top mainly by mass windows plus total pT and does not check for a W "
        "inside a top jet. ParT's b-tagging responds to the number of displaced tracks more than to their displacement, it "
        "uses lepton isolation, and it shows no response to π⁰ → γγ pairing or the heavy-quark dead cone. These come from "
        "small samples (400 jets per class for ParT) without intervals and still need a full-statistics rerun."
    ),
    'MASS_EXPLAINER': (
        "The networks' mass is the total pT times the pT-weighted squared spread about the given axis, without the "
        "recentring correction that makes the invariant mass independent of where the axis sits; on training jets the "
        "axis is the centre, so the two agree. Shifting all particles together keeps the invariant mass but raises the "
        "network's mass read-out by about the predicted amount for large shifts and less for small ones, and 77 % of "
        "JEDI-linear's W jets become Z calls; ParT shows the same behaviour. The total pT and the spread are carried by "
        "separate second-round sums; whether and where the dense layers multiply them is open, part of the response "
        "depends on the shift direction, and the decision uses more than this one mass variable (a real mass change that "
        "moves the read-out as much sends far fewer W jets to Z)."
    ),
    'SMALL_EXPLAINER': (
        "A Linformer and a Deep Sets model trained on JEDI-linear's inputs reach its accuracy, while plain transformers "
        "trained for the same time fall short; with the seeds run, Linformer and JEDI-linear are not distinguishable in "
        "accuracy. In all of them the mass is axis-relative, and variables appear over the layers roughly in order of "
        "complexity, partly because an untrained Deep Sets already reads many sums. JEDI-linear's sparse first layer of "
        "pure pT or pure direction cuts does not recur in Deep Sets, and the size of the mass response varies widely "
        "between architectures and seeds, so those details are specific to a model. A transformer trained for 100 epochs "
        "was also tested, and one of its seeds changes decisions under the shift although its mass read-out barely moves."
    ),
    'X6_EXPLAINER': (
        "In three other official JEDI-linear checkpoints the same elements recur: first-layer cuts on pT only or direction "
        "only, second-round sums that depend almost only on distance from the axis, and a shift response that turns many W "
        "jets into Z calls while a matched rotation does not. The pure pT-or-direction pattern is what random placement of "
        "the few nonzero weights would give, so it is not shown to be learned. Rank windows act causally only at the "
        "hardest-particle edge, with a strength that varies between checkpoints, and the permutation-invariant checkpoint, "
        "which has no rank windows, is slightly more accurate, so the objects do not require windows."
    ),
    'X7_EXPLAINER': (
        "When a tagger's inputs are measured about a different axis than in training, its W/Z decisions move in the "
        "direction its own axis-relative mass predicts: centred-axis models turn W into Z under a hardest-particle axis, "
        "and hardest-particle-trained models turn Z into W on the centred axis; shift-invariant controls change nothing, "
        "because their inputs are identical by construction. A translation of the same size that keeps the formula fixed "
        "still changes about a third as many decisions in centred-axis models, with no W-to-Z flow, and a rotation about "
        "the axis is not a valid control because the models' sums are nearly radial. Hardest-particle-trained models "
        "respond to that control with W-to-Z changes, so they read a second, shift-invariant spread besides the formula. "
        "So the direction and most of the size of the axis effect follow the formula, the rest is general sensitivity to "
        "moved inputs, and the verifier's confidence is low."
    ),
    'X8_EXPLAINER': (
        "Networks with the same layers, inputs and fixed-point masks but no trained function (labels shuffled, or trained "
        "weights rearranged) do not turn W calls into Z when the jet is shifted, while all three trained networks do, so "
        "that response is learned. The rank-window swap response, in contrast, appears just as strongly with rearranged "
        "weights as long as the zero-bit masks stay in place, so that test shows where the masks put the window, not that "
        "training uses it. The trained unit that tracks the pT-weighted spread follows it more closely than any null unit, "
        "but per unit of that correlation null units respond as much. The accuracy loss under the WTA axis is matched by a "
        "random shift of the same size. Taking a unit's value from another jet ranks units as replacing it with its "
        "average does; the context round then changes about 28 % of decisions instead of 15 %."
    ),
    'X5_EXPLAINER': (
        "A Deep Sets network trained in three ways (ordinary floating point, with an L1 sparsity penalty, and with the "
        "fixed-point HGQ method) on JEDI-linear's inputs always turns W calls into Z when the jet is shifted, with a mass "
        "read-out change of about the predicted size, so this response does not need fixed-point training. With L1 at "
        "JEDI-linear's sparsity every remaining first-layer unit uses pT only or angles only, but random placement of that "
        "many weights gives the same. Only the HGQ-trained models have more such units than random placement, in two of "
        "three seeds, and their training script also differs in optimizer and length, so quantization is not isolated. "
        "No model formed rank windows; short HGQ training did not form slot masks, so where windows come from is open."
    ),
    'X9_EXPLAINER': (
        "Two standard set networks of the field, a Particle Flow Network at the same accuracy and an Energy Flow Network at "
        "74 %, trained on the same inputs, also turn W calls into Z when the jet is shifted, and their mass read-out moves "
        "like a real mass increase. When the shifted jet's angles are shrunk so that the pT-weighted spread about the axis "
        "is unchanged, almost all of this disappears in all five networks, so the response runs through that spread. Only "
        "JEDI-linear responds to the pT-rank window edge, has first-layer units that are more often pT-only or angle-only "
        "than chance, and has nearly exactly radial objects; the flow networks cannot have rank windows by design, and the "
        "cut purity comparison mixes architecture with fixed-point training. Swapping the program's variables between jets "
        "agrees with the network no better than swapping random directions in the same space."
    ),
}

KEY_RESULT = {
    'Statistics for the claims (R1, R2, P2v)': (
        "Rerunning the claims on fresh jets with bootstrap intervals and correction for multiple tests kept the main mass "
        "and program results and narrowed several others. The invariant-mass hypothesis is rejected for every shift and "
        "axis change in JEDI-linear and ParT, ParT's pair features carry a small but nonzero share of its mass response, "
        "and the if-statement program's accuracy is not distinguishable from the network's. The corrections: fewer "
        "second-round sums are insensitive to every rotation than first stated, the set of sums that carries the jet "
        "context is not unique, Linformer and JEDI-linear accuracy cannot be separated with the seeds run, and the "
        "program's background rejection is not resolved at these sample sizes."
    ),
    "The networks' mass is not the invariant mass": (
        "JEDI-linear and ParT read the jet mass as the total pT times the pT-weighted squared spread about the given axis, "
        "which equals the invariant mass only when the axis is at the jet's centre, as on all training jets. Shifting all "
        "particles together, which keeps the invariant mass, raises this quantity and turns 77 % of JEDI-linear's W jets "
        "into Z calls, and fewer of ParT's. The same sign holds on unedited off-centre jets, but the response is not one "
        "exact formula: part of it depends on the shift direction, and where the two sums are multiplied is not located."
    ),
    'Why this matters: the jet axis is a convention': (
        "Because the networks' mass depends on the axis, measuring particle positions about another accepted axis "
        "(winner-take-all or the hardest particle) changes their decisions although the invariant mass does not change: "
        "with the winner-take-all axis about 30 % for JEDI-linear and 12 % for ParT with all input features, mostly W jets "
        "called Z. The direction of the changes follows each model's own formula, but some changes also occur under a "
        "shift of equal size that keeps the formula fixed, so part of the effect is general sensitivity to inputs unlike "
        "the training data."
    ),
    'The whole trigger network as if-statements': (
        "JEDI-linear's active units, written as nested hinge if-statements ('if Q > t, add a(Q − t)') fitted layer by "
        "layer from physical quantities, form a program that makes the same decision as the network on 94 % of held-out "
        "jets, with no network value inside it, and that follows the network on the jet shift. On fresh jets its accuracy "
        "is not distinguishable from the network's. Its variables match the network's internal variables only partly when tested by swapping single values "
        "between jets, and standard surrogates such as boosted trees reproduce the decisions with fewer fitted pieces. "
        "Mass-type terms first appear in the dense layers of the fit, which is consistent with, but does not locate, where "
        "the network combines total pT and spread."
    ),
}
