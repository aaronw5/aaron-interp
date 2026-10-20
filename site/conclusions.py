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
        "masks put the window, not that the trained weights use it beyond the masks. Networks retrained from scratch with "
        "the same architecture form the same kind of masks but not the official network's strong window-edge response, so "
        "that response belongs to the official checkpoint (or to its much longer training)."
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
        "exactly radial ones: their on-regions are polygons around the axis, not circles. A preregistered, blind fit "
        "confirms that most of these units (15 of 19) are each a short formula of pT and distance sums over rank windows, "
        "more often than any random mixing of the same units."
    ),
    'A4': (
        "JEDI-linear's internal mass behaves like the total pT times the pT-weighted squared spread of the particles about "
        "the given axis, without the term that corrects for the axis being off the jet's centre. Shifting every particle "
        "by the same small angle keeps the invariant mass fixed but raises this quantity; the network's mass read-out (a "
        "linear map from its internal units to log mass, fitted on ordinary jets) rises by about the predicted amount for "
        "large shifts and by less for small ones, and 77 % of W jets become Z calls, in four official checkpoints. On "
        "training jets the axis is the centre and the two definitions agree, so the difference appears only when the axis "
        "is moved. Retrained copies of the network, with and without fixed-point training, show the same behaviour. Not "
        "established: part of the response depends on the shift direction, and the total pT and the spread are each found "
        "as separate internal directions, but their combination (the logarithm of their product) is not found as one "
        "internal quantity in any layer."
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
        "description is not causally tested. The tally depends on chosen thresholds and has not been rerun independently; "
        "a later preregistered, blind test of the object layer (claim A11b) supports the naming part."
    ),
    'A11b': (
        "Most of JEDI-linear's pooled object units (15 of 19) can each be written as a short formula of particle pT and "
        "distance-from-axis sums over pT-rank windows, on three independent jet samples, and no random mixing of the same "
        "units reaches that rate. A sparse autoencoder (a learned dictionary of sparse directions, the usual alternative "
        "basis) makes the layer less readable by this measure, not more. Deep Sets also prefers its own unit basis at a "
        "lower rate, while the Particle Flow Network latent does not; whether the preference comes from training or from "
        "the per-particle ReLU-and-sum architecture is still open."
    ),
    'A4c': (
        "Swapping one or two learned directions of JEDI-linear's object layer between two jets changes its decision as an "
        "input edit of the total pT, or of the pT-weighted squared spread about the axis, would, for most of the decisions "
        "that edit changes; random directions of the same size almost never do. So these two moments are separate internal "
        "quantities from the object layer on. Their combination, the log of their product, is not found as one quantity: the "
        "directions found for it only follow a rescaling of all pT, and the same change made through the angles runs through "
        "the spread directions. For the spread, the gain over simply giving the edited jet the other jet's decision is small."
    ),
    'A1r': (
        "JEDI-linear retrained from scratch, with and without fixed-point (quantization-aware) training, reads the jet mass "
        "through the pT-weighted spread about the stored axis as the official network does, so that behaviour belongs to the "
        "architecture and inputs, not to one checkpoint. The official network's strong response at the 14-hardest-particle "
        "boundary appears in none of eight retrained copies, although the quantized ones form the same kind of zero-bit "
        "masks. Fixed-point training keeps very few first-layer weights, and at that sparsity every cut is pT-only or "
        "angle-only automatically. Uncertain: the retrained copies are 1–4 points less accurate and trained about ten times "
        "fewer epochs than the official one."
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
        "radial objects were found only in JEDI-linear. Retrained JEDI-linear copies with and without fixed-point training "
        "also respond to the shift about as much as to a real widening of the same spread."
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
    'B2m': (
        "The question is whether the program's variables are variables inside the network: if a variable's value is "
        "swapped between two jets inside the network, does the decision change as an input edit of that variable predicts? "
        "Swapping the program's per-unit formulas works no better than swapping random directions, but learned "
        "one- or two-dimensional directions for the total pT and the pT-weighted spread do reproduce most such decision "
        "changes. So the physics quantities are present as internal variables, not in the basis of the fitted formulas, and "
        "the log of their product is not found as one variable."
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
        "between the W and Z mass ranges. This was an earlier result on 400 jets per class. Re-measured at ten times the "
        "statistics with the corrected input build (P3, P3v), the switching head is confirmed (claim C2a), but the "
        "attention-dominated split is not: changes in where the heads look account for about a third of the neuron's "
        "change, and the switching head itself for about 5 % (claim C2b). This claim is kept only as the record of the "
        "earlier wording."
    ),
    'C2a': (
        "In ParT_full, one attention head of the last class-token layer gives most of its attention to the class token for "
        "jets lighter than about 87 GeV and almost none for heavier jets, with a switch about 4 GeV wide between the W and "
        "Z masses; no other head of the class-token stage does this, and the result reproduces on new jets. The switch "
        "also moves when all particles are shifted off the jet axis at fixed mass, so it follows the network's "
        "axis-relative mass rather than the invariant mass. What the switch is used for is not known, because it carries "
        "little of the mass information that reaches the decision (C2b)."
    ),
    'C2b': (
        "When the jet mass is changed, the change of ParT_full's W/Z output unit (n98, a coordinate of the class token "
        "that is high for W and low for Z) passes almost entirely through the first class-token attention layer, about "
        "half through one head; the 87 GeV switch head carries about 5 %. Most of the change is in what the particles "
        "carry, not in where the heads look. A three-unit circuit made of the two heads and n98, with everything else "
        "replaced, separates W from Z barely above chance (about 52 % against 83 % for the model), but under the same strict "
        "definition even all 16 heads reach only 63 %, so a small circuit that works on its own has not been found yet. "
        "The same first layer also carries the response when all particles are shifted off the jet axis at fixed "
        "invariant mass (99 %), so it reads the network's axis-relative mass, whatever its source."
    ),
    'C7': (
        "ParT does not build its axis-relative mass at any one layer. One direction in the jet-averaged particle "
        "representation, already present right after the input embedding, follows the axis-moment mass (the square root "
        "of total pT times the pT-weighted squared distance of the particles from the jet axis) and carries about two "
        "thirds of the network's response to a shift of all particles at fixed invariant mass. Copying this direction's "
        "value from real jets moves the W/Z output the way the shift predicts, at about 40 % of the strength; it is not an "
        "exact copy of the axis-moment mass, and nonlinear or per-particle forms have not been tested. Two newer checks "
        "lower confidence: in natural jets the W/Z score changes with this mass by only about a seventh of what the shift "
        "produces, and in ParticleNet directions of this kind also appear in networks with shuffled weights; neither "
        "control has been run on ParT's direction yet."
    ),
    'C8': (
        "When everything outside a chosen set of units is replaced by values from a jet of the other class, no small set "
        "reproduces ParT_full's W-versus-Z calls. With attention heads and neuron groups, sets of up to a tenth of the "
        "connections reach at most 63 % correct against 83 % for the full network; with learned sparse features (units "
        "trained to reconstruct the network's layers from few active components) as units, 50 features reach about 51 %, "
        "and only while the attention layers are computed on the real jet. This holds for the searches tried, which rank "
        "units by attribution; a search that trains the selection over features has not been run."
    ),
    'C3s': (
        "Six copies of ParT (two input sets, three random seeds) were retrained on the same 2 million jets used for "
        "L-GATr, and all six finished 50 epochs. After 3 of 50 epochs, the full-input copy already reacted like the "
        "released network to a shift of all particles that keeps the invariant mass fixed, while the kinematics-only "
        "copies reacted about half as strongly or less. Whether the axis response holds across seeds and at this data "
        "size is decided by the analysis of the final checkpoints, not yet run."
    ),
    'C9': (
        "ParticleNet's W-versus-Z score responds the same way when every particle is shifted by +0.06 or by −0.06 rad in "
        "azimuth, neither of which changes the jet mass. After the second graph layer one direction in its internal "
        "representation reproduces about three quarters of the response to either sign, but it is not a copy of the "
        "axis-moment mass, and networks with shuffled or untrained weights also contain such directions, so it does not "
        "show a variable built by training. The neighbour graphs that ParticleNet rebuilds carry at most about 4 % of the "
        "response, and a few dozen pooled channels kept alone reproduce the W/Z calls but not the 10-class calls."
    ),
    'C10': (
        "Pre-registered test, not yet run: Particle Transformers were trained on labels that depend on only one mass "
        "variable, either the invariant mass or the axis-moment mass. If the network trained on the invariant mass "
        "ignores a shift that keeps the invariant mass fixed, ParT's axis response is learned from the data; if it "
        "still responds, it comes from ParT's axis-relative inputs or architecture. Training has finished; a version "
        "given only pair features learns the invariant-mass label to 97 %."
    ),
    'C5': (
        "ParticleNet, a graph network without attention trained on the same JetClass inputs, also calls more true W jets "
        "Z when every particle is moved 0.06 rad off the jet axis at fixed invariant mass: from 12 % to 26–28 %, about "
        "two-thirds of ParT's increase. The response enters through each particle's distance to the axis and passes "
        "through terms that read each particle on its own; the neighbour graphs do not carry it, and rotations about the "
        "axis that keep every distance do not reproduce it; on three more samples, freezing the rebuilt neighbour graphs "
        "removes at most about 4 % of the response. Whether the axis-relative input format causes this, or both "
        "networks learned a similar solution, has not been tested."
    ),
    'C6': (
        "No trained L-GATr, a tagger built to respect the symmetries of special relativity, is published for JetClass, so "
        "one is being trained with the authors' code and settings. An early checkpoint ignores a rotation of all "
        "particles about the beam, as it must by construction, while a real mass change still moves its W/Z calls. The "
        "test that decides the claim, edits that keep the invariant mass but change the spread of particles about the "
        "axis, will run on the final checkpoint; until then the claim is untested."
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
        "call drops when the subjet-pair masses are rearranged in general. This holds for models trained on the multi-class "
        "hls4ml and JetClass data; top taggers trained on the Top Tagging Landscape do use the W-pair mass (claim D6). "
        "The momentum edit used here kept the jet mass fixed (it is a boost along the jet axis); scaling only the momenta "
        "lowers the W call to 48 % instead, and most calls changed by the boost are W or Z jets becoming top (claim D9)."
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
        "spread that does not change under a shift. A stand-in classifier that uses the mass about the given axis "
        "reproduces JEDI-linear's axis response (37.7 vs 37.6 % of Z vs W calls), but only partly ParT's. Confidence is "
        "limited, and a rotation about the axis turned out not to be a valid control for these near-radial models."
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
        "it is placed, so that test alone does not show counting. Quark–gluon taggers trained on Pythia jets are also not "
        "collinear safe, but they do not count particles (claim D7), so the multiplicity part holds only for the hls4ml "
        "and JetClass models tested. Pileup-like soft particles placed near the jet centre, where they add almost no "
        "mass, also lower the JetClass ParT and ParticleNet and the Top Landscape ParT, LorentzNet and PFN efficiencies, "
        "which supports the counting part for those models (claim D10)."
    ),
    'D6': (
        "We changed the mass of the W-candidate subjet pair inside top jets while holding the total jet mass and the "
        "three-prong measure τ32 fixed. ParT, ParticleNet and LorentzNet, trained on the Top Tagging Landscape, lowered "
        "their top score about as much as a classifier given the W-pair mass, and 2 to 4 times more than a classifier that "
        "sees only the edit's side effects (smaller subjets, lower τ21). So these taggers check the W mass inside the top, "
        "unlike JEDI-linear on the multi-class data. PFN and EFN respond only at the side-effect level, and the most "
        "extreme edits produce jets a classifier can tell from real ones."
    ),
    'D7': (
        "Trained quark–gluon taggers (ParT, ParticleNet, LorentzNet) do not count particles: adding a realistic soft "
        "particle or removing the softest one changes their output by at most about 0.03 in logit, while a count-only "
        "likelihood ratio moves 0.1. Splitting the hardest particle into two collinear pieces moves them toward gluon "
        "whether or not the particle count changes, so their collinear unsafety comes from how the hard particles share "
        "energy. Which function of that energy sharing they use is not identified, and the split jets are unrealistic. "
        "The same holds when the Pythia-trained taggers are tested on Herwig jets (added soft particle about −0.01 vs +0.09 "
        "for counting), and the leading particle's momentum fraction alone is not what they use; this newer test has not "
        "yet been checked by a verifier, and models trained on Herwig are still training."
    ),
    'D8': (
        "For each tagger, the closest stand-in built from standard IRC-safe jet shapes (mass, N-subjettiness, energy-flow "
        "polynomials) ranks W/QCD and top/QCD jets like the network in more than 90 % of signal–background pairs, but never "
        "responds to the jet-axis convention, because it is axis-free by construction. Writing the mass as the mass about "
        "the axis the network is given, which equals the invariant mass on natural jets, reproduces JEDI-linear's axis "
        "response (37.7 vs 37.6 % of Z vs W calls changed). The networks' response to collinear splits and to the total "
        "pT, and ParT's Z vs W ranking, are not described by either stand-in."
    ),
    'D9': (
        "We boosted each jet along its own axis, which lowers its pT by 10 % and keeps its invariant mass exactly "
        "fixed. Every tagger trained on the hls4ml jets changed a quarter to a third of its calls "
        "(JEDI-linear 33 %, against 2.4 % for a rotation of the same size), so these taggers do not treat the jet mass as "
        "a boost-invariant quantity; for JEDI-linear most changed calls are W or Z jets called top, the known total-pT "
        "shortcut. JetClass and Top Landscape taggers changed several times fewer calls under the boost than under "
        "pT-only or angle-only scaling. Why the two groups differ is unknown, since both were trained in narrow pT windows "
        "on equivalent inputs."
    ),
    'D10': (
        "Adding pileup-like soft particles over the jet lowers every network tagger's signal efficiency faster than it "
        "lowers a groomed (soft-drop) mass tagger. JEDI-linear reacts to the mass the particles add, like a plain mass "
        "window (W efficiency 87 % → 43 % at the lowest density). The JetClass ParT and ParticleNet and the Top Landscape "
        "ParT, LorentzNet and PFN react mainly to how many soft particles are added, even when they sit at the jet centre "
        "and add almost no mass. The jet-axis shift they cause is too small to matter; the pileup model is simplified and "
        "none of the networks was trained with pileup."
    ),
}

EXPLAINER = {
    'H12_EXPLAINER': (
        "A boost along the jet axis that keeps the invariant mass fixed changes about a third of the calls of every "
        "hls4ml-trained tagger (JEDI-linear 33 % vs 2.4 % for a rotation) and several times fewer than pT-only or "
        "angle-only scaling in the JetClass and Top taggers. For JEDI-linear most changed calls are W or Z jets called top, "
        "the total-pT shortcut. The test cannot tell the invariant mass from the axis-moment mass, and why the two groups "
        "of taggers differ is not known."
    ),
    'H13_EXPLAINER': (
        "Pileup-like soft particles lower every network's signal efficiency faster than a soft-drop mass tagger's; the "
        "few-milliradian axis shift they cause explains almost none of it. JEDI-linear reacts to the mass they add, "
        "while the particle-ID JetClass networks and the Top ParT, LorentzNet and PFN react mainly to how many are added. "
        "The pileup model is a simplified one and no network was trained with pileup."
    ),
    'H14_EXPLAINER': (
        "Quark–gluon taggers trained on Pythia ignore one added soft particle on Herwig jets as they do on Pythia jets "
        "(about −0.01 in logit vs +0.09 for a count-only classifier), so the no-counting result does not depend on the "
        "test generator. Their split response comes from how momentum is shared among the hardest particles, not from "
        "the leading fraction alone. Not yet verified; Herwig-trained models are still training."
    ),
    'H4_EXPLAINER': (
        "At fixed jet mass and τ32, moving the W-candidate pair mass off 80 GeV lowers the top score of ParT, ParticleNet "
        "and LorentzNet as much as a classifier that uses the W-pair mass, and 2 to 4 times more than a classifier of the "
        "edit's side effects; a classifier with only mass and τ32 does not respond. These Top-Landscape taggers use the W "
        "mass inside the top. For PFN and EFN the test does not separate W-mass use from the side effects."
    ),
    'H5_EXPLAINER': (
        "A count-only quark–gluon discriminant moves by 0.1 in logit per added particle (Poisson counting would predict "
        "0.46), but ParT, ParticleNet and LorentzNet barely move when a realistic soft particle is added or the softest "
        "removed. They move toward gluon when the hardest particle is split, equally with the count held fixed. So they do "
        "not count particles; the split response comes from the hard-particle energy sharing, in a form not yet identified."
    ),
    'H8_EXPLAINER': (
        "Stand-ins built from standard jet shapes rank jets like the taggers on natural jets but change no decisions when "
        "the jet axis is redefined, because they are axis-free. Replacing the invariant mass by the mass about the "
        "network's axis, which is the same on natural jets, reproduces JEDI-linear's axis response, so that response is "
        "mostly one known variable. The responses to collinear splits and to total pT, and ParT's Z vs W ranking, remain "
        "unexplained."
    ),
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
    'P3_EXPLAINER': (
        "One attention head in ParT_full's last class-token layer moves its attention off the class token as the jet mass "
        "passes 87 GeV, and no other head does this; the result reproduces on new jets. Under a real change of jet mass, "
        "however, this head carries only about 5 % of the change in the W/Z output unit, while the heads of the first "
        "class-token layer carry nearly all of it, about half through one head. The circuit of these two heads and the "
        "output unit, kept alone, separates W from Z barely above chance; how the first layer's output is turned into the "
        "W/Z unit is still open."
    ),
    'P4_EXPLAINER': (
        "Moving every particle 0.06 rad away from the jet axis leaves the jet mass unchanged, yet ParticleNet then calls "
        "26–28 % of true W jets Z instead of 12 %, about two-thirds of ParT's increase. In both networks the change "
        "enters through each particle's distance to the axis and passes through terms that read each particle on its "
        "own, not through the neighbour graphs; rotations that keep every distance do not reproduce it. So the response is "
        "not specific to attention; whether the input format causes it is tested next with L-GATr."
    ),
    'P5_EXPLAINER': (
        "No trained L-GATr is published for JetClass, so one is being trained with the authors' code and settings; the run "
        "is healthy and should finish around 21 October. An early checkpoint ignores a rotation of all particles about "
        "the beam, which strongly moves ParT's W/Z calls, but this holds by construction and says nothing yet about how "
        "L-GATr estimates mass. The deciding test, edits that keep the invariant mass but change the spread about the "
        "axis, is pre-registered for the final checkpoint."
    ),
    'P8_EXPLAINER': (
        "ParT does not form its axis-relative mass at one layer: each particle layer adds a little, and about half of the "
        "response to a shift at fixed invariant mass reaches the class token directly from the input embedding. One "
        "direction in the jet-averaged particle representation, present from the embedding onward, carries about two "
        "thirds of the response and follows the axis-moment mass; copying it from real jets moves the W/Z output at about "
        "40 % of the shift's strength. An earlier null came from donor jets that held the axis-moment mass fixed."
    ),
    'P9_EXPLAINER': (
        "Searching the whole of ParT_full for a small set of connections that, kept alone, reproduces its W-versus-Z "
        "calls finds none up to a tenth of the network: the best reach 63 % correct against 83 % for the model. A summed "
        "logit score can reach 98 % at that size without more correct calls, so accuracy is the measure used. What the "
        "found circuits add over random ones sits mostly in the final read-out; another search could still find a "
        "smaller circuit."
    ),
    'P10_EXPLAINER': (
        "ParT is being retrained six times on the 2 million jets used for L-GATr to check that its axis-relative mass "
        "response is not a property of one released checkpoint. After 3 of 50 epochs the full-input seed already responds "
        "like the released network (about 23 more true-W jets in 100 called Z); the kinematics-only seeds respond less. "
        "The final checkpoints decide, with pre-registered tests."
    ),
    'P13_EXPLAINER': (
        "Learned sparse features (units trained to reconstruct each layer of ParT_full from a few active components) "
        "did not give a small circuit for its W-versus-Z decision: 50 features kept alone give about 51 % correct calls "
        "against the model's 83 %, and 75 % needs 1,000–2,000 features, even with the attention layers left on the real "
        "jet. The feature most correlated with the axis-moment mass does not carry the response to a mass-preserving "
        "shift; how the rest of that response is carried is still open."
    ),
    'P14_EXPLAINER': (
        "Particle Transformers were trained on labels that depend on one known mass variable to test the methods and to "
        "ask whether the axis response is learned from data. Training has finished, but only the released-network "
        "control is analysed: it responds to a rotation that keeps the invariant mass fixed at about three quarters of "
        "its response to a matched real mass change, mostly through the per-particle angle inputs. The deciding "
        "comparison runs next round."
    ),
    'P15_EXPLAINER': (
        "Inside ParticleNet, the response to a mass-preserving azimuthal shift is the same for either sign, and a "
        "direction after the second graph layer reproduces about three quarters of it; the first-layer direction found "
        "for the positive shift only records which way the particles moved. Directions like these also exist in untrained "
        "networks and are not the axis-moment mass. The rebuilt neighbour graphs carry at most about 4 %, and a few dozen "
        "pooled channels reproduce the W/Z calls but not the 10-class calls."
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
        "neuron comes through changes in attention, with one head switching between the W and Z mass ranges. The "
        "attention result was re-measured at ten times the statistics (P3, P3v): the switching head is confirmed, but "
        "changes in attention account for only about a third of the neuron's change and the switching head for about 5 %; "
        "most of the change is carried by what the particles pass through the first class-token layer. The pair-bias "
        "result (C1) has no independent rerun."
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
    'X10_EXPLAINER': (
        "A method that learns which few directions of a layer to swap between two jets finds one or two directions in "
        "JEDI-linear's object layer that carry the total pT and the pT-weighted spread about the axis: swapping them "
        "reproduces most decision changes of the matching input edits, and random directions almost never do. The "
        "directions found for the log of their product only follow a rescaling of all pT; the same change made with angles "
        "runs through the spread directions, so the product is not one internal quantity. Untrained networks also beat "
        "random directions, so the meaningful reference is 'give the jet the other jet's decision', which the trained "
        "network beats for both moments and untrained networks never do."
    ),
    'X11_EXPLAINER': (
        "Fifteen of JEDI-linear's 19 pooled object units are each a short formula of pT and distance sums over pT-rank "
        "windows, on three jet samples, and none of 250 random mixings of the same units is as readable. Sparse-autoencoder "
        "features of the same layer are readable less often. Deep Sets also prefers its own units, the Particle Flow Network "
        "does not, and untrained networks have almost no readable units; whether the preferred basis comes from training "
        "or from the architecture is open."
    ),
    'X12_EXPLAINER': (
        "Eight JEDI-linear networks retrained from scratch, six with fixed-point training and two without, all read the jet "
        "mass through the pT-weighted spread about the stored axis: compared with a real widening of the jet that gives the "
        "same spread, they respond to an off-centre shift about equally. The official network's strong response at the "
        "14-hardest-particle boundary appears in none of them, although the fixed-point ones form the same zero-bit masks. "
        "Pure pT-only or angle-only first-layer cuts follow from how few weights fixed-point training keeps; the retrained "
        "copies are less accurate and trained far fewer epochs than the official one."
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
        "exact formula: part of it depends on the shift direction, and the two sums are found as separate internal "
        "directions while their product is not found as one internal quantity. JEDI-linear copies retrained from scratch, "
        "with and without fixed-point training, read the mass the same way."
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
