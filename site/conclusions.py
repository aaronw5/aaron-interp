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
        "replaced by any of several other sums with the same effect. An automated circuit search (round 14, E29) keeps "
        "only one or two of the eleven jet-level units and loses little accuracy, while dropping the summed-pT and core "
        "sums costs much more, so most of the context effect runs through very few units."
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
        "more often than any random mixing of the same units. Round 15 adds two limits: the sum named \"pT of the 14 "
        "hardest particles\" responds as much to pT added to softer particles, because its context input carries the total "
        "jet pT, so as a whole it tracks total pT (E33); and a pT-independent sum of how far particles lie beyond 0.2 from "
        "the axis carries most of the quark/gluon response to wide-angle particles (E34)."
    ),
    'A4': (
        "JEDI-linear's internal mass behaves like the total pT times the pT-weighted squared spread of the particles about "
        "the given axis, without the term that corrects for the axis being off the jet's centre. Shifting every particle "
        "by the same small angle keeps the invariant mass fixed but raises this quantity; the network's mass read-out (a "
        "linear map from its internal units to log mass, fitted on ordinary jets) rises by about the predicted amount for "
        "large shifts and by less for small ones, and 77 % of W jets become Z calls, in four official checkpoints. On "
        "training jets the axis is the centre and the two definitions agree, so the difference appears only when the axis "
        "is moved. Retrained copies of the network, with and without fixed-point training, show the same behaviour, and so "
        "does the official 16-feature JEDI-linear, which reads the spread through each particle's distance from the axis. Not "
        "established: part of the response depends on the shift direction, and the total pT and the spread are each found "
        "as separate internal directions, but their combination (the logarithm of their product) is not found as one "
        "internal quantity in any layer, and a direct test for a unit that multiplies them found none but can exclude only "
        "large effects. In the 16-feature network, shifting particles off the axis turns W jets into Z and then into top "
        "calls, and stretching only the particle distances to give the same change of the pT-weighted spread produces the "
        "same number of W-to-Z changes, while stretches matched on other radial sums do not (round 10, E25w). Tested "
        "against a family of 15 angular spreads (round 12, E27), the pT weighting of the spread is confirmed, but the "
        "power of the distance is not pinned to the square. In round 13 (E27w) the hardest particles were turned around "
        "the axis so that the true jet mass fell while every particle kept its pT and distance from the axis; the "
        "network's mass read-out did not move, so it does not read the jet mass, only each particle's pT and distance "
        "from the axis. Round 14 (E31) adds a limit to this: the shifted and turned jets differ from real jets in "
        "quantities measured about the axis, and in real jets the centre is never far enough off the axis to change the "
        "mass by more than 0.7 %, so the difference between the two masses describes how the network handles "
        "axis-displaced inputs, not something that can be seen on natural jets (claim A4o). Round 15 (E32) tests a "
        "realistic case: when the axis is computed from particles snapped to trigger-sized calorimeter towers, the network "
        "changes its W/Z decision about 0.9 as much as for a real mass change of the predicted size, and the response "
        "disappears when the spread about the new axis is restored (claim A4t)."
    ),
    'A4n': (
        "On real, unedited jets whose axis is slightly off the pT-weighted centre, moving the particles so that the jet is "
        "centred keeps the invariant mass fixed but lowers JEDI-linear's mass read-out, and moving them the opposite way or "
        "sideways raises it, as the axis-relative mass predicts in sign. So the axis dependence is not limited to the "
        "artificial off-centre jets made by the large shift. An earlier finding that the two parts of the response (one "
        "growing with the centre offset, one with the square of the shift) enter with clearly different weights did not "
        "hold up: measured without bias they are weighted nearly equally from the dense layers on, and unequally only at "
        "the first summed layer. Most of the direction-dependent part follows the plain average position of the particles, "
        "not the momentum-weighted one, which the axis-relative mass formula does not predict. That direction-dependent part "
        "does not pass mainly through the two jet-level units that count particles on each side of the axis: holding them "
        "fixed removes about a quarter of it in the later layers and none in the final W-versus-Z score (round 10, E23x)."
    ),
    'A5': (
        "This is partly supported. The class neurons (the units just before the final linear layer) can be approximated by "
        "hinge formulas (sums of terms of the form 'if X > t, add a(X − t)') on the second-round sums: substituting all of "
        "them changes about 10 % of decisions, and under input edits the formulas move in the same direction as the "
        "network's neurons but by roughly half as much. In interchange interventions (setting one internal value to its "
        "value on another jet, in both the network and the formulas) the formulas give the network's new decision in most "
        "cases, but do better than a guess that ignores the swap only at the class neurons. So the class neurons are "
        "approximately, not exactly, functions of the second-round sums. In round 14 (E30) the formulas failed a direct "
        "test: changing one second-round sum by half its spread moves the class neurons in ways the formulas do not "
        "predict, even on the jets where the neurons are strongest, so the formulas reproduce the neurons' values but not "
        "how they respond to each input (confidence lowered to 0.4)."
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
        "a later preregistered, blind test of the object layer (claim A11b) supports the naming part. A round-14 check of "
        "the weakest activations (E30) found no second job there: moving one particle changes two of the sums by the "
        "amount their formula predicts in every tenth of their range, and the third sum keeps the same quantity with a "
        "smaller gain in its strongest tenth; one sum responds at about 0.8 of the prediction in its weakest tenth, for a "
        "reason not yet found."
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
        "the spread directions. For the spread, the gain over simply giving the edited jet the other jet's decision is small. "
        "The same test on the official 16-feature JEDI-linear, and again on the 3-feature one with new jets, finds the two "
        "moments as separate directions in the object layer of both. A test of where they first appear, before that layer, "
        "was inconclusive: networks with entirely random weights allow the same swaps about as well at the first layer, and "
        "at the object layer the trained networks are above most but not all random networks. Holding the two directions "
        "fixed also removes most of the decision changes caused by a boost of the jet at fixed mass."
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
    'A10r': (
        "Three standard set networks (Particle Flow Network, Energy Flow Network, Deep Sets) trained on jets randomly "
        "shifted sideways by up to some amount ignore whole-jet shifts up to about that amount and change their decisions "
        "beyond it, towards heavier classes. Across training ranges from 0.04 to 0.30 rad the point where they start to "
        "respond moves one-for-one with the range, so how large a shift they tolerate is learned from the training jets, "
        "not fixed by the architecture. Uncertain: two seeds per setting, and JEDI-linear itself was not part of this scan."
    ),
    'A10c': (
        "Inside the range of shifts they were trained on, Deep Sets and Energy Flow Networks partly cancel a shift by "
        "reading off where the jet's centre is: removing the part of their internal response that lies along two "
        "directions which linearly encode the centre position makes them respond to the shift again, more than removing "
        "two random directions of the same size does. The Particle Flow Network does not show this clearly. Uncertain: "
        "one sample, and the result passes a false-discovery-rate correction but not the stricter Holm correction at the "
        "number of random draws used."
    ),
    'A3z': (
        "Moving every particle to a random direction around the jet axis, while keeping its momentum, its distance from the "
        "axis and the jet's mass moments, leaves JEDI-linear's named radial sums (total pT of the hardest particles, "
        "soft multiplicity, radial spreads) nearly unchanged. The extra calls that change, beyond a plain rotation of the "
        "jet, pass through two units that see only a jet-level summary built from the first-layer direction cuts. Deep "
        "Sets and Particle Flow Networks change as many calls under the same edit, so at the level of decisions this test "
        "does not distinguish JEDI-linear's description from a generic pooled network. The two summary units were later "
        "found to follow fixed-frame particle-count shapes (claim A3c)."
    ),
    'A3c': (
        "The two JEDI-linear units that carry its response to where particles sit in azimuth never look at individual "
        "particles; they read only the jet-level summary. That summary follows plain particle counts, not momentum: one "
        "unit tracks how far the centre of the particle count sits from the jet axis, the other how elongated the "
        "particle-count distribution is, both in a fixed (Δη, Δφ) frame rather than as rotation-invariant jet quantities. "
        "These descriptions explain about 93–95 % of the units' jet-level thresholds. Not explained: the elongation unit "
        "still changes when the jet is turned by 180°, which an elongation measure cannot see, and the network's per-slot "
        "rounding shapes the other unit's response. When the whole jet is shifted, holding these two units fixed removes "
        "about a quarter of the direction-dependent part of the response in the later layers and none of it in the final "
        "W-versus-Z score, so they are a minor route for that response (round 10, E23x)."
    ),
    'A10s': (
        "Every one of 56 trained networks tested reacts to a shift of the whole jet that leaves its true mass unchanged, "
        "but by very different amounts. The amount matches how much of each network's mass estimate is measured from the "
        "stored jet axis rather than from quantities that do not move with a shift (such as distances between particles, "
        "or positions measured from the jet's own centre; the tests used cannot tell these two apart). For the set-based "
        "networks (JEDI-linear, Particle and Energy Flow Networks, Deep Sets) that share is set by the network type; for "
        "transformer-type networks it differs between training runs of the same design. Training these networks is exactly "
        "repeatable, and the share changes when the starting weights, the order of training batches or the length of "
        "training is changed, with no single one of these detectably dominant; it forms in the middle of training and is "
        "not linked to accuracy. Uncertain: the share is measured on a mass read-out of the last layer, and in one case the "
        "network's own decision moved the other way; which internal units hold the shift-independent part is open."
    ),
    'A14': (
        "Boosting a jet along its axis keeps its mass but lowers or raises its total pT; at 10 % lower pT about a third of "
        "JEDI-linear's decisions change. About half of those changes disappear when the network's internal sum of the "
        "hardest particles' pT is held at its pre-boost value, and most disappear when two learned directions for total pT "
        "and pT-weighted angular spread are held, while the network's mass estimate does not move on average. So the "
        "network reads total pT as a class feature next to the mass. Internal directions that avoid total pT but vary as "
        "much in data undo far less, and most of the change in total pT reaches that sum through one context unit that "
        "summarises the whole jet. Uncertain: the boost moves the network mostly along total pT, so whether total pT is "
        "singled out or is simply where the boost acts cannot be separated at that layer, and at higher pT the response "
        "is spread over several later units."
    ),
    'A4w': (
        "When particles are moved toward or away from the jet axis with the total pT and the pT-weighted centre held, the "
        "network's mass-related decisions follow spreads in which each particle is weighted by its pT, not spreads that "
        "count particles equally or weight them by pT squared. Among pT-weighted spreads it does not respond to the "
        "squared-distance sum (the one in the invariant-mass formula) alone. Still open: which pT-weighted combination it "
        "uses, whether a girth-like spread enters, and whether it reads that sum or the mass itself, since the edits "
        "always changed both together."
    ),
    'A15': (
        "Splitting one particle into two particles moving in exactly the same direction leaves the jet's mass and every "
        "pT-weighted shape unchanged, yet it changes 2–4 % of JEDI-linear's decisions, more than moving the same particle "
        "without splitting it, and pushes quark calls toward gluon. So the trained network is neither collinear- nor "
        "soft-safe: part of its decision follows the number of particles. The effect reproduces on two further samples "
        "and in gluon-rich and high-multiplicity jets, and undoing a real near-collinear pair in data moves calls the "
        "other way. Its internal route is only partly known (claim A15r). A classifier cannot tell split jets or jets with "
        "one added soft particle from real jets, and the effect holds on those jets (round 14, E31), so it is not caused "
        "by unnatural inputs. With the jet's total momentum held exactly fixed (round 19, E40), splits move the "
        "quark/gluon score toward gluon at every distance from the axis, while very soft added particles matter only "
        "away from the jet core (claim A19qg)."
    ),
    'A16': (
        "We let an algorithm find the smallest part of JEDI-linear that still classifies jets, by removing internal "
        "connections whenever the output barely changes. It keeps every hand-named first-layer particle cut and nearly all "
        "of the hand-named jet quantities, and with about 115 of 186 active units it recovers about 95 % of the accuracy "
        "above chance; replacing its jet quantities by other active ones of the same number does worse. Because the hand "
        "names already cover almost every active unit in those layers, this shows agreement, not that the names are the "
        "only good description; and the hand-built program matches the algorithm's circuit only under one of the two "
        "standard ways of switching units off. Applied to ten trained versions of the network (round 15, E33, not yet "
        "rerun), the search needs about two thirds or more of the active units in the quantized networks and nearly all of "
        "them in the two trained without quantization."
    ),
    'A5n': (
        "A class neuron with a ReLU passes on only its positive side. For the two neurons that pick out jet masses near "
        "75 and 94 GeV, the hidden negative side keeps measuring the mass beyond the window: making a jet 5 % heavier or "
        "lighter moves it by about the amount the mass dependence predicts, while rotating the jet does not. The network "
        "throws this information away. Passing it on would cost accuracy, but mostly because of the size of the added "
        "values rather than what they say about each jet."
    ),
    'A4o': (
        "We checked whether the edited jets behind the mass result look like real jets. The shifted and turned jets "
        "differ from real jets only in quantities measured relative to the stored jet axis, and real jets never have "
        "their centre far enough off that axis to change the mass by more than 0.7 %. So in real data the jet mass and "
        "the spread-based mass the network uses are the same number, and the finding that the network reads the spread "
        "describes how it handles axis-displaced inputs. The boosted, split and soft-particle jets stay among real jets, "
        "and their effects hold there. Jets measured about a realistic trigger-style axis (round 15, E32) also look real "
        "on quantities that do not depend on the axis, but not on quantities measured about it, so a realistic axis error "
        "still produces inputs unlike stored-axis jets."
    ),
    'A4t': (
        "We measured the particles of each jet about an axis like a trigger would compute it: the momentum-weighted "
        "direction after snapping every particle to a calorimeter tower of 0.087 in η and φ, which moves the axis by "
        "about 18 mrad. JEDI-linear changes its W/Z decision about 0.9 as much as for a real mass change of the same "
        "predicted size, and the effect disappears when the spread of the particles about the new axis is restored, so "
        "the network reads the spread about whatever axis it is given. It also responds to which way the axis moved, "
        "about half as strongly; the tower model has no threshold or pileup, so how close it is to a real trigger is "
        "untested."
    ),
    'A1x v2': (
        "We repeated the naming of JEDI-linear's internal jet quantities on ten separately trained versions of the "
        "network. Two quantities come back in most of the quantized ones: the momentum-weighted squared distance of the "
        "particles from the axis, and a count of soft particles. The \"sum of pT of the hardest particles\" of the "
        "reference network turns out to track the total jet pT, and a true hardest-particles window appears in only "
        "three networks with different sizes. Formulas picked at random and fitted to the reference also recur in about "
        "a third of the other networks, so it is not yet shown that these quantities recur more than chance."
    ),
    'A17': (
        "We asked how JEDI-linear tells gluon jets from quark jets by changing one jet property by 5 % while holding the "
        "others: the number of soft particles, the jet width (girth), how concentrated the momentum is in a few "
        "particles (pT^D), and how far particles sit from the axis. Each moves the network's score in the direction that "
        "physics expects for gluons versus quarks, at about half to three quarters of the response of a classifier "
        "trained on these four properties. The \"mass\" part is a response to the particles' distance from the axis, not "
        "to the invariant mass, and an untrained network shows it too, so only the count, width and pT^D responses are "
        "shown to be learned."
    ),
    'A4s': (
        "We kept each W or Z jet's total momentum and mass exactly fixed and only made the momentum split between its "
        "two prongs more lopsided. JEDI-linear then calls W jets more Z-like and Z jets more W-like (about 1.5 in "
        "log-odds), while rotations of the same size that keep the split change almost nothing. Holding the network's "
        "internal mass direction fixed removes most of its response to a real mass change but almost none of this one, "
        "so the network uses the momentum sharing through a second route besides its mass variable. Which internal "
        "objects carry that route is not identified; it is spread over several of them. Round 17 adds that the response "
        "appears in 9 of 10 independently trained copies of the network, and that in the fourth layer one direction "
        "carries most of the average W-jet and Z-jet response when copied in, while the Z-jet part is also carried by "
        "the mass subspace (see A4s v2)."
    ),
    'A4s v2': (
        "Inside JEDI-linear's fourth layer there is one direction that acts like a signed 'momentum sharing' setting: "
        "copying it from a jet whose two prongs share momentum more unevenly reproduces about 85 % of the average change "
        "in the W-versus-Z score, and pushing an ordinary jet along it moves W jets toward Z and Z jets toward W. It is "
        "separate from the direction that carries a real mass change, and the response it belongs to appears in nearly "
        "every trained copy of the network. Removing it takes away about half of the W-jet response but none of the Z-jet "
        "response, which other parts of the network also carry, and which jet-shape quantity it computes is still open. "
        "Round 18 identifies it in part: it follows how unevenly the momentum is split between the prongs, plus how wide "
        "the prongs are (see A4s v3)."
    ),
    'A4s v3': (
        "We changed only how unevenly a W or Z jet's momentum is split between its two prongs, keeping the prong "
        "separation and the prong masses fixed, and compared with changing the separation or stretching the jet by the "
        "same mass change. The fourth-layer sharing coordinate follows the split: the other two changes move it about "
        "8 % as much, and shape variables that mix split and separation (D2, τ21) predict the wrong sign. It also "
        "responds to how wide the prongs are, so it is not a clean read-out of the split alone. Pushing the coordinate "
        "directly moves Z jets toward Z below about 85 GeV and toward W above, while W jets move toward Z up to about "
        "95 GeV, so its effect depends on the jet's class as well as its mass; and the Z-jet response is also carried "
        "by other directions in the same layer. Round 19 (E37y) qualifies two parts: the class dependence follows the "
        "network's last-layer state rather than the true class, and the prong-width part is not identified (see A4s v4)."
    ),
    'A4s v4': (
        "Pushing the network's prong-sharing coordinate moves W and Z jets of the same mass differently. W and Z jets of "
        "the same mass also start in different states of the network's last hidden layer (which mass-window neurons are "
        "on, how strongly W is already favoured), and once jets are matched on that state the difference disappears, so the "
        "response follows the network's state, not the true class. Holding three mass-window neurons fixed removes about "
        "0.6 of the difference, not clearly more than random holds of the same size. Two edits that both widen the prongs "
        "at fixed prong mass move the coordinate in opposite directions, so what besides the momentum split it reads is open."
    ),
    'A1s': (
        "We joined pairs of separately trained JEDI-linear networks: the first network's internal object layer (or the "
        "next layer) is passed through one fitted linear map into the second network, which finishes the decision. The "
        "joined network gives the second network's decision on 86–89 % of jets, about as often as the two networks "
        "agree on their own (90 %), against 34 % with an untrained first half and 66–78 % when only the first network's "
        "class scores are passed. So what these layers hold is shared between retrains up to a linear map and is more "
        "than the final answer; whether the networks compute it the same way is not shown, and of the edit responses "
        "only the Z-jet prong-sharing response can be assigned (to the receiving network). Round 19 (E39x) qualifies "
        "\"more than the final answer\": that holds only for maps built to copy the layer; maps trained on the receiving "
        "network's own task get as much from the class scores (see A1s v2)."
    ),
    'A1s v2': (
        "We joined two separately trained JEDI-linear networks with a linear map that passes only a few numbers from "
        "the first network's internal object layer to the second, trained so the second network classifies well. Four "
        "numbers give the second network's decision on 87 % of jets, close to how often the two agree on their own, but "
        "passing the first network's class scores the same way gives the same 87 %, and passing the whole layer adds only "
        "about 2 points. So these joins do not show that retrained networks share anything beyond their final class "
        "scores; the hand-named object units carry a little more than random units but reproduce only a minority of "
        "decisions on their own."
    ),
    'A19qg': (
        "Keeping the jet's total momentum and mass exactly fixed, splitting any particle into two moves JEDI-linear's "
        "quark/gluon score toward \"gluon\", and merging two particles moves it toward \"quark\", so the network counts "
        "particles; such a count is not infrared- and collinear-safe (IRC-safe means unchanged by such splits and by adding "
        "particles of vanishing momentum). Splits act at every distance from the jet axis, but a very soft added particle "
        "still counts only away from the jet core (beyond about 0.15 in angle). Inside the network the count enters through "
        "a wide-angle sum that ignores particle momentum, an empty-slot sum, and a soft-multiplicity sum where particles "
        "cross a pT-rank window edge; no single sum passes the multiple-test correction, and why the effect weakens toward "
        "the jet edge is open."
    ),
    'A18': (
        "We let a search find, for each jet, the smallest change of its particles that flips JEDI-linear's call while "
        "the jet's mass, momentum and axis stay fixed. These changes flip the real network on about 40 % of jets, against "
        "about 2 % for random changes of the same size. Over all jet types they mostly move momentum among the hardest "
        "particles, but the changes that turn W into Z mostly move particles toward or away from the axis, and moving "
        "momentum into the hardest particle alone flips few calls. The changed jets often look unlike real jets, so this "
        "maps the network's sensitivity rather than how it treats natural jets. A follow-up (round 17) found that the "
        "internal objects tied to the hardest particles carry about half of the gluon-to-quark flips and none of the W/Z "
        "flips, which run mostly through the radial (mass-like) objects (see A18h)."
    ),
    'A18h': (
        "We moved part of the momentum of one of a jet's hardest particles to another, keeping the jet's mass, momentum "
        "and axis fixed, to see whether JEDI-linear has a separate 'hardness' channel. The W-versus-Z score changes less "
        "than under a random reshuffle of the same momentum among soft particles, and even its sign depends on what else "
        "is adjusted to keep the mass fixed; with particle directions left alone the effect is signed and about 1 % of the "
        "score spread. The internal objects tied to the hardest particles carry about half of the gluon-to-quark minimal "
        "flips and none of the W/Z flips, so for W versus Z there is no separate hardness channel; the gluon-to-quark "
        "share is not significant after correcting for the four flip groups tested."
    ),
    'A1x v3': (
        "Ten separately trained JEDI-linear networks were compared with hundreds of random formulas that fit the "
        "reference network equally well. A unit that sums the particles' momentum times their squared distance from the "
        "axis reappears in 8 of the 9 other networks, which random formulas with the same number of fitted terms do in "
        "about 1–5 % of cases, so this recurrence is above chance at that level. The momentum weighting is what recurs; "
        "the squared-distance weight holds only for particles close to the axis (within 0.2), and the soft-particle count "
        "and hardest-particle window are not distinguished from random formulas."
    ),
    'A15r': (
        "We tested four ways a collinear split could reach JEDI-linear's decision. Two are excluded: the cut that "
        "switches on for particles below 11 GeV, and the softest particle being pushed out of the 64 input slots. Two "
        "carry part of it: the softer particles moving down one pT-rank position (up to about a third of the changed "
        "decisions), and the two internal sums that count empty input slots, which also carry 40 % of the effect of "
        "adding one soft particle. The rest is spread over many internal sums and is not yet assigned. On the "
        "quark/gluon score (round 15, E34v), extra particles near the jet core act through the same two empty-slot sums "
        "(about 0.8 of the response), while extra wide-angle particles act through a different sum. Round 19 (E40) "
        "separates two routes on the quark/gluon score: a split that pushes particles across the rank-13 window edge acts "
        "through the soft-multiplicity sum (about 0.8 of it, and nothing when no particle changes rank), while splits of "
        "soft particles act mainly through a wide-angle sum that ignores pT and an empty-slot sum; no single sum passes the "
        "multiple-test correction."
    ),
    'A4w v2': (
        "With the pT-weighted squared spread of the particles held fixed, smoothly reshaping how far particles sit from "
        "the jet axis still moves JEDI-linear's W-versus-Z decision. That response is predicted well by a single sum of "
        "pT times distance to a power slightly above 2, better than random smooth radial weightings. So the network's "
        "radial weight is close to, but measurably not exactly, the square in the mass formula; whether it is one power "
        "or a smooth weight near r² is open."
    ),
    'A10w': (
        "A Linformer, a transformer variant, weights particles by their pT rank through a learned table. Sharp jumps in "
        "that table sit where its random starting values put them and do not recur between training runs, so it does "
        "not form JEDI-linear's hard-edged rank windows. All four trained networks do give the five hardest particles "
        "more than twice the typical weight, which random tables never do, and removing that emphasis costs much "
        "accuracy. Open: whether that cost reflects the learned emphasis or simply that hard particles carry most of the "
        "jet's momentum."
    ),
    'A1f': (
        "The more accurate official JEDI-linear with 16 input features does not build its angular information from "
        "half-plane cuts as the 3-feature version does. Its first layer ignores the separate Δη and Δφ inputs and reads "
        "each particle's distance ΔR from the jet axis, often in the same cut as its pT, so its second-round sums are radial "
        "because their input already is. The detailed circuit therefore depends on which inputs the network is given, "
        "while the mass behaviour (claim A4) is the same in both. There is no usable untrained comparison for this model."
    ),
    'A13': (
        "On W-like jets with two prongs, moving the softer prong around the jet axis changes JEDI-linear's decisions, and "
        "this works through its jet-level summary units. Breaking that prong apart instead, with every particle's momentum, "
        "distance from the axis and the jet mass kept the same, changes the same share of decisions, while a classifier "
        "using the two-prong variable tau21 (N-subjettiness ratio) changes about 4 points more. So JEDI-linear responds to "
        "where the prong is, not detectably to whether it is intact; a small effect, or one hidden behind its response to "
        "jet mass, is not ruled out."
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
        "also respond to the shift about as much as to a real widening of the same spread. A comparison of 56 trained "
        "networks on the same jets (round 10, X13) finds the response in all of them; its size follows how much of each "
        "network's mass estimate is tied to the stored axis, which is fixed by the network type for set-based networks and "
        "varies between training runs for transformer-type networks (claim A10s)."
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
    'B2o': (
        "We tested whether the programs extracted from JEDI-linear on ordinary jets also follow it on edited jets they "
        "were not built for. When the momentum split between a jet's two prongs is made more lopsided at fixed mass, the "
        "floating-point copy of the network with its zero-bit masks changes the same calls as the network in about 85 % "
        "of cases, while the hinge-formula program gets the average shift right but changes many extra calls. On the "
        "smallest call-flipping edits found by a search, the programs predict which jets flip only slightly better than "
        "a rule that always says \"no change\", so faithfulness off ordinary jets depends strongly on the edit. Adding "
        "such edited jets, labelled by the network, to the program's training data does not change this (round 17, B2c)."
    ),
    'B2c': (
        "We refitted the extracted programs and standard surrogates with edited jets added to their training data, each "
        "labelled with the network's own output, to see whether this teaches them the network's response to edits. On the "
        "network's own smallest call-flipping edits they still do little better than a rule that always says \"no change\" "
        "(agreement score κ about 0.2, where 0 is that rule and 1 a perfect match), even when those edits are in the fit. "
        "An improvement on the prong-sharing edit appears just as much when ordinary jets in rotated form are added, so "
        "it comes from adding data, not from the network's labels; refitting on another sample shifts the scores by about "
        "as much as these effects."
    ),
    'B4 v3': (
        "We scored compact imitations of the network by how well they reproduce its decision changes when every particle "
        "of a jet is shifted by 0.06 rad. The large staged program does this better than formulas or trees of the same "
        "size on standard jet variables or energy-flow polynomials, but formulas and trees on the program's own input "
        "quantities do as well with far fewer bits, and a rule that only checks whether the network called the jet W or "
        "Z scores nearly as high. So the advantage comes from the choice of inputs, not from the staged structure, which "
        "adds a small gain only at the largest program size."
    ),
    'B2m': (
        "The question is whether the program's variables are variables inside the network: if a variable's value is "
        "swapped between two jets inside the network, does the decision change as an input edit of that variable predicts? "
        "Swapping the program's per-unit formulas works no better than swapping random directions, but learned "
        "one- or two-dimensional directions for the total pT and the pT-weighted spread do reproduce most such decision "
        "changes. So the physics quantities are present as internal variables, not in the basis of the fitted formulas, and "
        "the log of their product is not found as one variable. In round 18 the whole program's internal pieces, swapped "
        "between jets at the network units they were fitted to, reproduced the network's new decision in about 0.6 of "
        "decision-changing swaps against 0.25 for mismatched units; that match is partly built in by the fitting."
    ),
    'B3': (
        "Agreeing with the network on ordinary jets does not show that a surrogate computes the same thing. A formula "
        "fitted directly from physics observables to the class neurons agrees on most ordinary jets but sends almost no "
        "shifted W jets to Z, because it uses the invariant mass, while the nested program follows the network; programs "
        "of equal size on axis-relative and on shift-invariant observables differ by about one point on ordinary jets and "
        "by more than half of all decisions under the shift. The direct formula's failure depends on its inputs: without "
        "the invariant mass among them it does follow the shift, and no fitted program reproduces the network's dependence "
        "on the shift direction. So the agreement rate on ordinary jets cannot separate mechanisms; input edits that "
        "separate candidate mechanisms can. A later test (2026-10-20) found the same with formulas from symbolic "
        "regression and with trees on standard jet variables: close agreement on ordinary jets, but wrong decision "
        "changes under a shift that keeps the axis spread fixed."
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
        "Round 18 (B8, B8v): scored on how well it reproduces the network's decision changes under a small rigid shift of "
        "the jet, the large staged program beats formulas and trees of the same size built on standard jet variables or "
        "energy-flow polynomials, but formulas and trees built on the program's own input quantities do as well with "
        "fewer bits, so the lead comes from the inputs, not from the staged structure. "
        "Round 17 (B7v): with network-labelled edited jets in the fit, the program again gains no more than a boosted-tree "
        "model of the same size. "
        "This was not supported as a compactness claim. Standard surrogates fitted on the same jets, MARS (forward-selected "
        "hinge terms) fitted directly from physics inputs to the class scores and gradient-boosted trees, reach the "
        "agreement of the small nested programs with many fewer fitted pieces, and the largest nested programs are within "
        "about one point of them. Under the jet shift the nested programs agree with the network somewhat more often than "
        "the best boosted trees, but they lead by a similar margin under a random displacement that raises the axis spread "
        "by the same amount, so the lead is not specific to the shift mechanism. The surrogate the shift separates is the "
        "direct physics-to-score MARS fit, which uses the invariant mass and misses most of the network's changed decisions; "
        "symbolic regression was not run in that first comparison. A later comparison on four networks (2026-10-20) found that, once the boosted "
        "trees' settings are tuned on held-out jets, the if-statement program and trees of the same size follow the "
        "networks' changed decisions under the shift equally well, within about 2 points either way; the program keeps a "
        "small lead only for a shift that holds the axis spread fixed, in JEDI-linear and the transformer. With symbolic "
        "regression added and more samples, the program's lead over trees on the same sums swings from negative to "
        "positive depending on the sample, and is the same for a shift in a random direction."
    ),
    'B6t': (
        "The same extraction procedure, if-statements on sums over pT-ranked particle windows fitted from the inputs to the "
        "class scores without looking inside the network, works on Deep Sets, a Particle Flow Network and a small "
        "transformer as well as on JEDI-linear. On each, it follows how the network changes its decisions when the jet is "
        "shifted about as well as boosted trees, decision trees or small neural networks of the same size whose settings "
        "were tuned on held-out jets, but not better; much larger surrogates follow the network more closely. Symbolic "
        "regression was tested later (2026-10-20): on standard jet variables it falls behind, but so do trees on those "
        "variables. The size measure (32 bits per stored number) favours the program."
    ),
    'B6s': (
        "Formulas found by symbolic regression on standard jet variables (mass, groomed mass, N-subjettiness ratios, "
        "multiplicity, radial moments) reproduce the networks' decisions on ordinary jets almost as well as the "
        "if-statement program of the same size, but do not follow how those decisions change when the jet is shifted "
        "sideways; under a shift that keeps the axis spread fixed they do worse than a surrogate that never changes its "
        "decision. Trees and hinge fits on the same standard variables fail the same way, so the cause is which variables "
        "the surrogate is given, not the search method. Uncertain: the size of the gap depends on the sample and on "
        "which formula is selected; some formulas of similar size match the program in JEDI-linear."
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
        "little of the mass information that reaches the decision (C2b). ParTs retrained from scratch also develop such "
        "switch heads in four of six cases, but at different heads and masses, so this head is a property of the "
        "released checkpoint."
    ),
    'C2b': (
        "When the jet mass is changed, the change of ParT_full's W/Z output unit (n98, a coordinate of the class token "
        "that is high for W and low for Z) passes almost entirely through the first class-token attention layer, about "
        "half through one head; the 87 GeV switch head carries about 5 %. Most of the change is in what the particles "
        "carry, not in where the heads look. A three-unit circuit made of the two heads and n98, with everything else "
        "replaced, separates W from Z barely above chance (about 52 % against 83 % for the model), but under the same strict "
        "definition even all 16 heads reach only 63 %, so a small circuit that works on its own has not been found yet. "
        "The same first layer also carries the response when all particles are shifted off the jet axis at fixed "
        "invariant mass (99 %), so it reads the network's axis-relative mass, whatever its source. ParTs retrained from "
        "scratch on 2 million jets pass only 63–84 % through this layer, a range untrained networks also reach, so the "
        "concentration in one layer is a property of the released checkpoint."
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
        "Six Particle Transformers retrained from scratch on 2 million jets all keep the released network's main "
        "behaviour on two separate test samples: shifting every particle by 0.06 rad in azimuth while the stored jet axis "
        "stays where it was (a rotation about the beam line, which leaves the jet mass unchanged but widens the spread "
        "about the axis) turns 14–20 % more true W jets into Z, while untrained networks of the same design barely respond. "
        "The released network's internal route (almost all of the mass response through the first class-attention "
        "layer, one dominant head) is not reproduced; the retrained networks sit where untrained ones already do. "
        "Whether this difference comes from the random seed or from the 50 times larger training set of the released "
        "network is not yet separated."
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
        "Particle Transformers trained from scratch on 1.5 million W, Z and QCD jets, either on the true labels or only "
        "to flag a mass window, respond to a shift of the jet axis that leaves the invariant mass unchanged at 0.2 to 0.5 "
        "of their response to an equal real mass change, depending on the shift direction; the released models respond "
        "at about 0.75 to 0.85 in every direction. Trained with random axis shifts, the same network ignores the shift "
        "(ratio about 0.01) at the same accuracy, so the architecture does not force the axis-relative response. Which "
        "part of the released training (10 classes, 100 million jets or a longer schedule) gives the larger response was "
        "untested after round C6. (Round C7.) Training on all 10 classes with the same 1.5 million jets and step count raises the "
        "ratio by about 0.1 (to about 0.3), so the 10-class objective contributes but explains only a small part of the "
        "gap; runs with 5 million jets and with five times the training steps are still training, with one seed per run. "
        "(Round C9.) Those runs finished: neither 5 million jets with 10 classes nor five times the training steps brings "
        "the ratio above about 0.4, against 0.72 to 0.87 for the released models, so what produces the released models' "
        "larger axis response is still not identified; each run has one seed."
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
        "A boost of every particle along the beam keeps a jet's invariant mass but moves its particles relative to ParT's stored jet axis. Our L-GATr, a tagger built to respect the symmetries of special relativity, does not change its W-versus-Z score under this boost, while ParT responds at about 0.8 of a real mass change. So the axis-dependent mass reading found in ParT is absent in L-GATr. Whether L-GATr reads the invariant mass itself or the spread of particles about its own momentum direction is not yet separated, because the fixed-mass rearrangements tested keep both unchanged."
    ),
    'C23': (
        "ParT's attention is mostly a fixed formula of particle-pair quantities; our Lorentz-symmetric tagger L-GATr's "
        "is not. A fixed pair formula describes about a fifth of a typical L-GATr attention head, about as much as in an "
        "untrained copy. The heads the decision needs are in the first five blocks, and putting the formula there keeps "
        "about half the accuracy (50 % against 85 %) only while the network's own attention to its reference tokens is "
        "left in place, and none when that is fixed. Whether adding particle-identity information to the formula "
        "changes this is untested."
    ),
    'C24': (
        "ParticleNet's first graph layer does not compute a fixed formula of pair quantities that ignore orientation: it "
        "also uses how far a neighbouring particle sits from the jet axis compared with the particle itself. Taking "
        "that away lowers ParticleNet's response to a mass-preserving axis shift, from about 0.37 to 0.26 of a real "
        "mass change in the kinematics-only model, so it moves further from ParT, not closer; about half of that drop "
        "also appears with an equally poor fit that keeps the information. A training control for the kinematics-only "
        "model is still missing."
    ),
    'C25': (
        'In L-GATr each hidden channel of the feed-forward layers is switched on and off by its own scalar part. Read one at a time, these channels track a class or jet quantity more often than mixed directions of the same layer (about 19 % against 13 %), and this appears only after the switch. Acting on single channels does not move the network more specifically than acting on mixed directions, so they are a preferred basis for reading information but not separately controllable parts. Whether a few channels alone can carry the decision is untested, because the test used could not exceed 36 % accuracy even with all channels kept.'
    ),
    'C6 v3': (
        "L-GATr, the Lorentz-equivariant tagger, does not use the jet axis stored in the data file, and when the jet mass is changed in ways that keep every particle physical, its W-versus-Z score follows the invariant mass: a 5 % mass rise made by scaling all momenta moves it 0.84 as much as the same rise made by widening the angles. A larger angular spread at the same mass reaches it only through its per-particle angle inputs, at about an eighth of a real mass change. Making particles heavier or lighter than real particles also moves the score, which is why earlier particle-mass edits gave misleading ratios; this reading rests on one verification run and still needs an independent repeat."
    ),
    'C27': (
        "The axis dependence seen in ParT is tied to how the particle angles are measured, not to the type of network. The Lorentz-equivariant L-GATr, retrained with its particle angles measured from the stored jet axis that ParT uses, starts responding to particles moved off that axis at fixed invariant mass, at 0.38 of its response to a real mass change, where the unmodified network gives zero. This is from one early checkpoint of a training that is still running and has not been independently checked; the value is about half of ParT's, and whether it grows with training is open."
    ),
    'C28': (
        "Inside L-GATr, one direction among the 32 plain scalar channels each token carries holds about 40 % of the network's response to a change of the jet mass from the fourth block on: copying it between jets of different mass transfers that share, and holding it fixed removes 37–49 % of the response, far more than random directions. Under edits that keep particles physical it follows the invariant mass. The rest of the mass information is spread over many channels, so no small set of directions reproduces the network's calls."
    ),
    'C26': (
        "No set of 16 or fewer of L-GATr's 96 attention heads, kept alone, reproduces 80 % of its W-versus-Z or top-versus-QCD decision, whichever of three rules is used to switch the other heads off. When switched-off heads are replaced with outputs from jets pointing the same way, only six heads in the first three blocks matter individually and 32 heads recover about 0.7 of the top-versus-QCD decision. How small a faithful circuit can be, and whether the chosen heads beat random ones, remain open."
    ),
    'C3': (
        "ParT, which also receives shift-invariant pair features from which the invariant mass could be computed, still "
        "responds to a shift of all particles that keeps the invariant mass fixed: more of its W jets become Z calls, and "
        "its internal mass read-out rises by somewhat less than the axis-relative formula predicts. In ParT_full (ParT "
        "with all input features), feeding the unshifted per-particle distance to the axis removes almost all of this "
        "response, and under real mass changes the per-particle inputs carry most of the change while the pair features "
        "carry about a tenth to a fifth of the W-to-Z change. So ParT's mass sense comes mainly from particle positions "
        "measured from the axis, as in JEDI-linear, with a small nonzero contribution from the pair features. For mass "
        "decreases the two routes do not add, so a share is not defined there. The response reproduces on further fresh "
        "jets and with a second read-out of the network's output."
    ),
    'C4': (
        "ParT does not build its jet variables with the mass last, as JEDI-linear does. Particle count and pT can be read "
        "from the input embedding and the mass from the first block, but an untrained network of the same design gives "
        "nearly the same read-outs, so these do not show learning. Prong-shape variables (tau21, tau32, which measure "
        "whether the energy sits in one, two or three clusters) are read only by the trained network and build up over "
        "blocks 1–4. Edits that change one of them at a time at fixed mass (round C5) show that the network does not "
        "follow any of them as a single number, so what this trained stage reads is still not identified."
    ),
    'C11': (
        "At fixed jet mass, ParT does not respond to the standard prong variables the way classifiers built on them do. "
        "Making the two prongs of W, Z and top jets narrower, which lowers tau21 (a measure of how two-pronged a jet is) "
        "by about 11 %, makes ParT about 0.1 less W-like in logit, while a classifier on mass and tau21 predicts the "
        "opposite; QCD jets do follow that classifier. The edits also change subjet masses and the placement of soft "
        "particles, so which quantity ParT reads instead is not yet known."
    ),
    'C12': (
        "When all of a jet's particles are shifted by a small angle with the mass unchanged, about 93 % of ParT's "
        "response is the same for a shift and its mirror image, and its size follows the jet's response to a real "
        "widening of its angles, as expected if ParT reads the angular spread about the stored axis (the second moment "
        "M2). The direction-dependent centroid term found in JEDI-linear is not seen; ParticleNet has a larger "
        "direction-dependent part. At fixed shift size the response still varies with direction in a way not explained."
    ),
    'C13': (
        "Sophon, the Particle Transformer architecture trained by its authors on the larger JetClass-II dataset with "
        "mass-decorrelated training, has a W/Z output that barely reacts to a real change of the jet mass. Its internal "
        "size reading moves about 0.2 as much under a mass-preserving shift of the jet axis as under a real mass change, "
        "against 0.8 to 1.0 for the JetClass ParT, so the axis-relative mass is not a property of the architecture. "
        "Half of Sophon's size reading still uses the axis-relative angles, and which training difference causes the "
        "change (data size, decorrelation or normalised inputs) is unknown."
    ),
    'C14': (
        "When a single particle is moved slightly away from the jet axis, ParT and ParticleNet respond on average more "
        "for particles with more momentum and larger distance from the axis, at 0.6 to 0.8 of the slope that the "
        "angular second moment M2 predicts. For individual particles no simple formula, including JEDI-linear-style "
        "momentum-rank windows, explains more than about 8 % of the response. In ParT the rest comes in through the "
        "features between pairs of particles and changes non-smoothly with the step size; a later run traced it to "
        "particles with close neighbours (C16)."
    ),
    'C15': (
        "In ParT and ParticleNet models that are given particle-identity flags, the decision that a jet comes from a Higgs "
        "boson decaying to a lepton, a neutrino and two quarks rests almost entirely on the electron or muon flag of one "
        "particle: on two fresh sets of jets, removing only that flag moves about 98 % of these jets to other classes, "
        "against under 1.3 % for the same kind of label change on a hadron, and adding a muon label to a hadron in a jet "
        "without a lepton creates this decision in 18 % of jets. One attention head of ParT's final class stage reads the "
        "lepton; kept alone, with the rest of the network averaged out, it gives a lepton-present score but no decision, "
        "and that score does not separate these jets from top-quark jets with a lepton. Which parts of the full network "
        "turn the lepton label into the decision is not yet known."
    ),
    'C17': (
        "This is a planned test, not yet run: whether ParT's decision that a jet comes from a top quark decaying to three "
        "quarks is carried by a handful of attention heads, or spread over many heads as its W-versus-Z decision is. The "
        "test keeps only a few heads, averages out the rest of the network, and compares the result with random head sets "
        "of the same size. The first attempt failed on a broken cluster machine, so nothing is known yet. (Round C9.) "
        "The test ran: a few attention heads carry most of the top-quark decision, but keeping only them and averaging "
        "out everything else does not reproduce the top-quark call, and they do no better than random heads from the same "
        "final stage of the network. The stand-alone circuit stated here is therefore not supported; see C17 v2 and C17s."
    ),
    'C17 v2': (
        "Five attention heads in ParT's final class-token stage, where one summary token reads all particles, carry 59 % "
        "of the drop in the top-quark score that removing every head causes, against about 21 % for five random heads from "
        "the same stage. The same heads carry about two thirds of the response when the b-quark prong is merged into one "
        "particle at fixed jet momentum. Kept alone, without the rest of the network, they do not give the top-quark "
        "call, so they are a main route for the decision rather than a complete circuit; one network and two samples."
    ),
    'C17s': (
        "It is unsettled whether each jet class has its own attention heads in ParT. Removing the heads ranked for the "
        "top-quark call lowers mainly the top-quark score, about 3 standard deviations beyond random heads, which is only "
        "marginally significant after correcting for ten classes. On the other hand, the heads each class depends on "
        "one at a time come largely from the same small pool, so the classes may share most of their heads."
    ),
    'C18 v2': (
        "When all particles of a W or Z jet are shifted together at fixed jet mass, ParT responds to how far the "
        "particles spread about the stored jet axis, not to the jet's momentum direction moving off that axis: a shift "
        "that moves the direction but keeps the spread gives at most 0.2 of the response. Such shifts cannot tell the "
        "spread apart from the offset of the particles' momentum-weighted centre. In unedited jets that offset is set by "
        "the jet's pseudorapidity and width, so its effect on real decisions is inferred from edits, not observed."
    ),
    'C19': (
        "The 1,024 hidden neurons in ParT's final class-token layers are, as a set, no easier to read than random "
        "combinations of them: about half respond selectively to one class or physics quantity either way, and nudging a "
        "single neuron moves its labelled class the predicted way only about half the time, usually by a negligible "
        "amount. The predicted neuron-by-neuron readability is therefore excluded on two samples."
    ),
    'C19r': (
        "A small minority of the neurons in ParT's final class-token layers have clear causal roles that hold on fresh "
        "jets. Among them, a few that follow the jet mass measured about the jet axis reproduce about a quarter of the "
        "network's W-versus-other response to a real mass change, and a similar share under a sideways shift of the jet "
        "that keeps the mass. Which neurons count as mass neurons changes between samples, and two neurons carry much of it."
    ),
    'C20': (
        "The full-input ParT and ParticleNet change 8 to 12 % of their decisions when the jet is mirrored in azimuth. "
        "When the particle charges and transverse impact parameters are flipped as well, so that the result is the mirror "
        "image of a real jet, only about 3 % change, close to the kinematics-only models. The mirror sensitivity therefore "
        "comes from the sign pattern of charge and impact parameter, not from the jet axis."
    ),
    'C18 v3': (
        "The spread of a jet's particles about the stored jet axis has two parts: the spread about the jet's own "
        "momentum-weighted centre, and how far that centre sits from the axis. ParT responds to a change of the first "
        "part almost exactly as to a real mass change of the same size, and to the second at about three quarters of that, "
        "for offsets from 0.005 to 0.06 rad. ParticleNet gives the offset less weight, by an amount that depends on its "
        "direction and size; the spread edits were possible only in jets with two hard particles of similar momentum."
    ),
    'C21': (
        "ParticleNet (a graph network) and ParT (a transformer) build compatible early particle features: ParticleNet's "
        "features after its first or second layer, passed through one linear map, can replace ParT's own at ParT's first "
        "blocks at a cost of about 1 to 2 points of accuracy, against about 9 for an untrained ParticleNet. How strongly the "
        "combined network responds to a mass-preserving shift of the jet axis depends on both the features and the layers "
        "that read them. Only the full-input models and one read-out were tested."
    ),
    'C22': (
        "Our L-GATr, a tagger built to respect Lorentz symmetry, decides from its Lorentz-invariant channels, and two groups "
        "of them hold overlapping copies of the same information, so removing either group alone hides how much it carries. "
        "A boost along the beam changes about 3 % of its decisions, only through the inputs that tell it the lab frame (a "
        "time reference vector and the particle energies). Results are from an interim checkpoint; whether it, unlike ParT, "
        "ignores the jet axis at fixed invariant mass is still untested."
    ),
    'C18': (
        "In real jets the momentum-weighted centre of the particles sits a few hundredths of a radian from the stored jet "
        "axis. Moving the particles so that this centre lands on the axis, without changing the jet's mass, changes ParT's "
        "W-versus-Z score about as much as a real mass change of the same size in the spread about the axis (ratio about "
        "1.0). So ParT measures mass about the stored axis also for offsets of natural size; the effect is about a 1 % mass "
        "shift. The edited jets are no longer exactly natural, because the edit moves their momentum direction off the "
        "stored axis, so the effect on unedited jets is still to be measured."
    ),
    'C16': (
        "When one particle in a W or Z jet is moved slightly, the part of ParT's response that does not depend on the "
        "direction of the move comes from that particle's pairs with close neighbours (within about 0.05 rad) and passes "
        "through two of the eight attention heads that use the pair features. A first-order copy of the network given the "
        "same input changes produces only about a third of it, so most of it is made by the network's own nonlinear "
        "response rather than by the logarithm of the pair distance. Measured on W and Z jets in two models only; in the "
        "kinematic-input model, single-precision rounding of one pair feature also changes about 1 % of decisions."
    ),
    'D1': (
        "In earlier physics edits on JEDI-linear, changing only the mass of W jets (by scaling angles) to the Z mass turns "
        "almost all of them into Z calls, lowering the jet momentum by 10 % turns most W jets into top calls, and removing "
        "the W inside a top jet barely changes the top call. So JEDI-linear tags W, Z and top mainly by mass windows plus "
        "the total pT, and does not check that a top jet contains a W. The momentum-scaling result was reproduced at 10 "
        "times the statistics on two fresh samples (W call 87 % → 27 %, rotation control unchanged). With a control that "
        "holds the W pair fixed, JEDI-linear does not respond to the W-pair mass inside top. An earlier extension of this "
        "null to ParT is withdrawn: it came from feeding the edited jets about the unedited jet's axis, and when each "
        "edited jet is rebuilt about its own axis, JetClass ParT and ParticleNet do use the W-pair mass (claim D11), as "
        "the Top Landscape taggers do (claim D6). "
        "The momentum edit used here kept the jet mass fixed (it is a boost along the jet axis); scaling only the momenta "
        "lowers the W call to 48 % instead, and most calls changed by the boost are W or Z jets becoming top (claim D9). "
        "The JEDI-linear W-pair null was retested with each edited jet rebuilt about its own axis, the build that "
        "overturned the ParT null: JEDI-linear's response then lies within the range of retrained classifiers that never "
        "see the pair mass, and below every classifier given the pair mass at 100 GeV, on two samples. Still uncertain: "
        "at 90 GeV JEDI-linear raises its top score while every reference lowers it, and on hls4ml jets the classifiers "
        "with and without the pair mass are equally accurate, so the test for pair-mass use is weak. On W and Z jets "
        "JEDI-linear also responds to how momentum is shared between the two prongs while its mass variable and total "
        "pT stay fixed (claim D18), so it uses more than mass windows plus total pT there."
    ),
    'D2': (
        "In earlier physics edits on ParT_full (400 jets per class), making all tracks prompt removes most H → bb̄ calls "
        "and shrinking all displacements keeps most of them, but keeping only two displaced tracks removes many, so ParT "
        "responds to the number of displaced tracks more than to how far they travel. Removing lepton isolation lowers the "
        "leptonic-top calls, and edits of π⁰ → γγ pairing and of the heavy-quark dead cone have no measurable effect. At 10 "
        "times the statistics the same signs return; a sharper test caps every track's displacement significance just "
        "above threshold, which keeps the number of displaced tracks and costs 3 points, while keeping only two displaced "
        "tracks costs 61. Small new effects (lepton charge in H → ℓνqq′, charm dead cone) were not rerun. The count "
        "statement holds for H → bb̄ against QCD; for H → bb̄ against H → cc̄ the networks also respond to the "
        "displacement values at fixed counts (claim D19), without evidence that this is a lifetime reading. In "
        "H → ℓνqq′ jets the networks also use the lepton's angle to the quark particles, beyond lepton presence and "
        "isolation (claim D20)."
    ),
    'D4': (
        "Measuring particle positions from another standard jet axis (the winner-take-all axis, which follows the harder "
        "branch when the jet is reclustered, or the hardest particle) instead of the summed-momentum axis changes about 30 % "
        "of JEDI-linear's decisions and lowers its accuracy; ParT with all input features changes fewer, ParT with "
        "kinematic inputs only about as many. Most changed decisions are W jets called Z, and the size of the internal mass "
        "change matches the axis-relative formula. So the taggers depend on the axis convention, which the invariant mass "
        "does not. Because they were trained with one axis, any other axis is also a change of input distribution: the "
        "formula predicts the direction of the change, and some changes also occur under a shift of equal size that keeps "
        "the formula fixed. The same holds for taggers trained on top-quark and quark–gluon jets: the other axes change "
        "3–25 % of their decisions, while LorentzNet, whose inputs are four-vectors that do not refer to any axis, "
        "changes none."
    ),
    'D4m': (
        "On hls4ml and JetClass, when the axis convention changes, W and Z decisions move in the direction each model's "
        "own axis-relative mass predicts, but part of the change is general sensitivity to shifted inputs. On Top "
        "Landscape and quark–gluon taggers, the part of the response that depends on where the new axis points is no "
        "larger than in 12–20 untrained networks of the same type, and it changes between two training seeds by about as "
        "much as it differs from zero. So on those datasets the axis response is accounted for by generic sensitivity to "
        "translated inputs; a small learned direction effect hidden inside the untrained range is not excluded."
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
        "which supports the counting part for those models (claim D10). Splitting soft particles into exactly collinear "
        "halves, a pure count change, moves the JetClass W-versus-QCD taggers by 1.2–2.3 times a count-only classifier. "
        "On detector-level top jets (H61v), the same split lowers the Top Landscape ParT top score by about 0.3 "
        "log-odds while moving the same particles by the same small angle without splitting them changes nothing, so "
        "the response is to the number of objects, not to the edit's geometry."
    ),
    'D6': (
        "We changed the mass of the W-candidate subjet pair inside top jets while holding the total jet mass and the "
        "three-prong measure τ32 fixed. ParT, ParticleNet and LorentzNet, trained on the Top Tagging Landscape, lowered "
        "their top score about as much as a classifier given the W-pair mass, and 2 to 4 times more than a classifier that "
        "sees only the edit's side effects (smaller subjets, lower τ21). So these taggers check the W mass inside the top, "
        "unlike JEDI-linear on the multi-class data. The same models respond the same way on JetClass top jets, so the "
        "result does not depend on the jet sample. PFN and EFN respond only at the side-effect level, and the most "
        "extreme edits produce jets a classifier can tell from real ones. Rebuilding every edited jet about its own axis "
        "leaves these numbers unchanged, because the original test already did so. The same taggers also use the "
        "radiation pattern inside the prongs (claim D13)."
    ),
    'D7': (
        "When soft particles are split into exactly collinear halves, which adds particles and changes nothing "
        "infrared- and collinear-safe, quark–gluon taggers move by only a few percent of what a pure particle counter "
        "does, and a second training seed gives the same small response. This 'no response to the count beyond energy "
        "flow' holds for such hand-made edits only: when Pythia's hadronisation settings are changed so that real jets "
        "carry about 10 % more or fewer particles, the same networks do move toward the count (claim D22). How the "
        "networks would respond to a real-data difference in multiplicity is therefore not settled by the hand-made edits."
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
        "pT-only or angle-only scaling. Most of the hls4ml decisions that change are W or Z jets; identical small taggers "
        "trained on hls4ml or on Top Landscape jets respond alike (claim D12), so the training sample is not the cause in "
        "small networks, and large networks trained on hls4ml have not been tested. A later per-class measurement (H64) "
        "finds the same contrast in log-odds, but simple classifiers on mass and pT trained on each dataset show it too, "
        "so it does not show that the two groups of taggers read different quantities (claim D25)."
    ),
    'D10': (
        "Adding pileup-like soft particles over the jet lowers every network tagger's signal efficiency faster than it "
        "lowers a groomed (soft-drop) mass tagger. JEDI-linear reacts to the mass the particles add, like a plain mass "
        "window (W efficiency 87 % → 43 % at the lowest density). The JetClass ParT and ParticleNet and the Top Landscape "
        "ParT, LorentzNet and PFN react mainly to how many soft particles are added, even when they sit at the jet centre "
        "and add almost no mass. A pure count change (soft particles split into collinear halves) confirms that the "
        "JetClass W-versus-QCD networks count particles while quark–gluon networks barely do, so the count response "
        "belongs to the models and their training data, not to the pileup edit. The pileup model is simplified, none of "
        "the networks was trained with pileup, and part of the Top LorentzNet response comes from the added momentum. "
        "A later test with matched count, momentum and plain mass shows that for the JetClass W/Z networks where the "
        "soft particles sit also matters: wide-angle particles that grooming would remove cost several times more "
        "than the same number added at the jet centre (claim D21). "
        "Version 2 (density scan on four datasets, H65/H65v): no network keeps signal like a groomed or area-subtracted "
        "mass, which keep about 40 % W efficiency at 20 GeV per unit area while the networks keep almost none. The "
        "hls4ml tagger follows the plain mass, the Top Landscape ParT counts particles added without mass, and for the "
        "JetClass full-input ParT most of that count response comes from the particles being labelled neutral hadrons "
        "(as photons, about a quarter). The uniform massless haze is a toy for real pileup. "
        "Round D14 (H68/H68v): when the added soft particles carry the mix of types actually found among soft "
        "particles in JetClass (about three quarters photons), the full-input ParT responds about as much as the "
        "kinematics-only ParT, so the large label effect is specific to particles all labelled neutral hadrons."
    ),
    'D11': (
        "JetClass-trained ParT and ParticleNet lower their top score when the W-candidate subjet pair inside a top jet is "
        "moved off 80 GeV, by about as much as a reference classifier that is given that pair mass directly (0.8–1.25 "
        "times at 60 GeV), like the taggers trained on Top Landscape, and the drop grows with the distance from 80 GeV as "
        "expected for a W-mass window. This reproduces on three independent sets of jets once each edited jet is rebuilt "
        "about its own axis; the earlier nulls came from feeding edited jets about the unedited jet's axis, and these "
        "networks react to such a shift alone. Why the models with particle-ID and track inputs also react to the control "
        "edit beyond that shift is not known."
    ),
    'D12': (
        "Three small top taggers trained on hls4ml jets, on hls4ml jets in a narrow pT window, or on Top Landscape jets "
        "each change 4–7 % of their decisions when the jet is boosted along its axis to 90 % of its pT at fixed invariant "
        "mass, against under 1 % for a matched rotation. The training sample changes that share by about one point or "
        "less, so the sensitivity comes from the architecture, inputs and task; with absolute-pT inputs the training "
        "sample sets whether jets move toward top or away from it. Only two seeds and small networks were tested."
    ),
    'D13': (
        "Making the three subjets of a top jet narrower or wider, while keeping the jet mass and the W-pair mass fixed, "
        "moves every top tagger tested (five trained on Top Landscape, two on JetClass) the way a classifier using the "
        "three-prong variable τ32 predicts, and by a similar amount. When τ32 is changed with each subjet's mass held "
        "fixed, by reshaping the particle profile inside the subjets, the taggers respond with mixed signs. So they use "
        "the radiation pattern inside the prongs beyond the two masses, but not as the single number τ32; both edits make "
        "jets a classifier can tell from natural ones, and which inner-prong quantity each network reads is not known."
    ),
    'D14': (
        "With the number of displaced tracks (tracks that miss the collision point, as b-hadron decay products do) held "
        "fixed, the JetClass ParT and ParticleNet with track inputs lower their top score when the displacement is moved "
        "off the subjet that holds most of it, by about 1.7–2.0 and 1.1 log-odds against a matched control. So they use "
        "where the displaced tracks sit, not only how many there are. They do not follow the b position implied by the "
        "three-prong mass structure: moving displacement onto the mass-assigned b subjet also lowers the score. Still "
        "uncertain: the displacement-heavy subjet may be the true b or a charm quark from the W, which this dataset cannot tell."
    ),
    'D15': (
        "Rotating the W decay inside top jets so that one W daughter is emitted more along the W direction (larger "
        "|cos θ*| in the W rest frame), at fixed top and W mass, lowers the top score of the taggers beyond what the "
        "rotation's side effects on the prong shapes explain. This is clearest for the Deep Sets taggers PFN and EFN "
        "(about −1 to −2 log-odds per unit |cos θ*|); ParT, ParticleNet and LorentzNet show it only on a small set of "
        "JetClass top jets where the control matched the prong shapes. The taggers therefore use the top decay angle, "
        "as the W polarisation predicts, but no control yet reproduces the edit's full change of the three-prong variable τ32."
    ),
    'D16': (
        "Quark–gluon ParT and ParticleNet that see particle types use the sign of the jet charge (the momentum-weighted sum "
        "of particle charges): swapping every particle for its antiparticle, with all momenta unchanged, moves their quark "
        "score by about 0.24 log-odds against the charge sign, about 85 % of what a fit on jet charge alone does, while "
        "random-weight networks barely move. This is consistent with the quark jets of this Z+jet sample being mostly "
        "u-quark jets, a property of the sample rather than of quark versus gluon radiation. That sample property is now "
        "measured: quark jets carry net positive jet charge in both Pythia and Herwig, gluon jets none, and the networks "
        "keep their Pythia charge rule on Herwig jets, where the sample supports a weaker one. They also score soft "
        "photons toward quark and soft neutral hadrons toward gluon, but random-weight networks move by similar amounts "
        "and those relabelled jets are easy to tell from real ones. Round D16 (H59/H59v): networks trained with a "
        "second random seed reproduce the charge response within 7 %; the shuffled-label networks learned a constant "
        "output and give no test, and networks trained with 30 % random labels are still training as the replacement null."
    ),
    'D18': (
        "When momentum is moved between the two prongs of a jet while the jet's total four-momentum stays exactly "
        "fixed, every tagger tested on JetClass and hls4ml changes its W/Z score: balanced prongs look more W/Z-like, "
        "as in boson decays, and asymmetric prongs less, as in QCD splittings (about −2.5 log-odds for ParT on W jets). "
        "A rotation of the same angle that keeps both prongs fixed barely moves the scores, so the response follows the "
        "prong configuration. A classifier that never sees the momentum fraction z predicts the median response, "
        "because at fixed mass z, the prong separation and the prong masses change together, so whether the networks "
        "read z itself is not known. Putting the reclustered prong masses back after the edit leaves the response to "
        "asymmetric prongs unchanged, so it is not a prong-mass effect; but a classifier given the jet mass, the prong "
        "separation and the two prong masses, and not z, reproduces the response to a pure z change, so z is still "
        "not separated from those quantities or from D2."
    ),
    'D20': (
        "When the lepton in an H→ℓνqq′ jet is turned away from the other particles so that the visible mass rises "
        "across the Higgs mass (125 GeV), ParT and ParticleNet lower their Higgs score about twice as much as a "
        "classifier given only the jet mass predicts, with or without particle-ID inputs. Raising the visible mass by "
        "the same amount by turning a hadron instead gives no extra drop, so the networks respond to the lepton's "
        "angle to the quark system, not to the mass crossing 125 GeV. No network shows a two-way step at the "
        "leptonic-top lepton+b endpoint (152.6 GeV). The edit also makes the lepton more isolated, and the exact "
        "lepton–quark quantity read is not known."
    ),
    'D21': (
        "Three soft particles were added to jets with the same count, momentum and plain-mass change, once where "
        "soft-drop grooming removes them and once where it keeps them. The hls4ml W/Z taggers treat both alike, as a "
        "plain-mass tagger does. The JetClass and top taggers react strongly to particles grooming removes and, to "
        "kept particles, follow only about 0.4 of a groomed-mass classifier's response plus an extra penalty that does "
        "not depend on whether the groomed mass moves toward or away from the signal peak. What that penalty reads, "
        "possibly the shape of the prong the particles join, is not known."
    ),
    'D22': (
        "When only Pythia's hadronisation is changed on the same events, Pythia-trained quark/gluon taggers move beyond "
        "what an infrared- and collinear-safe network does, so they use hadron-level information. Settings that move the "
        "particle count and the momentum sharing among hadrons (pT^D) apart show that ParT and ParticleNet do not follow "
        "the total number of particles: at fixed pT^D a 7 % count change moves them by about a quarter of a count-only "
        "classifier's response. They follow how hard the fragmentation is, which pT^D and a count of hadrons above 1–4 GeV "
        "measure equally well; the decisive setting is far from nominal jets, and Herwig-trained networks were not tested "
        "with Herwig-side changes."
    ),
    'D23': (
        "With every tagger set to keep half of the signal jets, top taggers change their calls like groomed-mass and τ32 "
        "taggers when the momentum sharing between prongs changes at fixed jet mass, which a plain mass window cannot "
        "see. They still respond to soft radiation that grooming removes, at about half the rate of radiation it keeps, so "
        "they are not groomed-mass taggers; the hls4ml W/Z networks respond to both alike, as an ungroomed mass does. "
        "Whether the top taggers use τ32 itself or another prong-sharing quantity is open, and the comparison depends on "
        "how the edits are weighted."
    ),
    'D24': (
        "On detector-level top jets made as the Top Landscape training sample was made (Delphes detector simulation, no "
        "multi-parton interactions), a hadronisation change that adds about 7 % more reconstructed particles at the same "
        "mass and three-prong shape lowers the Top Landscape ParT top score by about 0.3 log-odds in two event samples, "
        "while an infrared- and collinear-safe network does not move. Splitting particles reproduces this whichever "
        "particles are split, and moving the same particles without adding any does nothing, so the taggers read the "
        "number of objects at fixed safe substructure as a sign of QCD. Uncertain: the generated jets are still "
        "distinguishable from Top Landscape jets, and the JetClass ParT_kin network barely moves under the hadronisation "
        "change although it responds to the count edit. (Version 1, particle level: a 9 % rise moved ParT by 0.45.) "
        "Version 3 (H66/H66v): on real Top Landscape top jets the four Top Landscape taggers lower their top score when "
        "objects are added and raise it when objects are removed, however the objects are added, while the energy-only "
        "EFN stays flat, and the Top Landscape training data show the same direction at fixed mass and shape (about "
        "−0.025 top log-odds per object). After reweighting the generated jets toward Top Landscape, ParT's hadronisation "
        "shift stays near −0.25 with pair-symmetric weights, but its size depends on which jet of each pair sets the "
        "weight; the networks also respond more per object than the data slope, and QCD-jet responses differ in sign."
    ),
    'D27': (
        "Multiplying every particle's energy in a jet by the same factor changes the jet mass and momentum but no "
        "dimensionless shape. It changed the output of every tagger that sees absolute momenta, by about 0.2–1.1 times "
        "the effect of the same mass change at fixed momentum, while classifiers given only shapes did not move, also on "
        "jets that stay inside the training momentum range. So these taggers read the absolute energy scale, and a "
        "jet-energy-scale error acts on them partly like a mass shift; the dependence is on mass and momentum together. "
        "Still open: how the mass and momentum parts combine, and whether a tracker-only versus calorimeter-only scale "
        "difference reflects particle-type inputs (it appears in kinematics-only models too)."
    ),
    'D28': (
        "When the particles beyond about half the jet radius are removed from top jets and the jet's mass and momentum "
        "are then restored, the large top taggers still lose most of their top calls, while a mass or groomed-mass cut "
        "does not. Putting the lost number of particles back without changing any energy-flow observable does not "
        "recover the loss, so the taggers are not counting particles here; the loss comes with weaker three-prong "
        "structure (τ32, a measure of how three-pronged the jet looks, rises from 0.52 to 0.64). Still open: the "
        "mass-restoring step stretches the jet's angles, so which shape feature the taggers lose is not identified."
    ),
    'D29': (
        "With all momenta fixed, flipping the electric charge of one particle at a time shows that the quark–gluon "
        "ParT and ParticleNet that see particle types weight each particle's charge by roughly its momentum fraction "
        "to the power 0.3–0.4 on quark jets, the weighting used in measured jet-charge observables; a plain count of "
        "charges and the leading particle alone do not describe them. ParticleNet adds several flipped charges almost "
        "linearly, while ParT's response saturates (two flips give about half the sum of the single ones). Still "
        "uncertain: the power is higher on gluon jets (up to about 0.5), and the label-noise null is not yet scored."
    ),
    'D30': (
        "When one classical quark/gluon angularity is moved with the others held fixed, the quark–gluon taggers respond "
        "mostly to edits of how the jet's energy is spread in angle (narrower spread reads as quark), much more when the "
        "energy is pulled toward a common radius than for the opposite move, and a random rearrangement of the same size "
        "does not reproduce it. Moving the particle count or the momentum sharing at fixed angularities gives no "
        "direction-dependent response, so this is not counting. Still uncertain: the energy-flow network, a tree on "
        "angularities and the leading particle's radius all give the same pattern, so the radial variable is not "
        "identified, and the dominant edits produce jets a classifier separates from real ones."
    ),
    'D25': (
        "Version 3 (H69/H69v): with jet mass and momentum held fixed, the full-input JetClass ParT and ParticleNet "
        "respond to a change in how radiation is spread inside the prongs (the energy-correlation ratio D2) 2–4.6 times "
        "more than to a random reshuffling of particle momenta of the same size; their response to how momentum is "
        "shared between prongs, and to τ21, is weaker and less certain. The kinematics-only ParT and the Top Landscape "
        "taggers show no response beyond that reshuffling or beyond their weight-scrambled copies. On signal jets the "
        "jet width cannot be changed without changing D2, so width and D2 cannot be told apart. "
        "Version 2 (H67/H67v): with the jet's mass and momentum held exactly fixed, the JetClass and Top Landscape "
        "taggers still change their output when the jet's momentum is shared less evenly between its two or three "
        "prongs (sub-jets), clearly more than when the prongs are only rotated by the same distance, and they also "
        "respond to a mass change at fixed width. So they read neither the mass alone nor the width and momentum "
        "alone. The test cannot say which second quantity they use (width, prong sharing, D2 or τ21), because the two "
        "edits change the prong sharing equally. "
        "Version 1: "
        "Boosting a jet along its own direction keeps its mass fixed while making it narrower and raising its momentum. "
        "JetClass and Top Landscape ParT and ParticleNet change their output by about 0.1–0.4 of what an equal narrowing "
        "at fixed momentum does, so they do not read the jet's angular size alone. The test cannot tell whether they "
        "compute the invariant mass or combine an angular width with the jet momentum, because a small classifier given "
        "only the width and momentum gives the same ratio; the larger response of the hls4ml and quark–gluon taggers "
        "also appears in such reference classifiers trained on those datasets."
    ),
    'D26': (
        "Changing only the particle-type labels in a jet (charged hadron, photon, neutral hadron), with every "
        "particle's momentum kept fixed, lowers the signal scores of the JetClass taggers that see these labels. The "
        "drop does not follow the jet's charged-energy fraction (the share of jet energy in charged particles): "
        "swapping labels between charged and neutral particles of equal total energy, which leaves that fraction "
        "almost unchanged, lowers the scores 1.5–2 times as much as relabelling 10 % of the energy one way. Whether "
        "this reflects real correlations between particle type and momentum or position, or a reaction to label "
        "patterns that never occur in real jets, has not been tested."
    ),
    'D19': (
        "When the number of displaced tracks (tracks well separated from the collision point) above every "
        "significance threshold is held fixed, ParT and ParticleNet still respond to how displaced the tracks are when "
        "telling H→bb from H→cc. About half of that response also appears when the displacements are changed in random "
        "directions by the same amount, and changing non-displaced tracks moves every class, so part of it is a "
        "reaction to any change of the impact-parameter pattern. In JetClass at equal track counts b-jet tracks are "
        "less displaced than c-jet tracks, opposite to the networks' direction, so use of the longer B-hadron lifetime "
        "is not shown; whether they use the mass of the displaced tracks is open."
    ),
    'D17': (
        "JetClass ParT and ParticleNet, including the versions that see only four-momenta, lower their W, Z and H→bb "
        "score when soft particles near a prong are moved from the side facing the other prong to the far side, and "
        "raise it for the reverse move, with jet mass, τ21, girth and D2 held within 5 %. A classifier on those shape "
        "variables does not respond, so the networks use where soft radiation sits relative to the two prongs, the "
        "colour-flow pattern of a colour-singlet decay. About half of the raw response also appears for a "
        "direction-neutral mirror, and the predicted opposite response on QCD jets is not resolved."
    ),
}

EXPLAINER = {
    'E25_EXPLAINER': (
        "The official 16-feature JEDI-linear reads particle angles only through each particle's distance from the stored "
        "jet axis and ignores the separate Δη and Δφ inputs, so it has no half-plane cuts. Its mass-like response is still "
        "not recentred: moving all particles 0.06 rad off the axis turns about half of true W jets into Z calls, through the "
        "distance column alone, and the response follows the pT-weighted spread of the particles about the axis rather than "
        "any change of the distances. There is no usable untrained comparison, and where the mass forms in later layers "
        "was not tested."
    ),
    'E24X_EXPLAINER': (
        "JEDI-linear's two context-only summary units, S58 and S31, do not read individual particles; they read only the "
        "jet-level summary, and their thresholds follow plain particle-count shapes in a fixed frame: the offset of the "
        "particle-count centre from the axis (S31) and the elongation of the particle-count distribution (S58), each with "
        "about 93–95 % of the threshold variance explained. The elongation name does not explain S58's change under a "
        "180° turn of the jet, and the network's per-slot rounding shapes S31's response to rotations."
    ),
    'E23W_EXPLAINER': (
        "When naturally off-centre jets are shifted sideways, the part of JEDI-linear's response that depends on the jet's "
        "centre offset and the part that depends on the shift size are weighted nearly equally from the dense layers on, "
        "once a measurement bias is removed; only the first summed layer weighs the first part less, as its radial sums "
        "predict. The earlier reading of a strong imbalance came from the estimator. Most of the direction-dependent "
        "response follows the plain average position of the particles, and no unit was found that multiplies total pT and "
        "spread, though that test can exclude only large effects."
    ),
    'E23X_EXPLAINER': (
        "When every particle of a jet is shifted, the part of JEDI-linear's response that depends on the shift direction "
        "passes only partly through the two jet-level units that count particles on each side of the axis (S31 and S58): "
        "holding them fixed removes about a quarter of it in the later layers and none of it in the final W-versus-Z score. "
        "Other second-round units that respond as strongly pass on about twice as much, so most of this response goes "
        "through the units that read individual particles. Which of those units carry it, and how it behaves on W and Z "
        "jets in particular, is still open."
    ),
    'E25W_EXPLAINER': (
        "In the 16-feature JEDI-linear, the total pT and the pT-weighted spread of particles about the jet axis are each "
        "held as a separate internal direction already at the second summed layer, as in the 3-feature network tested the "
        "same way; the two networks differ only in how the combined mass-like quantity develops one layer later. Shifting "
        "all particles off the axis turns W jets first into Z and then into top or quark/gluon calls, and a stretch of the "
        "particle distances that changes the spread by the same amount gives the same number of W-to-Z changes. Ten "
        "non-trained comparison networks respond less, but with ten of them the test cannot reach p below 0.09."
    ),
    'E26_EXPLAINER': (
        "When a jet is boosted along its axis so that its mass stays the same but its pT drops by 10 %, about a third of "
        "JEDI-linear's decisions change. Holding the network's internal sum of particle pT at its pre-boost value removes "
        "about half of those changes, and holding two learned directions for total pT and angular spread removes most; "
        "the network's mass estimate does not move on average, and the predicted route through soft-particle pT cuts is "
        "excluded. Repeated on new jets with a stricter random comparison, the result holds, but generic high-variance "
        "directions also carry much of the effect."
    ),
    'X14_EXPLAINER': (
        "In small transformer-type taggers, how much the last layer's mass code depends on the stored jet axis is not set "
        "by the starting weights: it forms in the middle of training and changes when the starting weights, the batch "
        "order or the training length is changed, with no single factor detectably dominant in 9 runs per model. Training "
        "is exactly repeatable, so the spread comes from the optimisation path, and it is not linked to accuracy. Whether "
        "this last-layer code reflects what the decision uses is uncertain, since the decision moved the other way in one case."
    ),
    'E25X_EXPLAINER': (
        "We looked for where JEDI-linear first holds the total pT and the pT-weighted angular spread of its particles, before "
        "the layer where they were found earlier. A single direction in each particle's first-layer values can be swapped to "
        "reproduce a change of the spread, but networks with entirely random weights allow the same swap equally well, so "
        "this does not show that training placed the spread there. Where training first builds it remains open."
    ),
    'E27_EXPLAINER': (
        "Moving particles toward or away from the jet axis, with total pT held, shows that JEDI-linear's mass response "
        "depends on spreads weighted by particle pT, not on spreads that count particles or weight them by pT squared. "
        "Among pT-weighted spreads the decision responds to more than the squared-distance sum that enters the jet mass: "
        "holding that sum fixed, other pT-weighted spreads still move the W-versus-Z decision. The exact combination, and "
        "whether the network reads that sum or the jet mass, is open."
    ),
    'E28_EXPLAINER': (
        "Splitting one particle into two collinear particles, which keeps the jet's mass and every pT-weighted shape, "
        "changes 2–4 % of JEDI-linear's decisions and pushes quark calls toward gluon, more than moving the same particle "
        "without splitting it. So the network is neither collinear- nor soft-safe. Holding the named count and "
        "rank-window sums removes no more of the effect than holding random sums, so the internal route is not identified."
    ),
    'E27W_EXPLAINER': (
        "We turned the hardest particles of each jet around the jet axis so that the true jet mass fell by up to 10 % "
        "while every particle kept its pT and distance from the axis. JEDI-linear's mass read-out did not move and Z jets "
        "were relabelled W at about the rate of a mass-neutral control, against 94 % for a real 10 % mass drop. So the "
        "network reads a function of each particle's pT and distance from the axis, not the jet mass; a separate fit puts "
        "that radial weight at a power of distance slightly above 2."
    ),
    'E28W_EXPLAINER': (
        "Splitting one particle into two collinear ones changes a few per cent of JEDI-linear's decisions. The soft-pT "
        "cut and the 64-slot input limit do not carry this. Moving the softer particles back to their original input "
        "positions removes up to about a third of it, and holding the two sums that count empty input slots removes "
        "about a sixth; the rest is spread over many internal sums."
    ),
    'E32_EXPLAINER': (
        "When JEDI-linear's inputs are measured about an axis computed from particles snapped to trigger-sized "
        "calorimeter towers, its W/Z decision moves about 0.9 as far as for a real mass change of the predicted size, "
        "and restoring the particles' spread about the new axis removes the effect. So the network reads the spread about "
        "the axis it is given, also for realistic axis errors. A smaller part of the response depends on which way the "
        "axis moved, and the tower model lacks threshold and pileup."
    ),
    'E33_EXPLAINER': (
        "Across ten separately trained JEDI-linear networks, a unit computing the momentum-weighted squared distance "
        "from the axis appears in nine and a soft-particle count in seven. The reference unit named \"pT of the hardest "
        "particles\" tracks the total jet pT, and a true hardest-particles window appears in only three networks. Random "
        "formulas fitted to the reference recur in about a third of the networks, so recurrence beyond chance is not yet "
        "shown."
    ),
    'E34_EXPLAINER': (
        "Changing one property of quark and gluon jets at a time shows that JEDI-linear's quark/gluon score follows the "
        "number of soft particles, the jet width, how concentrated the momentum is (pT^D), and how far particles sit from "
        "the axis, at about half to three quarters of the response of a classifier trained on those properties. The last "
        "one is not the invariant mass, and an untrained network shows it too."
    ),
    'E35_EXPLAINER': (
        "With a W or Z jet's mass and momentum exactly fixed, making the momentum split between its two prongs more "
        "lopsided moves JEDI-linear's call by about 1.5 log-odds (W toward Z, Z toward W), and same-size rotations that "
        "keep the split do almost nothing. Holding the network's mass direction removes most of a real mass change's "
        "effect but almost none of this one, so the sharing reaches the decision by another route. That route is spread "
        "over several internal objects and is not yet identified."
    ),
    'E36_EXPLAINER': (
        "A search for the smallest particle changes that flip JEDI-linear's call at fixed jet mass, momentum and axis "
        "finds changes that flip the real network on about 40 % of jets, against 2 % for random changes of the same "
        "size. They mostly move momentum among the hardest particles, except for W-to-Z flips, which move particles "
        "radially. Moving momentum into the leading particle alone flips few calls, and the extracted programs predict "
        "these flips only slightly better than always predicting no flip."
    ),
    'E37_EXPLAINER': (
        "One direction in JEDI-linear's fourth layer acts as a signed momentum-sharing setting: copied from a jet with a "
        "more lopsided prong split it reproduces about 85 % of the average W-versus-Z change, it carries almost none of a "
        "real mass change, and pushing jets along it moves W toward Z and Z toward W. It is needed for about half of the "
        "W-jet response and none of the Z-jet response, the response recurs in 9 of 10 trained networks, and the "
        "jet-shape quantity it computes is open."
    ),
    'E38_EXPLAINER': (
        "Moving momentum between a jet's hardest particles at fixed mass changes JEDI-linear's W-versus-Z score less than "
        "reshuffling the same momentum among soft particles, and the sign of the change depends on how the mass is kept "
        "fixed. The internal objects tied to the hardest particles carry about half of the gluon-to-quark minimal flips "
        "and none of the W/Z flips, so there is no separate hardness channel for W versus Z."
    ),
    'B7_EXPLAINER': (
        "Training the extracted programs on edited jets labelled by the network does not teach them the network's own "
        "smallest call-flipping edits (agreement κ about 0.2 beyond 'never changes'). An improvement on the prong-sharing "
        "edit comes back when plain rotated jets are added instead, so it reflects added data, not the network's labels."
    ),
    'E37X_EXPLAINER': (
        "The fourth-layer sharing coordinate follows how unevenly the momentum is split between a jet's two prongs, with "
        "a part that follows how wide the prongs are; changing the prong separation or stretching the jet by the same "
        "mass change moves it about 8 % as much. Its push on the W/Z decision changes sign near 85 GeV for Z jets but "
        "only above about 95 GeV for W jets, so it depends on class as well as mass, and the Z-jet response is also "
        "carried by other directions."
    ),
    'E40_EXPLAINER': (
        "With the jet's total momentum held exactly fixed, splitting a particle in two or adding a soft particle moves "
        "JEDI-linear's quark/gluon score toward gluon, and merging two particles toward quark, so the decision counts "
        "particles. Very soft added particles count only away from the jet core (beyond about 0.15 in angle); near the axis "
        "their effect fades with their momentum, as an infrared-safe quantity's would. The count enters through a "
        "wide-angle sum that ignores momentum, an empty-slot sum and the soft-multiplicity sum, with no single one "
        "passing the multiple-test correction."
    ),
    'E39X_EXPLAINER': (
        "Two retrained JEDI-linear networks joined at the object layer by a map passing 4 numbers, trained on the "
        "receiving network's own task, reproduce its decisions on 87 % of jets. The sending network's class scores, passed "
        "the same way, do equally well, so the join does not show shared content beyond the decision; the hand-named units "
        "carry slightly more than random units but explain only a minority of decisions."
    ),
    'E37Y_EXPLAINER': (
        "Pushing the fourth-layer sharing coordinate moves W and Z jets of the same mass differently, but the difference "
        "disappears once W and Z jets are matched on the network's own last-layer state, so it follows that state, not the "
        "true class. Holding three mass-window neurons removes about 0.6 of it, not clearly more than random holds; two "
        "prong-widening edits at fixed prong mass move the coordinate in opposite directions, so its width part is open."
    ),
    'E39_EXPLAINER': (
        "One fitted linear map joins any two trained JEDI-linear networks at the object layer or the next layer with "
        "86–89 % of the receiving network's decisions kept, about their own agreement rate (90 %), against 34 % for an "
        "untrained first half and 66–78 % when only class scores are passed. The layer content is shared between "
        "retrains; whether they compute it the same way, and which network the edit responses follow (except the Z-jet "
        "prong-sharing response, which follows the receiver), is not resolved with seven networks."
    ),
    'B8_EXPLAINER': (
        "Scored on the network's decision changes under a small rigid shift, the large staged program beats same-size "
        "formulas and trees on standard jet variables and energy-flow polynomials, but formulas and trees on the "
        "program's own inputs do as well with fewer bits, so the lead comes from the inputs. Its pieces swapped at the "
        "matching network units reproduce about 0.6 of the network's changed decisions against 0.25 for wrong units, "
        "partly by construction."
    ),
    'E33X_EXPLAINER': (
        "The unit that sums each particle's momentum times its squared distance from the axis reappears in 8 of 9 other "
        "trained JEDI-linear networks, which random formulas of the same complexity do in about 1–5 % of cases. The "
        "momentum weighting is what recurs, and the squared-distance weight holds only near the axis. The soft count and "
        "the hardest-particle window are not distinguished from random formulas."
    ),
    'E29_EXPLAINER': (
        "An algorithm that keeps only the internal connections JEDI-linear's output depends on picks out all the "
        "hand-named particle cuts and nearly all hand-named jet quantities, and recovers about 95 % of the accuracy above "
        "chance with about 115 of 186 active units, far above random unit sets. The hand-built program is as faithful at "
        "equal size when removed units are set to their average, and somewhat less faithful when they take values from a "
        "jet of the confusable class. Because the names cover nearly every active unit in those layers, the agreement "
        "does not single out the names over other descriptions."
    ),
    'E30_EXPLAINER': (
        "Moving one particle changes the network's second-round sums by the amount their formulas predict in their "
        "weakest jets as in their strongest, so each behaves like one jet observable over its whole range; the "
        "exceptions are one sum's weakest tenth (about 0.8 of the prediction, cause unknown) and another's strongest "
        "tenth (a smaller gain). The hidden negative side of two mass-window neurons keeps measuring the mass beyond the "
        "window and is discarded by the network. Whether those neurons' weak positive range follows one formula is open."
    ),
    'E31_EXPLAINER': (
        "The shifted and turned jets used to show that the network reads a spread about the axis rather than the jet "
        "mass differ from real jets only in quantities measured about that axis, and real jets never have their centre "
        "far enough off the axis to tell the two masses apart. So that result describes the network's response to "
        "axis-displaced inputs. The split, soft-particle and boost edits produce jets that lie among real jets, and "
        "their effects on the network hold on those jets."
    ),
    'X15_EXPLAINER': (
        "Linformers weight particles by pT rank through a learned table. Its sharp rank-to-rank jumps come from the "
        "random starting values and act no more strongly than in untrained networks, so these networks do not form "
        "hard-edged rank windows at fixed positions. They do learn, in every trained copy, a graded extra weight on the "
        "five hardest particles, which matters for their accuracy; whether that weight is learned for its own sake or "
        "follows the hard particles' share of the momentum is open."
    ),
    'E26W_EXPLAINER': (
        "When a jet's pT is lowered by 10 % at fixed mass, holding the network's total-pT sum at its old value undoes "
        "about 57 % of the changed decisions, far more than internal directions that avoid total pT but vary as much in "
        "data. Most of the change in that sum arrives through one context unit. Because the boost moves the network almost "
        "only along total pT, whether total pT is singled out or simply is where the boost acts cannot be separated there."
    ),
    'X13_EXPLAINER': (
        "All 56 trained networks tested react to a shift of the whole jet that keeps its true mass, so none of them "
        "computes the invariant mass alone. How strongly they react matches how much of their mass estimate is measured "
        "from the stored axis rather than from quantities a shift leaves unchanged. That share is set by the network type "
        "for the set-based networks, but differs between training runs for transformer-type networks, by a factor of "
        "about six in one case. Which internal parts compute the shift-independent share is still open."
    ),
    'X9X_EXPLAINER': (
        "Particle Flow Networks, Energy Flow Networks and Deep Sets trained on jets shifted at random by up to some "
        "amount, between 0.04 and 0.30 rad, ignore whole-jet shifts up to about that amount and change their decisions "
        "beyond it, towards heavier classes. The shift at which they start to respond moves one-for-one with the training "
        "range in all three architectures, so the tolerated shift is learned from the data. In Deep Sets and Energy Flow "
        "Networks part of the tolerance comes from a linear read-out of the jet centre; JEDI-linear was not in this scan."
    ),
    'E24W_EXPLAINER': (
        "On two-prong jets, rotating the softer prong intact around the axis and scattering its particles in azimuth, both "
        "with each particle's momentum, distance from the axis and the jet mass kept, change the same share of "
        "JEDI-linear's decisions, about 7 %. A classifier using the two-prong variable tau21 changes about 4 points more "
        "when the prong is broken up. JEDI-linear's response follows where the prong sits and runs through its jet-level "
        "summary; a small effect of prong integrity, or one hidden behind a mass response, is not ruled out."
    ),
    'B6W_EXPLAINER': (
        "Symbolic regression on standard jet variables finds short formulas that reproduce the networks' decisions on "
        "ordinary jets almost as well as the rank-window if-statement program of the same size, but not how those "
        "decisions change under a sideways shift of the jet. Trees and hinge fits on the same standard variables fail the "
        "same way, so the difference comes from the variables, not from symbolic regression. The two edits meant to "
        "separate the program from trees on the same sums changed under 2 % of decisions and could not decide between them."
    ),
    'X9W_EXPLAINER': (
        "Particle Flow Networks, Energy Flow Networks and Deep Sets trained on jets randomly shifted by up to 0.08 rad stop "
        "calling W jets Z when the jet is shifted by 0.06 rad, while the same networks trained on unshifted jets send about "
        "two thirds of W jets to Z; the accuracy cost is about one point at most. A larger shift of 0.12 rad still moves "
        "most W jets to Z in the augmented networks, so they ignore offsets inside the trained range rather than measuring "
        "mass about the jet's own centre. For JEDI-linear the augmented retrainings lost several accuracy points and two of "
        "three kept the response, so whether its architecture matters is open."
    ),
    'E24_EXPLAINER': (
        "Randomising each particle's direction around the jet axis, at fixed momenta, distances and mass moments, destroys "
        "the two-prong pattern of W jets and changes about 7 % of JEDI-linear's calls, against 4 % for rotating the whole "
        "jet. A classifier built on the two-prong variable tau21 changes the same share, because prong shape adds little "
        "beyond mass and radial shape on these jets, so the test cannot tell whether JEDI-linear uses prong structure. "
        "JEDI-linear's named radial sums barely move; the extra change runs through two units that read only a jet-level "
        "summary, and Deep Sets and Particle Flow Networks change as many calls."
    ),
    'B6_EXPLAINER': (
        "A short list of if-statements on sums over pT-ranked particle windows follows how JEDI-linear, Deep Sets, a "
        "Particle Flow Network and a transformer change their calls under a small jet shift about as well as boosted trees, "
        "decision trees or small neural networks of the same size tuned on held-out jets, with differences up to about 2 "
        "points either way. An earlier lead came from comparing with one under-fitted boosted-tree setting. The method "
        "therefore transfers across architectures at baseline-level fidelity per bit. Symbolic regression was not run here; "
        "a later comparison (B6w) found that symbolic-regression formulas on standard jet variables fall behind under the "
        "shift, as trees on those variables do."
    ),
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
        "the leading fraction alone. H14 had no verifier of its own; the Herwig-trained models it was waiting for were "
        "tested in H24, verified by H24v, and they do not count particles either."
    ),
    'H18_EXPLAINER': (
        "Soft particles that add energy flow move every tagger toward the high-multiplicity class; for quark–gluon "
        "networks the move is about that of a network that cannot count. When only the count changes (soft particles "
        "split into collinear halves), quark–gluon networks move by 0.1–0.3 of a count-only classifier and JetClass "
        "W-versus-QCD networks by 1.2–2.3 times it, so counting depends on the model and its data, not on the edit. Why "
        "top taggers move toward QCD when particles are added is not explained."
    ),
    'H19_EXPLAINER': (
        "Rebuilt about each edited jet's own axis, JetClass- and Top-trained ParT and ParticleNet all lower their top score "
        "when the W-candidate pair mass moves off 80 GeV, at about 3–4 times a side-effect-only classifier, on both jet "
        "samples. The first analysis showed a JetClass null only because the edited jets were fed about the unedited "
        "axis, and these networks react to an axis shift alone. An extra JetClass response to the control edit remains "
        "unexplained."
    ),
    'H16_EXPLAINER': (
        "Identical small top taggers trained on hls4ml or Top Landscape jets change 4–7 % of decisions under a "
        "mass-preserving boost to 90 % of the jet pT, against under 1 % for a rotation, so the training sample is not "
        "what makes them boost-sensitive. With absolute-pT inputs the sample decides the direction of the change. Large "
        "networks trained on hls4ml were not tested."
    ),
    'H23_EXPLAINER': (
        "Rebuilt about each edited jet's own axis, JetClass-trained ParT and ParticleNet lower their top score when the "
        "W-candidate pair mass moves off 80 GeV about as much as a classifier given that mass, more the farther it moves, "
        "on two independent jet samples. The earlier ParT null came mostly from an input build that also shifted the jet "
        "relative to its axis. Part of the full-input models' response to the control edit is not explained by that shift."
    ),
    'H24_EXPLAINER': (
        "ParT and ParticleNet trained on Herwig jets, like those trained on Pythia jets, barely move when soft particles "
        "are split to raise only the particle count (about a tenth of a count-only classifier or less), so not counting "
        "is not a Pythia artefact. The generators' different multiplicities do not explain the networks' different "
        "outputs on their gluon jets. Removing the softest particles at fixed count does move the networks toward quark, "
        "and only one training per generator was available."
    ),
    'H25_EXPLAINER': (
        "Making the subjets of top and QCD jets narrower or wider at fixed jet mass and W-pair mass moves all seven top "
        "taggers the way a τ32 classifier predicts, by similar amounts, while a matched rotation inside the subjets moves "
        "them much less. The edit also changes the subjet masses, so it shows that the taggers use the inner-prong "
        "pattern but not which quantity of it."
    ),
    'H29_EXPLAINER': (
        "Rebuilt about each edited jet's own axis, JEDI-linear lowers its top score under a W-pair-mass edit by about "
        "what a classifier without the pair mass predicts, and less than one given the pair mass, so its \"no W-mass "
        "constraint\" survives the rebuild that overturned the ParT null. The fixed-axis build gave the opposite sign "
        "because the control edit moved the jet axis, which JEDI-linear reads through its axis-relative mass. The exact "
        "ratios depend on which reference fit is used."
    ),
    'H29V_EXPLAINER': (
        "On fresh jets and against five retrained reference classifiers, JEDI-linear's W-pair response sits inside the "
        "range of classifiers without the pair mass and, at 100 GeV, below every classifier with it. The ratios H29 "
        "quoted vary by about ±50 % across retrains. At 90 GeV JEDI-linear raises its top score while every reference "
        "lowers it; that response is not explained."
    ),
    'H30_EXPLAINER': (
        "Moving one soft particle across the Lund plane (angle and transverse momentum relative to the jet axis) at fixed "
        "particle count and jet pT changes the quark–gluon networks' output in a reproducible map, and hard wide-angle "
        "placements move them toward quark while every classical reference moves toward gluon. The edit also took up to "
        "about 50 GeV from the hardest particle, so the map does not isolate the emission's position (see H30v)."
    ),
    'H30V_EXPLAINER': (
        "Splitting H30's edit into its two parts shows that, at small angle and kt about 1.3 GeV, the gluon-ward rise "
        "comes from the momentum taken out of the hardest particle, while moving the particle at fixed momentum pushes "
        "the other way; on Herwig jets the rise is absent. The networks use both the hard-core energy and the soft "
        "particle's position, and which emission quantity they follow is still open."
    ),
    'H31_EXPLAINER': (
        "In a small trial run (150 jets per class), rotating the W decay inside top jets at fixed top and W mass lowered "
        "the top score of nine taggers by about 1 log-odds on JetClass tops. A classifier given only the edit's side "
        "effects (τ32 and how energy is shared between prongs) moved as much and no result passed the multiple-test "
        "correction at this size. The full run (H32, verified by H32v) found a response to the decay angle beyond a "
        "control matched in those side effects, clearest for PFN and EFN and, for the other taggers, on a small matched "
        "set of JetClass tops (claim D15)."
    ),
    'H32_EXPLAINER': (
        "At full statistics, rotating the W decay to raise |cos θ*| at fixed top and W mass lowers every tagger's top score "
        "more than a control that copies the prong energy sharing at a fixed angle. The control reproduces only about half "
        "of the τ32 change, so an MLP that sees neither the angle nor the pair masses also drops; after a per-jet linear "
        "correction PFN and EFN keep a clear angle dependence of −1.4 to −1.8 log-odds per unit |cos θ*|."
    ),
    'H32V_EXPLAINER': (
        "On a fresh sample the raw response reproduces. Fitting each edit direction separately shows that the corrected "
        "angle slope of ParT, ParticleNet and LorentzNet came from a difference between directions, not an angle slope; "
        "but on JetClass jets where the control matches τ32 and the prong fractions to 0.01, these taggers drop by 0.3–0.7 "
        "log-odds while the angle-blind MLP stays within 0.09. So they also use the angle, on that small subset."
    ),
    'H35_EXPLAINER': (
        "Swapping the impact parameters of displaced tracks from the mass-assigned b subjet to a W subjet, at fixed jet-level "
        "count, lowers ParT_full's top score by about 2.9 log-odds more than the same swap inside the b subjet "
        "(ParticleNet 2.1), while a shuffled-weight ParT and a count-only classifier give 0. The control swap was smaller "
        "and onto harder tracks, which the verification corrects."
    ),
    'H35V_EXPLAINER': (
        "With a control matched in the number and momentum of swapped tracks, the location effect is about 1.7 (ParT) and "
        "1.1 (ParticleNet) log-odds. Moving displacement from a W-assigned subjet that holds most of it onto the "
        "mass-assigned b subjet also lowers the score, so the taggers follow the subjet that already carries the "
        "displacement, not the b position from the masses."
    ),
    'H36_EXPLAINER': (
        "Measuring particle angles about the winner-take-all or hardest-particle axis changes 3–25 % of the decisions of "
        "top and quark–gluon taggers, in the order of each model's sensitivity to a plain shift of its inputs; LorentzNet "
        "changes none. A prediction made beforehand from that shift sensitivity got the size within a factor 0.4–1.3, a "
        "miss it shares with a random-direction shift."
    ),
    'H36V_EXPLAINER': (
        "On two fresh samples the call changes reproduce. Comparing the axis shift with a random-direction shift of the same "
        "size, trained networks differ by −1.7 to +4.0 points, inside the range of untrained shuffled-weight networks "
        "(−3.0 to +3.2), so no learned use of the axis direction is shown. LorentzNet's zero holds because its inputs do "
        "not depend on the axis."
    ),
    'H39_EXPLAINER': (
        "Compared with six untrained copies of the same architecture, the trained taggers' extra response to shifting "
        "the jet image toward its hardest particle is never outside the untrained range, and its sign changes with "
        "where the decision threshold is placed. Untrained energy-weighted networks (EFN) already show such a direction "
        "effect of a few points, so it does not require training. The test is weak with six untrained networks, and the "
        "second-seed and shuffled-label trainings are still running."
    ),
    'H40_EXPLAINER': (
        "With every momentum fixed, relabelling particle types moves the quark score of the particle-ID quark–gluon "
        "networks, while networks without particle ID do not move. Swapping all particles for antiparticles moves the "
        "score against the sign of the jet charge, at about the rate of a fit on jet charge; soft photons count toward "
        "quark and soft neutral hadrons toward gluon; kaons and protons cannot be told from pions in these inputs."
    ),
    'H40V_EXPLAINER': (
        "On fresh jets the jet-charge response reproduces (about 0.24 log-odds, 85 % of a jet-charge fit) and lies far "
        "outside 12 random-weight networks. ParticleNet does respond to flipping the hardest charged particle once the "
        "flip is signed by that particle's charge. The neutral-species response reproduces in direction, but random "
        "PN networks move by similar amounts, so only its direction separates it from the null."
    ),
    'H42_EXPLAINER': (
        "Quark jets in the quark–gluon Z+jet samples carry net positive jet charge (+0.097 in Pythia, +0.067 in Herwig) "
        "and gluon jets none, so a tagger can learn 'positive charge means quark' from the sample's u-quark excess. "
        "Swapping all charges in gluon jets gives jets a classifier cannot tell from real ones, so that edit stays "
        "realistic. The Pythia-trained networks keep the same charge rule on Herwig jets; second-seed and "
        "shuffled-label trainings are still running."
    ),
    'H44_EXPLAINER': (
        "Shrinking track displacements inside each significance bin, so that the number of displaced tracks above "
        "every threshold is unchanged, moves H→bb jets toward H→cc in ParT and ParticleNet, while random-weight "
        "networks do not move. The H→bb-versus-QCD score barely changes under the same edit. The edit of the "
        "displaced-track mass gave signs that differ between networks and jet types, so the vertex-mass question is open."
    ),
    'H44V_EXPLAINER': (
        "On fresh jets the count-preserving shrink reproduces, but changing each displacement in a random direction "
        "by the same amount moves H→bb jets toward H→cc just as far, and the same shrink on non-displaced tracks moves "
        "every class. In the data at equal counts b-jet tracks are less displaced than c-jet tracks, so the networks' "
        "direction is not the lifetime difference; they use impact-parameter values, not shown to be lifetime."
    ),
    'H48_EXPLAINER': (
        "Turning the lepton of an H→ℓνqq′ jet outward so the visible mass crosses 125 GeV lowers the Higgs score of "
        "ParT and ParticleNet about twice as much as a mass-only classifier predicts. For leptonic top jets no network "
        "shows a two-way step at the 152.6 GeV lepton+b endpoint. The verifier showed that the H effect follows the "
        "moved lepton, not the mass crossing (see H48v)."
    ),
    'H48V_EXPLAINER': (
        "On fresh jets the lepton-edit effect returns at the same size, largest where the mass crosses 125 GeV. Raising "
        "the visible mass by the same amount by turning a hadron gives no extra drop, so the networks react to the "
        "lepton's angle to the quark system rather than to the Higgs-mass edge. Which lepton–quark quantity they read, "
        "and how much is lepton isolation, is open."
    ),
    'H49_EXPLAINER': (
        "Soft particles added with equal count, momentum and plain-mass change, once where grooming removes them and "
        "once where it keeps them, are treated alike by the hls4ml taggers, as a plain-mass tagger does. The JetClass "
        "and top taggers react strongly to the removed particles and further to the kept ones; the verifier showed "
        "that this further response is only partly groomed-mass-like (see H49v)."
    ),
    'H49V_EXPLAINER': (
        "On fresh jets the responses reproduce. Jet by jet, the JetClass and top taggers follow about 0.4 of a "
        "groomed-mass classifier's response and add a penalty that does not depend on the direction of the groomed-mass "
        "change, and the response also depends on where the removed particles sit. What the penalty reads is not tested."
    ),
    'H50_EXPLAINER': (
        "Restoring the reclustered prong masses after moving momentum toward one prong leaves every tagger's response "
        "unchanged, so that response is not a prong-mass effect. Changing only z at fixed prong separation and prong "
        "masses moves the JetClass scores toward W/Z for more balanced prongs, but the verifier showed that a classifier "
        "without z predicts the same (see H50v)."
    ),
    'H50V_EXPLAINER': (
        "A classifier that sees jet mass, prong separation and the two prong masses, but not z or D2, reproduces the "
        "networks' response to a pure z change within about 0.15 log-odds, because those quantities fix z. So the test "
        "cannot tell whether the networks read z, D2, or separation with masses. The prong-mass result holds only for "
        "edits toward more asymmetric prongs."
    ),
    'H45_EXPLAINER': (
        "Moving momentum between the two prongs at exactly fixed jet four-momentum changes the W/Z score of every "
        "tagger: balanced prongs look more W/Z-like, asymmetric ones less. A classifier that never sees the momentum "
        "fraction predicts a similar median response, because prong separation and prong masses change with it, so "
        "whether the taggers read the momentum fraction itself is open."
    ),
    'H45V_EXPLAINER': (
        "On fresh jets every number reproduces within 0.24 log-odds. Rotating by the same angle while keeping both "
        "prongs fixed barely moves the JetClass and JEDI-linear scores, so the response follows the prong configuration. "
        "Cell by cell the full JetClass models respond up to twice as strongly as the reference on asymmetric jets and "
        "less on balanced ones, and the edit also changes the reclustered prong masses, so the variable read is open."
    ),
    'H41_EXPLAINER': (
        "Moving soft radiation from between the two prongs to outside them, at fixed mass and τ21, lowers the W, Z and "
        "H→bb scores of ParT and ParticleNet, while a random-weight network and a mass/τ21/multiplicity classifier "
        "do not respond. This is the radiation pattern of a colour-singlet decay; the opposite response on QCD jets was "
        "not resolved, and other shape variables were not yet held fixed."
    ),
    'H41V_EXPLAINER': (
        "On fresh jets the colour-flow response holds with girth and D2 also fixed within 5 %, where a shape-variable "
        "classifier shows none, and the two halves of the edit act in opposite directions. About half of the raw "
        "response also appears for a direction-neutral mirror, and at a 2 % shape match the W-jet effect is not resolved."
    ),
    'H25V_EXPLAINER': (
        "On fresh jets the subjet-width result reproduces. When τ32 is changed with the subjet masses held fixed, by "
        "making the cores inside the subjets sharper or flatter, four of seven taggers move toward top for sharper cores, "
        "the Top Landscape networks barely move for flatter ones, and PFN moves against τ32. The taggers therefore do not "
        "follow τ32 as one number; the edits are not fully realistic and only two profile shapes were tested."
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
        "gluons carry 1.59 times as many particles as quarks. The networks needed for the edits were then trained and "
        "edited in H4 (W-pair mass inside top) and H5 (splits and removals on quark–gluon jets)."
    ),
    'H3_EXPLAINER': (
        "The earlier physics edits return with the same sign at 10 times the statistics: JEDI-linear's W call collapses "
        "under a 10 % momentum scaling that leaves every jet shape unchanged, and ParT's H → bb call follows the number of "
        "significantly displaced tracks, not how far they travel. With a control that keeps the W pair fixed, no network "
        "responded to the W-pair mass inside top in this build; for ParT that null was later withdrawn, because rebuilding "
        "each edited jet about its own axis shows a W-pair response (H23, claim D11), while for JEDI-linear it holds (H29)."
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
        "not specific to attention; whether the input format causes it is to be tested with L-GATr, which was still "
        "training on 20 October."
    ),
    'P5_EXPLAINER': (
        "No trained L-GATr is published for JetClass, so one is being trained with the authors' code and settings; the run "
        "was still training on 20 October (job anrunw-ai-c-p5-lgatr2, expected end 21 October). An early checkpoint ignores a rotation of all particles about "
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
        "ParT was retrained six times on the 2 million jets used for L-GATr to check that its axis-relative mass "
        "response is not a property of one released checkpoint. After 3 of 50 epochs the full-input seed already "
        "responded like the released network; the kinematics-only seeds responded less. On the final checkpoints (P11, "
        "reproduced by P11v) all six seeds respond at the call level (14–20 more true-W jets in 100 called Z), but the "
        "released network's concentration of the response in the first class-attention layer is not reproduced."
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
        "ask whether the axis response is learned from data. P14 trained the networks and analysed only the "
        "released-network control: it responds to a shift of all particles in azimuth with the stored axis kept, which "
        "leaves the invariant mass unchanged, at about three quarters of its response to a matched real mass change, "
        "mostly through the per-particle angle inputs. The arm comparison ran in later rounds (P16, P22): ParTs trained "
        "on the planted or true labels respond at only 0.2 to 0.5 of a matched real mass change, so the architecture "
        "does not force the released response."
    ),
    'P15_EXPLAINER': (
        "Inside ParticleNet, the response to a mass-preserving azimuthal shift is the same for either sign, and a "
        "direction after the second graph layer reproduces about three quarters of it; the first-layer direction found "
        "for the positive shift only records which way the particles moved. Directions like these also exist in untrained "
        "networks and are not the axis-moment mass. The rebuilt neighbour graphs carry at most about 4 %, and a few dozen "
        "pooled channels reproduce the W/Z calls but not the 10-class calls."
    ),
    'P11_EXPLAINER': (
        "Six Particle Transformers retrained from scratch all turn 14–20 % more true W jets into Z when every particle is "
        "shifted by 0.06 rad in azimuth with the stored jet axis kept (a rotation about the beam line, which keeps the jet "
        "mass but widens the spread about the axis), while untrained networks barely respond, so this behaviour comes with "
        "training on JetClass. The released network's internal route, almost all through the first class-attention "
        "layer with one dominant head, is not reproduced; the retrained values lie in the range untrained networks reach. "
        "Seed and the 50 times smaller training set are not yet separated."
    ),
    'P16_EXPLAINER': (
        "Particle Transformers trained only to flag a jet-mass window respond to a shift of all particles in azimuth with "
        "the stored axis kept, which leaves the invariant mass unchanged, at about 0.2 of their response to a real mass "
        "change, against about 0.75 for the released models, and compute the mass mostly from pairs of particles. So the "
        "architecture does not force the released models' axis-relative mass. Later runs tested the source: networks "
        "trained on the true labels (P22) also stay well below the released level, and training on all 10 classes raises "
        "the response only slightly (P31); what produces the released level, and whether the label decides which mass "
        "is read, is still open."
    ),
    'P18_EXPLAINER': (
        "In ParT, particle count and pT can be read from the input and the jet mass from the first block, nearly as well "
        "as in an untrained network of the same shape, so there is no mass-last stage and these read-outs do not show "
        "learning. Prong-shape variables are what training adds: only the trained network reads them, and the share of "
        "the response to randomising particle directions about the axis that one direction carries grows from about 0 to "
        "about 0.7 over the blocks, while untrained networks stay flat. Which prong observable is used is not resolved."
    ),
    'P21_EXPLAINER': (
        "Changing one prong variable at a time at fixed jet mass shows that ParT does not track tau21, tau32 or D2 the "
        "way classifiers built on them do: narrower prongs make W, Z and top jets less W-like, the reverse of a tau21 "
        "classifier. Moving soft particles toward the axis makes ParT about 0.3 more W-like, but that edit also raises "
        "the mass of the hard particles, which can explain it. Head patching gives the same picture for edits that leave "
        "the variables unchanged, so it cannot show where the prong quantity is read; what ParT reads is still open."
    ),
    'P22_EXPLAINER': (
        "Particle Transformers trained on 1.5 million W, Z and QCD jets with the true labels respond to a "
        "mass-preserving shift of the jet axis only 0.2 to 0.5 times as strongly as to a matched real mass change, "
        "depending on the shift direction, against about 0.75 to 0.85 for the released JetClass models, and they do not "
        "approach that level during training. Training with random axis shifts removes the response at the same "
        "accuracy, so it is learned from the inputs rather than built into the architecture. Which of 100 million jets, "
        "10 classes or a longer schedule produces the released level is open: a later run (P31) found that 10 classes "
        "raise the response only slightly, and the larger-data and longer-schedule runs were still training."
    ),
    'P26_EXPLAINER': (
        "Sophon, a Particle Transformer released after training on the much larger JetClass-II dataset, hardly changes "
        "its W/Z output when the jet mass is really increased. Its internal size reading moves about 0.2 as much under "
        "a mass-preserving shift of the jet axis as under a real mass change, against 0.8 to 1.0 for the JetClass ParT, "
        "so it mostly ignores where the axis sits, although half of it still uses the axis-relative angles. Whether the "
        "larger data set, the mass-decorrelated training or the momentum-normalised inputs causes the difference is open."
    ),
    'P27_EXPLAINER': (
        "Moving one particle slightly away from the jet axis shifts ParT's and ParticleNet's W-versus-Z score; on "
        "average the shift grows with the particle's momentum share and its distance from the axis, at 0.6 to 0.8 of "
        "the steepness that the angular second moment M2 would give. Particle by particle no simple formula predicts the "
        "shift, and the momentum-rank windows that JEDI-linear uses do no better than M2. In ParT most of the scatter "
        "comes from the features between pairs of particles; a later run (P33) traced that term to particles with close "
        "neighbours, read through two of the pair-feature attention heads."
    ),
    'P31_EXPLAINER': (
        "A Particle Transformer trained on all 10 JetClass classes, with the same 1.5 million jets and step count as the "
        "3-class runs, responds to a rigid shift of the jet about its axis about 0.3 times as strongly as to a real mass "
        "change of the same size; the 3-class runs give about 0.22 and the released models 0.72 to 0.87. The 10-class "
        "objective therefore raises the axis response measurably but accounts for only a small part of the gap. There is "
        "one seed per run, and the larger-data and longer-training runs are still training."
    ),
    'P32_EXPLAINER': (
        "In the ParT and ParticleNet models that receive particle-identity inputs, removing the electron or muon flag from "
        "one particle moves about 99 % of Higgs-to-lepton-neutrino-two-quark jets to other classes, while the same change "
        "on a similar hadron moves far fewer. In ParT one head of the final class stage reads that particle, and a second "
        "route through the earlier blocks backs it up. A three-head subset reproduces the call for this class but also "
        "gives it to almost all top-quark jets with a lepton, so it is not a specific classifier; a fresh-sample rerun is "
        "pending."
    ),
    'P33_EXPLAINER': (
        "When one particle in a W or Z jet is moved by 0.01 rad, ParT's direction-independent response through its pair "
        "features comes mostly from particles with a neighbour closer than about 0.01 rad and passes through two of the "
        "eight pair-bias heads; both results reproduce on fresh jets. A linearised copy of the network produces only about "
        "a third of that response, so the network's own nonlinearity makes most of it, not the logarithm of the pair "
        "distance alone. Which stage of the network produces it is not yet known."
    ),
    'P34_EXPLAINER': (
        "Removing the electron or muon label from one particle changes about 98 % of ParT's decisions that a jet comes "
        "from a Higgs boson decaying to a lepton, a neutrino and two quarks, against under 1 % for a matched label change "
        "on a hadron; adding a muon label to a hadron in a jet without a lepton creates that decision in 18 % of jets. The "
        "attention head that reads the lepton, kept alone with everything else averaged out, still scores lepton presence "
        "but makes no decision and does not tell these jets from top-quark jets with a lepton. Which units make the "
        "decision in the full network is still open."
    ),
    'P36_EXPLAINER': (
        "Moving the particles of naturally off-centre W and Z jets so that their momentum-weighted centre sits on the "
        "stored axis, at fixed jet mass, changes ParT's W-versus-Z score about as much as a real mass change of the same "
        "size in the spread about the axis (ratio about 1.0). So ParT measures mass about the stored axis also for offsets "
        "of natural size, which here amount to about a 1 % mass shift and change a few percent of decisions on the most "
        "off-centre jets. The edit moves the jets' momentum direction off the stored axis, so unedited jets are still to "
        "be tested."
    ),
    'P37_EXPLAINER': (
        "Five attention heads in ParT's final class-token stage, where one summary token reads all particles, carry 59 % "
        "of the top-quark score drop, against about 21 % for five random heads from that stage, and about two thirds of "
        "the response when the b-quark prong is merged into one particle. Removing them mainly affects the top-quark class, "
        "but only marginally significantly. Kept alone, they do not give the top-quark call, so they are not a stand-alone circuit."
    ),
    'P39_EXPLAINER': (
        "Comparing unedited off-centre jets with centred jets of equal mass cannot test ParT's axis dependence, because the "
        "off-centre jets sit at larger pseudorapidity and every network, including one that cannot see the stored axis, "
        "scores them differently for that reason. Edits show that ParT responds to the particles' spread about the stored "
        "axis, not to the jet's momentum direction moving off it. The full models' sensitivity to a mirror image comes from "
        "the signs of charge and impact parameter; how much the natural offset affects real decisions is still inferred."
    ),
    'P40_EXPLAINER': (
        "As a set, the neurons in ParT's final class-token layers are about as readable as random rotated directions in "
        "the same space, and nudging one neuron moves its labelled class the predicted way only slightly more often than a "
        "random direction does, usually by a negligible amount. A small minority have reproducible causal roles; a few that "
        "follow the axis-based jet mass reproduce about a quarter of the W response to a real mass change. Which neurons "
        "count as mass neurons changes between samples."
    ),
    'P43_EXPLAINER': (
        "When two hard particles are moved apart or together so that the jet's spread changes while its momentum-weighted "
        "centre stays fixed, ParT responds almost exactly as to a real mass change of the same size. When the centre is "
        "moved off the stored axis on unedited jets, it responds at about three quarters of that, independent of the shift "
        "size. An earlier estimate of one third came from measuring on jets that had already been edited."
    ),
    'P44_EXPLAINER': (
        "In our Lorentz-symmetric tagger the class decision runs through its Lorentz-invariant channels, two groups of which "
        "back each other up. A boost along the beam changes about 3 % of calls, only through the time reference and the "
        "particle energies; boosting those together with the particles leaves the calls unchanged, as the design requires. "
        "Its response to the jet axis at fixed invariant mass waits for the final checkpoint."
    ),
    'P45_EXPLAINER': (
        "ParticleNet's particle features after its first two graph layers can stand in for ParT's own at ParT's first "
        "blocks through one linear map, at about 1 point of accuracy; after its third layer they cannot. Running the "
        "stitch in both directions shows that the size of the response to a mass-preserving axis shift depends on both the "
        "features and the layers that read them, not on ParticleNet's last layers alone. Which part of ParticleNet's first "
        "two layers makes its response smaller is open."
    ),
    'P49_EXPLAINER': (
        "In our L-GATr a fixed formula of particle-pair kinematics accounts for about a fifth of an attention head's "
        "pattern, and the heads the decision needs are all in the first five blocks. Replacing the early attention with "
        "the formula keeps about half the accuracy only while the network's own attention to its reference tokens stays "
        "live; 17 of the recovered points come from the pair geometry. Unlike ParT's pair bias, L-GATr's early attention "
        "does not work as a fixed geometric prior; particle identity was not included in the formula."
    ),
    'P50_EXPLAINER': (
        "ParticleNet's first graph layer uses, beyond rotation-invariant pair quantities, how far each neighbour sits "
        "from the jet axis, and its messages are not sums of one-variable functions. Restricting it to the invariant "
        "quantities lowers the kinematics-only model's axis response from about 0.37 to 0.26 of a real mass change, "
        "away from ParT's 0.8; about half of that drop is generic fit error. The full-input model could not be compared "
        "at matched accuracy."
    ),
    'P46_EXPLAINER': (
        "The pre-registered test of whether L-GATr ignores the jet axis at fixed invariant mass did not run, because of a "
        "failed GPU node and a blocked relaunch. From the network's code, a rotation about the beam and a move of the "
        "stored axis cannot change its input, so only a boost along the beam and rearrangements at fixed mass can test "
        "the claim when it is relaunched."
    ),
    'P51_EXPLAINER': (
        "Boosting every particle along the beam keeps the jet mass but moves particles relative to ParT's stored axis; ParT reads this as a mass change (about 0.8 of a real one) and our L-GATr does not respond at all, on two samples. The rearrangement edits that hold the mass fixed also hold the particles' spread fixed, so they cannot tell whether L-GATr reads the invariant mass or the spread about its own momentum direction; a wide-angle test is needed."
    ),
    'P54_EXPLAINER': (
        "L-GATr's gated hidden channels track a class or physics quantity more often than mixed directions of the same layer (19 % against 13 % on two samples), and only after the gate. Steering or removing single channels gives no consistent advantage over mixed directions, so the channels are a preferred read-out basis but not separately controllable units. The small-circuit test could not reach the full tagger even with every channel kept, so it does not decide whether small channel circuits exist."
    ),
    'P56_EXPLAINER': (
        "Raising L-GATr's jet mass by making every particle heavier at fixed directions moved its W-versus-Z score only a third as much as the same mass rise made by widening the angles, all through its four-vector inputs; widening the spread at fixed mass moved it about a fifth as much. The verification showed that the heavier-particle edit also pushes particles off their physical mass, which by itself moves the score the other way, so this ratio does not measure how much L-GATr reads the invariant mass."
    ),
    'P56V_EXPLAINER': (
        "On new jets the heavier-particle response was smaller (a quarter instead of a third of a real mass change). Reversing every energy change of the fixed-mass energy shuffle gave the same response, so L-GATr reacts to particles being off their physical mass in either direction, by about half of a real mass change and opposite in sign to the mass response. The share of the angular spread that reaches it through the angle inputs, about 0.13, reproduced; the four-vector share did not."
    ),
    'P59_EXPLAINER': (
        "A search for a small part of L-GATr's internal state that stores the jet mass found one direction among its 32 plain scalar channels that carries about 40 % of the response to a real mass change from the fourth block on, well above random directions and mismatched-pair controls. The scalar parts of its multivectors carry little, and no subset of four or fewer directions reproduces the calls. Its response to particle-mass edits was later shown to come from unphysical particle masses."
    ),
    'P59V_EXPLAINER': (
        "On new files the same direction again carries about 40 % of L-GATr's response to a real mass change, and holding it fixed removes 37–49 % of that response, against at most 8 % for random directions. With edits that keep particles physical, both the direction and the network output follow the invariant mass (0.84 of an angle-made change when all momenta are scaled), not the angular spread. About 60 % of the mass response is held elsewhere."
    ),
    'P60_EXPLAINER': (
        "L-GATr retrained with its particle angles measured from the stored jet axis, as ParT does, responds to particles moved off that axis at fixed invariant mass with 0.38 of a real mass change after 5,000 training iterations, against zero for the unmodified network, and the response passes entirely through those angle inputs. So the axis dependence comes from the inputs, not from the architecture; it is about half of ParT's so far, and the training is still running."
    ),
    'P55_EXPLAINER': (
        'No set of 16 or fewer attention heads in L-GATr reproduces 80 % of its W/Z or top/QCD decision under any of three replacement rules. Replacing switched-off heads with outputs from jets that point elsewhere made most heads look necessary; with direction-matched replacements only six early heads are individually needed and 32 heads recover about 0.7 of the top/QCD decision. Whether the chosen heads beat random ones was not testable with 100 random sets.'
    ),
    'P23_EXPLAINER': (
        "Shifting all of a jet's particles by a small angle changes ParT's W-versus-Z score mostly in the same way for a "
        "shift and its mirror image, about 93 % per jet, with a size that follows the jet's own response to a real "
        "widening of its angles. This is what reading the angular second moment about the stored axis predicts, and it "
        "differs from JEDI-linear's direction-dependent response; ParticleNet has a larger direction-dependent part. "
        "Why the response still varies with the shift's direction at fixed size is not explained."
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
        "but with large scatter from jet to jet. The repeat on a second sample was not reviewed by the verifier, and a later "
        "test with small input edits of the two moments (E21w) found no class neuron and not the W-versus-Z score outside a "
        "band of random controls, so the product-like reading is not reproduced; that test can exclude only large effects. "
        "No layer that forms the product has been located."
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
        "mechanism. This run also found two different gains, below the prediction for the part quadratic in the shift and "
        "above it for the part linear in the centre offset; a later measurement without the estimator bias (E23w and its "
        "verification) did not reproduce that split: the two parts are weighted nearly equally from the dense layers on. At "
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
        "small samples (400 jets per class for ParT) without intervals. H3 reran them at 10 times the statistics with a "
        "matched control for each edit and the signs returned (claim D3); lepton isolation, π⁰ pairing and the dead cone "
        "were not rerun by the verifier."
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
        "No model formed rank windows; short HGQ training did not form slot masks. Later full-length HGQ retrainings of "
        "JEDI-linear (X12) did form slot masks, but with their edges at other pT ranks (24–45) and without the official "
        "network's response at the 14-hardest-particle edge."
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
    'H54_EXPLAINER': (
        "Regenerating quark/gluon jets with only Pythia's hadronisation changed moves the particle count by about 11 % "
        "while infrared- and collinear-safe shapes stay nearly fixed. The IRC-safe network does not move, a count "
        "classifier moves strongly, and the Pythia-trained taggers move part of the way toward the count classifier. "
        "Count and soft-hadron momentum change together, so which of them is read is open."
    ),
    'H54V_EXPLAINER': (
        "A rerun with new seeds reproduces every shift, and changing only the string transverse-momentum width, where the "
        "IRC-safe network moves the other way, gives the same beyond-energy-flow response. Over 30 hadronisation settings "
        "each tagger's shift is a weighted mixture of the IRC-safe network's and the count classifier's, with a smaller "
        "count part for networks trained on Herwig jets."
    ),
    'H55_EXPLAINER': (
        "Infrared- and collinear-safe classical taggers ignore exact collinear splits, as expected, while trained networks "
        "change a few percent of their calls. Under the full edit library, JetClass and Top networks respond most like "
        "groomed-mass-plus-substructure taggers; the verifier showed that hls4ml networks respond most like a plain mass "
        "window (see H55v)."
    ),
    'H55V_EXPLAINER': (
        "The nearest-tagger result depends on the dataset: groomed-mass-plus-substructure for JetClass and Top, plain mass "
        "window for hls4ml Z and top. The reported significances rested on an approximation to too few random-weight "
        "copies, and a groomed-mass-only tagger is needed to tell groomed mass from substructure use."
    ),
    'H56_EXPLAINER': (
        "A second training seed leaves the axis-direction effect inside the range of untrained networks and moves it by up "
        "to 1–2 percentage points. The small response to particle number at fixed energy flow repeats across seeds. "
        "Networks trained on unrelated labels learn a constant output and give no spread, and the jet-charge test waits "
        "for its retrained networks."
    ),
    'H56V_EXPLAINER': (
        "On new jets and against 12–20 untrained networks per architecture, no trained tagger shows an axis-direction "
        "effect outside the untrained range. The count response at fixed energy flow reproduces, and the change between "
        "Herwig- and Pythia-trained networks is several times the seed-to-seed difference. Error bars at a fixed decision "
        "threshold understate how much such numbers move between jet samples."
    ),
    'H57_EXPLAINER': (
        "Pythia settings that change the particle count and the momentum sharing among hadrons (pT^D) separately, on the "
        "same parton-level events, show that the Pythia-trained ParT and ParticleNet quark/gluon taggers do not follow "
        "the total count: at fixed pT^D a 7 % drop in count moves them by about a quarter of what a count-only classifier "
        "does beyond an IRC-safe tagger. Where the two quantities disagree, the taggers follow pT^D. The setting that "
        "decides this is far from nominal jets, and particle-ID networks are confounded by the changed hadron species."
    ),
    'H57V_EXPLAINER': (
        "On new events the result repeats: the taggers do not follow the total number of hadrons. A count of only the "
        "harder hadrons (above 1–4 GeV) predicts their response as well as pT^D, because the two move together in every "
        "setting, so the experiment does not decide between momentum sharing and counting hard hadrons. The decisive "
        "setting is a cancellation of two large opposite effects."
    ),
    'H60_EXPLAINER': (
        "When only Pythia's hadronisation changes so that top jets carry about 9 % more particles at the same mass and "
        "three-prong structure, all six top taggers tested call them less top-like, while an IRC-safe network does not "
        "move. This is the direction of a classifier that reads the count at fixed substructure, opposite to a count-only "
        "classifier. These are particle-level jets with many more particles than the detector-level training jets."
    ),
    'H60V_EXPLAINER': (
        "Adding only extra collinear particles to top jets reproduces most of the shift, and merging the extra particles "
        "away removes it, so on top jets the cause is the particle count at fixed substructure. Under a crude "
        "calorimeter-like merging that brings the count into the training range, ParT's shift falls to about a quarter "
        "and one JetClass network shows none, so the size for real detector-level tagging is open."
    ),
    'H61_EXPLAINER': (
        "On detector-level jets made with the Top Landscape detector simulation and no multi-parton interactions, a "
        "hadronisation change that adds about 7 % more reconstructed particles to top jets at the same mass and "
        "three-prong shape lowers the Top Landscape ParT top score by 0.31 log-odds, while an IRC-safe network does not "
        "move. The generated jets remain distinguishable from the Top Landscape jets (classifier AUC 0.74)."
    ),
    'H61V_EXPLAINER': (
        "On new events the shift repeats (ParT −0.29 log-odds). Moving the same particles by the same small angle "
        "without adding any changes no score, and splitting randomly chosen particles moves the scores as much as "
        "splitting the softest, so the taggers respond to the number of objects at fixed safe substructure, not to the "
        "geometry of the edit or the softness of the added objects."
    ),
    'H64_EXPLAINER': (
        "Under a boost along the jet axis that keeps the mass fixed, JetClass and Top Landscape ParT and ParticleNet "
        "respond far less than to an equal narrowing at fixed momentum, while the hls4ml and quark–gluon taggers respond "
        "more. A classifier given only mass and momentum also responds strongly on hls4ml, so a large response does not "
        "by itself show that a network reads angles."
    ),
    'H64V_EXPLAINER': (
        "Measured with class-summed magnitudes, JetClass and Top ParT and ParticleNet respond at 0.12–0.36 of the "
        "narrowing response, also inside the training momentum range. A classifier given only the jet's angular width "
        "and momentum, with no mass, gives the same ratio, so the test excludes a pure angular-size reader but cannot "
        "tell an invariant-mass reader from a width-plus-momentum reader."
    ),
    'H67_EXPLAINER': (
        "Edits that hold a jet's mass and momentum exactly fixed but change its width (by sharing momentum less evenly "
        "between prongs), or change its mass at fixed width, move every JetClass and Top Landscape tagger, so none reads "
        "mass alone or width alone. The hls4ml taggers respond much less to the width edit on W and Z jets."
    ),
    'H67V_EXPLAINER': (
        "The numbers repeat on new jets, but both edits change how momentum is shared between the prongs by the same "
        "amount, and classifiers on mass plus prong sharing, D2, τ21 or width give similar responses. So the test "
        "excludes pure mass and pure width readers without naming the second variable; against a rotation of matched "
        "size, every tagger responds to the width edit, the hls4ml ones included."
    ),
    'H68_EXPLAINER': (
        "Relabelling particles between charged hadron, photon and neutral hadron while keeping their momenta fixed "
        "lowers the signal scores of the JetClass taggers that see particle types, roughly in proportion to the "
        "relabelled energy; taggers without type inputs do not move."
    ),
    'H69_EXPLAINER': (
        "Edits that move one shape variable at a time at fixed jet mass and momentum change the JetClass and Top "
        "Landscape tagger outputs more than a prong rotation of the same particle displacement, for prong sharing, D2 "
        "and τ21 alike; the jet width could almost never be changed alone on signal jets."
    ),
    'H69V_EXPLAINER': (
        "Against a random reshuffling of particle momenta of the same size, and with ratios taken over the same jets, "
        "about half of the response disappears: the full-input JetClass taggers still respond to D2, less clearly to "
        "prong sharing and τ21, while kinematics-only ParT and the Top taggers do not exceed the floor or their "
        "scrambled copies. Width can be changed on some signal jets, but only together with D2."
    ),
    'H71_EXPLAINER': (
        "Scaling all particle energies by a common factor, which leaves every dimensionless shape unchanged, moves the "
        "output of every tagger that sees absolute momenta, by 0.2–1.1 times the response to the same mass change at "
        "fixed momentum; shape-only classifiers do not move."
    ),
    'H71V_EXPLAINER': (
        "The response is the same on jets that stay inside the training momentum range, so it is not caused by leaving "
        "that range. A momentum change at fixed mass also moves the outputs, by about a quarter of the mass effect, and "
        "the effects do not add, so the taggers use mass and momentum together; the tracker-versus-calorimeter "
        "difference appears in kinematics-only models too."
    ),
    'H72_EXPLAINER': (
        "Removing the outer particles of top jets moves top taggers from about the same radius as a mass cut, and "
        "restoring the mass and momentum afterwards leaves most of their loss, unlike a mass or groomed-mass cut."
    ),
    'H72V_EXPLAINER': (
        "Putting the removed number of particles back, by splitting particles into exactly collinear halves, does not "
        "recover the loss, so it is not particle counting; it comes with weaker three-prong structure. The small net "
        "responses of the Top PFN and quark–gluon taggers are a count effect and a shape effect cancelling."
    ),
    'H66_EXPLAINER': (
        "After reweighting our simulated top jets until a classifier can barely tell them from Top Landscape jets "
        "(AUC 0.61), a hadronisation change that adds about 4 objects at fixed mass and three-prong shape still lowers "
        "ParT's top score (−0.25 log-odds), while the energy-only EFN does not move. In the Top Landscape data, jets with "
        "more objects at fixed mass and shape are more often QCD. Uncertain: the size depends on the weighting convention "
        "(see H66v), and QCD-jet responses differ between networks."
    ),
    'H66V_EXPLAINER': (
        "Weighting each event pair by one of its two jets biases the comparison in opposite directions, so the size of "
        "the reweighted hadronisation shift is not fixed (ParT −0.47 to −0.05 depending on the convention), while the "
        "response per added object is stable. On real Top Landscape top jets, adding objects in three different ways "
        "lowers the top score and removing them raises it for all four taggers, and the data slope (about −0.025 per "
        "object at fixed shape) reproduces on fresh rows."
    ),
    'H59_EXPLAINER': (
        "A second training seed reproduces the quark–gluon networks' jet-charge response within 7 %. Flipping the "
        "charges of hard and soft particles shows that the score follows the jet charge with each particle weighted by "
        "its momentum fraction to a power of about 0.33, not a plain charge count or the leading particle alone. "
        "Uncertain: one such charge explains 80 % of ParticleNet's response but only 40 % of ParT's."
    ),
    'H59V_EXPLAINER': (
        "Flipping one particle's charge at a time gives a weighting power of 0.32–0.41 on quark jets and up to 0.48 on "
        "gluon jets. ParticleNet adds several flips almost linearly; ParT's response to two flips is about half the sum "
        "of the single ones and depends on the flip direction, so for ParT the per-particle weighting is measured but a "
        "linear jet-charge read-out is not."
    ),
    'H76_EXPLAINER': (
        "Moving one classical quark/gluon angularity at a time with the others fixed, the trained networks respond "
        "almost only to the angular-spread edits (LHA and width), with a pattern matching the energy-flow network "
        "(cosine about 0.99), and barely to the particle count or pT^D. Uncertain: these edits make jets a classifier "
        "separates from real ones (AUC about 0.9)."
    ),
    'H76V_EXPLAINER': (
        "The angular-spread response is lopsided: pulling energy toward a common radius moves ParT toward quark four to "
        "six times more than the opposite edit moves it toward gluon, and a random edit of the same size does not. A tree "
        "on angularities and the leading particle's radius give the same pattern as the energy-flow network, so the "
        "radial variable the networks read is not identified; the count edits give the same response whether particles "
        "are added or removed."
    ),
    'H68V_EXPLAINER': (
        "Swapping labels between charged and neutral particles of equal total energy, which leaves the jet's "
        "charged-energy fraction almost unchanged, lowers the scores even more, so the response follows how much "
        "labelling is rearranged, not the jet's composition. Whether real jets would ever show such patterns is open."
    ),
    'H65_EXPLAINER': (
        "Under a uniform haze of soft particles, every network loses signal faster than groomed- or area-subtracted-mass "
        "taggers. The hls4ml tagger follows the plain-mass increase, the larger JetClass and Top taggers respond to soft "
        "particles even when they add no mass, and quark–gluon taggers respond mostly to wide-angle ones."
    ),
    'H65V_EXPLAINER': (
        "On fresh seeds the numbers repeat. Top Landscape ParT responds the same whether the added particles are stacked "
        "on the axis or spread over 0.1 rad, so it counts them. JetClass full-input ParT's response to particles that add "
        "no mass falls to about a quarter when they are labelled photons instead of neutral hadrons, so the label "
        "carries most of it."
    ),
    'H58_EXPLAINER': (
        "With every tagger set to keep half of the signal jets, top taggers' call changes under edits look more like a "
        "groomed-mass window than an ungroomed one, and an extra τ32 cut seems to add nothing. At the published cuts "
        "every network instead looks closest to the substructure taggers, so the earlier H55 picture depended on the "
        "working point. The verifier showed that this reading does not hold (H58v)."
    ),
    'H58V_EXPLAINER': (
        "The groomed-mass preference of top taggers comes from a single edit that changes how momentum is shared between "
        "the prongs at fixed mass; without it there is none. Top taggers still respond to soft radiation that grooming "
        "removes, at about half the rate of radiation it keeps, while the hls4ml W/Z networks respond to both alike, as "
        "an ungroomed mass does. Compared at the same mass window and efficiency, a τ32 cut does add similarity; whether "
        "the networks compute τ32 itself is open."
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
        "with and without fixed-point training, read the mass the same way. Three standard set networks trained on "
        "randomly shifted jets stop responding to shifts inside the trained range, whatever its size, and respond beyond "
        "it, so the behaviour comes from training jets whose axis is always at the centre. Across 56 trained networks of "
        "several types the shift response is present in all; its size follows how much of each network's mass estimate is "
        "tied to the stored axis, which depends on the network type for set networks and on the training run for "
        "transformer-type networks (the whole training path, not the starting weights alone). At fixed mass, JEDI-linear "
        "also reads total pT as a class feature, so boosting a jet along its axis changes about a third of its decisions "
        "while its mass estimate stays the same on average. The spread it reads is weighted by particle pT, as in the "
        "mass formula, but its decision is not a function of the squared-distance sum alone; and splitting one particle "
        "into two collinear ones, which keeps the mass, still changes a few per cent of its decisions. Lowering the true "
        "jet mass by turning particles around the axis, with each particle's pT and distance from the axis fixed, leaves "
        "JEDI-linear's mass estimate unchanged, and its radial weight is a power of distance slightly above 2. In real "
        "jets the centre is never far enough from the axis to change the mass by more than 0.7 %, so the difference "
        "between the two masses shows only for axis-displaced inputs, such as jets measured about another axis; the "
        "edited jets differ from real jets only in quantities measured about the axis (round 14, E31)."
    ),
    'Why this matters: the jet axis is a convention': (
        "Because the networks' mass depends on the axis, measuring particle positions about another accepted axis "
        "(winner-take-all or the hardest particle) changes their decisions although the invariant mass does not change: "
        "with the winner-take-all axis about 30 % for JEDI-linear and 12 % for ParT with all input features, mostly W jets "
        "called Z. The direction of the changes follows each model's own formula, but some changes also occur under a "
        "shift of equal size that keeps the formula fixed, so part of the effect is general sensitivity to inputs unlike "
        "the training data. For JEDI-linear on a trigger-style axis built from calorimeter towers (round 15, E32), the "
        "W/Z response is about 0.9 of the predicted mass change plus a smaller part that depends on the offset direction."
    ),
    'The whole trigger network as if-statements': (
        "JEDI-linear's active units, written as nested hinge if-statements ('if Q > t, add a(Q − t)') fitted layer by "
        "layer from physical quantities, form a program that makes the same decision as the network on 94 % of held-out "
        "jets, with no network value inside it, and that follows the network on the jet shift. On fresh jets its accuracy "
        "is not distinguishable from the network's. Its variables match the network's internal variables only partly when tested by swapping single values "
        "between jets, and standard surrogates such as boosted trees reproduce the decisions with fewer fitted pieces. "
        "Mass-type terms first appear in the dense layers of the fit, which is consistent with, but does not locate, where "
        "the network combines total pT and spread. A blind version of the procedure, applied to three other architectures, "
        "follows their decision changes as well as tuned trees of the same size, not better, and better than symbolic "
        "regression on standard jet variables, which copies decisions on ordinary jets but not under the jet shift. "
        "Off ordinary jets (round 16) the floating-point copy with masks follows about 85 % of the network's changed "
        "decisions under a prong-sharing edit at fixed mass, the hinge program over-reacts, and on search-found minimal "
        "edits the programs predict little about which jets flip; training them on such edited jets labelled by the "
        "network does not change that (round 17). Scored on edit responses at equal size (round 18), the program's lead "
        "over standard-variable and energy-flow-polynomial surrogates is matched by surrogates on its own inputs, so "
        "it comes from the inputs rather than the staged structure."
    ),
}
