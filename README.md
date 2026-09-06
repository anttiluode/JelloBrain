# JelloBrain

### Can meaningless signals write a medium, grow private conventions, and learn when an internal event is worth sending outside?

**Live Phase I:** https://anttiluode.github.io/JelloBrain/  
**Live Phase II dialogue lab:** https://anttiluode.github.io/JelloBrain/dialogue.html  
**Live spatial launcher lab:** https://anttiluode.github.io/JelloBrain/spatial.html

JelloBrain follows [GelatinIsland](https://github.com/anttiluode/GelatinIsland). The recurring idea is simple:

> **Signals make roads. Roads change later signals. A boundary decides what becomes an outward event.**

This is a sequence of deliberately small toy experiments, not a brain model.

```text
W0-W5   one spatial gel
        ordered traffic -> persistent operator geometry

D0-D4   two private protocol gels
        interaction -> pair-specific pulse convention

S0-S4   one full spatial body + launch boundary
        inherited geometry -> prediction -> residual -> route repair

S5-S8   launch-boundary salience isolator
        novelty != relevance; expected != unimportant
```

The current hard problem is to reunite those pieces without hiding a lookup table beside the launcher.

---

## Phase I — words write the gel

Six meaningless primitives `A B C D E F` are only spatiotemporal pulses. A pseudoword is a sequence such as `ABC` or `ACB`. There are no embeddings, pretrained semantics, word IDs, or symbolic meanings inside the substrate.

Fast excitation can be erased while slow directed material remains.

Run:

```bash
python gates.py
```

Frozen receipt: [`results/gates.json`](results/gates.json)

| Gate | Result | Interpretation |
|---|---:|---|
| **W0 persistent trace** | **1.00** vs .50 chance | history survives a complete fast-state wipe |
| **W1 order not frequency** | **1.00** | identical primitive counts, different order, still decodable |
| **W2 geometry matters** | **1.00 -> .50** after shuffle | preserving every slow value but scrambling its placement/direction destroys the code |
| **W3 context** | **1.00** | identical target `E` differs after `AB` versus `CD` |
| **W4 computation** | gel **1.00**, raw linear XOR **.75** | the dynamics create a useful nonlinear temporal expansion |
| **W5 cross-world** | **1.00** vs .25 chance | order-only pseudowords transfer across individualized gels |

The important result is W2: information is not only in how much material exists. It is in **material arrangement as an operator**.

---

## Phase II — two private systems negotiate a pulse convention

A sees private bit `x`. B sees private bit `y`. One of two initially meaningless pulse identities crosses A -> B. B must output `x XOR y`. Neither side can inspect the other's slow material.

Run:

```bash
python dialogue_gates.py
```

Frozen receipt: [`results/dialogue_gates.json`](results/dialogue_gates.json)  
Readable audit: [DIALOGUE_RESULT.md](DIALOGUE_RESULT.md)

| Gate | Result | Interpretation |
|---|---:|---|
| **D0 useful channel** | intact **1.000**, swapped **0.000**, foreign crossplay **0.475** | learned pulse use is causal and pair-specific |
| **D1 predictive pulse** | **0.458 -> 0.996**, no-pulse **0.500** | the pulse becomes immediately useful to B |
| **D2 coupling tracks success** | **r = 0.885** | stronger cue/pulse coupling accompanies better joint performance |
| **D3 repair** | **0.000 -> 1.000** after wire swap + adaptation | a pair can renegotiate |
| **D4 private code / shared relation** | opposite conventions split 10/10; foreign **0.474**; canonical **1.000** | raw token identity is private while the task relation is shared |

The protocol gel is intentionally minimal. It establishes a signaling-game control before asking whether the full spatial substrate can do the same thing.

---

## S0-S4 — inherited geometry attacks the launcher

Moving directly to a full `JelloWorld` exposed a confound. With naturally aligned ports, virgin geometry solved the task before learning.

The body is intentionally born with two strong lanes:

```text
A ==================> C

F ==================> D
```

`A/F` are private cue ports. `C/D` are candidate axon-like launch ports.

Run:

```bash
python spatial_gates.py
```

Frozen receipt: [`results/spatial_gates.json`](results/spatial_gates.json)  
Readable audit: [SPATIAL_RESULT.md](SPATIAL_RESULT.md)

| Gate | Result | Interpretation |
|---|---:|---|
| **S0 inherited shortcut** | normal virgin **1.000** | geometry can masquerade as learning |
| **S1 channel causality** | **1.000 -> 0.000** after wire swap | the physical mapping really matters |
| **S2 naive repair** | greedy **0.000**, late reward **0.000** | the inherited route monopolizes the launcher |
| **S3 negative-image repair** | greedy **1.000**, late reward **1.000** | context-keyed prediction error opens exploration/credit and real spatial writing builds the alternate road |
| **S3 no-write attacker** | late reward **0.492** | subtraction alone is not the solution |
| **S3 global predictor attacker** | greedy **0.000** | this is not global normalization |
| **S3 wrong-context attacker** | greedy **0.000** | the cancellation signal must be context keyed |
| **S4 fast predictor attacker** | slow predictor **1.000**, fast predictor **0.500** | a surprise-only launcher can learn away the innovation it needs |

S3 was inspired by adaptive sensory cancellation / negative-image ideas: compare what the body produced with what its current context predicts, then use the residual rather than raw magnitude for exploration and credit.

But S4 originally tempted us into an over-broad rule: "prediction must be slow." S5-S8 show that is not the stronger conclusion.

---

## S5-S8 — surprise is not relevance

Recent chandelier-cell work made an edge case impossible to ignore: a familiar stimulus can habituate, then regain salience after becoming behaviorally predictive.

So S5-S8 deliberately isolate **ping economy** from the spatial problem.

Two event identities occur in two contexts. A fast expectation learns ordinary event magnitude:

```text
surprise = |observed - predicted(event)|
```

A separate state learns contextual consequence:

```text
relevance = R(context, event)
```

The dual gate is only:

```text
ping if surprise + relevance > threshold
```

The current outcome is delivered after the current ping decision, so there is no same-trial target leak.

Run:

```bash
python salience_gates.py
```

Frozen receipt: [`results/salience_gates.json`](results/salience_gates.json)  
Readable audit: [SALIENCE_RESULT.md](SALIENCE_RESULT.md)

| Gate | Result | Interpretation |
|---|---:|---|
| **S5 novelty habituation** | ping **0.851 -> 0.000**; surprise **0.467 -> 0.017** | repeated harmless events stop demanding pings |
| **S6 familiar but relevant** | dual gate target **1.000**, surprise-only **0.000** | expected does not mean unimportant |
| **S6 ping economy** | dual gate pings **0.250** of uniform events at precision **1.000**; always-ping precision **0.250** | relevance restores only the useful pings |
| **S6 relevance ablation** | target **1.000 -> 0.000** | the separate relevance state is causal in this isolator |
| **S7 contextual relevance** | correct-context **1.000**, wrong-context **0.000** | the same familiar event can matter here and be ignorable there |
| **S7 global relevance attacker** | correct **1.000**, wrong-context **1.000** | one salience scalar per token overgeneralizes |
| **S8 fast expectation** | prediction rate **0.80**, dual target **1.000**, surprise-only **0.000** | fast expectation is compatible with useful output when relevance is separate |

All S5-S8 means use **24 independent seeds**.

The revised structural hypothesis is therefore:

> **Prediction, relevance, and structural consolidation should not be collapsed into one state variable.**

A useful shorthand for the next architecture is:

```text
FAST
expectation / mismatch
"what is surprising?"

MEDIUM
contextual relevance
"what matters?"

SLOW
roads / AIS-like launch apparatus
"what has become part of the system?"
```

Those labels are toy-design hypotheses, not claims that chandelier cells literally implement those three timescales.

---

## A more careful chandelier/AIS analogy

The biology motivates experimental questions, not component renaming.

- The **axon initial segment (AIS)** is the specialized boundary where action potentials are initiated/shaped and where somatodendritic and axonal compartments are separated.
- **Chandelier / axo-axonic cells** target the AIS specifically.
- Recent visual-cortex work reports strong chandelier responses to arousal-related events including visuomotor mismatch, reduced responses after repeated visuomotor experience, and accompanying ChC/AIS structural plasticity.
- Recent prefrontal work reports chandelier responses related to novelty/intensity and **acquired salience**: a familiar cue can become responsive again after learning makes it behaviorally predictive.
- Separate 2024 work reports that changing chandelier input, unlike PV basket-cell input, can drive homeostatic tuning of AIS morphology, sodium-channel expression, and principal-cell excitability.

References used as inspiration:

- Leterrier C. *The Axon Initial Segment: An Updated Viewpoint.* J Neurosci. 2018. doi:10.1523/JNEUROSCI.1922-17.2018
- Seignette K et al. *Experience shapes chandelier cell function and structure in the visual cortex.* eLife. 2024. doi:10.7554/eLife.91153.3
- Zhao R et al. *Axo-axonic synaptic input drives homeostatic plasticity by tuning the axon initial segment structurally and functionally.* Sci Adv. 2024. doi:10.1126/sciadv.adk4331
- Zhang K et al. *Prefrontal chandelier cells encode stimulus salience to influence learning in male mice.* Nat Commun. 2026. doi:10.1038/s41467-026-68959-3
- Bell CC. *An efference copy which is modified by reafferent input.* Science. 1981.
- Bell CC et al. *Synaptic plasticity in a cerebellum-like structure depends on temporal order.* Nature. 1997.

JelloBrain does **not** claim that chandelier cells are prediction-error neurons, that the AIS performs mormyrid negative-image subtraction, or that the toy reproduces cortical physiology.

---

## The next hard gate — S9: relevance must bootstrap the road, then get out of the way

S5-S8 are intentionally an abstract two-by-two isolator. Their relevance state can learn the consequential context/event pair directly. That is useful as a falsification control, but it would be cheating to leave that table permanently beside a spatial organism.

So S9 has a stricter requirement:

```text
candidate spatial route
       ↓
fast expectation
       +
transient contextual relevance
       ↓
AIS-like launch eligibility
       ↓
rewarded real JelloWorld traffic
       ↓
slow material route strengthens
       ↓
REMOVE / RELAX RELEVANCE
       ↓
behavior must survive in the spatial substrate
```

If behavior dies when the relevance state is removed, S9 fails: we merely hid the policy in another table.

There is also a conceptual correction to make in S9:

> **which output wins** and **whether that output deserves an outward ping** should be tested as separate operations.

The present S3 launcher uses a residual partly to choose between `C/D`. A more AIS-like architecture should let spatial competition create candidate outputs and let a boundary circuit control their **emission/eligibility**, rather than making the boundary itself secretly store the content of the decision.

Only after that passes should the repo return to the grander target: two full spatial bodies with private interiors, sparse axon-like events, corollary/context signals, foreign-code attacks, wire swaps, and repair.

---

## Run everything

```bash
python -m pip install -r requirements.txt
python gates.py
python dialogue_gates.py
python spatial_gates.py
python salience_gates.py
pytest -q
```

CI runs all four gate suites on every push.

## Main files

- `jellobrain.py` — writable 2-D substrate
- `gates.py` / `results/gates.json` — W0-W5 Phase-I gates
- `dialogue.py` / `dialogue_gates.py` — minimal two-private-system signaling control
- `DIALOGUE_RESULT.md` — readable dialogue audit
- `spatial_launcher.py` / `spatial_gates.py` — S0-S4 inherited-route / negative-image launcher audit
- `SPATIAL_RESULT.md` — readable spatial audit
- `salience_gate.py` / `salience_gates.py` — S5-S8 surprise/relevance/ping isolator
- `results/salience_gates.json` / `SALIENCE_RESULT.md` — frozen salience receipt and audit
- `MODEL.md` — Phase-I equations and claim boundary
- `index.html`, `dialogue.html`, `spatial.html` — browser labs

## Claim boundary

The strongest safe statement currently is:

> **JelloBrain contains a spatial toy in which ordered traffic writes persistent operator geometry; a minimal signaling control in which initially meaningless pulses acquire pair-specific, repairable roles; a spatial launcher audit in which context-specific prediction can expose alternatives hidden by inherited geometry; and a separate launch-boundary isolator showing that novelty and learned contextual relevance must be distinguished if familiar-but-important events are to remain eligible for sparse output.**

Not understanding. Not consciousness. Not a biological model of chandelier cells, the AIS, or electric fish. Not yet two full spatial communicators.

That last gap is why the repo is still called **JelloBrain**.
