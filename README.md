# JelloBrain

### Can meaningless signals write a medium, grow private conventions, and learn when an internal event is worth sending outside?

**Live Phase I:** https://anttiluode.github.io/JelloBrain/  
**Live Phase II dialogue lab:** https://anttiluode.github.io/JelloBrain/dialogue.html  
**Live spatial launcher lab:** https://anttiluode.github.io/JelloBrain/spatial.html  
**Final research write-up:** [FINAL_RESULT.md](FINAL_RESULT.md)

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

S9-S12  spatial consolidation diagnosis
        exploration alone fails -> interference -> failure-gated plasticity succeeds

S13     contingency reversal
        reactivation wakes learning, but the shared scalar sheet does not remap robustly
```

The research arc now has a useful stopping point. The current substrate can learn stable competing routes, but repeated contingency reversal exposes a stability/plasticity and representational-interference problem. The next clean move would change the representation, not add another hidden controller.

Machine-readable synthesis: [`results/final_summary.json`](results/final_summary.json)

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

But S4 originally tempted us into an over-broad rule: “prediction must be slow.” S5-S8 show that is not the stronger conclusion.

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

A useful shorthand is:

```text
FAST
expectation / mismatch
"what is surprising?"

MEDIUM
contextual relevance
"what matters?"

SLOW
roads / launch apparatus
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

## S9-S13 — where the project actually ends

The later gates turned out to be more useful than simply pushing toward another positive score.

### S9 — POKE is not enough

A mismatch-triggered POKE improves behavior while the controller is active, but after the controller is erased the raw spatial policy is only **0.500**. Always-explore also ends at **0.500**. Therefore the remaining bottleneck is not simply exploration.

Frozen receipt: [`results/poke_gates.json`](results/poke_gates.json)

### S10-S11 — the shared sheet has a margin problem

Oracle writing shows that one scalar sheet can *formally* contain both routes at **1.000** accuracy, falsifying a simple hard-capacity story. But sequential writes leave one route correct at a time (**0.500**), and the shared correct margin is only about **8.59e-6**, roughly **1.31e-5** of an isolated route's margin. Tiny readout perturbation collapses the shared solution to **0.503** while isolated routes stay at **1.000**.

Receipts: [`results/crossing_gates.json`](results/crossing_gates.json), [`results/robustness_gates.json`](results/robustness_gates.json)

### S12 — failure-gated plasticity works

Writing after every rewarded success creates a rich-get-richer loop. Stopping plasticity once a route is familiar and successful fixes the two-route task:

| condition | final raw accuracy | mean writes |
|---|---:|---:|
| reward every success | **0.500** | 999.7 |
| failure-gated plasticity | **1.000** | 121.0 |
| failure-gated + assurance | **1.000** | 130.8 |

The controller can then be erased. The learned behavior remains in the material.

Readable audit: [ASSURANCE_RESULT.md](ASSURANCE_RESULT.md)  
Frozen receipt: [`results/assurance_gates.json`](results/assurance_gates.json)

The reusable principle is:

> **The stable thing should not keep teaching itself that it is stable.**

### S13 — contingency reversal fails

After the mapping is stable and quiet, the outward contingency is changed.

The system correctly wakes up and resumes POKEs/writes, but it does not cleanly remap the shared sheet:

```text
failure-only reactivation:
new mapping 0.500
old mapping 0.500

reactivation + assurance:
new mapping 0.375
old mapping 0.625

no reactivation:
new mapping 0.000
old mapping 1.000
```

A final local route-retirement sweep also finds no clean reversible regime. Depending on weakening rate, one contingency or the other tends to win.

That is the stop point:

> **The missing mechanism is now representational separation / robust addressing between competing routes, not another layer of exploration or attention logic.**

Full synthesis: [FINAL_RESULT.md](FINAL_RESULT.md)

---

## Run everything

```bash
python -m pip install -r requirements.txt
pytest -q
python gates.py
python dialogue_gates.py
python spatial_gates.py
python salience_gates.py
python poke_gates.py
python crossing_gates.py
python robustness_gates.py
python assurance_gates.py
python reversal_gates.py
python retirement_sweep.py
```

CI runs the full sequence on every push. Some late gates are intentionally negative diagnostics; green CI means the result reproduced, not that every scientific hypothesis passed.

## Main files

- `jellobrain.py` — writable 2-D substrate
- `gates.py` / `results/gates.json` — W0-W5 Phase-I gates
- `dialogue.py` / `dialogue_gates.py` — minimal two-private-system signaling control
- `DIALOGUE_RESULT.md` — readable dialogue audit
- `spatial_launcher.py` / `spatial_gates.py` — S0-S4 inherited-route / negative-image launcher audit
- `SPATIAL_RESULT.md` — readable spatial audit
- `salience_gate.py` / `salience_gates.py` — S5-S8 surprise/relevance/ping isolator
- `poke_launcher.py` / `poke_gates.py` — S9 POKE bootstrap failure
- `crossing_gates.py` / `robustness_gates.py` — S10-S11 interference/margin diagnostics
- `assurance_gates.py` / `ASSURANCE_RESULT.md` — S12 failure-gated plasticity
- `reversal_gates.py` / `retirement_sweep.py` — S13 reversal and route-retirement stop point
- `results/final_summary.json` / `FINAL_RESULT.md` — final synthesis
- `MODEL.md` — Phase-I equations and claim boundary
- `index.html`, `dialogue.html`, `spatial.html` — browser labs

## Claim boundary

The strongest safe statement is now:

> **JelloBrain contains a reproducible sequence of toy experiments in which ordered traffic writes persistent spatial operator geometry; initially meaningless pulses acquire pair-specific, repairable roles; context-specific prediction error exposes alternatives hidden by inherited geometry; novelty and learned relevance are separable; and failure-gated plasticity stabilizes competing routes. The same shared scalar substrate still fails at robust repeated contingency reversal because route changes interfere with one another.**

Not understanding. Not consciousness. Not a biological model of chandelier cells, the AIS, electric fish, or dendritic Takens reconstruction.

That failure is a good ending: it says what the next architecture would actually have to change.
