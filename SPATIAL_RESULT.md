# S-series — the launcher learns to ignore inherited roads

The first attempt to move Phase II into the full 2-D `JelloWorld` hit a confound: fixed geometry could solve the task before learning. That failure is now frozen as **S0–S2** rather than tuned away.

This file reports the next controlled experiment. It isolates one full spatial body and its two candidate output ports before returning to two full communicators.

## The setup

A `JelloWorld` is born with two deliberately strong horizontal lanes:

```text
A ==================> C   launch 0

F ==================> D   launch 1
```

`A/F` are private cue ports. `C/D` are candidate axon-like launch ports.

The inherited lane structure makes the virgin body prefer `A -> C` and `F -> D`. A two-wire channel carries the chosen launch identity downstream. The receiver is intentionally trivial in this audit: delivered pulse identity is the action. That keeps S0–S4 about the **spatial launch boundary**, not about a second hidden learner.

After the channel wires are swapped, the old spatial preference is exactly wrong.

All route changes after birth use the ordinary local `JelloWorld` writing rule. No diagonal road is edited into the material by hand.

Run:

```bash
python spatial_gates.py
```

Frozen receipt: [`results/spatial_gates.json`](results/spatial_gates.json).

## Receipt

| Gate | Result | Meaning |
|---|---:|---|
| **S0 — inherited geometric shortcut** | virgin normal channel **1.000** | geometry alone can masquerade as learned protocol |
| **S1 — channel causality** | normal **1.000 → 0.000** after wire swap | the physical mapping is genuinely causal |
| **S2 — naive raw repair** | final greedy **0.000**, final-100 reward **0.000** | raw excitation is so dominated by the inherited lane that the counterfactual launch is never sampled enough to grow |
| **S3 — negative-image repair** | final greedy **1.000**, final-100 reward **1.000** | context-keyed expected-response subtraction opens credit/exploration and rewarded traffic grows the alternate spatial road |
| **S3 attacker — residual but no material writing** | final-100 reward **0.492** | subtraction alone does not solve the task |
| **S3 attacker — global prediction** | greedy **0.000** | a single global normalization does not remove the cue-specific geometric bias |
| **S3 attacker — wrong prediction context** | greedy **0.000** | the negative image must be keyed to the correct corollary/context signal |
| **S4 — predictor timescale** | slow tracker **1.000**, fast tracker **0.500** | if expectation absorbs new route changes too quickly, it cancels the innovation needed for repair |

Numbers above are means over **12 independent seeds**.

## What changed in S3

The launcher no longer asks only:

```text
which output is loudest?
```

It first learns an expectation of what its own inherited geometry normally produces for each internal cue:

```text
cue / corollary context
        ↓
expected C,D response
```

Then current traffic is compared with that expectation:

```text
innovation = log(observed launcher response) - predicted log response
```

The prediction model never sees the reward target. It is calibrated only from self-generated cue responses, with small amplitude jitter, and then tracks them slowly.

At the moment of the wire swap, the inherited geometry is still huge in the raw signal but close to zero in the residual. Exploration is no longer monopolized by the old lane. When a counterfactual launch happens to earn reward, the ordinary spatial replay rule deposits material along that cue/launch trajectory. That learned deviation now appears in the residual and wins more often, which deposits more material.

So the loop becomes:

```text
inherited geography
       ↓
expected self-response ─────────┐
       ↓                        │ subtract
current spatial response ───────┘
       ↓
innovation
       ↓
launch exploration / credit
       ↓
consequence
       ↓
local material writing
       ↓
new response exceeds old expectation
```

## Why the electric-fish analogy is useful — and where it stops

Mormyrid electrosensory circuits provide a real biological precedent for **context-linked prediction and subtraction**. Bell's classic efference-copy work and later anti-Hebbian studies describe learned negative images that oppose predictable self-generated sensory input, allowing unexpected sensory components to remain prominent.

That motivates the computational move here: don't globally weaken the dominant signal; predict the part attributable to the system's own context/action and subtract that prediction.

But this S-series mechanism is **not a model of the fish ELL**. In the fish, negative images filter sensory reafference. Here the same algorithmic idea is used as a launch/credit filter in a toy spatial substrate.

## Why the axon initial segment belongs in the architecture

The AIS is a much better biological analogy for the **boundary** than for the negative image itself. It sits between somatodendritic and axonal compartments, initiates and shapes action potentials, and is itself structurally and functionally plastic.

That suggests a cleaner decomposition for JelloBrain:

```text
PRIVATE MATERIAL BODY
    distributed integration / route history
                ↓
AIS-LIKE LAUNCH BOUNDARY
    decides whether/which travelling event leaves the body
                ↓
AXON / NARROW CHANNEL
    external pulse

parallel internal copy of launch/context
                ↓
NEGATIVE-IMAGE FILTER
    predicts self-generated consequences
                ↓
innovation / credit
```

The AIS and the fish mechanism should not be collapsed into one biological claim. One motivates **launch + compartment boundary**; the other motivates **prediction + cancellation**.

## The unexpected result: S4

The negative image cannot simply be "as adaptive as possible."

When its tracking rate is made very fast (`0.20` per observation in this toy), it learns away the newly useful residual almost as soon as that residual appears. Across 12 seeds the final greedy score falls to **0.500** and late reward returns to chance.

With the slow tracker (`0.0002`), every seed reaches **1.000** greedy repair and the final 100 episodes average **1.000** reward.

This is not evidence that those numerical timescales map to biology. It is a structural result of the toy:

> **To distinguish an expected road from a newly relevant road, the expectation and the road cannot be rewritten on the same effective timescale.**

That is very close to the fast / medium / slow operator problem that motivated the wider repo family.

## Next hard gate — S5

S0–S4 deliberately isolate one spatial launch boundary. The receiver is still trivial.

Now restore the actual Phase-II problem:

```text
spatial A
  private x
  spatial launcher
      |
      | meaningless pulse
      v
spatial B
  private y
  prediction / residual filter
  spatial action launcher
      ↓
  x XOR y reward
```

No table receiver. No hard-coded token meaning. Two full `JelloWorld` interiors. The launch event should also produce a private corollary-copy signal inside the body rather than giving the predictor magical access to the other side.

Then rerun the old communication attackers: wire swap, foreign crossplay, frozen plasticity, context scrambling, and repair.

That is the point where "geography carries a negotiable protocol" becomes a real question again.

## Claim boundary

S0–S4 support a narrower statement:

> **In this controlled spatial toy, a learned context-specific prediction of inherited launcher responses can remove a fixed geometric bias from credit assignment, allowing reward-modulated local material writing to build an alternate route after a channel remap.**

Not a brain model. Not an AIS model. Not an explanation of corollary discharge. Not yet two spatial communicators.
