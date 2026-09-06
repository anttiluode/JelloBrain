# Phase II — two private gels and a narrow channel

The first JelloBrain experiment asked whether meaningless pulse sequences can leave slow, order-sensitive structure and whether the resulting substrate can compute. Phase II preserves the boundary between two systems:

```text
private gel A  -> tiny pulse channel -> private gel B
private gel A  <-       no shared state        -> private gel B
```

Neither side can inspect the other's slow material. A sees private bit `x`, B sees private bit `y`, one meaningless binary pulse crosses A→B, and B is rewarded only when its action equals `x XOR y`.

No symbol is assigned a meaning by the experimenter. Different random seeds are free to settle on opposite pulse conventions.

## Why the Stephens–Silbert–Hasson paper matters

Stephens, Silbert & Hasson (PNAS, 2010, doi:10.1073/pnas.1008662107) measured a speaker telling an unrehearsed story and listeners hearing it. They reported widespread spatial and temporal speaker–listener coupling during successful communication. The coupling largely disappeared when communication was broken by an unintelligible Russian story or by pairing the listener with a different English story. On average the listener response lagged the speaker, while some frontal/striatal regions showed anticipatory activity; the extent of anticipatory coupling was associated with better comprehension.

That paper is **inspiration for tests, not validation of this toy**. Its signal is fMRI/BOLD, its temporal precision is limited, and it does not claim a gelatin substrate or pulse vocabulary.

## D0–D4 are implemented

Run:

```bash
python dialogue_gates.py
```

The frozen receipt is in [`results/dialogue_gates.json`](results/dialogue_gates.json), with a readable audit in [`DIALOGUE_RESULT.md`](DIALOGUE_RESULT.md).

| Gate | Current receipt | Interpretation |
|---|---:|---|
| **D0 — communication-specific channel** | intact **1.000**, swapped **0.000**, foreign code **0.475** | function depends causally on the learned pair-specific pulse convention |
| **D1 — predictive pulse effect** | post-pulse correct probability **0.458 → 0.996**, no-pulse **0.500** | the learned pulse becomes immediately useful to B; this is only a discrete predictive analogue |
| **D2 — coupling tracks success** | **r = 0.885**, 48 pairs | stronger cue→pulse coupling accompanies better joint performance across learning stages |
| **D3 — repair** | **0.000 → 1.000** after wire swap + adaptation | interaction can renegotiate after the physical channel mapping changes |
| **D4 — private code / shared relation** | 10 pairs chose each opposite convention; crossplay **0.474**; canonical relation **1.000** | raw token identity is private/idiosyncratic while the relational task structure is shared |

All five frozen thresholds pass.

## What the current protocol gel actually is

Phase II deliberately isolates communication before forcing it into the full spatial substrate.

A owns private slow material:

```text
sender_material[x, pulse]
```

B owns separate private slow material:

```text
receiver_material[y, pulse, action]
```

Successful traffic deposits the chosen road. Failed traffic erodes it. Unused material relaxes toward baseline. Only the pulse identity crosses the channel.

This is a minimal reward-modulated material signaling game. It is **not yet the full 2-D JelloWorld**.

## The important attackers

The dialogue demo is attacked rather than judged by visual synchrony:

- physically swap the two pulse wires after learning;
- replay a sender's code into a foreign receiver;
- freeze adaptation after the channel is perturbed;
- compare pulse-visible prediction with a no-pulse attacker;
- inspect whether independent pairs converge on one imposed token identity or genuinely choose private conventions;
- report task reward as the primary outcome.

A pretty synchronized animation is not communication. **The pulses must change what the pair can jointly do.**

# Spatial transition — the first direct attempt failed

The obvious next move was to replace the protocol tables with two actual spatial `JelloWorld` bodies:

```text
private spatial world A
    cue enters at a port
    activity traverses grown directed material
    a boundary launcher chooses a travelling pulse
                |
                v
          narrow channel
                |
                v
private spatial world B
    pulse enters at a boundary port
    combines with B's private cue
    spatial dynamics choose an action launcher
```

But the first version revealed a confound before learning even began.

With geometrically aligned ports, virgin space already preferred one cue→launcher mapping. The system scored perfectly on the normal channel **before any adaptation**. Swapping the physical channel then turned that inherited mapping into the exactly wrong answer, and naive rewarded replay could not escape it.

That failure is now frozen instead of being tuned away.

## S0–S4 — launch boundary audit

See [`SPATIAL_RESULT.md`](SPATIAL_RESULT.md), [`spatial_launcher.py`](spatial_launcher.py), and [`spatial_gates.py`](spatial_gates.py).

The question changed from:

> Can geography carry a protocol?

into:

> **How does a launch boundary tell the road the body inherited from the road experience has made relevant?**

The controlled S-series uses one full spatial body with two deliberately inherited lanes and then swaps its output wires.

| Gate | Receipt | Interpretation |
|---|---:|---|
| **S0 inherited shortcut** | virgin normal **1.000** | geometry itself can fake a learned protocol |
| **S1 channel causality** | **1.000 → 0.000** after swap | the physical output mapping is causal |
| **S2 naive raw repair** | greedy **0.000** | inherited excitation monopolizes launch choice |
| **S3 negative-image repair** | greedy **1.000**, late reward **1.000** | context-keyed expected-response subtraction opens exploration/credit and local material writing grows the alternate route |
| **S3 no-write attacker** | late reward **0.492** | prediction/subtraction alone does not solve the task |
| **S3 global predictor attacker** | greedy **0.000** | global normalization is insufficient |
| **S3 wrong-context attacker** | greedy **0.000** | prediction has to be keyed to the correct private context |
| **S4 predictor timescale** | slow **1.000**, fast **0.500** | expectation can adapt so quickly that it erases the innovation needed for repair |

All numbers are means across 12 independent seeds.

## Why electric fish and the axon belong here — separately

The mormyrid electric-fish literature gives a real precedent for a system learning a **negative image** of predictable self-generated input and subtracting it so unexpected residual structure remains visible. That motivates the S3 computation:

```text
innovation
 = observed launcher response
 - predicted self-generated response given context
```

The axon initial segment motivates a different part of the architecture. It is a specialized boundary between somatodendritic and axonal compartments and a major site of action-potential initiation and excitability control. Its composition, position, and morphology are plastic.

So JelloBrain should not pretend that one biological mechanism does everything. The cleaner decomposition is:

```text
PRIVATE SPATIAL BODY
        ↓
AIS-LIKE LAUNCH BOUNDARY
        ├──────────── external travelling pulse / axon
        │
        └──────────── private corollary copy
                             ↓
                    EXPECTATION / NEGATIVE IMAGE
                             ↓
                         innovation
                             ↓
                    credit / future launch
```

The fish inspires **prediction + cancellation**. The AIS inspires **launch + compartment boundary**. They are not the same biological claim.

## S4 adds a timescale constraint

The predictor cannot simply chase the body as fast as possible.

When the S-series predictor tracks the current response too quickly, it absorbs the newly useful route into expectation almost immediately and cancels the residual that was giving that route credit. Repair falls back to chance.

So the current structural hypothesis is:

> **Expectation must be slower than the innovation it is supposed to reveal.**

Or in the language of the wider project:

```text
fast       current traffic / exploration
medium     learned consequences / route change
slower     expected self-generated operator response
```

The exact numerical rates in the toy are not biological claims. The dependency on relative timescale is the thing to test.

# Next gate — S5: two full spatial communicators

Now return to the original Phase-II task, but carry the S3/S4 lesson with us.

```text
SPATIAL A                         SPATIAL B
private x                         private y
   ↓                                 ↓
material body                    material body
   ↓                                 ↑
AIS-like launch ── pulse ────────────┘
   │                                 ↓
   └─ private corollary copy     residual filter
                                     ↓
                                action launcher
                                     ↓
                                  XOR reward
```

Requirements:

- two full `JelloWorld` interiors;
- no table receiver;
- no globally assigned pulse meaning;
- only the travelling pulse crosses the boundary;
- launch events create their own private corollary-copy signal;
- prediction is learned from self-generated/context-linked responses, not the reward target;
- spatial route changes still use local material writing;
- rerun wire swap, foreign crossplay, frozen plasticity, prediction-context scrambling, and repair.

If S5 fails while D0–D4 and S0–S4 pass, that is useful: the protocol and single-boundary mechanisms do not compose into spatial communication. If it survives, then the original question becomes legitimate again:

> **Can geography itself carry a negotiable protocol?**

## Claim boundary

D0–D4 support an emergent, pair-specific, repairable signaling convention in a minimal artificial material system. S0–S4 separately support a context-specific residual mechanism for repairing a remapped launch boundary in one spatial body. Neither establishes language understanding, biological speaker–listener coupling, an AIS model, electric-fish circuitry, qualia transfer, consciousness, or thought.
