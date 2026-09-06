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

## D0–D4 are now implemented

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

That limitation is now the next experiment rather than something to hide.

## The important attackers

The dialogue demo is attacked rather than judged by visual synchrony:

- physically swap the two pulse wires after learning;
- replay a sender's code into a foreign receiver;
- freeze adaptation after the channel is perturbed;
- compare pulse-visible prediction with a no-pulse attacker;
- inspect whether independent pairs converge on one imposed token identity or genuinely choose private conventions;
- report task reward as the primary outcome.

A pretty synchronized animation is not communication. **The pulses must change what the pair can jointly do.**

## Next gate — Spatial Dialogue

The protocol isolator works. Now remove its biggest convenience.

Replace:

```text
cue -> pulse associative material
cue × pulse -> action associative material
```

with two actual spatial `JelloWorld` bodies:

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

Then rerun D0–D4 unchanged as far as possible.

That is the hard question now:

> **Can geography itself carry a negotiable protocol?**

If spatial D0–D4 fail while the protocol gel passes, we learned that the clean signaling-game result does not survive the physical substrate. If they survive, Phase I's strange writable geography has become a genuine communication medium.

## Claim boundary

Passing D0–D4 supports an emergent, pair-specific, repairable signaling convention in a minimal artificial material system. It does not establish language understanding, biological speaker–listener coupling, qualia transfer, consciousness, or thought.
