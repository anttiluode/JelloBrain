# Phase II receipt — private gels learn a pulse convention

`python dialogue_gates.py` generates `results/dialogue_gates.json`.

The task is intentionally tiny. A sees private bit `x`, B sees private bit `y`, exactly one of two meaningless pulse identities crosses A→B, and B must output `x XOR y`. The two slow materials are private. No line of code declares what pulse 0 or pulse 1 means.

## Current result

| Gate | Result |
|---|---:|
| D0 useful channel | intact **1.000**, swapped wires **0.000**, foreign-code crossplay **0.475** |
| D1 predictive pulse effect | post-pulse correct probability **0.458 → 0.996**; no-pulse stays **0.500** |
| D2 coupling tracks success | **r = 0.885** across 48 independently seeded / partially trained pairs |
| D3 repair | swap breaks reward to **0.000**; adaptive repair returns greedy reward to **1.000** |
| D4 private code / shared relation | 10 pairs chose `(0→0,1→1)`, 10 chose the opposite; canonical relation accuracy **1.000** |

All five gates pass under the frozen thresholds in `dialogue_gates.py`.

## The strongest result is not 1.000

Perfect XOR performance is easy to engineer. The useful receipt is the *cross-play structure*.

Twenty independently trained pairs split exactly 10/10 between the two possible binary conventions. When a sender is connected to a foreign receiver without translation, mean reward is **0.474**, essentially chance. Yet after pulse identity is canonicalized by its learned relational role, the receiver relation is perfect across pairs.

So the toy has both:

```text
private arbitrary token identity
        +
shared task relation
```

That is a much cleaner artificial analogue of “same communicative relation, privately instantiated code” than making every pair converge on the same hand-coded symbol.

## Repair is also causal

The channel attacker physically swaps the two pulse wires after training. Nothing inside either gel is told that the mapping changed.

```text
before swap       1.000
immediately after 0.000
frozen protocol   0.000
adaptive repair   1.000
```

The old convention therefore is not merely correlated with success: changing the channel mapping destroys the function, and continued interaction can rebuild a new working convention.

## Important limitation

This Phase-II model is a deliberately minimal **protocol gel**, not yet the full 2-D `JelloWorld`. Its slow material is an associative transition substrate:

- A has `cue → pulse` material;
- B has `cue × pulse → action` material;
- rewarded traffic deposits the used road;
- failed traffic erodes it;
- unused material relaxes toward baseline.

That isolation is useful because it tells us the communication experiment itself works before we bury it in spatial dynamics. The next hard gate is therefore obvious:

> **Can the same D0–D4 phenomena survive when those associative roads are replaced by the actual spatial, directed, writable JelloWorld?**

If not, Phase II remains a signaling-game control. If yes, then the strange geography from Phase I has become a communication substrate.

## Claim boundary

This receipt supports an emergent, pair-specific, repairable signaling convention in a tiny reward-modulated artificial material system. It does **not** establish language understanding, speaker–listener neural coupling, consciousness, qualia transfer, or a biological mechanism.

D1 is especially modest: it is a discrete causal analogue in which an incoming learned pulse becomes predictive of the correct joint action. It is not a reproduction of the anticipatory fMRI effect in Stephens, Silbert & Hasson (2010).
