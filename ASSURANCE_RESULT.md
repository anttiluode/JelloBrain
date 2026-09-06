# S12 — Habituated plasticity and assurance

S12 asks a narrower question than the earlier POKE experiments:

> **Once an outward route is working reliably, should reward keep rewriting it?**

In the current shared spatial sheet, the answer is **no**.

This is a toy plasticity-scheduling result. It is not a chandelier-cell model, an AIS self-test model, or a claim about biological learning rules.

## Setup

The body starts with the same inherited two-lane `JelloWorld` used in S0-S11. The physical pulse wires are swapped, so the inherited launcher preference is wrong.

The transient controller stores only one scalar per cue:

```text
needs_attention[cue]
```

It stores no desired launcher identity and no correct-output table.

A failed outward consequence sets `needs_attention` high. While attention is high, an exploratory POKE can sample another launcher. If that alternative succeeds, normal local JelloWorld traffic writes the route into the material. Repeated success decays attention.

The crucial S12 change is that **reward alone is not enough to authorize a material write**:

```text
failure
  -> attention high
  -> POKE / plasticity eligible

successful alternative
  -> write route
  -> attention decays

familiar success
  -> use route
  -> no POKE
  -> no further write
```

The final test erases the controller and reads the launchers from raw JelloWorld activity only.

## Result

Across **12 independent seeds**:

| condition | final raw accuracy after controller erase | last-200 reward | last-200 POKE rate | last-200 write rate | mean total writes |
|---|---:|---:|---:|---:|---:|
| reward every success | **0.500** | 0.674 | 0.332 | 0.674 | 999.7 |
| failure-gated plasticity | **1.000** | **1.000** | **0.000** | **0.000** | **121.0** |
| failure-gated + explicit assurance probes | **1.000** | **1.000** | **0.000** | **0.000** | 130.8 |

Frozen machine-readable receipt: [`results/assurance_gates.json`](results/assurance_gates.json)

Run it with:

```bash
python assurance_gates.py
```

## What failed

The earlier S9 learner wrote material after every rewarded trial. That sounds harmless, but in a shared sheet it creates positive feedback: whichever route becomes reliable first gets rewarded more often simply because it already wins, and therefore receives still more material.

The result is continual rehearsal of the already-dominant route while the competing route is still trying to become robust.

S10/S11 exposed the same problem from another angle. Two routes could be made formally correct in one scalar sheet, but the shared launcher margin was only about `8.6e-6`, versus about `0.658` for an isolated route. Tiny readout perturbation collapsed the apparently perfect shared solution to chance.

S12 fixes the learning schedule rather than adding another representational channel.

## The useful negative result

We also tried a more elaborate **assurance loop**. After every material write it re-probed both private contexts and re-opened attention for any context whose raw output had been damaged.

That variant works, but it is not needed here.

Failure-gated plasticity **without** explicit assurance probes already reaches 1.000 raw accuracy in every seed and then stops both POKEs and writes.

So the simpler conclusion is stronger:

> **The stable thing should not keep teaching itself that it is stable.**

Or, in the language that motivated the experiment:

> **Do not waste pings — and do not waste plasticity.**

## Why this matters for the next gate

Habituation creates an immediate new danger: a quiet boundary could become permanently complacent.

The next test therefore changes the outward consequence **after** the route is already familiar and quiet.

A useful controller must do all of the following:

1. remain silent while the embodied mapping is working;
2. wake when the same familiar event suddenly has a different consequence;
3. reopen exploration and material plasticity without being told the correct output;
4. embody the replacement mapping in JelloWorld;
5. become quiet again;
6. still work after the controller is erased.

That is S13.
