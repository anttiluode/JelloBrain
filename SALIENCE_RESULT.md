# S5-S8 — surprise is not relevance

This audit follows the spatial S0-S4 launcher experiment.

S4 showed a failure: when the launch boundary used only prediction error (`observed - expected`), a fast predictor could learn away the very residual needed to repair a newly useful route.

That result was real, but the broad verbal conclusion was too strong.

The next isolator asks a sharper question:

> Can an event become perfectly expected and still remain worth pinging?

The answer in this toy is yes — but only if **relevance is stored separately from surprise**.

## Model

There are two familiar event identities (`e0`, `e1`) and two contexts (`c0`, `c1`). A fast predictor learns the ordinary magnitude of each event:

```text
surprise = |observed - predicted(event)|
```

A separate state learns whether an event has become consequential in the current context:

```text
relevance = R(context, event)
```

The dual launch gate is:

```text
ping if surprise + relevance > threshold
```

The current outcome is applied only **after** the ping decision, so there is no same-trial target leakage.

The relevance signal is deliberately external in this isolator. This experiment does not claim to model how reward/salience reaches chandelier cells or the AIS.

## S5 — novelty habituates

All events begin harmless. The sensory event itself does not change.

Across 24 seeds:

- initial ping rate: **0.851**
- late ping rate after repetition: **0.000**
- mean surprise: **0.467 -> 0.017**

So a fast expectation can economically suppress repeated harmless pings.

## S6 — familiar but relevant

After familiarization, `e0` becomes consequential only in context `c0`.

Frozen evaluation:

| gate | relevant `c0/e0` | wrong-context same event `c1/e0` | uniform ping rate | ping precision |
|---|---:|---:|---:|---:|
| surprise only | **0.000** | 0.000 | ~0 | — |
| surprise + contextual relevance | **1.000** | **0.000** | **0.250** | **1.000** |
| always ping | 1.000 | 1.000 | **1.000** | **0.250** |

Erase only the learned relevance state after training and the relevant ping falls **1.000 -> 0.000**.

The important result is therefore not simply "the gate can learn the answer". It is this distinction:

```text
familiar != irrelevant
expected != unimportant
```

A surprise-only machine confuses those pairs.

## S7 — relevance is contextual

A dangerous shortcut is to learn one salience scalar per event.

That attacker does learn that `e0` matters, but then pings for it in both contexts:

```text
context-specific relevance
    c0/e0  1.000
    c1/e0  0.000

global event relevance
    c0/e0  1.000
    c1/e0  1.000
```

So "this token matters" is insufficient. The same token can be important here and ignorable there.

## S8 — fast expectation is allowed again

This is the correction to our first interpretation of S4.

Set the prediction learning rate to **0.80** — deliberately fast.

With surprise only:

```text
familiar relevant ping = 0.000
```

With surprise and relevance separated:

```text
familiar relevant ping = 1.000
wrong-context ping      = 0.000
```

So the stronger hypothesis is **not**:

> prediction must always be slower than structural learning.

It is:

> prediction, relevance, and structural consolidation must not be collapsed into one state variable.

That allows the cleaner architecture:

```text
FAST
expectation / mismatch
"What is surprising?"

MEDIUM
contextual relevance
"What matters?"

SLOW
roads / AIS / launch apparatus
"What has become part of the system?"
```

Those labels are hypotheses about the toy architecture, not claims about the timescales of chandelier cells.

## Why this was prompted by chandelier cells

The biological literature motivates the *question*, not the implementation.

Recent work reports chandelier-cell responses to unexpected/arousal-related events and visuomotor mismatch, experience-dependent changes in those responses, and structural changes in chandelier/AIS connectivity. A 2026 prefrontal study also reports responses to novelty and acquired salience: a familiar cue can regain chandelier-cell response after becoming behaviorally predictive.

That is exactly the edge case a pure negative-image launcher misses.

But this code is not a chandelier-cell model. It does not reproduce GABA, membrane conductance, AIS cable theory, circuit inputs, or calcium dynamics.

## What to build next

The next spatial experiment should not simply append a "salience neuron" table to the old launcher.

Instead, the target is a **ping economy**:

1. the body produces candidate launch activity;
2. a fast expectation marks ordinary/self-predictable drive;
3. a contextual relevance process can keep a familiar but important candidate eligible;
4. the AIS-like boundary emits only a sparse external event;
5. successful repeated events alter the actual spatial material;
6. eventually the slow body should carry enough of the policy that the transient relevance signal can relax.

That final handoff is the hard part. If relevance permanently stores the policy, we have merely hidden another lookup table beside the AIS.

The next falsification gate should therefore ask:

> **Can relevance bootstrap a route and then get out of the way?**

That would connect S5-S8 back to the spatial JelloWorld rather than leaving the new distinction in an abstract two-by-two isolator.
