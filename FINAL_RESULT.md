# JelloBrain — final result of this research arc

JelloBrain started with a deliberately loose question:

> **Can meaningless signals write a medium, become meaningful through use, and eventually control what a bounded system sends outside?**

The answer at the end is neither “yes, it thinks” nor “no, nothing happened.”

The useful result is a much narrower architecture, together with a clean failure point.

> **History can become geometry. Geometry can become a predictive prior. Prediction error can expose alternatives. Relevance can decide what deserves attention. Plasticity can be gated by need. But a single shared scalar substrate does not yet support robust repeated reversal without interference.**

That is where this version of JelloBrain should stop.

Machine-readable summary: [`results/final_summary.json`](results/final_summary.json)

The latest full CI run executed the complete chain through `reversal_gates.py` and `retirement_sweep.py` successfully as code. Several of those late gates are intentionally negative scientific results: a green CI run means the audit reproduced, not that every hypothesis passed.

---

## 1. The beginning: signals really can write a spatial operator

Phase I used six meaningless primitives, `A` through `F`, as nothing more than spatially injected temporal pulses. There are no embeddings, semantic labels, pretrained token meanings, or hidden word IDs in the substrate.

The important findings were:

| result | outcome |
|---|---:|
| persistent trace after complete fast-state wipe | **1.00** vs .50 chance |
| order-only discrimination with identical primitive counts | **1.00** |
| intact geometry vs shuffled same material values | **1.00 -> .50** |
| temporal XOR from gel state | **1.00** vs raw-linear upper bound .75 |
| transfer to unseen individualized gels | **1.00** vs .25 chance |

The strongest part is still the geometry attack. Preserving every slow material value while destroying its spatial/directional arrangement destroys the learned distinction.

So the medium is not merely a bag of weights.

> **The arrangement of material acts as an operator compiled by history.**

That was the first thing worth keeping.

---

## 2. Meaningless pulses can become a private communication convention

Phase II then stripped away most of the spatial machinery and asked a cleaner communication question.

Agent A sees private bit `x`. Agent B sees private bit `y`. Only one of two meaningless pulse identities can cross from A to B. B must produce `x XOR y`.

The pair learns without being told that pulse 0 “means” anything.

Across the frozen dialogue gates:

- intact reward reached **1.000**;
- physically swapping the two pulse wires dropped reward to **0.000**;
- foreign-code crossplay was **0.475**, near chance;
- task-relevant coupling tracked success with **r = 0.885**;
- after the wire swap, continued interaction repaired reward from **0.000 to 1.000**;
- independent pairs split evenly between opposite raw binary conventions while retaining **1.000 canonical relational accuracy**.

This is not language. It is a small signaling game.

But it supports one clean sentence:

> **A meaningless token can acquire a pair-specific role because of the history of interaction, while the raw token identity remains private to that pair.**

That was the second thing worth keeping.

---

## 3. Full spatial geometry immediately revealed a cheat

Putting the problem back into a full `JelloWorld` produced the first major negative.

The virgin body was born with strong spatial lanes. Before any learning, ordinary geometry already solved the unswapped launch task at **1.000** accuracy.

Swap the physical output wires and accuracy becomes **0.000**.

Then naive rewarded replay also remains at **0.000**.

The reason is important: raw excitation is not neutral evidence. Inherited geometry already tells the launcher which output “looks strongest.” The system cannot easily discover the alternative because the inherited route wins the competition before experience gets a chance to rewrite it.

This killed a tempting but wrong story:

> strong activation is not automatically learned relevance.

The launcher needs a reference for what the current body would have done anyway.

---

## 4. The useful negative image: observed minus expected

S3 added a context-keyed prediction of the inherited launch response and used the residual rather than raw magnitude:

```text
observed launch activity
        -
predicted activity for this context/body
        =
residual / innovation
```

With that residual opening exploration and actual JelloWorld material writes doing the learning, the swapped spatial mapping repaired to **1.000** across 12 seeds.

The attacks mattered more than the headline score:

- residual but **no material write** stayed near chance, late reward **0.492**;
- one **global** prediction baseline gave **0.000**;
- a **wrong-context** prediction gave **0.000**.

So subtraction itself was not secretly solving the task. The predictor had to be keyed to the current intervention/context, and the successful alternative still had to be embodied by real spatial writing.

This is the point where the efference-copy / negative-image idea became computationally useful in JelloBrain:

> **compare the current body against what this body was expected to do, then let the unexplained part become eligible for learning.**

That does not make the toy a biological corollary-discharge circuit. It gives the toy a reference signal with the same causal purpose.

---

## 5. Prediction is not relevance

The next mistake was also useful.

A fast predictor can make a repeated event unsurprising. If launch is controlled only by surprise, a familiar but newly important event becomes invisible.

S5-S8 therefore separated two variables:

```text
prediction / mismatch:
    what is surprising?

contextual relevance:
    what matters here?
```

The result was clean:

| test | result |
|---|---:|
| repeated harmless event ping rate | **0.851 -> 0.000** |
| familiar event after becoming relevant | **1.000** target ping rate |
| same event, wrong context | **0.000** |
| surprise-only attacker on familiar relevant event | **0.000** |
| dual-gate precision | **1.000** |

The stronger conclusion was therefore not “prediction must be slow.”

It was:

> **Prediction, relevance, and plasticity eligibility are different jobs and should not be stored in one scalar.**

That distinction survived later gates.

---

## 6. POKE alone was not enough

S9 asked whether a transient mismatch-triggered exploratory POKE could bootstrap the real spatial route and then disappear.

It failed.

With POKE plus real material writing, behavior improved while the controller was present, but after erasing the controller the raw spatial policy remained only **0.500**.

Always exploring also ended at **0.500**.

With no material write, final raw behavior was **0.000**.

This was valuable because it removed a whole class of easy fixes:

> **The remaining problem was not lack of exploration.**

Something about how multiple routes coexist in the same writable sheet was unstable.

---

## 7. The shared sheet can hold two routes — but almost without margin

S10 first asked a hard-capacity question with the correct routes supplied by an oracle.

Each route alone works at **1.000**. Balanced interleaved writing can also make the shared sheet formally produce both correct outputs at **1.000**.

So the strongest hard-capacity claim was falsified.

But sequential writing leaves only one route correct at a time: **0.500** after the first route and **0.500** after the second.

S11 then measured the actual launcher margin.

For isolated routes, the mean correct margin is about:

```text
0.658
```

For the formally correct shared two-route sheet:

```text
8.59e-6
```

The shared margin is only about **1.31e-5** of the isolated margin.

Readout perturbation with sigma `0.001` — only about **8.8e-5 of mean response** — leaves the isolated routes at **1.000** but collapses the shared solution to **0.503**, essentially chance.

This changed the diagnosis again.

The issue was not simply whether two routes *exist* in the same material.

> **The issue was whether they are separated strongly enough to survive noise and continued learning.**

---

## 8. The best positive result late in the repo: do not keep rewriting success

S12 changed the plasticity schedule instead of adding a new representational channel.

The previous learner rewrote material after every rewarded success. That creates a rich-get-richer loop: whichever route wins first keeps receiving additional material because it already wins.

S12 allowed writes only while a context still `needs_attention`.

The result across 12 seeds:

| rule | final raw accuracy after controller erase | mean writes |
|---|---:|---:|
| reward every success | **0.500** | **999.7** |
| failure-gated plasticity | **1.000** | **121.0** |
| failure-gated + assurance | **1.000** | **130.8** |

For failure-gated plasticity, the last-200 reward was **1.000**, POKE rate **0.000**, and write rate **0.000**.

The controller can then be erased and the body still works.

This is probably the sharpest reusable principle JelloBrain found:

> **The stable thing should not keep teaching itself that it is stable.**

Or more generally:

> **Plasticity should be driven by unresolved error/need, not by success alone.**

That result is much more interesting than another layer of POKE logic.

---

## 9. The stop point: repeated reversal still breaks the shared representation

S13 deliberately changed the outward contingency *after* the learned spatial mapping had become familiar, successful, and quiet.

Before reversal, the embodied mapping is healthy:

- raw accuracy **1.000**;
- late reward **1.000**;
- late POKE rate **0.000**;
- late write rate **0.000**.

Then the environment changes.

No-reactivation and no-write controls correctly preserve the old mapping but cannot learn the new one:

```text
new mapping 0.000
old mapping 1.000
```

Reactivation does wake the learner, but the shared sheet does not cleanly remap:

```text
failure-only reactivation:
new mapping 0.500
old mapping 0.500

reactivation + assurance:
new mapping 0.375
old mapping 0.625
```

The learner explores and writes. It simply cannot update one relation without damaging another strongly enough to settle into a robust new two-route configuration.

That is a genuine negative result, not a failed run.

---

## 10. Route retirement did not rescue repeated reversibility

The final sweep added local weakening of the route that had just produced a failed consequence.

If naive bidirectional adaptation were the missing ingredient, some retirement rate should have produced reliable learning of:

```text
swapped -> normal -> swapped again
```

Instead the mean raw accuracies across the three phases were:

| retirement rate | learn swapped | reverse normal | swapped again |
|---:|---:|---:|---:|
| 0.000 | **1.000** | 0.375 | **1.000** |
| 0.005 | 0.875 | 0.250 | **1.000** |
| 0.020 | 0.000 | **1.000** | 0.000 |
| 0.080 | **1.000** | 0.000 | **1.000** |

There is no clean reversible regime in this sweep.

Different weakening strengths mostly decide **which contingency wins** rather than letting the same shared sheet represent and repeatedly switch between both robustly.

So this is where adding more controller tricks becomes less informative.

The next change should be representational.

---

# What JelloBrain actually found

The full arc can now be compressed into this:

```text
ordered interaction
      ↓
writable material geometry
      ↓
body-specific expected response
      ↓
observed - expected
      ↓
innovation / mismatch
      ↓
contextual relevance
      ↓
sparse outward POKE / launch eligibility
      ↓
failure-gated plasticity
      ↓
embodied route
```

This chain works surprisingly far.

But once multiple mappings must remain robust **and** change repeatedly, the current scalar spatial material develops interference.

The final architectural lesson is therefore a separation principle:

```text
FAST      prediction / mismatch
          what happened that I did not already expect?

MEDIUM    relevance / attention
          which mismatch matters in this context?

SLOW      structural route memory
          what should become part of the body?

MISSING   robust separation / addressing between competing routes
          how can one route change without erasing another?
```

That missing fourth line is the legitimate successor problem.

---

# What should come next, if anything

Do **not** add another hidden table to the launcher.

Do **not** add more unconditional exploration.

Do **not** keep tuning route-retirement rates until one seed looks good.

The clean next experiments should change how routes are represented:

1. explicit orthogonal/subspace channels;
2. spatial compartments or branch-local memories;
3. frequency- or phase-separated channels;
4. context-dependent gating of a shared physical substrate;
5. a transport operator whose modes can be independently strengthened/retired.

The old audio-loop experiments with controllable band-limited noise become mildly interesting here, not because of their old “quantum” framing, but because **frequency-selective perturbation is a good attacker for channel separation**. If a future system claims two frequency-isolated routes, inject noise into one band and ask whether only the corresponding route degrades.

That would be a new project, not another JelloBrain gate.

---

# Claim boundary

JelloBrain does **not** show a brain, thought, understanding, consciousness, language, biological corollary discharge, a chandelier-cell mechanism, an AIS learning rule, or a Takens implementation in dendrites.

It does show a reproducible sequence of toy results in which:

- traffic history writes spatial operator geometry;
- token order matters after fast state is erased;
- meaningless pulses acquire pair-specific interaction-dependent roles;
- inherited geometry can masquerade as learned preference;
- a context-keyed negative image can expose useful innovation;
- novelty and learned relevance are separable;
- exploration alone is insufficient;
- continual reward-driven rewriting destabilizes shared routes;
- failure-gated plasticity can stabilize them;
- repeated contingency reversal remains unsolved because the shared scalar representation interferes with itself.

That is enough for this repo.

> **Signals made roads. Prediction learned which roads were ordinary. Relevance learned when ordinary was still important. Plasticity learned when to stop. The remaining failure was not attention — it was how multiple roads share the same ground.**
