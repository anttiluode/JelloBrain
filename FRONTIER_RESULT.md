# Frontier result — conflict, copies, receipts, and forced delay history

This file freezes the sequence after `ADDRESSABILITY_RESULT.md`.  The point is to
keep the failures next to the positive results so the next pass does not
rediscover the same mechanism under a new metaphor.

## S17 — interference can cause an address to be born

A single slow material sheet begins with no cue->band table.  The system retains
a temporary before-write reference, performs an ordinary local material write,
measures the resulting write-effect vector, and compares it with the previous
write effect.

When successive effect vectors become strongly opposed (`cosine <= -0.90`), a
new anonymous slow band is created.

Across 8 seeds the split occurred in 8/8 runs, at mean episode ~252.9.  The
minimum observed effect cosines were typically between -0.93 and almost -1.0.
After the split, the two cues occupied different anonymous bands in 8/8 runs.

An attacker copying the *post-write* state instead of the preserved pre-write
state also succeeded in this easy address-birth assay.  Therefore S17 does **not**
earn a claim that a literal pre-intervention snapshot is necessary.

The narrow result is:

> **Opposed measured intervention effects can themselves provide a trigger for
> creating a protected write direction in this toy.**

## S17R — address birth is not relevance

The hard cycle was restored:

```text
SWAPPED -> NORMAL -> SWAPPED
```

Mean raw accuracy:

```text
no split             1.000 -> 0.375 -> 1.000
pre-write split      1.000 -> 0.500 -> 1.000
post-write split     1.000 -> 0.125 -> 1.000
```

The split protects old structure and improves robustness in some phases, but it
does not reliably tell the launcher which preserved structure is currently
relevant after meaning changes.

So:

```text
ADDRESS     = where can incompatible knowledge live?
RELEVANCE   = which preserved knowledge applies now?
```

Those are different problems.

## S18 — scalar relevance is too coarse

A single consequence-tracked scalar was attached to each anonymous band.  The
hope was that current success/failure could gate which old address should speak.
It failed: the final reversal settled around chance (~0.5).

A whole band can contain useful and obsolete relations simultaneously.  Calling
the entire address "relevant" or "irrelevant" is not enough.

## S19 — exact copy/delta planning can preserve the wrong thing perfectly

After a successful POKE, every candidate write address was tested on an internal
copy.  The allocator chose the address whose write least disturbed the other
currently working behavior.

This implements the old question literally:

> "What may I change without damaging something else?"

It failed worse than random allocation in the middle reversal:

```text
random write address     reverse NORMAL raw accuracy 0.25
copy/delta allocator     reverse NORMAL raw accuracy 0.00
```

The reason is precise: when the outside mapping changes, the behavior being
protected may itself be obsolete.

The question therefore has to be qualified:

> **What may I change without damaging something I currently have evidence is
> still valid?**

## S20 — validation receipts help, but do not solve reversal

S20 added one binary validation receipt per encountered cue.  A receipt says only
that the currently committed embodied response for that cue has recently
succeeded.  It stores neither the desired launcher nor a cue->band identity.
Successful random POKEs cannot mint receipts.

Mean raw accuracy over 4 seeds:

```text
                         SWAPPED    NORMAL    SWAPPED again
blind assurance            0.75      0.25          0.75
local receipt clearing     0.75      0.50          0.75
global receipt clearing    0.75      0.375         1.00
```

Local invalidation improves the difficult middle phase, while global clearing
restores the final return-to-SWAPPED phase most reliably.  Neither is a complete
solution.  Global clearing is also a strong generic change-point assumption and
must not be smuggled in as a biological mechanism.

## S21 — the Takens/efference-copy bridge

The side question was whether the familiar efference-copy equation and the old
Takens/delay-embedding work are actually the same mathematical object.

They are not.  The useful composition is:

```text
delayed observation history  -> estimate hidden state relevant to prediction
copy of own action            -> identify the forcing the system itself issued
forward model                 -> predict the self-generated return
actual return                 -> observe what happened
innovation                    -> actual - predicted self-effect
```

This is closer to a **forced/action-conditioned delay embedding** than to the
simplest autonomous Takens theorem.

`takens_efference_probe.py` hides the Jello material arrays from a predictor.  A
single fixed scalar observation is measured before/after random local writes.
The predictor may or may not receive delayed scalar history and a copy of the
write action.

Held-out self-effect prediction:

```text
model                 MSE          R^2
CURRENT            0.009580      -0.011
DELAY only         0.009654      -0.019
EFFERENCE          0.004316       0.545
FORCED DELAY       0.004116       0.566
```

The dominant gain is the efference copy.  Delay history alone does nothing in
this assay.  Adding an 8-step forced history gives a modest additional gain.
Scrambling the efference/action correspondence increases forced-delay MSE by
**3.50x**.

Then occasional unreported extra material writes were inserted.  Raw absolute
change could not detect them:

```text
AUC |raw delta|                    0.483
AUC |innovation|                   0.646
AUC innovation, action scrambled  0.472
```

So the computational statement analogous to the electric-fish idea survives:
**predict the component attributable to one's own action, subtract it, and an
unmodelled perturbation becomes more visible.**

## S21R — delay depth attacker

The delay depth was swept over `1,2,4,8,16,32`.  `k=1` is effectively the
current-state + current-action model, so improvements at `k>1` measure what
history itself buys.

The best relative point was `k=8`:

```text
k=1   forced/efference MSE ratio 1.000   shock innovation AUC 0.597
k=2                               0.987                        0.611
k=4                               0.972                        0.617
k=8                               0.954                        0.646
k=16                              1.004                        0.619
k=32                              0.995                        0.607
```

There is therefore a **finite-memory optimum in this particular predictor**, but
not a monotone Takens miracle.  Delay-only R^2 remains negative at every tested
length.  The action copy is the indispensable part; temporal history contributes
a smaller state-reconstruction correction.

## Connection back to GeometricNeuronV24

Gate 6C of `GeometricNeuronV24` already contained the relevant architectural
fork.  A local same-field writer developed a real write-timescale optimum, while
an exact `HISTORY_REPLAY` estimator that kept all pulse equations removed the
fast-write penalty.

S21 now supplies a possible object between those extremes:

```text
no history                     exact full history
   |                                  |
local same-field      finite forced delay predictor
```

The clean next cross-repo gate is therefore not "prove dendrites are Takens."
It is:

> **Replace exact HISTORY_REPLAY with a finite action-conditioned delay state and
> measure how much of the fast-write penalty it removes.**

That would test whether a compact temporal basis can substitute for perfect
bookkeeping.

## AIS boundary

The ~190 nm actin/spectrin periodic scaffold at the AIS should stay out of the
delay-bank claim.  Its spatial period corresponds to microsecond-or-shorter
propagation intervals at plausible axonal speeds, and current evidence does not
establish it as a temporal diffraction/delay processor.

AIS remains relevant here as a plastic launch boundary.  The predictive
negative-image / forced-delay computation is a separate role unless evidence
shows otherwise.

## Current frontier

The sequence now looks like:

```text
S3      cancel predictable inherited/self-generated response
S16X    measure collateral write coupling
S17     conflict can create a protected address
S17R    address != current relevance
S19     copy/delta can protect obsolete structure
S20     validation evidence helps but is incomplete
S21     action-conditioned prediction exposes innovation
```

The strongest next question is whether a finite delayed, action-conditioned
state can become the **receipt/prediction object itself**, replacing both exact
material copies and brittle one-bit validity bookkeeping.
