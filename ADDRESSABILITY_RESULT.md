# Addressability result — S15, S16 and the cross-write matrix

This file freezes the experimental state after reviewing the wider repo lineage. See [`ADDRESSABILITY_LEDGER.md`](ADDRESSABILITY_LEDGER.md) for the historical warning: stable addressability has appeared repeatedly before and is **not** itself a new discovery here.

## S15 — explicit slow-state separation is a complete upper-bound rescue

Cycle:

```text
SWAPPED -> NORMAL -> SWAPPED
```

Four-seed raw greedy accuracy:

| condition | learn swapped | reverse normal | swapped again |
|---|---:|---:|---:|
| one collapsed additive sheet | 1.000 | 0.375 | 1.000 |
| two cue-addressed slow bands | **1.000** | **1.000** | **1.000** |
| two bands + committed-route retirement | **1.000** | **1.000** | **1.000** |

Late reward, POKE and write activity also return to 1.0 / 0 / 0 in every separated phase.

The important diagnostic is that **retirement becomes unnecessary once the two contexts stop overwriting the same slow sheet**. S14's apparent forgetting problem was therefore secondary to representational interference in this assay.

S15 is deliberately an upper-bound control. `cue -> band` is supplied and must not be advertised as a learned address mechanism.

## S16 — remove the cue->band table

S16 starts two **exact-copy anonymous bands**. Both see every cue. There is no stored cue-to-band mapping. Each band proposes its strongest launcher and a generic confidence `|C-D|`; the most confident band gets the committed output. Failure transiently reopens random POKEs; successful traffic writes only the band that was actually tried.

Four-seed result:

| condition | raw SWAPPED | raw NORMAL | raw SWAPPED again | noisy SWAPPED again |
|---|---:|---:|---:|---:|
| one band | 1.000 | 0.375 | 1.000 | 0.513 phase mean |
| two anonymous bands, bound write | 1.000 | 0.375 | 1.000 | **0.823** |
| two anonymous bands, scrambled write address | 1.000 | 0.500 | 1.000 | 0.788 |

The noise test uses launcher noise sigma `0.001`, 4,000 samples per cue.

### What did emerge

After the first SWAPPED phase, and again after returning to SWAPPED, **all four seeds assigned the two cues to different anonymous bands**. Final assignments were:

```text
seed 0   cue bands [0, 1]
seed 1   cue bands [0, 1]
seed 2   cue bands [1, 0]
seed 3   cue bands [0, 1]
```

The arbitrary label reversal in seed 2 is desirable evidence that band identity was not semantically planted. The slow material itself broke the symmetry and carried the assignment after transient alarm state was erased.

This specialization improves final noisy robustness dramatically over the collapsed sheet, even though formal argmax accuracy is 1.0 in both.

### What did not emerge

The anonymous-band system **still does not reorganize successfully during the middle NORMAL reversal**. Raw accuracy remains 0.375 and noisy accuracy about 0.350 in the bound-write condition. The emergent partition is useful but sticky.

The scrambled-write attacker is also not a clean kill: it degrades final noisy accuracy only from roughly 0.823 to 0.788. Therefore S16 is **not a full pass** for address binding.

The honest S16 sentence is:

> **Two anonymous slow bands can spontaneously specialize to different cue histories without a cue->band table and thereby create a much more robust embodied mapping, but the current generic confidence router does not reassign those bands when outward meaning reverses.**

## S16X — measure the interference instead of naming it

To make the wall mathematical, define each cue's signed correct-launcher margin `m_i`. After an oracle-balanced two-route solution, make exactly one additional rewarded material write for cue `j` and measure every margin again:

```text
C[i,j] = delta m_i after write j
```

Across 12 seeds, the shared sheet gives:

```text
mean margin before write = 8.589e-6

mean cross-write matrix

        write cue0       write cue1
cue0   +4.609e-6        -4.627e-6
cue1   -4.624e-6        +4.611e-6
```

So one local write helps its own context and damages the other by essentially the **same magnitude**:

```text
mean |off diagonal| / mean |diagonal| = 1.0034
```

This is the clearest witness of the current failure so far. The two contexts are almost a zero-sum direction in the material coordinates. A single extra write is about half the size of the already tiny shared-sheet decision margin.

For explicitly separated bands:

```text
mean pre-write margin = 0.65799
mean cross-band off diagonal = 0
```

The self-effect of one additional write at that already saturated state is slightly negative (`~ -0.001995`), so it should not be interpreted as a generic Hebbian derivative. The useful comparison is the collateral term: **shared material has one-for-one cross-context coupling; separated material has none by construction.**

## Current interpretation

The recurring problem can now be written without neuron vocabulary:

```text
consequence says:        something was wrong
eligibility says:        here is what was recently active
shared material says:    changing here also changes another task
```

The missing property is not merely more error signal. It is a read/write coordinate system in which useful corrections have sufficiently small harmful off-diagonal effects.

That suggests a general diagnostic for this whole repo family:

> **addressability = how close the consequence-to-write / write-to-read operator is to a useful block structure.**

The next work should therefore stop asking whether explicit addresses help. They do. The frontier is whether interference can cause a system to **create or reorganize** those protected directions itself.

## Next clean gate

Two possibilities remain distinct:

1. **S16R — reassign existing anonymous bands.** On a failed *committed* band, transiently suppress only that band's confidence so another anonymous band can compete. Erase this suppression at evaluation. If the final mapping survives, assignment can move without a cue->band table.
2. **S17 — create the address space itself.** Begin with one undifferentiated medium containing a local degree of freedom such as orientation / channel / phase preference. Ask whether repeated interference makes different histories carve distinct persistent transport directions. No pre-separated bands.

S17 is the more important scientific frontier. S16R is the cleaner mechanistic bridge.
