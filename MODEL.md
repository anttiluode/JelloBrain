# Model and gate contract

JelloBrain is a computational toy built from the line that kept recurring in the earlier repos:

> **structure compiles an operator.**

Here the structure is literal directed material on a 2-D lattice.

## State

The fast state is excitation `x` plus a refractory field `r`. A medium eligibility trace `q` remembers recent local activation long enough for ordered activity to write the slow state. The slow state is four directed edge fields:

- `M_E`: left cell → right cell
- `M_W`: right cell → left cell
- `M_S`: upper cell → lower cell
- `M_N`: lower cell → upper cell

The effective directed conductivity is

```text
G_dir = G_base + gain * M_dir
```

and a step computes a local weighted neighbour drive, subtracts refractory pressure, and passes it through a bounded nonlinearity.

A schematic update is

```text
x[t+1] = relu(tanh(decay*x[t] + propagation*L_M(x[t]) - beta*r[t]))
```

where `L_M` is the nearest-neighbour propagation operator compiled by the current material.

## Slow writing

Material is not updated from a word label. It is updated locally from a source eligibility trace and current activity at the adjacent target:

```text
M_(i->j)[t+1] = clip(M_(i->j)[t] + eta * q_i[t] * x_j[t+1])
q[t+1]        = rho*q[t] + (1-rho)*x[t]
```

This makes the material directional and order-sensitive. `A→B` can leave a different trace from `B→A` even when the primitive counts are identical.

## Artificial language

There are six meaningless primitives `A..F`. Each is only a short spatial pulse pattern entering the same dynamical substrate. A pseudoword such as `ABC` is the sequence `A`, then `B`, then `C`.

There is deliberately no:

- embedding;
- tokenizer;
- word ID stored in the gel;
- semantic vector;
- pretrained language model;
- lookup table from a pseudoword to an internal state.

If order becomes recoverable after fast state is wiped, it has to be present in the slow material and in how that material changes future dynamics.

## Gate contract

**W0 — persistent word trace.** Grow on two corpora, wipe all fast variables, then use a neutral A–F probe battery. A decoder is trained on some world seeds and evaluated on unseen seeds.

**W1 — order, not frequency.** Both corpora contain exactly the same counts of A/B/C/D. Only their ordering differs. Above-chance held-out decoding means the slow substrate contains temporal organization beyond a bag of primitives.

**W2 — geometry matters.** Preserve every slow material value but randomly relocate values across position and direction. The decoder is trained only on intact worlds. If intact decoding remains high while shuffled-material decoding falls to chance, *where* the material lives matters, not only its histogram.

**W3 — context.** Freeze slow material. Present `AB → E` or `CD → E`, with the same target `E` in both conditions. Decode context from the response to the target.

**W4 — computation.** Freeze slow material and present two bits as primitive choices. The target is XOR. A linear readout sees the nonlinear gel state; an attacker gets the two raw bits but is restricted to a single linear boundary. A linear separator can score at most 0.75 on the four XOR corners. The current gel readout scores 1.0 on held-out world seeds.

**W5 — shared vs individual code.** Increase initial material heterogeneity strongly, grow 20 independent worlds, train on 10 and test on 10. The four classes (`ABC`, `ACB`, `BAC`, `BCA`) use the same primitive counts and differ only in order.

## Claim boundary

Passing these gates supports statements about **this model**: persistent history, order sensitivity, geometry-dependent response, context dependence, useful nonlinear temporal transformation, and some cross-world response regularity.

It does **not** establish that the model thinks, understands words, is conscious, is a biological neuron, or explains speaker–listener neural coupling.
