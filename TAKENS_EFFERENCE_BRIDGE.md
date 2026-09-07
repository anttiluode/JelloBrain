# Takens, efference copy, and the JelloBrain control loop

This note exists to prevent another attractive conflation.

## The connection is real, but the objects are different

Classical Takens delay embedding addresses a reconstruction problem.  From delayed
samples of an observable,

```text
z_t = [y_t, y_(t-tau), y_(t-2tau), ...]
```

one may, under the theorem's assumptions and with a sufficiently large generic
embedding, obtain coordinates for the hidden dynamical state.

Efference copy addresses a prediction / attribution problem.  The system retains
information about its own intervention `u_t`, predicts the sensory consequence of
that intervention, and compares prediction with the actual return:

```text
predicted self-effect = F(hidden-state estimate, u_t)
innovation             = observed effect - predicted self-effect
```

The useful synthesis for JelloBrain is therefore not `Takens = efference copy`.
It is:

```text
DELAY HISTORY        -> estimate the hidden state relevant to prediction
EFFERENCE COPY       -> identify the forcing / action that the system itself issued
FORWARD MODEL        -> predict the self-generated return
ACTUAL RETURN        -> observe what happened
INNOVATION / DELTA   -> retain what the self-model did not explain
```

For a driven system the natural mathematical object is an **action-conditioned
(or forced) delay embedding**, not the autonomous Takens theorem in its simplest
form.

## Why this connects to the old GeometricNeuron line

`GeometricNeuronV24` already made the important distinction that one scalar pulse
is not enough: the evidence is the history of **address plus pulse**.  Gate 3 then
showed that an addressed perturbation can make a hidden operator observable.
Gate 6C compared a same-field local writer with a deliberately boring
`HISTORY_REPLAY` attacker that stores the measurement history and recomputes a
consistent estimate.

That leaves a clean question between the two extremes:

> Can a finite delayed history of scalar observations, together with a copy of
> the interventions that produced them, predict the self-generated change well
> enough that the residual exposes unmodelled change?

That is the purpose of `takens_efference_probe.py`.

## The dendritic-cable intuition

A branched cable can provide many differently filtered and differently delayed
versions of a source signal.  Computationally, such a bank can serve as a temporal
basis from which a downstream learner constructs a predictor.  Calling the cable
itself a literal implementation of Takens' theorem would be too strong: real
dendrites are distributed filters, not ideal delay taps, and the theorem has
specific assumptions.

The useful engineering analogy is narrower:

```text
one outgoing / reference signal
        -> many path-dependent temporal transforms
        -> weighted combination
        -> prediction of expected return
```

This is close to adaptive-filter architectures used in cerebellar and
cerebellum-like modelling.

## The AIS grating boundary

The ~190 nm actin/spectrin periodicity at the axon initial segment is a real
structural periodicity, but it is the wrong scale for a millisecond delay bank.
Even at a very slow 0.1 m/s propagation speed, 190 nm corresponds to about
1.9 microseconds; at 1 m/s it is about 0.19 microseconds.  Current evidence
supports mechanical organization, membrane-protein / channel positioning,
polarity and excitability roles for the periodic scaffold.  A signal-processing
`diffraction grating` or Takens delay-line interpretation is not established.

The AIS remains relevant to this project for a different reason: it is an
output boundary whose excitability and structure can be modified.  That makes it
a plausible analogy for **whether / how a travelling event is launched**, while
the predictive negative image belongs conceptually to another computation.

## Claim boundary

The next experiment is a mathematical toy.  If a forced delay history improves
prediction and shock detection, it shows that **history + efference information is
useful for reconstructing enough hidden state to cancel predictable self-effects
in this substrate**.  It does not establish that a dendrite implements Takens,
that the AIS periodic scaffold is a delay grating, or that the brain uses the
specific estimator tested here.
