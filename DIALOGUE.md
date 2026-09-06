# Phase II — two private gels and a narrow channel

The first JelloBrain experiment asks whether meaningless pulse sequences can leave slow, order-sensitive structure and whether the resulting substrate can compute. The next experiment should **not** join two gels into one state. It should preserve the boundary.

```text
private gel A  -> tiny pulse channel -> private gel B
private gel A  <- tiny pulse channel <- private gel B
```

Neither side can inspect the other's material. Each has a private hidden cue and the pair receives reward only when their joint action is correct. No symbol is assigned a meaning by the experimenter. If a stable code appears, it must be useful because of the consequences of interaction.

## Why the Stephens–Silbert–Hasson paper matters

Stephens, Silbert & Hasson (PNAS, 2010, doi:10.1073/pnas.1008662107) measured a speaker telling an unrehearsed story and listeners hearing it. They reported widespread spatial and temporal speaker–listener coupling during successful communication. The coupling largely disappeared when communication was broken by an unintelligible Russian story or by pairing the listener with a different English story. On average the listener response lagged the speaker, while some frontal/striatal regions showed anticipatory activity; the extent of anticipatory coupling was associated with better comprehension.

That paper is **inspiration for tests, not validation of this toy**. Its signal is fMRI/BOLD, its temporal precision is limited, and it does not claim a gelatin substrate or pulse vocabulary.

## Dialogue gates

| Gate | Question | Strong outcome |
|---|---|---|
| **D0 — communication-specific coupling** | Do A and B become more dynamically coupled only when the channel carries task-useful interaction? | intact cooperative pair > shuffled-channel / foreign-code controls |
| **D1 — temporal asymmetry** | Does B first lag A, then develop predictive activity as repeated interaction makes the task predictable? | learned lead/lag structure changes with competence |
| **D2 — coupling tracks success** | Across independently seeded pairs, does useful coupling predict held-out task reward? | coupling–reward relation survives seed variation |
| **D3 — repair** | If the channel mapping is perturbed, can the pair renegotiate a working pulse vocabulary? | recovery beats frozen-code baseline |
| **D4 — private code vs shared relation** | Are pulse identities idiosyncratic while relational structure generalizes across pairs? | within-pair code differs, cross-pair relational decoder survives |

## Attackers

A dialogue demo is easy to fake accidentally. The following controls are mandatory:

- randomize pulse labels after training;
- replay one partner's old messages to a new partner;
- keep message frequencies but shuffle their temporal order;
- give one side the other side's private cue directly (upper bound / leakage check);
- freeze material on A, B, or both;
- compare an adaptive channel with a fixed hand-coded vocabulary;
- report task reward, not visual synchrony, as the primary outcome.

A pretty synchronized animation is not communication. **The pulses must change what the pair can jointly do.**
