# Query-Clean Temporal Learning

## Problem

Temporal query state is useful inside a sequence, but harmful when it leaks across unrelated random training samples. If the model keeps target-history tokens from the previous batch, the next batch may receive an unrelated object's memory.

## Training Rule

For each sampled training item:

1. Initialize temporal query state as empty.
2. Process the sampled search frames in order.
3. Propagate detached query state only inside that sampled item.
4. Reset query state before the next independent item.

## Inference Rule

During inference, temporal query state is allowed to persist because frames belong to the same video.

## Dynamic Template Rule

The initial template is always retained. Dynamic templates are added only when:

- the response score is above a confidence threshold
- the update interval is satisfied
- the template pool has available capacity or can replace a lower-confidence template

This balances adaptation and drift resistance.
