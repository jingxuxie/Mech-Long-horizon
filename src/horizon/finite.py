"""Exact finite-state auditing. State equality and outputs must be exact."""
from collections import deque


def audit(initials, actions, transition, output, keep, horizon=None, max_pairs=100000):
    """Breadth-first state-pair search with a persistent zero mask.

    Returns a shortest falsifying sequence, a finite-horizon certificate,
    an all-horizon closure certificate, or unresolved on budget exhaustion.
    Protocol memory, when needed, must be included as unmasked state.
    """
    if horizon is not None and horizon < 0:
        raise ValueError('Negative horizon.')
    if max_pairs < 1:
        raise ValueError('max_pairs must be positive.')
    def mask(x):
        return tuple(v if i in keep else 0 for i, v in enumerate(x))
    queue, seen = deque(), set()
    for x in initials:
        pair = (tuple(x), mask(x))
        if pair not in seen:
            seen.add(pair)
            queue.append((pair, ()))
    evaluated, transitions, truncated = 0, 0, False
    while queue:
        if evaluated >= max_pairs:
            return dict(status='unresolved', pairs=evaluated, transitions=transitions)
        (x,z), word = queue.popleft()
        evaluated += 1
        if output(x) != output(z):
            return dict(status='falsified', time=len(word), word=list(word),
                        original=x, reduced=z, pairs=evaluated, transitions=transitions)
        if horizon is not None and len(word) == horizon:
            truncated = True
            continue
        for a in actions:
            pair = (tuple(transition(x,a)), mask(transition(z,a)))
            transitions += 1
            if pair not in seen:
                seen.add(pair)
                queue.append((pair, word+(a,)))
    return dict(status='certified_horizon' if truncated else 'certified_all',
                horizon=horizon if truncated else None, pairs=evaluated, transitions=transitions)
