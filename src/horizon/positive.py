"""Exact support certificates for nonnegative switched linear systems.

Convention: x[0] has independent coordinates in [0,1] on `sources` and
zero elsewhere; y[t] = C x[t]; x[t+1] = A[a[t]] x[t]. A persistent mask
initializes z[0] = P x[0] and uses z[t+1] = P A[a[t]] z[t]. Outputs at
all prefix times 0,...,H are compared. The protocol is prefix-closed.
"""
from collections import deque
from dataclasses import dataclass
from itertools import product
import math
import numpy as np


@dataclass
class System:
    matrices: tuple
    readout: np.ndarray
    sources: tuple
    protocol: dict  # (protocol state, symbol) -> next protocol state
    start: int = 0

    def __post_init__(self):
        self.matrices = tuple(np.asarray(a) for a in self.matrices)
        self.readout = np.atleast_2d(np.asarray(self.readout))
        if not self.matrices:
            raise ValueError('At least one transition matrix is required.')
        n = self.matrices[0].shape[0]
        if any(a.shape != (n, n) or not np.isfinite(a).all() or (a < 0).any()
               for a in self.matrices):
            raise ValueError('Matrices must be finite, square and nonnegative.')
        if self.readout.shape[1] != n or (self.readout < 0).any() or not np.isfinite(self.readout).all():
            raise ValueError('Readout must be finite and nonnegative.')
        if any(not 0 <= s < n for s in self.sources):
            raise ValueError('Invalid source.')
        if any(not 0 <= a < len(self.matrices) for q, a in self.protocol):
            raise ValueError('Invalid protocol symbol.')

    @property
    def n(self):
        return self.matrices[0].shape[0]


def unrestricted(symbols):
    return {(0, a): 0 for a in range(symbols)}


def no_consecutive_ones():
    return {(0, 0): 0, (0, 1): 1, (1, 0): 0}


def _bfs(adjacency, roots):
    distance = {v: 0 for v in roots}
    parent = {v: None for v in roots}
    todo = deque(roots)
    while todo:
        u = todo.popleft()
        for v, label in adjacency.get(u, ()):
            if v not in distance:
                distance[v] = distance[u] + 1
                parent[v] = (u, label)
                todo.append(v)
    return distance, parent


def support_certificate(system):
    """Return all birth horizons, a product graph and shortest-path records.

    Exact combinatorial result: a mask K is faithful through H iff K
    contains all v with birth[v] <= H. No floating tolerance is used to
    decide whether a supplied coefficient is positive.
    """
    states = {system.start}
    for (q, a), r in system.protocol.items():
        states.update((q, r))
    forward = {(i, q): [] for i in range(system.n) for q in states}
    reverse = {v: [] for v in forward}
    supports = [list(zip(*np.nonzero(matrix > 0))) for matrix in system.matrices]
    for (q, a), r in system.protocol.items():
        for i, j in supports[a]:
            u, v = (int(j), q), (int(i), r)
            forward[u].append((v, a))
            reverse[v].append((u, a))
    roots = [(s, system.start) for s in system.sources]
    outputs = [(i, q) for i in range(system.n) for q in states
               if np.any(system.readout[:, i] > 0)]
    din, pin = _bfs(forward, roots)
    dout, pout = _bfs(reverse, outputs)
    birth, via = {}, {}
    for i in range(system.n):
        options = [(din.get((i, q), math.inf) + dout.get((i, q), math.inf), q)
                   for q in states]
        value, q = min(options)
        birth[i], via[i] = value, (i, q)
    return dict(birth=birth, via=via, pin=pin, pout=pout,
                vertices=len(forward), edges=sum(map(len, forward.values())))


def live_mask(certificate, horizon):
    if horizon < 0:
        raise ValueError('Horizon must be nonnegative.')
    return {i for i, birth in certificate['birth'].items() if birth <= horizon}


def witness(certificate, keep):
    """Shortest failure time and positive path for any nonfaithful mask.

    Returns None iff the mask is faithful at every finite horizon.
    The returned source basis vector and symbol word are a real witness.
    """
    omitted = [(b, i) for i, b in certificate['birth'].items()
               if i not in keep and math.isfinite(b)]
    if not omitted:
        return None
    length, i = min(omitted)
    v = certificate['via'][i]
    left_nodes, left_labels = [v], []
    u = v
    while certificate['pin'][u] is not None:
        p, a = certificate['pin'][u]
        left_nodes.append(p)
        left_labels.append(a)
        u = p
    nodes = list(reversed(left_nodes))
    labels = list(reversed(left_labels))
    u = v
    while certificate['pout'][u] is not None:
        p, a = certificate['pout'][u]
        nodes.append(p)
        labels.append(a)
        u = p
    assert len(labels) == length
    return dict(time=int(length), source=nodes[0][0],
                word=labels, path=[x[0] for x in nodes], omitted=i)


def numeric_risk(system, keep, horizon, initial=None):
    """Enumerate allowed words; exact arithmetic when inputs are integer.

    For nonnegative systems the worst [0,1]-box initial state is the
    all-ones source vector. Floats yield numerical evaluations, NOT an
    outward-rounded formal certificate. Use support_certificate for an
    exact zero-error support guarantee.
    """
    if horizon < 0:
        raise ValueError('Horizon must be nonnegative.')
    integer = all(a.dtype.kind in 'iu' for a in system.matrices) and system.readout.dtype.kind in 'iu'
    integer = integer and (initial is None or np.asarray(initial).dtype.kind in 'iu')
    dtype = object if integer else float
    x = np.zeros(system.n, dtype=dtype)
    if initial is None:
        x[list(system.sources)] = 1
    else:
        x[:] = initial
    p = np.array([int(i in keep) for i in range(system.n)], dtype=dtype)
    frontier = [(x, p*x, system.start, ())]
    worst, first, arg = 0, None, ()
    transitions = 0
    for t in range(horizon+1):
        nxt = []
        for x, z, q, word in frontier:
            error = max(abs(v) for v in (system.readout @ (x-z)))
            if error > worst:
                worst, arg = error, word
            if error > 0 and first is None:
                first = t
            if t < horizon:
                for a, matrix in enumerate(system.matrices):
                    if (q, a) in system.protocol:
                        nxt.append((matrix @ x, p*(matrix @ z), system.protocol[q, a], word+(a,)))
                        transitions += 1
        frontier = nxt
    return dict(risk=float(worst), first_failure=first, word=list(arg), transitions=transitions)


def all_masks(n):
    for bits in product((0, 1), repeat=n):
        yield {i for i, b in enumerate(bits) if b}


def chain_system(lengths, dead=0):
    n = sum(d+1 for d in lengths) + dead
    matrix = np.zeros((n, n), dtype=int)
    readout = np.zeros((len(lengths), n), dtype=int)
    sources, offset = [], 0
    for k, d in enumerate(lengths):
        sources.append(offset)
        for j in range(d):
            matrix[offset+j+1, offset+j] = 1
        readout[k, offset+d] = 1
        offset += d+1
    # Unreachable, unobservable dynamics test removal of genuinely dead units.
    for j in range(offset, n):
        matrix[j, j] = 2
    return System((matrix,), readout, tuple(sources), unrestricted(1))


def gated_system(patterns, protocol=None, dead=0):
    n = sum(len(p)+1 for p in patterns)+dead
    matrices = [np.zeros((n,n), dtype=int) for _ in range(2)]
    readout = np.zeros((len(patterns), n), dtype=int)
    sources, offset = [], 0
    for k, pattern in enumerate(patterns):
        sources.append(offset)
        for j, a in enumerate(pattern):
            matrices[a][offset+j+1, offset+j] = 1
        readout[k, offset+len(pattern)] = 1
        offset += len(pattern)+1
    return System(tuple(matrices), readout, tuple(sources),
                  unrestricted(2) if protocol is None else protocol)
