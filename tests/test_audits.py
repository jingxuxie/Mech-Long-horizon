import itertools
import numpy as np
import pytest
from horizon.positive import (System, unrestricted, no_consecutive_ones,
    support_certificate, live_mask, witness, numeric_risk, all_masks,
    chain_system, gated_system)
from horizon.finite import audit


def test_chain_birth_and_witness():
    s = chain_system([4], dead=2)
    c = support_certificate(s)
    assert live_mask(c, 3) == set()
    assert live_mask(c, 4) == set(range(5))
    w = witness(c, {0})
    assert w['time'] == 4
    assert w['path'] == list(range(5))
    assert numeric_risk(s, {0}, 4)['risk'] == 1
    assert witness(c, set(range(5))) is None


def test_zero_horizon():
    s = chain_system([0,2])
    assert live_mask(support_certificate(s), 0) == {0}
    assert numeric_risk(s, set(), 0)['first_failure'] == 0


def test_protocol_must_be_in_product():
    s = gated_system([(0,1,0), (1,1)], no_consecutive_ones(), dead=1)
    assert live_mask(support_certificate(s), 5) == {0,1,2,3}
    unrestricted_s = gated_system([(0,1,0), (1,1)], dead=1)
    assert len(live_mask(support_certificate(unrestricted_s), 5)) == 7
    assert numeric_risk(s, {0,1,2,3}, 5)['risk'] == 0


@pytest.mark.parametrize('seed', range(12))
def test_support_equals_arithmetic_on_all_masks(seed):
    rng = np.random.default_rng(seed)
    n = 3+seed%3
    mats = tuple((rng.random((n,n)) < .3).astype(int) for _ in range(2))
    s = System(mats, (rng.random((1,n)) < .4).astype(int), (0,),
               unrestricted(2) if seed%2 else no_consecutive_ones())
    c, H = support_certificate(s), 3
    required = live_mask(c,H)
    for keep in all_masks(n):
        r = numeric_risk(s,keep,H)
        assert (r['risk'] == 0) == required.issubset(keep)
        w = witness(c,keep)
        predicted = None if w is None or w['time'] > H else w['time']
        assert r['first_failure'] == predicted


def test_signed_coefficients_are_rejected():
    with pytest.raises(ValueError):
        System((np.array([[-1]]),), np.ones((1,1)), (0,), unrestricted(1))


def test_finite_shortest_witness_and_budget():
    transition = lambda x,a: (0,x[0],x[1])
    output = lambda x:x[2]
    assert audit([(1,0,0)], [0], transition, output, {0}, 1)['status'] == 'certified_horizon'
    r = audit([(1,0,0)], [0], transition, output, {0}, 8)
    assert r['status'] == 'falsified' and r['time'] == 2
    assert audit([(1,0,0)], [0], transition, output, {0}, 8, 1)['status'] == 'unresolved'
    assert audit([(1,0,0)], [0], transition, output, {0,1,2})['status'] == 'certified_all'


def test_interventions_expand_scope():
    def transition(x,a):
        return (1,x[1],x[2]) if a == 'set' else (x[0],x[0],x[1])
    natural = audit([(0,0,0)], ['tick'], transition, lambda x:x[2], set())
    expanded = audit([(0,0,0)], ['tick','set'], transition, lambda x:x[2], set())
    assert natural['status'] == 'certified_all'
    assert expanded['time'] == 3


def test_cancellation_breaks_mask_monotonicity():
    transition = lambda x,a: (0,x[0],x[0])
    output = lambda x:x[1]-x[2]
    assert audit([(1,0,0)], [0], transition, output, set())['status'] == 'certified_all'
    assert audit([(1,0,0)], [0], transition, output, {0,1})['status'] == 'falsified'


def test_nonnegative_worst_state_is_box_corner():
    s = chain_system([2,3])
    keep = {0,1,2}
    corner = numeric_risk(s,keep,4)['risk']
    for a,b in itertools.product([0,.25,.5,1], repeat=2):
        x = np.zeros(s.n)
        x[list(s.sources)] = [a,b]
        # Float initial conditions require float matrices, rather than integer casting.
        sf = System(tuple(m.astype(float) for m in s.matrices), s.readout.astype(float), s.sources, s.protocol)
        assert numeric_risk(sf,keep,4,initial=x)['risk'] <= corner


def test_float_initial_not_truncated_by_integer_system():
    s = chain_system([1])
    result = numeric_risk(s, set(), 1, initial=np.array([.25, 0.]))
    assert result['risk'] == .25

def test_worst_case_deletion_is_not_submodular():
    s = System((np.zeros((3,3)),), np.array([[1.5,0,0],[0,1,1]]),
               (0,1,2), unrestricted(1))
    def f(deleted):
        return numeric_risk(s,set(range(3))-set(deleted),0)['risk']
    assert f({0,2})-f({0}) == 0
    assert f({0,1,2})-f({0,1}) == .5

def test_no_sources_or_no_outputs_has_empty_live_set():
    a = np.ones((2,2), dtype=int)
    for sources, c in [((), np.ones((1,2), dtype=int)), ((0,), np.zeros((1,2), dtype=int))]:
        s = System((a,),c,sources,unrestricted(1))
        assert live_mask(support_certificate(s),100) == set()
        assert numeric_risk(s,set(),4)['risk'] == 0

def test_multiplicity_factor_lower_bound():
    from fractions import Fraction
    from math import factorial, prod
    for d in range(1,9):
        for a in range(1,d+1):
            alpha = [a]+([d-a] if d>a else [])
            factor = prod(Fraction(factorial(k),k**k) for k in alpha)
            assert factor >= Fraction(factorial(d),d**d)
