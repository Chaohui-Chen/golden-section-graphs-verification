#!/usr/bin/env python3
"""Exact verification of finite computations in the accompanying paper.

Paper: A complete characterization of graphs with second largest eigenvalue
at most (sqrt(5)-1)/2.

Run: python verify.py
Requirements: Python >= 3.10 and NetworkX >= 3.0, < 4.0.

Correspondence with the paper:
  Lemma 2.5: the seven exceptional graphs in H_plus (the finite part);
  Lemma 3.1: the H_plus classification through order seven;
  Lemma 3.2: the specified forbidden induced subgraphs;
  Proposition 3.6: the family F of minimal forbidden induced subgraphs for
    lambda_min >= -phi that also satisfy lambda_2(complement) <= sigma.

The W(t,j) formulas for arbitrary parameters are proved in the paper.
All spectral decisions use rational arithmetic in Q(sqrt(5)); no floating-
point eigensolver or numerical tolerance is used. The arbitrary-order
structural assertions are proved in the paper, not by this script.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import networkx as nx


@dataclass(frozen=True, slots=True)
class Q5:
    """Exact a+b*sqrt(5), with a,b rational."""

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)

    def __add__(self, other):
        other = q(other)
        return Q5(self.a + other.a, self.b + other.b)

    __radd__ = __add__

    def __neg__(self):
        return Q5(-self.a, -self.b)

    def __sub__(self, other):
        return self + (-q(other))

    def __rsub__(self, other):
        return q(other) - self

    def __mul__(self, other):
        other = q(other)
        return Q5(self.a * other.a + 5 * self.b * other.b,
                  self.a * other.b + self.b * other.a)

    __rmul__ = __mul__

    def inverse(self):
        den = self.a * self.a - 5 * self.b * self.b
        if den == 0:
            raise ZeroDivisionError('division by zero in Q(sqrt(5))')
        return Q5(self.a / den, -self.b / den)

    def __truediv__(self, other):
        return self * q(other).inverse()

    def __rtruediv__(self, other):
        return q(other) * self.inverse()

    def sign(self) -> int:
        """Exact sign via rational comparison of a^2 and 5*b^2."""
        a, b = self.a, self.b
        if b == 0:
            return (a > 0) - (a < 0)
        if a == 0:
            return (b > 0) - (b < 0)
        if (a > 0) == (b > 0):
            return 1 if a > 0 else -1
        # a and b have opposite signs: compare |a| with |b*sqrt(5)|.
        aa, bb = a * a, 5 * b * b
        if aa == bb:
            raise AssertionError('sqrt(5) is irrational')
        return ((aa > bb) == (a > 0)) * 2 - 1

    def __str__(self):
        if self.b == 0:
            return str(self.a)
        return f'({self.a})+({self.b})*sqrt(5)'


def q(value) -> Q5:
    if isinstance(value, Q5):
        return value
    return Q5(Fraction(value))


ZERO, ONE = q(0), q(1)
PHI = Q5(Fraction(1, 2), Fraction(1, 2))
SIGMA = Q5(Fraction(-1, 2), Fraction(1, 2))


def inertia(matrix):
    """(positive, zero, negative) inertia by exact symmetric congruence.

    Use 1x1 pivots when a nonzero diagonal exists. If all diagonals are
    zero, a nonzero 2x2 off-diagonal pivot has inertia (1,0,1).
    This handles singular matrices without numerical rank decisions.
    """
    a = [list(map(q, row)) for row in matrix]
    pos = neg = zero = 0
    while a:
        n = len(a)
        k = next((i for i in range(n) if a[i][i] != ZERO), None)
        if k is not None:
            rem = [i for i in range(n) if i != k]
            d = a[k][k]
            if d.sign() > 0:
                pos += 1
            else:
                neg += 1
            a = [[a[i][j] - a[i][k] * a[k][j] / d for j in rem]
                 for i in rem]
            continue
        pair = next(((i, j) for i in range(n) for j in range(i + 1, n)
                     if a[i][j] != ZERO), None)
        if pair is None:
            zero += n
            break
        i, j = pair
        rem = [k for k in range(n) if k != i and k != j]
        d = a[i][j]
        pos += 1
        neg += 1
        a = [[a[k][l] - (a[k][i] * a[j][l] + a[k][j] * a[i][l]) / d
              for l in rem] for k in rem]
    return pos, zero, neg


def shifted_matrix(g, rank_one=False):
    """A(g)+phi*I or A(g)+phi*I-J, with an integer vertex ordering."""
    vertices = list(g)
    n = len(vertices)
    result = [[PHI if i == j else q(int(g.has_edge(vertices[i], vertices[j])))
               for j in range(n)] for i in range(n)]
    if rank_one:
        return [[z - ONE for z in row] for row in result]
    return result


def least_ge_minus_phi(g):
    return inertia(shifted_matrix(g))[2] == 0


def complement_lambda2_le_sigma(g):
    """For order >= 2: lambda_2(complement(g)) <= sigma iff n_-(M-J) <= 1."""
    return inertia(shifted_matrix(g, rank_one=True))[2] <= 1


def complement_lambda2_gt_sigma(g):
    return inertia(shifted_matrix(g, rank_one=True))[2] >= 2


def solve_ones(matrix):
    """Exact solve M*x=1 (some free variables set to zero).

    Return None iff 1 is not in range(M); otherwise return one solution.
    """
    n = len(matrix)
    aug = [list(map(q, row)) + [ONE] for row in matrix]
    pivots = []
    row = 0
    for col in range(n):
        pivot = next((i for i in range(row, n) if aug[i][col] != ZERO), None)
        if pivot is None:
            continue
        aug[row], aug[pivot] = aug[pivot], aug[row]
        factor = aug[row][col]
        aug[row] = [v / factor for v in aug[row]]
        for i in range(n):
            if i != row and aug[i][col] != ZERO:
                factor = aug[i][col]
                aug[i] = [v - factor * w for v, w in zip(aug[i], aug[row])]
        pivots.append(col)
        row += 1
    if any(aug[i][-1] != ZERO for i in range(row, n)):
        return None
    result = [ZERO] * n
    for i, col in enumerate(pivots):
        result[col] = aug[i][-1]
    assert all(sum((q(matrix[i][j]) * result[j] for j in range(n)), ZERO) == ONE
               for i in range(n))
    return result


def shifted_coronal(g):
    """1^T (A+phi I)^dagger 1 when 1 belongs to the range."""
    sol = solve_ones(shifted_matrix(g))
    if sol is None:
        raise AssertionError('1 is not in range(A+phi*I)')
    return sum(sol, ZERO)


def union(*graphs):
    result = nx.Graph()
    for g in graphs:
        result = nx.disjoint_union(result, g)
    return result


def join(a, b):
    result = union(a, b)
    na = len(a)
    result.add_edges_from((i, j) for i in range(na) for j in range(na, na + len(b)))
    return result


def empty():
    return nx.empty_graph(0)


def clique(n):
    return nx.complete_graph(n)


def isolated(n):
    return nx.empty_graph(n)


def W(t, j):
    assert 1 <= t and 0 <= j <= t
    result = clique(t)
    result.add_edges_from((i, t + i) for i in range(j))
    return result


# The seven exceptional graphs in H_plus (Lemma 2.5).
def positive_exceptions():
    return {
        'K1 join P3': join(clique(1), nx.path_graph(3)),
        'K1 join P4': join(clique(1), nx.path_graph(4)),
        'K1 join C5': join(clique(1), nx.cycle_graph(5)),
        'C5': nx.cycle_graph(5),
        'complement W(3,3)': nx.complement(W(3, 3)),
        'K1 join (K2 union K2)': join(clique(1), union(clique(2), clique(2))),
        'K1 join (K2 union K3)': join(clique(1), union(clique(2), clique(3))),
    }


# The graphs S1,...,S5 defining the family F in Proposition 3.6.
# They are minimal forbidden induced subgraphs for lambda_min >= -phi
# whose complements also satisfy lambda_2 <= sigma (not all minimal
# forbidden induced subgraphs for the least-eigenvalue bound).
def minimal_forbidden_graphs():
    return {
        'S1': nx.star_graph(3),
        'S2': join(clique(2), union(clique(1), clique(2))),
        'S3': join(clique(3), union(clique(1), clique(1))),
        'S4': join(clique(1), union(clique(2), clique(4))),
        'S5': join(clique(1), union(clique(3), clique(3))),
    }


# The four fixed forbidden graphs F1,...,F4 (Lemma 3.2).
def forbidden_graphs():
    # F1 = triangle, with a path of length two attached at vertex 2.
    f1 = clique(3)
    f1.add_edges_from([(2, 3), (3, 4)])
    # F2 = claw, with one edge subdivided once.
    f2 = nx.Graph([(0, 1), (0, 2), (0, 4), (4, 3)])
    # F3 = isolated K1 plus K2 join 3K1.
    f3 = union(isolated(1), join(clique(2), isolated(3)))
    # F4 = diamond K4-e, with a leaf attached to a degree-2 vertex.
    f4 = clique(4)
    f4.remove_edge(0, 1)
    f4.add_edge(0, 4)
    return {'F1': f1, 'F2': f2, 'F3': f3, 'F4': f4}


def same(g, h):
    return len(g) == len(h) and g.number_of_edges() == h.number_of_edges() and nx.is_isomorphic(g, h)


# Lemma 2.5: verify positive semidefiniteness, the range condition,
# and the stated shifted coronals for the seven exceptional graphs.
# The W(t,j) assertions are proved for all parameters in the paper.
def check_positive_exceptions():
    print('[Lemma 2.5 / seven exceptional graphs in H_plus]')
    r = positive_exceptions()
    values = {
        'K1 join P3': q(2), 'K1 join P4': q(2),
        'K1 join C5': q(2), 'C5': q(2) - SIGMA,
        'complement W(3,3)': q(2),
        'K1 join (K2 union K2)': q(4) + SIGMA,
        'K1 join (K2 union K3)': q(12) + 6 * SIGMA,
    }
    for name, g in r.items():
        assert least_ge_minus_phi(g), name
        assert shifted_coronal(g) == values[name], name
        print(f'  PASS  {name:27s}  eta={values[name]}')


# Graph Atlas input for Lemma 3.1 and Proposition 3.6 (orders <= 7).
def enumerate_graphs():
    atlas = nx.graph_atlas_g()  # exactly one graph per unlabeled graph of order <= 7
    return [g for g in atlas if len(g) and nx.is_connected(g)]


# Lemma 3.1: the H_plus classification of connected graphs through
# order seven, including the order-seven assertion.
def check_positive_classification(atlas):
    print('[Lemma 3.1 / connected graphs with least eigenvalue >= -phi]')
    candidates = []
    for t in range(1, 8):
        for j in range(t + 1):
            if t + j <= 7:
                candidates.append(W(t, j))
    candidates += list(positive_exceptions().values())
    count = Counter()
    for g in atlas:
        spectral = least_ge_minus_phi(g)
        classified = any(same(g, h) for h in candidates)
        assert spectral == classified, f'H_plus classification mismatch: {nx.to_graph6_bytes(g, header=False)}'
        if spectral:
            count[len(g)] += 1
    print('  PASS  all connected unlabeled graphs through order 7')
    print('  PSD counts by order:', dict(sorted(count.items())))
    # Explicitly check none of the seven exceptional shapes has order 7.
    assert all(len(g) <= 6 for g in positive_exceptions().values())


# Lemma 3.2(i)-(ii): exact least-eigenvalue and complement-lambda_2
# tests for S1,...,S5 and the fixed forbidden induced subgraphs.
def check_fixed_forbidden():
    print('[Lemma 3.2 / fixed forbidden induced subgraphs]')
    S = minimal_forbidden_graphs()
    F = forbidden_graphs()
    for name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        assert not least_ge_minus_phi(S[name]), name
    assert not least_ge_minus_phi(nx.cycle_graph(4))
    assert not least_ge_minus_phi(F['F1'])
    print('  PASS  min eigenvalue < -phi for S1..S5, C4, F1')
    for name, g in [('C4', nx.cycle_graph(4)), *F.items()]:
        assert complement_lambda2_gt_sigma(g), name
    print('  PASS  lambda2(complement) > sigma for C4, F1..F4')
    for i, j in product(S, repeat=2):
        assert complement_lambda2_gt_sigma(union(S[i], S[j])), (i, j)
    print('  PASS  lambda2(complement(Si union Sj)) > sigma for all 25 ordered pairs')
    for name, g in S.items():
        # Sp*(S,1) = (K1 join S) union K1.
        bad = union(join(clique(1), g), isolated(1))
        assert complement_lambda2_gt_sigma(bad), name
    print('  PASS  lambda2(complement(Sp*(Si,1))) > sigma for i=1..5')


# Proposition 3.6: determine the family F through order seven using
# minimality, the least-eigenvalue threshold, and the complement bound.
# Lemma 3.5 supplies the order bound; it is proved in the paper.
def check_minimal_forbidden_classification(atlas):
    print('[Proposition 3.6 / the family F of minimal forbidden induced subgraphs]')
    S = minimal_forbidden_graphs()
    identified = []
    for g in atlas:
        if least_ge_minus_phi(g) or not complement_lambda2_le_sigma(g):
            continue
        # Deletion minimality implies all proper induced subgraphs are PSD.
        if any(not least_ge_minus_phi(nx.convert_node_labels_to_integers(
                nx.subgraph(g, [u for u in g if u != v]))) for v in g):
            continue
        matches = [name for name, s in S.items() if same(g, s)]
        assert len(matches) == 1, f'unexpected member of F: {nx.to_graph6_bytes(g, header=False)}'
        identified.append(matches[0])
    assert sorted(identified) == sorted(S), f'family F classification incomplete: {identified}'
    print('  PASS  F consists exactly of:', ', '.join(sorted(identified)))
    # Check the three defining conditions directly for each S_i.
    for name, s in S.items():
        assert nx.is_connected(s)
        assert not least_ge_minus_phi(s)
        assert complement_lambda2_le_sigma(s)
        for v in s:
            h = s.copy()
            h.remove_node(v)
            assert least_ge_minus_phi(h), (name, v)


def main():
    if not __debug__:
        raise RuntimeError("Run with plain python, not python -O: assertions must be enabled.")
    assert PHI * PHI == PHI + ONE
    assert SIGMA * PHI == ONE
    atlas = enumerate_graphs()
    counts = Counter(map(len, atlas))
    print('NetworkX version:', nx.__version__)
    print('Connected unlabeled Graph Atlas counts:', dict(sorted(counts.items())))
    assert [counts[n] for n in range(1, 8)] == [1, 1, 2, 6, 21, 112, 853]
    check_positive_exceptions()
    check_positive_classification(atlas)
    check_fixed_forbidden()
    check_minimal_forbidden_classification(atlas)
    print('ALL CHECKS PASSED (exact arithmetic in Q(sqrt(5))).')


if __name__ == '__main__':
    main()
