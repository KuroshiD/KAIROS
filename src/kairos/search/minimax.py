"""Minimax com poda alfa-beta sobre a arvore pronta: U = v(folha) + lambda * soma log pi(lances proprios)."""
import random

from kairos.search.common import INF
from kairos.search.policy import node_logpi


class Ctx:
    def __init__(self, lam=0.0, pol=None, T=1.0, renorm=True, prune=False, order="policy", seed=0):
        self.lam, self.pol, self.T, self.renorm = lam, pol, T, renorm
        self.prune, self.order = prune, order
        self.n_own = self.n_opp = self.n_leaf = 0
        self.rng = random.Random(seed)


def logterms(node, ctx):
    if ctx.pol is None or ctx.lam == 0:
        return [0.0] * len(node.cands)
    return (ctx.lam * node_logpi(node, ctx.pol, ctx.T, ctx.renorm)).tolist()


def order_own(node, lt, ctx):
    idx = list(range(len(node.cands)))
    if ctx.order == "policy":
        idx.sort(key=lambda i: -(lt[i] + node.cands[i].cp))
    elif ctx.order == "engine":
        idx.sort(key=lambda i: -node.cands[i].cp)
    elif ctx.order == "random":
        ctx.rng.shuffle(idx)
    return idx


def _cand_value(c, t, rem, ctx, alpha, beta):
    if c.terminal is not None:
        ctx.n_leaf += 1
        return t + c.terminal
    if rem <= 1 or c.opp is None:
        ctx.n_leaf += 1
        return t + c.cp
    return t + solve_opp(c.opp, rem - 1, ctx, alpha - t, beta - t)


def solve_own(node, rem, ctx, alpha, beta):
    ctx.n_own += 1
    lt = logterms(node, ctx)
    best = -INF
    for i in order_own(node, lt, ctx):
        val = _cand_value(node.cands[i], lt[i], rem, ctx, alpha, beta)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if ctx.prune and alpha >= beta:
            break
    return best


def solve_opp(op, rem, ctx, alpha, beta):
    ctx.n_opp += 1
    best = INF
    reps = list(op.replies)
    if ctx.order == "random":
        ctx.rng.shuffle(reps)
    for r in reps:
        val = r.terminal if r.terminal is not None else solve_own(r.own, rem, ctx, alpha, beta)
        if val < best:
            best = val
        if best < beta:
            beta = best
        if ctx.prune and alpha >= beta:
            break
    return best


def root_values(node, rem, ctx):
    """Q(a) para cada candidato da raiz. Exato quando prune=False; com poda, os candidatos
    descartados recebem apenas um limite superior (o melhor valor continua exato)."""
    lt = logterms(node, ctx)
    Q = [None] * len(node.cands)
    alpha = -INF
    ctx.n_own += 1
    for i in order_own(node, lt, ctx):
        val = _cand_value(node.cands[i], lt[i], rem, ctx, alpha if ctx.prune else -INF, INF)
        if ctx.prune and val <= alpha:
            val = val - 1e-9        # podado: so ha um limite superior; nunca empata com o melhor
        Q[i] = val
        if val > alpha:
            alpha = val
    return Q


def argbest(Q, cands):
    return max(range(len(Q)), key=lambda i: (Q[i], cands[i].cp, -i))
