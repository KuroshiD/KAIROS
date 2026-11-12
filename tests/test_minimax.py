"""Testes de logica do minimax (sem Stockfish, sem GPU, sem checkpoints).

Um "motor" simulado e deterministico e uma politica aleatoria substituem os reais.
Uso:  pytest
"""
import hashlib
import os
import random
import tempfile
from types import SimpleNamespace

import chess
import numpy as np

from kairos.search import common, evaluation, minimax, policy, tree

CLIP = common.CLIP


# ----------------------------------------------------------------- motor simulado
class MockAnalyzer:
    """MultiPV falso e deterministico: cp pseudo-aleatorio por (fen, lance, nos); mate em 1 vira 10000."""

    def __init__(self):
        self.calls = 0

    def multipv(self, board, k, nodes, root_moves=None):
        self.calls += 1
        out = []
        for m in board.legal_moves:
            h = int(hashlib.md5(f"{board.fen()}|{m.uci()}|{nodes}".encode()).hexdigest()[:8], 16)
            cp = (h % 401) - 250
            mate = None
            board.push(m)
            if board.is_checkmate():
                cp, mate = 10000, 1
            board.pop()
            out.append({"move": m, "pv": [m], "cp": cp, "mate": mate})
        out.sort(key=lambda c: -c["cp"])
        return out[:k]


def params(**kw):
    d = dict(k_own=4, k_opp=2, eps=120, opp_eps=200, nodes=500, root_k=8, leaf_nodes=100, opp_nodes=100,
             no_renorm=False, seed=1)
    d.update(kw)
    return SimpleNamespace(**d)


def random_positions(n, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(6, 30)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(rng.choice(ms))
        if not b.is_game_over() and b.legal_moves.count() >= 6:
            out.append(b.fen())
    return out


def build(fens, P, depth, an=None, db=None):
    db = db or tree.TreeDB(":memory:")
    prov = tree.Provider(db, an or MockAnalyzer())
    return [tree.make_own(prov, f, depth, P, root=True) for f in fens], prov, db


def policy_for(trees, temps=(1.0,), seed=0):
    need = {}
    for t in trees:
        tree.collect_need(t, need)
    return policy.RandomPolicy(seed).raw(need, list(temps))


# ----------------------------------------------------------------- testes
def test_alfa_beta_igual_minimax():
    P = params()
    trees, _, _ = build(random_positions(25, 1), P, 3)
    pol = policy_for(trees)
    for lam in (0.0, 10.0, 60.0):
        for order in ("policy", "engine", "random"):
            for node in trees:
                pure = minimax.root_values(node, 3, minimax.Ctx(lam=lam, pol=pol, prune=False))
                ab = minimax.root_values(node, 3, minimax.Ctx(lam=lam, pol=pol, prune=True, order=order, seed=3))
                assert abs(max(pure) - max(ab)) < 1e-9, (lam, order, max(pure), max(ab))
                # o candidato escolhido tem de ter valor igual ao melhor valor exato
                j = minimax.argbest(ab, node.cands)
                assert abs(pure[j] - max(pure)) < 1e-9, "alfa-beta escolheu um lance de valor menor"


def test_poda_reduz_nos():
    P = params()
    trees, _, _ = build(random_positions(25, 2), P, 3)
    pol = policy_for(trees)
    tot_pure = tot_ab = 0
    for node in trees:
        c1 = minimax.Ctx(lam=20.0, pol=pol, prune=False)
        minimax.root_values(node, 3, c1)
        c2 = minimax.Ctx(lam=20.0, pol=pol, prune=True, order="policy")
        minimax.root_values(node, 3, c2)
        tot_pure += c1.n_own + c1.n_opp
        tot_ab += c2.n_own + c2.n_opp
        assert c2.n_own + c2.n_opp <= c1.n_own + c1.n_opp
    assert tot_ab < tot_pure, "a poda nunca cortou nada"


def test_profundidade_1_e_argmax_cp_mais_lambda_logpi():
    P = params()
    trees, _, _ = build(random_positions(30, 3), P, 1)
    pol = policy_for(trees)
    for lam in (0.0, 7.0, 50.0):
        for node in trees:
            Q = minimax.root_values(node, 1, minimax.Ctx(lam=lam, pol=pol))
            lp = policy.node_logpi(node, pol, 1.0, True)
            for i, c in enumerate(node.cands):
                assert abs(Q[i] - (c.cp + lam * lp[i])) < 1e-9


def test_lambda_zero_reproduz_top1_do_motor_na_profundidade_1():
    P = params()
    trees, _, _ = build(random_positions(30, 4), P, 1)
    pol = policy_for(trees)
    for node in trees:
        Q = minimax.root_values(node, 1, minimax.Ctx(lam=0.0, pol=pol))
        j = minimax.argbest(Q, node.cands)
        top1 = max(range(len(node.cands)), key=lambda k: (node.cands[k].cp, -k))
        assert j == top1


def test_lambda_enorme_escolhe_argmax_da_politica():
    P = params()
    trees, _, _ = build(random_positions(30, 5), P, 1)
    pol = policy_for(trees)
    for node in trees:
        Q = minimax.root_values(node, 1, minimax.Ctx(lam=1e6, pol=pol))
        lp = policy.node_logpi(node, pol, 1.0, True)
        assert minimax.argbest(Q, node.cands) == int(np.argmax(lp))


def test_decisao_suave_e_prior_vezes_exp_valor_sobre_lambda():
    P = params()
    trees, _, _ = build(random_positions(20, 6), P, 1)
    pol = policy_for(trees)
    lam = 25.0
    for node in trees:
        Q = np.array(minimax.root_values(node, 1, minimax.Ctx(lam=lam, pol=pol)))
        q = common.softmax(Q / lam)
        lp = policy.node_logpi(node, pol, 1.0, True)
        cp = np.array([c.cp for c in node.cands])
        want = np.exp(lp) * np.exp(cp / lam)
        want = want / want.sum()
        assert np.allclose(q, want, atol=1e-9)


def test_renormalizacao_soma_um_e_sem_renorm_nao():
    P = params()
    trees, _, _ = build(random_positions(15, 7), P, 1)
    pol = policy_for(trees)
    some_below = False
    for node in trees:
        a = np.exp(policy.node_logpi(node, pol, 1.0, True)).sum()
        assert abs(a - 1) < 1e-9
        b = np.exp(policy.node_logpi(node, pol, 1.0, False)).sum()
        assert b <= 1 + 1e-9
        some_below |= b < 1 - 1e-6
    assert some_below, "sem renormalizar, a soma deveria ser < 1 em alguma posicao"


def test_temperatura_alta_aproxima_uniforme():
    P = params()
    trees, _, _ = build(random_positions(15, 8), P, 1)
    pol = policy_for(trees, temps=(1.0, 1000.0))
    for node in trees:
        v = np.exp(policy.node_logpi(node, pol, 1000.0, True))
        assert np.allclose(v, 1 / len(v), atol=1e-2)


def test_mate_em_um_vale_o_teto_e_vira_terminal_na_profundidade_2():
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"       # Ra8# e mate em 1
    P = params(k_own=4, eps=20000)
    trees, _, _ = build([fen], P, 2)
    node = trees[0]
    mate = [c for c in node.cands if c.uci == "a1a8"]
    assert mate, "o lance Ra8# deveria estar entre os candidatos"
    assert mate[0].terminal == CLIP and mate[0].cp == CLIP
    pol = policy_for(trees)
    Q = minimax.root_values(node, 2, minimax.Ctx(lam=0.0, pol=pol))
    assert abs(Q[node.cands.index(mate[0])] - CLIP) < 1e-9


def test_rede_de_seguranca_remove_mate_contra():
    class MateContra(MockAnalyzer):
        def multipv(self, board, k, nodes, root_moves=None):
            res = super().multipv(board, k, nodes)
            if res:
                res[0]["cp"], res[0]["mate"] = 10000, -1   # melhor linha "mate contra" (cp ruim com mate<0)
                res[0]["cp"] = -10000
            res.sort(key=lambda c: -c["cp"])
            return res
    P = params()
    fen = random_positions(1, 9)[0]
    db = tree.TreeDB(":memory:")
    prov = tree.Provider(db, MateContra())
    node = tree.make_own(prov, fen, 1, P, root=True)
    assert all(c.cp > -CLIP for c in node.cands), "candidato com mate contra deveria ter sido removido"


def test_banco_ida_e_volta_sem_chamar_o_motor():
    fens = random_positions(8, 10)
    P = params()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = os.path.join(d, "t.sqlite")
        db = tree.TreeDB(path)
        db2 = None
        try:
            an = MockAnalyzer()
            prov = tree.Provider(db, an)
            trees1 = [tree.make_own(prov, f, 2, P, root=True) for f in fens]
            db.commit()
            assert an.calls > 0
            db2 = tree.TreeDB(path)
            prov2 = tree.Provider(db2, None)          # sem motor: tudo tem de vir do banco
            trees2 = [tree.make_own(prov2, f, 2, P, root=True) for f in fens]
            assert prov2.stats["sf"] == 0 and prov2.stats["db"] > 0
            pol = policy_for(trees1)
            pol2 = policy_for(trees2)
            for a_, b_ in zip(trees1, trees2):
                Qa = minimax.root_values(a_, 2, minimax.Ctx(lam=15.0, pol=pol))
                Qb = minimax.root_values(b_, 2, minimax.Ctx(lam=15.0, pol=pol2))
                assert np.allclose(Qa, Qb)
        finally:
            # no Windows o arquivo nao pode ser apagado com a conexao aberta
            db.close()
            if db2 is not None:
                db2.close()


def test_analise_ausente_levanta_erro_explicito():
    prov = tree.Provider(tree.TreeDB(":memory:"), None)
    try:
        tree.make_own(prov, chess.STARTING_FEN, 1, params(), root=True)
    except tree.MissingAnalysis as e:
        assert "|" in str(e)
        return
    raise AssertionError("deveria ter levantado MissingAnalysis")


def test_arvore_nao_depende_da_politica_nem_de_lambda():
    P = params()
    trees, prov, _ = build(random_positions(5, 11), P, 2)
    n1 = prov.stats["sf"]
    pol_a, pol_b = policy_for(trees, seed=1), policy_for(trees, seed=2)
    for node in trees:
        minimax.root_values(node, 2, minimax.Ctx(lam=5.0, pol=pol_a))
        minimax.root_values(node, 2, minimax.Ctx(lam=80.0, pol=pol_b))
    assert prov.stats["sf"] == n1, "resolver nao pode consultar o motor"


def test_valores_com_lambda_zero_nao_dependem_da_politica():
    P = params()
    trees, _, _ = build(random_positions(10, 12), P, 2)
    pa, pb = policy_for(trees, seed=1), policy_for(trees, seed=2)
    for node in trees:
        qa = minimax.root_values(node, 2, minimax.Ctx(lam=0.0, pol=pa))
        qb = minimax.root_values(node, 2, minimax.Ctx(lam=0.0, pol=pb))
        q0 = minimax.root_values(node, 2, minimax.Ctx(lam=0.0, pol=None))
        assert np.allclose(qa, qb) and np.allclose(qa, q0)


def test_pipeline_completo_de_metricas_roda_e_respeita_invariantes():
    P = params()
    fens = random_positions(40, 13)
    trees_nodes, _, _ = build(fens, P, 2)
    datasets = ["romantic_test"] * 14 + ["positional_test"] * 14 + ["fischer"] * 12
    players = (["Morphy", "Tal"] * 7 + ["Karpov", "Petrosian"] * 7 + ["Fischer"] * 12)
    rng = random.Random(0)
    trees = []
    for i, (f, nd) in enumerate(zip(fens, trees_nodes)):
        human = rng.choice(nd.cands).uci if rng.random() < 0.8 else nd.cands[0].uci
        trees.append(SimpleNamespace(idx=i, pid=i, dataset=datasets[i], player=players[i], fen=f,
                                     human=human, root=nd))
    pols = {n: policy_for(trees_nodes, seed=s) for n, s in (("rom", 1), ("pos", 2), ("base", 3))}
    args = SimpleNamespace(no_renorm=False, temps=[1.0], prune_n=10)
    lams, depths = [0.0, 10.0, 80.0], [1, 2]
    quiet = lambda *_: None
    M = evaluation.run_configs(trees, pols, args, depths, lams, [1.0], log_fn=quiet)
    for name in pols:
        m0 = M[(name, 1.0, 1, 0.0)]
        assert np.nansum(m0["div_sf1"]) == 0, "lambda=0 na profundidade 1 tem de coincidir com o top-1"
        assert np.nansum(m0["div_obj"]) == 0
        assert np.nanmax(M[(name, 1.0, 2, 80.0)]["cost_obj"]) >= 0
        assert np.nanmin(M[(name, 1.0, 2, 80.0)]["cost_obj"]) >= -1e-9
    lines = []
    res = evaluation.summarize(trees, M, list(pols), depths, lams, [1.0], 10.0, 2, log_fn=lines.append)
    assert any("INTERACAO" in s for s in lines) and any("POR JOGADOR" in s for s in lines)
    pr = evaluation.prune_stats(trees, pols, args, depths, 10.0, log_fn=lines.append)
    assert pr[2]["mismatches"] == 0
    assert res, "summarize nao produziu resultados"
