"""
app/server.py: serve o kairos_play.html e decide os lances com o KAIROS v2
(busca adversarial hibrida, pacote kairos.search).

Requer o pacote instalado (pip install -e .). Uso, a partir da raiz do projeto:
  python app/server.py --stockfish stockfish/<exe> \
      --policy rom=runs/style_rom/best.pt --policy pos=runs/style_pos/best.pt \
      --policy base=runs/p1/best.pt --port 8000 --threads 2 --hash 256
  e abra http://localhost:8000

Formulação (v2): U = v(folha) + lambda * soma log pi(lances próprios); nós próprios
maximizam, nós do adversário minimizam. lambda em centipawns por nat.
A árvore (candidatos e valores do Stockfish) é construída ao vivo para a posição
pedida; os valores Q(a) são resolvidos por minimax (ou alfa-beta) com a política
escolhida.

Tempo de pensamento: se o pedido traz "think" (segundos), a árvore é construída por
aprofundamento iterativo com orçamento de tempo: a raiz recebe uma fatia, e cada
profundidade extra só entra se terminar dentro do prazo (senão vale a última
profundidade completa). Sem "think", vale o modo por nós (reprodutível), com a
profundidade pedida. Para os experimentos, use `kairos build|solve`:
o modo por tempo não é reprodutível.
"""

import argparse
import json
import math
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import chess
import chess.engine
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from kairos.engine import Analyzer  # noqa: E402
from kairos.search import common, minimax, tree
from kairos.search import policy as policies  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--stockfish", required=True)
p.add_argument("--policy", action="append", required=True,
               help="nome=caminho.pt ou nome=random")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--engines", type=int, default=2)
p.add_argument("--threads", type=int, default=1, help="threads por engine")
p.add_argument("--hash", type=int, default=128, help="MB de hash por engine")
p.add_argument("--think-max", type=float, default=20.0, help="teto do tempo por lance (s)")
p.add_argument("--cpu", action="store_true")
# parâmetros da busca (mesmos nomes e padrões do 05_kairos_minimax.py)
p.add_argument("--k-own", type=int, default=6, help="largura nos nós próprios")
p.add_argument("--k-opp", type=int, default=3, help="largura nos nós do adversário")
p.add_argument("--eps", type=float, default=100, help="janela (cp) dos candidatos próprios")
p.add_argument("--opp-eps", type=float, default=150, help="janela (cp) das respostas do adversário")
p.add_argument("--nodes", type=int, default=5000, help="nós do Stockfish na raiz (modo sem tempo)")
p.add_argument("--root-k", type=int, default=16, help="MultiPV da raiz")
p.add_argument("--leaf-nodes", type=int, default=1000, help="nós nos nós próprios internos")
p.add_argument("--opp-nodes", type=int, default=1000, help="nós nos nós do adversário")
p.add_argument("--no-renorm", action="store_true",
               help="não renormaliza pi dentro dos candidatos")
p.add_argument("--lam", type=float, default=20.0, help="lambda padrão (cp/nat)")
p.add_argument("--max-depth", type=int, default=2,
               help="teto de profundidade (lances próprios); 3 é caro")
p.add_argument("--analysis-nodes", type=int, default=20000, help="nós na aba de análise")
p.add_argument("--cache-max", type=int, default=200000, help="análises em cache (RAM)")
A = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() and not A.cpu else "cpu")
S = policies.parse_policies(A.policy, dev)

an = Analyzer(A.stockfish, A.engines)
for engine in an.engines:
    engine.configure({"Threads": A.threads, "Hash": A.hash})

P = SimpleNamespace(k_own=A.k_own, k_opp=A.k_opp, eps=A.eps, opp_eps=A.opp_eps,
                    nodes=A.nodes, root_k=A.root_k, leaf_nodes=A.leaf_nodes,
                    opp_nodes=A.opp_nodes)
RENORM = not A.no_renorm

LOCK = threading.Lock()
EV = chess.engine.SimpleEngine.popen_uci(A.stockfish)
EV_LOCK = threading.Lock()
HTML = os.path.join(HERE, "kairos_play.html")


# =============================================================================
# CACHE E PROVEDORES DE ANÁLISE
# =============================================================================

class MemDB:
    """Cache em RAM com a interface get/put que o Provider usa (limpa ao encher)."""

    def __init__(self, cap):
        self.cap, self.d = cap, {}

    def get(self, key):
        return self.d.get(key)

    def put(self, key, val):
        if len(self.d) >= self.cap:
            self.d.clear()
        self.d[key] = val


DB = MemDB(A.cache_max)


class TimeUp(Exception):
    pass


def multipv_time(analyzer, board, k, seconds):
    """Igual a Analyzer.multipv, mas limitado por tempo em vez de nós."""
    engine = analyzer.q.get()
    try:
        infos = engine.analyse(board, chess.engine.Limit(time=seconds), multipv=k)
    finally:
        analyzer.q.put(engine)
    if isinstance(infos, dict):
        infos = [infos]
    out = []
    for d in infos:
        pv, sc = d.get("pv"), d.get("score")
        if not pv or sc is None:
            continue
        s = sc.pov(board.turn)
        out.append([pv[0].uci(), int(s.score(mate_score=10000)), s.mate()])
    return out


class TimedProvider(tree.Provider):
    """Provider com orçamento de tempo.

    A raiz recebe `root_sec` (sempre completa). Cada análise interna recebe uma
    fatia fixa `step`; se não sobrar uma fatia inteira até o prazo, levanta TimeUp
    e o aprofundamento iterativo mantém a última profundidade completa.
    """

    def __init__(self, db, an, root_fen, root_k, root_sec, step, deadline):
        super().__init__(db, an)
        self.root_fen, self.root_k = root_fen, root_k
        self.root_sec, self.step, self.deadline = root_sec, step, deadline

    def analysis(self, fen, k, nodes):
        if fen == self.root_fen and k == self.root_k:
            sec = self.root_sec
        else:
            if self.deadline - time.monotonic() < self.step:
                raise TimeUp()
            sec = self.step
        key = f"{fen}|{k}|t{sec:.3f}"
        v = self.db.get(key)
        if v is not None:
            self.stats["db"] += 1
            return v
        v = multipv_time(self.an, chess.Board(fen), k, sec)
        self.stats["sf"] += 1
        self.db.put(key, v)
        return v


def build_tree(board, depth, lam, think):
    """Devolve (raiz, profundidade_usada, stats)."""
    fen = board.fen()
    if think <= 0:
        prov = tree.Provider(DB, an)
        return tree.make_own(prov, fen, depth, P, root=True), depth, prov.stats
    deep = depth > 1
    start = time.monotonic()
    root_sec = max(0.05, think * (0.4 if deep else 1.0))
    step = min(0.5, max(0.03, 0.6 * think / 25))  # ~25 chamadas na profundidade 2
    prov = TimedProvider(DB, an, fen, P.root_k, root_sec, step, start + think)
    root = tree.make_own(prov, fen, 1, P, root=True)
    used = 1
    for d in range(2, depth + 1):
        try:
            root_d = tree.make_own(prov, fen, d, P, root=True)
        except TimeUp:
            break
        root, used = root_d, d
    return root, used, prov.stats


# =============================================================================
# DECISÃO
# =============================================================================

def decide(q):
    board = chess.Board(q["fen"])
    if board.is_game_over():
        raise ValueError("posição sem lances")
    name = q.get("policy")
    if name not in S:
        raise ValueError(f"política desconhecida: {name}")
    lam = max(0.0, float(q.get("lam", A.lam)))
    depth = max(1, min(A.max_depth, int(q.get("depth", A.max_depth))))
    if "budget" in q and "depth" not in q and int(q["budget"]) == 0:
        depth = 1  # compatibilidade: budget 0 = sem busca funda
    if lam <= 0:
        depth = 1  # lambda 0: lance objetivo do motor, aprofundar não muda nada
    T = max(0.05, min(5.0, float(q.get("temp", 1.0))))
    think = max(0.0, min(A.think_max, float(q.get("think") or 0)))
    prune = bool(q.get("prune", False))
    sample = bool(q.get("sample", False)) and lam > 0
    if sample:
        prune = False  # a decisão suave precisa do Q exato de todos os candidatos
    seed = int(q.get("seed", 0))

    t0 = time.monotonic()
    with LOCK:
        root, used, stats = build_tree(board, depth, lam, think)
        cands = root.cands
        if not cands:
            raise ValueError("Stockfish não retornou candidatos")
        need = {}
        tree.collect_need(root, need)
        pol = S[name].raw(need, [T])
        q0 = minimax.root_values(root, used, minimax.Ctx(lam=0.0))
        ctx = minimax.Ctx(lam=lam, pol=pol, T=T, renorm=RENORM, prune=prune)
        Q = minimax.root_values(root, used, ctx)
        lp = policies.node_logpi(root, pol, T, RENORM)
    n = len(cands)
    top1 = max(range(n), key=lambda k: (cands[k].cp, -k))
    a0 = minimax.argbest(q0, cands)
    qs = common.softmax([x / lam for x in Q]) if lam > 0 and not prune else None
    if sample:
        j = random.Random(seed).choices(range(n), weights=list(qs))[0]
    else:
        j = minimax.argbest(Q, cands)
    elapsed = time.monotonic() - t0

    def san(i):
        return board.san(chess.Move.from_uci(cands[i].uci))

    ranking = [{"uci": cands[i].uci, "san": san(i), "cp": cands[i].cp,
                "logp": float(lp[i]), "f": float(Q[i]),
                "q": None if qs is None else float(qs[i])}
               for i in sorted(range(n), key=lambda k: -Q[k])]
    return {
        "move": cands[j].uci, "san": san(j),
        "top": cands[top1].uci,                      # topo do Stockfish
        "cost": float(cands[top1].cp - cands[j].cp),  # perda vs. topo do Stockfish (cp)
        "cost_obj": float(max(q0) - q0[j]),           # perda objetiva (valor minimax)
        "diverge": bool(j != top1),
        "diverge_obj": bool(j != a0),
        "ranking": ranking,
        "depth_used": used, "lam": lam, "temp": T,
        "engine_calls": stats["sf"], "cache_hits": stats["db"],
        "time": round(elapsed, 2),
    }


# =============================================================================
# AVALIAÇÃO E ANÁLISE
# =============================================================================

def eval_pos(fen, depth=12):
    b = chess.Board(fen)
    if b.is_checkmate():
        return {"cp": -10000 if b.turn else 10000, "mate": None}
    if b.is_game_over():
        return {"cp": 0, "mate": None}
    with EV_LOCK:
        info = EV.analyse(b, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    if score.is_mate():
        return {"cp": None, "mate": score.mate()}
    return {"cp": score.score(), "mate": None}


PLY = 10


def fmt(board, moves):
    position = board.copy(stack=False)
    result = []
    for index, move in enumerate(moves):
        if move not in position.legal_moves:
            break
        if position.turn == chess.WHITE:
            result.append(f"{position.fullmove_number}.")
        elif index == 0:
            result.append(f"{position.fullmove_number}...")
        result.append(position.san(move))
        position.push(move)
    return " ".join(result)


def policy_logp(scorer, board):
    """log pi(a|s) sobre TODOS os lances legais (T=1)."""
    fen = board.fen()
    ucis = {m.uci() for m in board.legal_moves}
    d, _ = scorer.raw({fen: ucis}, [1.0])[fen]
    return {chess.Move.from_uci(u): v for u, v in d.items()}


def model_line(board, first, scorer, n=PLY):
    """Segue o lance mais provável do modelo para os dois lados."""
    position = board.copy()
    position.push(first)
    moves = [first]
    while len(moves) < n and not position.is_game_over():
        lps = policy_logp(scorer, position)
        move = max(lps, key=lps.get)
        position.push(move)
        moves.append(move)
    return moves


def analyze(fen, policy=None, k=3):
    board = chess.Board(fen)
    if board.is_game_over():
        return {**eval_pos(fen), "lines": [], "model": []}

    sign = 1 if board.turn == chess.WHITE else -1
    with LOCK:
        candidates = an.multipv(board, k, A.analysis_nodes)
        scorer = S[policy if policy in S else next(iter(S))]
        lps = policy_logp(scorer, board)
        top_model = sorted(lps.items(), key=lambda item: -item[1])[:k]
        model = [
            {
                "uci": move.uci(),
                "san": board.san(move),
                "p": math.exp(lpv),
                "txt": fmt(board, model_line(board, move, scorer)),
            }
            for move, lpv in top_model
        ]

    lines = []
    for candidate in candidates[:k]:
        mate = candidate["mate"]
        lines.append({
            "uci": candidate["move"].uci(),
            "txt": fmt(board, [candidate["move"]] + list(candidate["pv"][1:PLY])),
            "cp": None if mate is not None else sign * candidate["cp"],
            "mate": None if mate is None else sign * mate,
        })
    if not lines:
        raise RuntimeError("Stockfish não retornou linhas de análise")
    return {
        "cp": lines[0]["cp"],
        "mate": lines[0]["mate"],
        "lines": lines,
        "model": model,
    }


# =============================================================================
# HTTP
# =============================================================================

class H(BaseHTTPRequestHandler):

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        if self.path.startswith("/api/info"):
            info = {"policies": list(S), "version": "v2-minimax",
                    "defaults": {"lam": A.lam, "depth": A.max_depth, "temp": 1.0,
                                 "think_max": A.think_max, "k_own": A.k_own,
                                 "k_opp": A.k_opp, "eps": A.eps, "opp_eps": A.opp_eps}}
            return self._send(200, json.dumps(info).encode())
        if self.path == "/logo.png":
            try:
                with open(os.path.join(HERE, "logo.png"), "rb") as f:
                    return self._send(200, f.read(), "image/png")
            except FileNotFoundError:
                return self._send(404, b"")
        with open(HTML, "rb") as f:
            self._send(200, f.read(), "text/html; charset=utf-8")

    def do_POST(self):
        try:
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.startswith("/api/eval"):
                return self._send(200, json.dumps(eval_pos(q["fen"])).encode())
            if self.path.startswith("/api/analyze"):
                return self._send(200, json.dumps(
                    analyze(q["fen"], q.get("policy"), int(q.get("k", 3)))
                ).encode())
            self._send(200, json.dumps(decide(q)).encode())
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode())


print(f"KAIROS Play (v2 minimax) em http://localhost:{A.port}  (políticas: {', '.join(S)})")
try:
    ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
finally:
    try:
        an.close()
    finally:
        EV.quit()