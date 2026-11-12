"""Banco SQLite de analises do Stockfish e arvore de candidatos (independente de politica e de lambda)."""
import json
import os
import sqlite3
import threading
from types import SimpleNamespace

import chess

from kairos.search.common import CLIP, log, clip


# =============================================================================
# BANCO (SQLite) E PROVEDOR DE ANALISES DO STOCKFISH
# =============================================================================

class MissingAnalysis(Exception):
    pass


class TreeDB:
    def __init__(self, path):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.mem = {}
        self.pending = 0
        with self.lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS analysis(key TEXT PRIMARY KEY, val TEXT)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS positions(id INTEGER PRIMARY KEY, dataset TEXT, "
                              "player TEXT, fen TEXT, human TEXT)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
            self.conn.commit()

    def get(self, key):
        v = self.mem.get(key)
        if v is not None:
            return v
        with self.lock:
            row = self.conn.execute("SELECT val FROM analysis WHERE key=?", (key,)).fetchone()
        if row:
            v = json.loads(row[0])
            self.mem[key] = v
            return v
        return None

    def put(self, key, val):
        self.mem[key] = val
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO analysis(key,val) VALUES (?,?)", (key, json.dumps(val)))
            self.pending += 1
            if self.pending >= 500:
                self.conn.commit()
                self.pending = 0

    def commit(self):
        with self.lock:
            self.conn.commit()
            self.pending = 0

    def close(self):
        with self.lock:
            self.conn.commit()
            self.conn.close()

    def n_analysis(self):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM analysis").fetchone()[0]

    def set_meta(self, k, v):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES (?,?)", (k, json.dumps(v)))
            self.conn.commit()

    def get_meta(self, k):
        with self.lock:
            row = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def positions(self):
        with self.lock:
            return self.conn.execute("SELECT id,dataset,player,fen,human FROM positions ORDER BY id").fetchall()

    def add_positions(self, recs):
        with self.lock:
            self.conn.executemany("INSERT INTO positions(dataset,player,fen,human) VALUES (?,?,?,?)",
                                  [(r["dataset"], r["player"], r["fen"], r["human"]) for r in recs])
            self.conn.commit()


class Provider:
    """analysis(fen,k,nodes) -> [[uci, cp, mate], ...] (cp do ponto de vista de quem joga)."""

    def __init__(self, db, an=None, seeds=None):
        self.db, self.an, self.seeds = db, an, seeds or {}
        self.stats = {"db": 0, "sf": 0, "seed": 0}

    def analysis(self, fen, k, nodes):
        key = f"{fen}|{k}|{nodes}"
        v = self.db.get(key)
        if v is not None:
            self.stats["db"] += 1
            return v
        if key in self.seeds:
            v = self.seeds[key]
            self.stats["seed"] += 1
        elif self.an is None:
            raise MissingAnalysis(key)
        else:
            cs = self.an.multipv(chess.Board(fen), k, nodes)
            v = [[c["move"].uci(), int(c["cp"]), c["mate"]] for c in cs]
            self.stats["sf"] += 1
        self.db.put(key, v)
        return v


# =============================================================================
# ARVORE (independente de politica e de lambda)
# =============================================================================

class Cand:
    __slots__ = ("uci", "cp", "opp", "terminal")

    def __init__(self, uci, cp):
        self.uci, self.cp, self.opp, self.terminal = uci, cp, None, None


class OwnNode:
    __slots__ = ("fen", "cands", "lp")

    def __init__(self, fen):
        self.fen, self.cands, self.lp = fen, [], {}


class Reply:
    __slots__ = ("uci", "cp", "own", "terminal")

    def __init__(self, uci, cp):
        self.uci, self.cp, self.own, self.terminal = uci, cp, None, None


class OppNode:
    __slots__ = ("replies",)

    def __init__(self):
        self.replies = []


def own_candidates(prov, fen, P, root):
    raw = prov.analysis(fen, P.root_k, P.nodes) if root else prov.analysis(fen, P.k_own, P.leaf_nodes)
    if not raw:
        return []
    valid = [r for r in raw if not (r[2] is not None and r[2] < 0)] or raw[:1]
    best = max(r[1] for r in valid)
    sel = [r for r in valid if r[1] >= best - P.eps]
    sel.sort(key=lambda r: -r[1])
    return sel[:P.k_own]


def make_own(prov, fen, d, P, root=False):
    node = OwnNode(fen)
    cands = own_candidates(prov, fen, P, root)
    b = chess.Board(fen)
    for r in cands:
        c = Cand(r[0], clip(r[1]))
        if d > 1:
            b.push(chess.Move.from_uci(r[0]))
            if b.is_game_over():
                c.terminal = CLIP if b.is_checkmate() else 0.0
            else:
                c.opp = make_opp(prov, b.fen(), d, P)
            b.pop()
        node.cands.append(c)
    return node


def make_opp(prov, fen, d, P):
    raw = prov.analysis(fen, P.k_opp, P.opp_nodes)
    if not raw:
        return None
    best = max(r[1] for r in raw)
    sel = [r for r in raw if r[1] >= best - P.opp_eps]
    sel.sort(key=lambda r: -r[1])
    op = OppNode()
    b = chess.Board(fen)
    for r in sel[:P.k_opp]:
        rep = Reply(r[0], clip(r[1]))
        b.push(chess.Move.from_uci(r[0]))
        if b.is_game_over():
            rep.terminal = -CLIP if b.is_checkmate() else 0.0
        else:
            rep.own = make_own(prov, b.fen(), d - 1, P)
        b.pop()
        op.replies.append(rep)
    return op


def collect_need(node, out):
    """fen -> conjunto de lances candidatos (para pontuar com a politica)."""
    s = out.setdefault(node.fen, set())
    for c in node.cands:
        s.add(c.uci)
        if c.opp is not None:
            for r in c.opp.replies:
                if r.own is not None:
                    collect_need(r.own, out)


def load_trees(db, P, depth, log_fn=log):
    prov = Provider(db, an=None)
    rows = db.positions()
    trees, missing, errors = [], 0, []
    for (pid, dataset, player, fen, human) in rows:
        try:
            root = make_own(prov, fen, depth, P, root=True)
        except MissingAnalysis as e:
            missing += 1
            if len(errors) < 3:
                errors.append(str(e))
            continue
        trees.append(SimpleNamespace(idx=len(trees), pid=pid, dataset=dataset, player=player,
                                     fen=fen, human=human, root=root))
    log_fn(f"arvores completas: {len(trees)}/{len(rows)} (incompletas: {missing}; consultas ao banco={prov.stats['db']})")
    if missing:
        log_fn("AVISO: ha posicoes incompletas para estes parametros/profundidade; rode 'build' com os mesmos "
               f"parametros. Exemplos de chaves ausentes: {errors}")
    return trees


TREE_KEYS = ("k_own", "k_opp", "eps", "opp_eps", "nodes", "root_k", "leaf_nodes", "opp_nodes")


def tree_params(args):
    return {k: getattr(args, k) for k in TREE_KEYS}
