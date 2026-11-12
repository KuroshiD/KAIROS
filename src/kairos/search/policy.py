"""Politica de estilo como prior pi(a|s): rede treinada ou controle aleatorio; pontuacao em lote."""
import hashlib
import os
import random
import time

import chess
import numpy as np

from kairos.search.common import log, logsumexp
from kairos.search.tree import collect_need


def _pack(lp_by_uci, ucis, temps):
    arr = np.array(list(lp_by_uci.values()), float)
    return ({u: float(lp_by_uci[u]) for u in ucis}, {T: logsumexp(arr / T) for T in temps})


class RandomPolicy:
    """h aleatoria, deterministica por posicao (controle)."""

    def __init__(self, seed=0):
        self.seed = seed

    def raw(self, need, temps):
        out = {}
        for fen, ucis in need.items():
            b = chess.Board(fen)
            rng = random.Random(int(hashlib.md5((fen + str(self.seed)).encode()).hexdigest()[:8], 16))
            moves = [m.uci() for m in b.legal_moves]
            x = np.array([rng.gauss(0, 1) for _ in moves])
            lp = x - logsumexp(x)
            out[fen] = _pack(dict(zip(moves, lp)), ucis, temps)
        return out


class NetPolicy:
    def __init__(self, path, device):
        import torch
        self.torch, self.device = torch, device
        from kairos.chess_ext import board_to_planes, legal_action_ids
        from kairos.models.policy_net import PolicyNet
        self.board_to_planes, self.legal_action_ids = board_to_planes, legal_action_ids
        ck = torch.load(path, map_location=device, weights_only=False)
        self.net = PolicyNet(**ck["config"]).to(device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()

    def raw(self, need, temps, bs=256):
        torch = self.torch
        out, items = {}, list(need.items())
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            boards = [chess.Board(f) for f, _ in chunk]
            x = torch.from_numpy(np.stack([self.board_to_planes(b) for b in boards])).float().to(self.device)
            with torch.no_grad():
                logits = self.net(x)["policy"].float().cpu().numpy()
            for j, (b, (fen, ucis)) in enumerate(zip(boards, chunk)):
                moves = list(b.legal_moves)
                ids = list(self.legal_action_ids(b))
                s = logits[j, ids]
                s = s - s.max()
                lp = s - np.log(np.exp(s).sum())
                out[fen] = _pack(dict(zip([m.uci() for m in moves], lp)), ucis, temps)
        return out


def node_logpi(node, pol, T, renorm):
    key = (id(pol), T, renorm)
    v = node.lp.get(key)
    if v is None:
        d, lses = pol[node.fen]
        x = np.array([d[c.uci] for c in node.cands], float) / T
        z = logsumexp(x) if renorm else lses[T]
        v = x - z
        node.lp[key] = v
    return v


def parse_policies(specs, device):
    pols = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        if path == "random":
            pols[name] = RandomPolicy()
        else:
            if not os.path.exists(path):
                raise SystemExit(f"checkpoint nao encontrado: {path}")
            pols[name] = NetPolicy(path, device)
        log(f"politica '{name}': {path}")
    return pols


def score_all(trees, pols, temps):
    need = {}
    for t in trees:
        collect_need(t.root, need)
    log(f"pontuando {len(need)} posicoes proprias distintas com {len(pols)} politicas ...")
    out = {}
    for name, p in pols.items():
        t0 = time.time()
        out[name] = p.raw(need, temps)
        log(f"  politica {name}: ok em {time.time() - t0:.0f}s")
    return out
