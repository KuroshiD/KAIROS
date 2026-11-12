"""
style_common.py  (KAIROS)

Corpus de estilo (M3) e utilitarios compartilhados pelo ajuste fino (03) e pela busca:

  - flags de estilo por lance (captura, xeque, sacrificio, ...): usadas para
    MEDIR a direcao do estilo (H2) e adaptar pesos por epoca. Elas NAO entram
    na funcao de decisao: no KAIROS, h(a) = log p_estilo(a|s) (PDF, Secao 5.3).
  - leitura de PGN por jogador (so os lances do jogador-alvo)
  - CLI `build`: monta os corpora de estilo (jsonl) a partir de

        PGN_DIR/romantic/{Morphy,Tal,Kasparov,Anderssen}.pgn
        PGN_DIR/positional/{Karpov,Petrosian,Capablanca}.pgn
        PGN_DIR/test/Fischer.pgn

    Anderssen (2 partidas) vai para split "ref" (so qualitativo, sem gradiente).
    Fischer vai para split "all" (nunca treina).

Uso:
    kairos-corpus build --pgn-dir data/corpus --out data/style-data \
      --stockfish stockfish/<exe> --engines 7
"""

import argparse
import glob
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.pgn
import numpy as np

from kairos.engine import Analyzer

PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
      chess.QUEEN: 9, chess.KING: 100}

FLAGS = ["capture", "check", "sacrifice", "quiet", "castle",
         "queen", "pawn_push", "king_zone"]

REF_ONLY = set()
GROUPS = ("romantic", "positional", "test")


# ============================================================
# FLAGS DE ESTILO
# ============================================================

def is_sacrifice(b, m):
    """Entrega liquida >= 2 pontos numa casa atacada e mal defendida."""

    pt = b.piece_type_at(m.from_square)

    if pt in (None, chess.KING) or m.promotion:
        return False

    if b.is_capture(m):
        victim = 1 if b.is_en_passant(m) else PV[b.piece_type_at(m.to_square)]
    else:
        victim = 0

    if PV[pt] - victim < 2:
        return False

    b.push(m)

    try:
        att = b.attackers(b.turn, m.to_square)      # adversario
        if not att:
            return False
        dfn = b.attackers(not b.turn, m.to_square)  # quem jogou
        low = min(PV[b.piece_type_at(s)] for s in att)
        return (not dfn) or low < PV[pt]
    finally:
        b.pop()


def move_flags(b, m):

    pt = b.piece_type_at(m.from_square)
    ek = b.king(not b.turn)
    gives_check = b.gives_check(m)
    cap = b.is_capture(m)

    f = np.zeros(len(FLAGS), dtype=np.float32)
    f[0] = cap
    f[1] = gives_check
    f[2] = is_sacrifice(b, m)
    f[3] = (not cap) and (not gives_check)
    f[4] = b.is_castling(m)
    f[5] = pt == chess.QUEEN
    f[6] = pt == chess.PAWN and not cap
    f[7] = (pt not in (chess.PAWN, chess.KING) and ek is not None
            and chess.square_distance(m.to_square, ek) <= 2
            and chess.square_distance(m.from_square, ek) > 2)
    return f


def annotate_entry(an, e, k, nodes):
    """Preenche rank do lance humano no MultiPV e a perda em cp."""

    b = chess.Board(e["fen"])
    hm = chess.Move.from_uci(e["uci"])
    c = an.multipv(b, k, nodes)

    if not c:
        return

    best = c[0]["cp"]

    for i, x in enumerate(c):
        if x["move"] == hm:
            rank, cp = i, x["cp"]
            break
    else:
        r = an.multipv(b, 1, nodes, root_moves=[hm])
        rank, cp = -1, (r[0]["cp"] if r else best - 1000)

    e["rank"] = rank
    e["cp_loss"] = int(max(0, best - cp))
    e["best"] = c[0]["move"].uci()


# ============================================================
# PGN POR JOGADOR
# ============================================================

def game_key(ucis):
    return hashlib.md5(" ".join(ucis).encode()).hexdigest()


def split_of(key):
    v = int(key[:8], 16) % 10
    return "val" if v == 0 else ("test" if v == 1 else "train")


def _is_player(name, sn):
    """Compara sobrenome ou nome completo com o header do jogador."""

    name = name.strip().lower().replace(".", " ")
    want = sn.strip().lower().replace(",", " ").split()

    if "," in name and len(want) == 1:
        return name.split(",")[0].strip() == want[0]

    return set(want) <= set(name.replace(",", " ").split())


def read_player(path, surname=None, skip_plies=8):
    """[(game_key, [entradas])]; sem jogador, inclui lances dos dois lados."""

    sn = surname.lower() if surname else None
    out, seen = [], set()

    with open(path, encoding="utf-8", errors="ignore") as f:

        while True:

            g = chess.pgn.read_game(f)

            if g is None:
                break

            if g.errors:
                continue

            if sn is None:
                colors = (chess.WHITE, chess.BLACK)
            else:
                w = _is_player(g.headers.get("White", ""), sn)
                bl = _is_player(g.headers.get("Black", ""), sn)
                if w == bl:
                    continue
                colors = (chess.WHITE if w else chess.BLACK,)

            board = g.board()
            ucis, ents = [], []

            for ply, mv in enumerate(g.mainline_moves()):

                if (ply >= skip_plies and board.turn in colors
                        and board.legal_moves.count() >= 2):
                    ents.append({"fen": board.fen(), "uci": mv.uci(),
                                 "ply": ply})

                ucis.append(mv.uci())
                board.push(mv)

            if len(ucis) < 10 or not ents:
                continue

            key = game_key(ucis)

            if key in seen:
                continue

            seen.add(key)
            out.append((key, ents))

    return out


def load_jsonl(path, split=None):

    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if split is None or r["split"] in split:
                rows.append(r)

    return rows


# ============================================================
# BUILD
# ============================================================

def collect(path, surname, group, args, forced_split=None):
    """Le um PGN e devolve as entradas, com split por partida."""

    tag = surname or os.path.splitext(os.path.basename(path))[0]
    ref = forced_split == "ref"
    games = read_player(path, surname, args.skip_plies)
    n_all = len(games)
    rng = random.Random(f"{args.seed}-{tag}")

    if not ref and args.max_games and len(games) > args.max_games:
        games = rng.sample(games, args.max_games)

    ents = []

    for key, es in games:
        sp = forced_split or split_of(key)
        if args.pos_per_game and len(es) > args.pos_per_game:
            es = rng.sample(es, args.pos_per_game)
        for e in es:
            e.update(player=tag, group=group, split=sp, game=key)
            ents.append(e)

    if args.pos_per_player and len(ents) > args.pos_per_player:
        ents = rng.sample(ents, args.pos_per_player)

    print(f"{group}/{tag}: {len(games)}/{n_all} partidas, "
          f"{len(ents):,} posicoes", flush=True)

    return ents


def finish_group(rows, group, args, an):
    """Anota com Stockfish (opcional) e grava style_<group>.jsonl."""

    if not rows:
        return

    if an is not None:

        t0 = time.time()
        print(f"  anotando {len(rows):,} posicoes com Stockfish "
              f"({args.label_nodes} nodes, multipv {args.k})", flush=True)

        with ThreadPoolExecutor(args.engines) as ex:
            list(ex.map(lambda e: annotate_entry(
                an, e, args.k, args.label_nodes), rows))

        print(f"  ok ({(time.time() - t0) / 60:.1f} min)")

        for pl in sorted({r["player"] for r in rows}):
            rs = [r for r in rows if r["player"] == pl and "rank" in r]
            if rs:
                print(f"    {pl}: diverge do top-1 SF em "
                      f"{np.mean([r['rank'] != 0 for r in rs]):.1%} | "
                      f"perda media {np.mean([r['cp_loss'] for r in rs]):.0f} cp"
                      f" | perda>200cp {np.mean([r['cp_loss'] > 200 for r in rs]):.1%}")

    path = os.path.join(args.out, f"style_{group}.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"  -> {path}")

    sp = {s: sum(r["split"] == s for r in rows)
          for s in ("train", "val", "test")}
    print(f"  splits (posicoes): {sp}")


def build_legacy(args, an):

    for group in GROUPS:

        rows = []

        for p in sorted(glob.glob(os.path.join(args.pgn_dir, group, "*.pgn"))):
            surname = os.path.splitext(os.path.basename(p))[0]
            forced = ("ref" if surname.lower() in REF_ONLY
                      else "all" if group == "test" else None)
            rows.extend(collect(p, surname, group, args, forced))

        finish_group(rows, group, args, an)


def cmd_build(args):

    os.makedirs(args.out, exist_ok=True)

    if not args.pgn and not args.pgn_dir:
        raise SystemExit("use --pgn ARQ.pgn [...] ou --pgn-dir")

    an = Analyzer(args.stockfish, args.engines) if args.stockfish else None

    if args.pgn:
        name = args.name or os.path.splitext(os.path.basename(args.pgn[0]))[0]
        rows = []
        for p in args.pgn:
            rows.extend(collect(p, args.player, name, args))
        finish_group(rows, name, args, an)
    else:
        build_legacy(args, an)

    if an is not None:
        an.close()


def main():

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--pgn", nargs="+", default=None,
                   help="um ou mais PGNs genericos (modo livre)")
    b.add_argument("--player", default=None,
                   help="jogador-alvo; omitido = lances de brancas e pretas")
    b.add_argument("--name", default=None,
                   help="nome do corpus (padrao: nome do primeiro PGN)")
    b.add_argument("--pgn-dir", default=None,
                   help="modo antigo: PGN_DIR/<grupo>/<Sobrenome>.pgn")
    b.add_argument("--out", required=True)
    b.add_argument("--stockfish", default=None,
                   help="se dado, anota rank/perda em cp de cada lance humano")
    b.add_argument("--engines", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    b.add_argument("--max-games", type=int, default=260,
                   help="0 = sem limite")
    b.add_argument("--pos-per-game", type=int, default=24,
                   help="maximo de posicoes amostradas por partida (0 = sem limite)")
    b.add_argument("--pos-per-player", type=int, default=8000,
                   help="0 = sem limite")
    b.add_argument("--skip-plies", type=int, default=8,
                   help="ignora a teoria de abertura")
    b.add_argument("--k", type=int, default=12, help="MultiPV da anotacao")
    b.add_argument("--label-nodes", type=int, default=3000)
    b.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.cmd == "build":
        cmd_build(args)


if __name__ == "__main__":
    main()