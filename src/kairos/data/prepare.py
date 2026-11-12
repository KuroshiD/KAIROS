"""
01_prepare_policy_dataset.py

Dataset UNICO de policy pura: PGN humano (Lichess) + puzzles, com
train / val / test contendo os DOIS tipos de dado.

PGN
  * filtro por ANO (--years, padrao 2013 2014 2015), Elo, controle de
    tempo, variante e termino da partida;
  * split por PARTIDA (hash do Site/indice): 75% train, 12,5% val,
    12,5% test. Nenhuma partida aparece em dois splits;
  * aleatorizacao em 4 niveis: (1) varios arquivos lidos em
    intercalado aleatorio, (2) amostragem aleatoria de partidas,
    (3) buffer de embaralhamento das partidas, (4) posicoes sorteadas
    dentro da partida, balanceadas entre opening/middlegame/endgame,
    e linhas embaralhadas dentro de cada chunk;
  * tags: SOMENTE opening / middlegame / endgame (+ os 4 bits de tipo
    de lance: castle, enPassant, promotion, capture).

PUZZLES (lichess_db_puzzle.csv[.zst])
  * cada lance do solver vira uma amostra (o 1o lance do CSV e o do
    adversario e so avanca o tabuleiro);
  * split por PuzzleId com as mesmas proporcoes;
  * tags: fase (do tema opening/middlegame/endgame), temas (mate,
    advantage, ...) e tipo de lance.

LANCES RAROS: en passant e promocao entram SEMPRE; roque entra com
probabilidade --castle-keep (senao dominaria a amostra: 1-2 por
partida). Assim a policy ve exemplos suficientes de castle e en
passant.

Uso:
  python 01_prepare_policy_dataset.py \
      --pgn lichess_2013-*.pgn.zst lichess_2014-*.pgn.zst \
      --puzzles lichess_db_puzzle.csv.zst \
      --out data_policy --games 400000 --puzzle-count 1200000 \
      --workers 8
"""

import argparse
import csv
import io
import json
import math
import os
import random
import re
import time

import chess
import chess.pgn
import numpy as np

from kairos.chess_ext import (
    PHASE_NAMES,
    PUZZLE_GROUP_OFFSET,
    TAG_BIT,
    THEME_TAG_NAMES,
    MOVE_TAG_NAMES,
    SplitWriter,
    assign_split,
    board_to_packed,
    encode_move,
    legal_action_ids,
    move_tags,
    open_text_stream,
    raw_phase,
    rows_to_chunk,
    run_chunked_pipeline,
)

SPLITS = ("train", "val", "test")

_HEADER = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')

_RARE_BITS = TAG_BIT["enPassant"] | TAG_BIT["promotion"]


# ============================================================
# WORKERS (top-level: picklable para o spawn)
# ============================================================

def _rows_from_pgn(task):
    """task = ("pgn", text, group, seed, k, skip_plies, castle_keep)"""

    _, text, group, seed, k, skip_plies, castle_keep = task

    game = chess.pgn.read_game(io.StringIO(text))

    if game is None or game.errors:
        raise ValueError("bad pgn")

    moves = list(game.mainline_moves())

    if len(moves) < skip_plies + 8:
        raise ValueError("short game")

    rng = random.Random(seed)

    # passada 1: fase / tipo de lance de cada ply candidato
    board = game.board()
    by_phase = {0: [], 1: [], 2: []}
    forced = []

    for ply, move in enumerate(moves):

        if ply >= skip_plies and board.legal_moves.count() >= 2:

            mt = move_tags(board, move)

            if mt & _RARE_BITS:
                forced.append(ply)

            elif mt & TAG_BIT["castle"]:
                if rng.random() < castle_keep:
                    forced.append(ply)

            else:
                by_phase[raw_phase(board)].append(ply)

        board.push(move)

    # sorteio balanceado entre as 3 fases (round-robin aleatorio)
    for lst in by_phase.values():
        rng.shuffle(lst)

    chosen = set(forced)
    quota = k
    order = [0, 1, 2]

    while quota > 0 and any(by_phase[p] for p in order):

        rng.shuffle(order)

        for p in order:
            if quota > 0 and by_phase[p]:
                chosen.add(by_phase[p].pop())
                quota -= 1

    # passada 2: codifica so os plies escolhidos
    rows = []
    board = game.board()

    for ply, move in enumerate(moves):

        if ply in chosen:

            tags = (
                TAG_BIT[PHASE_NAMES[raw_phase(board)]]
                | move_tags(board, move)
            )

            rows.append((
                board_to_packed(board),
                encode_move(move),
                tags,
                group,
                legal_action_ids(board),
            ))

        board.push(move)

    return rows


def _theme_bits(themes):

    bits = 0

    for theme in themes:

        if theme in THEME_TAG_NAMES:
            bits |= TAG_BIT[theme]

        elif theme.startswith("mate"):  # mateIn1, mateIn2, ...
            bits |= TAG_BIT["mate"]

    return bits


def _rows_from_puzzle(task):
    """task = ("puz", fen, moves_uci, themes, group)"""

    _, fen, moves_uci, themes_text, group = task

    themes = set(themes_text.split())

    board = chess.Board(fen)

    moves = [chess.Move.from_uci(m) for m in moves_uci.split()]

    if len(moves) < 2:
        raise ValueError("puzzle too short")

    phase = None

    for i, name in enumerate(PHASE_NAMES):
        if name in themes:
            phase = i

    theme_bits = _theme_bits(themes)

    if move_is_illegal(board, moves[0]):
        raise ValueError("illegal first move")

    board.push(moves[0])  # lance do adversario: so avanca

    rows = []

    for i in range(1, len(moves), 2):

        move = moves[i]

        if move_is_illegal(board, move):
            raise ValueError("illegal solver move")

        if board.legal_moves.count() >= 2:

            ph = raw_phase(board) if phase is None else phase

            tags = (
                TAG_BIT[PHASE_NAMES[ph]]
                | theme_bits
                | move_tags(board, move)
            )

            rows.append((
                board_to_packed(board),
                encode_move(move),
                tags,
                group,
                legal_action_ids(board),
            ))

        board.push(move)

        if i + 1 < len(moves):

            if move_is_illegal(board, moves[i + 1]):
                raise ValueError("illegal reply")

            board.push(moves[i + 1])

    return rows


def move_is_illegal(board, move):
    return move not in board.legal_moves


def worker(tasks):

    all_rows = []
    skipped = 0

    for task in tasks:

        try:

            if task[0] == "pgn":
                all_rows.extend(_rows_from_pgn(task))
            else:
                all_rows.extend(_rows_from_puzzle(task))

        except Exception:
            skipped += 1

    # embaralha as linhas do chunk (mistura partidas e puzzles)
    random.Random(len(all_rows) * 7919 + skipped).shuffle(all_rows)

    X, y, tags, group, legal, lengths = [], [], [], [], [], []

    for packed, action, tg, grp, ids in all_rows:
        X.append(packed)
        y.append(action)
        tags.append(tg)
        group.append(grp)
        legal.extend(ids)
        lengths.append(len(ids))

    return rows_to_chunk(X, y, tags, group, legal, lengths, skipped)


# ============================================================
# LEITURA / FILTROS (processo principal, so texto, sem python-chess)
# ============================================================

def iter_pgn_games(path):
    """Gera (headers, texto_bruto) de cada partida do arquivo."""

    with open_text_stream(path) as f:

        lines, headers, has_moves = [], {}, False

        for line in f:

            if line.startswith("[Event ") and has_moves:
                yield headers, "".join(lines)
                lines, headers, has_moves = [], {}, False

            lines.append(line)

            if line.startswith("["):

                m = _HEADER.match(line)

                if m:
                    headers[m.group(1)] = m.group(2)

            elif line.strip():
                has_moves = True

        if lines and has_moves:
            yield headers, "".join(lines)


def game_year(h):

    date = h.get("UTCDate") or h.get("Date") or ""

    return int(date[:4]) if date[:4].isdigit() else None


def accept_game(h, args):
    """Sem restricoes de partida; so o filtro opcional --years."""

    if args.years and game_year(h) not in args.years:
        return False

    return True


class YearSplitter:
    """
    Split ESTRATIFICADO POR ANO, em blocos aleatorios.

    Para cada ano, as partidas (que chegam embaralhadas) sao atribuidas
    em blocos de --split-block posicoes (padrao 8 = 6 train + 1 val +
    1 test), com a ordem sorteada dentro de cada bloco. Efeitos:
      * cada ano fica com 75/12,5/12,5 (+-1 partida) em QUALQUER ponto
        do stream, entao a mistura de anos e a mesma em train/val/test;
      * anos pequenos nao desandam: o desvio maximo e o de um bloco;
      * funciona em streaming, sem contar as partidas antes.
    """

    def __init__(self, ratios, block, seed):

        raw = [ratios[s] * block for s in SPLITS]
        counts = [int(x) for x in raw]

        order = sorted(range(3), key=lambda i: raw[i] - counts[i],
                       reverse=True)

        for i in order[:block - sum(counts)]:
            counts[i] += 1

        self.template = [s for s, c in zip(SPLITS, counts) for _ in range(c)]
        self.counts = dict(zip(SPLITS, counts))
        self.rng = random.Random(seed)
        self.queues = {}

    def next(self, year):

        queue = self.queues.get(year)

        if not queue:
            queue = self.template[:]
            self.rng.shuffle(queue)
            self.queues[year] = queue

        return queue.pop()


def interleave(generators, rng):
    """Le varios iteradores em ordem aleatoria (nao um arquivo por vez)."""

    generators = list(generators)

    while generators:

        i = rng.randrange(len(generators))

        try:
            yield next(generators[i])
        except StopIteration:
            generators.pop(i)


def shuffle_buffer(iterator, size, rng):
    """Embaralhamento aproximado em streaming."""

    buf = []

    for item in iterator:

        if len(buf) < size:
            buf.append(item)
            continue

        j = rng.randrange(size)

        yield buf[j]

        buf[j] = item

    rng.shuffle(buf)

    yield from buf


def count_or_all(text):
    """Inteiro ou 'ALL' (case-insensitive) -> None = sem limite."""

    if text.strip().lower() == "all":
        return None

    value = int(text)

    if value < 0:
        raise argparse.ArgumentTypeError("deve ser >= 0 ou ALL")

    return value


def quotas(total, ratios):
    return {s: int(math.ceil(total * ratios[s])) for s in SPLITS}


def pgn_items(args, ratios, rng, stats):

    limit = float("inf") if args.games is None else args.games

    quota = {s: limit * ratios[s] for s in SPLITS}
    taken = {s: 0 for s in SPLITS}

    splitter = YearSplitter(ratios, args.split_block, args.seed)

    for s in SPLITS:
        if abs(splitter.counts[s] / args.split_block - ratios[s]) > 1e-9:
            print(f"[aviso] --split-block {args.split_block} nao representa "
                  f"exatamente as proporcoes ({splitter.counts}); use um "
                  f"bloco multiplo de 1/proporcao (8 para 75/12,5/12,5).")
            break

    stats["pgn_years"] = {}

    def one(path):
        for h, text in iter_pgn_games(path):
            yield h, text

    stream = shuffle_buffer(
        interleave([one(p) for p in args.pgn], rng),
        args.shuffle_buffer,
        rng,
    )

    counter = 0

    for h, text in stream:

        stats["pgn_seen"] += 1

        if (args.years and stats["pgn_seen"] == 200000
                and stats["pgn_accepted"] == 0):
            raise SystemExit(
                "200k partidas lidas e nenhuma passou no --years. "
                "Confira os arquivos."
            )

        if not accept_game(h, args):
            continue

        stats["pgn_accepted"] += 1

        if rng.random() > args.game_sample:
            continue

        year = game_year(h)
        ykey = str(year) if year is not None else "?"

        split = splitter.next(ykey)

        if taken[split] >= quota[split]:

            if all(taken[s] >= quota[s] for s in SPLITS):
                return

            continue

        taken[split] += 1
        counter += 1

        by = stats["pgn_years"].setdefault(ykey, {s: 0 for s in SPLITS})
        by[split] += 1

        task = (
            "pgn", text, counter,
            args.seed * 1000003 + counter,
            args.positions_per_game, args.skip_plies, args.castle_keep,
        )

        yield split, task


def puzzle_phase(row):
    """Fase do puzzle: tema opening/middlegame/endgame (fallback: posicao)."""

    themes = row["Themes"].split()

    for i, name in enumerate(PHASE_NAMES):
        if name in themes:
            return i

    return raw_phase(chess.Board(row["FEN"]))


def puzzle_items(args, ratios, rng, stats):
    """
    Split 75/12,5/12,5 POR FASE: dentro de opening, de middlegame e de
    endgame, cada uma tem suas proprias cotas train/val/test. O total
    (--puzzle-count) e dividido entre as fases por --puzzle-phase-weights.
    """

    w = args.puzzle_phase_weights
    w = [x / sum(w) for x in w]

    use_all = args.puzzle_count is None

    # ALL: sem cota (usa todo puzzle que passar nos filtros); o split
    # 75/12,5/12,5 por fase vem do hash, entao e aproximado.
    quota = {
        (p, s): (
            float("inf") if use_all
            else int(math.ceil(args.puzzle_count * w[p] * ratios[s]))
        )
        for p in range(3) for s in SPLITS
    }

    taken = {k: 0 for k in quota}

    stats["puz_quota"] = {
        f"{PHASE_NAMES[p]}/{s}": (None if use_all else v)
        for (p, s), v in quota.items()
    }
    stats["puz_taken"] = {
        f"{PHASE_NAMES[p]}/{s}": 0 for (p, s) in quota
    }

    def rows():

        with open_text_stream(args.puzzles) as f:

            for row in csv.DictReader(f):
                yield row

    counter = 0

    for row in shuffle_buffer(rows(), args.shuffle_buffer, rng):

        stats["puz_seen"] += 1

        try:

            if not args.puzzle_min_rating <= int(row["Rating"]) <= args.puzzle_max_rating:
                continue

            if int(row["Popularity"]) < args.puzzle_min_popularity:
                continue

            if int(row["NbPlays"]) < args.puzzle_min_plays:
                continue

            phase = puzzle_phase(row)

        except (KeyError, ValueError):
            continue

        if rng.random() > args.puzzle_sample:
            continue

        split = assign_split(
            f"{phase}:{row['PuzzleId']}", args.seed + 1,
            ratios["train"], ratios["val"],
        )

        key = (phase, split)

        if taken[key] >= quota[key]:

            if all(taken[k] >= quota[k] for k in quota):
                return

            continue

        taken[key] += 1
        stats["puz_taken"][f"{PHASE_NAMES[phase]}/{split}"] += 1
        counter += 1

        yield split, (
            "puz", row["FEN"], row["Moves"], row["Themes"],
            PUZZLE_GROUP_OFFSET + counter,
        )


def mixed_items(args, ratios, rng, stats):
    """Mistura PGN e puzzles na proporcao das cotas ate acabarem."""

    sources = []

    if args.pgn and args.games != 0:

        weight = 200_000 if args.games is None else args.games

        sources.append([pgn_items(args, ratios, rng, stats), weight])

    if args.puzzles and args.puzzle_count != 0:

        # peso so para intercalar as fontes; em ALL usa uma estimativa
        weight = 4_000_000 if args.puzzle_count is None else args.puzzle_count

        sources.append([puzzle_items(args, ratios, rng, stats), weight])

    n = 0
    t0 = time.time()

    while sources:

        pick = rng.random() * sum(w for _, w in sources)

        for src in sources:

            pick -= src[1]

            if pick <= 0:
                break

        try:
            item = next(src[0])
        except StopIteration:
            sources.remove(src)
            continue

        n += 1

        if n % 50000 == 0:
            print(
                f"  {n:>9,} tarefas enviadas "
                f"({n / (time.time() - t0):,.0f}/s) | "
                f"pgn lidas {stats['pgn_seen']:,} aceitas "
                f"{stats['pgn_accepted']:,} | puzzles lidos "
                f"{stats['puz_seen']:,}",
                flush=True,
            )

        yield item


# ============================================================
# MAIN
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--out", required=True)
    p.add_argument("--pgn", nargs="*", default=[],
                   help="arquivos .pgn / .pgn.zst / .pgn.gz")
    p.add_argument("--puzzles", default=None,
                   help="lichess_db_puzzle.csv[.zst]")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunk-size", type=int, default=500)

    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--val-ratio", type=float, default=0.125,
                   help="o resto (0,125) vai para test")

    g = p.add_argument_group("PGN")
    g.add_argument("--years", type=int, nargs="*", default=[],
                   help="restringe a estes anos (padrao: todos os anos\n"
                        "encontrados nos arquivos)")
    g.add_argument("--games", type=count_or_all, default=None,
                   help="TOTAL de partidas (train+val+test) ou ALL (padrao)")
    g.add_argument("--split-block", type=int, default=8,
                   help="tamanho do bloco do split por ano (8 = 6/1/1)")
    g.add_argument("--positions-per-game", type=int, default=10)
    g.add_argument("--skip-plies", type=int, default=2)
    g.add_argument("--castle-keep", type=float, default=0.25)
    g.add_argument("--game-sample", type=float, default=1.0,
                   help="prob. de aceitar cada partida que passou nos filtros")
    g.add_argument("--shuffle-buffer", type=int, default=20000)

    g = p.add_argument_group("Puzzles")
    g.add_argument("--puzzle-count", type=count_or_all, default=1200000,
                   help="TOTAL de puzzles (~2 amostras cada) ou ALL para usar\n"
                        "todos que passarem nos filtros (ignora "
                        "--puzzle-phase-weights)")
    g.add_argument("--puzzle-phase-weights", type=float, nargs=3,
                   default=[1, 1, 1], metavar=("OPEN", "MID", "END"),
                   help="fatia do --puzzle-count por fase (normalizado)")
    g.add_argument("--puzzle-sample", type=float, default=1.0)
    g.add_argument("--puzzle-min-rating", type=int, default=600)
    g.add_argument("--puzzle-max-rating", type=int, default=3000)
    g.add_argument("--puzzle-min-popularity", type=int, default=60)
    g.add_argument("--puzzle-min-plays", type=int, default=100)

    args = p.parse_args()

    if not args.pgn and not args.puzzles:
        p.error("informe --pgn e/ou --puzzles")

    return args


def main():

    args = parse_args()

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": 1.0 - args.train_ratio - args.val_ratio,
    }

    assert ratios["test"] > 0

    rng = random.Random(args.seed)

    os.makedirs(args.out, exist_ok=True)

    writers = {s: SplitWriter(os.path.join(args.out, s)) for s in SPLITS}

    stats = {"pgn_seen": 0, "pgn_accepted": 0, "puz_seen": 0}

    t0 = time.time()

    run_stats = run_chunked_pipeline(
        mixed_items(args, ratios, rng, stats),
        worker,
        writers,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    print("\nResumo")

    for s in SPLITS:

        w = writers[s]

        meta = w.close()

        group = np.fromfile(
            os.path.join(args.out, s, "group.bin"), dtype=np.int32
        )

        n_puz = int((group >= PUZZLE_GROUP_OFFSET).sum())

        meta.update({
            "split": s,
            "puzzle_group_offset": PUZZLE_GROUP_OFFSET,
            "rows_pgn": int(len(group) - n_puz),
            "rows_puzzle": n_puz,
            "seed": args.seed,
        })

        with open(os.path.join(args.out, s, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        c = meta["tag_counts"]

        print(
            f"  {s:<5} n={meta['n']:>9,}  pgn={len(group) - n_puz:>9,}  "
            f"puzzle={n_puz:>9,} | "
            + " ".join(f"{t}={c[t]:,}" for t in
                       PHASE_NAMES + MOVE_TAG_NAMES)
        )

    with open(os.path.join(args.out, "dataset_info.json"), "w") as f:
        json.dump({"args": vars(args), "ratios": ratios,
                   "stats": stats, "skipped": run_stats["skipped"],
                   "seconds": time.time() - t0}, f, indent=2)

    if stats.get("pgn_years"):

        print("\nPartidas por ano (train / val / test e % de cada)")

        for year in sorted(stats["pgn_years"]):

            c = stats["pgn_years"][year]
            tot = sum(c.values())

            print(f"  {year:<6} n={tot:>8,}  "
                  + "  ".join(f"{sp}={c[sp]:,} ({100 * c[sp] / tot:.1f}%)"
                              for sp in SPLITS)
                  + ("   <-- poucas partidas: split pode desviar ate 1 bloco"
                     if tot < 2 * args.split_block else ""))

    if "puz_taken" in stats:

        print("\nPuzzles por fase (obtido"
              + ("" if args.puzzle_count is None else "/cota")
              + ") - 75/12,5/12,5 em CADA fase")

        for p in PHASE_NAMES:

            cells = []

            for sp in SPLITS:
                k = f"{p}/{sp}"
                q = stats["puz_quota"][k]
                cells.append(
                    f"{sp}={stats['puz_taken'][k]:,}"
                    + ("" if q is None else f"/{q:,}")
                )

            short = any(
                stats["puz_quota"][f"{p}/{sp}"] is not None
                and stats["puz_taken"][f"{p}/{sp}"] < stats["puz_quota"][f"{p}/{sp}"]
                for sp in SPLITS
            )

            print(f"  {p:<11} " + "  ".join(cells)
                  + ("   <-- faltou puzzle nessa fase; ajuste "
                     "--puzzle-phase-weights ou os filtros" if short else ""))

    print(f"\nTarefas puladas (erro/ilegal): {run_stats['skipped']:,}")
    print(f"Tempo: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()