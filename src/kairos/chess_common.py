"""
chess_common.py

Definitions shared by 01_prepare_policy_dataset.py,
01_prepare_tactical_dataset.py, 02_train_policy.py and any duel /
diagnostic script. Keeping them in ONE place guarantees that the
data preparation, the training and the play-time code agree on the
board encoding and on the action ids. This module does not import
torch, so the multiprocessing workers of the prepare scripts stay light.

BOARD ENCODING: 18 planes of 8x8
    0-11  pieces (white P N B R Q K, then black p n b r q k)
    12    side to move (whole plane = 1 when white moves)
    13    white can castle kingside      (whole plane)
    14    white can castle queenside     (whole plane)
    15    black can castle kingside      (whole plane)
    16    black can castle queenside     (whole plane)
    17    en passant target square (1 on the square, only when an
          en passant capture is actually legal)

ACTION SPACE (compact): every (from, to, promotion) that can ever be
legal in chess. 1,792 queen-like/knight-like (from, to) pairs (this
already covers castling, en passant and pawn moves) + 176 promotion
variants = 1,968 ids, instead of the old 64*64*5 = 20,480. The legal
move mask is still needed: not all 1,968 are legal in a given position.

ON-DISK FORMAT (one directory per split, raw binaries + meta.json):
    X.bin            [N, 144] uint8   bit-packed 18x8x8 planes
    y.bin            [N]      int16   target action id
    tags.bin         [N]      uint16  bit mask over TAGS (see below)
    group.bin        [N]      int32   source game / puzzle index
    legal_moves.bin  [M]      int16   legal action ids, CSR layout
    legal_offsets.bin[N+1]    int64
The raw layout lets the trainer np.memmap everything: no RAM copy of
the dataset, and DataLoader workers only need the directory path.

TAGS: bit i of `tags` = TAGS[i]. Bits 0-2 are the game phase (exactly
one is set for human positions and for every puzzle); the rest come
from the Lichess puzzle Themes column (puzzles only).
"""

import contextlib
import hashlib
import io
import json
import os
import subprocess
from collections import deque

import chess
import numpy as np


# ============================================================
# BOARD ENCODING
# ============================================================

INPUT_CHANNELS = 18

PIECE_PLANES = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,

    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

PLANE_SIDE_TO_MOVE = 12

# (color, kingside?) -> plane
PLANE_CASTLING = {
    (chess.WHITE, True): 13,
    (chess.WHITE, False): 14,
    (chess.BLACK, True): 15,
    (chess.BLACK, False): 16,
}

PLANE_EN_PASSANT = 17

PACKED_BYTES = INPUT_CHANNELS * 64 // 8  # 144


def board_to_planes(board):

    planes = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.uint8)

    for square, piece in board.piece_map().items():

        plane = PIECE_PLANES[(piece.color, piece.piece_type)]

        planes[
            plane,
            chess.square_rank(square),
            chess.square_file(square)
        ] = 1

    if board.turn == chess.WHITE:
        planes[PLANE_SIDE_TO_MOVE, :, :] = 1

    for (color, kingside), plane in PLANE_CASTLING.items():

        if kingside:
            has_right = board.has_kingside_castling_rights(color)
        else:
            has_right = board.has_queenside_castling_rights(color)

        if has_right:
            planes[plane, :, :] = 1

    if board.has_legal_en_passant():

        square = board.ep_square

        planes[
            PLANE_EN_PASSANT,
            chess.square_rank(square),
            chess.square_file(square)
        ] = 1

    return planes


def pack_planes(planes):
    """[18, 8, 8] uint8 (0/1) -> [144] uint8 (bit-packed, MSB first)."""
    return np.packbits(planes.reshape(-1))


def board_to_packed(board):
    return pack_planes(board_to_planes(board))


# ============================================================
# COMPACT ACTION SPACE
# ============================================================

PROMO_CODE = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

_PROMO_PIECES = [
    None,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
]


def _build_action_space():

    actions = []

    # Plain moves: every (from, to) a queen or a knight could make on
    # an empty board. Covers castling (king two squares along a rank),
    # en passant (one diagonal step) and every pawn push/capture.
    for f in range(64):

        f_file, f_rank = f & 7, f >> 3

        for t in range(64):

            if t == f:
                continue

            d_file = abs((t & 7) - f_file)
            d_rank = abs((t >> 3) - f_rank)

            if (
                d_file == 0
                or d_rank == 0
                or d_file == d_rank
                or (d_file, d_rank) in ((1, 2), (2, 1))
            ):
                actions.append((f, t, 0))

    # Promotions: white pawn on rank 7 / black pawn on rank 2, push or
    # capture, four promotion pieces each.
    for f in range(64):

        f_file, f_rank = f & 7, f >> 3

        if f_rank == 6:

            targets = [f + 8]

            if f_file > 0:
                targets.append(f + 7)

            if f_file < 7:
                targets.append(f + 9)

        elif f_rank == 1:

            targets = [f - 8]

            if f_file > 0:
                targets.append(f - 9)

            if f_file < 7:
                targets.append(f - 7)

        else:
            continue

        for t in targets:
            for promo in (1, 2, 3, 4):
                actions.append((f, t, promo))

    lookup = [-1] * (64 * 64 * 5)

    for action_id, (f, t, promo) in enumerate(actions):
        lookup[f * 320 + t * 5 + promo] = action_id

    return actions, lookup


ACTIONS, _LOOKUP = _build_action_space()

ACTION_SIZE = len(ACTIONS)


def encode_move(move):

    action_id = _LOOKUP[
        move.from_square * 320
        + move.to_square * 5
        + PROMO_CODE[move.promotion]
    ]

    if action_id < 0:
        raise ValueError(f"Move outside the action space: {move}")

    return action_id


def decode_action(action_id):

    f, t, promo = ACTIONS[action_id]

    return chess.Move(f, t, _PROMO_PIECES[promo])


def legal_action_ids(board):
    return [encode_move(move) for move in board.legal_moves]


# ============================================================
# TAGS AND GAME PHASE
# ============================================================

TAGS = [
    "opening",
    "middlegame",
    "endgame",
    "mate",
    "advantage",
    "equality",
    "long",
    "master",
    "masterVsMaster",
    "oneMove",
    "sacrifice",
    "short",
]

TAG_BIT = {tag: 1 << i for i, tag in enumerate(TAGS)}

PHASE_NAMES = TAGS[:3]


def raw_phase(board):
    """
    0 opening, 1 middlegame, 2 endgame -- from the position alone.

    Approximation of the Lichess "Divider" (which the puzzle phase
    tags come from): endgame when 6 or fewer knights/bishops/rooks/
    queens remain, middlegame when 10 or fewer remain or a back rank
    is sparse. The real Divider also uses a "mixedness" score that
    is NOT reproduced here, so borders differ by a few plies.
    """

    minors_majors = chess.popcount(
        board.knights | board.bishops | board.rooks | board.queens
    )

    if minors_majors <= 6:
        return 2

    if minors_majors <= 10:
        return 1

    white_back = chess.popcount(
        board.occupied_co[chess.WHITE] & chess.BB_RANK_1
    )

    black_back = chess.popcount(
        board.occupied_co[chess.BLACK] & chess.BB_RANK_8
    )

    if white_back < 4 or black_back < 4:
        return 1

    return 0


# ============================================================
# HASHING / SPLITS
# ============================================================

def hash01(text):
    """Deterministic pseudo-random float in [0, 1) from a string."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big") / float(2 ** 64)


def assign_split(key, seed, train_ratio, val_ratio):

    bucket = hash01(f"{seed}:split:{key}")

    if bucket < train_ratio:
        return "train"

    if bucket < train_ratio + val_ratio:
        return "val"

    return "test"


# ============================================================
# READING TEXT FILES (.gz / .zst / plain)
# ============================================================

@contextlib.contextmanager
def open_text_stream(path):
    """Opens .zst, .gz or plain text. Downloads nothing."""

    lower = str(path).lower()

    if lower.endswith(".zst"):

        try:
            import zstandard as zstd
        except ImportError:
            zstd = None

        if zstd is not None:

            fh = open(path, "rb")

            try:
                # Lichess dumps use a very long window; the default
                # decoder limit rejects them.
                dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)

                yield io.TextIOWrapper(
                    dctx.stream_reader(fh),
                    encoding="utf-8",
                    errors="replace"
                )

            finally:
                fh.close()

            return

        try:
            proc = subprocess.Popen(
                ["zstd", "-dc", "--long=31", str(path)],
                stdout=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Cannot read .zst: install the 'zstandard' Python "
                "package (pip install zstandard) or the 'zstd' "
                "command-line tool, or decompress the file first."
            ) from exc

        stream = io.TextIOWrapper(
            proc.stdout,
            encoding="utf-8",
            errors="replace"
        )

        try:
            yield stream
        finally:
            stream.close()
            proc.wait()

        return

    if lower.endswith(".gz"):

        import gzip

        with gzip.open(
            path, "rt", encoding="utf-8", errors="replace"
        ) as f:
            yield f

        return

    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        yield f


# ============================================================
# ON-DISK SPLIT: WRITER
# ============================================================

_DTYPES = {
    "X": np.uint8,
    "y": np.int16,
    "tags": np.uint16,
    "group": np.int32,
    "legal_moves": np.int16,
    "legal_offsets": np.int64,
}


class SplitWriter:
    """
    Appends chunks to the raw binaries of one split. A chunk is a dict:
        X [n,144] uint8, y [n], tags [n], group [n],
        legal_flat [sum(lengths)], legal_lengths [n]
    (extra keys such as 'skipped' are ignored here).
    """

    def __init__(self, directory):

        self.directory = directory

        os.makedirs(directory, exist_ok=True)

        self._files = {
            name: open(os.path.join(directory, name + ".bin"), "wb")
            for name in _DTYPES
        }

        self.n = 0
        self.total_legal = 0
        self.tag_counts = np.zeros(16, dtype=np.int64)

    def _write(self, name, array):

        np.ascontiguousarray(
            array, dtype=_DTYPES[name]
        ).tofile(self._files[name])

    def add(self, chunk):

        n = len(chunk["y"])

        if n == 0:
            return

        lengths = np.asarray(chunk["legal_lengths"], dtype=np.int64)

        starts = self.total_legal + np.cumsum(lengths) - lengths

        self._write("X", chunk["X"])
        self._write("y", chunk["y"])
        self._write("tags", chunk["tags"])
        self._write("group", chunk["group"])
        self._write("legal_moves", chunk["legal_flat"])
        self._write("legal_offsets", starts)

        tags = np.asarray(chunk["tags"], dtype=np.int64)

        for bit in range(16):
            self.tag_counts[bit] += int(((tags >> bit) & 1).sum())

        self.n += n
        self.total_legal += int(lengths.sum())

    def close(self, extra=None):

        np.array([self.total_legal], dtype=np.int64).tofile(
            self._files["legal_offsets"]
        )

        for f in self._files.values():
            f.close()

        meta = {
            "n": self.n,
            "total_legal": self.total_legal,
            "packed_bytes": PACKED_BYTES,
            "input_channels": INPUT_CHANNELS,
            "action_size": ACTION_SIZE,
            "tags": TAGS,
            "tag_counts": {
                tag: int(self.tag_counts[i])
                for i, tag in enumerate(TAGS)
            },
        }

        if extra:
            meta.update(extra)

        with open(
            os.path.join(self.directory, "meta.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return meta


# ============================================================
# ON-DISK SPLIT: READER (memory-mapped, picklable, worker-safe)
# ============================================================

class SplitData:

    def __init__(self, directory):

        self.directory = directory

        with open(
            os.path.join(directory, "meta.json"), "r", encoding="utf-8"
        ) as f:
            self.meta = json.load(f)

        self.n = int(self.meta["n"])

        if int(self.meta["action_size"]) != ACTION_SIZE:
            raise RuntimeError(
                f"{directory} was built with another action space "
                f"({self.meta['action_size']} vs {ACTION_SIZE}). "
                f"Re-run the prepare scripts."
            )

        self._arrays = None

    def __getstate__(self):

        state = self.__dict__.copy()
        state["_arrays"] = None  # each worker re-opens its own memmaps

        return state

    def _open(self):

        if self._arrays is None:

            if self.n == 0:
                raise RuntimeError(f"{self.directory} is empty.")

            d = self.directory
            n = self.n
            total = int(self.meta["total_legal"])

            def mm(name, shape):
                return np.memmap(
                    os.path.join(d, name + ".bin"),
                    dtype=_DTYPES[name],
                    mode="r",
                    shape=shape
                )

            self._arrays = {
                "X": mm("X", (n, PACKED_BYTES)),
                "y": mm("y", (n,)),
                "tags": mm("tags", (n,)),
                "group": mm("group", (n,)),
                "legal_moves": mm("legal_moves", (total,)),
                "legal_offsets": mm("legal_offsets", (n + 1,)),
            }

        return self._arrays

    def tags_array(self):
        return self._open()["tags"]

    def gather(self, idx):
        """
        Batch of positions `idx` (array of int). Legal moves come back
        as one flat int16 array plus per-batch offsets (CSR), built
        with a vectorized gather (no Python loop per position).
        """

        a = self._open()

        idx = np.asarray(idx, dtype=np.int64)

        offsets = a["legal_offsets"]

        starts = np.asarray(offsets[idx])
        ends = np.asarray(offsets[idx + 1])
        lengths = ends - starts

        batch_offsets = np.zeros(len(idx) + 1, dtype=np.int64)
        batch_offsets[1:] = np.cumsum(lengths)

        total = int(batch_offsets[-1])

        flat_index = (
            np.arange(total, dtype=np.int64)
            + np.repeat(starts - batch_offsets[:-1], lengths)
        )

        return {
            "X": np.asarray(a["X"][idx]),
            "y": np.asarray(a["y"][idx]).astype(np.int64),
            "tags": np.asarray(a["tags"][idx]).astype(np.int32),
            "group": np.asarray(a["group"][idx]),
            "legal_flat": np.asarray(a["legal_moves"][flat_index]),
            "legal_offsets": batch_offsets,
        }


# ============================================================
# CHUNKED MULTIPROCESSING PIPELINE
# ============================================================

def run_chunked_pipeline(
    items,
    worker,
    writers,
    workers=1,
    chunk_size=1000
):
    """
    items:   iterable of (split_name, task), consumed lazily in the
             main process (so the whole input is never held in RAM).
    worker:  top-level picklable function, list_of_tasks -> chunk dict
             (see SplitWriter.add) plus an optional 'skipped' count.
    writers: {split_name: SplitWriter}

    Tasks are grouped per split, dispatched to a process pool in
    chunks, and the results are written in submission order.
    """

    import multiprocessing as mp

    buffers = {name: [] for name in writers}
    pending = deque()
    stats = {"skipped": 0}

    pool = (
        mp.get_context("spawn").Pool(workers)
        if workers > 1
        else None
    )

    max_pending = max(2, 3 * workers)

    def collect(split, out):
        writers[split].add(out)
        stats["skipped"] += int(out.get("skipped", 0))

    def submit(split):

        chunk = buffers[split]
        buffers[split] = []

        if not chunk:
            return

        if pool is None:
            collect(split, worker(chunk))
            return

        pending.append(
            (split, pool.apply_async(worker, (chunk,)))
        )

        while len(pending) > max_pending:
            s, result = pending.popleft()
            collect(s, result.get())

    try:

        for split, task in items:

            buffers[split].append(task)

            if len(buffers[split]) >= chunk_size:
                submit(split)

        for split in list(buffers):
            submit(split)

        while pending:
            s, result = pending.popleft()
            collect(s, result.get())

    except BaseException:

        if pool is not None:
            pool.terminate()

        raise

    else:

        if pool is not None:
            pool.close()

    finally:

        if pool is not None:
            pool.join()

    return stats


def empty_chunk():

    return {
        "X": np.zeros((0, PACKED_BYTES), dtype=np.uint8),
        "y": np.zeros(0, dtype=np.int16),
        "tags": np.zeros(0, dtype=np.uint16),
        "group": np.zeros(0, dtype=np.int32),
        "legal_flat": np.zeros(0, dtype=np.int16),
        "legal_lengths": np.zeros(0, dtype=np.int64),
        "skipped": 0,
    }


def rows_to_chunk(X, y, tags, group, legal, lengths, skipped):

    if not y:
        chunk = empty_chunk()
        chunk["skipped"] = skipped
        return chunk

    return {
        "X": np.stack(X).astype(np.uint8),
        "y": np.asarray(y, dtype=np.int16),
        "tags": np.asarray(tags, dtype=np.uint16),
        "group": np.asarray(group, dtype=np.int32),
        "legal_flat": np.asarray(legal, dtype=np.int16),
        "legal_lengths": np.asarray(lengths, dtype=np.int64),
        "skipped": skipped,
    }


# ============================================================
# SELF-TEST:  python chess_common.py
# ============================================================

def _selftest():

    import random

    print(f"Action space: {ACTION_SIZE} ids (expected 1968)")

    assert ACTION_SIZE == 1968, ACTION_SIZE

    rng = random.Random(0)

    positions = 0

    for _ in range(40):

        board = chess.Board()

        for _ in range(120):

            if board.is_game_over():
                break

            moves = list(board.legal_moves)

            ids = [encode_move(m) for m in moves]

            assert len(set(ids)) == len(ids)

            for move, action_id in zip(moves, ids):
                assert decode_action(action_id) == move

            planes = board_to_planes(board)

            assert planes.shape == (INPUT_CHANNELS, 8, 8)

            restored = np.unpackbits(
                pack_planes(planes)
            ).reshape(INPUT_CHANNELS, 8, 8)

            assert np.array_equal(planes, restored)

            positions += 1

            board.push(rng.choice(moves))

    # castling rights + en passant planes
    board = chess.Board()

    planes = board_to_planes(board)

    assert planes[13:17].all() and not planes[17].any()

    for san in ["e4", "a6", "e5", "d5"]:
        board.push_san(san)

    planes = board_to_planes(board)

    assert planes[17, 5, 3] == 1 and planes[17].sum() == 1

    print(f"OK: {positions} random positions checked")


if __name__ == "__main__":
    _selftest()