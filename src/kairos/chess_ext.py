"""
chess_common_ext.py

Extensao do chess_common.py 

Importe SEMPRE este modulo (e nao o chess_common direto) nos scripts
novos: ele ocupa os 4 bits mortos do campo `tags` (bits 12-15, que o
uint16 ja reservava e o writer ja contava) com o TIPO DO LANCE ALVO:

    bit 12 castle      o lance alvo e roque
    bit 13 enPassant   o lance alvo e en passant
    bit 14 promotion   o lance alvo e promocao
    bit 15 capture     o lance alvo e captura (inclui en passant)

O formato em disco nao muda (action_size continua 1968); datasets
antigos continuam legiveis, apenas sem esses 4 bits.
As mutacoes sao in-place e idempotentes, entao valem tambem nos
workers do multiprocessing (spawn), que reimportam este modulo.
"""

import chess
import numpy as np

from kairos import chess_common as cc
from kairos.chess_common import *  # noqa: F401,F403  (reexporta tudo)

_NEW_TAGS = ["castle", "enPassant", "promotion", "capture"]

if len(cc.TAGS) == 12:
    cc.TAGS.extend(_NEW_TAGS)

for _i, _t in enumerate(cc.TAGS):
    cc.TAG_BIT[_t] = 1 << _i

TAGS = cc.TAGS
TAG_BIT = cc.TAG_BIT
PHASE_NAMES = TAGS[:3]                 # opening, middlegame, endgame
THEME_TAG_NAMES = TAGS[3:12]           # temas de puzzle (multi-label)
MOVE_TAG_NAMES = TAGS[12:16]           # tipo do lance alvo

assert len(TAGS) == 16, TAGS

# group.bin >= este valor  =>  a linha veio de um puzzle
PUZZLE_GROUP_OFFSET = 1_000_000_000

ACTION_FROM = np.array([a[0] for a in cc.ACTIONS], dtype=np.int64)
ACTION_TO = np.array([a[1] for a in cc.ACTIONS], dtype=np.int64)


def move_tags(board, move):
    """Bits 12-15 do lance `move` jogado em `board` (antes do push)."""

    bits = 0

    if board.is_castling(move):
        bits |= TAG_BIT["castle"]

    if board.is_en_passant(move):
        bits |= TAG_BIT["enPassant"]

    if move.promotion:
        bits |= TAG_BIT["promotion"]

    if board.is_capture(move):
        bits |= TAG_BIT["capture"]

    return bits