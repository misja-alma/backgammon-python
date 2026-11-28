"""eXtreme Gammon position files (.xgp): reading, and writing from a template.

The layout follows XG's published file format (extremegammon.com/xgformat.aspx: XG_format.pas
and ZLibArchive.pas), with the offsets as bgsage uses them (~/workspace/bgsage/python/bgsage/
xg_file.py) and checked against position files saved by XG:

    [RichGameHeader, 8232 bytes: 'RGMH', version, header size, thumbnail offset and size, a GUID,
     the title (UTF-16, at byte 2088)] [thumbnail: a JPEG of the board, 192 x 113]
    [the archive's files, zlib-compressed] [their registry, 532 bytes per file, compressed]
    [trailer, 36 bytes: CRC32 of files + registry, file count, version, sizes, flags]

The files: temp.xg, a sequence of 2560-byte records (match header, game header, then a cube record
for the position before the roll and, when the dice are set, a checker play record); temp.xgi, the
first and last record; temp.xgr, rollouts.

Boards (26 signed bytes) are from player 1's side: index 1..24 player 1's points, positive player
1's checkers, negative player 2's; 0 is player 2's bar, 25 player 1's. Player 1 (the user in XG,
at the bottom) is ME. The cube is a signed log2 (0 centred at 1, +n player 1 owns 2^n, -n player
2); the side on roll ("actif") is 1 for player 1, -1 for player 2. A money game has match length
99999 ("unlimited").

Writing starts from a real XG position file (data/xg_position_template.xgp, saved by XG with its
default player names) and replaces the position, side on roll, dice, cube and score, clears the
analysis and rollouts, and puts in a new title and, when given, thumbnail. Only the fields XG needs
to show the position are changed; everything else stays as XG wrote it.
"""
import struct
import zlib
from pathlib import Path
from typing import Optional

from backgammon.game.position import Player

RICH_HEADER_SIZE = 8232
MAGIC_RGMH = 0x484D4752
TITLE_OFFSET = 2088            # the header's title (UTF-16, zero-terminated)
TITLE_CHARS = 1024
THUMBNAIL_SIZE_OFFSET = 20
RECORD = 2560
TRAILER = 36
REGISTRY_ENTRY = 532

HEADER_MATCH, HEADER_GAME, CUBE, MOVE = 0, 1, 2, 3

# Record fields (offsets in the 2560-byte record).
MATCH_LENGTH, MATCH_JACOBY = 92, 101
MONEY_LENGTH = 99999           # XG's match length of a money game ("unlimited")
GAME_SCORE1, GAME_SCORE2, GAME_CRAWFORD = 12, 16, 20
CUBE_ACTIF, CUBE_DOUBLE, CUBE_TAKE, CUBE_VALUE, CUBE_POSITION = 12, 16, 20, 32, 36
CUBE_LEVEL, CUBE_ROLLOUT, CUBE_ANALYZED = 64 + 28, 224, 232
MOVE_POSITION, MOVE_END, MOVE_PLAYED, MOVE_ACTIF, MOVE_DICE, MOVE_CUBE = 9, 35, 68, 64, 100, 108
MOVES = 124                    # the analysis of the move (EngineStructBestMove)
MOVES_BOARD, MOVES_DIE1, MOVES_DIE2, MOVES_LEVEL, MOVES_COUNT = 0, 28, 32, 36, 64
MOVE_ERROR, MOVE_ROLLOUTS, MOVE_ANALYZED = 2312, 2344, 2472

TEMPLATE = Path(__file__).parent / "data" / "xg_position_template.xgp"


class XgError(ValueError):
    pass


# --- the archive --------------------------------------------------------------------------------

def _crc(data: bytes) -> int:
    return struct.unpack("<i", struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF))[0]


def read_archive(raw: bytes) -> tuple[bytes, dict[str, bytes]]:
    """(header and thumbnail, {file name: content}) of an XG file."""
    if len(raw) < RICH_HEADER_SIZE + TRAILER or struct.unpack_from("<I", raw, 0)[0] != MAGIC_RGMH:
        raise XgError("not an eXtreme Gammon file")
    trailer = len(raw) - TRAILER
    crc, count, _, registry_size, data_size, flags = struct.unpack_from("<iiiiii", raw, trailer)
    registry_start = trailer - registry_size
    data_start = registry_start - data_size
    if data_start < RICH_HEADER_SIZE or _crc(raw[data_start:trailer]) != crc:
        raise XgError("damaged eXtreme Gammon file (bad layout or checksum)")
    registry = raw[registry_start:trailer]
    if flags & 0xFF:
        registry = zlib.decompress(registry)
    files = {}
    for i in range(count):
        entry = registry[i * REGISTRY_ENTRY:(i + 1) * REGISTRY_ENTRY]
        name = entry[1:1 + entry[0]].decode("ascii", errors="replace")
        size, stored, start, _ = struct.unpack_from("<iiii", entry, 512)
        blob = raw[data_start + start:data_start + start + stored]
        files[name] = zlib.decompress(blob) if entry[528] == 0 else blob
    return raw[:data_start], files


def write_archive(prefix: bytes, files: dict[str, bytes]) -> bytes:
    blobs, registry = bytearray(), bytearray()
    for name, content in files.items():
        blob = zlib.compress(content)
        entry = bytearray(REGISTRY_ENTRY)
        encoded = name.encode("ascii")
        entry[0] = len(encoded)
        entry[1:1 + len(encoded)] = encoded
        struct.pack_into("<iiii", entry, 512, len(content), len(blob), len(blobs), _crc(content))
        entry[528] = 0          # compressed
        entry[529] = 2          # default compression level
        blobs += blob
        registry += entry
    compressed_registry = zlib.compress(bytes(registry))
    stream = bytes(blobs) + compressed_registry
    trailer = struct.pack("<iiiii", _crc(stream), len(files), 1, len(compressed_registry), len(blobs))
    trailer += bytes([1]) + bytes(TRAILER - len(trailer) - 1)
    return prefix + stream + trailer


def records(stream: bytes) -> list[bytes]:
    return [stream[o:o + RECORD] for o in range(0, len(stream) - RECORD + 1, RECORD)]


# --- boards -------------------------------------------------------------------------------------

def _from_xg_board(board) -> tuple[list[int], list[int]]:
    """(player 1's, player 2's) checkers, gnubg style: points 1..24 at 0..23, the bar at 24."""
    p1, p2 = [0] * 25, [0] * 25
    for i in range(1, 25):
        if board[i] > 0:
            p1[i - 1] = board[i]
        elif board[i] < 0:
            p2[24 - i] = -board[i]
    p1[24], p2[24] = abs(board[25]), abs(board[0])
    return p1, p2


def _to_xg_board(p1: list[int], p2: list[int]) -> list[int]:
    board = [0] * 26
    for i in range(1, 25):
        board[i] = p1[i - 1] if p1[i - 1] else -p2[24 - i]
    board[0], board[25] = -p2[24], p1[24]
    return board


def _flip(board: list[int]) -> list[int]:
    """The board from the other player's side (bgsage's flip_board)."""
    return [-board[25 - i] for i in range(26)]


# --- reading ------------------------------------------------------------------------------------

def read_xgp(raw: bytes):
    """A SavedPosition from an XG position file: the position of its last cube or checker play
    record, with the match length, score and Crawford flag of its headers."""
    from backgammon.game.position_formats import SavedPosition, _position, _check_dice
    _, files = read_archive(raw)
    stream = files.get("temp.xg")
    if not stream:
        raise XgError("the XG file has no game records")
    match_length = score1 = score2 = 0
    crawford = jacoby = False
    last = None
    for rec in records(stream):
        kind = rec[8]
        if kind == HEADER_MATCH:
            match_length = struct.unpack_from("<i", rec, MATCH_LENGTH)[0]
            if match_length >= MONEY_LENGTH or match_length < 0:
                match_length = 0
            jacoby = bool(rec[MATCH_JACOBY])
        elif kind == HEADER_GAME:
            score1, score2 = struct.unpack_from("<ii", rec, GAME_SCORE1)
            crawford = bool(rec[GAME_CRAWFORD])
        elif kind in (CUBE, MOVE):
            last = rec
    if last is None:
        raise XgError("the XG file holds no position")
    if last[8] == MOVE:
        actif = struct.unpack_from("<i", last, MOVE_ACTIF)[0]
        board = struct.unpack_from("<26b", last, MOVE_POSITION)
        dice = _check_dice(struct.unpack_from("<2i", last, MOVE_DICE))
        cube = struct.unpack_from("<i", last, MOVE_CUBE)[0]
    else:
        actif = struct.unpack_from("<i", last, CUBE_ACTIF)[0]
        board = struct.unpack_from("<26b", last, CUBE_POSITION)
        dice = None
        cube = struct.unpack_from("<i", last, CUBE_VALUE)[0]
    me, opponent = _from_xg_board(board)
    owner = None if cube == 0 else (Player.ME if cube > 0 else Player.OPPONENT)
    on_roll = Player.ME if actif == 1 else Player.OPPONENT
    position = _position(me, opponent, on_roll, 1 << abs(cube), owner)
    return SavedPosition(position, dice, match_length, score1, score2, crawford, jacoby and not match_length)


# --- writing ------------------------------------------------------------------------------------

def title(saved) -> str:
    if saved.match_length:
        return f"Position: {saved.match_length} point match {saved.score_me}-{saved.score_opponent}"
    return "Position: money game"


def write_xgp(saved, thumbnail_jpeg: Optional[bytes] = None) -> bytes:
    """An XG position file of saved (see the module docstring), with thumbnail_jpeg (192 x 113)
    as its picture, or the template's when None."""
    from backgammon.game.position_formats import _board, _log2
    template = TEMPLATE.read_bytes()
    prefix, files = read_archive(template)
    header = bytearray(prefix[:RICH_HEADER_SIZE])
    thumbnail = thumbnail_jpeg if thumbnail_jpeg is not None else prefix[RICH_HEADER_SIZE:]
    text = title(saved)[:TITLE_CHARS - 1].encode("utf-16-le")
    header[TITLE_OFFSET:TITLE_OFFSET + 2 * TITLE_CHARS] = text + bytes(2 * TITLE_CHARS - len(text))
    struct.pack_into("<I", header, THUMBNAIL_SIZE_OFFSET, len(thumbnail))

    p = saved.position
    on_roll = p.get_turn()
    actif = 1 if on_roll == Player.ME else -1
    log_cube = _log2(p.cube_value)
    cube = 0 if p.cube_owner is None else (log_cube if p.cube_owner == Player.ME else -log_cube)
    board = _to_xg_board(_board(p, Player.ME), _board(p, Player.OPPONENT))
    mover_board = board if actif == 1 else _flip(board)

    by_kind = {rec[8]: bytearray(rec) for rec in records(files["temp.xg"])}
    match, game, cube_rec, move = by_kind[HEADER_MATCH], by_kind[HEADER_GAME], by_kind[CUBE], by_kind[MOVE]
    struct.pack_into("<i", match, MATCH_LENGTH, saved.match_length or MONEY_LENGTH)
    match[MATCH_JACOBY] = int(saved.jacoby and not saved.match_length)
    struct.pack_into("<ii", game, GAME_SCORE1, saved.score_me, saved.score_opponent)
    game[GAME_CRAWFORD] = int(saved.crawford)

    struct.pack_into("<i", cube_rec, CUBE_ACTIF, actif)
    # Before the roll a cube decision without an action (as XG saves one); with the dice, the
    # template's "no cube decision" before the checker play.
    if saved.dice is None:
        struct.pack_into("<ii", cube_rec, CUBE_DOUBLE, -1, -1)
    struct.pack_into("<i", cube_rec, CUBE_VALUE, cube)
    struct.pack_into("<26b", cube_rec, CUBE_POSITION, *board)
    struct.pack_into("<i", cube_rec, CUBE_LEVEL, -100)        # not analyzed
    struct.pack_into("<i", cube_rec, CUBE_ROLLOUT, -1)
    struct.pack_into("<i", cube_rec, CUBE_ANALYZED, -1)
    out = [match, game, cube_rec]

    if saved.dice is not None:
        die1, die2 = saved.dice
        struct.pack_into("<26b", move, MOVE_POSITION, *board)
        struct.pack_into("<26b", move, MOVE_END, *([0] * 26))
        struct.pack_into("<8i", move, MOVE_PLAYED, *([0] * 8))
        struct.pack_into("<i", move, MOVE_ACTIF, actif)
        struct.pack_into("<2i", move, MOVE_DICE, die1, die2)
        struct.pack_into("<i", move, MOVE_CUBE, cube)
        struct.pack_into("<26b", move, MOVES + MOVES_BOARD, *mover_board)
        struct.pack_into("<ii", move, MOVES + MOVES_DIE1, die1, die2)
        struct.pack_into("<i", move, MOVES + MOVES_COUNT, 0)  # no analyzed moves
        struct.pack_into("<d", move, MOVE_ERROR, -1000.0)     # no error known
        struct.pack_into("<32i", move, MOVE_ROLLOUTS, *([-1] * 32))
        struct.pack_into("<i", move, MOVE_ANALYZED, -1)
        out.append(move)

    stream = b"".join(bytes(r) for r in out)
    new_files = {"temp.xg": stream, "temp.xgr": b"", "temp.xgi": bytes(out[0]) + bytes(out[-1])}
    return write_archive(bytes(header) + thumbnail, new_files)
