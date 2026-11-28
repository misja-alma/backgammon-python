"""Saving and loading positions in the formats of other backgammon programs.

Formats (encodings as in GNU Backgammon's positionid.c, matchid.c, set.c (XGID), import.c and
export.c (Snowie text), sgf.c; see ~/workspace/gnubg):

- XGID: eXtreme Gammon's position string, also what XG, gnubg and most other programs copy to and
  paste from the clipboard: `XGID=-b----E-C---eE---c-e----B-:0:0:1:00:0:0:0:0:10`.
- GNUbg ID: GNU Backgammon's Position ID and Match ID (base64 keys), `4HPwATDgc/ABMA:cAkAAAAAAAAA`.
- Snowie text (.txt): one line of 40 semicolon-separated fields, from the player on roll's view.
- GNU Backgammon (.sgf): a game record that sets up the position (gnubg's "export position").
- eXtreme Gammon (.xgp): XG's binary position file (see xg_format.py).

Players: ME (white, at the bottom, home board bottom right) is the bottom player of XG and
gnubg's player 1; OPPONENT is the top player, gnubg's player 0. Points are numbered from each
player's own side, as in Position (1..24, 0 borne off, 25 the bar).

Besides the checkers, the side on roll, the cube and the dice, the formats hold the match length
and score; they are kept in a SavedPosition (money game, 0-0, by default) so they survive loading
and saving, but the app itself only uses the position, cube and dice.
"""
import base64
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from backgammon.game.position import Player, Position
from backgammon.game.settings import APP_NAME

CHECKERS = 15


@dataclass
class SavedPosition:
    position: Position                       # checkers, side on roll, cube
    dice: Optional[tuple[int, int]] = None   # None: before the roll
    match_length: int = 0                    # 0: money game
    score_me: int = 0
    score_opponent: int = 0
    crawford: bool = False
    jacoby: bool = False


class Format(Enum):
    XGID = ("XGID", ".txt")
    GNUBG_ID = ("GNUbg ID", ".txt")
    SNOWIE = ("Snowie text", ".txt")
    SGF = ("GNU Backgammon", ".sgf")
    XG = ("eXtreme Gammon", ".xgp")       # binary: files only, see xg_format.py

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def extension(self) -> str:
        return self.value[1]


class FormatError(ValueError):
    """Text or a file that isn't a valid position in any supported format."""


# --- helpers ------------------------------------------------------------------------------------

def _board(position: Position, player: Player) -> list[int]:
    """The player's checkers on points 1..24 (index 0..23) and the bar (index 24), gnubg style."""
    checkers = position.me_checkers if player == Player.ME else position.opponent_checkers
    return [checkers[point] for point in range(1, 25)] + [checkers[25]]


def _position(me: list[int], opponent: list[int], turn: Player, cube_value: int = 1,
              cube_owner: Optional[Player] = None) -> Position:
    """A Position from gnubg-style boards (points 1..24 at index 0..23, the bar at 24)."""
    for board in (me, opponent):
        if any(c < 0 for c in board) or sum(board) > CHECKERS:
            raise FormatError("more than 15 checkers for a player")
    if any(me[i] and opponent[23 - i] for i in range(24)):
        raise FormatError("a point holds checkers of both players")
    p = Position()
    p.me_checkers = [CHECKERS - sum(me)] + me[:24] + [me[24]]
    p.opponent_checkers = [CHECKERS - sum(opponent)] + opponent[:24] + [opponent[24]]
    p.set_turn(turn)
    p.set_cube(cube_value, cube_owner)
    return p


def _check_dice(dice) -> Optional[tuple[int, int]]:
    if dice is None or dice == (0, 0):
        return None
    if not all(1 <= d <= 6 for d in dice):
        raise FormatError(f"invalid dice {dice}")
    return dice


def _log2(cube: int) -> int:
    if cube < 1 or cube & (cube - 1):
        raise FormatError(f"cube value {cube} is not a power of 2")
    return cube.bit_length() - 1


# --- XGID ---------------------------------------------------------------------------------------

def to_xgid(saved: SavedPosition) -> str:
    """`XGID=<checkers>:<cube>:<owner>:<turn>:<dice>:<score bottom>:<score top>:<rules>:<match
    length>:<max cube>`. Checkers: 26 characters, the top player's bar, the points 1..24 from
    the bottom player's side, the bottom player's bar; '-' empty, 'A'.. the bottom player's
    count, 'a'.. the top player's."""
    p = saved.position
    me, opp = _board(p, Player.ME), _board(p, Player.OPPONENT)
    chars = [chr(ord('a') + opp[24] - 1) if opp[24] else '-']
    for point in range(1, 25):
        if me[point - 1]:
            chars.append(chr(ord('A') + me[point - 1] - 1))
        elif opp[24 - point]:
            chars.append(chr(ord('a') + opp[24 - point] - 1))
        else:
            chars.append('-')
    chars.append(chr(ord('A') + me[24] - 1) if me[24] else '-')
    owner = {None: 0, Player.ME: 1, Player.OPPONENT: -1}[p.cube_owner]
    turn = 1 if p.get_turn() == Player.ME else -1
    dice = f"{saved.dice[0]}{saved.dice[1]}" if saved.dice else "00"
    rules = int(saved.crawford) if saved.match_length else int(saved.jacoby)
    return (f"XGID={''.join(chars)}:{_log2(p.cube_value)}:{owner}:{turn}:{dice}:"
            f"{saved.score_me}:{saved.score_opponent}:{rules}:{saved.match_length}:10")


def from_xgid(text: str) -> SavedPosition:
    match = re.search(r"(?:XGID=)?([-a-pA-P]{26})((?::[-\w]+){9})", text.strip())
    if not match:
        raise FormatError("not an XGID")
    chars, fields = match.group(1), match.group(2)[1:].split(":")
    me, opp = [0] * 25, [0] * 25
    for i, c in enumerate(chars):
        count = ord(c.lower()) - ord('a') + 1 if c != '-' else 0
        if c == '-':
            continue
        if c.isupper():                       # the bottom player: ME
            if i == 0:
                raise FormatError("XGID: the bottom player's checkers on the top player's bar")
            me[24 if i == 25 else i - 1] = count
        else:                                  # the top player: OPPONENT
            if i == 25:
                raise FormatError("XGID: the top player's checkers on the bottom player's bar")
            opp[24 if i == 0 else 24 - i] = count
    try:
        cube_log, owner, turn = int(fields[0]), int(fields[1]), int(fields[2])
        dice_field = fields[3]
        score_me, score_opp, rules, length = int(fields[4]), int(fields[5]), int(fields[6]), int(fields[7])
    except ValueError:
        raise FormatError("XGID: invalid number")
    turn_player = Player.ME if turn == 1 else Player.OPPONENT
    if dice_field == "D":                      # a double offered: the roll hasn't happened
        dice = None
    elif re.fullmatch(r"[0-6]{2}", dice_field):
        dice = _check_dice((int(dice_field[0]), int(dice_field[1])))
    else:
        raise FormatError(f"XGID: invalid dice {dice_field}")
    owner_player = {0: None, 1: Player.ME, -1: Player.OPPONENT}.get(owner)
    position = _position(me, opp, turn_player, 1 << cube_log, owner_player)
    return SavedPosition(position, dice, length, score_me, score_opp,
                         crawford=bool(length and rules == 1), jacoby=bool(not length and rules & 1))


# --- GNUbg Position ID and Match ID -------------------------------------------------------------

def _bits_to_bytes(bits: list[int], n_bytes: int) -> bytes:
    """Little-endian packing: bit k goes to byte k // 8, bit k % 8."""
    out = bytearray(n_bytes)
    for k, bit in enumerate(bits):
        if bit:
            out[k // 8] |= 1 << (k % 8)
    return bytes(out)


def _bytes_to_bits(data: bytes) -> list[int]:
    return [(data[k // 8] >> (k % 8)) & 1 for k in range(8 * len(data))]


def _decode_base64(text: str, n_bytes: int) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    try:
        return base64.b64decode(padded, validate=True)[:n_bytes]
    except ValueError:
        raise FormatError("invalid base64")


def position_id(position: Position) -> str:
    """gnubg's Position ID (14 characters): for the player not on roll and then the player on
    roll, per point 1..24 and the bar a 1 per checker and a 0; 80 bits, base64."""
    on_roll = position.get_turn()
    bits = []
    for player in (on_roll.other_player(), on_roll):
        for count in _board(position, player):
            bits += [1] * count + [0]
    return base64.b64encode(_bits_to_bytes(bits, 10)).decode()[:14]


def boards_from_position_id(text: str) -> tuple[list[int], list[int]]:
    """(board of the player on roll, board of the other player) of a Position ID."""
    if not re.fullmatch(r"[A-Za-z0-9+/]{14}", text):
        raise FormatError("not a Position ID")
    boards, current, point = [[0] * 25, [0] * 25], 0, 0
    for bit in _bytes_to_bits(_decode_base64(text, 10)):
        if current == 2:
            break
        if bit:
            boards[current][point] += 1
        else:
            point += 1
            if point == 25:
                current, point = current + 1, 0
    not_on_roll, on_roll = boards
    return on_roll, not_on_roll


def _set_bits(bits: list[int], start: int, count: int, value: int) -> None:
    for i in range(count):
        bits[start + i] = (value >> i) & 1


def _get_bits(bits: list[int], start: int, count: int) -> int:
    return sum(bits[start + i] << i for i in range(count))


def match_id(saved: SavedPosition) -> str:
    """gnubg's Match ID (12 characters): cube, cube owner, player on roll, Crawford, game state,
    turn, double and resignation flags, dice, match length and score in a 66-bit key, base64.
    Players: 0 = OPPONENT, 1 = ME."""
    p = saved.position
    bits = [0] * 72
    player = {Player.OPPONENT: 0, Player.ME: 1}
    on_roll = player[p.get_turn()]
    _set_bits(bits, 0, 4, _log2(p.cube_value))
    _set_bits(bits, 4, 2, 3 if p.cube_owner is None else player[p.cube_owner])
    _set_bits(bits, 6, 1, on_roll)                         # the player on roll (fMove)
    _set_bits(bits, 7, 1, int(saved.crawford))
    _set_bits(bits, 8, 3, 1)                               # game state: playing
    _set_bits(bits, 11, 1, on_roll)                        # whose turn it is (fTurn)
    if saved.dice:
        _set_bits(bits, 15, 3, max(saved.dice))
        _set_bits(bits, 18, 3, min(saved.dice))
    _set_bits(bits, 21, 15, saved.match_length)
    _set_bits(bits, 36, 15, saved.score_opponent)
    _set_bits(bits, 51, 15, saved.score_me)
    _set_bits(bits, 66, 1, int(not saved.jacoby))
    return base64.b64encode(_bits_to_bytes(bits, 9)).decode()


def to_gnubg_id(saved: SavedPosition) -> str:
    return f"{position_id(saved.position)}:{match_id(saved)}"


def from_gnubg_id(text: str) -> SavedPosition:
    """`PositionID:MatchID`, `Position ID: ... Match ID: ...` (gnubg's own output) or a Position
    ID alone (then ME is on roll, before the roll, money game)."""
    text = text.strip()
    labelled = re.search(r"Position ID:\s*([A-Za-z0-9+/]{14})(?:.*?Match ID:\s*([A-Za-z0-9+/]{12}))?", text, re.S)
    plain = re.fullmatch(r"(?:GNUbg ID:\s*)?([A-Za-z0-9+/]{14})(?::([A-Za-z0-9+/]{12}))?", text)
    found = labelled or plain
    if not found:
        raise FormatError("not a GNUbg ID")
    pos_id, m_id = found.group(1), found.group(2)
    on_roll_board, other_board = boards_from_position_id(pos_id)
    if m_id is None:
        return SavedPosition(_position(on_roll_board, other_board, Player.ME))
    bits = _bytes_to_bits(_decode_base64(m_id, 9))
    players = {0: Player.OPPONENT, 1: Player.ME}
    on_roll = players[_get_bits(bits, 6, 1)]
    owner_bits = _get_bits(bits, 4, 2)
    owner = None if owner_bits == 3 else players.get(owner_bits)
    dice = _check_dice((_get_bits(bits, 15, 3), _get_bits(bits, 18, 3)))
    if on_roll == Player.ME:
        me, opp = on_roll_board, other_board
    else:
        me, opp = other_board, on_roll_board
    position = _position(me, opp, on_roll, 1 << _get_bits(bits, 0, 4), owner)
    return SavedPosition(position, dice, match_length=_get_bits(bits, 21, 15),
                         score_opponent=_get_bits(bits, 36, 15), score_me=_get_bits(bits, 51, 15),
                         crawford=bool(_get_bits(bits, 7, 1)), jacoby=not _get_bits(bits, 66, 1))


# --- Snowie text --------------------------------------------------------------------------------

def to_snowie(saved: SavedPosition) -> str:
    """40 fields: match length; Jacoby; 0; 1 for money; player on roll (ME = 1); the names (on
    roll first); Crawford; the scores (on roll first); cube value; cube owner (1 on roll, -1 the
    other, 0 centred); the other player's bar (negative); points 1..24 from the side on roll
    (positive its checkers, negative the other's); the bar of the side on roll; the dice (0 0
    before the roll). As gnubg's export."""
    p = saved.position
    on_roll = p.get_turn()
    other = on_roll.other_player()
    mine, theirs = _board(p, on_roll), _board(p, other)
    name = {Player.ME: "ME", Player.OPPONENT: "OPPONENT"}
    score = {Player.ME: saved.score_me, Player.OPPONENT: saved.score_opponent}
    owner = 0 if p.cube_owner is None else (1 if p.cube_owner == on_roll else -1)
    fields = [saved.match_length, int(saved.jacoby and not saved.match_length), 0,
              0 if saved.match_length else 1, 1 if on_roll == Player.ME else 0,
              name[on_roll], name[other], int(saved.crawford), score[on_roll], score[other],
              p.cube_value, owner, -theirs[24]]
    for i in range(24):
        fields.append(mine[i] if mine[i] else -theirs[23 - i])
    fields += [mine[24], *(saved.dice or (0, 0))]
    return ";".join(str(f) for f in fields) + ";"


def from_snowie(text: str) -> SavedPosition:
    fields = text.strip().split(";")
    if len(fields) < 40:
        raise FormatError("not a Snowie text position (needs 40 fields)")
    try:
        numbers = {i: int(fields[i]) for i in range(40) if i not in (5, 6)}
    except ValueError:
        raise FormatError("Snowie text: invalid number")
    on_roll = Player.ME if numbers[4] == 1 else Player.OPPONENT
    mine, theirs = [0] * 25, [0] * 25
    for i in range(24):
        v = numbers[13 + i]
        if v > 0:
            mine[i] = v
        elif v < 0:
            theirs[23 - i] = -v
    # The bars: gnubg writes the other player's (negative) first and the player on roll's last;
    # its reader takes the first as the player on roll's. The signs tell them apart.
    for v in (numbers[12], numbers[37]):
        if v > 0:
            mine[24] = v
        elif v < 0:
            theirs[24] = -v
    owner_field = numbers[11]
    owner = None if owner_field == 0 else (on_roll if owner_field == 1 else on_roll.other_player())
    me, opp = (mine, theirs) if on_roll == Player.ME else (theirs, mine)
    position = _position(me, opp, on_roll, numbers[10] or 1, owner)
    score_on_roll, score_other = numbers[8], numbers[9]
    score_me, score_opp = (score_on_roll, score_other) if on_roll == Player.ME else (score_other, score_on_roll)
    return SavedPosition(position, _check_dice((numbers[38], numbers[39])), numbers[0], score_me, score_opp,
                         crawford=bool(numbers[7]), jacoby=bool(numbers[1]))


# --- GNU Backgammon SGF -------------------------------------------------------------------------

def to_sgf(saved: SavedPosition) -> str:
    """A game record that sets up the position, as gnubg's "save position": the side on roll (PL;
    B, gnubg's player 1, is ME, W is OPPONENT), the checkers (AE clears the board; AW White's,
    letter a + point - 1 in its own numbering, y its bar; AB Black's, in White's numbering:
    letter x - (point - 1)), the cube (CV, CP) and the dice (DI). The board doesn't depend on
    who is on roll (checked against gnubg 1.08)."""
    p = saved.position
    colour = {Player.ME: "B", Player.OPPONENT: "W"}
    white, black = _board(p, Player.OPPONENT), _board(p, Player.ME)
    aw = "".join(f"[{chr(ord('a') + i)}]" * white[i] for i in range(25))
    ab = "".join(f"[{'y' if i == 24 else chr(ord('x') - i)}]" * black[i] for i in range(25))
    rules = []
    if saved.crawford:
        rules.append("Crawford:CrawfordGame")
    if saved.jacoby:
        rules.append("Jacoby")
    on_roll = colour[p.get_turn()]
    out = ["(;FF[4]GM[6]CA[UTF-8]AP[" + APP_NAME + "]"
           f"MI[length:{saved.match_length}][game:0][ws:{saved.score_opponent}][bs:{saved.score_me}]"
           "PW[OPPONENT]PB[ME]" + (f"RU[{':'.join(rules)}]" if rules else ""),
           f";PL[{on_roll}]AE[a:y]" + (f"AW{aw}" if aw else "") + (f"AB{ab}" if ab else "")]
    if p.cube_value != 1:
        out.append(f";CV[{p.cube_value}]")
    if p.cube_owner is not None:
        out.append(f";CP[{'b' if p.cube_owner == Player.ME else 'w'}]")
    if saved.dice:
        out.append(f";PL[{on_roll}]DI[{saved.dice[0]}{saved.dice[1]}]")
    return "\n".join(out) + ")\n"


def _sgf_points(values: list[str]) -> list[int]:
    """Point letters (and gnubg's 'a:y' ranges) as indexes 0..24."""
    points = []
    for v in values:
        if ":" in v:
            a, b = v.split(":")
            points += range(ord(a) - ord('a'), ord(b) - ord('a') + 1)
        elif len(v) == 1 and 'a' <= v <= 'y':
            points.append(ord(v) - ord('a'))
    return points


def from_sgf(text: str) -> SavedPosition:
    """The position a gnubg game record sets up (its setup properties PL, AE, AW, AB, CV, CP, DI,
    see to_sgf) in the first game of the file. A record that plays moves (a game or match)
    isn't a position."""
    if not text.lstrip().startswith("(;"):
        raise FormatError("not an SGF file")
    game = text.split("\n(;")[0]
    props = re.findall(r"([A-Z]{1,2})((?:\[(?:[^\]\\]|\\.)*\]\s*)+)", game)
    on_roll = Player.ME
    white, black = [0] * 25, [0] * 25    # OPPONENT's and ME's, each in its own numbering
    cube, owner, dice, setup = 1, None, None, False
    match_length, score_me, score_opp, crawford, jacoby = 0, 0, 0, False, False
    for name, raw in props:
        values = re.findall(r"\[((?:[^\]\\]|\\.)*)\]", raw)
        if name in ("B", "W") and values and values[0] not in ("", "double", "take", "drop"):
            raise FormatError("the SGF file records a game, not a position")
        if name == "PL":
            on_roll = Player.ME if values[0] == "B" else Player.OPPONENT
        elif name == "AE":
            setup = True
            for i in _sgf_points(values):
                white[i] = 0
                black[24 if i == 24 else 23 - i] = 0
        elif name == "AW":
            setup = True
            for i in _sgf_points(values):
                white[i] += 1
        elif name == "AB":
            setup = True
            for i in _sgf_points(values):
                black[24 if i == 24 else 23 - i] += 1
        elif name == "CV":
            cube = int(values[0])
        elif name == "CP":
            owner = {"b": Player.ME, "w": Player.OPPONENT}.get(values[0])
        elif name == "DI" and re.fullmatch(r"[1-6]{2}", values[0]):
            dice = (int(values[0][0]), int(values[0][1]))
        elif name == "MI":
            for v in values:
                key, _, val = v.partition(":")
                if key == "length":
                    match_length = int(val)
                elif key == "ws":
                    score_opp = int(val)
                elif key == "bs":
                    score_me = int(val)
        elif name == "RU":
            crawford = "CrawfordGame" in values[0]
            jacoby = "Jacoby" in values[0]
    if not setup:
        raise FormatError("the SGF file sets up no position")
    position = _position(black, white, on_roll, cube, owner)
    return SavedPosition(position, dice, match_length, score_me, score_opp, crawford, jacoby)


# --- any format ---------------------------------------------------------------------------------

TEXT_FORMATS = [Format.XGID, Format.GNUBG_ID, Format.SNOWIE, Format.SGF]
WRITERS = {Format.XGID: to_xgid, Format.GNUBG_ID: to_gnubg_id, Format.SNOWIE: to_snowie, Format.SGF: to_sgf}


def to_text(saved: SavedPosition, fmt: Format) -> str:
    return WRITERS[fmt](saved)


def detect(text: str) -> Format:
    stripped = text.strip()
    if stripped.startswith("(;"):
        return Format.SGF
    if re.search(r"(?:XGID=)?[-a-pA-P]{26}(?::[-\w]+){9}", stripped):
        return Format.XGID
    if stripped.count(";") >= 39:
        return Format.SNOWIE
    if re.search(r"Position ID:|^(?:GNUbg ID:\s*)?[A-Za-z0-9+/]{14}(?::[A-Za-z0-9+/]{12})?$", stripped):
        return Format.GNUBG_ID
    raise FormatError("not a position in a known format (XGID, GNUbg ID, Snowie text, SGF, XG)")


READERS = {Format.XGID: from_xgid, Format.GNUBG_ID: from_gnubg_id, Format.SNOWIE: from_snowie, Format.SGF: from_sgf}


def from_text(text: str) -> tuple[SavedPosition, Format]:
    """The position in text of any supported format (detected from its content)."""
    fmt = detect(text)
    return READERS[fmt](text), fmt


def save(path, saved: SavedPosition, fmt: Format, thumbnail_jpeg: Optional[bytes] = None) -> None:
    """Write saved to path in fmt. thumbnail_jpeg: the picture of an XG file (192 x 113 JPEG)."""
    if fmt == Format.XG:
        from backgammon.game.xg_format import write_xgp
        Path(path).write_bytes(write_xgp(saved, thumbnail_jpeg))
        return
    Path(path).write_text(to_text(saved, fmt) + ("" if fmt == Format.SGF else "\n"), encoding="utf-8")


def load(path) -> tuple[SavedPosition, Format]:
    """The position in a file of any supported format (detected from its content)."""
    data = Path(path).read_bytes()
    if data[:4] == b"RGMH":
        from backgammon.game.xg_format import XgError, read_xgp
        try:
            return read_xgp(data), Format.XG
        except XgError as e:
            raise FormatError(str(e))
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return from_text(text)
