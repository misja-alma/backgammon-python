"""Position formats, checked against GNU Backgammon 1.08 (strings it wrote or printed)."""
import os
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.game import position_formats as pf  # noqa: E402
from backgammon.game.position import Player, Position  # noqa: E402

START_XGID = "XGID=-b----E-C---eE---c-e----B-:0:0:1:00:0:0:0:0:10"
# A middle-game position with ME (bottom) on roll; the same with OPPONENT on roll; and with
# OPPONENT on roll 5-2, a 2-cube owned by ME, a 7-point match at 3-2.
XGID_ME = "XGID=-a-B--C-B---eD---c-dA---B-:0:0:1:00:0:0:0:0:10"
XGID_OPP = "XGID=-a-B--C-B---eD---c-dA---B-:0:0:-1:00:0:0:0:0:10"
XGID_MATCH = "XGID=-a-B--C-B---eD---c-dA---B-:1:1:-1:52:3:2:0:7:10"
# What gnubg printed after `set xgid` / `set turn`, and after loading our SGF and Snowie files.
GNUBG_IDS = {XGID_ME: "4Dn4AAjjDB4QBg:cAkAAAAAAAAE", XGID_OPP: "jDN4QBjwHHwABA:MAEAAAAAAAAE",
             XGID_MATCH: "jDN4QBjwHHwABA:EYHqACAAGAAE"}
# gnubg's `export position snowietxt` of XGID_MATCH (gnubg had swapped the players: its player on
# roll, "gnubg", sits at the bottom, so it is the mirrored position with ME on roll).
GNUBG_SNOWIE = "7;0;0;0;1;gnubg;misja;0;2;3;2;-1;0;-2;0;0;0;-1;4;0;3;0;0;0;-4;5;0;0;0;-2;0;-3;0;0;-2;0;1;0;5;2;"
GNUBG_SNOWIE_XGID = "XGID=-b---aD-C---dE---b-c--b-A-:1:-1:1:52:2:3:0:7:10"
# gnubg's `save position` of XGID_OPP (after `set turn 0`).
GNUBG_SGF = ("(;FF[4]GM[6]CA[UTF-8]AP[GNU Backgammon:1.08.003]MI[length:0][game:0][ws:0][bs:0]PW[gnubg]PB[misja]\n"
             ";PL[W]AE[a:y]AW[f][f][f][f][h][h][h][m][m][m][m][m][x]"
             "AB[v][v][s][s][s][q][q][l][l][l][l][e][a][a])\n")


def random_saved(rng: random.Random) -> pf.SavedPosition:
    """A random legal position, side on roll, cube, dice and match state."""
    me, opp = [0] * 25, [0] * 25
    for board, other in ((me, opp), (opp, me)):
        for _ in range(rng.randint(0, 15)):
            point = rng.randint(0, 24)
            if point < 24 and other[23 - point]:
                continue
            board[point] += 1
    turn = rng.choice([Player.ME, Player.OPPONENT])
    owner = rng.choice([None, Player.ME, Player.OPPONENT])
    cube = 1 if owner is None else 2 ** rng.randint(1, 6)
    position = pf._position(me, opp, turn, cube, owner)
    dice = rng.choice([None, (rng.randint(1, 6), rng.randint(1, 6))])
    length = rng.choice([0, 0, 5, 11])
    scores = (rng.randint(0, length - 1), rng.randint(0, length - 1)) if length else (0, 0)
    return pf.SavedPosition(position, dice, length, *scores)


def key(saved: pf.SavedPosition):
    p = saved.position
    dice = tuple(sorted(saved.dice)) if saved.dice else None
    return (p.me_checkers, p.opponent_checkers, p.get_turn(), p.cube_value, p.cube_owner, dice,
            saved.match_length, saved.score_me, saved.score_opponent)


class PositionFormatsTest(unittest.TestCase):

    def test_starting_position(self):
        start = Position()
        start.setup_starting_position()
        saved = pf.SavedPosition(start)
        self.assertEqual(START_XGID, pf.to_xgid(saved))
        self.assertEqual("4HPwATDgc/ABMA", pf.position_id(start))
        self.assertEqual(key(saved), key(pf.from_xgid(START_XGID)))

    def test_gnubg_ids_as_gnubg_prints_them(self):
        for xgid, ids in GNUBG_IDS.items():
            saved = pf.from_xgid(xgid)
            self.assertEqual(ids, pf.to_gnubg_id(saved), xgid)
            self.assertEqual(key(saved), key(pf.from_gnubg_id(ids)))

    def test_gnubg_manual_match_id_example(self):
        # Score 2-4 in a 9-point match, player 0 (OPPONENT) holds a 2-cube, player 1 (ME) rolled 52.
        saved = pf.from_gnubg_id("4HPwATDgc/ABMA:QYkqASAAIAAA")
        p = saved.position
        self.assertEqual((2, Player.OPPONENT, Player.ME), (p.cube_value, p.cube_owner, p.get_turn()))
        self.assertEqual(((5, 2), 9, 4, 2), (saved.dice, saved.match_length, saved.score_me, saved.score_opponent))
        self.assertEqual("QYkqASAAIAAA", pf.match_id(saved))

    def test_gnubg_output_formats(self):
        self.assertEqual(GNUBG_SNOWIE_XGID, pf.to_xgid(pf.from_snowie(GNUBG_SNOWIE)))
        self.assertEqual(XGID_OPP, pf.to_xgid(pf.from_sgf(GNUBG_SGF)))
        # gnubg's labelled output also parses.
        labelled = "GNU Backgammon  Position ID: 4Dn4AAjjDB4QBg\n                 Match ID   : cAkAAAAAAAAE"
        self.assertEqual(XGID_ME, pf.to_xgid(pf.from_gnubg_id(labelled)))

    def test_every_format_round_trips(self):
        rng = random.Random(3)
        for _ in range(300):
            saved = random_saved(rng)
            for fmt in pf.TEXT_FORMATS:
                text = pf.to_text(saved, fmt)
                back, detected = pf.from_text(text)
                self.assertEqual(fmt, detected, text)
                self.assertEqual(key(saved), key(back), f"{fmt.label}: {text}")

    def test_xg_files_round_trip(self):
        from backgammon.game import xg_format
        rng = random.Random(5)
        for _ in range(100):
            saved = random_saved(rng)
            self.assertEqual(key(saved), key(xg_format.read_xgp(xg_format.write_xgp(saved, b"\xff\xd8 picture"))))

    def test_xg_template_is_an_xg_position_file(self):
        # Saved by eXtreme Gammon: OPPONENT (player 2) to play 1-2 at double match point (1-point match).
        from backgammon.game import xg_format
        saved = xg_format.read_xgp(xg_format.TEMPLATE.read_bytes())
        self.assertEqual("XGID=----BBB-bA-AcB-b-B-ccbBA--:0:0:-1:12:0:0:0:1:10", pf.to_xgid(saved))
        self.assertEqual((1, 2), saved.dice)

    def test_xg_file_layout(self):
        # What XG expects: the title, the thumbnail size in the header, the files of the archive.
        from backgammon.game import xg_format
        saved = pf.from_xgid(XGID_MATCH)
        raw = xg_format.write_xgp(saved, b"JPEG")
        prefix, files = xg_format.read_archive(raw)
        self.assertEqual(b"JPEG", prefix[xg_format.RICH_HEADER_SIZE:])
        title = prefix[xg_format.TITLE_OFFSET:xg_format.TITLE_OFFSET + 80].decode("utf-16-le").split("\0")[0]
        self.assertEqual("Position: 7 point match 3-2", title)
        records = xg_format.records(files["temp.xg"])
        self.assertEqual([0, 1, 2, 3], [r[8] for r in records])          # match, game, cube, move
        self.assertEqual(files["temp.xgi"], records[0] + records[-1])
        money = pf.SavedPosition(saved.position)                        # a money game: length 99999
        _, money_files = xg_format.read_archive(xg_format.write_xgp(money))
        match_record = xg_format.records(money_files["temp.xg"])[0]
        self.assertEqual(99999, struct.unpack_from("<i", match_record, xg_format.MATCH_LENGTH)[0])
        before_roll = pf.SavedPosition(saved.position)                  # no dice: no move record
        _, files = xg_format.read_archive(xg_format.write_xgp(before_roll))
        self.assertEqual([0, 1, 2], [r[8] for r in xg_format.records(files["temp.xg"])])

    def test_files(self):
        saved = pf.from_xgid(XGID_MATCH)
        with tempfile.TemporaryDirectory() as d:
            for fmt in pf.Format:
                path = os.path.join(d, "p" + fmt.extension)
                pf.save(path, saved, fmt)
                back, detected = pf.load(path)
                self.assertEqual((fmt, key(saved)), (detected, key(back)))

    def test_invalid_input(self):
        for text in ["", "hello", "XGID=-b----E-C---eE---c-e----B-", "4HPwATDgc/AB", "1;2;3",
                     "(;FF[4]GM[6];B[52hgec])", "XGID=-p----E-C---eE---c-e----B-:0:0:1:00:0:0:0:0:10"]:
            with self.assertRaises(pf.FormatError, msg=text):
                pf.from_text(text)


class BoardPositionClipboardTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from backgammon.ui.board import Board
        cls.board = Board()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_paste_sets_up_the_position_and_dice_and_copy_gives_it_back(self):
        board = self.board
        with patch("backgammon.ui.board.paste_from_clipboard", return_value=XGID_MATCH + "\n"):
            board.paste_position()
        self.assertEqual(Player.OPPONENT, board.position.get_turn())
        self.assertEqual((5, 2), board.edit_dice)
        copied = []
        with patch("backgammon.ui.board.copy_to_clipboard", side_effect=copied.append):
            board.copy_position()
        # The match score isn't kept by the app: the copy is a money position.
        self.assertEqual("XGID=-a-B--C-B---eD---c-dA---B-:1:1:-1:52:0:0:0:0:10", copied[0])

    def test_pasting_something_else_changes_nothing(self):
        board = self.board
        board.restore_starting_position()
        with patch("backgammon.ui.board.paste_from_clipboard", return_value="not a position"):
            board.paste_position()
        self.assertEqual(START_XGID, pf.to_xgid(pf.SavedPosition(board.position)))


if __name__ == "__main__":
    unittest.main()
