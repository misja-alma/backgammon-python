from typing import List
from enum import Enum


class Player(Enum):
    OPPONENT = 0
    ME = 1

    def other_player(self):
        if self == Player.ME:
            return Player.OPPONENT
        else:
            return Player.ME


class Position:
    def __init__(self):
        # Arrays representing checkers for each player on points 0-25
        # Point 0: off the board (bear off), Points 1-24: board positions, Point 25: on the bar
        # Both players use same indexing: move from high numbers to low numbers, bear off to 0
        self.white_checkers: List[int] = [0] * 26
        self.black_checkers: List[int] = [0] * 26

        # Whose turn it is
        self.turn: Player = Player.ME

        # Doubling cube state
        self.cube_value: int = 1
        self.cube_owner: Player = None  # None means centered, otherwise owned by a player

        self._hash: int | None = None

    def hash_code(self) -> int:
        if self._hash is None:
            self._hash = hash((
                tuple(self.white_checkers),
                tuple(self.black_checkers),
                self.turn,
                self.cube_value,
                self.cube_owner,
            ))
        return self._hash

    def _invalidate_hash(self) -> None:
        self._hash = None

    def set_checkers(self, player: Player, point: int, count: int) -> None:
        """Set the number of checkers for a player on a specific point."""
        if not (0 <= point <= 25):
            raise ValueError("Point must be between 0 and 25")
        if count < 0:
            raise ValueError("Checker count cannot be negative")

        if player == Player.OPPONENT:
            self.white_checkers[point] = count
        else:
            self.black_checkers[point] = count
        self._invalidate_hash()

    def get_checkers(self, player: Player, point: int) -> int:
        """Get the number of checkers for a player on a specific point."""
        if not (0 <= point <= 25):
            raise ValueError("Point must be between 0 and 25")

        if player == Player.OPPONENT:
            return self.white_checkers[point]
        else:
            return self.black_checkers[point]

    def switch_turn(self) -> None:
        """Switch to the other player's turn."""
        self.turn = Player.ME if self.turn == Player.OPPONENT else Player.OPPONENT
        self._invalidate_hash()

    def get_turn(self):
        return self.turn

    def set_turn(self, player: Player):
        self.turn = player
        self._invalidate_hash()

    def double_cube(self, player: Player) -> None:
        """Double the cube value and assign ownership to the player."""
        self.cube_value *= 2
        self.cube_owner = player
        self._invalidate_hash()

    def clear(self) -> None:
        """Move all checkers for both players to point 0 (bear-off area)."""
        for checkers in [self.white_checkers, self.black_checkers]:
            for i in range(1, 26):
                checkers[i] = 0
            checkers[0] = 15
        self._invalidate_hash()

    def setup_starting_position(self) -> None:
        """Set up the standard backgammon starting position."""
        # Clear all positions
        self.white_checkers = [0] * 26
        self.black_checkers = [0] * 26

        # Both players have identical starting positions
        # 2 checkers on point 24
        self.white_checkers[24] = 2
        self.black_checkers[24] = 2

        # 5 checkers on point 13
        self.white_checkers[13] = 5
        self.black_checkers[13] = 5

        # 3 checkers on point 8
        self.white_checkers[8] = 3
        self.black_checkers[8] = 3

        # 5 checkers on point 6
        self.white_checkers[6] = 5
        self.black_checkers[6] = 5

        # Reset game state
        self.turn = Player.ME
        self.cube_value = 1
        self.cube_owner = None
        self._invalidate_hash()