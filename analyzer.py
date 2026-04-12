from backgammon_position import Position, Player
from utils import Utils
import copy

class Analyzer:
    dice_rolls = []
    for d1 in range(1,7):
        for d2 in range(d1, 7):
            dice_rolls.append((d1, d2))

    @staticmethod
    def find_known_pw(position: Position, player: Player):
        if position.get_checkers(player, 0) == 15:
            return 1.0
        if position.get_checkers(player.other_player(), 0) == 15:
            return 0.0
        return None

    @staticmethod
    def winning_chances(position: Position, player: Player) -> float:
        pw_known = Analyzer.find_known_pw(position, player)
        if pw_known is not None:
            return pw_known

        total = 0
        for die1, die2 in Analyzer.dice_rolls:
            _, pw = Analyzer.best_move(position, die1, die2)
            multiplier = 2 if die1 != die2 else 1
            total += multiplier * pw

        return total / 36 if player == position.get_turn() else 1 - total / 36

    # best move for player on roll:
    # generate all legal moves
    # get winning chances for each for player on roll
    # return the resulting position with highest winning chances
    @staticmethod
    def best_move(position: Position, die1: int, die2: int):
        best = None # best move for player on roll
        for half_moves in Utils.valid_possible_moves(position, die1, die2):
            move = Utils.apply_move(position, half_moves)
            pw = Analyzer.winning_chances(move, position.get_turn())
            if not best:
                best = move, pw
            else:
                _, best_pw = best
                if pw > best_pw:
                    best = move, pw

        if not best:
            # no move, return winning chances after passing
            after_pass = copy.deepcopy(position)
            after_pass.switch_turn()
            pw = Analyzer.winning_chances(after_pass, position.get_turn())
            return after_pass, pw
        else:
            return best



