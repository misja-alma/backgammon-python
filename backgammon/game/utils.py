from backgammon.game.position import Position, Player
import copy


class Utils:
    @staticmethod
    def pipcount(position: Position, player: Player) -> int:
        """Calculate the pipcount for a player - sum of positions of all checkers."""
        total = 0
        for point in range(1, 26):  # Points 1-25 (exclude 0 as those are off the board)
            checkers = position.get_checkers(player, point)
            total += point * checkers
        return total

    @staticmethod
    def can_bear_off(position: Position) -> bool:
        """Check if current player can bear off (all checkers in home board)."""
        player = position.turn
        # Check if player has any checkers outside home board (points 7-24) or on bar (point 25)
        for point in range(7, 26):
            if position.get_checkers(player, point) > 0:
                return False
        return True

    @staticmethod
    def can_move(position: Position, start_point: int, target_point: int) -> bool:
        """Check if a move from start_point to target_point is legal for current player."""
        player = position.turn
        opponent = Player.ME if player == Player.OPPONENT else Player.OPPONENT

        # Check if player has checkers on the bar and trying to move from elsewhere
        if position.get_checkers(player, 25) > 0 and start_point != 25:
            return False

        # Check if target is bearing off (point 0)
        if target_point == 0:
            if not Utils.can_bear_off(position):
                return False

        # Check if target point is occupied by opponent (2 or more opponent checkers)
        # Opponent's point numbering is reversed: opponent's point = 25 - target_point
        if target_point > 0 and position.get_checkers(opponent, 25 - target_point) >= 2:
            return False

        return True

    @staticmethod
    def apply_half_move(pos, start, target, player):
        """Apply a die move to a position and return new position."""
        new_pos = copy.deepcopy(pos)
        new_pos.set_checkers(player, start, new_pos.get_checkers(player, start) - 1)
        new_pos.set_checkers(player, target, new_pos.get_checkers(player, target) + 1)
        return new_pos

    @staticmethod
    def apply_move(pos, half_moves):
        new_pos = copy.deepcopy(pos)
        for start, target in half_moves:
            new_pos = Utils.apply_half_move(new_pos, start, target, pos.get_turn())
        new_pos.switch_turn()
        return new_pos

    @staticmethod
    def possible_moves(position: Position, dice: list):
        """Generate all possible move sequences for given dice for current player."""
        from itertools import permutations

        player = position.turn

        def find_moves_recursive(pos, remaining_dice, current_moves, all_moves_set):
            """Recursively find all possible move sequences."""
            if not remaining_dice:
                all_moves_set.add(tuple(current_moves))
                return

            die = remaining_dice[0]
            found_move = False

            # Try all possible starting points
            for start in range(1, 26):
                if pos.get_checkers(player, start) == 0:
                    continue

                target = start - die if start != 25 else 25 - die
                if target < 0:
                    target = 0  # Bear off

                if Utils.can_move(pos, start, target):
                    found_move = True
                    new_pos = Utils.apply_half_move(pos, start, target, player)
                    new_move = (start, target)
                    find_moves_recursive(new_pos, remaining_dice[1:], current_moves + [new_move], all_moves_set)

            # If no moves possible with remaining dice, add current partial sequence
            if not found_move and current_moves:
                all_moves_set.add(tuple(current_moves))

        # Use set to avoid duplicates
        all_moves_set = set()

        # Try all permutations of dice
        for dice_perm in permutations(dice):
            find_moves_recursive(position, list(dice_perm), [], all_moves_set)

        return all_moves_set

    @staticmethod
    def valid_possible_moves(position: Position, die1: int, die2: int):
        """Generate legal move sequences for two dice with backgammon rules for current player.
           Returns a list of tuples that contain all die moves.
        """
        # Convert to dice list, handling doubles
        if die1 == die2:
            dice = [die1, die1, die1, die1]  # Four moves for doubles
        else:
            dice = [die1, die2]

        # Get all possible moves
        all_moves = Utils.possible_moves(position, dice)

        if not all_moves:
            return set()

        # Filter to keep only moves with maximum number of dice used
        max_dice_used = max(len(moves) for moves in all_moves)
        max_dice_moves = {moves for moves in all_moves if len(moves) == max_dice_used}

        # For non-doubles, if only one die could be played, keep only moves using the higher die
        if die1 != die2 and max_dice_used == 1:
            higher_die = max(die1, die2)
            highest_die_moves = {moves for moves in max_dice_moves
                                 if any(abs(start - target) == higher_die for start, target in moves)}

            # If no moves with highest die, return original filtered moves
            return highest_die_moves if highest_die_moves else max_dice_moves

        return max_dice_moves