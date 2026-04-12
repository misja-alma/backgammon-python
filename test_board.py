import pygame
from board import Board
from backgammon_position import Position, Player

# TODO move eventloop elsewhere
# Have different modes, selectable by file menu;
# entry mode; on checker click, fill/empty the point up to the mouse position
# Have analysis menu option that gives winning chances
# Create simple recursive analyser for that

def main():
    """Test the Board class by displaying a starting position."""
    board = Board()
    position = Position()
    position.setup_starting_position()
    # position.set_checkers(Player.ME, 0, 5)
    # position.set_checkers(Player.ME, 25, 2)
    # position.set_checkers(Player.OPPONENT, 0, 5)
    # position.set_checkers(Player.OPPONENT, 25, 2)
    # position.set_checkers(Player.ME, 1, 6)
    # position.set_checkers(Player.OPPONENT, 1, 6)
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    board.handle_key_press(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    if not board.handle_menu_click(event.pos):
                        if not board.handle_turn_indicator_click(event.pos):
                            board.handle_click_entering(event.pos, Player.ME)
                elif event.button == 3:  # Right mouse button
                    if not board.handle_menu_click(event.pos):
                        if not board.handle_turn_indicator_click(event.pos):
                            board.handle_click_entering(event.pos, Player.OPPONENT)
        
        board.draw(position)
        clock.tick(60)
    
    board.quit()


if __name__ == "__main__":
    main()