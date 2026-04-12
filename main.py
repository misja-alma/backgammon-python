import pygame

from backgammon.ui.board import Board
from backgammon.game.position import Position, Player


def main():
    board = Board()
    position = Position()
    position.setup_starting_position()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.WINDOWRESIZED:
                board.handle_resize(event.x, event.y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    board.handle_key_press(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                buttons = pygame.mouse.get_pressed()

                mods = pygame.key.get_mods()
                left, _, right = buttons
                # Workaround for Mac 'magic' mouse
                is_right_click = (event.button == 3) or right or (mods & pygame.KMOD_CTRL)
                is_left_click = left and not is_right_click

                if is_left_click:
                    if not board.handle_menu_click(event.pos):
                        if not board.handle_turn_indicator_click(event.pos):
                            board.handle_click_entering(event.pos, Player.ME)
                elif is_right_click:
                    if not board.handle_menu_click(event.pos):
                        if not board.handle_turn_indicator_click(event.pos):
                            board.handle_click_entering(event.pos, Player.OPPONENT)

        board.draw(position)
        clock.tick(60)

    board.quit()


if __name__ == "__main__":
    main()