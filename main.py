import pygame

from backgammon.ui.board import Board
from backgammon.game.position import Player


def main():
    board = Board()

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
            elif event.type in (pygame.MOUSEWHEEL, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                board.text_area.handle_event(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if board.text_area.handle_event(event):
                    continue
                buttons = pygame.mouse.get_pressed()
                mods = pygame.key.get_mods()
                left, _, right = buttons
                # Workaround for Mac 'magic' mouse
                is_right_click = (event.button == 3) or right or (mods & pygame.KMOD_CTRL)
                is_left_click = left and not is_right_click
                if not (is_left_click or is_right_click):
                    continue
                if board.handle_menu_click(event.pos):
                    continue

                if board.is_game_running:
                    # During a game clicks enter moves; editing the position is disabled.
                    board.handle_game_click(event.pos, is_left_click=bool(is_left_click))
                    continue

                position = board.position
                player = Player.ME if is_left_click else Player.OPPONENT
                if not board.map.handle_cube_click(event.pos, is_left_click=bool(is_left_click), position=position):
                    if not board.handle_turn_indicator_click(event.pos):
                        board.handle_click_entering(event.pos, player, is_left_click=bool(is_left_click))

        board.update()
        board.draw()
        clock.tick(60)

    board.quit()


if __name__ == "__main__":
    main()
