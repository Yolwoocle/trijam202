from engine.slimyengine import *
from player import *


game = Game().init().target_fps(60).set_background_color(Colors.darkgrey)
game.set_debug(False)
game.load_image("player", "assets/art/player.png")

world = game.get_world()

main_menu = Scene()
world.enable_physics()

scene = Scene()
world.load_scene(scene)

player = Player()
scene.register_component(player.root)

game.update_size()
while game.is_alive():
    game.begin_frame()

    if game.is_key_down(pygame.K_SPACE):
        player.jump()

    game.tick()

    game.end_frame()
game.quit()