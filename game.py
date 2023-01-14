from engine.slimyengine import *



game = Game().init().target_fps(60).set_background_color(Colors.darkgrey)
game.set_debug(True)
game.load_image("player", "data/player.png")

world = game.get_world()

main_menu = Scene()
world.enable_physics()


game.update_size()
while game.is_alive():
    game.begin_frame()

    game.tick()

    game.end_frame()
game.quit()