from engine.slimyengine import *
from player import *


game = Game().init().target_fps(60).set_background_color(Colors.darkgrey)
game.set_debug(True)
game.load_image("player", "assets/art/player.png")

world = game.get_world()

main_menu = Scene()
world.enable_physics()
world.get_physics_world().set_limits(vec3(-4, -4, 0), vec3(4, 4, math.inf))

scene = Scene()
world.load_scene(scene)

player = Player()
scene.register_actor(player)
scene.register_component(player.root)



game.update_size()
while game.is_alive():
    game.begin_frame()
    
    game.tick()

    game.end_frame()
game.quit()