from engine.slimyengine import *
from player import *
from objects import *
from globals import *

game = Game().init("Trijam 202").target_fps(60).set_background_color(Colors.darkgrey)
game.set_debug(True)
game.load_image("player", "assets/art/player.png")
game.load_image("cement_mixer", "assets/art/cement_mixer.png")
game.load_image("speech_bubble", "assets/art/speech_bubble.png")
game.load_font("game_font", "data/debug_font.ttf")

world = game.get_world()

main_menu = Scene()
world.enable_physics()
world.get_physics_world().set_limits(vec3(-4, -4, 0), vec3(4, 4, math.inf))

scene = Scene()
world.load_scene(scene)

player = Player()
cement_mixer = CementMixer()

scene.register_actor(player)
scene.register_actor(cement_mixer)

scene.register_component(player.root)
scene.register_component(cement_mixer.root)


game.update_size()
while game.is_alive():
    game.begin_frame()
    
    if game.is_key_down(pygame.K_SPACE):
        cement_mixer.show_tooltip()
    game.tick()

    game.end_frame()
game.quit()