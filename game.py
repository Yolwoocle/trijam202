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

    if game.is_key_down(pygame.K_SPACE):
        player.jump()
    
    direction = vec2()
    if game.is_key_down(pygame.K_LEFT):  direction+=vec2(-1,  0)
    if game.is_key_down(pygame.K_RIGHT): direction+=vec2( 1,  0)
    if game.is_key_down(pygame.K_UP):    direction+=vec2( 0, -1)
    if game.is_key_down(pygame.K_DOWN):  direction+=vec2( 0,  1)
    if direction.length_squared()>0: player.add_input(direction.normalize())

    game.tick()

    game.end_frame()
game.quit()