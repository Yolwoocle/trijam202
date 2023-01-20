from engine.slimyengine import *
from globals import *
from main_menu import MainMenu
from test_scene import TestScene    # In order be able to jump right into the level while testing


game = Game().init("Trijam 202").target_fps(60).set_background_color(Colors.darkgrey)
game.set_debug(True)
game.load_image("player", "assets/art/player.png")
game.load_image("cement_mixer", "assets/art/cement_mixer.png")
game.load_image("speech_bubble", "assets/art/speech_bubble.png")
game.load_font("game_font", "data/debug_font.ttf")

Settings.debug_level = debugLevel.collisions    # Juste les collisions

world = game.get_world()
world.load_scene(MainMenu)

while game.is_alive():
    game.begin_frame()    
    game.tick()
    game.end_frame()

game.quit()