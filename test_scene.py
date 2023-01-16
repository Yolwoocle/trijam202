from engine.slimyengine import *
from player import *
from objects import *


class TestScene(Scene):
    def __init__(self):
        Scene.__init__(self)
        self._world = Globals.game.get_world()
    
    def load(self):
        self._world = Globals.game.get_world()
        self._world.enable_physics()
        self._world.get_physics_world().set_limits(vec3(-4, -4, 0), vec3(4, 4, math.inf))

        player = Player()
        cement_mixer = CementMixer()

        self.register_actor(player)
        self.register_actor(cement_mixer)

        self.register_component(player.root)
        self.register_component(cement_mixer.root)
