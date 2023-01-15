from engine.slimyengine import *
from globals import *
import math

class Placeable(Actor):
    def __init__(self, pos: vec3 | None = None):
        Actor.__init__(self, pos)


class CementMixer(Placeable):
    def __init__(self, pos: vec3 | None = None):
        Placeable.__init__(self, pos)
        self._root = AnimatedSprite(None, pos, image_names=["cement_mixer"], sprite_time=0.5)
        self._root.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._tooltip = SpriteComponent(self._root, vec3(0, 0, 0), image_name="speech_bubble")
        self._tooltip.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._t = 0.
    
    def tick(self, dt:float):
        self._t += dt
        self._tooltip.set_local_position(vec3(0, 0, 2+0.2*math.sin(5*self._t)))
    
    def show_tooltip(self):
        pass