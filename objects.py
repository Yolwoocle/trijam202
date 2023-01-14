from engine.slimyengine import *
from globals import *

class Placeable(Actor):
    def __init__(self, pos: vec3 | None = None):
        Actor.__init__(self, pos)


class CementMixer(Placeable):
    def __init__(self, pos: vec3 | None = None):
        Placeable.__init__(self, pos)
        self._root = SpriteComponent(None, pos, image_name="cement_mixer")
        self._root.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._tooltip = SpriteComponent(self._root, vec3(), image_name="speech_bubble")
        self._tooltip.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
    
    def show_tooltip(self):
        pass