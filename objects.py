from engine.slimyengine import *

class Placeable(Actor):
    def __init__(self, pos: vec3 | None = None):
        Actor.__init__(pos)
        self._root = SpriteComponent(None, self._root)


class CementMixer(Placeable):
    def __init__(self, pos: vec3 | None = None):
        Placeable.__init__(self, pos)
    
    def show_tooltip(self):
        pass