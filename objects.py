from engine.slimyengine import *

class Placeable(Actor):
    def __init__(self, pos: vec3 | None = None):
        Actor.__init__(pos)