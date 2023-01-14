from engine.slimyengine import *

class Player(Pawn):
    def __init__(self, pos: vec3 | None = None):
        Pawn.__init__(self, pos, "player")
        self._character.set_size(vec3(2, 2, 2))
    
    def tick(self, dt:float):
        Pawn.tick(self, dt)
        self._root.apply_force(GravityForce(-10))

    def jump(self):
        self._root.apply_force(Force(vec3(0, 0, 10)))