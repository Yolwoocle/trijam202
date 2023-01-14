from engine.slimyengine import *

class Player(Pawn):
    def __init__(self, pos: vec3 | None = None):
        Pawn.__init__(self, pos, "player")
        self._character.set_size(vec3(2, 2, 2))
        self._root.mass=0.01
    
    def tick(self, dt:float):
        Pawn.tick(self, dt)
        self._root.apply_force(GravityForce())
        self._root.apply_force(FrictionForce(0.05))

    def jump(self):
        self._root.apply_force(Force(vec3(0, 0, 20)))
    
    def add_input(self, direction:vec2):
        self._root.apply_force(Force(1*vec3(direction.x, direction.y, 0)))