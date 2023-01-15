from engine.slimyengine import *
from globals import *

class Player(Pawn):
    def __init__(self, pos: vec3 | None = None):
        Pawn.__init__(self, pos, "player")
        self._character.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._shadow.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._root.mass=0.01
        self._game = Globals.game
    
    def tick(self, dt:float):
        Pawn.tick(self, dt)
        self._root.apply_force(GravityForce())
        self._root.apply_force(FrictionForce(0.05))
        
        direction = vec2()
        if self._game.is_key_down(pygame.K_LEFT):  direction+=vec2(-1,  0)
        if self._game.is_key_down(pygame.K_RIGHT): direction+=vec2( 1,  0)
        if self._game.is_key_down(pygame.K_UP):    direction+=vec2( 0, -1)
        if self._game.is_key_down(pygame.K_DOWN):  direction+=vec2( 0,  1)
        if direction.length_squared()>0: self.add_input(direction.normalize())

    def jump(self):
        self._root.apply_force(Force(vec3(0, 0, 20)))
    
    def add_input(self, direction:vec2):
        self._root.apply_force(Force(1*vec3(direction.x, direction.y, 0)))