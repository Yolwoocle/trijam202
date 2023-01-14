from engine.slimyengine import *

class Player(Pawn):
    def __init__(self, pos: vec3 | None = None):
        Pawn.__init__(self, pos, "player")
    
    def jump(self):
        print("Jump")