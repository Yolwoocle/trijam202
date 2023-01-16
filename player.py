from engine.slimyengine import *
from globals import *

class Player(Pawn):
    def __init__(self, pos: vec3 | None = None):
        Pawn.__init__(self, pos, "player")
        self._root.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._character.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._shadow = SpriteComponent(self.root, image_name="default_shadow")
        self._shadow.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        # self._shadow.set_draw_offset(vec2(0, 10))
        self._root.mass=0.01
        self._game = Globals.game
        self._game.register_event_listener(EventListenerFunctionCallback(EventKeyPressed, self.on_key_pressed))
        self._game.get_world().get_current_scene().register_player_actor(self)
    
    def tick(self, dt:float):
        Pawn.tick(self, dt)
        # self._shadow.set_world_position(vec3(self.root.get_world_position().x, self.root.get_world_position().y+PIXEL_SIZE*3, Globals.game._world._physics_world.line_trace(self.root.get_local_position(), vec3(0, 0, -1)).z))
        self._shadow.set_world_position(vec3(self.root.get_world_position().x, self.root.get_world_position().y, 0))
        self._shadow.set_z_bias(-1)
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
    
    def on_key_pressed(self, event:EventKeyPressed):
        # log(f"The key ({event.get_key()}) was pressed")
        # log(f"{self}")
        if event.get_key()==Key.space:
            self.jump()