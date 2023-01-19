from engine.slimyengine import *
from globals import *
import math


class Placeable(Actor):
    def __init__(self, pos: vec3 | None = None):
        Actor.__init__(self, pos)

class CementMixer(Placeable):
    def __init__(self, pos: vec3 | None = None):
        Placeable.__init__(self, pos)
        self._root:PhysicsComponent = PhysicsComponent(None, pos).set_simulate_physics(True).set_mass(0.01)
        self._root.set_size(vec3(p_to_w(16), p_to_w(19), p_to_w(16)))
        self._sprite = AnimatedSprite(self._root, vec3(), image_names=["cement_mixer"], sprite_time=0.5)
        self._sprite.set_size(self._root.get_size())
        self._tooltip = Tooltip(self._root, vec3(0, 0, 0))
        self._t = 0.
        self._tooltip.render()
    
    def tick(self, dt:float):
        self._t += dt
        self._tooltip.set_local_position(vec3(0, 0, p_to_w(19)+p_to_w(1)*math.sin(5*self._t)))
        self._root.apply_force(FrictionForce(0.05))

        if Math.distance_max_vec3(Globals.game.get_world().get_player_actor().root.get_world_position(), self._root.get_world_position())<3:
            self._tooltip.show()
        else:
            self._tooltip.hide()
    
    def show_tooltip(self):
        pass

class Tooltip(DrawableComponent):
    def __init__(self, parent: Union['SceneComponent', None] = None, pos: vec3 | None = None):
        DrawableComponent.__init__(self, parent, pos)
        self._bg = SpriteComponent(self, vec3(0, 0, 0), image_name="speech_bubble")
        self._bg.set_size(vec3(p_to_w(18), p_to_w(21), p_to_w(16)))
        self._bg._z_bias=-1
        self._txt:str = "PRESS E"
        self._font = Globals.game.load_font("game_font", size=10)
        self._txt_surface = SpriteComponent(self._bg, vec3(0, 0, 0))
        # self._txt_surface.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._txt_surface._skip_resize=True
        Globals.game.register_event_listener(EventListenerFunctionCallback(EventWindowResize, self.on_resize))
        Globals.game.register_event_listener(EventListenerFunctionCallback(EventZoomLevelChanged, self.on_resize))
    
    def render(self):        
        txt_data = self._font.data.render(self._txt, True, (10, 10, 10))
        dim = vec2(Globals.game.camera.world_size2_to_screen(self._bg.get_size().xy))
        surf = pygame.surface.Surface(dim).convert_alpha()
        surf.fill((255, 255, 255, 0))
        surf.blit(txt_data, dim/2-vec2(txt_data.get_size()[0], txt_data.get_size()[1])/2 - vec2(0, 0.1)*dim.y)
        self._txt_surface.set_sprite(Image("tooltip", dim, "", surf))
    
    def on_resize(self, event:EventWindowResize):
        dim = vec2(Globals.game.camera.world_size2_to_screen(self._bg.get_size().xy))
        self._font = Globals.game.load_font("game_font", size=dim.y/10)
        self.render()
    
    def show(self):
        if not Drawable.show(self): return
        self._bg.show()
        self._txt_surface.show()
        pass

    def hide(self):
        if not Drawable.hide(self): return
        self._bg.hide()
        self._txt_surface.hide()
        self._visible=False
        pass