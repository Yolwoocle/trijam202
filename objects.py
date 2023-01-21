from engine.slimyengine import *
from globals import *
import math


class Placeable(Actor):
    def __init__(self, pos: vec3 | None = None):
        Actor.__init__(self, pos)

class CementMixer(Placeable):
    def __init__(self, pos: vec3 | None = None):
        Placeable.__init__(self, pos)
        self._root:PhysicsComponent = PhysicsComponent(None, pos).set_simulate_physics(False)
        self._sprite = AnimatedSprite(self._root, vec3(), image_names=["cement_mixer"], sprite_time=0.5)
        self._tooltip = Tooltip(self._root, vec3(0, 0, 0))
        
        self._t = 0.
        
        self._root.set_size(vec3(p_to_w(14), p_to_w(10), p_to_w(4)))
        self._sprite.set_size(vec3(p_to_w(16), p_to_w(19), p_to_w(10))).set_local_position(vec3(0, -p_to_w(4), 0))
        self._tooltip.render()
        register_event_listener(EventListenerFunctionCallback(EventKeyPressed, self.on_keypress))
    
    def tick(self, dt:float):
        self._t += dt
        self._tooltip.set_local_position(vec3(0, 0, p_to_w(23)+p_to_w(1)*math.sin(5*self._t)))
        self._root.apply_force(FrictionForce(0.05))

        if Math.distance_max_vec3(Globals.game.get_world().get_player_actor().root.get_world_position(), self._root.get_world_position())<3:
            self._tooltip.show()
        else:
            self._tooltip.hide()
    
    def show_tooltip(self):
        pass

    def on_keypress(self, event:EventKeyPressed):
        if event.get_key()==Key.e and self._tooltip.is_visible():
            self._tooltip.set_message("Success! :)")
            self._tooltip.render()
            after_time(lambda: (self._tooltip.set_message("PRESS E") and False) or (self._tooltip.render() and False), 1.)

class Tooltip(DrawableComponent):
    def __init__(self, parent: Union['SceneComponent', None] = None, pos: vec3 | None = None):
        DrawableComponent.__init__(self, parent, pos)
        self._bg = SpriteComponent(self, vec3(), image_name="speech_bubble")
        self._bg.set_size(vec3(p_to_w(18), p_to_w(21), p_to_w(16))).set_z_bias(-1)
        self._txt:str = "PRESS E"
        self._font = Globals.game.load_font("game_font", size=14)
        self._txt_surface = SpriteComponent(self._bg, vec3())
        self._txt_surface._skip_resize = True

        Globals.game.register_event_listener(EventListenerFunctionCallback(EventWindowResize, self.on_resize))
        Globals.game.register_event_listener(EventListenerFunctionCallback(EventZoomLevelChanged, self.on_resize))
        
        self.render()
    
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
    
    def set_message(self, msg:str) -> 'Tooltip':
        self._txt = msg
        return self

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