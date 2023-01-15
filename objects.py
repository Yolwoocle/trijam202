from engine.slimyengine import *
from globals import *
import math

class Placeable(Actor):
    def __init__(self, pos: vec3 | None = None):
        Actor.__init__(self, pos)


class CementMixer(Placeable):
    def __init__(self, pos: vec3 | None = None):
        Placeable.__init__(self, pos)
        self._root = AnimatedSprite(None, pos, image_names=["cement_mixer"], sprite_time=0.5)
        self._root.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._tooltip = Tooltip(self._root, vec3(0, 0, 0))
        self._t = 0.
        self._tooltip.render()
    
    def tick(self, dt:float):
        self._t += dt
        self._tooltip.set_local_position(vec3(0, 0, 2+0.6*math.sin(5*self._t)**5))
    
    def show_tooltip(self):
        pass

class Tooltip(DrawableComponent):
    def __init__(self, parent: Union['SceneComponent', None] = None, pos: vec3 | None = None):
        DrawableComponent.__init__(self, parent, pos)
        self._bg = SpriteComponent(self, vec3(0, 0, 0), image_name="speech_bubble")
        self._bg.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._bg._z_bias=-1
        self._txt:str = "Press E..."
        self._font = Globals.game.load_font("game_font", size=10)
        self._txt_surface = SpriteComponent(self._bg, vec3(0, 0, 0)).set_local_position(vec3(self._bg.get_size().x/2, self._bg.get_size().y/2, 0))
        self._txt_surface.set_size(vec3(SPRITE_16_SIZE, SPRITE_16_SIZE, SPRITE_16_SIZE))
        self._txt_surface._skip_resize=True
    
    def render(self):
        self._txt_surface.sprite = Image("tooltip", vec2(), "", self._font.data.render(self._txt, True, (10, 10, 10)))