from engine.slimyengine import *
from test_scene import TestScene


class MainMenu(Scene):
    def __init__(self):
        Scene.__init__(self)
        self._world = Globals.game.get_world()
    
    def load(self) -> 'MainMenu':
        self._world = Globals.game.get_world()
        self._world.disable_physics()

        self._play_btn = Button(None, vec2(100, 100), vec2(100, 30), text="Play")
        self._play_btn.set_callback(self.on_click_play).register()

        return self
    
    def unload(self) -> 'MainMenu':
        self._play_btn._button.remove()
        self._play_btn._button.kill()
    
    def on_click_play(self):
        log("Play pressed")
        # scene = TestScene()
        self._world.unload_scene(self)
        self._world.load_scene(TestScene)