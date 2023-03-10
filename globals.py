from engine.slimyengine import Settings

PIXEL_SIZE = Settings.pixel_size
SPRITE_16_SIZE = 16*PIXEL_SIZE

def p_to_w(pixels:float):
    return PIXEL_SIZE*pixels

def w_to_p(world:float):
    return world/PIXEL_SIZE