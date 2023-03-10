import math
import os
import random
import sys
import time
import colorit
from typing import List, Tuple, Union, Literal
from collections import deque
import numpy as np
import json
from pathlib import Path
import socket
import threading

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = '1'
import pygame
import pygame_gui

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    INSTALLED=True
else:
    INSTALLED=False

try:
    import pyi_splash # type: ignore

    pyi_splash.update_text("Loading the engine...")

    pyi_splash.close()
except:
    pass

# Basic definitions
vec2 = pygame.math.Vector2
vec3 = pygame.math.Vector3
vec  = vec2|vec3

sqrt2 = math.sqrt(2)

# Helpers
def random_vec3_in_sphere(origin : vec3, radius : float) -> vec3:
    dir = vec3(2*random.random()-1, 2*random.random()-1, 2*random.random()-1)/sqrt2
    return origin + radius*dir

def random_vec3_at_height(origin : vec3, radius : float) -> vec3:
    dir = vec3(2*random.random()-1, 2*random.random()-1, 0).normalize()*random.random()
    return origin + radius*dir

def perpendicular_vector(v:vec3):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('Zero vector')
        else:
            return v.cross(vec3(0, 1, 0)).normalize()
    return v.cross(vec3(1, 0, 0)).normalize()

def random_vec3_in_cone(direction : vec3, angle : float) -> vec3:
    side = np.arctan(angle)
    v = vec3((2*random.random()-1)*side, (2*random.random()-1)*side, 0)
    p1 = perpendicular_vector(direction)
    p2 = direction.cross(p1)
    return direction+v[0]*p1+v[1]*p2

def random_vec2_in_cone(direction : vec2, angle : float) -> vec2:
    r, theta = direction.as_polar()
    v=vec2()
    v.from_polar([r, theta+(2*random.random()-1)*angle])
    return v

def randrange(a:float=0.9, b:float=1.1) -> float:
    return random.random()*(b-a)+a

def random_float_in_range(a:float=0.9, b:float=1.1) -> float:
    return randrange(a, b)

def set_x(vector : vec3, x : float) -> vec3:
    return vec3(x, vector.y, vector.z)

def set_y(vector : vec3, y : float) -> vec3:
    return vec3(vector.x, y, vector.z)

def set_z(vector : vec3, z : float) -> vec3:
    return vec3(vector.x, vector.y, z)

def is_null_vec2(v : vec2) -> bool:
    return v.x==0 and v.y==0
def clamp(x, a, b):
    return max(a, min(x, b))

def replace_extension(path:str, ext:str)->str:
    p = Path(path)
    p=p.with_suffix('.'+ext)
    return p.resolve().as_posix()

def color_from_vec3(vector:vec3) -> pygame.Color:
    return pygame.Color(int(vector.x), int(vector.y), int(vector.z))


def generate_radial_gradient(color1:vec3, alpha1:int, color2:vec3, alpha2:int, size:vec2=vec2(512, 512)):
    temp_surface = pygame.Surface(size, pygame.SRCALPHA)

    circular_smoothness_steps = 5

    colour_1 = pygame.Color((int(color1.x), int(color1.y), int(color1.z), alpha1))
    colour_1.r, colour_1.g = colour_1.r//circular_smoothness_steps, colour_1.g//circular_smoothness_steps
    colour_1.b, colour_1.a = colour_1.b//circular_smoothness_steps, colour_1.a//circular_smoothness_steps

    colour_2 = pygame.Color((int(color2.x), int(color2.y), int(color2.z), alpha2))
    colour_2.r, colour_2.g = colour_2.r//circular_smoothness_steps, colour_2.g//circular_smoothness_steps
    colour_2.b, colour_2.a = colour_2.b//circular_smoothness_steps, colour_2.a//circular_smoothness_steps

    # 3x3 - starter
    radial_grad_starter = pygame.Surface((3, 3), pygame.SRCALPHA)
    radial_grad_starter.fill(colour_1)
    radial_grad_starter.fill(colour_2, pygame.Rect(1, 1, 1, 1))
    radial_grad = pygame.transform.smoothscale(radial_grad_starter, size)

    for i in range(0, circular_smoothness_steps):
        radial_grad_rot = pygame.transform.rotate(radial_grad, (360.0/circular_smoothness_steps) * i)

        pos_rect = pygame.Rect((0, 0), size)

        area_rect = pygame.Rect(0, 0, size.x, size.y)
        area_rect.center = radial_grad_rot.get_width()//2, radial_grad_rot.get_height()//2
        temp_surface.blit(radial_grad_rot, pos_rect,
                        area=area_rect,
                        special_flags=pygame.BLEND_RGBA_ADD)
    return temp_surface

def get_image_size_tuple(size):
    if size==None:
        return None
    elif type(size)==list and len(size)==2:
        return (size[0], size[1])
    elif type(size)==vec2 and len(size)==2:
        return (size[0], size[1])
    elif type(size)==tuple and len(size)==2:
        return size
    raise RuntimeError("Unknown type for image size")

# Debug log
class logTypes:
    """
    Log types
    """
    info = colorit.Colors.green
    timer = colorit.Colors.blue
    warning = colorit.Colors.yellow
    error = colorit.Colors.red
    trace = colorit.Colors.white

class debugLevel:
    """
    Log levels
    """
    none        = 0
    physics     = 1<<0
    drawing     = 1<<1
    collisions  = 1<<2
    
    all = ~0


def log(msg, type:Tuple[int, int, int]=logTypes.info) -> None:
    """
    Log the data
    """
    pretext = "[WARN]" if type==logTypes.warning else "[INFO]" if type==logTypes.info or type==logTypes.trace else "[TIME]" if type==logTypes.timer else "[ERRO]"
    print(colorit.color(pretext+" {}".format(msg), type))

def log_newline() -> None:
    """
    Log a new line
    """
    print("")

def logf(frame: int, target_frame: int, *args, **kwargs) -> None:
    """
    Log if target_frame matches frame
    """
    if frame==target_frame:
        log(*args, **kwargs)

# Basic classes
class MutableBool:
    def __init__(self, val:bool) -> None:
        self._val = val
        pass

    def get(self):
        return self._val
    
    def set(self, val):
        self._val=val

class BoundingBox:
    def __init__(self, owner:'PhysicsComponent', begin:None|vec3=None, end:None|vec3=None) -> None:
        self._owner:PhysicsComponent = owner
        self._begin = begin if begin else vec3()
        self._end = end if end else vec3()
    
    def intersect(self, other:'BoundingBox') -> tuple[bool, vec3]:
        # log(f"{self._begin} | {self._end}")
        # log(f"{other._begin} | {other._end}")
        t_a = self._owner.get_world_position()
        t_b = other._owner.get_world_position()
        if self._begin.x+t_a.x > other._end.x+t_b.x or self._end.x+t_a.x < other._begin.x+t_b.x:
            return False
        if self._begin.y+t_a.y > other._end.y+t_b.y or self._end.y+t_a.y < other._begin.y+t_b.y:
            return False
        if self._begin.z+t_a.z > other._end.z+t_b.z or self._end.z+t_a.z < other._begin.z+t_b.z:
            return False
        return True


class Ray:
    def __init__(self, origin:vec3, dir:vec3) -> None:
        self._origin = origin
        self._dir = dir
        pass

    def intersect(self, collider:'Collider') -> tuple[bool, vec3, vec3]:
        c = collider.__class__
        if c==BoxCollider:
            return self.intersect_box(collider)
        log(f"ERROR {c}", logTypes.error)
    
    def intersect_box(self, box:'BoxCollider') -> tuple[bool, vec3, vec3]:
        wp = box._owner.get_world_position()
        top = wp-vec3(0, 0, box._size.z/2)
        intersect, pt = ray_intersect_rectangle(top, self._dir, wp-set_z(box._size, 0)/2, wp+set_z(box._size, 0)/2, vec3(0, 0, 1))
        return intersect, pt, pt


def ray_intersect_rectangle(origin:vec3, direction:vec3, p1:vec3, p2:vec3, normal:vec3) -> tuple[bool, vec3]:
    if normal.dot(direction)==0: return False, vec3()
    t = normal.dot(origin-p1)/normal.dot(direction)
    A = origin+t*direction
    # if ((p1.x==p2.x and abs(p1.x-A.x)<=0.1) or min(p1.x, p2.x)<=A.x<=max(p1.x,p2.x)) and \
    #     ((p1.y==p2.y and abs(p1.y-A.y)<=0.1) or min(p1.y, p2.y)<=A.y<=max(p1.y,p2.y)) and \
    #         ((p1.z==p2.z and abs(p1.z-A.z)<=0.1) or min(p1.z, p2.z)<=A.z<=max(p1.z, p2.z)):
    #     return True, A
    s1 = set_z(p2-p1, p1.z)
    s2 = set_z(p1-p2, p2.z)
    if 0<=(A-p1).dot(s1)<=s1.length() and 0<=(A-p2).dot(s2)<s2.length():
        return True, A
    return False, vec3(.5, .5, .5)


class Collider:
    def __init__(self, owner:'PhysicsComponent') -> None:
        self._owner:PhysicsComponent = owner
        pass

    def set_size(self, size:vec3):
        pass

class SphereCollider(Collider):
    def __init__(self, owner:'PhysicsComponent', radius:float) -> None:
        Collider.__init__(self, owner)
        self._radius = radius
    
    def set_size(self, size:vec3):
        self._radius = max(size.x, size.y, size.z)

class BoxCollider(Collider):
    def __init__(self, owner: 'PhysicsComponent', size:vec3()) -> None:
        Collider.__init__(self, owner)
        self._size:vec3 = size
    
    def set_size(self, size:vec3):
        self._size = size

def collide_sphere_sphere(circle1:SphereCollider, circle2:SphereCollider) -> tuple[bool, vec3]:
    t_a = circle1._owner.get_world_position()
    t_b = circle2._owner.get_world_position()
    l = (t_b-t_a).length()
    if l>0:
        return l<(circle1._radius+circle2._radius), -(t_b-t_a).normalize()*(circle1._radius+circle2._radius-l)
    return True, vec3(0, 1, 0)

def collide_box_box(box1:BoxCollider, box2:BoxCollider) -> tuple[bool, vec3]:
    t_a = box1._owner.get_world_position()
    t_b = box2._owner.get_world_position()

    if Settings.debug_level&debugLevel.collisions:
        Globals.game.draw_debug_box(t_a-set_z(box1._size/2, 0), t_a+set_z(box1._size/2, 0), color=(0, 255, 0))
        Globals.game.draw_debug_box(t_b-set_z(box2._size/2, 0), t_b+set_z(box2._size/2, 0), color=(0, 255, 0))

    px = 0
    py = 0
    pz = 0
    if t_a.x < t_b.x:
        px = (-box2._size.x/2+t_b.x) - (box1._size.x/2+t_a.x)
        if px>0: return False, vec3()
    else:
        px = (box2._size.x/2+t_b.x) - (-box1._size.x/2+t_a.x)
        if px<0: return False, vec3()

    if t_a.y < t_b.y:
        py = (-box2._size.y/2+t_b.y) - (box1._size.y/2+t_a.y)
        if py>0: return False, vec3()
    else:
        py = (box2._size.y/2+t_b.y) - (-box1._size.y/2+t_a.y)
        if py<0: return False, vec3()

    if t_a.z < t_b.z:
        pz = (-box2._size.z/2+t_b.z) - (box1._size.z/2+t_a.z)
        if pz>0: return False, vec3()
    else:
        pz = (box2._size.z/2+t_b.z) - (-box1._size.z/2+t_a.z)
        if pz<0: return False, vec3()
    
    if abs(px)<abs(py) and abs(px)<abs(pz):
        return True, vec3(px, 0, 0)
    elif abs(py)<abs(px) and abs(py)<abs(pz):
        return True, vec3(0, py, 0)
    else:
        return True, vec3(0, 0, pz)
    
    """
    delta = t_b-t_a
    if abs(delta.x)>abs(delta.y) and abs(delta.x)>abs(delta.z):
        return True, vec3(px, 0, 0)
    elif abs(delta.y)>abs(delta.x) and abs(delta.y)>abs(delta.z):
        return True, vec3(0, py, 0)
    else:
        return True, vec3(0, 0, pz)
    """

def collide_generic(colA : Collider, colB : Collider) -> tuple[bool, vec3]:
    cA = colA.__class__
    cB = colB.__class__
    if cA==BoxCollider:
        if cB==BoxCollider:
            return collide_box_box(colA, colB)
        if cB==SphereCollider:
            return True, vec3(1, 0, 0)
    elif cA == SphereCollider:
        if cB==BoxCollider:
            return True, vec3(1, 0, 0)
        elif cB==SphereCollider:
            return collide_sphere_sphere(colA, colB)

class Timeline:
    def __init__(self) -> None:
        pass

    def get(self, t:float):
        pass

class FloatTimelineConstant(Timeline):
    def __init__(self, val:float) -> None:
        Timeline.__init__(self)
        self._val = val
    
    def get(self, t:float)->float:
        return self._val

class FloatTimelineFadeIn(Timeline):
    def __init__(self, percentage:float) -> None:
        Timeline.__init__(self)
        self._percentage:float=percentage
    
    def get(self, t:float)->float:
        if t>self._percentage:
            return 1.
        return t/self._percentage

class FloatTimelineFadeOut(Timeline):
    def __init__(self, percentage:float) -> None:
        Timeline.__init__(self)
        self._percentage:float=percentage
    
    def get(self, t:float)->float:
        if t<self._percentage:
            return 1.
        return 1-(t-self._percentage)/(1-self._percentage)

class FloatTimelineFadeInOut(Timeline):
    def __init__(self, percentage:float) -> None:
        Timeline.__init__(self)
        self._percentage:float=percentage
    
    def get(self, t:float)->float:
        if t<self._percentage:
            return t/self._percentage
        elif t>1-self._percentage:
            return 1-(t-(1-self._percentage))/(self._percentage)
        else:
            return 1.

class Colors:
    Color = pygame.Color
    black       = (0, 0, 0)
    white       = (255, 255, 255)
    green       = (0, 255, 0)
    red         = (255, 0, 0)
    darkgreen   = (150, 215, 140)
    darkblue    = (0, 0, 88)

    darkgrey    = (42, 42, 42)

    white_a     = (255, 255, 255, 255)
    
    def get_random():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class Key:
    unknown = pygame.K_UNKNOWN
    up = pygame.K_UP
    left = pygame.K_LEFT
    down = pygame.K_DOWN
    right = pygame.K_RIGHT
    space = pygame.K_SPACE
    leftCtrl = pygame.K_LCTRL
    rightCtrl = pygame.K_RCTRL

    a = pygame.K_a
    b = pygame.K_b
    c = pygame.K_c
    d = pygame.K_d
    e = pygame.K_e
    f = pygame.K_f
    g = pygame.K_g
    h = pygame.K_h
    i = pygame.K_i
    j = pygame.K_j
    k = pygame.K_k
    l = pygame.K_l
    m = pygame.K_m
    n = pygame.K_n
    o = pygame.K_o
    p = pygame.K_p
    q = pygame.K_q
    r = pygame.K_r
    s = pygame.K_s
    t = pygame.K_t
    u = pygame.K_u
    v = pygame.K_v
    w = pygame.K_w
    x = pygame.K_x
    y = pygame.K_y
    z = pygame.K_z

class Math:
    @staticmethod
    def lerp(a, b, k):
        return k*(b-a)+a
    
    @staticmethod
    def lerp_squared(a, b, k) -> vec:
        c:vec = b - a
        return k * c * c.length() + a
    
    @staticmethod
    def distance(a:vec, b:vec) -> float:
        return (b-a).length()
    
    @staticmethod
    def distance_manhattan_vec3(a:vec3, b:vec3) -> float:
        return abs(b.x-a.x)+abs(b.y-a.y)+abs(b.z-a.z)

    @staticmethod
    def distance_max_vec3(a:vec3, b:vec3) -> float:
        return max(abs(b.x-a.x),abs(b.y-a.y),abs(b.z-a.z))
    
    @staticmethod
    def snap_to_grid(value:float, pas:float) -> float:
        return int(value/pas)*pas

class Line2d:
    def __init__(self, point:vec2, dir:vec2) -> None:
        self._pt  = point
        self._dir = dir
    
    def is_dot(self) -> bool:
        return is_null_vec2(self._dir)

class Rect2d:
    def __init__(self, top_left: vec2, bottom_right: vec2) -> None:
        self._tl = top_left
        self._br = bottom_right
    
    @property
    def width(self):
        return abs(self._br.x-self._tl.x)

    @property
    def height(self):
        return abs(self._br.y-self._tl.y)
    
    @property
    def size_x(self):
        return abs(self._br.x-self._tl.x)
    
    @property
    def size_y(self):
        return abs(self._br.y-self._tl.y)
    
    def corners(self) -> Tuple[float, float, float, float]:
        xa = min(self._tl.x, self._br.x)
        ya = min(self._tl.y, self._br.y)
        xb = max(self._tl.x, self._br.x)
        yb = max(self._tl.y, self._br.y)
        return xa, ya, xb, yb

    def intersect_box(self, other : 'Rect2d') -> bool:
        xa, ya, xb, yb = self.corners()
        xa_o, ya_o, xb_o, yb_o = other.corners()
        return not (xb<xa_o or yb<ya_o or xa>xb_o or ya>yb_o)
    
    # def intersect_line(self, line : Line2d) -> bool:

        
class Rect3d:
    def __init__(self, begin:vec3|None=None, end:vec3|None=None) -> None:
        self._begin = begin if begin else vec2()
        self._end   = end if end else vec2()

    @property
    def size_x(self):
        return abs(self._end.x-self._begin.x)
    
    @property
    def size_y(self):
        return abs(self._end.y-self._begin.y)

    @property
    def size_z(self):
        return abs(self._end.y-self._begin.y)


class Image:
    def __init__(self, name, size, path:str="", data:pygame.Surface|None=None, flags:int=0):
        self.name:str = name
        self.size:vec2 = size
        self._path:str= path
        self._data:pygame.Surface|None = data
        if not self._data:
            if size:
                self._data = pygame.Surface(size, flags)
    
    def set_data(self, data):
        self._data = data
    
    def get_data(self) -> pygame.Surface:
        if not self._data: raise RuntimeError("Uninitialized texture")
        return self._data
    
    def resize(self, size) -> 'Image':
        assert self._data!=None
        self._data = pygame.transform.scale(self._data, size)
        self.size = size
        return self
    
    def get_size(self) -> vec2:
        return self.size

    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, val):
        self._path = val

class Animation:
    def __init__(self, name, size, path:str="") -> None:
        pass

class Font:
    def __init__(self, name:str, path:str, data:pygame.font.Font, size:int, flags:int=0):
        self._name = name
        self._path = path
        self._data = data
        self._size = size
        self._flags = flags
    
    def set_data(self, data:pygame.font.Font):
        self._data = data

    def get_data(self) -> pygame.font.Font:
        if not self._data: raise RuntimeError("Uninitialized font")
        return self._data
    
    @property
    def data(self) -> pygame.font.Font:
        return self.get_data()

    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, val):
        self._path = val
    
    @property
    def size(self):
        return self._size

class Tileset:
    def __init__(self, name:str, path:str, tile_width:int, tile_height:int) -> None:
        self.global_image = Globals.game.load_image("tileset_"+name, path)
        self._tiles:list[Image] = []
        self._name = name
        self._sx = int(tile_width)
        self._sy = int(tile_height)
        self._xr = int(self.global_image.size.x/float(tile_width))
        self._yr = int(self.global_image.size.y/float(tile_height))
        self._start_index:int = 1
        for y in range(0, int(self.global_image.size.y), self._sy):
            for x in range(0, int(self.global_image.size.x), self._sx):
                im = Image("tileset_"+name+str(int(x))+"x"+str(int(y)), vec2(self._sx, self._sy), path)
                im.get_data().blit(self.global_image.get_data(), vec2(0, 0), pygame.rect.Rect(x, y, self._sx, self._sy))
                self._tiles.append(im)
    
    def get_random_tile(self) -> Image:
        return self._tiles[random.randrange(0, len(self._tiles))]

    def get_tile(self, index:int) -> Image:
        return self._tiles[index]
    
    def get_tile_component(self, i:int) -> 'SpriteComponent':
        sc = SpriteComponent(None, vec3(), vec2(self._sx, self._sy))
        sc.sprite = self._tiles[i]
        sc._size_locked = True
        return sc


    def get_random_tile_component(self) -> 'SpriteComponent':
        i = random.randint(0, len(self._tiles))
        return self.get_tile_component(i)
    
    
    def get_tile_size(self) -> Tuple[int, int]:
        return (self._sx, self._sy)

class Tilemap:
    def __init__(self, name:str, tilesets:list[Tileset], size:vec2, tile_size:vec2) -> None:
        self._tilesets = dict({t._start_index:t for t in tilesets})
        self._sx = tile_size.x
        self._sy = tile_size.y
        self._size = size if size else vec2()
        self.map   = np.zeros((int(size[0]), int(size[1])), dtype=int)
        self.image = Image(name, (size[0]*self._sx, size[1]*self._sy))
    
    def get_tile(self, idx:int) -> Image:
        index=idx+1
        i=1
        for j in self._tilesets.keys():
            if index<j:
                if index<=i:
                    raise RuntimeError("Tile does not exist")
                return self._tilesets[i].get_tile(index-i-1)
            i=j
        if index>=i:
            return self._tilesets[i].get_tile(index-i-1)
        log("Tile not found {}".format(index), logTypes.error)
        return self._tilesets[self._tilesets.keys()[0]].get_tile(index)

    def compute(self):
        m, n = self.map.shape
        for i in range(m):
            for j in range(n):
                tile = self.get_tile(self.map[i, j])
                self.image.get_data().blit(tile.get_data(), (j*self._sy, i*self._sx))
    
    def set_random(self):
        n = max(self._tilesets.keys())+self._tilesets[max(self._tilesets.keys())]._start_index - 1
        self.map = np.random.randint(n, size=(int(self._size[0]), int(self._size[1])))
        self.compute()
    
    @property
    def width(self):
        m, n = self.map.shape
        return n
    
    @property
    def height(self):
        m, n = self.map.shape
        return m

"""
def import_tiled_tilemap(name:str, path:str, tileset:Tileset) -> Tilemap:
    data = None
    with open(path) as f:
        data=json.load(f)
    log("Loaded tilemap with {} layer{}".format(len(data["layers"]), "s" if len(data["layers"])>1 else ""))
    layer = data["layers"][0]
    width = int(layer["width"])
    height = int(layer["height"])
    size = vec2(width, height)
    map = np.array(layer["data"])
    map = map.reshape(height, width)
    tm = Tilemap(name, [tileset], size, vec2(data["tilewidth"], data["tileheight"]))
    tm.map = map
    tm.compute()
    return tm
"""

# Helpers
def register_event_listener(event_listener:'EventListener') -> 'Game':
    return Globals.game.register_event_listener(event_listener)

def after_time(function:callable, time:float) -> 'Game':
    Globals.game.register_periodic_function(function, time, 1)

# Main classes
class Object:
    def __init__(self):
        self._delete_me = False

class Settings:
    optimize_particles_alpha = False
    max_swaps_per_frame = 10
    debug_level = debugLevel.collisions
    pixel_size = 0.1

class Globals:
    game : 'Game' = None                # type: ignore
    settings : 'Settings' = Settings()  # type: ignore

class Game:
    def __init__(self, size:vec2=vec2(1280, 720)):
        if Globals.game: raise RuntimeError("There can exist only one game")
        Globals.game = self
        self._world = World()
        self._size:vec2 = size
        self._title = ""
        self._fonts : dict[str, dict[size, Font]] = {}
        self._clock : pygame.time.Clock = pygame.time.Clock()
        self._start_time : int = time.time_ns()
        self._ctime : float = 0.0
        self._delta_time:float = 0.0
        self._alive = True
        self._target_fps = 60
        self._background_color = Colors.black
        self._events = []
        self._images = {}
        self._keydowns:map[int, bool] = {}
        
        self._event_listeners : dict[Event.__class__, list[EventListener]] = {}
        
        self._running_threads:List['StoppableThread'] = []
        self._perioric_functions:list[list[float, float, callable]] = []    # (last_time_executed, delay, function)

        self._multiplayer:bool = False
        self._server:None|Server = None

        self._frame_debugs = []
        self.debug_infos:dict[str, str] = {"fps": "0", "deltatime": "0"}
        self._debug_font:Font = None
        self._no_debug = False
    
    def init(self, title="Slimy Engine"):
        pygame.init()
        flags = pygame.RESIZABLE | pygame.DOUBLEBUF
        self._screen = pygame.display.set_mode(self._size, flags)
        # self._screen_to_draw = pygame.display.set_mode(self._size, flags)
        self._title = title
        pygame.display.set_caption(self._title)
        self._clock = pygame.time.Clock()
        self._global_clock = pygame.time.Clock()
        self._start_time = time.time_ns()
        self._debug_font = self.load_font("debug_default", "data/debug_font.ttf")

        self.load_image("default", "data/default.png")
        self.load_image("default_shadow", "data/default_shadow.png")
        self.load_image("default_particle", "data/default_particle.png")

        self._gui_manager:pygame_gui.UIManager = pygame_gui.UIManager(self._size)
        self._gui_refs:dict[pygame_gui.core.UIElement, Widget]={}

        self.set_size(self._size)

        return self
    
    def quit(self) -> bool:
        self._alive=False
        pygame.quit()
        self.kill_threads()
        return True
    
    def kill_threads(self) -> None:
        log("Killing threads...", logTypes.warning)
        for thread in self._running_threads:
            if thread.is_alive():
                thread.join(0.2)
                thread.stop()
        self._running_threads.clear()

    def register_thread(self, thread:'StoppableThread') -> 'Game':
        self._running_threads.append(thread)
        return self
    
    def register_event_listener(self, event_listener:'EventListener') -> 'Game':
        cl = event_listener._event_class
        if cl not in self._event_listeners:
            self._event_listeners[cl] = []
        self._event_listeners[cl].append(event_listener)
        return self
    
    def fire_event(self, event:'Event') -> 'Game':
        if event.__class__ in self._event_listeners:
            for listener in self._event_listeners[event.__class__]:
                listener.trigger(event)
        return self
    
    def register_periodic_function(self, function:callable, interval:float, max_calls=-1) -> bool:
        self._perioric_functions.append([self._ctime, interval, function, max_calls])
        return True
    
    def load_font(self, name, path:str|None=None, size=28, force_reload=False):
        if self._fonts.get(name):
            fonts = self._fonts[name]
            if (size in fonts) and (not force_reload):
                return fonts[int(size)]
            else:
                p=self.resource_path(next(iter(self._fonts[name].values())).path)
                f = Font(name, p, pygame.font.Font(p, int(size)), int(size))
                self._fonts[name][int(size)]=f
                return f
        else:
            if path:
                f = Font(name, path, pygame.font.Font(self.resource_path(path), int(size)), int(size))
                self._fonts[name] = {}
                self._fonts[name][int(size)]=f
                return f
            else:
                log(f"Unable to load font {name}, no path provided", logTypes.error)
                return None
    
    def resource_path(self, relative_path):
        try:
            # PyInstaller creates a temporary folder and stores path in _MEIPASS
            base_path = sys._MEIPASS  # type: ignore # pylint: disable=no-member
        except Exception:
            base_path = os.path.abspath(".")
        
        return os.path.join(base_path, relative_path)

    def load_image(self, name, path="", size=None, force_reload=False):
        if self._images.get(name):
            if (not force_reload) and self._images[name].get(get_image_size_tuple(size)):
                return self._images[name][get_image_size_tuple(size)]
            else:
                if size:
                    p=self.resource_path(next(iter(self._images[name].values())).path)
                    log("Loading image {} from disk with size ({}, {})".format(name, size[0], size[1]), logTypes.trace)
                    im = pygame.image.load(p).convert_alpha()
                    im = pygame.transform.scale(im, size).convert_alpha()
                    s = im.get_size()
                    img = Image(name, vec2(s[0], s[1]), path, im)
                    self._images[name][s] = img
                    return img
                else:
                    return self._images[name][max(self._images[name])]
        if not path:
            raise RuntimeError("Never loaded this resource and no path specified ("+name+")")
        im = pygame.image.load(self.resource_path(path)).convert_alpha()
        log("Loading image {} from disk with {}".format(name, "size {}".format(size) if size else "default size"), logTypes.trace)
        if size:
            im = pygame.transform.scale(im, size).convert_alpha()
        s = im.get_size()
        img = Image(name, vec2(s[0], s[1]), path, im)
        self._images[name] = {}
        self._images[name][s] = img
        return img
    
    def get_world(self) -> 'World':
        return self._world
    
    def set_world(self, world:'World') -> 'Game':
        self._world = world
        return self
    
    def get_ui_manager(self) -> pygame_gui.UIManager:
        return self._gui_manager

    def is_alive(self):
        return self._alive
    
    def target_fps(self, fps):
        self._target_fps = fps
        return self
    
    def get_delta_time(self) -> float:
        return self._delta_time
    
    def get_time(self) -> float:
        return self._ctime
    
    def set_background_color(self, color):
        assert type(color)==tuple
        self._background_color = color
        return self
    
    def get_size(self) -> vec2:
        return self._size
    
    def set_size(self, size:vec2):
        self._size = size
        self.update_size()
    
    def update_size(self) -> None:
        self.get_world().update_screen_size(self._size)
        self._gui_manager.set_window_resolution(self._size)
        self.fire_event(EventWindowResize(self._size))
    
    def on_resize(self, event:pygame.event.Event):
        size = vec2(event.dict['size'][0], event.dict['size'][1])
        self.set_size(size)
        pass

    def is_key_down(self, key):
        if key in self._keydowns:
            return self._keydowns[key]
        return False
    
    def begin_frame(self, dont_clear=False):
        if not self._alive: return
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self._alive=False
                return
            if event.type == pygame.VIDEORESIZE:
                self.on_resize(event)
                pygame.display.update()
            if event.type == pygame.KEYDOWN:
                self._keydowns[event.key]=True
                self.fire_event(EventKeyPressed(event.key))
            if event.type == pygame.KEYUP:
                self._keydowns[event.key]=False
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if self._gui_refs.get(event.ui_element):
                    self._gui_refs[event.ui_element].on_click()
            self._gui_manager.process_events(event)
            
        if not dont_clear:
            self._screen.fill(self._background_color)
    
    def tick(self):
        if not self._alive: return
        self._ctime = (time.time_ns() - self._start_time)/1E9
        for function in self._perioric_functions:
            lt, dt, func, count = function
            if count==0: self._perioric_functions.remove(function)
            if lt+dt<=self._ctime:
                func()
                function[0]=self._ctime
                function[3]=function[3]-1
        # self._delta_time = self._clock.get_time()
        self._gui_manager.update(self._delta_time)
        self._world.tick(self._delta_time)
        self.draw()
    
    def draw(self):
        self._gui_manager.draw_ui(self._screen)
        self._world.draw(self._screen)
        self.debug_pass()

    
    if INSTALLED:
        def debug_pass(self):
            pass
        def draw_debug_vector(self, start : vec3, end : vec3, color=(255,0,0), immediate=False):
            pass
        def draw_debug_spring(self, start : vec3, end : vec3, color=(255,0,0), immediate=False):
            pass
        def draw_debug_rectangle(self, start : vec2, end : vec2, color=(0,0,255), immediate=False, thickness=1):
            pass
        def draw_debug_box(self, start : vec3, end : vec3, color=(0,0,255), immediate=False, thickness=1, offset:None|vec2=None):
            pass
    else:
        def debug_pass(self):
            if not self._no_debug:
                for debug in self._frame_debugs:
                    debug.draw(self._screen)
                self._frame_debugs = []
                current_height = 10
                self.debug_infos["fps"] = str(round(self._clock.get_fps()))
                self.debug_infos["deltatime"] = str(round(self._clock.get_time(), 1))
                for debug in self.debug_infos:
                    txt = str(debug) + ": " + str(self.debug_infos[debug])
                    img = self._debug_font.data.render(txt, True, (255, 255, 255))
                    rect = img.get_rect()
                    self._screen.blit(img, (self._size.x-rect.width-10, current_height))
                    current_height+=rect.height+5

        def draw_debug_vector(self, start : vec3, end : vec3, color=(255,0,0), immediate=False):
            if self._no_debug: return
            vector = DebugVector(self)
            vector.start = start
            vector.end = end
            vector.color = color
            if immediate:
                vector.draw(self._screen)
            else:
                self._frame_debugs.append(vector)
            
        def draw_debug_spring(self, start : vec3, end : vec3, color=(255,0,0), immediate=False):
            if self._no_debug: return
            spring = DebugSpring(self)
            spring._start = start
            spring._end = end
            spring._color = color
            if immediate:
                spring.draw(self._screen)
            else:
                self._frame_debugs.append(spring)
            
        def draw_debug_rectangle(self, start : vec2, end : vec2, color=(0,0,255), immediate=False, thickness=1):
            if self._no_debug: return
            square = DebugRectangle(self)
            square.start = start
            square.end = end
            square.color = color
            square.thickness = thickness
            if immediate:
                square.draw(self.screen)
            else:
                self._frame_debugs.append(square)

        def draw_debug_dot(self, pos : vec2, color=(0,0,255), immediate=False, thickness=1):
            if self._no_debug: return
            dot = DebugDot(self)
            dot.pos = pos
            dot.color = color
            dot.thickness = thickness
            if immediate:
                dot.draw(self._screen)
            else:
                self._frame_debugs.append(dot)

        def draw_debug_box(self, start : vec3, end : vec3, color=(0,0,255), immediate=False, thickness=1, offset:None|vec2=None):
            if self._no_debug: return
            square = DebugBox(self)
            square.start = start
            square.end = end
            square.color = color
            square.thickness = thickness
            square.offset = offset if offset else vec2()
            if immediate:
                square.draw(self._screen)
            else:
                self._frame_debugs.append(square)
    
    def end_frame(self):
        if not self._alive: return        
        # self._screen = pygame.transform.scale(self._screen, self._size)
        # pygame.transform.scale(self._screen_to_draw, self._size)
        # print(self._size)
        # self._screen_to_draw.blit(self._screen, pygame.rect.Rect(0, 0, self._size.x, self._size.y))
        pygame.display.flip()
        self._delta_time = self._clock.tick(self._target_fps) / 1000.0
        # self._screen = pygame.transform.scale(self._screen, vec2(256, 256))
        return
    
    
    # def load_scene(self, scene:'Scene'):
    #     self.active_scene = scene
    #     self.active_scene.get_active_camera().update_screen_size(self._size)
    #     return self
    
    def set_debug(self, debug):
        self._no_debug = not debug
        return self
    
    @property
    def camera(self) -> 'Camera':
        return self.get_world().get_current_scene().get_active_camera()
    
    def host(self, port:int=5050) -> 'Game':
        self._multiplayer = True
        self._server = Server()
        self._server.open(port=port)
        return self


class World:
    def __init__(self) -> None:
        self._game = Globals.game
        self._current_scene:Scene|None = None
        self._physics_world:PhysicsWorld|None = None
        self._particle_systems:list[ParticleSystem] = []
        # self._physics_tick_rate = 1/60.
        # self._physics_last_update = 0.
    
    def get_current_scene(self) -> 'Scene':
        return self._current_scene
    
    def get_player_actor(self) -> 'Actor':
        return self._current_scene.get_player_actor()
    
    def update_screen_size(self, size:vec2):
        if self._current_scene: self._current_scene.update_screen_size(size)

    def load_scene(self, scene:'Scene.__class__') -> 'World':
        sc:Scene = scene()
        self._current_scene = sc
        sc.load()
        Globals.game.update_size()
        return self

    def unload_scene(self, scene:'Scene') -> 'World':
        scene.unload()
        if scene==self._current_scene: self._current_scene = None
        return self
    
    def get_physics_world(self) -> 'PhysicsWorld':
        return self._physics_world
    
    def enable_physics(self) -> 'World':
        if not self._physics_world:
            self._physics_world = PhysicsWorld()
        return self

    def disable_physics(self) -> 'World':
        self._physics_world = None
        return self

    def register_particle_system(self, system:'ParticleSystem') -> 'World':
        assert isinstance(system, ParticleSystem)
        self._particle_systems.append(system)
        return self
    
    def tick(self, dt:float):
        # ctime = self._game.get_time()
        # if (ctime-self._physics_last_update)>self._physics_tick_rate:
        if self._physics_world is not None:
            self._physics_world.tick(dt)
            # self._physics_last_update=ctime
        # if self._current_scene is not None:
        #     self._current_scene.physics_update(delta_time)
        if self._current_scene is not None:
            self._current_scene.update(dt)
        for system in self._particle_systems:
            system.tick(dt)
    
    def draw(self, screen:pygame.Surface):
        for system in self._particle_systems:
            system.draw(screen)
        if self._current_scene is not None:
            self._current_scene.draw(screen)
            self._current_scene.light_pass(screen)
        
        Globals.game.debug_infos["particles_count"] = sum([s.get_particles_count() for s in self._particle_systems])

class Scene:
    def __init__(self):
        self._objects : List[SceneComponent] = []
        self._actors : list[Actor] = []
        self.manual_rendering : bool = False
        self._active_camera : Camera = OrthographicCamera() # type: ignore
        self._drawables : list[DrawableComponent] = []
        self._tilemaps : list[Tilemap] = []
        self._tilesets : list[Tileset] = []
        self._backgrounds : list[SpriteComponent] = []
        self._ambient_light = vec3(1., 1., 1.)
        self._lights : list[Light] = []
        self._lightmap : pygame.Surface = pygame.surface.Surface(vec2(10, 10))
        
        self._player_actor : 'Actor' = None
    
    def load(self) -> 'Scene':
        return self

    def unload(self) -> 'Scene':
        return self
    
    def get_active_camera(self) -> 'Camera':
        return self._active_camera
    
    def add_drawable_rec(self, obj : 'SceneComponent'):
        if isinstance(obj, DrawableComponent):
            self._drawables.append(obj)
        if obj.any_child():
            for child in obj.children:
                if isinstance(child, SceneComponent):
                    self.add_drawable_rec(child) # type: ignore
    
    def register_actor(self, actor:'Actor') -> 'Scene':
        self._actors.append(actor)
        if actor.root:
            self.register_component(actor.root)
        return self
    
    def register_component(self, component : 'SceneComponent'):
        self._objects.append(component) # Add only the root component
        self.add_drawable_rec(component)
        return self
    
    def register_light(self, light:'Light'):
        self._lights.append(light)
        light._scene = self
        return self
    
    def update_screen_size(self, size:vec2):
        if self._active_camera: self._active_camera.update_screen_size(size)
        for light in self._lights:
                light.render()
    
    def register_player_actor(self, actor : 'Actor') -> 'Scene':
        self._player_actor = actor
        return self
    
    def get_player_actor(self) -> 'Actor':
        return self._player_actor
    
    def draw(self, screen:pygame.Surface):
        # Lazy bubble sort with only few swaps per frame
        for _ in range(Settings.max_swaps_per_frame):
            for i in range(len(self._drawables)-1):
                if self._drawables[i]>self._drawables[i+1]:
                    c=self._drawables[i]
                    self._drawables[i]=self._drawables[i+1]
                    self._drawables[i+1]=c
                    break
        if not self.manual_rendering:
            for background in self._backgrounds:
                background.draw(screen)
            for obj in self._drawables:
                obj.draw(screen)
    
    def set_ambient_light(self, val:vec3):
        self._ambient_light = val
        return self
    
    def light_pass(self, screen:pygame.Surface):
        if self._lightmap.get_size()!=Globals.game._size:
            self._lightmap = pygame.surface.Surface(Globals.game._size)
        self._lightmap.fill(color_from_vec3(self._ambient_light*255))
        if not self.manual_rendering:
            for light in self._lights:
                light.draw()
            
            screen.blit(self._lightmap, (0, 0), special_flags=pygame.BLEND_MULT)
    
    def update(self, dt:float):
        # for actor in self._actors:
        #     actor.pre_tick(dt)
        for obj in self._objects:
            obj.update()
        for actor in self._actors:
            actor.tick(dt)
    
    # def physics_update(self, dt:float):
    #     for actor in self._actors:
    #         actor.physics_tick(dt)
    
    def clear(self):
        self._backgrounds.clear()
    

    def load_map(self, name:str, path:str) -> list['SpriteComponent']:
        sprites = []
        data = None
        with open(path) as f:
            data=json.load(f)
        d_tilesets = data["tilesets"]
        d_layers = data["layers"]
        log("Loading map with:")
        log(" => {} tileset{}".format(len(d_tilesets), "s" if len(d_tilesets)>1 else ""))
        i=0
        tw = data["tilewidth"]
        th = data["tileheight"]
        for tileset in d_tilesets:
            ts = Tileset(name+"_"+str(i), replace_extension(tileset["source"], "png"), int(data["tilewidth"]), int(data["tileheight"]))
            ts._start_index = tileset["firstgid"]
            self._tilesets.append(ts)
            i+=1
        
        for layer in d_layers:
            width = int(layer["width"])
            height = int(layer["height"])
            size = vec2(width, height)
            map = np.array(layer["data"])
            map = map.reshape(height, width)
            tm = Tilemap(name, self._tilesets, size, vec2(tw, th))
            tm.map = map
            tm.compute()
            self._tilemaps.append(tm)
            map_sprite = SpriteComponent(None, vec3(2, 2, 0), vec2(0, 0))
            map_sprite.size = vec3(tm.width*2, tm.height*2, 0)
            map_sprite.sprite = tm.image
            map_sprite._size_locked = True
            sprites.append(map_sprite)
            self._backgrounds.append(map_sprite)
        
        return sprites
    
    def get_light_map(self) -> pygame.Surface:
        return self._lightmap


# Networking
class StoppableThread(threading.Thread):
    def __init__(self) -> None:
        threading.Thread.__init__(self)
        self._should_stop = False

    def stop(self):
        self._should_stop = True

class ServerWaitingThread(StoppableThread):
    def __init__(self, server:'Server') -> None:
        StoppableThread.__init__(self)
        self.name = "server_waiting_thread"
        self._server = server
    
    def run(self):
        while not self._should_stop:
            try:
                conn, adr = self._server._socket.accept()
                conn.setblocking(0)
                self._server._new_connection(conn, adr)
            except socket.timeout:
                pass
            except KeyboardInterrupt:
                return
            except BlockingIOError:
                pass
        self._server._socket.close()
        return

class ClientConnectThread(StoppableThread):
    def __init__(self, client:'Client') -> None:
        StoppableThread.__init__(self)
        self.name = "client_connect_thread"
        self._client = client
    
    def run(self):
        while not self._should_stop:
            try:
                self._client._socket.connect((self._client._address, self._client._port))
                self._client.on_connect()
                break
            except socket.timeout:
                log("Socket connect timeout", logTypes.warning)
            except ConnectionRefusedError as e:
                log(f"Connection refused ({e})", logTypes.warning)
                pass
            except KeyboardInterrupt:
                break

class Reflected:
    def __init__(self) -> None:
        self._version:int = 0
    
    def get_version(self) -> int:
        return self._version

class Server:
    def __init__(self) -> None:
        self._open:bool = False
        self._address:str = ""
        self._port:int = 5050
        self._socket:socket.socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._clients:list[socket.socket]=[]
        self._max_clients:int=32

    def open(self, address:str="127.0.0.1", port:int=5050, backlog:int=5) -> 'Server':
        if self._open: return self
        self._address = address
        self._port = port
        try:
            self._socket.bind((self._address, self._port))
        except socket.error as e:
            log("Failed to open server", logTypes.error)
            return self
        # self.set_timeout(0.5)
        # self._socket.setblocking(False)
        self._socket.listen(backlog)
        log("Server is waiting for connection...")
        thread = ServerWaitingThread(self)
        Globals.game.register_thread(thread)
        self._open = True
        thread.start()
        return self
    
    def set_timeout(self, timeout:float) -> 'Server':
        self._socket.settimeout(timeout)
        return self
    
    def _new_connection(self, conn:socket.socket, adr):
        a, b = conn.getpeername()
        log(f"New connection from {a} on {b}")
        if len(self._clients)>=self._max_clients:
            log("Already full, can't accept the connection", logTypes.error)
            return
        self._clients.append(conn)

        while self._open:
            log("Waiting for a message...")
            data = conn.recv(2048)
            log("Passed")
            if not data:
                log("Client disconnected", logTypes.warning)
                return
            log(f"Received data : {data}")
            self.broadcast(str.encode("Received!"))
            log("Broadcasted")
    
    def broadcast(self, message:bytes) -> 'Server':
        print("Before")
        # self._socket.send(message) # THIS SEEMS TO BE BLOCKING
        print("After")
        return self
    
    def close(self) -> 'Server':
        self._open=False
        return self

class Client:
    def __init__(self) -> None:
        self._connected:bool = False
        self._address:str = ""
        self._port:int = 5050
        self._socket:socket.socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def connect(self, address:str="127.0.0.1", port:int=5050) -> 'Client':
        self._address=address
        self._port=port
        thread = ClientConnectThread(self)
        Globals.game.register_thread(thread)
        self._connected=True
        thread.start()
        return self
    
    def disconnect(self) -> 'Client':
        log("Disconnecting...")
        self._connected = False
        return self
    
    def on_connect(self) -> 'Client':
        self._connected = True
        self._socket.send(b"Hello")
        while False:
            data = self._socket.recv(2048).decode()
            if not data:
                log("Connection lost", logTypes.error)
            print(data)
        return self
    
    def send(self, data:bytes) -> 'Client':
        self._socket.send(data)
        return self
    
    def close(self) -> 'Client':
        self._socket.close()
        self._connected = False
        return self

class Event:
    def __init__(self):
        self._consumed = False

class EventWindowResize(Event):
    def __init__(self, size:vec2):
        Event.__init__(self)
        self._size:vec2 = size
    
    def get_size(self) -> vec:
        return self._size

class EventZoomLevelChanged(Event):
    def __init__(self, size:vec2):
        Event.__init__(self)
        self._size:vec = size
    
    def get_size(self) -> vec:
        return self._size

class EventKeyPressed(Event):
    def __init__(self, key:int=Key.unknown):
        Event.__init__(self)
        pygame.K_RCTRL
        self._key:int = key
    
    def get_key(self) -> int:
        return self._key

class WidgetClick(Event):
    def __init__(self):
        Event.__init__(self)

class EventListener:
    def __init__(self, event_class:Event.__class__) -> None:
        self._event_class : Event.__class__ = event_class
    
    def register(self) -> 'EventListener':
        Globals.game.register_event_listener(self)
        return self
    
    def trigger(self, event:Event):
        pass

class EventListenerFunctionCallback(EventListener):
    def __init__(self, event_class:Event.__class__, callback:callable):
        EventListener.__init__(self, event_class)
        self._callback = callback
    
    def trigger(self, event:Event):
        self._callback(event)


class Actor(Object):
    def __init__(self, pos:vec3|None=None):
        Object.__init__(self)
        self._root:SceneComponent = SceneComponent(None, pos)
    
    @property
    def root(self) -> 'SceneComponent':
        return self._root
    
    def pre_tick(self, dt:float):
        pass

    def tick(self, dt:float):
        pass

    # def physics_tick(self, dt:float):
    #     pass

class Pawn(Actor):
    def __init__(self, pos:vec3|None=None, image_name:str="default"):
        Actor.__init__(self, pos=pos)
        self._root:PhysicsComponent = PhysicsComponent(None, pos=pos, mass=0.1)
        self._character = SpriteComponent(self.root, image_name=image_name)
    
    def tick(self, dt:float):
        Actor.tick(self, dt)


class Component:
    def __init__(self, parent:Union[None,'Component']=None) -> None:
        self._parent   : Component | None  = parent
        self._children : List[Component]   = []
        if parent:
            assert isinstance(parent, Component)
            parent.add_child(self)

    def add_child(self, child):
        self._children.append(child)
    
    def attach(self, parent):
        self._parent = parent
    
    def update(self):
        pass
        
    @property
    def children(self):
        return self._children
    
    def any_child(self):
        return len(self._children)>0

class SceneComponent(Component):
    def __init__(self, parent:Union['SceneComponent',None]=None, pos:vec3|None=None):
        Component.__init__(self, parent)
        self._parent:SceneComponent|None = parent
        self._pos:vec3 = pos if pos else vec3()
        self._size:vec3 = vec3(0.5, 0.5, 0.5)
        self._scene_parent:bool = True if isinstance(parent, SceneComponent) else False
        self._parent_pos:vec3 = parent.get_world_position() if self._scene_parent else vec3() # type: ignore
        self._valid:bool = False
        self._update_count:int = 0
        self._inherit_parent_position = True
    
    def get_local_position(self):
        return self._pos
    
    def set_local_position(self, val:vec3) -> 'SceneComponent':
        self._pos = val
        self._valid = False
        # for c in self._children:
        #     c.invalidate()
        return self
    
    def set_world_position(self, val:vec3) -> 'SceneComponent':
        self._pos = val-self._parent_pos
        return self
    
    def set_inherit_parent_location(self, val:bool) -> None:
        self._inherit_parent_position = val
    
    def get_world_position(self):
        return self._parent_pos+self._pos

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, s):
        self._size = s
    
    def set_size(self, size:vec3):
        self._size = size
        return self
    
    def get_size(self):
        return self._size
    
    def attach(self, parent:'SceneComponent'):
        Component.attach(self, parent)
        self._parent_pos = parent.get_world_position() if self._inherit_parent_position else vec3()
        self.invalidate()
    
    def invalidate(self):
        self._valid = False

    def update(self):
        # if self._valid: return
        # self._update_count+=1
        if self._parent:
            self._parent_pos=self._parent.get_world_position() if self._inherit_parent_position else vec3()
        for child in self._children:
            # child._parent_pos = vec3(self._parent_pos+self._pos)
            child.update()
        self._valid = True
        return
    
class Camera(SceneComponent):
    def __init__(self, parent:SceneComponent|None=None, pos:vec3|None=None) -> None:
        SceneComponent.__init__(self, parent, pos)
        self.game = Globals.game
        self.zoom = 1.
        pass

    def world_to_screen(self, world : vec3) -> vec2:
        return vec2()
    
    def world_to_cam(self, world : vec3) -> vec3:
        return vec3()

    def update_screen_size(self, size : vec2) -> None:
        pass

    def set_zoom(self, zoom : float) -> None:
        pass

    def get_zoom(self) -> float:
        return self.zoom
    
    def world_size2_to_screen(self, dim : vec2) -> vec2:
        return vec2()


class OrthographicCamera(Camera):
    def __init__(self) -> None:
        Camera.__init__(self)
        self.offset = vec2(0.5, 0.5)                    # Camera center is...centered
        self.aspect_ratio = 1
        self.bounds = vec2(10./self.aspect_ratio, 10.)
        self.screen_size = vec2(640, 480)
    
    def update_screen_size(self, size : vec2) -> None:
        self.aspect_ratio = size[1]/size[0]
        self.screen_size = vec2(size[0], size[1])
        self.bounds = vec2(self.bounds[1]/self.aspect_ratio, self.bounds[1])

    def world_to_cam(self, world : vec3) -> vec3:
        return world-self.get_world_position()

    def world_to_screen(self, world : vec3) -> vec2:
        p = world-self.get_world_position()
        pas = 0.1
        p = vec3(Math.snap_to_grid(p.x, pas), Math.snap_to_grid(p.y, pas), Math.snap_to_grid(p.z, pas))
        x = ((p.x)/self.bounds.x + self.offset.x)*self.screen_size.x
        y = ((p.y-p.z)/self.bounds.y + self.offset.y)*self.screen_size.y
        return vec2(x, y)
        
    def set_bounds_height(self, height:float):
        self.bounds = vec2(height/self.aspect_ratio, height)
        Globals.game.fire_event(EventZoomLevelChanged(vec2(self.bounds)))
    
    def world_size2_to_screen(self, dim : vec2) -> vec2:
        x = Math.snap_to_grid(dim.x, 0.1)*self.screen_size.x/self.bounds.x
        y = Math.snap_to_grid(dim.y, 0.1)*self.screen_size.y/self.bounds.y
        return vec2(int(x), int(y))


class Drawable():
    def __init__(self):
        self._z_bias:int = 0
        self._visible:bool = True
        pass

    def set_z_bias(self, bias:int) -> 'Drawable':
        self._z_bias = bias
        return self
    
    def draw(self, screen:pygame.Surface):
        return
    
    def is_visible(self) -> bool:
        return self._visible
    
    def show(self) -> bool:
        if self._visible: return False
        self._visible = True
        return True
    
    def hide(self) -> bool:
        if not self._visible: return False
        self._visible = False
        return True

class DrawableComponent(SceneComponent, Drawable):
    def __init__(self, parent=None, pos=vec3()):
        SceneComponent.__init__(self, parent, pos)
        Drawable.__init__(self)
        
    # def __le__(self, other:'DrawableComponent'):
    #     a=self.get_world_position()
    #     b=other.get_world_position()
    #     # if (a.z-b.z)<=0.005:
    #         # return a.y<=b.y
    #     return a.z+a.y<=b.z+b.y
    
    def __lt__(self, other:'DrawableComponent'):
        a=self.get_world_position()
        b=other.get_world_position()
        if abs(a.z-b.z)<=0.01:
            if self._z_bias!=other._z_bias:
                return self._z_bias<other._z_bias
            return a.y<b.y
        return a.z<b.z


class Light(DrawableComponent):
    def __init__(self, scene:'Scene', parent: Union['SceneComponent', None] = None, pos: vec3 | None = None):
        DrawableComponent.__init__(self, parent, pos)
        self._strength:float = 1.
        self._scene = scene
        self._light_surface:pygame.Surface=pygame.surface.Surface((10, 10))
        self._color:vec3 = vec3(255, 255, 255)
    
    def set_color(self, color:vec3):
        self._color.x, self._color.y, self._color.z = color.x, color.y, color.z
        return self
    
    def draw(self, screen:pygame.Surface):
        Drawable.draw(self, screen)
    
    def render(self):
        pass

class PointLight(Light):
    def __init__(self, scene:'Scene', parent: Union['SceneComponent', None] = None, pos: vec3 | None = None):
        Light.__init__(self, scene, parent, pos)
        # self._light_surface = pygame.Surface((100, 100))
        # self._light_surface.fill('White')
        self.render()
    
    def draw(self, screen:pygame.Surface):
        Light.draw(self, screen)
        self._scene.get_light_map().blit(self._light_surface, Globals.game.camera.world_to_screen(self._pos), special_flags=pygame.BLEND_ADD)
    
    def render(self):
        Light.render(self)
        screen_size = Globals.game.camera.world_size2_to_screen(self.size.xy)
        self._light_surface = generate_radial_gradient(vec3(0, 0, 0), 255, self._color, 255, screen_size)
        return self

class ParticleEmitter(Drawable):
    def __init__(self, system:Union[None,'ParticleSystem']=None):
        Drawable.__init__(self)
        self._rate = 1.0
        self._elapsed_time:int = 0
        self._particles:deque[tuple[int, MutableBool, vec3, vec3, list[int], int]]=deque()  # particle: id, is_alive, position, velocity, color, creation_time
        self._sprite:Image=Globals.game.load_image("default_particle").resize((8, 8))
        self._started = False
        self._system = system
        self.draw_size = vec2()
        self._sprite_size = 2*vec2(1., 1.)
        self._size_locked = False
        self._pos = vec3()
        self._track_component:None|SceneComponent = None
        self._alpha_animate=FloatTimelineFadeInOut(0.1)
    
    def track_component(self, component:SceneComponent):
        self._track_component = component
    
    def start(self) -> 'ParticleEmitter':
        self._started = True
        return self
    
    def tick(self, dt) -> None:
        if not self._started: return
        self._elapsed_time+=dt
        if self._track_component:
            self._pos = self._track_component.get_world_position()
        
        self.draw_size = Globals.game.camera.world_size2_to_screen(self._sprite_size)
        if self.draw_size!=self._sprite.size:
            log("Resizing particle sprite", logTypes.warning)
            if (not self._size_locked):
                self._sprite = Globals.game.load_image(self._sprite.name, self._sprite.path, self.draw_size)
            else:
                self._sprite.resize(self.draw_size)
        
        alive_up_to=0
        i=-1
        for particle in self._particles:
            i+=1
            id, is_alive, position, velocity, color, creation_time, max_age = particle
            if not is_alive.get():
                continue
            alive_up_to=i
            age = self._elapsed_time-creation_time
            normalized_age:float = age/max_age
            if normalized_age>1:
                is_alive.set(False)
            color[3]=int(self._alpha_animate.get(normalized_age)*255)
            
            position+=velocity*dt
        dead_to=len(self._particles)-alive_up_to-1
        for _ in range(dead_to): self._particles.pop()
    
    def get_particles_count(self)->int:
        return len(self._particles)
    
    def draw(self, screen:pygame.Surface) -> None:
        assert self._system!=None
        camera = Globals.game.camera
        screen_size = Globals.game.get_size()
        for particle in self._particles:
            id, is_alive, position, _, color, _, _ = particle
            if not is_alive.get(): continue
            screen_pos = camera.world_to_screen(position+self._system.get_world_position())

            if screen_pos.x+self.draw_size.x<0 or screen_pos.y+self.draw_size.y<0 or screen_pos.x-self.draw_size.x>screen_size[0] or screen_pos.y-self.draw_size.y>screen_size[1]: is_alive.set(False)
            # to_paint = self._sprite.get_data().copy()
            # to_paint.fill(color[:3], special_flags=pygame.BLEND_MULT)
            # to_paint.set_alpha(color[3])
            # screen.blit(to_paint, screen_pos)

            if not Globals.settings.optimize_particles_alpha:
                self._sprite.get_data().set_alpha(color[3])
            # self._sprite.get_data().fill(color[:3], special_flags=pygame.BLEND_MULT)
            screen.blit(self._sprite.get_data(), screen_pos)

class FountainEmitter(ParticleEmitter):
    def __init__(self, system: Union[None, 'ParticleSystem'] = None):
        ParticleEmitter.__init__(self, system)
        self._spawn_target = 0.
        self._spawn_size = 2.
        self._spawn_range = (0.05, 0.1)
    
    def tick(self, dt) -> None:
        if not self._started: return

        if self._elapsed_time>=self._spawn_target:
            self._spawn_target=self._elapsed_time+random_float_in_range(self._spawn_range[0], self._spawn_range[1])
            pos = self._pos.copy()
            pos = pos if self._spawn_size==0 else random_vec3_in_sphere(pos, self._spawn_size)
            self._particles.appendleft((0, MutableBool(True), pos, vec3(), list(Colors.get_random()), self._elapsed_time, 10.))

        ParticleEmitter.tick(self, dt)


class ParticleSystem(DrawableComponent):
    def __init__(self, parent=None, pos=vec3()):
        DrawableComponent.__init__(self, parent, pos)
        self._emitters:list[ParticleEmitter]=[]
    
    def start(self) -> 'ParticleSystem':
        for emitter in self._emitters:
            emitter.start()
        return self

    def tick(self, dt) -> 'ParticleSystem':
        for emitter in self._emitters:
            emitter.tick(dt)
        
        return self

    def draw(self, screen:pygame.Surface) -> 'ParticleSystem':
        DrawableComponent.draw(self, screen)
        for emitter in self._emitters:
            emitter.draw()
        return self
    
    def get_particles_count(self)->int:
        s=0
        for e in self._emitters: s+=e.get_particles_count()
        return s

class DebugDraw:
    def __init__(self, game) -> None:
        self.game : Game = game
        pass

    def draw(self, screen):
        return

class DebugVector(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self.start:vec3         = vec3()
        self.end:vec3           = vec3()
        self.color:Colors.Color = Colors.Color(255, 0, 0)
        self.thickness:int      = 1
    
    def draw(self, screen):
        DebugDraw.draw(self, screen)
        if self.start==self.end: return
        s_2d = self.game.camera.world_to_screen(self.start)
        e_2d = self.game.camera.world_to_screen(self.end)
        if (e_2d-s_2d).length()<=0: return
        dir = (e_2d-s_2d).normalize()
        length = (e_2d-s_2d).length()
        pygame.draw.line(screen, self.color, s_2d, e_2d)
        pygame.draw.lines(screen, self.color, False, [length*0.3*(-dir).rotate(20)+e_2d, e_2d, length*0.3*(-dir).rotate(-20)+e_2d])

class DebugRectangle(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self.start:vec2 = vec2()
        self.end:vec2 = vec2()
        self.color:Colors.Color = Colors.Color(255, 0, 0)
        self.thickness = 1
    
    def draw(self, screen):
        DebugDraw.draw(self, screen)
        if self.start==self.end: return
        pygame.draw.rect(screen, self.color, pygame.Rect(self.start.x, self.start.y, self.end.x-self.start.x, self.end.y-self.start.y), self.thickness)

class DebugDot(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self.pos:vec2 = vec2()
        self.color:Colors.Color = Colors.Color(255, 0, 0)
        self.thickness = 1
    
    def draw(self, screen):
        DebugDraw.draw(self, screen)
        pygame.draw.circle(screen, self.color, self.pos, self.thickness*2)

class DebugBox(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self.start:vec3 = vec3()
        self.end:vec3 = vec3()
        self.color:Colors.Color = Colors.Color(255, 0, 0)
        self.thickness = 1
        self.offset:vec2 = vec2()
    
    def draw(self, screen):
        DebugDraw.draw(self, screen)
        s2d = self.game.camera.world_to_screen(self.start)+self.offset
        e2d = self.game.camera.world_to_screen(self.end)+self.offset
        if self.start==self.end: return
        pygame.draw.rect(screen, self.color, pygame.Rect(s2d.x, s2d.y, e2d.x-s2d.x, e2d.y-s2d.y), self.thickness)


class DebugSpring(DebugDraw):
    def __init__(self, game) -> None:
        DebugDraw.__init__(self, game)
        self._start:vec3 = vec3()
        self._end:vec3 = vec3()
        self._color:Colors.Color = Colors.Color(255, 0, 0)
        self._thickness = 1
    
    def draw(self, screen):
        ends_length = 0.5
        num_spires = 15
        width=0.5
        camera = self.game.camera
        if self._start==self._end: return
        unit = (self._end-self._start).normalize()
        length = (self._end-self._start).length()
        ends_length=min(ends_length, length/2)
        side=vec3(0,0,1).cross(unit).normalize() if vec3(0,0,1).dot(unit)==0 else vec3(0,1,0).cross(unit).normalize()
        offset = self._start+unit*ends_length
        pygame.draw.line(screen, self._color, camera.world_to_screen(self._start), camera.world_to_screen(offset), self._thickness)
        stride = (length-2*ends_length)/num_spires
        for i in range(num_spires):
            pygame.draw.lines(screen, self._color, False, [camera.world_to_screen(offset+i*stride*unit), camera.world_to_screen(offset+i*stride*unit+stride/2*unit+(width*side if i%2 else -width*side)), camera.world_to_screen(offset+(i+1)*stride*unit)])
        pygame.draw.line(screen, self._color, camera.world_to_screen(self._end-unit*ends_length), camera.world_to_screen(self._end), self._thickness)


class PhysicsWorld:
    def __init__(self):
        self._objects:List[PhysicsComponent] = []
        self._limits = [vec3(-math.inf, -math.inf, -math.inf), vec3(math.inf, math.inf, math.inf)]
        self._draw_borders = True
        self._last_tick = time.time_ns()
        self._tmp_tick = time.time_ns()
    
    def set_limits(self, min, max):
        assert type(min)==vec3 and type(max)==vec3
        self._limits = [min, max]
        return self
    
    def register_physics_component(self, obj:'PhysicsComponent'):
        assert isinstance(obj, PhysicsComponent)
        self._objects.append(obj)

        obj.set_physics_world(self)

    def tick(self, dt:float):
        # self._tmp_tick=time.time_ns()
        # dt = (self._tmp_tick-self._last_tick)*1.0E-9
        # self._last_tick=self._tmp_tick

        if self._draw_borders:
            a = set_z(self._limits[0], 0)
            b = set_z(self._limits[1], 0)
            if a.length_squared()<math.inf and b.length_squared()<math.inf:
                if Settings.debug_level&debugLevel.physics: Globals.game.draw_debug_box(a, b, vec3(255, 0, 0), thickness=2)

        for obj in self._objects:
           obj.tick(dt)
        
        # Collisions
        n=len(self._objects)
        for i in range(n):
            for j in range(i+1, n):
                objA = self._objects[i]
                objB = self._objects[j]
                if objA==objB or (not objA._collide) or (not objB._collide): continue
                col = objA.get_bounding_box().intersect(objB.get_bounding_box())
                if not col: continue
                # objA.apply_force(Force(100*dir.normalize()*dir.length()**7))
                colA = objA.get_collider()
                colB = objB.get_collider()
                col, direction = collide_generic(colA, colB)
                if not col: continue
                if objA._simulate_physics:
                    if objB._simulate_physics:
                        M = objA._mass+objB._mass
                        objA._pos += (objB._mass)/M*direction
                        objB._pos -= (objA._mass)/M*direction
                        objA._vel += (objB._mass)/M*direction/dt
                        objB._vel -= (objA._mass)/M*direction/dt
                        
                    else:
                        objA._pos += direction
                        objA._vel += direction/dt
                else:
                    if objB._simulate_physics:
                        objB._pos -= direction
                        objB._vel -= direction/dt


    def line_trace(self, ray:Ray, ignore:list['PhysicsComponent']=[]):
        closest = None
        closest_dir = 0
        return set_z(ray._origin, min(self._limits[0].z, self._limits[1].z))
        
        for obj in self._objects:
            if obj in ignore: continue
            intersect, p1, _ = ray.intersect(obj.get_collider())
            if intersect and (not closest or (p1-ray._origin).length_squared()<closest_dir):
                closest = p1
                closest_dir = (p1-ray._origin).length_squared()
        if not closest:
            return set_z(ray._origin, min(self._limits[0].z, self._limits[1].z))
            intersect, p = ray_intersect_rectangle(ray._origin, ray._dir, self._limits[0], set_z(self._limits[1], self._limits[0].z), vec3(0, 0, 1))
            print(intersect)
            if Settings.debug_level&debugLevel.collisions: Globals.game.draw_debug_dot(Globals.game.camera.world_to_screen(p), (255, 0, 0))
            return p
        return closest

class Force:
    def __init__(self, value=vec3()):
        self.value = value
    
    def get(self, object:Union['PhysicsComponent',None]=None):
        return self.value
    
    @property
    def x(self):
        return self.value.x

    @x.setter
    def x(self, val):
        self.value.x=val
    
    @property
    def y(self):
        return self.value.y
    
    @y.setter
    def y(self, val):
        self.value.y=val

    @property
    def z(self):
        return self.value.z
    
    @z.setter
    def z(self, val):
        self.value.z=val

    @property
    def length(self):
        return self.value.length()
    
    def normalize_ip(self):
        self.value.normalize_ip()
    
    def scale(self, val):
        self.value*=val

class FrictionForce(Force):
    def __init__(self, value=0.9):
        Force.__init__(self, vec3())
        self.friction = value
    
    def get(self, object:Union['PhysicsComponent',None]=None):
        if object and object._vel.length_squared()>0:
            return -self.friction * object._vel.normalize()*(object._vel.length()**1.2) if object.get_world_position().z<1 else vec3()
        return vec3()

class GravityForce(Force):
    def __init__(self, strength:float=-2, axis=None):
        Force.__init__(self, (axis if axis else vec3(0, 0, 1))*strength)

"""
class CollisionPoint:
    def __init__(self, a:None|vec3, b:None|vec3) -> None:
        self._a:vec3 = a if a else vec3()
        self._b:vec3 = b if b else vec3()
        self._normal:vec3=(self._b-self._a).normalize()
        self._depth:float=(self._b-self._a).length()
        pass

class Collision:
    def __init__(self, objectA:'PhysicsComponent', objectB:'PhysicsComponent', collision_point:CollisionPoint) -> None:
        self._objA = objectA
        self._objB = objectB
        self._collision_point = collision_point

class Solver:
    def __init__(self) -> None:
        pass

    def solve(self, collisions:list[Collision], dt:float):
        pass
"""

class PhysicsComponent(DrawableComponent, SceneComponent):
    def __init__(self, parent, pos=None, mass:float=1, world:None|World=None):
        DrawableComponent.__init__(self, parent, pos)
        self._physics_world : PhysicsWorld = world.get_physics_world() if world else Globals.game.get_world().get_physics_world()
        self._mass = mass
        self._vel = vec3()
        self._last_vel = vec3()
        self._acc = vec3()

        self._simulate_physics = True
        self._collide = True

        self._bounding_box = BoundingBox(self, -self._size/2, self._size/2)

        self._collider = BoxCollider(self, self._size)

        self._forces:list[Force] = []
        self._forces_count:int   = 0

        self._physics_world.register_physics_component(self)
    
    @SceneComponent.size.setter
    def size(self, s):
        self._size = s
        self._bounding_box = BoundingBox(self, self._pos-self._size/2, self._pos+self._size/2)
        self._collider.set_size(self._size)
    
    def set_size(self, size:vec3):
        DrawableComponent.set_size(self, size)
        self._bounding_box = BoundingBox(self, self._pos-self._size/2, self._pos+self._size/2)
        self._collider.set_size(self._size)
        return self
    
    @property
    def mass(self) -> float:
        return self._mass
    
    def set_mass(self, mass:float) -> 'PhysicsComponent':
        self._mass = mass
        return self
    
    def set_simulate_physics(self, simulate:bool) -> 'PhysicsComponent':
        self._simulate_physics = simulate
        return self
    
    def set_collide(self, collide:bool) -> 'PhysicsComponent':
        self._collide = collide
        return self
    
    def get_bounding_box(self) -> BoundingBox:
        return self._bounding_box
    
    def set_physics_world(self, pworld:PhysicsWorld):
        self._physics_world = pworld
    
    def get_collider(self) -> Collider:
        return self._collider

    def draw(self, screen:pygame.Surface):
        if Settings.debug_level&debugLevel.physics:
            Globals.game.draw_debug_box((set_z(self.get_world_position()+self._bounding_box._begin, 0)),
                                        (set_z(self.get_world_position()+self._bounding_box._end, 0)), color=vec3(255, 150, 0), thickness=1)
        pass

    def apply_force(self, force:Force) -> 'PhysicsComponent':
        if self._forces_count>=len(self._forces):
            self._forces.append(force)
            self._forces_count+=1
        else:
            self._forces[self._forces_count]=force
            self._forces_count+=1
        return self
    
    def tick(self, dt:float):
        if not self._simulate_physics: return

        self._acc = vec3()
        for i in range(self._forces_count):
            f = self._forces[i].get(self)
            self._acc += f
            if Settings.debug_level&debugLevel.physics: Globals.game.draw_debug_vector(self.get_world_position(), self.get_world_position()+0.1*f)

        self._forces_count = 0
        # log("Acc??l??ration : {}".format(self.acc))
        self._last_vel = self._vel
        self._vel += self._acc / self._mass * dt

        vel = self._vel * dt

        if vel.x<0:
            if self._pos.x+vel.x>self._physics_world._limits[0].x:
                self._pos.x += vel.x
            else:
                self._vel.x = 0
                self._pos.x = self._physics_world._limits[0].x
        
        if vel.x>0:
            if self._pos.x+vel.x<self._physics_world._limits[1].x:
                self._pos.x += vel.x
            else: 
                self._vel.x = 0
                self._pos.x = self._physics_world._limits[1].x
        
        if vel.y<0:
            if self._pos.y+vel.y>self._physics_world._limits[0].y:
                self._pos.y += vel.y
            else:
                self._vel.y = 0
                self._pos.y = self._physics_world._limits[0].y
        
        if vel.y>0:
            if self._pos.y+vel.y<self._physics_world._limits[1].y:
                self._pos.y += vel.y
            else:
                self._vel.y = 0
                self._pos.y = self._physics_world._limits[1].y

        if vel.z<0:
            if self._pos.z+vel.z>self._physics_world._limits[0].z:
                self._pos.z += vel.z
            else:
                self._vel.z = 0
                self._pos.z = self._physics_world._limits[0].z
        
        if vel.z>0:
            if self._pos.z+vel.z<self._physics_world._limits[1].z:
                self._pos.z += vel.z
            else:
                self._vel.z = 0
                self._pos.z = self._physics_world._limits[1].z
        
        if Settings.debug_level&debugLevel.physics: Globals.game.draw_debug_vector(self._pos, self._pos+0.1*self._vel, (10,255,10))
        # Globals.game.draw_debug_box(self._pos-set_z(self.size/2, 0), self._pos+set_z(self.size/2, 0), (0, 0, 255), thickness=1)

class SpriteComponent(DrawableComponent):
    def __init__(self, parent, pos=vec3(), size=vec2(1, 1), image_name="default"):
        SceneComponent.__init__(self, parent=parent, pos=pos)
        Drawable.__init__(self)
        self._draw_size = size
        self._sprite = Globals.game.load_image(image_name, size=self._draw_size)
        self._size_locked = False
        self._skip_resize = False
        self._draw_offset:vec2 = vec2()
        self._anchor:vec2 = vec2(0.5, 0.5)
    
    def draw(self, screen:pygame.Surface):
        Drawable.draw(self, screen)
        if self._sprite and self.is_visible():
            draw_pos = Globals.game.camera.world_to_screen(self.get_world_position()-vec3(self._size.x*(1-self._anchor.x), self._size.y*self._anchor.y, 0))
            self._draw_size = Globals.game.camera.world_size2_to_screen(self.size.xy) if not self._skip_resize else self._sprite.get_size()
            if Settings.debug_level&debugLevel.drawing:
                Globals.game.draw_debug_dot(draw_pos+self._draw_offset)
                t = vec2(self._draw_size.x*self._anchor.x, self._draw_size.y*self._anchor.y)
                Globals.game.draw_debug_rectangle(draw_pos - t + self._draw_offset, draw_pos - t + self._draw_size + self._draw_offset, (255, 0, 0))
            if not self._skip_resize and self._sprite.size != self._draw_size:
                if (not self._size_locked):
                    # log("Size if wrong, reloading sprite", logTypes.warning)
                    self._sprite = Globals.game.load_image(self._sprite.name, self._sprite.path, self._draw_size)
                else:
                    self._sprite.resize(self._draw_size)
            screen.blit(self._sprite.get_data(), draw_pos + self._draw_offset)
    
    def get_sprite(self) -> Image:
        return self._sprite
    
    def set_sprite(self, sprite:Image) -> 'SpriteComponent':
        self._sprite = sprite
        return self
    
    def set_draw_offset(self, offset:vec2):
        self._draw_offset = offset
    
    def get_draw_offset(self) -> vec2:
        return self._draw_offset
    
    def set_anchor(self, anchor:vec2) -> 'SpriteComponent':
        self._anchor = anchor
        return self
    
    def get_anchor(self) -> vec2:
        return self._anchor

class AnimatedSprite(SpriteComponent):
    def __init__(self, parent, pos=vec3(), size=vec2(1, 1), image_names:list[str]=["default"], sprite_time:float=0.5):
        assert len(image_names)>0, "AnimatedSprite must have at least one image"
        self.draw_size = Globals.game.camera.world_size2_to_screen(size.xy)
        self._sprites = [Globals.game.load_image(image_name, size=self.draw_size) for image_name in image_names]
        SpriteComponent.__init__(self, parent=parent, pos=pos, size=size, image_name=image_names[0])
        self._current_sprite = 0
        self._time = 0
        self._time_per_sprite = sprite_time
        self._loop = True
        self._playing = True
    
    def draw(self, screen:pygame.Surface):
        if self._playing:
            self._time += Globals.game.get_delta_time()
            if self._time > self._time_per_sprite:
                self._time -= self._time_per_sprite
                self._current_sprite += 1
                if self._current_sprite >= len(self._sprites):
                    if self._loop:
                        self._current_sprite = 0
                    else:
                        self._current_sprite = len(self._sprites)-1
                        self._playing = False
        self.sprite = self._sprites[self._current_sprite]
        SpriteComponent.draw(self, screen)
        

class Widget:
    def __init__(self, parent, pos:None|vec2=None, size:None|vec2=None, color=vec3(255, 255, 255)):
        self._parent = parent
        self._pos:vec2 = pos if pos else vec2()
        self._size:vec2 = size if size else vec2()
        self._color = color
        self._children:list[Widget] = []
        self._visible = True
    
    def register(self):
        pass
    
    def debug_draw(self):
        if self._visible:
            Globals.game.draw_debug_box(self._pos, self._pos+self._size, self._color)
            for child in self._children:
                child.debug_draw()
    def on_click(self):
        pass

class Button(Widget):
    def __init__(self, parent, pos:None|vec2=None, size:None|vec2=None, color=vec3(255, 255, 255), text=""):
        Widget.__init__(self, parent, pos, size, color)
        self._text = text
        self._text_color = vec3(0, 0, 0)
        self._text_size = 1
        self._callback:callable = None
        self._button:None|pygame_gui.elements.UIButton = None
    
    def register(self) -> 'Button':
        relative_rect = pygame.Rect(self._pos.x, self._pos.y, self._size.x, self._size.y)
        self._button = pygame_gui.elements.UIButton(relative_rect=relative_rect,
                                            text=self._text,
                                            manager=Globals.game.get_ui_manager(),
                                            anchors={"left": "left", "centery":"centery"})
        Globals.game._gui_refs[self._button]=self
        # Globals.game.get_ui_manager().set_visual_debug_mode(True)
        # log(f"{}")
        return self
    
    def set_callback(self, callback:callable) -> 'Button':
        self._callback = callback
        return self
    
    def on_click(self):
        if self._callback:
            self._callback()
    
    def is_valid(self):
        return self._button!=None

class UIImage(Widget):
    def __init__(self, parent, pos: None | vec2 = None, size: None | vec2 = None, image:str="default"):
        Widget.__init__(self, parent, pos, size)
        self._image = Globals.game.load_image(image)
        self._widg = None
    
    def register(self):
        relative_rect = pygame.Rect(self._pos.x, self._pos.y, self._size.x, self._size.y)
        self._widg = pygame_gui.elements.UIImage(relative_rect=relative_rect,
                                                manager=Globals.game.get_ui_manager(),
                                                image_surface=self._image.get_data())
        Globals.game._gui_refs[self._widg]=self
        return self
    
    def hide(self):
        self._widg.hide()
    
    def show(self):
        self._widg.show()

def testSlimyEngine():
    pass

if __name__=="__main__":
    testSlimyEngine()