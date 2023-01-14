from engine.slimyengine import *

class Map(Drawable):
    def __init__(self, w:int, h:int, tile_size:int) -> None:
        Drawable.__init__(self)
        
        self.width = w
        self.height = h
        self.tile_size = tile_size
        
        self.grid = [[None for i in range(h)] for j in range(w)]
        self.dynamic_tiles = []
        
        self.texture = ...
        
    def in_bounds(self, x:int, y:int):
        return (0 <= x < self.width and 0 <= y < self.height)
        
    def get(self, x:int, y:int):
        if not self.in_bounds(x, y):
            return None
        return self.grid[x][y]

    def set(self, x:int, y:int, tile):
        if not self.in_bounds(x, y):
            return False
        
        old_obj = self.grid[x][y]
        
        self.grid[x][y] = tile
        if type(tile) == type:
            self.dynamic_tiles.append(tile)
        
        if type(old_obj) == type:
            old_obj._delete_me = True 
        
        return old_obj

