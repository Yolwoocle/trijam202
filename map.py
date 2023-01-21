import pygame
from engine.slimyengine import *

class Tile:
    def __init__(self, name:str) -> None:
        """Tile represents a tile in the map.

        Args:
            name (str): Name of the Tile.
        """
        
        self.name = name



class Map(Drawable):
    def __init__(self, size:vec3, tile_size:int) -> None:
        """Map that stores data about the world.

        Args:
            size (vec3): Size of the map area.
            tile_size (int): Size of a tile of the map.
        """
        Drawable.__init__(self)
        
        self.size = size
        self.tile_size = tile_size
        
        self.grid = [[[None for i in range(size.z)] for j in range(size.y)] for j in range(size.x)]
        self.dynamic_tiles = []
        
        self.map_surface = pygame.Surface((size.x * tile_size, size.y * tile_size))
        
        
    def in_bounds(self, x:int, y:int) -> bool:
        return (0 <= x < self.width and 0 <= y < self.height)
        
        
    def get(self, x:int, y:int):
        if not self.in_bounds(x, y):
            return None
        return self.grid[x][y]


    def set(self, x:int, y:int, tile:Tile):
        if not self.in_bounds(x, y):
            return False
        
        old_obj = self.grid[x][y]
        
        self.grid[x][y] = tile
        if tile.is_dynamic:
            self.dynamic_tiles.append(tile)
        
        if old_obj.is_dynamic:
            old_obj._delete_me = True 
            
        return old_obj
    

    def tick(self, dt):
        for tile in self.dynamic_tiles: 
            tile.tick()
        
    def draw(self, screen):
        for z in range(self.size.z):
            for y in range(self.size.y):
                for x in range(self.size.x):
                    self.draw_tile(screen, self.grid[x][y][z], x, y, z)

    def draw_tile(self, screen, tile, x, y, z):
        tile.texture.blit(screen, self.tile_size * vec2(x, y))