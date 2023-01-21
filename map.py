import pygame
from engine.slimyengine import *

class Tile:
    def __init__(self, name:str) -> None:
        """Tile represents a tile in the map.

        Args:
            name (str): Name of the Tile.
        """
        
        self.name = name


class StaticTile(Tile):
    def __init__(self, name: str) -> None:
        super().__init__(name, )


class DynamicTile(Tile):
    def __init__(self, name: str) -> None:
        super().__init__(name)
    
    def tick(self, dt):
        raise Exception("`tick` function of DynamicTile is not defined")


class StaticTiles:
    Floor = StaticTile("floor")
    Wall = StaticTile("wall")



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
        
        self.grid = [
            [
                [None for i in range(int(size.z))] for j in range(int(size.y))
            ] for j in range(int(size.x))
        ]
        self.dynamic_tiles = []
        
        self.map_surface = pygame.Surface((size.x * tile_size, size.y * tile_size))
        
        
    def in_bounds(self, pos:vec3) -> bool:
        return (
            0 <= pos.x < self.size.x and 
            0 <= pos.y < self.size.y and 
            0 <= pos.z < self.size.z
        )
        
        
    def get(self, pos:vec3):
        if not self.in_bounds(pos):
            return None
        return self.grid[pos.x][pos.y][pos.z]


    def set(self, pos:vec3, tile:Tile):
        if not self.in_bounds(pos):
            return False
        
        old_tile = self.grid[pos.x][pos.y][pos.z]
        
        self.grid[pos.x][pos.y][pos.z] = tile
        if isinstance(tile, DynamicTile):
            self.dynamic_tiles.append(tile)
        
        if isinstance(old_tile, DynamicTile):
            old_tile._delete_me = True 
            
        return old_tile
    

    def tick(self, dt):
        for tile in self.dynamic_tiles: 
            tile.tick()
        
    def draw(self, screen):
        for z in range(int(self.size.z)):
            for y in range(int(self.size.y)):
                for x in range(int(self.size.x)):
                    self.draw_tile(screen, self.grid[x][y][z], x, y, z)

    def draw_tile(self, screen, tile, x, y, z):
        tile.texture.blit(screen, self.tile_size * vec2(x, y))