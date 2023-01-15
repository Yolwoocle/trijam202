# SlimyEngine reference

## Script de base


```python
from engine.slimyengine import *

game = Game().init().target_fps(60).set_background_color(Colors.darkgrey)
game.set_debug(False)
game.load_image("name", "path/to/texture.png")

world = game.get_world()

world.enable_physics()
world.get_physics_world().set_limits(vec3(-4, -4, 0), vec3(4, 4, math.inf))

scene = Scene()
world.load_scene(scene)

player = Pawn()

scene.register_actor(player)

scene.register_component(player.root)

game.update_size()
while game.is_alive():
    game.begin_frame()
    
    game.tick()

    game.end_frame()
game.quit()
```

---

## Les classes

> Seules figurent ici les classes essentielles/ les plus utilisées

### Game

- Gère le jeu dans sa globalité
- Une seule instance de cette classe est possible, lors de la construction la variable ```Globals.game``` est mise à jour en conséquence
- Gère les textures, via la méthode ```load_image("name", "path/to/texture.png")```
- Gère les fonts, via la méthode ```load_font("name", "path/to/font.ttf")```
- Gère les évènements, récupérés lors de l'appel à ```begin_frame()```

### World

- Possédé par la classe ```Game``` et *a priori* persistent pour la durée entière du jeu
- Permet de charger des scènes, c'est dans cette classe qu'on gère par exemple les scores, la progression, ce qui doit être persistent d'un niveau à l'autre. Coordonne les changements de niveau de manière générale
- Possède le ```PhysicsWorld``` qui est une représentation *physique* du niveau en cours, appelé chaque dixième de seconde pour mettre à jour les ```PhysicsObjects``` (actuellement non fonctionnel)
- Gère les systèmes de particules...bon ça ça va très vite changer, c'est une erreur de design x)
- Method ```tick``` -> update ```PhysicsWorld```, update ```Scene```, draw ```Scene```, draw lights

### Scene

- Ne possède *a priori* personne, garde uniquement des listes de références vers:
  - Les ```SceneComponent```, components avec une position/ rotation/ taille donc une présence dans le niveau
  - Les ```Actor```, entité abstraite qui ne possède pas de présence dans le monde. Possède une méthode tick, vide par défaut mais appelée (dans un ordre non déterminé) lors de l'update de la scène à chaque frame.
  - La ```Camera``` active. Unique, c'est le point de vue actuel de la scène. La caméra est un ```SceneComponent``` avec des méthodes en plus telles ```world_to_screen(position:vec3)->vec2``` qui transforme une position 3D (dans la scène) en une position 2D à afficher sur l'écran.

### Actor, Pawn, Component, ... c'est quoi ce bowdel?

- Un ```Actor``` est une sorte de conteneur logique, il sert à exécuter du code client, comme s'occuper du déplacement d'un personnage selon les entrées clavier. Il possède un ```self._root``` de type ```SceneComponent|None``` qui représente son ancrage (éventuel, peut-être None) dans la scène. Un actor n'a en soi ni position, ni possibilité d'être affiché. C'est la hierarchie ayant pour parent son ```self._root``` qui représente sa "matérialisation". Par exemple dans le cas d'un joueur classique, ```self._root``` sera un ```PhysicsComponent``` avec un collider de type capsule, et un ```SpriteComponent``` avec une texture de joueur sera attaché au root. Les ```Actor``` n'ont a priori pas de lien de parenté. La possiblité est laissée de définir un *parent* pour un actor mais (pour le moment) cela n'a aucun impact du point de vue du moteur...
- Un ```Pawn``` est un type d'```Actor``` qui implémente par défaut la structure définie précédemment, avec un ```PhysicsComponent``` en root, auquel est attaché un ```SpriteComponent``` (nom de variable: ```self._character```) et une ombre (```SpriteComponent``` qui reste constamment projeté sur le sol tout en suivant les coordonnées (x, y) du root). Cette classe très courte a tout intérêt *a priori* à être réimplémentée au cas par cas par le client, elle est là uniquement pour faciliter la mise en place rapide de prototypes.
- Un ```Component``` est un objet ayant pour vocation d'être membre d'une hiérarchie de ```Component```. Il possède un parent de type ```Component|None``` (peut-être None si le component est la racine de sa hiérarchie) et une liste d'enfants. Il possède une méthode ```update()```, typiquement utilisée par les ```SceneComponent``` pour mettre à jour les positions hiérarchiques (cf. paragraphe suivant).
- Un ```SceneComponent``` est un ```Component``` doté notamment d'une position. Cette position est **locale**, relative à celle de son parent. Ainsi pour déterminer la position dans le monde d'un ```SceneComponent``` il est nécessaire d'additioner récursivement chaque position, c'est pourquoi par soucis d'optimisation la position du parent est stockée (self._parent_pos) et mise à jour (en "descendant" la hiérarchie) uniquement en cas de modification. Pour avoir accès à la position "absolue" d'un ```SceneComponent``` il faut appeler la méthode ```component.get_world_position()``` qui sera donc pratiquement *systématiquement* celle utilisée.
- Un ```PhysicsComponent``` est un ```SceneComponent``` qui possède en plus une vélocitée, une accélération, une masse, une bounding_box, etc... Peut-être registered dans le ```PhysicsWorld``` pour voir sa position mise à jour selon les forces, la vélocitée, etc...
- Un ```Drawable``` est **uniquement** une classe qui implémente la méthode ```draw```, à implémenter dans chaque classe dérivée et à la discrétion de celles-ci (se conclue *a priori* par un ```Globals.game.screen.blit```)
- Un ```DrawableComponent``` est à la fois un ```SceneComponent``` et un ```Drawable```
- Un ```SpriteComponent``` est un ```DrawableComponent``` avec une texture (self.sprite) et la gestion automatique du redimensionnement pour rester pixel-perfect. Possiblité de mettre la variable membre ```self._size_locked``` à ```True``` pour empêcher ce comportement, mais risque de produire des images quelques peu incohérentes si la taille demandée pour le draw est éloignée de la taille réelle de la texture.
- Une ```Light``` est un ```DrawableComponent``` qui est logiquement censé lors du draw faire une commande du type         ```scene.get_light_map().blit(..., special_flags=pygame.BLEND_ADD)```. Il possède de plus une méthode ```render()``` pour permettre de précalculer la lumière. Cf. ```PointLight```
- ```ParticleEmitter``` est un ```Drawable``` chargé de gérer (émission, update, ...) un ensemble de particules
- ```ParticleSystem``` est un ```DrawableComponent``` qui possède une liste de ```ParticleEmitter```. C'est donc le ```ParticleSystem``` qui possède une position dans le monde, dont se servent les ```ParticleEmitter```. Pourquoi séparer les deux? Dans le cas d'un feu par exemple il est intéressant de le manipuler (déplacer, afficher/ cacher) en tant qu'unique ```ParticleSystem``` alors qu'il possède deux ```ParticleEmitter```: les flammes et la fumée
- Un ```DebugDraw``` est une sorte de ```Drawable``` avec une reference au ```Game``` en plus (existence plutôt historique et par soucis de clareté, pas d'intérêt particulier à ne pas simplement hériter de ```Drawable```). Destiné aux élément de debug, automatiquement caché en production.

---

## Le déroulement d'une frame

1) Appel par le client de ```Game.begin_frame()```
2) Appel par le client de ```Game.tick()```
3) Appel par le client de ```Game.end_frame()```