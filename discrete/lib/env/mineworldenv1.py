import functools
from collections import Counter
from random import Random
from typing import Tuple, TypeVar, Union, List, Dict, Collection

import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, circleShape, fixtureDef, contactListener)
import numpy as np
from gym import spaces
from collections import Counter
import copy
from copy import deepcopy

from discrete.lib.env.gridenv import GridEnv
from discrete.lib.env.saveloadenv import SaveLoadEnv
from discrete.lib.env.util import element_add

# for the continuous env

Goal_reached_Dist = 0.4
COLLISION_DIST = 0.35
hit_wall_reward = -0.1



class MineWorldTileType:
    """A single special tile in the mine world"""

    def __init__(self, consumable: bool, inventory_modifier: Counter, action_name: str, grid_letter: str,
                 wall: bool = False, reward: int = 0, terminal: bool = False, inventory_requirements: Counter = None,
                 movement_requirements: Counter = None):
        """
        :param consumable: Does this tile disappear after being activated
        :param inventory_modifier: How does this modify the inventory (e.g. wood -2, desk +1)
        :param action_name: What atomic proposition should be true the round that this tile is activated
        :param grid_letter: What letter should be displayed on the grid
        """
        self.consumable = consumable
        self.inventory = inventory_modifier
        self.action_name = action_name
        self.grid_letter = grid_letter
        self.wall = wall
        self.reward = reward
        self.terminal = terminal
        self.inventory_requirements = inventory_requirements or Counter()
        self.movement_requirements = movement_requirements or Counter()

    def apply_inventory(self, prev_inventory: Counter):
        """
        Get the new inventory of the player after interacting with this tile, or errors if the player is unable to
        interact with the tile
        :param prev_inventory: The current inventory of the player
        """

        # Apply all the inventory changes and make sure that no item is negative
        new_inv = prev_inventory.copy()
        new_inv.update(self.inventory)
        if any([(new_inv[i] < 0) for i in new_inv]):
            raise ValueError()
        else:
            return new_inv

    def meets_requirements(self, current_inventory: Counter):
        inv_requirements_temp = current_inventory.copy()
        inv_non_neg_temp = current_inventory.copy()

        inv_requirements_temp.subtract(self.inventory_requirements)
        inv_non_neg_temp.update(self.inventory)

        requirements_ok = not any([(inv_requirements_temp[i] < 0) for i in inv_requirements_temp])
        inv_non_neg_ok = not any([(inv_non_neg_temp[i] < 0) for i in inv_non_neg_temp])

        return requirements_ok and inv_non_neg_ok
    
    def move_requirements(self, current_inventory: Counter):
        inv_requirements_temp = current_inventory.copy()
        inv_requirements_temp.subtract(self.movement_requirements)

        requirements_ok = not any([(inv_requirements_temp[i] < 0) for i in inv_requirements_temp])

        return requirements_ok

    @staticmethod
    def from_dict(dict):
        wall = dict.get("wall", False)
        reward = dict.get("reward", 0)
        terminal = dict.get("terminal", False)
        inventory_requirements = Counter(dict.get("inventory_requirements", {}))
        return MineWorldTileType(consumable=dict["consumable"], inventory_modifier=Counter(dict["inventory_modifier"]),
                                 action_name=dict["action_name"], grid_letter=dict["grid_letter"], wall=wall,
                                 reward=reward, terminal=terminal, inventory_requirements=inventory_requirements)


T = TypeVar("T")
MaybeRand = Union[T, str]


class TilePlacement:
    def __init__(self, tile: MineWorldTileType, fixed_placements: Collection[Tuple[int, int]] = tuple(),
                 random_placements: int = 0):
        self.tile = tile
        self.fixed_placements = fixed_placements
        self.random_placements = random_placements

    @staticmethod
    def from_dict(dict):
        tile = MineWorldTileType.from_dict(dict["tile"])
        fixed_raw = dict.get("fixed_placements", [])
        fixed_placements = [tuple(coord) for coord in fixed_raw]
        random_placements = dict.get("random_placements", 0)
        return TilePlacement(tile=tile,
                             fixed_placements=fixed_placements,
                             random_placements=random_placements)


class InventoryItemConfig:
    def __init__(self, name: str, default_quantity: int, capacity: int):
        """
        :param name: Name of the item, like wood or iron
        :param default_quantity: How many of these items to start with
        :param capacity: Maximum amount of this item the agent can hold. Also used for scaling of NN inputs.
        """
        self.name = name
        self.default_quantity = default_quantity
        self.capacity = capacity

    @staticmethod
    def from_dict(dict):
        return InventoryItemConfig(**dict)


class MineWorldConfig:
    def __init__(self, shape: Tuple[int, int], initial_position: Union[Tuple[int, int], None],
                 placements: List[TilePlacement], inventory: List[InventoryItemConfig]):
        self.placements = placements
        self.shape = shape
        self.initial_position = initial_position
        self.inventory = inventory

    @staticmethod
    def from_dict(dict):
        shape = tuple(dict["shape"])
        ip = dict["initial_position"]
        initial_position = ip if ip is None else tuple(ip)
        placement = [TilePlacement.from_dict(i) for i in dict["placements"]]
        inventory = list(map(InventoryItemConfig.from_dict, dict["inventory"]))

        return MineWorldConfig(shape=shape, initial_position=initial_position, placements=placement,
                               inventory=inventory)


def n_hot_grid(shape: Tuple[int, int], grid_positions: Union[None, List[Tuple[int, int]]], grid=None):
    if grid is None:
        grid = np.zeros(shape, dtype=np.int8)

    if grid_positions is None:
        grid_positions = []

    for pos in grid_positions:
        grid[pos] = 1

    return grid


def const_plane(shape, val):
    result = np.full(shape, val)
    return result


@functools.lru_cache(16384)
def obs_rewrite(shape, obs):
    position, tile_locs, inventories = obs
    # Convert to float?
    position_tile_layers = tuple(n_hot_grid(shape, layer) for layer in ((position,), *tile_locs))
    inventory_layers = tuple(np.full(shape, layer, dtype=np.int8) for layer in inventories)
    return np.stack((*position_tile_layers, *inventory_layers), axis=0)



class AgentTileContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)
        self.agent_tile_contacts = set()
    
    def BeginContact(self, contact):
        fA = contact.fixtureA
        fB = contact.fixtureB
        if "type" in fA.userData and "type" in fB.userData:
            types = (fA.userData["type"], fB.userData["type"])
            if "agent" in types and "tile" in types:
                tile_fixture = fA if fA.userData["type"] == "tile" else fB
                self.agent_tile_contacts.add(tile_fixture)
    
    def EndContact(self, contact):
        fA = contact.fixtureA
        fB = contact.fixtureB
        if "type" in fA.userData and "type" in fB.userData:
            types = (fA.userData["type"], fB.userData["type"])
            if "agent" in types and "tile" in types:
                tile_fixture = fA if fA.userData["type"] == "tile" else fB
                if tile_fixture in self.agent_tile_contacts:
                    self.agent_tile_contacts.remove(tile_fixture)
 
    def reset(self):
        self.agent_tile_contacts.clear()

class MineWorldEnvContinuous(GridEnv, SaveLoadEnv):

    @staticmethod
    def from_dict(dict):
        return MineWorldEnvContinuous(MineWorldConfig.from_dict(dict))
    
    def __init__(self, config: MineWorldConfig, width=7.0, height=7.0, time_step=1.0/60.0, seed=None, *args, **kwargs):
        super().__init__(shape=config.shape, *args, **kwargs)
        self.config = config
        self.width = width
        self.height = height
        self.time_step = time_step

        self.rng = np.random.RandomState(seed=seed)

        # Action = (force_x, force_y, interact_flag)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: [agent_x, agent_y, agent_vx, agent_vy, inventory...]
        obs_dim = 4 + len(config.inventory)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.inventory = Counter()
        self.default_inventory = Counter({inv.name: inv.default_quantity for inv in self.config.inventory})

        self.world = None
        self.agent_body = None
        self.tiles = []  
        self.done = True

        # Parameters for tuning
        self.force_magnitude = 7.0
        self.damping = 1.0
        
        # Set up a custom contact listener
        self.contact_listener = AgentTileContactListener()
        self.world = world(gravity=(0, 0), doSleep=True)
        self.world.contactListener = self.contact_listener

    def reset(self):
        self.done = False
        # print("Reset Begins")
        
        # Destroy the old world if it exists.
        if self.world is not None:
            for body in list(self.world.bodies):
                self.world.DestroyBody(body)

        
        # Create a new Box2D world and attach the contact listener.
        self.world = world(gravity=(0, 0), doSleep=True)
        self.world.contactListener = self.contact_listener

        # Reset inventory
        self.inventory = self.default_inventory.copy()

        # Place agent
        if self.config.initial_position is None:
            ax = self.rng.uniform(0, self.width)
            ay = self.rng.uniform(0, self.height)
        else:
            ax, ay = self.config.initial_position
            ax, ay = float(ax), float(ay)

        self.agent_body = self.world.CreateDynamicBody(
            position=(ax, ay),
            angle=0.0, 
            linearDamping=self.damping,
            angularDamping=self.damping
        )
        circle = circleShape(radius=0.3)
        agent_fixture = self.agent_body.CreateFixture(shape=circle, density=1.0, friction=0.3)
        agent_fixture.userData = {"type": "agent"}

        # Place tiles
        self.tiles.clear()
        self._place_tiles_box2d()

        # Step the world a few times to let things settle
        for _ in range(5):
            self.world.Step(self.time_step, 6, 2)
        self.contact_listener.reset()
        # print('reset done')

        return self._get_obs()

    def step(self, action):
        assert not self.done, "Episode finished, call reset() first."
        fx, fy, raw_interact_flag = action
        fx = float(np.clip(fx, -1, 1))
        fy = float(np.clip(fy, -1, 1))
        interact_flag = 1 if raw_interact_flag > 0 else 0

        # Apply force to the agent's body center
        self.agent_body.ApplyForceToCenter((fx * self.force_magnitude, fy * self.force_magnitude), wake=True)

        self.world.Step(self.time_step, 6, 2)

        # Boundary management: Clamp agent within [0, width] x [0, height]
        px, py = self.agent_body.position
        clamped_px = np.clip(px, 0.0, self.width)
        clamped_py = np.clip(py, 0.0, self.height)
        if (clamped_px, clamped_py) != (px, py):
            self.agent_body.position = (clamped_px, clamped_py)
            # Optionally, also set velocity to zero when clamped:
            self.agent_body.linearVelocity = (0.0, 0.0)

        reward = 0.0
        tile_action_names = set()

        # Handle interactions if the interact_flag is active
        if interact_flag > 0:
            
            for tile_fixture in list(self.contact_listener.agent_tile_contacts):
                tile_data = tile_fixture.userData.get("tile_data", None)
                if tile_data is None or not tile_data.get("active", False):
                    continue
                tile_type = tile_data["mineworld_type"]
                if tile_type.wall:
                    continue  
                if tile_type.meets_requirements(self.inventory):
                    try:
                        new_inv = tile_type.apply_inventory(self.inventory)

                    except ValueError:
                        continue 

                    # Clamp inventory to capacity
                    for inv_cfg in self.config.inventory:
                        if new_inv[inv_cfg.name] > inv_cfg.capacity:
                            new_inv[inv_cfg.name] = inv_cfg.capacity
                    self.inventory = new_inv
                    reward += tile_type.reward
                    

                    # Record the tile's action name (useful for terminal checking)
                    tile_action_names.add(tile_type.action_name)

                    if tile_type.consumable:
                        # Remove the tile from the world
                        tile_body = tile_data["body"]
                        self.world.DestroyBody(tile_body)
                        tile_data["active"] = False
                    if tile_type.terminal:
                        self.done = True
                        print ('done')
                        break
            
            self.contact_listener.reset()

        obs = self._get_obs()
        info = {
            "inventory": dict(self.inventory),
            "position": (self.agent_body.position.x, self.agent_body.position.y),
            "tile_action_names": list(tile_action_names)
        }
        print(info)
        return obs, reward, self.done, info


    def render(self, mode='human'):
        agent_pos = self.agent_body.position
        agent_vel = self.agent_body.linearVelocity
        print(f"Agent pos=({agent_pos.x:.2f}, {agent_pos.y:.2f}), vel=({agent_vel.x:.2f}, {agent_vel.y:.2f})")
        print(f"Inventory: {dict(self.inventory)}")
        for tile in self.tiles:
            pos = tile["body"].position
            print(f"{tile['mineworld_type'].grid_letter} | Active: {tile['active']} | Pos: ({pos.x:.2f}, {pos.y:.2f})")

    def _place_tiles_box2d(self):
        """
        Create static bodies or sensor fixtures for each tile in the configuration.
        For wall tiles, we create a static shape with physical collision.
        For non-wall tiles, we create a sensor fixture for overlap detection.
        """
        for placement in self.config.placements:
            tile_type = placement.tile
            # Fixed placements
            for (fx, fy) in placement.fixed_placements:
                self._create_tile_body(tile_type, fx, fy)
            # Random placements
            for _ in range(placement.random_placements):
                rx = self.rng.uniform(0, self.width)
                ry = self.rng.uniform(0, self.height)
                self._create_tile_body(tile_type, rx, ry)

    def _create_tile_body(self, tile_type, x, y):
        if tile_type.wall:
            body = self.world.CreateStaticBody(position=(x, y))
            box = polygonShape(box=(0.5, 0.5))
            fix = body.CreateFixture(shape=box, density=0, friction=0.5)
            fix.userData = {"type": "tile", "tile_data": None}
            tile_data = {"mineworld_type": tile_type, "body": body, "active": True}
            self.tiles.append(tile_data)
        else:
            body = self.world.CreateStaticBody(position=(x, y))
            circle = circleShape(radius=0.3)
            fix = body.CreateFixture(shape=circle, isSensor=True)
            tile_data = {"mineworld_type": tile_type, "body": body, "active": True}
            fix.userData = {"type": "tile", "tile_data": tile_data}
            self.tiles.append(tile_data)

    def _get_obs(self):
        # Construct the observation from agent state and inventory
        ax, ay = self.agent_body.position
        vx, vy = self.agent_body.linearVelocity
        obs_list = [ax, ay, vx, vy]
        for inv_cfg in self.config.inventory:
            obs_list.append(float(self.inventory[inv_cfg.name]))
        return np.array(obs_list, dtype=np.float32)

    def save_state(self):
        
        agent_pos = (self.agent_body.position.x, self.agent_body.position.y)
        agent_vel = (self.agent_body.linearVelocity.x, self.agent_body.linearVelocity.y)
        tiles_state = [tile.copy() for tile in self.tiles]
        return {
            "agent_pos": agent_pos,
            "agent_vel": agent_vel,
            "done": self.done,
            "tiles": tiles_state,
            "inventory": self.inventory.copy()
        }

    def load_state(self, state):
        pos = state["agent_pos"]
        vel = state["agent_vel"]
        self.agent_body.position = pos 
        self.agent_body.linearVelocity = vel
        self.done = state["done"]
        self.tiles = [tile.copy() for tile in state["tiles"]]
        self.inventory = state["inventory"].copy()


class MineWorldEnv(GridEnv, SaveLoadEnv):
    """A basic minecraft-like environment, with a global view of the state space"""
    @staticmethod
    def from_dict(dict):
        return MineWorldEnv(MineWorldConfig.from_dict(dict))

    def __init__(self, config: MineWorldConfig, *args, **kwargs):
        super().__init__(shape=config.shape, *args, **kwargs)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0, 1,
                                            shape=(1 + len(config.placements) + len(config.inventory), *config.shape),
                                            dtype=np.int8)
        self.config = config
        self.default_inventory = Counter(
            {inv_type.name: inv_type.default_quantity for inv_type in self.config.inventory})
        self.rand = Random()

        """
        Up: 0,
        Right:1,
        Down: 2,
        Left: 3,
        No-op: 4,
        Tile action: 5"""

        self.done = True
        self.position: Tuple[int, int] = (0, 0)
        self.special_tiles: Dict[Tuple[int, int], MineWorldTileType] = dict()
        self.inventory = Counter()

    def step(self, action: int):
        assert self.action_space.contains(action)
        assert not self.done

        action_names = set()

        reward = 0

        if action < 5:
            # Movement or no-op
            action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
            new_place = element_add(self.position, action_offsets[action])

            can_move = self._in_bounds(new_place)

            if new_place in self.special_tiles:
                tile = self.special_tiles[new_place]
                if tile.wall or not tile.move_requirements(self.inventory):
                    can_move = False

            if can_move:
                self.position = new_place
        else:
            if self.position in self.special_tiles:
                this_tile: MineWorldTileType = self.special_tiles[self.position]
                if this_tile.meets_requirements(self.inventory):
                    new_inv = this_tile.apply_inventory(self.inventory)
                    for inv_config in self.config.inventory:
                        if new_inv[inv_config.name] > inv_config.capacity:
                            new_inv[inv_config.name] = inv_config.capacity
                    self.inventory = new_inv
                    action_names.add(this_tile.action_name)
                    if this_tile.consumable:
                        del self.special_tiles[self.position]
                    if this_tile.terminal:
                        self.done = True
                    reward += this_tile.reward

        info = {
            'tile_action_names': action_names,
            'inventory': self.inventory.copy(),
            'position': self.position
        }

        return obs_rewrite(self.shape, self._get_observation()), reward, self.done, info

    def seed(self, seed=None):
        self.rand.seed(seed)

    def reset(self):
        self.done = False
        self.position = self.config.initial_position
        if not self.position:
            self.position = self.rand.randrange(0, self.shape[0]), self.rand.randrange(0, self.shape[1])
        self.inventory = self.default_inventory.copy()
        self.special_tiles = self._get_tile_positioning()

        return obs_rewrite(self.shape, self._get_observation())

    def _get_tile_positioning(self) -> Dict[Tuple[int, int], MineWorldTileType]:

        tiles = {}

        for tile_type in self.config.placements:
            for fixed in tile_type.fixed_placements:
                tiles[fixed] = tile_type.tile

        # noinspection PyTypeChecker
        all_spaces = set(np.ndindex(self.config.shape))
        open_spaces = all_spaces.difference(tiles.keys())
        if (0, 0) in open_spaces:
            open_spaces.remove((0, 0))

        for tile_type in self.config.placements:
            tile, num_placements = tile_type.tile, tile_type.random_placements
            spaces = self.rand.sample(open_spaces, num_placements)
            open_spaces.difference_update(spaces)

            for space in spaces:
                tiles[space] = tile

        return tiles

    def _get_observation(self):

        tiles = tuple(
            frozenset(space for space, content in self.special_tiles.items() if content is placement.tile) for
            placement in self.config.placements)

        inv = tuple(self.inventory[inv_config.name] for inv_config in self.config.inventory)

        return (
            self.position,
            tiles,
            inv
        )

    def render(self, mode='human'):
        def render_func(x, y):
            agent_str = "A" if self.position == (x, y) else " "
            tile_str = self.special_tiles[(x, y)].grid_letter if (x, y) in self.special_tiles else " "
            return agent_str + tile_str, False, False

        print(self._render(render_func, 2), end="")
        print(dict(self.inventory))

    def save_state(self):
        return self.position, self.done, self.special_tiles.copy(), self.inventory.copy()

    def load_state(self, state):
        self.position, self.done, spec_tile, inv = state
        self.special_tiles = spec_tile.copy()
        self.inventory = inv.copy()
