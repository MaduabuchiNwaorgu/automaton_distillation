import functools
from collections import Counter
from random import Random
from typing import Tuple, TypeVar, Union, List, Dict, Collection

import numpy as np
from gym import spaces

from automaton_transfer.lib.env.gridenv import GridEnv
from automaton_transfer.lib.env.saveloadenv import SaveLoadEnv
from automaton_transfer.lib.env.util import element_add


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
