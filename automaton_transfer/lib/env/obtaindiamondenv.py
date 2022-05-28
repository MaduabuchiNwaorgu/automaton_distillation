import abc
import random
from collections import Counter
from typing import Tuple, List, Dict, Type
import numpy as np
from gym import spaces

from automaton_transfer.lib.env.saveloadenv import SaveLoadEnv

actions_name = ['Noop', 'Forward', 'Backward', 'Right', 'Left', 'Craft', 'Mine Forward',
                'Mine Backward', 'Mine Right', 'Mine Left', 'Mine Down', 'Mine Up']


class Block(abc.ABC):
    def __init__(self, consumable: bool, action_name: str, grid_letter: str,
                 inventory_requirements: Counter, representation: int):
        self.consumable = consumable
        self.action_name = action_name
        self.grid_letter = grid_letter
        self.inventory_requirements = inventory_requirements
        self.representation = representation

    def break_block(self, inv):
        pass

    def apply_inventory(self):
        pass

    def requirements(self):
        pass

    def get_representation(self):
        return self.representation


class Air(Block):
    def __init__(self):
        super().__init__(consumable=False, action_name=None, grid_letter='a',
                         inventory_requirements=None, representation=0)

    def break_block(self, inv):
        return None, None


class Wood(Block):
    def __init__(self):
        super().__init__(consumable=True, action_name='wood', grid_letter='W',
                         inventory_requirements=None, representation=1)

    def break_block(self, inv):
        return Air(), Counter(wood=1)


class Grass(Block):
    def __init__(self):
        super().__init__(consumable=True, action_name=None, grid_letter='G',
                         inventory_requirements=None, representation=2)

    def break_block(self, inv):
        return Air(), None


class Stone(Block):
    def __init__(self):
        super().__init__(consumable=True, action_name='stone', grid_letter='S',
                         inventory_requirements=None, representation=3)

    def break_block(self, inv):
        if inv['woodpickaxe'] == 1:
            return Air(), Counter(stone=1)
        else:
            return None, None


class Iron(Block):
    def __init__(self):
        super().__init__(consumable=True, action_name='iron', grid_letter='I',
                         inventory_requirements=None, representation=4)

    def break_block(self, inv):
        if inv['stonepickaxe'] == 1:
            return Air(), Counter(iron=1)
        else:
            return None, None


class Diamond(Block):
    def __init__(self):
        super().__init__(consumable=True, action_name='diamond', grid_letter='D',
                         inventory_requirements=None, representation=5)

    def break_block(self, inv):
        if inv['ironpickaxe'] == 1:
            return Air(), Counter(diamond=1)
        else:
            return None, None


class Layer:
    def __init__(self, shape: Tuple[int, int], base_block: Type[Block], args: List[Dict]):
        # Args should consist of a list of dictionaries, with element of 'type', 'num', 'min_per', 'max_per'
        self.tiles = [[base_block() for _ in range(shape[0])] for _ in range(shape[1])]
        self.shape = shape
        n, m = shape[0], shape[1]

        for arg in args:
            num = arg.get('num')
            coords = random.sample(range(shape[0] * shape[1]), num)
            for coord in coords:
                self.tiles[coord % n][coord // n] = arg.get('type')()
                num_pocket = random.randint(arg.get('min_per'), arg.get('max_per'))
                for _ in range(1, num_pocket):
                    coord = np.random.choice([max(coord - 1, 0), min(coord + 1, n),
                                              min(coord + n, n), max(coord - n, 0)])  # Change later
                    self.tiles[coord % n][coord // n] = arg.get('type')()

    def get(self, coords: Tuple[int, int]):
        return self.tiles[coords[0]][coords[1]]

    def break_block_at(self, coords: Tuple[int, int], inv):
        new_block, update = self.tiles[coords[0]][coords[1]].break_block(inv)
        if new_block is not None:
            self.tiles[coords[0]][coords[1]] = new_block
        return update, True if new_block is not None else False

    @staticmethod
    def from_dict(dict):
        return Layer(dict.get('shape'), dict.get('base'), dict.get('args', {}))


def wood_layer(shape):
    return {'shape': shape, 'base': Air, 'args': [{'type': Wood, 'num': 10, 'min_per': 1, 'max_per': 1}]}


def grass_layer(shape):
    return {'shape': shape, 'base': Grass, 'args': []}


def stone_layer(shape):
    return {'shape': shape, 'base': Stone, 'args': []}


def iron_layer(shape):
    return {'shape': shape, 'base': Stone, 'args': [{'type': Iron, 'num': 2, 'min_per': 2, 'max_per': 4}]}


def diamond_layer(shape):
    return {'shape': shape, 'base': Stone, 'args': [{'type': Diamond, 'num': 1, 'min_per': 1, 'max_per': 1}]}


class ObtainDiamondGridworld(SaveLoadEnv):
    # @staticmethod
    # def from_dict(dict):
    #     dict = dict['config']
    #     return ObtainDiamondGridworld(shape=tuple(dict['shape']))

    def __init__(self, config: Dict):
        self.shape = config['shape']
        self.obs_shape = (5, 5)  # Should be odd for both
        self.prev_action = 0

        self.action_space = spaces.Discrete(11)

        # Observation should only be what is visible in a 3x5x5 area around the agent
        self.observation_space = spaces.Box(-2, 5, shape=(12,) + self.obs_shape)
        # self.observation_space = spaces.Box(-2, 5, shape=(8 + self.shape[-1],) + self.obs_shape)

        self.layers = []
        self.max_time = self.shape[2] * self.shape[0] * self.shape[1] // 4
        self.current_time = 0
        self.pos = (0, 0, 0)
        self.inv = Counter()
        self.prev_inv = self.inv.copy()

    def load_state(self, state):
        self.pos, self.layers, self.inv, self.current_time = state
        self.prev_inv = self.inv.copy()

    def step(self, action):
        """
        :param action: Actions can take on a value from 0 to 10
        0: Noop
        1: Forward
        2: Backward
        3: Right
        4: Left
        5: Craft
        6: Mine Forward (These just break the block and don't change the agent's location, except for break down)
        7: Mine Backward
        8: Mine Right
        9: Mine Left
        10: Mine Down
        11: Mine Up
        :return: The observation in the environment after the agent takes the action
        (0 represents air, 1 represents wood, 2 represents grass, 3 represents stone,
        4 represents iron, 5 represents diamond, -1 represents unseen, -2 represents border)
        """
        self.prev_action = action
        self.current_time += 1
        if action == 1:
            new_pos = (max(self.pos[0] - 1, 0), self.pos[1], self.pos[2])
            if isinstance(self.get(new_pos), Air):
                self.pos = new_pos
        elif action == 2:
            new_pos = (min(self.pos[0] + 1, self.shape[0] - 1), self.pos[1], self.pos[2])
            if isinstance(self.get(new_pos), Air):
                self.pos = new_pos
        elif action == 3:
            new_pos = (self.pos[0], min(self.pos[1] + 1, self.shape[1] - 1), self.pos[2])
            if isinstance(self.get(new_pos), Air):
                self.pos = new_pos
        elif action == 4:
            new_pos = (self.pos[0], max(self.pos[1] - 1, 0), self.pos[2])
            if isinstance(self.get(new_pos), Air):
                self.pos = new_pos
        elif action == 5:
            self.craft()
        elif action == 6:
            block_coords = (max(self.pos[0] - 1, 0), self.pos[1], self.pos[2])
            self.break_block_at(block_coords)
        elif action == 7:
            block_coords = (min(self.pos[0] + 1, self.shape[0] - 1), self.pos[1], self.pos[2])
            self.break_block_at(block_coords)
        elif action == 8:
            block_coords = (self.pos[0], min(self.pos[1] + 1, self.shape[1] - 1), self.pos[2])
            self.break_block_at(block_coords)
        elif action == 9:
            block_coords = (self.pos[0], max(self.pos[1] - 1, 0), self.pos[2])
            self.break_block_at(block_coords)
        elif action == 10:
            block_coords = (self.pos[0], self.pos[1], min(self.pos[2] + 1, self.shape[2] - 1))
            broken = self.break_block_at(block_coords)
            if broken:
                self.pos = block_coords
        elif action == 11:
            block_coords = (self.pos[0], self.pos[1], max(self.pos[2] - 1, 0))
            self.break_block_at(block_coords)
        return np.asarray(self.observe()), self.reward_tracker(), \
               self.inv['diamond'] == 1 or self.current_time > self.max_time, \
               {'obs': np.asarray(self.observe()), 'shape': self.shape, 'pos': self.pos, 'inventory': self.inv}

    def reset(self):
        self.inv.clear()
        self.prev_inv.clear()
        self.current_time = 0
        self.layers.clear()
        self.layers.append(Layer.from_dict(wood_layer(self.shape[:-1])))
        for _ in range(2):
            self.layers.append(Layer.from_dict(grass_layer(self.shape[:-1])))

        # Some arbitrary numbers for how much iron, for the purposes of the experiments it's 2-4
        # for both depths of 10 and 16
        num_iron_layers = np.random.randint(max(2, (self.shape[-1] - 3) // 5), max((self.shape[-1] - 3) // 4 + 1, 4))
        num_diamond_layers = np.random.randint(1, 4)
        iron_layers = np.random.choice([i for i in range(3, self.shape[-1] - 4)], num_iron_layers, replace=False)
        diamond_layers = np.random.choice([i for i in range(self.shape[-1] - 4, self.shape[-1])], num_diamond_layers,
                                          replace=False)

        for i in range(3, self.shape[-1]):
            if i in iron_layers:
                self.layers.append(Layer.from_dict(iron_layer(self.shape[:-1])))
            elif i in diamond_layers:
                self.layers.append(Layer.from_dict((diamond_layer(self.shape[:-1]))))
            else:
                self.layers.append(Layer.from_dict((stone_layer(self.shape[:-1]))))

        pos = random.randint(0, self.shape[0] * self.shape[1] - 1)
        self.pos = (pos % self.shape[0], pos // self.shape[0], 0)

        return np.asarray(self.observe())

    def render(self, mode='human'):
        print('\n' * 10)
        print(actions_name[self.prev_action])
        print(self.inv)
        for i in range(self.shape[0]):
            row = ''
            for j in range(self.shape[1]):
                if (i, j) != self.pos[:-1]:
                    row += f'{self.get((i, j, self.pos[2])).grid_letter} '
                else:
                    row += 'X '
            print(row)

    def save_state(self):
        return self.pos, self.layers, self.inv, self.current_time

    def get(self, coords):
        return self.layers[coords[2]].get(coords[:-1])

    def break_block_at(self, coords):
        inv_update, broken = self.layers[coords[2]].break_block_at(coords[:-1], self.inv)
        if inv_update is not None:
            self.inv += inv_update
        return broken

    def observe_layer(self, layer: Layer, coords: Tuple[int, int]):
        obs = [[-1 for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])]
        x, y = coords
        ind = 0
        num_x = (self.obs_shape[0] - 1) // 2
        num_y = (self.obs_shape[1] - 1) // 2
        for j in range(y - num_y, y + num_y + 1):
            for i in range(x - num_x, x + num_x + 1):
                is_air = True
                if i < 0 or i >= layer.shape[0]:
                    obs[ind % self.obs_shape[0]][ind // self.obs_shape[1]] = -2
                    ind += 1
                    continue
                if j < 0 or j >= layer.shape[1]:
                    obs[ind % self.obs_shape[0]][ind // self.obs_shape[1]] = -2
                    ind += 1
                    continue
                if is_air:
                    try:
                        obs[ind % self.obs_shape[0]][ind // self.obs_shape[1]] = layer.get((i, j)).get_representation()
                    except TypeError:
                        print(layer.get((i, j)))
                    ind += 1
                    continue
        return obs

    def observe(self):
        stack = []
        for i in range(self.pos[2] - 1, self.pos[2] + 2):
            if i < 0 or i >= self.shape[2]:
                stack.append([[-2 for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])
            else:
                stack.append(self.observe_layer(self.layers[i], self.pos[:-1]))

        stack.append([[self.pos[2] for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])

        stack.append([[self.inv.get('wood', 0) for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])
        stack.append(
            [[self.inv.get('workbench', 0) for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])
        stack.append(
            [[self.inv.get('woodpickaxe', 0) for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])
        stack.append([[self.inv.get('stone', 0) for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])
        stack.append(
            [[self.inv.get('stonepickaxe', 0) for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])
        stack.append([[self.inv.get('iron', 0) for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])
        stack.append(
            [[self.inv.get('ironpickaxe', 0) for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])
        stack.append([[self.inv.get('diamond', 0) for _ in range(self.obs_shape[0])] for _ in range(self.obs_shape[1])])

        return stack

    def craft(self):
        if self.inv['wood'] >= 1 and self.inv['iron'] >= 1 and self.inv['ironpickaxe'] == 0 and \
                self.inv['workbench'] == 1:
            self.inv += Counter(wood=-1, iron=-1, ironpickaxe=1)
        elif self.inv['wood'] >= 1 and self.inv['stone'] >= 1 and self.inv['stonepickaxe'] == 0 and \
                self.inv['workbench'] == 1:
            self.inv += Counter(wood=-1, stone=-1, stonepickaxe=1)
        elif self.inv['wood'] >= 2 and self.inv['workbench'] == 1 and self.inv['woodpickaxe'] == 0:
            self.inv += Counter(wood=-2, woodpickaxe=1)
        elif self.inv['wood'] >= 1 and self.inv['workbench'] == 0:
            self.inv += Counter(wood=-1, workbench=1)

    def reward_tracker(self):
        reward = 0
        temp = self.inv.copy()
        temp.subtract(self.prev_inv)
        if self.inv['diamond'] == 1:
            reward = 128
        elif temp['ironpickaxe'] == 1:
            reward = 64
        elif temp['iron'] == 1 and self.prev_inv['iron'] == 0 and self.inv['ironpickaxe'] == 0:
            reward = 32
        elif temp['stonepickaxe'] == 1:
            reward = 16
        elif temp['stone'] == 1 and self.prev_inv['stone'] == 0 and self.inv['stonepickaxe'] == 0:
            reward = 8
        elif temp['woodpickaxe'] == 1:
            reward = 4
        elif temp['workbench'] == 1:
            reward = 2
        elif temp['wood'] == 1 and self.prev_inv['wood'] == 0 and self.inv['workbench'] == 0:
            reward = 1

        self.prev_inv = self.inv.copy()
        return reward


if __name__ == '__main__':
    env = ObtainDiamondGridworld({'shape': (10, 10, 20)})
    obs = env.reset()
    reward = 0
    for i in range(50):
        print(i)
        for _ in range(10000):
            # env.render()
            # print(reward)
            # _, reward, _, _ = env.step(
            #     max(min(int(input("0: Noop 1: Forward 2: Backward 3: Right 4: Left 5: Craft 6: Mine Forward 7: Mine "
            #                       "Backward 8: Mine Right 9: Mine Left 10: Mine Down 11: Mine Up: ")), 11), 0))
            _, _, done, _ = env.step(np.random.choice(range(12)))
            if done:
                break
