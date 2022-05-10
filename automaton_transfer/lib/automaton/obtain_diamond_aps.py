NUM_TO_MATERIAL = {1: 'wood', 3: 'stone', 4: 'iron', 5: 'diamond'}
MATERIAL_TO_NUM = {'wood': 1, 'stone': 3, 'iron': 4, 'diamond': 5}


class CraftableAP:
    """Checks if an item is currently craftable given the inventory requirements"""
    RECIPES = ['workbench', 'woodpickaxe', 'stone_pickaxe', 'iron_pickaxe']

    def __init__(self, item_name: str = None):
        self.item_name = item_name

    def __call__(self, info):
        inv = info['inventory']
        if self.item_name == 'workbench':
            return inv['wood'] > 1 and inv['workbench'] != 1
        if self.item_name == 'woodpickaxe':
            return inv['wood'] > 2 and inv['woodpickaxe'] != 1
        if self.item_name == 'stone_pickaxe':
            return inv['wood'] > 1 and inv['stone'] > 1 and inv['stonepickaxe'] != 1
        if self.item_name == 'iron_pickaxe':
            return inv['wood'] > 1 and inv['iron'] > 1 and inv['workbench'] != 1


class AnyCraftableAP:
    """Checks if there is any craftable item given the current inventory"""

    def __init__(self):
        pass

    def __call__(self, info):
        inv = info['inventory']
        if inv['wood'] > 1 and inv['workbench'] != 1:
            return True
        if inv['wood'] > 2 and inv['woodpickaxe'] != 1:
            return True
        if inv['wood'] > 1 and inv['stone'] > 1 and inv['stonepickaxe'] != 1:
            return True
        if inv['wood'] > 1 and inv['iron'] > 1 and inv['workbench'] != 1:
            return True

        return False


class InventoryAP:
    """Same as other MineInventoryAP, but in this case at least some quantity"""

    def __init__(self, inventory_item, quantity):
        self.item = inventory_item
        self.quantity = quantity

    def __call__(self, info):
        return info["inventory"][self.item] >= self.quantity


class NewLayerAP:
    """Checks if the agent is on a new layer"""

    def __init__(self):
        self.current_location = 0

    def __call__(self, info):
        new_layer = info['pos'][-1] == self.current_location
        self.current_location = info['pos'][-1]
        return new_layer


class LayerAP:
    """Checks if the agent is on the layer for a specific material"""
    MATERIALS = ['wood', 'stone', 'iron', 'diamond']

    def __init__(self, material: str = None):
        self.material = material

    def __call__(self, info):
        layers = info['shape'][-1]

        if self.material == 'wood':
            return info['pos'][-1] == 0
        elif self.material == 'stone' or self.material == 'iron':
            return info['pos'][-1] > 2
        elif self.material == 'diamond':
            return info['pos'][-1] >= layers - 4


class MaterialInView:
    """Checks if the material is in the observation that the agent received"""

    def __init__(self, material):
        self.material = material

    def __call__(self, info):
        return MATERIAL_TO_NUM[self.material] in info['obs']


def materials_in_view(info):
    return set(info['obs'].flatten().tolist())
