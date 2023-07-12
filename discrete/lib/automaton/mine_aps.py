class MineInfoAutAP:
    """Mineworldenv saves information about the current action to tile_action_names;
    does this frame contain a given key?"""

    def __init__(self, ap_name: str = None):
        self.name = ap_name

    def __call__(self, info):
        return self.name in info["tile_action_names"]


class MineInventoryAP:
    """Do we have inventory of a given item matching some quantity"""
    def __init__(self, inventory_item, quantity):
        self.item = inventory_item
        self.quantity = quantity

    def __call__(self, info):
        return info["inventory"][self.item] == self.quantity


class MineLocationAP:
    """Is the agent at a specific location?"""
    def __init__(self, location):
        self.location = tuple(location)

    def __call__(self, info):
        return info['position'] == self.location
