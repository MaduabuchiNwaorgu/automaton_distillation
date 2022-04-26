from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.automaton.obtain_diamond_aps import CraftableAP, InventoryAP, NewLayerAP, LayerAP, \
    MATERIAL_TO_NUM, NUM_TO_MATERIAL, materials_in_view

diamond_basic = {'shape': (16, 16, 16)}

diamond_basic_env_config = EnvConfig(
    env_name="ObtainDiamondGridworld-v0",
    kwargs={"config": diamond_basic})

obtain_diamond_aps = [
    AP(name="craftable", func=lambda x: True in [CraftableAP(recipe)(x) for recipe in CraftableAP.RECIPES]),
    AP(name="new_layer", func=NewLayerAP()),
    AP(name="has_viewable_materials",
       func=lambda x: True not in [InventoryAP(material, 1)(x) for material in materials_in_view(x)]),
]

obtain_diamond_ltlf = "(F craftable) & (F new_layer) & (F has_viewable_materials)"
