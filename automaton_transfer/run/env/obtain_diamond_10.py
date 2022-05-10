from automaton_transfer.lib.config import EnvConfig
from automaton_transfer.lib.automaton.mine_env_ap_extractor import AP
from automaton_transfer.lib.automaton.obtain_diamond_aps import CraftableAP, InventoryAP, NewLayerAP, LayerAP, \
    MATERIAL_TO_NUM, NUM_TO_MATERIAL, materials_in_view, AnyCraftableAP, HasViewableMaterials

diamond_basic = {'shape': (10, 10, 10)}

diamond_basic_env_config = EnvConfig(
    env_name='ObtainDiamondGridworld-v0',
    kwargs={'config': diamond_basic})


obtain_diamond_aps = [
    AP(name="craftable", func=AnyCraftableAP()),
    AP(name="new_layer", func=NewLayerAP()),
    AP(name='woodpickaxe', func=InventoryAP('woodpickaxe', 1)),
    AP(name='stone', func=InventoryAP('stone', 1)),
    AP(name='stonepickaxe', func=InventoryAP('stonepickaxe', 1)),
    AP(name='iron', func=InventoryAP('iron', 1)),
    AP(name='ironpickaxe', func=InventoryAP('ironpickaxe', 1)),
    AP(name='diamond', func=InventoryAP('diamond', 1)),
    AP(name='stone_layers', func=LayerAP('stone')),
    AP(name='iron_layers', func=LayerAP('iron')),
    AP(name='diamond_layers', func=LayerAP('diamond'))
]

obtain_diamond_aps_medium = [
    AP(name="craftable", func=AnyCraftableAP()),
    AP(name="new_layer", func=NewLayerAP()),
    AP(name="has_viewable_materials", func=HasViewableMaterials())
]

obtain_diamond_ltlf = '(F craftable) & ((F new_layer) U diamond_layers) & ' \
                      '((woodpickaxe & stone_layers) -> F stone) & ((stonepickaxe & iron_layers) -> F iron) &' \
                      ' ((ironpickaxe & diamond_layers) -> F diamond)'

obtain_diamond_ltlf_medium = '((F craftable) U ironpickaxe) & ((F new_layer) U diamond_layers) ' \
                            '& (F has_viewable_materials)'
