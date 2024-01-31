from .desp import NpxProbeDesp, NpxElectrodeDesp
from .npx import ChannelMap

__all__ = ['electrode_select']

BUILTIN_SELECTOR = {
    'default': 'chmap.probe_npx.select_default:electrode_select',
    'weaker': 'chmap.probe_npx.select_weaker_arrangement:electrode_select',
}


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, s: list[NpxElectrodeDesp], *,
                     selector: str = 'default',
                     **kwargs) -> ChannelMap:
    selector = BUILTIN_SELECTOR.get(selector, selector)
    selector = load_select(selector)

    return selector(desp, chmap, s, **kwargs)


def load_select(selector: str):
    module, _, name = selector.partition(':')
    if len(name) == 0:
        raise ValueError(f'not a selector pattern "module_path:name" : {selector}')

    import importlib
    module = importlib.import_module(module)

    return getattr(module, name)
