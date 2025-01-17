import os

from .config_md import DefaultConfig as Config_md
from .config_twob_dct import DefaultConfig as Config_twob_dct
from .config_twob_dwt import DefaultConfig as Config_twob_dwt

name_to_config = {
    "twob_dwt": Config_twob_dwt,
    "twob_dct": Config_twob_dct,
    "md": Config_md,
}


def load_config(name):
    config_name = name.lower()
    print(f"Loading config: {config_name}")
    if config_name not in name_to_config:
        raise ValueError(f"Unknown config name: {config_name}. Valid names: {list(name_to_config.keys())}")
    return name_to_config[config_name]()
