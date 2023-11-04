import pathlib

import ruamel.yaml as yaml

SURVIVAL_LOG = False

root = pathlib.Path(__file__).parent
for key, value in yaml.safe_load((root / 'data.yaml').read_text()).items():
    globals()[key] = value
