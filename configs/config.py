import os
import json
from datetime import datetime

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.join(FILE_DIR, '../')


def process_config(cfg):

    json_path = os.path.join(FILE_DIR, f'{cfg.JSON_CFG}.json')
    with open(json_path) as f:
        json_cfg = json.load(f)

    for key, val in json_cfg.items():
        if key not in cfg:
            setattr(cfg, key, val)

    init_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if cfg.MODE == 'train':
        cfg.CHECKPOINT_DIR = os.path.join(ROOT_DIR, f'output/{init_time}')
    else:
        cfg.CHECKPOINT_DIR = os.path.join(ROOT_DIR, f'output/{cfg.VER_TO_LOAD}')

    return cfg
