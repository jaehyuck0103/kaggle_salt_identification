import argparse
import os
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from skimage.io import imsave
from tqdm import tqdm

from torch.utils.data import DataLoader
from agents.unet import UNetAgent
from datasets.salt import SaltTest
from utils.imgproc import remove_small_mask
from utils.misc import rle_encode
from configs.config import process_config

# Filter out the low contrast warning in imsave()
warnings.filterwarnings('ignore', message='.*low contrast')
warnings.filterwarnings('ignore', message='.*Anti')

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def train(cfg):

    '''
    # temp
    from shutil import copy2  # NOQA
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    copy2('./nets/unet_res_open.py', cfg.CHECKPOINT_DIR)
    copy2('./datasets/salt.py', cfg.CHECKPOINT_DIR)
    '''

    for KFOLD_I in cfg.COMMON.KFOLD_I_LIST:
        for CYCLE_I in range(cfg.COMMON.CYCLE_N):

            filename = f'UNet_{KFOLD_I}_{CYCLE_I}.ckpt'
            if os.path.exists(os.path.join(cfg.COMMON.CHECKPOINT_DIR, filename)):
                continue

            cfg_dyn = cfg.get_train_config(CYCLE_I)
            cfg_dyn.KFOLD_I = KFOLD_I
            cfg_dyn.CYCLE_I = CYCLE_I
            logging.info(vars(cfg_dyn))

            agent = UNetAgent(cfg_dyn)
            if CYCLE_I > 0:
                agent.load_checkpoint(cfg.COMMON.CHECKPOINT_DIR)
            agent.train()

    # logging configs
    logging.info(vars(cfg.COMMON))


def test(cfg):
    cfg = cfg.get_test_config()
    cfg.CYCLE_I = cfg.CYCLE_N

    test_dataset = SaltTest(cfg)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.TEST_BATCH_SIZE,
                             shuffle=False, num_workers=8)

    agents = []
    for i in cfg.KFOLD_I_LIST:
        cfg.KFOLD_I = i
        agent = UNetAgent(cfg, predict_only=True)
        agent.load_checkpoint(cfg.CHECKPOINT_DIR)
        agents.append(agent)

    tqdm_batch = tqdm(test_loader, f'Test')
    os.makedirs(os.path.join(cfg.CHECKPOINT_DIR, 'test_imgs'), exist_ok=True)
    pred_dict = {}
    for x in tqdm_batch:
        pred = [a.predict(x['img']) for a in agents]
        pred = np.mean(pred, axis=0)
        pred = np.squeeze(pred)

        for mask, fname in zip(pred, x['file_name']):
            mask = np.round(mask)
            mask = remove_small_mask(mask)

            save_path = os.path.join(cfg.CHECKPOINT_DIR, f'test_imgs/{fname}.png')
            imsave(save_path, mask)
            pred_dict[fname] = rle_encode(mask[13:-14, 13:-14])

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(os.path.join(cfg.CHECKPOINT_DIR, 'submit.csv'))


if __name__ == '__main__':

    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument('MODE', type=str, choices=['train', 'test'])
    parser.add_argument('CFG_NAME', type=str)

    # Optional arguments
    parser.add_argument('--VER_TO_LOAD', type=str)

    # Fixed configs
    args = parser.parse_args()
    cfgs = process_config(args)

    # ------
    # Setting Root Logger
    # -----
    level = logging.INFO
    format = '%(asctime)s: %(message)s'
    log_dir = os.path.join(ROOT_DIR, 'output', 'log')
    log_path = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S.log'))
    os.makedirs(log_dir, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode='w')
    ]
    logging.basicConfig(format=format, level=level, handlers=handlers)

    # -------
    # Run
    # -------
    func = globals()[args.MODE]
    func(cfgs)
