import argparse
import os
import logging
import warnings
from datetime import datetime
import random

import numpy as np
import pandas as pd
from skimage.io import imsave
from tqdm import tqdm

import torch
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

# Fix seed
random.seed(910103)
np.random.seed(910103)
torch.manual_seed(910103)
torch.backends.cudnn.deterministic = True


def train(cfg):

    '''
    # temp
    from shutil import copy2  # NOQA
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    copy2('./nets/unet_res_open.py', cfg.CHECKPOINT_DIR)
    copy2('./datasets/salt.py', cfg.CHECKPOINT_DIR)
    '''

    for cfg.KFOLD_I in cfg.KFOLD_I_LIST:
        for cfg.CYCLE_I in range(cfg.CYCLE_N):
            agent = UNetAgent(cfg)
            if hasattr(cfg, 'FINETUNE_DIR') and cfg.CYCLE_I == 0:
                agent.load_checkpoint(cfg.FINETUNE_DIR, 0)
            elif cfg.CYCLE_I > 0:
                agent.load_checkpoint(cfg.CHECKPOINT_DIR, cfg.CYCLE_I-1)
            agent.train()

    # logging configs
    logging.info(cfg)


def test(cfg):

    test_dataset = SaltTest(cfg, mode='train')
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.TEST_BATCH_SIZE,
                             shuffle=False, num_workers=8)

    agents = []
    for i in cfg.KFOLD_I_LIST:
        cfg.KFOLD_I = i
        agent = UNetAgent(cfg)
        agent.load_checkpoint(cfg.CHECKPOINT_DIR, cfg.CYCLE_N-1)
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
    parser.add_argument('JSON_CFG', type=str)

    # Optional arguments
    parser.add_argument('--VER_TO_LOAD', type=str)

    # Fixed configs
    cfg = parser.parse_args()
    cfg = process_config(cfg)

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

    # logging configs
    logging.info(cfg)

    # -------
    # Run
    # -------
    func = globals()[cfg.MODE]
    func(cfg)
