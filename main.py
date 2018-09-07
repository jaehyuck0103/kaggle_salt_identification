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

# Filter out the low contrast warning in imsave()
warnings.filterwarnings('ignore', message='.*low contrast')
warnings.filterwarnings('ignore', message='.*Anti')

ROOT_DIR = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(ROOT_DIR)


def train(cfg):

    for i in range(cfg.KFOLD_N):
        cfg.KFOLD_I = i
        agent = UNetAgent(cfg)
        agent.train()


def test(cfg):

    test_dataset = SaltTest(cfg, mode='train')
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.TEST_BATCH_SIZE,
                             shuffle=False, num_workers=8)

    agents = []
    for i in range(cfg.KFOLD_N):
        cfg.KFOLD_I = i
        agent = UNetAgent(cfg)
        agent.load_checkpoint()
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
            pred_dict[fname] = rle_encode(mask)

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(os.path.join(cfg.CHECKPOINT_DIR, 'submit.csv'))


if __name__ == '__main__':

    init_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument('MODE', type=str, choices=['train', 'test'])

    # Optional arguments
    parser.add_argument('--VER_TO_LOAD', type=str)

    # Fixed configs
    cfg = parser.parse_args()
    cfg.TRAIN_BATCH_SIZE = 32
    cfg.VALID_BATCH_SIZE = 32
    cfg.TEST_BATCH_SIZE = 32
    cfg.LEARNING_RATE = 0.1
    cfg.MAX_EPOCH = 2000
    cfg.KFOLD_N = 5
    cfg.KFOLD_I = 0
    cfg.PATIENCE = 5
    if cfg.MODE == 'train':
        cfg.CHECKPOINT_DIR = os.path.join(ROOT_DIR, f'output/{init_time}')
    else:
        cfg.CHECKPOINT_DIR = os.path.join(ROOT_DIR, f'output/{cfg.VER_TO_LOAD}')

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
    func = globals()[cfg.MODE]
    func(cfg)
