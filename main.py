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


# Filter out the low contrast warning in imsave()
warnings.filterwarnings('ignore', message='.*low contrast')

ROOT_DIR = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(ROOT_DIR)


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def train(cfg):

    for i in range(cfg.KFOLD_N):
        cfg.KFOLD_I = i
        agent = UNetAgent(cfg)
        agent.train()


def test(cfg):

    df = pd.read_csv(os.path.join(ROOT_DIR, 'data/sample_submission.csv'), index_col='id')

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
    os.makedirs(os.path.join(cfg.CHECKPOINT_DIR, 'test_imgs'))
    for x in tqdm_batch:
        pred = [a.predict(x['img']) for a in agents]
        pred = np.mean(pred, axis=0)

        for mask, fname in zip(pred, x['file_name']):
            save_path = os.path.join(cfg.CHECKPOINT_DIR, f'test_imgs/{fname}.png')
            imsave(save_path, mask[0])

            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            df.loc[fname, 'rle_mask'] = rle_encode(mask[0])

    df.to_csv(os.path.join(cfg.CHECKPOINT_DIR, 'submit.csv'))


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
