import os
import numpy as np
import logging

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nets.unet import UNet
from datasets.salt import Salt

from utils.metrics import AverageMeter


class UNetAgent():
    def __init__(self, cfg):
        self.cfg = cfg

        # Device Setting
        self.device = torch.device('cuda')
        # #torch.cuda.manual_seed_all(self.config.seed)

        # Dataset Setting
        train_dataset = Salt(cfg, mode='train')
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE,
                                       shuffle=True, num_workers=8)
        valid_dataset = Salt(cfg, mode='valid')
        self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.VALID_BATCH_SIZE,
                                       shuffle=True, num_workers=8)

        # Network Setting
        self.net = UNet().to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.scheduler.step()

        self.loss = nn.BCEWithLogitsLoss()

        # Counter Setting
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

    def save_checkpoint(self):

        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        filename = f'UNet_{self.cfg.KFOLD_I}.ckpt'
        os.makedirs(self.cfg.CHECKPOINT_DIR, exist_ok=True)
        torch.save(state, os.path.join(self.cfg.CHECKPOINT_DIR, filename))
        '''
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')
        '''

    def load_checkpoint(self):

        filename = f'UNet_{self.cfg.KFOLD_I}.ckpt'
        logging.info("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(os.path.join(self.cfg.CHECKPOINT_DIR, filename))

        self.current_epoch = checkpoint['epoch']
        self.current_iteration = checkpoint['iteration']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        logging.info("Checkpoint loaded successfully at (epoch {}) at (iteration {})\n"
                     .format(checkpoint['epoch'], checkpoint['iteration']))

    def train(self):

        num_bad_epochs = 0
        for epoch in range(self.current_epoch, self.cfg.MAX_EPOCH):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            # LR Decaying
            if num_bad_epochs == self.cfg.PATIENCE:
                self.scheduler.step()
                num_bad_epochs = 0

                cur_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f'LR decaying to {cur_lr}')

                # Early Stopping
                if cur_lr < 5e-4:
                    break

        self.save_checkpoint()
        return self.best_valid_acc

    def train_one_epoch(self):

        # Set the model to be in training mode
        self.net.train()

        # Initialize average meters
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        tqdm_batch = tqdm(self.train_loader, f'Epoch-{self.current_epoch}-')
        for x in tqdm_batch:
            # prepare data
            imgs = torch.tensor(x['img'], dtype=torch.float, device=self.device)
            masks = torch.tensor(x['mask'], dtype=torch.float, device=self.device)

            # model
            pred = self.net(imgs)

            # loss
            cur_loss = self.loss(pred, masks)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            # metrics
            cur_acc = torch.sum((pred > 0) == (masks > 0.5)).item() / imgs.numel()

            epoch_loss.update(cur_loss.item(), imgs.size(0))
            epoch_acc.update(cur_acc, imgs.size(0))

            self.current_iteration += 1

        tqdm_batch.close()

        logging.info(f'Training at epoch- {self.current_epoch} |'
                     f'loss: {epoch_loss.val} - Acc: {epoch_acc.val}')

    def validate(self):

        # set the model in eval mode
        self.net.eval()

        # Initialize average meters
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        tqdm_batch = tqdm(self.valid_loader, f'Epoch-{self.current_epoch}-')
        with torch.no_grad():
            for x in tqdm_batch:
                # prepare data
                imgs = torch.tensor(x['img'], dtype=torch.float, device=self.device)
                masks = torch.tensor(x['mask'], dtype=torch.float, device=self.device)

                # model
                pred = self.net(imgs)

                # loss
                cur_loss = self.loss(pred, masks)
                if np.isnan(float(cur_loss.item())):
                    raise ValueError('Loss is nan during training...')

                # metrics
                cur_acc = torch.sum((pred > 0) == (masks > 0.5)).item() / imgs.numel()

                epoch_loss.update(cur_loss.item(), imgs.size(0))
                epoch_acc.update(cur_acc, imgs.size(0))

        tqdm_batch.close()

        logging.info(f'Validation at epoch- {self.current_epoch} |'
                     f'loss: {epoch_loss.val} - Acc: {epoch_acc.val}')

        return epoch_acc.val

    def predict(self, imgs):

        self.net.eval()

        with torch.no_grad():
            imgs = torch.tensor(imgs, dtype=torch.float, device=self.device)

            # model
            pred = self.net(imgs)
            pred = torch.sigmoid(pred)

        return pred.cpu().numpy()
