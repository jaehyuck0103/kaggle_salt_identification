import os
import numpy as np
import logging

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nets.unet_basic import UNetBasic
from nets.unet_res import UNetRes
from nets.unet_res34 import UNetRes34
from nets.unet_res_open import UNetResOpen
from datasets.salt import Salt

from utils.metrics import AverageMeter, iou_pytorch
from utils.imgproc import remove_small_mask_batch

from nets.lovasz_losses import lovasz_hinge


class UNetAgent():
    def __init__(self, cfg):
        self.cfg = cfg

        # Device Setting
        self.device = torch.device('cuda')

        # Dataset Setting
        train_dataset = Salt(cfg, mode='train')
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE,
                                       shuffle=True, num_workers=8)
        valid_dataset = Salt(cfg, mode='valid')
        self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.VALID_BATCH_SIZE,
                                       shuffle=False, num_workers=8)

        '''
        for i in range(100):
            train_dataset[i]
        exit()
        '''

        # Network Setting
        if cfg.NET == 'UNetBasic':
            self.net = UNetBasic(cfg).to(self.device)
        elif cfg.NET == 'UNetRes':
            self.net = UNetRes(cfg).to(self.device)
        elif cfg.NET == 'UNetRes34':
            self.net = UNetRes34(cfg).to(self.device)
        elif cfg.NET == 'UNetResOpen':
            self.net = UNetResOpen().to(self.device)
        else:
            raise ValueError(f'Unknown Network: {cfg.NET}')

        if cfg.OPTIMIZER == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LEARNING_RATE)
        elif cfg.OPTIMIZER == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=cfg.LEARNING_RATE,
                                             momentum=0.9, weight_decay=0.0001)
        else:
            raise ValueError(f'Unknown Optimizer: {cfg.OPTIMIZER}')

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1,
                                                         gamma=cfg.LR_DECAY_RATE)
        self.scheduler.step()

        if cfg.LOSS == 'CrossEntropy':
            self.loss = nn.BCEWithLogitsLoss()
        elif cfg.LOSS == 'LOVASZ':
            self.loss = lovasz_hinge
        else:
            raise ValueError(f'Unknown Loss: {cfg.LOSS}')

        # Counter Setting
        self.current_epoch = 0
        self.best_valid_iou = 0

    def save_checkpoint(self):

        state = {
            'epoch': self.current_epoch,
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

    def load_checkpoint(self, load_dir):

        filename = f'UNet_{self.cfg.KFOLD_I}.ckpt'
        logging.info("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(os.path.join(load_dir, filename))

        self.current_epoch = checkpoint['epoch'] + 1
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        logging.info(f'Checkpoint loaded successfully at (epoch {checkpoint["epoch"]})')

    def train(self):

        num_bad_epochs = 0
        for epoch in range(self.current_epoch, self.cfg.MAX_EPOCH):
            self.current_epoch = epoch
            self.train_one_epoch()

            # Validation
            valid_iou = self.validate()
            is_best = valid_iou > self.best_valid_iou
            if is_best:
                self.best_valid_iou = valid_iou
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
                if cur_lr < self.cfg.MIN_LR:
                    break

        self.save_checkpoint()
        return self.best_valid_iou

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

        tqdm_batch.close()

        logging.info(f'Training at epoch- {self.current_epoch} |'
                     f'loss: {epoch_loss.val:.5} - Acc: {epoch_acc.val:.5}')

    def validate(self):

        # set the model in eval mode
        self.net.eval()

        # Initialize average meters
        epoch_loss = AverageMeter()
        epoch_iou = AverageMeter()
        epoch_filtered_iou = AverageMeter()

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
                    raise ValueError('Loss is nan during validation...')

                # metrics
                pred_t = torch.sigmoid(pred) > 0.5
                masks_t = masks > 0.5

                cur_iou = iou_pytorch(pred_t, masks_t)
                cur_filtered_iou = iou_pytorch(remove_small_mask_batch(pred_t), masks_t)

                batch_size = imgs.shape[0]
                epoch_loss.update(cur_loss.item(), batch_size)
                epoch_iou.update(cur_iou.item(), batch_size)
                epoch_filtered_iou.update(cur_filtered_iou.item(), batch_size)

        tqdm_batch.close()

        logging.info(f'Validation at epoch- {self.current_epoch} |'
                     f'loss: {epoch_loss.val:.5} - IOU: {epoch_iou.val:.5}'
                     f' - Filtered IOU: {epoch_filtered_iou.val:.5}')

        return epoch_filtered_iou.val

    def predict(self, imgs):

        self.net.eval()

        with torch.no_grad():
            imgs = torch.tensor(imgs, dtype=torch.float, device=self.device)

            # model
            pred = self.net(imgs)
            pred = torch.sigmoid(pred)

        return pred.cpu().numpy()
