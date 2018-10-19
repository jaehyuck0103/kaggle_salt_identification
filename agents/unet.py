import os
import numpy as np
import logging

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nets.unet_res_light import UNetResLight
from nets.unet_res_supervision import UNetResSupervision
from datasets.salt import Salt

from utils.metrics import AverageMeter, iou_pytorch
from utils.imgproc import remove_small_mask_batch

from nets.lovasz_losses import lovasz_hinge


class UNetAgent():
    def __init__(self, cfg, predict_only=False):
        self.cfg = cfg

        # Device Setting
        self.device = torch.device('cuda')

        # Network Setting
        if cfg.NET == 'UNetResLight':
            self.net = UNetResLight().to(self.device)
        elif cfg.NET == 'UNetResSupervision':
            self.net = UNetResSupervision().to(self.device)
        else:
            raise ValueError(f'Unknown Network: {cfg.NET}')

        if predict_only:
            return

        # Dataset Setting
        train_dataset = Salt(cfg, mode='train')
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE,
                                       shuffle=True, num_workers=8, drop_last=True)
        valid_dataset = Salt(cfg, mode='valid')
        self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.VALID_BATCH_SIZE,
                                       shuffle=False, num_workers=8)

        '''
        for i in range(100):
            train_dataset[i]
        exit()
        '''

        if cfg.OPTIMIZER == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LEARNING_RATE)
        elif cfg.OPTIMIZER == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=cfg.LEARNING_RATE,
                                             momentum=0.9, weight_decay=0.0001)
        else:
            raise ValueError(f'Unknown Optimizer: {cfg.OPTIMIZER}')

        if cfg.SCHEDULER == 'Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=cfg.LR_DECAY_RATE,
                patience=cfg.PATIENCE, verbose=True, threshold=0)
        elif cfg.SCHEDULER == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.NUM_EPOCH, eta_min=cfg.ETA_MIN)
        else:
            raise ValueError(f'Unknown Scheduler: {cfg.SCHEDULER}')

        if cfg.LOSS == 'CrossEntropy':
            self.loss = nn.BCEWithLogitsLoss()
        elif cfg.LOSS == 'LOVASZ':
            self.loss = lovasz_hinge
        else:
            raise ValueError(f'Unknown Loss: {cfg.LOSS}')

        # Counter Setting
        self.current_epoch = 0

    def save_checkpoint(self):

        state = {
            'epoch': self.current_epoch,
            'state_dict': self.net.state_dict(),
            'best_thres': self.best_thres,
        }

        filename = f'UNet_{self.cfg.KFOLD_I}_{self.cfg.CYCLE_I}.ckpt'
        logging.info("Saving checkpoint '{}'".format(filename))
        os.makedirs(self.cfg.CHECKPOINT_DIR, exist_ok=True)
        torch.save(state, os.path.join(self.cfg.CHECKPOINT_DIR, filename))

        logging.info(f'Checkpoint saved successfully at (epoch {self.current_epoch})')

    def load_checkpoint(self, cycle_i):

        filename = f'UNet_{self.cfg.KFOLD_I}_{cycle_i}.ckpt'
        logging.info("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(os.path.join(self.cfg.CHECKPOINT_DIR, filename))

        self.current_epoch = checkpoint['epoch'] + 1
        self.net.load_state_dict(checkpoint['state_dict'])
        self.best_thres = checkpoint['best_thres']

        logging.info(f'Checkpoint loaded successfully at (epoch {checkpoint["epoch"]})')

    def train(self):
        num_bad_epochs = 0
        best_valid_iou = 0
        for epoch in range(self.current_epoch, self.current_epoch + self.cfg.NUM_EPOCH):
            self.current_epoch = epoch

            if self.cfg.NET == 'UNetResLight':
                self.train_one_epoch()
            elif self.cfg.NET == 'UNetResSupervision':
                self.train_one_epoch_supervision()
            else:
                raise ValueError(f'Unknown Network: {self.cfg.NET}')

            # Validation
            valid_iou = self.validate()

            # LR Scheduling
            if self.cfg.SCHEDULER == 'Plateau':
                self.scheduler.step(valid_iou)
            else:
                self.scheduler.step()

            logging.info(f"Current LR: {self.optimizer.param_groups[0]['lr']}")

            # Early Stop
            is_best = valid_iou > best_valid_iou
            if is_best:
                best_valid_iou = valid_iou
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            if num_bad_epochs == self.cfg.EARLY_STOP:
                break

        # Find best threshold
        self.best_thres = self.find_best_thres()

        self.save_checkpoint()
        return best_valid_iou

    def train_one_epoch(self):

        # Set the model to be in training mode
        self.net.train()

        # Initialize average meters
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()
        epoch_iou = AverageMeter()
        epoch_filtered_iou = AverageMeter()

        tqdm_batch = tqdm(self.train_loader, f'Epoch-{self.current_epoch}-')
        for x in tqdm_batch:
            # prepare data
            imgs = torch.tensor(x['img'], dtype=torch.float, device=self.device)
            masks = torch.tensor(x['mask'], dtype=torch.float, device=self.device)

            # model
            pred, *_ = self.net(imgs)

            # loss
            cur_loss = self.loss(pred, masks)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            # metrics
            pred_t = torch.sigmoid(pred) > 0.5
            masks_t = masks > 0.5

            cur_acc = torch.sum(pred_t == masks_t).item() / masks.numel()
            cur_iou = iou_pytorch(pred_t, masks_t)
            cur_filtered_iou = iou_pytorch(remove_small_mask_batch(pred_t), masks_t)

            batch_size = imgs.shape[0]
            epoch_loss.update(cur_loss.item(), batch_size)
            epoch_acc.update(cur_acc, batch_size)
            epoch_iou.update(cur_iou.item(), batch_size)
            epoch_filtered_iou.update(cur_filtered_iou.item(), batch_size)

        tqdm_batch.close()

        logging.info(f'Training at epoch- {self.current_epoch} |'
                     f'loss: {epoch_loss.val:.5} - Acc: {epoch_acc.val:.5}'
                     f'- IOU: {epoch_iou.val:.5} - Filtered IOU: {epoch_filtered_iou.val:.5}')

    def train_one_epoch_supervision(self):

        # Set the model to be in training mode
        self.net.train()

        # Initialize average meters
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()
        epoch_iou = AverageMeter()
        epoch_filtered_iou = AverageMeter()

        tqdm_batch = tqdm(self.train_loader, f'Epoch-{self.current_epoch}-')
        for x in tqdm_batch:
            # prepare data
            imgs = torch.tensor(x['img'], dtype=torch.float, device=self.device)
            masks = torch.tensor(x['mask'], dtype=torch.float, device=self.device)
            salty = torch.tensor(x['salty'], dtype=torch.float, device=self.device)

            # model
            pred, pred_seg_pure, pred_salty = self.net(imgs)

            # loss
            cur_pred_loss = self.loss(pred, masks)
            cur_seg_pure_loss = self.loss(pred_seg_pure[salty.squeeze() > 0.5],
                                          masks[salty.squeeze() > 0.5])
            cur_salty_loss = F.binary_cross_entropy_with_logits(pred_salty, salty)
            cur_loss = 0.005 * cur_salty_loss + 0.5 * cur_seg_pure_loss + cur_pred_loss
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            # metrics
            pred_t = torch.sigmoid(pred) > 0.5
            masks_t = masks > 0.5

            cur_acc = torch.sum(pred_t == masks_t).item() / masks.numel()
            cur_iou = iou_pytorch(pred_t, masks_t)
            cur_filtered_iou = iou_pytorch(remove_small_mask_batch(pred_t), masks_t)

            batch_size = imgs.shape[0]
            epoch_loss.update(cur_loss.item(), batch_size)
            epoch_acc.update(cur_acc, batch_size)
            epoch_iou.update(cur_iou.item(), batch_size)
            epoch_filtered_iou.update(cur_filtered_iou.item(), batch_size)

        tqdm_batch.close()

        logging.info(f'Training at epoch- {self.current_epoch} |'
                     f'loss: {epoch_loss.val:.5} - Acc: {epoch_acc.val:.5}'
                     f'- IOU: {epoch_iou.val:.5} - Filtered IOU: {epoch_filtered_iou.val:.5}')

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
                pred, *_ = self.net(imgs)

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

    def find_best_thres(self):
        # set the model in eval mode
        self.net.eval()

        pred_TTA_list = []
        masks_t_list = []

        tqdm_batch = tqdm(self.valid_loader, f'Epoch-{self.current_epoch}-')
        with torch.no_grad():
            for x in tqdm_batch:
                # prepare data
                imgs = torch.tensor(x['img'], dtype=torch.float, device=self.device)
                masks = torch.tensor(x['mask'], dtype=torch.float, device=self.device)

                # model
                pred, *_ = self.net(imgs)
                pred_flip, *_ = self.net(imgs.flip(dims=[3]))
                pred_TTA = (torch.sigmoid(pred) + torch.sigmoid(pred_flip.flip(dims=[3]))) / 2

                # metrics
                masks_t = masks > 0.5

                pred_TTA_list.append(pred_TTA.cpu())
                masks_t_list.append(masks_t.cpu())
        tqdm_batch.close()

        pred_TTA = torch.cat(pred_TTA_list, dim=0)
        masks_t = torch.cat(masks_t_list, dim=0)

        thresholds = np.linspace(0.3, 0.7, 50)
        filtered_ious = np.array([iou_pytorch(remove_small_mask_batch(pred_TTA > t), masks_t)
                                  for t in thresholds])
        best_thres_idx = np.argmax(filtered_ious)
        best_thres = thresholds[best_thres_idx]

        logging.info(f'Best Threshold- {best_thres}')

        return best_thres

    def predict(self, imgs):

        self.net.eval()

        with torch.no_grad():
            imgs = torch.tensor(imgs, dtype=torch.float, device=self.device)

            # model
            pred, *_ = self.net(imgs)
            pred = torch.sigmoid(pred)

        return pred.cpu().numpy()
