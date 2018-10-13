import os
import numpy as np
from skimage.io import imread, ImageCollection  # , imsave
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
from imgaug import augmenters as iaa


ROOT_DIR = os.path.join(os.path.dirname(__file__), '../')
ROOT_DIR = os.path.abspath(ROOT_DIR)

TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'data/train/images')
TRAIN_MASK_DIR = os.path.join(ROOT_DIR, 'data/train/masks')
TEST_IMG_DIR = os.path.join(ROOT_DIR, 'data/test/images')


def _imread_img(f):
    img = imread(f, as_gray=True).astype(np.float32)
    return img


def _imread_mask(f):
    mask = (imread(f) != 0)
    mask = mask.astype(np.float32)
    return mask


class Salt(Dataset):
    def __init__(self, cfg, mode):
        super(Salt, self).__init__()

        self.cfg = cfg
        self.mode = mode

        self.imgs = ImageCollection(os.path.join(TRAIN_IMG_DIR, '*.png'),
                                    conserve_memory=False, load_func=_imread_img)
        self.masks = ImageCollection(os.path.join(TRAIN_MASK_DIR, '*.png'),
                                     conserve_memory=False, load_func=_imread_mask)

        H, W = self.imgs[0].shape

        coverages = np.array([np.sum(m) / (H * W) for m in self.masks])
        coverage_labels = np.ceil(coverages * 10).astype(int)

        kf = StratifiedKFold(n_splits=cfg.KFOLD_N, shuffle=True, random_state=910103)
        train_idx, valid_idx = list(kf.split(self.imgs, coverage_labels))[cfg.KFOLD_I]
        if mode == 'train':
            self.idx_map = train_idx
        elif mode == 'valid':
            self.idx_map = valid_idx
        else:
            raise ValueError('Unknown Mode')

        self.aug_geo1 = iaa.Sequential([
            # General
            iaa.Fliplr(0.5),
            iaa.Crop(px=(5, 15), keep_size=False),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10), mode='edge')),

            # Deformations
            # iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
            iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
        ], random_order=True)

        self.aug_geo2 = iaa.Sequential([
            iaa.Scale({"height": 128, "width": 128}),
        ], random_order=False)

        self.aug_intensity = iaa.Sequential([
            iaa.Invert(0.3),
            iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
            iaa.OneOf([
                 iaa.Noop(),
                 # iaa.OneOf([
                 #     iaa.Add((-10, 10)),
                 #     iaa.AddElementwise((-10, 10)),
                 #     iaa.Multiply((0.95, 1.05)),
                 #     iaa.MultiplyElementwise((0.95, 1.05)),
                 # ]),
                 # iaa.OneOf([
                 #     iaa.GaussianBlur(sigma=(0.0, 1.0)),
                 #     iaa.AverageBlur(k=(2, 5)),
                 #     iaa.MedianBlur(k=(3, 5))
                 # ])
             ])
        ], random_order=False)

    def __getitem__(self, idx):

        idx = self.idx_map[idx]

        img = self.imgs[idx]
        mask = self.masks[idx]

        '''
        imsave(f'output/image/{idx}_img.png', img)
        imsave(f'output/image/{idx}_mask.png', mask)
        '''

        if self.mode == 'train':
            img_mask = np.stack([img, mask], axis=-1)

            img_mask = np.pad(img_mask, ((23, 24), (23, 24), (0, 0)), mode='edge')

            # float -> uint8
            img_mask *= 255
            img_mask = img_mask.astype(np.uint8)

            # augment
            img_mask = self.aug_geo1.augment_image(img_mask)
            img_mask = self.aug_geo2.augment_image(img_mask)
            img_mask[:, :, 0] = self.aug_intensity.augment_image(img_mask[:, :, 0])

            # uint8 -> float
            img_mask = img_mask.astype(np.float32)
            img_mask /= 255

            #
            img = img_mask[:, :, 0].copy()
            mask = img_mask[:, :, 1].copy()
        else:
            img = np.pad(img, ((13, 14), (13, 14)), 'edge')
            mask = np.pad(mask, ((13, 14), (13, 14)), 'edge')

        '''
        imsave(f'output/image/{idx}_img_aug.png', img)
        imsave(f'output/image/{idx}_mask_aug.png', mask)
        '''

        # Add depth channels
        yy, _ = np.mgrid[1:129, 1:129] / 128
        ch3 = img * yy

        img = np.stack([img, yy, ch3], axis=0)

        # Normalize
        img[0] = (img[0] - 0.485) / 0.229
        img[1] = (img[1] - 0.456) / 0.224
        img[2] = (img[2] - 0.406) / 0.225

        # sample return
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        mask = np.expand_dims(mask, axis=0)
        sample = {'img': img, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.idx_map)


class SaltTest(Dataset):
    def __init__(self, cfg):
        super(SaltTest, self).__init__()

        self.cfg = cfg

        self.imgs = ImageCollection(os.path.join(TEST_IMG_DIR, '*.png'),
                                    conserve_memory=False, load_func=_imread_img)

    def __getitem__(self, idx):

        img = self.imgs[idx]

        file_name = self.imgs._files[idx]
        file_name = os.path.basename(file_name)
        file_name = os.path.splitext(file_name)[0]

        img = np.pad(img, ((13, 14), (13, 14)), 'edge')

        # Add depth channels
        yy, _ = np.mgrid[1:129, 1:129] / 128
        ch3 = img * yy

        img = np.stack([img, yy, ch3], axis=0)

        # Normalize
        img[0] = (img[0] - 0.485) / 0.229
        img[1] = (img[1] - 0.456) / 0.224
        img[2] = (img[2] - 0.406) / 0.225

        # sample return
        sample = {'img': img, 'file_name': file_name}

        return sample

    def __len__(self):
        return len(self.imgs)
