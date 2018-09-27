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

        self.aug_geo = iaa.Sequential([
            iaa.Fliplr(p=0.5),
            iaa.Crop(px=(0, 20)),
            iaa.Scale({"height": 128, "width": 128}),
            iaa.Affine(rotate=(-10, 10), mode='reflect'),
        ])

        self.aug_intensity = iaa.Sequential([
            iaa.Add(value=(-20, +20)),
            iaa.ContrastNormalization(alpha=(0.9, 1.1)),
        ])

    def __getitem__(self, idx):

        idx = self.idx_map[idx]

        img = self.imgs[idx]
        mask = self.masks[idx]

        H = 128
        W = 128

        if self.mode == 'train':
            img_mask = np.stack([img, mask], axis=-1)

            img_mask *= 255
            img_mask = img_mask.astype(np.uint8)

            # pad and crop
            img_mask = np.pad(img_mask, ((23, 24), (23, 24), (0, 0)), mode='reflect')
            '''
            crop_idx_W = np.random.randint(20)
            crop_idx_H = np.random.randint(20)
            img_mask = img_mask[crop_idx_H:crop_idx_H+H, crop_idx_W:crop_idx_W+W, :]

            # FlipLR (50%)
            flip_idx = np.random.randint(20)
            if flip_idx < 10:
                img_mask = img_mask[:, ::-1, :]
            '''
            img_mask = self.aug_geo.augment_image(img_mask)
            img_mask[:, :, 0] = self.aug_intensity.augment_image(img_mask[:, :, 0])

            #
            img_mask = img_mask.astype(np.float32)
            img_mask /= 255

            #
            img = img_mask[:, :, 0].copy()
            mask = img_mask[:, :, 1].copy()
        else:
            img = np.pad(img, ((13, 14), (13, 14)), 'reflect')
            mask = np.pad(mask, ((13, 14), (13, 14)), 'reflect')

        # sample return
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        sample = {'img': img, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.idx_map)


class SaltTest(Dataset):
    def __init__(self, cfg, mode):
        super(SaltTest, self).__init__()

        self.cfg = cfg

        self.imgs = ImageCollection(os.path.join(TEST_IMG_DIR, '*.png'),
                                    conserve_memory=False, load_func=_imread_img)

    def __getitem__(self, idx):

        img = self.imgs[idx]

        file_name = self.imgs._files[idx]
        file_name = os.path.basename(file_name)
        file_name = os.path.splitext(file_name)[0]

        img = np.pad(img, ((13, 14), (13, 14)), 'reflect')

        # sample return
        img = np.expand_dims(img, axis=0)
        sample = {'img': img, 'file_name': file_name}

        return sample

    def __len__(self):
        return len(self.imgs)
