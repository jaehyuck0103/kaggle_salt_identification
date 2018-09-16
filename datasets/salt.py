import os
import numpy as np
from skimage.io import imread, ImageCollection  # , imsave
from sklearn.model_selection import KFold


from torch.utils.data import Dataset


ROOT_DIR = os.path.join(os.path.dirname(__file__), '../')
ROOT_DIR = os.path.abspath(ROOT_DIR)

TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'data/train/images')
TRAIN_MASK_DIR = os.path.join(ROOT_DIR, 'data/train/masks')
TEST_IMG_DIR = os.path.join(ROOT_DIR, 'data/test/images')


def _imread_img(f):
    img = imread(f, as_gray=True).astype(np.float32)
    img = np.pad(img, ((13, 14), (13, 14)), 'reflect')
    return img


def _imread_mask(f):
    mask = (imread(f) != 0)
    mask = mask.astype(np.float32)
    mask = np.pad(mask, ((13, 14), (13, 14)), 'reflect')
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

        kf = KFold(n_splits=cfg.KFOLD_N, shuffle=False)
        train_idx, valid_idx = list(kf.split(self.imgs))[cfg.KFOLD_I]
        if mode == 'train':
            self.idx_map = train_idx
        elif mode == 'valid':
            self.idx_map = valid_idx
        else:
            raise ValueError('Unknown Mode')

    def __getitem__(self, idx):

        idx = self.idx_map[idx]

        img = self.imgs[idx]
        mask = self.masks[idx]

        H, W = img.shape

        if self.mode == 'train':
            img_mask = np.stack([img, mask], axis=-1)

            # pad and crop
            img_mask = np.pad(img_mask, ((10, 10), (10, 10), (0, 0)), mode='reflect')
            crop_idx = np.random.randint(20)
            img_mask = img_mask[crop_idx:crop_idx+H, crop_idx:crop_idx+W, :]

            # FlipLR (50%)
            if crop_idx < 10:
                img_mask = img_mask[:, ::-1, :]

            #
            img = img_mask[:, :, 0].copy()
            mask = img_mask[:, :, 1].copy()

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

        # sample return
        img = np.expand_dims(img, axis=0)
        sample = {'img': img, 'file_name': file_name}

        return sample

    def __len__(self):
        return len(self.imgs)
