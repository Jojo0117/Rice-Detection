from pathlib import Path
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial



import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


def load_image(filename):
    ext = Path(filename).suffix
    if ext == '.npy':
        return np.load(filename)  # Load .npy files directly as numpy arrays
    else:
        return np.array(Image.open(filename))  # Convert other types of images to numpy arrays


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        assert 0 < self.scale <= 1, 'Scale must be between 0 and 1'

        # Generate list of image IDs based on image files
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, scale, is_mask):
        # Resize images (if necessary) and normalize
        if is_mask:
            img = np.array(img).astype(np.int64)
        else:
            img = np.array(img).astype(np.float32)
            if img.max() > 1:
                img /= 255.0

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = self.images_dir / f'{name}.npy'
        mask_file = self.mask_dir / f'mask_{name}.npy'

        assert img_file.is_file(), f'Image file not found: {img_file}'
        assert mask_file.is_file(), f'Mask file not found: {mask_file}'

        img = load_image(img_file)
        mask = load_image(mask_file)

        assert img.shape == mask.shape, f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

