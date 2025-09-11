from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import random
import os
from typing import List


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_img_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except IOError:
        raise Exception('{} not found'.format(path))
    else:
        return img


def _read_txt_list(list_path: str, root: str = "") -> List[str]:
    image_list: List[str] = []
    with open(list_path, "r") as f:
        for line in f:
            image_list.append(os.path.join(root, line.strip()))
    return image_list


def _scan_dir(dir_path: str) -> List[str]:
    files = sorted(os.listdir(dir_path))
    return [
        os.path.join(dir_path, f)
        for f in files
        if os.path.splitext(f)[1] in IMG_EXTENSIONS
    ]


class BaseDataset(data.Dataset):
    def __init__(self, cfg):
        src_list = getattr(cfg, "source_list", None)
        src_root = getattr(cfg, "source_root", "")
        src_dir = getattr(cfg, "source_dir", None)
        tgt_list = getattr(cfg, "target_list", None)
        tgt_root = getattr(cfg, "target_root", "")
        tgt_dir = getattr(cfg, "target_dir", None)

        self.src_list: List[str] = []
        self.tgt_list: List[str] = []

        if src_dir:
            self.src_list = _scan_dir(src_dir)
        elif src_list:
            self.src_list = _read_txt_list(src_list, src_root)
        else:
            raise ValueError("source_dir or source_list must be specified.")

        if tgt_dir:
            self.tgt_list = _scan_dir(tgt_dir)
        elif tgt_list:
            self.tgt_list = _read_txt_list(tgt_list, tgt_root)
        else:
            raise ValueError("target_dir or target_list must be specified.")

        self.src_len = len(self.src_list)
        self.tgt_len = len(self.tgt_list)

        transform_options = cfg.transform
        height, width = cfg.height, cfg.width
        scale_lower, scale_higher = cfg.scale_l, cfg.scale_h
        transform_list = []

        if 'random_resized_crop' in transform_options:
            transform_list.append(
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=(scale_lower, scale_higher),
                    ratio=(0.75, 1.3333333333333333),
                )
            )
        else:
            transform_list.append(transforms.Resize((height, width)))
        if 'crop' in transform_options:
            transform_list.append(transforms.RandomCrop(height))
        if 'h_flip' in transform_options:
            transform_list.append(transforms.RandomHorizontalFlip(0.5))
        if 'v_flip' in transform_options:
            transform_list.append(transforms.RandomVerticalFlip(0.5))
        transform_list.append(transforms.ToTensor())
        if 'normalize' in transform_options:
            transform_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            )
        self.transform = transforms.Compose(transform_list)
        self.image_loader = pil_img_loader
        self.random_pair = cfg.random_pair
        self.return_name = cfg.return_name

    def __getitem__(self, index):
        if self.random_pair:
            tgt_idx = random.randint(0, self.tgt_len - 1)
        else:
            tgt_idx = index % self.tgt_len
        src_img = pil_img_loader(self.src_list[index])
        tgt_img = pil_img_loader(self.tgt_list[tgt_idx])
        src_img = self.transform(src_img)
        tgt_img = self.transform(tgt_img)

        if not self.return_name:
            return src_img, tgt_img
        else:
            return (
                src_img,
                tgt_img,
                '{}_to_{}.png'.format(
                    os.path.splitext(os.path.basename(self.src_list[index]))[0],
                    os.path.splitext(os.path.basename(self.tgt_list[tgt_idx]))[0],
                ),
            )

    def __len__(self):
        return self.src_len


def get_dataset(cfg):
    return BaseDataset(cfg)

