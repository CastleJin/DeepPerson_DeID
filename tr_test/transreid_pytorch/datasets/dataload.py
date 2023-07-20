import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import ImageDataset
from .market_human import Market_human
from .msmt17 import MSMT17
from .mm import MM
__factory = {
    'market1501': Market_human,
    'msmt17': MSMT17,
    'mm': MM,
}

def train_collate_fn(batch):
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def test_gan(cfg):
    # make transform
    v_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # set the data
    # you can check the dataset format from datasets > market1501.py
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    
    # train and val
    train = ImageDataset(dataset.train, v_transforms)
    gallery = ImageDataset(dataset.gallery, v_transforms)
    query = ImageDataset(dataset.query, v_transforms)

    # dataloader
    train_dataloader = DataLoader(
        train, batch_size=128, shuffle=False, num_workers=2, collate_fn=val_collate_fn
    )
    gallery_dataloader = DataLoader(
        gallery, batch_size=128, shuffle=False, num_workers=2, collate_fn=val_collate_fn
    )
    query_dataloader = DataLoader(
        query, batch_size=128, shuffle=False, num_workers=2, collate_fn=val_collate_fn
    )

    # info
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    return train_dataloader, gallery_dataloader, query_dataloader, num_classes, cam_num, view_num

