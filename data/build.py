# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader
import random

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms
from .datasets.bases import BaseImageDataset


class MergedDataLoader(BaseImageDataset):
    def __init__(self, names, props, root, verbose=True):
        assert len(names) == len(props), f'Diff len of names, props: {len(names)} vs. {len(props)}'
        assert all([0. <= prop <= 1. for prop in props]), f'Wrong props {props}'
        datasets = [init_dataset(name, root=root) for name in names]

        # TODO how to sample?
        trains = [random.choices(d.train, k=int(d.num_train_imgs * p)) for d, p in zip(datasets, props)]
        if len(trains) == 1:
            train = trains[0]
            query = datasets[0].query
            gallery = datasets[0].gallery
        elif len(trains) == 2:
            self.pid_base, self.camid_base = datasets[0].num_train_pids, datasets[0].num_train_imgs
            train = trains[0] + [(i, p + self.pid_base, c + self.camid_base) for i, p, c in trains[1]]
            query = datasets[1].query
            gallery = datasets[1].gallery
        else:
            raise IndexError('Wrong datasets len!')

        pid_container = set([p for _, p, _ in train])
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        train = [(i, pid2label[p], c) for i, p, c in train]

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        if verbose:
            print(f"=> MergedDataLoader {', '.join(names)} loaded")
            self.print_dataset_statistics(train, query, gallery)
        pass


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1 and cfg.DATASETS.PROPS == 1.:
        dataset = init_dataset(cfg.DATASETS.NAMES[0], root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        # dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
        dataset = MergedDataLoader(cfg.DATASETS.NAMES, cfg.DATASETS.PROPS, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes, dataset
