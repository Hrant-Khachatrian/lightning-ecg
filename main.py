import numpy as np
import torch

import argparse

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision.models import squeezenet1_1
from torchvision.models.resnet import BasicBlock, conv1x1

from datasets import MITBIHDataset


class MainECG(pl.LightningModule):
    def __init__(self, batch_size=16, conv_filters=64, data_source='aligned180'):
        super().__init__()

        self.batch_size = batch_size

        if data_source == 'aligned180':
            self.data_path = '/nfs/c9_2tb/tigrann/domainbed/mit_bih_data.npy'  # aligned beats
            beat_length = 180
            class_weights = [1, 32, 13, 112]
        else:
            self.data_path = '/nfs/c9_2tb/tigrann/domainbed/mit-bih.npy'  # non-aligned beats
            beat_length = 280
            class_weights = [1, 32, 13, 112, 6428]

        self.num_classes = len(class_weights)

        cf = conv_filters

        self.block1 = BasicBlock(1, cf)
        self.block2 = BasicBlock(cf, cf, stride=2, downsample=self._create_downsampler(cf, cf, 2))
        self.block3 = BasicBlock(cf, cf, stride=2, downsample=self._create_downsampler(cf, cf, 2))
        self.block4 = BasicBlock(cf, cf, stride=2, downsample=self._create_downsampler(cf, cf, 2))
        self.block5 = BasicBlock(cf, cf, stride=2, downsample=self._create_downsampler(cf, cf, 2))

        self.linear = nn.Linear(768, self.num_classes)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.train_f1 = pl.metrics.classification.F1(5, average=None)
        self.valid_f1 = pl.metrics.classification.F1(5, average=None)

    def _create_downsampler(self, inplanes, planes, stride):
        return nn.Sequential(
            conv1x1(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        batch_size, beat_length = x.size()

        x = x.view(batch_size, 1, 1, beat_length)
        x = x.repeat(1, 1, 11, 1)   # (b, 1, 11, 180)

        x = self.block1(x)  # (b, 64, 11, 180)  64=conv_filters
        x = self.block2(x)  # (b, 64, 6, 90)
        x = self.block3(x)  # (b, 64, 3, 45)
        x = self.block4(x)  # (b, 64, 2, 23)
        x = self.block5(x)  # (b, 64, 1, 12)

        x = x.view(batch_size, -1)  # (b, 768)
        x = self.linear(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        logits = self(batch['x'])
        loss = F.nll_loss(logits, batch['y'])
        self.train_acc(logits, batch['y'])
        Nf1, Sf1, Vf1, Ff1, Qf1 = self.train_f1(logits, batch['y'])
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_S_f1', Sf1, on_step=True, on_epoch=False)
        self.log('train_V_f1', Vf1, on_step=True, on_epoch=False)
        self.log('loss', loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        logits = self(batch['x'])
        # loss = F.nll_loss(logits, batch['y'])
        self.valid_acc(logits, batch['y'])
        Nf1, Sf1, Vf1, Ff1, Qf1 = self.valid_f1(logits, batch['y'])
        prefix = 'DS1_qdev' if dataloader_idx == 0 else 'DS2'
        self.log(f'{prefix}_acc', self.valid_acc, on_step=False, on_epoch=True)
        self.log(f'{prefix}_S_f1', Sf1, on_step=False, on_epoch=True)
        self.log(f'{prefix}_V_f1', Vf1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        train_dataset = MITBIHDataset(self.data_path, 'DS1', qdev=False)

        num_samples = len(train_dataset)
        class_counts = [(train_dataset.labels == i).sum() for i in range(self.num_classes)]

        class_weights = [num_samples / class_counts[i] for i in range(self.num_classes)]
        weights = [class_weights[train_dataset.labels[i]] for i in range(num_samples)]
        sampler = WeightedRandomSampler(weights, num_samples)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        return train_loader

    def val_dataloader(self):
        DS1_qdev = MITBIHDataset(self.data_path, 'DS1', qdev=True)
        DS2 = MITBIHDataset(self.data_path, 'DS2', qdev=None)
        loader1 = DataLoader(DS1_qdev, batch_size=self.batch_size)
        loader2 = DataLoader(DS2, batch_size=self.batch_size)
        return [loader1, loader2]

    def test_dataloader(self):
        test_dataset = MITBIHDataset(self.data_path, 'DS2', qdev=None)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return test_loader

from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb_path', '-tb', default='/nfs/c9_home/hrant/tb_logs/')
    parser.add_argument('--tb_name', '-n', default='ecg180-custom-wrs')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--filters', '-f', type=int, default=64)
    args = parser.parse_args()

    model = MainECG(batch_size=args.batch_size, conv_filters=args.filters).cuda()
    logger = TensorBoardLogger(args.tb_path, name=args.tb_name)
    trainer = pl.Trainer(logger=logger)
    # trainer = pl.Trainer()
    trainer.fit(model)
