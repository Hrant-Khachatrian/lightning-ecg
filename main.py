import numpy as np
import torch

import argparse

from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision.models.resnet import BasicBlock, conv1x1

from datasets import MITBIHDataset


class MainECG(pl.LightningModule):
    def __init__(self, batch_size=16, conv_filters=(32, 64, 256, 64, 32), learning_rate=0.001, lr_decay_milestones=None,
                 data_source='aligned180', data_workers=0):
        super().__init__()

        if lr_decay_milestones is None:
            lr_decay_milestones = []
        self.batch_size = batch_size
        self.data_workers = data_workers
        self.learning_rate = learning_rate
        self.lr_decay_milestones = lr_decay_milestones

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

        self.convs = []
        last_filter_size = 1
        for filter_size in conv_filters:
            self.convs.append(
                nn.Conv1d(last_filter_size, filter_size, 5, stride=2).cuda()
            )
            last_filter_size = filter_size

        self.linear = nn.Linear(64, self.num_classes)

        self.train_acc = pl.metrics.Accuracy()
        self.DS1_qdev_acc = pl.metrics.Accuracy()
        self.DS2_acc = pl.metrics.Accuracy()
        self.train_f1 = pl.metrics.classification.F1(5, average=None)
        self.DS1_qdev_f1 = pl.metrics.classification.F1(5, average=None)
        self.DS2_f1 = pl.metrics.classification.F1(5, average=None)

    def _create_downsampler(self, inplanes, planes, stride):
        return nn.Sequential(
            conv1x1(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        batch_size, beat_length = x.size()

        x = x.view(batch_size, 1, beat_length) # (b, 1, 180)
        # x = x.repeat(1, 1, 11, 1)   # (b, 1, 11, 180)

        x = self.convs[0](x)  # (b, 32, 88)  64=conv_filters
        x = self.convs[1](x)  # (b, 64, 42)
        x = self.convs[2](x)  # (b, 256, 19)
        x = self.convs[3](x)  # (b, 64, 8)
        x = self.convs[4](x)  # (b, 32, 2)

        x = x.view(batch_size, -1)  # (b, 64)
        x = self.linear(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x = batch['x'].float().cuda()
        y = batch['y'].long().cuda()
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.train_acc(logits, y)
        Nf1, Sf1, Vf1, Ff1, Qf1 = self.train_f1(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_S_f1', Sf1, on_step=True, on_epoch=False)
        self.log('train_V_f1', Vf1, on_step=True, on_epoch=False)
        self.log('loss', loss, on_step=True, on_epoch=False)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x = batch['x'].float().cuda()
        y = batch['y'].long().cuda()
        logits = self(x)
        loss = F.nll_loss(logits, y)
        if dataloader_idx == 0:
            prefix = 'DS1_qdev'
            acc = self.DS1_qdev_acc(logits, y)
            Nf1, Sf1, Vf1, Ff1, Qf1 = self.DS1_qdev_f1(logits, y)
        elif dataloader_idx == 1:
            prefix = 'DS2'
            acc = self.DS2_acc(logits, y)
            Nf1, Sf1, Vf1, Ff1, Qf1 = self.DS2_f1(logits, y)
        else:
            raise Exception(f"Unknown dataloader_idx={dataloader_idx}")

        self.log(f'{prefix}_loss', loss, on_step=False, on_epoch=True)
        self.log(f'{prefix}_acc', acc, on_step=False, on_epoch=True)
        self.log(f'{prefix}_S_f1', Sf1, on_step=False, on_epoch=True)
        self.log(f'{prefix}_V_f1', Vf1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # optimizer = SGD(self.parameters(), lr=5e-3, momentum=0.9, weight_decay=0.001)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_decay_milestones, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = MITBIHDataset(self.data_path, 'DS1', qdev=False)

        num_samples = len(train_dataset)
        class_counts = [(train_dataset.labels == i).sum() for i in range(self.num_classes)]

        class_weights = [num_samples / class_counts[i] for i in range(self.num_classes)]
        weights = [class_weights[train_dataset.labels[i]] for i in range(num_samples)]
        sampler = WeightedRandomSampler(weights, num_samples)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  sampler=sampler, num_workers=self.data_workers)
        return train_loader

    def val_dataloader(self):
        DS1_qdev = MITBIHDataset(self.data_path, 'DS1', qdev=True)
        DS2 = MITBIHDataset(self.data_path, 'DS2', qdev=None)
        loader1 = DataLoader(DS1_qdev, batch_size=self.batch_size, num_workers=self.data_workers)
        loader2 = DataLoader(DS2, batch_size=self.batch_size, num_workers=self.data_workers)
        return [loader1, loader2]

    def test_dataloader(self):
        test_dataset = MITBIHDataset(self.data_path, 'DS2', qdev=None)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.data_workers)
        return test_loader

from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb_path', '-tb', default='/nfs/c9_home/hrant/tb_logs/')
    parser.add_argument('--tb_name', '-n', default='ecg180-custom-wrs')
    parser.add_argument('--batch_size', '-bs', type=int, default=12)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--filters', '-f', type=int, default=16)
    parser.add_argument('--accumulate_gradient', '-ag', type=int, default=1)
    args = parser.parse_args()

    model = MainECG(batch_size=args.batch_size,
                    # conv_filters=args.filters,
                    learning_rate=args.learning_rate,
                    lr_decay_milestones=[50 * args.accumulate_gradient],
                    data_workers=4).cuda()
    logger = TensorBoardLogger(args.tb_path, name=args.tb_name)

    log_speed = max(1, 5000 // args.batch_size)
    trainer = pl.Trainer(logger=logger, log_every_n_steps=log_speed, max_epochs=500 * args.accumulate_gradient,
                         accumulate_grad_batches=args.accumulate_gradient)
    # trainer = pl.Trainer()
    trainer.fit(model)
