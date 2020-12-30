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
                 data_source='aligned180', data_workers=0, depth_multiplier = 1):
        super().__init__()

        if depth_multiplier not in [1, 2]:
            raise Exception(f"Depth multiplier {depth_multiplier} is not implemented")

        if lr_decay_milestones is None:
            lr_decay_milestones = []
        self.batch_size = batch_size
        self.data_workers = data_workers
        self.learning_rate = learning_rate
        self.lr_decay_milestones = lr_decay_milestones
        self.depth_multiplier = depth_multiplier

        if data_source == 'aligned180':
            self.data_path = '/nfs/c9_2tb/tigrann/domainbed/mit_bih_data.npy'  # aligned beats
            beat_length = 180
            class_weights = [1, 32, 13, 112]
        else:
            self.data_path = '/nfs/c9_2tb/tigrann/domainbed/mit-bih.npy'  # non-aligned beats
            beat_length = 280
            class_weights = [1, 32, 13, 112, 6428]

        self.num_classes = len(class_weights)

        c1, c2, c3, c4, c5 = conv_filters

        self.conv1 = nn.Conv1d(1, c1, 5, stride=2).cuda()
        if self.depth_multiplier == 2:
            self.conv1s = nn.Conv1d(c1, c1, 5, stride=1, padding=2).cuda()
        self.conv2 = nn.Conv1d(c1, c2, 5, stride=2).cuda()
        if self.depth_multiplier == 2:
            self.conv2s = nn.Conv1d(c2, c2, 5, stride=1, padding=2).cuda()
        self.conv3 = nn.Conv1d(c2, c3, 5, stride=2).cuda()
        if self.depth_multiplier == 2:
            self.conv3s = nn.Conv1d(c3, c3, 5, stride=1, padding=2).cuda()
        self.conv4 = nn.Conv1d(c3, c4, 5, stride=2).cuda()
        if self.depth_multiplier == 2:
            self.conv4s = nn.Conv1d(c4, c4, 5, stride=1, padding=2).cuda()
        self.conv5 = nn.Conv1d(c4, c5, 5, stride=2).cuda()
        if self.depth_multiplier == 2:
            self.conv5s = nn.Conv1d(c5, c5, 5, stride=1, padding=2).cuda()

        self.linear = nn.Linear(c5 * 2, self.num_classes)

        self.train_acc = pl.metrics.Accuracy()
        self.DS1_qdev_acc = pl.metrics.Accuracy()
        self.DS2_acc = pl.metrics.Accuracy()
        self.train_f1 = pl.metrics.classification.F1(4, average=None)
        self.train_cm = pl.metrics.classification.ConfusionMatrix(4)
        self.DS1_qdev_cm = pl.metrics.classification.ConfusionMatrix(4)
        self.DS2_cm = pl.metrics.classification.ConfusionMatrix(4)

    def _create_downsampler(self, inplanes, planes, stride):
        return nn.Sequential(
            conv1x1(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        batch_size, beat_length = x.size()

        x = x.view(batch_size, 1, beat_length) # (b, 1, 180)
        # x = x.repeat(1, 1, 11, 1)   # (b, 1, 11, 180)

        x = self.conv1(x)  # (b, 32, 88)  64=conv_filters
        if self.depth_multiplier == 2:
            x = self.conv1s(x)  # (b, 32, 88)  64=conv_filters
        x = self.conv2(x)  # (b, 64, 42)
        if self.depth_multiplier == 2:
            x = self.conv2s(x)  # (b, 64, 42)
        x = self.conv3(x)  # (b, 256, 19)
        if self.depth_multiplier == 2:
            x = self.conv3s(x)  # (b, 256, 19)
        x = self.conv4(x)  # (b, 64, 8)
        if self.depth_multiplier == 2:
            x = self.conv4s(x)  # (b, 64, 8)
        x = self.conv5(x)  # (b, 32, 2)
        if self.depth_multiplier == 2:
            x = self.conv5s(x)  # (b, 32, 2)

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

        Nf1, Sf1, Vf1, Ff1 = self.train_f1(logits, y)
        cm = self.train_cm(logits, y)
        Vf1_cm = self._calc_f1(cm, 2)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_S_f1', Sf1, on_step=True, on_epoch=False)
        self.log('train_V_f1', Vf1, on_step=True, on_epoch=False)
        self.log('train_Vf1', Vf1_cm, on_step=True, on_epoch=False)
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
            cm = self.DS1_qdev_cm(logits, y)
        elif dataloader_idx == 1:
            prefix = 'DS2'
            acc = self.DS2_acc(logits, y)
            cm = self.DS2_cm(logits, y)
        else:
            raise Exception(f"Unknown dataloader_idx={dataloader_idx}")

        self.log(f'{prefix}_loss', loss, on_step=False, on_epoch=True)
        self.log(f'{prefix}_acc', acc, on_step=False, on_epoch=True)
        return cm

    @classmethod
    def _calc_f1(cls, cm, index):
        f1_TP = cm[index, index]
        f1_FPTP = cm[:, index].sum()  # - Vf1_TP
        f1_FNTP = cm[index, :].sum()  # - Vf1_TP
        f1_cm = 2 / (f1_FPTP / f1_TP + f1_FNTP / f1_TP)
        return f1_cm

    def validation_epoch_end(self, outputs):
        outputs_DS1qdev, outputs_DS2 = outputs

        outputs_DS1qdev = np.array(outputs_DS1qdev)
        cm = outputs_DS1qdev.sum(axis=0)
        self.log('DS1_qdev_VF1', self._calc_f1(cm, 2))
        self.log('DS1_qdev_SF1', self._calc_f1(cm, 1))

        outputs_DS2 = np.array(outputs_DS2)
        cm = outputs_DS2.sum(axis=0)
        self.log('DS2_VF1', self._calc_f1(cm, 2))
        self.log('DS2_SF1', self._calc_f1(cm, 1))

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
        class_counts[-1] = 1e7

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
    parser.add_argument('--filters', '-f', type=int, default=1)
    parser.add_argument('--accumulate_gradient', '-ag', type=int, default=1)
    parser.add_argument('--gradient_clip', '-gc', type=float, default=0)
    args = parser.parse_args()

    model = MainECG(batch_size=args.batch_size,
                    conv_filters=np.array((32, 64, 256, 64, 32)) * args.filters,
                    learning_rate=args.learning_rate,
                    lr_decay_milestones=[50000 * args.accumulate_gradient],
                    data_workers=0).cuda()
    logger = TensorBoardLogger(args.tb_path, name=args.tb_name)

    log_speed = max(1, 5000 // args.batch_size)
    trainer = pl.Trainer(logger=logger, log_every_n_steps=log_speed, max_epochs=5000 * args.accumulate_gradient,
                         accumulate_grad_batches=args.accumulate_gradient, log_gpu_memory='all', track_grad_norm=2,
                         gradient_clip_val=args.gradient_clip)
    # trainer = pl.Trainer()
    trainer.fit(model)
