import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl


class MITBIHDataset(Dataset):
    def __init__(self, npy_file, part, qdev=False):
        """
        Args:
            npy_file (string): Path to the csv file with annotations.
            part (string): DS1 or DS2
            qdev (boolean): Use the first 20% if True, the other 80% otherwise
        """
        self.raw_data = np.load(npy_file, allow_pickle=True).tolist()
        if part not in ['DS1', 'DS2']:
            raise Exception("Incorrect part {}".format(part))

        def inside_qdev(idx, length):
            if qdev:
                return idx < 0.2 * length
            else:
                return idx > 0.2 * length

        self.data = [
            {
                "patient": patient,
                "x": x,
                "y": y
            }
            for patient in self.raw_data.keys()
            for i, (x, y) in enumerate(zip(self.raw_data[patient]['x'], self.raw_data[patient]['y']))
            if patient.startswith(part) and inside_qdev(i, len(self.raw_data[patient]['y']))
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        sample['x'] = torch.tensor(sample['x'], dtype=torch.float)
        sample['y'] = torch.tensor(sample['y'], dtype=torch.long)

        return sample



class MainECG(pl.LightningModule):
    def __init__(self, data_source='aligned180'):
        super().__init__()

        if data_source == 'aligned180':
            self.data_path = '/mnt/2tb/tigrann/domainbed/mit_bih_data.npy'  # aligned beats
            beat_length = 180
            class_weights = [1, 32, 13, 112]
        else:
            self.data_path = '/mnt/2tb/tigrann/domainbed/mit-bih.npy'  # non-aligned beats
            beat_length = 280
            class_weights = [1, 32, 13, 112, 6428]

        self.layer_1 = nn.Linear(beat_length, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, len(class_weights))

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.train_f1 = pl.metrics.classification.F1(5, average=None)
        self.valid_f1 = pl.metrics.classification.F1(5, average=None)

        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def forward(self, x):
        batch_size, beat_length = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        logits = self(batch['x'])
        loss = F.nll_loss(logits, batch['y'], weight=self.class_weights)
        self.train_acc(logits, batch['y'])
        Nf1, Sf1, Vf1, Ff1, Qf1 = self.train_f1(logits, batch['y'])
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_S_f1', Sf1, on_step=True, on_epoch=False)
        self.log('train_V_f1', Vf1, on_step=True, on_epoch=False)
        self.log('loss', loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['x'])
        # loss = F.nll_loss(logits, batch['y'])
        self.valid_acc(logits, batch['y'])
        Nf1, Sf1, Vf1, Ff1, Qf1 = self.valid_f1(logits, batch['y'])
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)
        self.log('valid_S_f1', Sf1, on_step=False, on_epoch=True)
        self.log('valid_V_f1', Vf1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        train_dataset = MITBIHDataset(self.data_path, 'DS1', qdev=False)
        train_loader = DataLoader(train_dataset, batch_size=64)
        return train_loader

    def val_dataloader(self):
        train_dataset = MITBIHDataset(self.data_path, 'DS1', qdev=True)
        train_loader = DataLoader(train_dataset, batch_size=64)
        return train_loader

    def test_dataloader(self):
        test_dataset = MITBIHDataset(self.data_path, 'DS2', qdev=False)
        test_loader = DataLoader(test_dataset, batch_size=64)
        return test_loader

from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    print("hello")
    model = MainECG()
    logger = TensorBoardLogger('/home/hrant/tb_logs/', name='ecg180-mlp-cw')
    trainer = pl.Trainer(logger=logger)
    # trainer = pl.Trainer()
    trainer.fit(model)
