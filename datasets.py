import numpy as np
import torch
from torch.utils.data import Dataset


class MITBIHDataset(Dataset):
    def __init__(self, npy_file, part, qdev=None):
        """
        Args:
            npy_file (string): Path to the csv file with annotations.
            part (string): DS1 or DS2
            qdev (boolean|None): Use the first 20% if True, the other 80% if False, all if None
        """
        self.raw_data = np.load(npy_file, allow_pickle=True).tolist()
        if part not in ['DS1', 'DS2']:
            raise Exception("Incorrect part {}".format(part))

        def inside_qdev(idx, length):
            if qdev is None:
                return True
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

        self.labels = np.array([sample['y'] for sample in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        # sample['x'] = torch.tensor(sample['x'], dtype=torch.float).cuda()
        # sample['y'] = torch.tensor(sample['y'], dtype=torch.long).cuda()

        return sample
