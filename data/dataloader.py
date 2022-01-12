from dataset import ScriptDataset
from torch.utils.data import DataLoader
import sys
sys.path.append("../")


def get_dataloader(mode, batch_size, base_dir):
    dataset = ScriptDataset(mode, base_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
