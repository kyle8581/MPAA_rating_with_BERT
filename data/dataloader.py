from torch.utils.data import DataLoader
import sys
sys.path.append("../")
from dataset import ScriptDataset


def get_dataloader(mode, batch_size):
    dataset = ScriptDataset(mode)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

