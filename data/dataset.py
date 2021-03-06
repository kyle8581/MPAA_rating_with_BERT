import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
import json
import pandas as pd
import ast


class ScriptDataset(Dataset):
    def __init__(self, mode, base_dir):
        """
        params:

            mode : ("train" || "val" || "test")

        """
        partition_file = open(os.path.join(
            base_dir, "data/partition.json"), "r")
        self.script_id_list = json.load(partition_file)[mode]
        self.data = pd.read_csv(os.path.join(
            base_dir, "data/preprocessed_data.csv"))

    def __len__(self):
        """
        returns length of this dataset
        """
        return len(self.script_id_list)

    def __getitem__(self, index):

        ret_dict = {}

        script_id = self.script_id_list[index]
        row = self.data[self.data["IMDB_id"] == script_id]
        print(row["tokenized_script"])
        ret_dict["x"] = torch.LongTensor(
            ast.literal_eval(list(row["tokenized_script"])[0]))
        ret_dict["y"] = torch.LongTensor(list(row["rating"]))

        return ret_dict
