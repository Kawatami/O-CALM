import json
import pathlib
from torch.utils.data import Dataset
import random
from transformers import AutoTokenizer

class SelectorDataset(Dataset) :

    def __init__(
            self,
            data,
            exclude_prompt
    ):

        self.exclude_prompt = exclude_prompt

        self.data = data

        self.random_index = [random.randint(0, 7) for _ in range(len(data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_point = self.data[idx]

        if data_point['best_context'] in self.exclude_prompt :
            context_selector = self.random_index[idx]
        elif data_point['best_context'] == "equal" :
            context_selector = self.random_index[idx]
        else :
            context_selector = list(data_point['context']).index(data_point['best_context'])


        sample = {
            "input" : data_point['input'],
            "labels" : context_selector
        }


        return sample