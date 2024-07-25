from __future__ import annotations
import warnings
from argparse import ArgumentParser, Namespace
from typing import Callable, Type, Optional, List
import json
import logging
import pathlib
from source.data.data_modules.base_datamodule import BaseDataModule
from torch.utils.data.dataloader import DataLoader
from source.data.datasets.SelectorDataset import SelectorDataset
from functools import cached_property
from transformers import AutoTokenizer
import os
from source.utils.register import register

import torch

@register("DATASETS")
class SelectorDataModule(BaseDataModule) :
    """
    Base class for NER DataLoading
    """

    def __init__(self,
                 data_dir: pathlib.Path,
                 seed: int = 42,
                 batch_size: int = 32,
                 batch_size_test: int = 10,
                 single_sample_training : bool = False,
                 tokenizer_local_load : Optional[str] = None,
                 tokenizer_key : str = "xlm-roberta-large",
                 gpus : int = None,
                 shuffle : bool = True,
                 **kwargs
        ) :
        """

        :param collector_log_dir:
        :param data_dir:
        :param file_loader:
        :param verbose:
        :param seed:
        :param batch_size:
        :param batch_size_test:
        :param single_sample_training:
        :param kwargs:
        """

        super().__init__(**kwargs)

        self.tokenizer_key = tokenizer_key
        self.shuffle = shuffle

        # checking datadir exist
        self.data_dir = data_dir

        assert 1 <= batch_size, "batch size should be at least 1"
        self.batch_size = batch_size

        assert 1 <= batch_size_test, "batch size test should be at least 1"
        self.batch_size_test = batch_size_test

        self.seed = seed

        self.single_sample = single_sample_training
        if single_sample_training:
            logging.warn("#### SINGLE SAMPLE MODE ENABLED")

        self.data_loaded = False

        self.tokenizer_local_load = tokenizer_local_load

        if gpus is None :
            warnings.warn(f"### WARNING : No GPU is defined, assuming only one is used or CPU only. Setting to 1.")
            gpus = 1
        self.gpus = gpus



    def update_args(self, args : Namespace) :
        return args

    @classmethod
    def from_args(cls, args : Namespace) -> SelectorDataModule :
        """
        Build Rotowire Task 1 Datamodule from parser Namspace args
        :param args: namespace from main parser
        :return: RotoWireTask1DataModule object
        """
        return cls(**vars(args))

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add Data Module specific args to the main parser
        :param parent_parser: main parser
        :return: updated main parser
        """
        parent_parser = BaseDataModule.add_data_specific_args(parent_parser)

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_dir", type = pathlib.Path, default = None)

        parser.add_argument("--batch_size", type = int, default = 32)
        parser.add_argument("--batch_size_test", type = int, default = 10)

        parser.add_argument("--verbose", action = "store_true", default = False)
        parser.add_argument("--seed", type = int, default = 42)
        parser.add_argument("--single_sample_training", action="store_true", default = False)
        parser.add_argument("--no_shuffle", action="store_false", default = True)
        parser.add_argument("--tokenizer_local_load", type = str, default = None)
        parser.add_argument("--tokenizer_key", type = str, default = "xlm-roberta-large")

        return parser


    def collate_function(self, samples):

        # tokenizing
        encoded_input = self.model_tokenizer(
            [data_point['input'] for data_point in samples],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        encoded_input['labels'] = torch.tensor([data_point['labels'] for data_point in samples])

        return encoded_input

    def prepare_data(self) -> None :

        with self.data_dir.open("r") as file :
            self.data = json.load(file)

    def setup(self, stage: Optional[str] = None) -> None :

        # creating datasets
        # REMOVE BEFORE FLIGHT : hard coded constraints
        datasets = SelectorDataset(self.data, ['results_NER_all'])

        self.sets = {
            "train" : DataLoader(datasets, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_function),
            "val": DataLoader(datasets, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_function),
            "test": DataLoader(datasets, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_function)

        }

    @cached_property
    def model_tokenizer(self) :

        if self.tokenizer_local_load is None :

            print("######### LOADING FROM WEB")
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_key,
                use_fast = True,
                add_prefix_space=True,
                model_max_length=512
            )
        else :
            tokenizer = self.load_tokenizer_locally(self.tokenizer_local_load)

        tokenizer.add_tokens(["<EOS>"], special_tokens=True)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def load_tokenizer_locally(self, tokenizer_local_load) :
        """
        Load the tokenizer from local files given the env variable MODEL_FILES is defined.
        :param tokenizer_local_load: subdirectory holding the model files
        :return : the tokenizer
        """

        print("## LOCAL LOAD TOKENIZER")

        # sanity check
        if "MODEL_FILES" not in os.environ :
            raise EnvironmentError(f"No $MODEL_FILES env variable have been defined. Make sure to set it "
                                   f"model file location in order to load the tokenizer locally.")

        # creating path
        path_tokenizer = pathlib.Path(os.environ['MODEL_FILES']) / tokenizer_local_load

        if not path_tokenizer.exists() :
            raise FileNotFoundError(f"Directory {path_tokenizer} not found.")

        # loading model
        model = AutoTokenizer.from_pretrained(
            path_tokenizer,
            use_fast = True,
            add_prefix_space = True,
            padding="max_length",
            is_split_into_words=True,
            truncation=True,
            return_tensors="pt",
            model_max_length=512
        )

        return model

    def _get_dataloader(self, split='train') -> Type[DataLoader]:

        if split not in self.sets :
            raise ValueError(f"Unavailable split {split}. Available {' '.join(self.sets.keys())}")

        return self.sets[split]









