from __future__ import annotations
from argparse import ArgumentParser, Namespace
from source.utils.register import register
from typing import Callable, Type, Optional, List
from enum import Enum
from source.utils.misc import load_from_file, list_file_bc5cdr
from source.data.data_modules.NERDataModule import NERDataModule
from datasets import Dataset
from source.utils.HyperParametersManagers import HyperParametersManager

# defining BC5CDR labels
class BC5CDR_label(Enum):
    B_Chemical = 0
    B_Disease = 1
    I_Chemical = 2
    I_Disease = 3
    O = 4
    B_X = -100 # always ignore context token in loss processing


@register("DATASETS")
class Bc5cdrDataModule(NERDataModule) :

    def __init__(
            self,
            merge_train_dev : bool = False,
            **kwargs
    ) :

        super().__init__(
            file_loader = load_from_file,
            label_enum = BC5CDR_label,
            file_detector = list_file_bc5cdr,
            **kwargs
        )

        self.merge_train_dev = merge_train_dev

    def setup(self, stage: Optional[str] = None) -> None:

        # listing text files
        files = self.list_file(self.data_dir)

        # sanity check
        if "train" not in files:
            raise RuntimeError(f"ERROR loading files : Missing \"train\" file. Found {' '.join(files.keys())}"
                               f". This error might occur when selecting the wrong folder (e.g bc5cdrDataLoader with WNUT17 folder)")
        if "test" not in files:
            raise RuntimeError(f"ERROR loading files : Missing \"test\" file. Found {' '.join(files.keys())}"
                               f". This error might occur when selecting the wrong folder (e.g bc5cdrDataLoader with WNUT17 folder)")
        if "dev" not in files:
            raise RuntimeError(f"ERROR loading files : Missing \"dev\" file. Found {' '.join(files.keys())}"
                               f". This error might occur when selecting the wrong folder (e.g bc5cdrDataLoader with WNUT17 folder)")

        # loading from files
        val = self.load_from_file(files['dev'])
        train = self.load_from_file(files['dev'])

        if self.merge_train_dev :
            train += val

        self.sets = {
            "train": train,
            "val": val,
            "test": self.load_from_file(files['test'])
        }

        def gen(samples):
            for sample in samples:
                yield sample

        # loading from files
        self.sets = {
            "train": self.create_dataLoader(Dataset.from_generator(gen, gen_kwargs={"samples": self.sets['train']})),
            "val": self.create_dataLoader(Dataset.from_generator(gen, gen_kwargs={"samples": self.sets['val']})),
            "test": self.create_dataLoader(Dataset.from_generator(gen, gen_kwargs={"samples": self.sets['test']}))
        }

        # storing info
        HyperParametersManager()['n_training_steps'] = len(self.sets['train']) / self.gpus
        HyperParametersManager()['n_val_steps'] = len(self.sets['val']) / self.gpus
        HyperParametersManager()['n_test_steps'] = len(self.sets['test']) / self.gpus

        self.data_loaded = True


    @classmethod
    def from_args(cls, args : Namespace) -> Bc5cdrDataModule :
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
        parser = NERDataModule.add_data_specific_args(parent_parser)

        parser.add_argument("--merge_train_dev", action='store_true', default = False)

        return parser












