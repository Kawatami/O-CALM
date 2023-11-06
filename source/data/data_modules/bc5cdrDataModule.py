from __future__ import annotations
from argparse import ArgumentParser, Namespace
from source.utils.register import register
from enum import Enum
from source.utils.misc import load_from_file, list_file_bc5cdr
from source.data.data_modules.NERDataModule import NERDataModule

# defining WNUT17 labels
class BC5CDR_label(Enum):
    B_Chemical = 0
    B_Disease = 1
    I_Chemical = 2
    I_Disease = 3
    O = 4
    B_X = -100 # always ignore context token in loss processing


@register("DATASETS")
class Bc5cdrDataModule(NERDataModule) :

    def __init__(self, **kwargs) :

        super().__init__(
            file_loader = load_from_file,
            label_enum = BC5CDR_label,
            file_detector = list_file_bc5cdr,
            **kwargs
        )


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

        return parser












