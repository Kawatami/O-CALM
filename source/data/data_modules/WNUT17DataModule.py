from __future__ import annotations
from argparse import ArgumentParser, Namespace
from source.utils.register import register
from enum import Enum
from source.utils.misc import load_from_file, list_file
from source.data.data_modules.NERDataModule import NERDataModule

# defining WNUT17 labels
class WNUT17_label(Enum):
    B_corporation = 0
    B_creative_work = 1
    B_group = 2
    B_location = 3
    B_person = 4
    B_product = 5
    I_corporation = 6
    I_creative_work = 7
    I_group = 8
    I_location = 9
    I_person = 10
    I_product = 11
    O = 12
    B_X = -100 # always ignore context token in loss processing


@register("DATASETS")
class WNUT17DataModule(NERDataModule) :

    def __init__(self, **kwargs) :

        super().__init__(
            file_loader = load_from_file,
            label_enum = WNUT17_label,
            **kwargs
        )


    @classmethod
    def from_args(cls, args : Namespace) -> WNUT17DataModule :
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












