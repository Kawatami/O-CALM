from __future__ import annotations
from argparse import ArgumentParser, Namespace
from source.utils.register import register
from enum import Enum
from source.utils.misc import load_from_file_cnllpp, list_file_conll
from source.data.data_modules.NERDataModule import NERDataModule

# defining WNUT17 labels
class CNLLPP_label(Enum):
    B_LOC = 0
    B_MISC = 1
    B_ORG = 2
    B_PER = 3
    I_LOC = 4
    I_MISC = 5
    I_ORG = 6
    I_PER = 7
    I_group = 8
    O = 9
    B_X = -100 # always ignore context token in loss processing


@register("DATASETS")
class CNLLPPDataModule(NERDataModule) :

    def __init__(self, **kwargs) :

        super().__init__(
            file_loader = load_from_file_cnllpp,
            file_detector= list_file_conll,
            label_enum = CNLLPP_label,
            **kwargs
        )


    @classmethod
    def from_args(cls, args : Namespace) -> CNLLPPDataModule :
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












