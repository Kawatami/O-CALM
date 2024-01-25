from __future__ import annotations
from argparse import ArgumentParser, Namespace
from source.utils.register import register
from enum import Enum
from source.utils.misc import load_from_file, list_file_bc5cdr
from source.data.data_modules.NERDataModule import NERDataModule

# defining BC5CDR labels
class ONTONOTE_label(Enum):
    O= 0
    B_CARDINAL= 1
    B_DATE= 2
    I_DATE= 3
    B_PERSON= 4
    I_PERSON= 5
    B_NORP= 6
    B_GPE= 7
    I_GPE= 8
    B_LAW= 9
    I_LAW= 10
    B_ORG= 11
    I_ORG= 12
    B_PERCENT= 13
    I_PERCENT= 14
    B_ORDINAL= 15
    B_MONEY= 16
    I_MONEY= 17
    B_WORK_OF_ART= 18
    I_WORK_OF_ART= 19
    B_FAC= 20
    B_TIME= 21
    I_CARDINAL= 22
    B_LOC= 23
    B_QUANTITY= 24
    I_QUANTITY= 25
    I_NORP= 26
    I_LOC= 27
    B_PRODUCT= 28
    I_TIME= 29
    B_EVENT= 30
    I_EVENT= 31
    I_FAC= 32
    B_LANGUAGE= 33
    I_PRODUCT= 34
    I_ORDINAL= 35
    I_LANGUAGE= 36
    B_X = -100 # always ignore context token in loss processing


@register("DATASETS")
class OntonoteDataModule(NERDataModule) :

    def __init__(self, **kwargs) :

        super().__init__(
            file_loader = load_from_file,
            label_enum = ONTONOTE_label,
            file_detector = list_file_bc5cdr,
            **kwargs
        )


    @classmethod
    def from_args(cls, args : Namespace) -> OntonoteDataModule :
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












