from pytorch_lightning.callbacks import Callback
from argparse import Namespace


class BaseCallback(Callback) :
    """
    Base class for callbacks.
    """

    @staticmethod
    def add_callback_specific_args(parent_parser):
        return parent_parser

    @classmethod
    def build_from_args(cls, args : Namespace) :
        raise NotImplementedError("Base class callbacks should be inherited")