from __future__ import annotations
from pytorch_lightning.utilities.types import STEP_OUTPUT
from source.callbacks.base_callback import  BaseCallback
from argparse import Namespace, ArgumentParser
import pytorch_lightning as pl
from source.utils.register import register
from pytorch_lightning.callbacks import Callback
import pathlib
import json
import torch
import numpy as np
import warnings
from typing import *
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

@register("CALLBACKS")
class LearningRateMonitor(BaseCallback, LearningRateMonitor) :

    _name = ["LearningRateMonitor"]

    def __init__(self,
                 logging_interval='step',
                 log_momentum=False,
                 **kwargs):
        super().__init__(logging_interval=logging_interval,
                         log_momentum=log_momentum)

    @staticmethod
    def add_callback_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add early stopping specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('LearningRateMonitor')

        group.add_argument("--logging_interval", type=str, default="step")
        group.add_argument("--log_momentum", action='store_true',default=False)

        return parser

    @classmethod
    def build_from_args(cls, args: Namespace) -> LearningRateMonitor:
        """
        Build Early stopping object from args object issued by the parser
        :param args:
        :return:
        """
        return cls(**vars(args))


@register("CALLBACKS")
class EarlyStoppingWrapper(BaseCallback, EarlyStopping) :
    """
    Early stopping wrapper for class from pytorch lightning. Used to be
    with the call back factory.
    """
    _names = ["EarlyStopping"]

    def __init__(self,
                 earlyStopping_monitor : str,
                 patience : int = 5,
                 mode : str = "min"
                 ) :
        super().__init__(monitor=earlyStopping_monitor,
                         patience=patience,
                         mode=mode,
                         save_weights_only=True,
                         every_n_train_steps=10000)

    @staticmethod
    def add_callback_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :
        """
        Add early stopping specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('EarlyStopping')

        group.add_argument("--earlyStopping_monitor", type=str,
                           choices=['trainset', 'valset'],
                           default = "valset")
        group.add_argument("--earlyStopping_mode", type=str,
                           choices=['min', 'max'],
                           default="min")
        group.add_argument("--earlyStopping_patience", type=int,
                           default=20)
        return parser

    @classmethod
    def build_from_args(cls, args : Namespace) -> EarlyStoppingWrapper:
        """
        Build Early stopping object from args object issued by the parser
        :param args:
        :return:
        """
        monitor = f"{args.loss}/{args.earlyStopping_monitor}"
        return cls(monitor,
                   args.earlyStopping_patience,
                   args.earlyStopping_mode)

@register("CALLBACKS")
class ModelCheckpointWrapper(BaseCallback, ModelCheckpoint) :
    """
    Model checkpoint wrapper for class from pytorch lightning. Used to be
    with the call back factory.
    """
    _names = ["ModelCheckpoint"]

    def __init__(
            self,
            modelCheckpoint_monitor : str,
            mode : str = "min",
            modelCheckpoint_every_n_train_steps : int = None
        ) :
        super().__init__(
            monitor=modelCheckpoint_monitor,
            mode=mode,
            every_n_train_steps=modelCheckpoint_every_n_train_steps,
            save_last=True,
            every_n_epochs=1,
            verbose=True,
            save_top_k=1

        )

    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('modelCheckpoint')

        group.add_argument("--modelCheckpoint_monitor", type=str,
                           default = "valset")
        group.add_argument("--modelCheckpoint_mode", type=str,
                           choices=['min', 'max'],
                           default="min")
        group.add_argument("--modelCheckpoint_every_n_train_steps", type = int, default = None)
        return parser

    @classmethod
    def build_from_args(cls, args : Namespace) -> ModelCheckpointWrapper :
        """
        Build Early stopping object from args object issued by the parser
        :param args:
        :return:
        """
        if args.modelCheckpoint_monitor == "trainset" or args.modelCheckpoint_monitor == "valset" :
            monitor = f"{args.loss}/{args.modelCheckpoint_monitor}"
        else :
            monitor = args.modelCheckpoint_monitor

        return cls(
            monitor,
            args.modelCheckpoint_mode,
            args.modelCheckpoint_every_n_train_steps
        )

