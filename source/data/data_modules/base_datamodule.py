from __future__ import annotations
from argparse import ArgumentParser, Namespace
from typing import Type, Optional, Callable
import pathlib
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import torch


class BaseDataModule(pl.LightningDataModule):
    """
    Base data module used by pytorch lighning for data management
    """
    def __init__(self, **kwargs):
        """
        Super class constructor. To be overrided.
        :param data_directory: path pointing to the data
        """
        super().__init__()


    def update_args(self, args : Namespace) :
        """
        Method used to update the args Namespace, this is typically used to pass
        process data to the task object
        """

        return args

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser :
        """
        Add data module specific args. To be overrided
        :param parent_parser: main parser
        :return: main  parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Base Dataset')

        return parser

    def process_class_weights(self, data) :
        """
        Process class weights in the case of classification task
        """
        raise NotImplementedError

    @classmethod
    def from_args(cls, args : Namespace) -> Type[BaseDataModule]:
        """
        Build data module from namespace `args` issued by the main parser
        :param args: namespace from the main parser
        :return: unstance of subclass BaseDataModule
        """
        raise NotImplementedError

    def update_batch_size(self, rate : int) -> None :
        """
        Update batchsize on the fly
        :param rate: rate of change
        """
        self.batch_size *= rate

    def train_dataloader(self) -> Type[DataLoader]:
        """
        Get the train dataloader
        :return: train dataloader
        """
        return self._get_dataloader('train')

    def val_dataloader(self) -> Type[DataLoader]:
        """
        Get the val dataloader
        :return: val dataloader
        """
        return self._get_dataloader('val')

    def test_dataloader(self) -> Type[DataLoader]:
        """
        Get the test dataloader
        :return: test dataloader
        """
        return self._get_dataloader('test')

    def _get_dataloader(self, split='train') -> Type[DataLoader] :
        """
        Private methods to get a specific dataloader
        :param split: split to get amoong "train", "val", "test"
        :return: DataLoader
        """
        raise NotImplementedError

    def prepare_data(self, *args, **kwargs) -> None :
        """
        Load data.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Format the data into model feedable structure and produce DataLoader for
        each split
        :param stage:
        """
        pass

