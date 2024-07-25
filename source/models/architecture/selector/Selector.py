from __future__ import annotations

import os
import pathlib

import torch
from typing import Dict, Optional, Tuple, List, Type
from transformers import AutoModel, PreTrainedModel
from source.models.common.CRF import CRF, PartialCRF
from source.utils.register import register
from argparse import ArgumentParser, Namespace
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import math
from source.utils.HyperParametersManagers import HyperParametersManager
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float
):
    # warmup steps
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


@register("MODELS")
class SelectorModel(torch.nn.Module) :
    """
    Baseline model as defined in https://arxiv.org/abs/2105.03654
    """

    def __init__(
            self,
            num_label : int = 13,
            dropout : Optional[float] = None,
            training_key : str = "xlm-roberta-large",
            transformer_local_load : Optional[str] = None,
            classifier_hidden_size : int = 512,
            dropout_classifier : float = 0.3,
            freeze_transformer : bool = True,
            **kwargs
    ):
        """
        baseline model coomposed of several submodules call in this order :
            1. model : pre-trained transformer, xlm-RoBERTa in the original paper
            2. classifier : Simple Linear layer for prediction\
            3. CRF : Conditional random Field used to model labels transitions

        :param num_label: (int) number of output classes
        :param dropout: dropout rate before classifier
        :param kwargs:
        """
        super().__init__()

        # storing haparams
        self.hparams = {
            "num_label" : num_label,
            "dropout" : dropout if dropout is not None else 0.0,
            "training_key" : training_key
        }

        # defining model
        if transformer_local_load is None :
            self.model = AutoModel.from_pretrained(
                training_key
            )
        else :
            self.model = self.load_model_locally(transformer_local_load)

        self.model.requires_grad_(not freeze_transformer)

        # setting classifier
        self.classifier = torch.nn.Sequential(*[
            torch.nn.Dropout(p=dropout_classifier),
            torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=classifier_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_classifier),
            torch.nn.Linear(in_features=classifier_hidden_size, out_features=num_label)
        ])


    def forward(self, batch : Dict) -> Dict :
        """
        forward function of the model. Process prediction using viterbi decoding, CRF posterior. Some
        intermediate representation are stored within the dict "batch"
        :param batch: Dict holding results and hidden states
        :return: Dict updated
        """

        # forward with context
        hidden_states = self.model(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            #token_type_ids = batch['token_type_ids'],
        ).last_hidden_state

        # storing info
        batch['prediction'] =  self.classifier(hidden_states[:, 0])

        return batch

    def get_optimizer_config(
            self,
            learning_rate,
            accumulate_grad_batches : int = 1,
            max_epochs : int = None,
            **kwargs
    )  :
        """
        Set up the optimizer and learning rate scheduler. AdamW is used as optimizer and Cycling cosine scheduler
        :param learning_rate: base learning rate
        """

        # defining paramters groups and their associated learning rates
        optimizer_param_groups = [
            {
                "params": self.model.parameters(),
                "lr" : learning_rate,
                "weight_decay" : 0.1,
                'name': 'language_model'
            },
            {
                "params": self.classifier.parameters(),
                "lr": learning_rate,
                "weight_decay": 0.1,
                'name': 'classifier'
            }
        ]

        # defining optimizer
        optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=learning_rate,
            weight_decay=0.1
        )


        return optimizer, None

    def load_model_locally(self, transformer_local_load) -> Type[PreTrainedModel] :
        """
        Load the transformers model from local files given the env variable MODEL_FILES is defined.
        :param transformer_local_load: subdirectory holding the model files
        :return : the loaded model
        """

        # sanity check
        if "MODEL_FILES" not in os.environ :
            raise EnvironmentError(f"No $MODEL_FILES env variable have been defined. Make sure to set it "
                                   f"model file location in order to load them locally.")

        # creating path
        path_model = pathlib.Path(os.environ['MODEL_FILES']) / transformer_local_load

        # loading model
        model = AutoModel.from_pretrained(
            path_model
        )

        return model

    def export_hparams(self, hparams) -> None :
        """
        Store the model's hyperparameters
        :param hparams: hparams attribute of main pl.LightningModule
        """

        hparams.model_hparams = self.hparams

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add model specific args to main parser
        :param parent_parser: main parser
        :return: main parser updated
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--num_label", type=int, default=13)
        parser.add_argument("--training_key", type=str, default='xlm-roberta-large')
        parser.add_argument("--transformer_local_load", type=str, default=None)
        return parser

    @classmethod
    def from_args(cls, args: Namespace) -> SelectorModel :
        """
        Build model from argument namsepace
        :param args: namespace from main parser
        :return: RotoWireTask1DataModule object
        """
        return cls(**vars(args))
