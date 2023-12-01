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
class BaselineModel(torch.nn.Module) :
    """
    Baseline model as defined in https://arxiv.org/abs/2105.03654
    """

    def __init__(
            self,
            num_label : int = 13,
            dropout : Optional[float] = None,
            training_key : str = "xlm-roberta-large",
            transformer_local_load : Optional[str] = None,
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

        # setting classifier
        self.classfier = torch.nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=num_label
        )

        # droptout
        self.dropout = torch.nn.Dropout(p=dropout) if dropout is not None else None

        # defining CRF
        self.CRF = PartialCRF(
            num_tags = num_label,
            batch_first = True
        )


    def forward(self, batch : Dict) -> Dict :
        """
        forward function of the model. Process prediction using viterbi decoding, CRF posterior. Some
        intermediate representation are stored within the dict "batch"
        :param batch: Dict holding results and hidden states
        :return: Dict updated
        """

        # forward with context
        hidden_states_with_context = self.model(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            #token_type_ids = batch['token_type_ids'],
        ).last_hidden_state

        # forward without context
        hidden_states_without_context = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask_without_context'],
        ).last_hidden_state

        # applying drop out
        if self.dropout is not None :
            hidden_states_with_context = self.dropout(hidden_states_with_context)
            hidden_states_without_context = self.dropout(hidden_states_without_context)

        # storing info
        batch['context_hidden_states'] = hidden_states_with_context
        batch['hidden_states'] = hidden_states_without_context


        # classifying
        logits_with_context = self.classfier(hidden_states_with_context)
        logits_without_context = self.classfier(hidden_states_without_context)

        # apply mask
        logits_with_context = logits_with_context * batch['attention_mask_without_context'].unsqueeze(-1)
        logits_without_context = logits_without_context * batch['attention_mask_without_context'].unsqueeze(-1)

        # CRF layer posterior
        mask_crf = batch['attention_mask_without_context'] # != -100 # ignoring sub tokens
        CRF_prediction_with_context = self.CRF.compute_posterior(logits_with_context.detach(), mask_crf)
        CRF_prediction_without_context = self.CRF.compute_posterior(logits_without_context, mask_crf)
        batch["CRF_posterior_with_context"] = CRF_prediction_with_context # detach cause we back-propagate only to
        batch["CRF_posterior_without_context"] = CRF_prediction_without_context # sequence without context

        # processing CRF loss
        CRF_loss_with_context = - self.CRF(logits_with_context, batch['labels'], mask_crf)
        CRF_loss_without_context = - self.CRF(logits_without_context, batch['labels'], mask_crf)
        batch["CRF_loss_with_context"] = CRF_loss_with_context
        batch["CRF_loss_without_context"] = CRF_loss_without_context

        # prediction
        prediction_with_context_viterbi = self.CRF.decode(logits_with_context, mask_crf)
        prediction_without_context_viterbi = self.CRF.decode(logits_without_context, mask_crf)
        batch["prediction_with_context"] = prediction_with_context_viterbi[0]
        batch["prediction_without_context"] = prediction_without_context_viterbi[0]

        return batch

    def get_optimizer_config(
            self,
            learning_rate,
            accumulate_grad_batches : int = 1,
            max_epochs : int = None,
            **kwargs
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict]] :
        """
        Set up the optimizer and learning rate scheduler. AdamW is used as optimizer and Cycling cosine scheduler
        :param learning_rate: base learning rate
        """

        # sanity check
        if 'n_training_steps' not in HyperParametersManager() :
            raise ValueError(f"No \"n_training_steps\" is defined in the HyperParametersManager object. Verify it is defined in "
                             f"during the datasets set up phase.")
        if max_epochs is None :
            raise ValueError(f"No \"max_epochs\" is defined in the main Namespace object. Verify it is defined in "
                             f"the high level parser.")

        # defining paramters groups and their associated learning rates
        optimizer_param_groups = [
            {
                "params": self.model.parameters(),
                "lr" : learning_rate,
                "weight_decay" : 0.1,
                'name': 'language_model'
            },
            {
                "params": self.classfier.parameters(),
                "lr": learning_rate,
                "weight_decay": 0.1,
                'name': 'classifier'
            },
            {
                "params": self.CRF.parameters(),
                "lr": 5e-2,
                "weight_decay": 0.01,
                'name': 'CRF Layer'
            }
        ]

        # defining optimizer
        optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=learning_rate,
            weight_decay=0.1
        )

        # TODO : handling training configuration (number of steps, number of GPUS and gradient accumulation)
        num_training_steps = (HyperParametersManager()['n_training_steps']  * max_epochs) // accumulate_grad_batches
        num_warmup_steps = int(0.1 * num_training_steps)

        print(f"num train step : {num_warmup_steps}")
        print(f"num warmup step : {num_warmup_steps}")

        # setting up the learning rate scheduler
        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda, -1),
            'interval': 'step',  # or 'epoch'
            'frequency' : 1
        }


        return [optimizer], [scheduler]

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
            path_model,
            device_map='auto',
            local_files_only=True
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
    def from_args(cls, args: Namespace) -> BaselineModel :
        """
        Build model from argument namsepace
        :param args: namespace from main parser
        :return: RotoWireTask1DataModule object
        """
        return cls(**vars(args))
