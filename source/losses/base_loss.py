from __future__ import annotations
from argparse import ArgumentParser, Namespace
from source.utils.register import register, Registers
import torch
from typing import Tuple, Type, List

class BaseLoss(torch.nn.Module):
    """
    Base class for Loss object
    """
    def __init__(self, log_name : str = None, **kwargs) :
        """
        Base class constructor
        :param log_name: name to use in the case of metric usage
        """
        super().__init__()
        self.log_name = log_name

    @staticmethod
    def _prep_inputs(model_outputs : dict, batch : dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the model output and labels to be processed by a loss object
        :param model_outputs: model output
        :param batch: batch object
        :return: Associated tensors
        """
        prd, tgt = model_outputs['prediction'], batch['labels'].float()
        prd, tgt = prd.view(-1), tgt.view(-1)
        return prd, tgt

    @staticmethod
    def add_loss_specific_args(parent_parser : ArgumentParser) -> ArgumentParser:
        return parent_parser

    @classmethod
    def from_args(cls, args : Namespace) -> BaseLoss:
        """
        Build Loss object from Namespace issued by main parser
        :param args:
        :return:
        """

        return cls(**vars(args))

    @property
    def abbrv(self):
        return self.__class__.__name__

@register('LOSSES')
class MultiTaskLoss(torch.nn.Module):
    """
    Base class for Multi task learning.

    ARGS :
        losses Union[BaseLoss] : list of losses
        weights Union[float] : one weight per loss, they sum to one.
        inputs Union[str] : list of prediction keys on which to apply the losses
        outputs Union[str] : list of gold standard keys for each loss
    """

    _names = ['MultiTask']

    @property
    def name(self):
        return 'MT'

    def __init__(
            self,
            losses,
            loss_weights,
            loss_input_keys,
            loss_target_keys,
            loss_names : List[str],
            **kwargs
    ):
        super().__init__()

        # input sanity check
        if len(loss_weights) == 1 and len(losses) == 2:
            loss_weights.append(1 - loss_weights[0])
        else:
            assert len(loss_weights) == len(losses), f'{len(loss_weights)} weights specified, but you have {len(losses)} losses!'

        unknown_losses = [loss for loss in losses if loss not in Registers['LOSSES']]
        if len(unknown_losses):
            raise ValueError(f'Unknown losses: {unknown_losses}')

        if loss_names is not None :
            assert len(losses) == len(loss_names), f'{len(loss_names)} names specified, but you have {len(losses)} losses!'
            self.losses_names = loss_names
        else :
            self.losses_names = [None] * len(losses)


        # defining losses
        # TODO: deal with args in losses (in a future version)
        self.losses = torch.nn.ModuleList([
            Registers['LOSSES'][loss](**kwargs) for loss in losses
        ])
        self.weights = loss_weights

        # setting dict names
        self.input_keys = loss_input_keys
        self.target_keys = loss_target_keys

        #
        self.last_state = {}

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))

    @staticmethod
    def add_loss_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Multi-task loss')

        group.add_argument("--losses", type=str, nargs='+',
                           default=['LogCosh', 'LogCosh'])
        group.add_argument("--loss_weights", type=float, nargs='+',
                           default=[0.5, 0.5])
        group.add_argument("--loss_input_keys", type=str, nargs='+',
                           default=['prediction', 'ae_prediction'])
        group.add_argument("--loss_target_keys", type=str, nargs='+',
                           default=['tgt', 'values'])
        group.add_argument("--loss_names", type=str, nargs='+',)
        return parser

    def update_weights(self, weights):
        self.weights = weights

    def _prep_inputs(self, batch):
        """
        Bottle inputs the correct way to be processed by BaseLoss
        """
        return [
            {
                'input_key' : input_key,
                'output_key' : output_key,
                'input1': batch[input_key],
                'input2': batch[output_key]
            }
            for input_key, output_key in zip(self.input_keys, self.target_keys)
        ]



    def forward(self, batch):

        iterable = zip(
            self.losses,
            self.weights,
            self._prep_inputs(batch),
            self.input_keys,
            self.target_keys,
            self.losses_names
        )

        individual_logs = {}

        loss = 0
        for loss_func, weight, supervision, in_key, tgt_key, name in iterable :

            try :
                current_loss = loss_func(supervision, **batch)
                loss += weight * loss_func(supervision, **batch)

                name = f"{loss_func.__class__.__name__}_{name}" if name is not None else \
                    f"{loss_func.__class__.__name__}"

                individual_logs.update({
                   name : current_loss
                })

            except Exception as e :
                raise RuntimeError(f"ERROR : During loss {name} process\nmsg :\n{str(e)}")

        return loss, individual_logs