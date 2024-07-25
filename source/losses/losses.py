import torch
from source.losses.base_loss import BaseLoss
from source.utils.register import register
from argparse import ArgumentParser
from torch.nn import KLDivLoss, BCELoss
import torch.nn.functional as F
from argparse import Namespace

@register('LOSSES')
class CRFLoss(BaseLoss):
    _names = ['CRF']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, model_output, **kwargs) :
        return model_output["input1"]

@register('LOSSES')
@register('METRICS')
class CrossEntropyLoss(BaseLoss):
    _names = ['CrossEntropy']


    def __init__(self,
                 class_weights: torch.Tensor = None,
                 log_name : str = None,
                 label_smoothing : float = 0.0,
                 **kwargs):


        super().__init__(log_name=log_name)
        self.criterion = torch.nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=-100,
            weight=class_weights,
            label_smoothing=label_smoothing
        )


    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    @property
    def abbrv(self):
        return 'xent'

    def __call__(self, batch, **kwargs):

        prd, tgt = self._prep_inputs(batch)

        print("======")
        print(prd)
        print("~~")
        print(tgt)

        loss = self.criterion(prd, tgt,)

        return loss

    @classmethod
    def from_args(cls, args : Namespace) -> BaseLoss:
        """
        Build Loss object from Namespace issued by main parser
        :param args:
        :return:
        """

        return cls(**vars(args))

    @staticmethod
    def _prep_inputs(batch):
        prd, tgt = batch['prediction'], batch['labels']


        n_classes = prd.shape[-1]

        if tgt.type() != torch.LongTensor :
            tgt = tgt.long()
        return prd.view(-1, n_classes), tgt.view(-1)

    @staticmethod
    def add_loss_specific_args(parent_parser):

        # calling baseloss arguments
        parser = BaseLoss.add_loss_specific_args(parent_parser)

        parser = ArgumentParser(parents=[parser], add_help=False)
        group = parser.add_argument_group('Cross Entropy loss')

        group.add_argument("--label_smoothing", type=float, default=0.0)

        return parser

@register('LOSSES')
@register('METRICS')
class BinaryCrossEntropy(BaseLoss):
    _names = ['CrossEntropy']


    def __init__(self,
                 class_weights: torch.Tensor = None,
                 log_name : str = None,
                 label_smoothing : float = 0.0,
                 **kwargs):


        super().__init__(log_name=log_name)
        self.criterion = torch.nn.BCEWithLogitsLoss(
            reduction="none",
            weight=class_weights,
        )


    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    @property
    def abbrv(self):
        return 'xent'

    def __call__(self, batch, **kwargs):

        # getting data
        prd, tgt = self._prep_inputs(batch)

        # preparing mask
        mask = (tgt != -100).float()

        # replacing ignore index
        tgt[tgt == -100] = 0

        # processing loss
        loss = self.criterion(prd, tgt,)

        # processing finale loss
        loss = (loss * mask).mean()

        return loss

    @classmethod
    def from_args(cls, args : Namespace) -> BaseLoss:
        """
        Build Loss object from Namespace issued by main parser
        :param args:
        :return:
        """

        return cls(**vars(args))

    @staticmethod
    def _prep_inputs(batch):
        prd, tgt = batch['input1'], batch['input2']

        return prd.view(-1), tgt.view(-1).float()

    @staticmethod
    def add_loss_specific_args(parent_parser):

        # calling baseloss arguments
        parser = BaseLoss.add_loss_specific_args(parent_parser)

        parser = ArgumentParser(parents=[parser], add_help=False)
        group = parser.add_argument_group('Cross Entropy loss')

        group.add_argument("--label_smoothing", type=float, default=0.0)

        return parser

@register("LOSSES")
class L2(BaseLoss) :

    _names = ['L2']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, batch, **kwargs) :

        prd, target = batch['input1'], batch['input2']

        return (prd - target).norm(p=2, dim = -1).sum(dim=-1).mean()

@register("LOSSES")
class KLDivergence(KLDivLoss) :

    _names = ['KLDivergence']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(reduction="none", log_target=True)

        self.temperature = 0.3

    def __call__(self, batch, apply_mask = False, **kwargs) :

        input1, input2 = batch['input1'], batch['input2']
        attention_mask_without_context = kwargs['attention_mask_without_context'].unsqueeze(-1)

        # applying mask
        if apply_mask :
            input1 *= attention_mask_without_context
            input2 *= attention_mask_without_context

        input1 = F.log_softmax(input1 / self.temperature, dim = -1)
        input2 = F.log_softmax(input2 / self.temperature, dim = -1)

        # processing point wise loss
        point_wise_loss = F.kl_div(
            input1,
            input2,
            reduction="none",
            log_target=True
        )

        # applying mask
        point_wise_loss = point_wise_loss * attention_mask_without_context

        # reduce
        point_wise_loss =  (point_wise_loss * (self.temperature ** 2)).sum() / attention_mask_without_context.shape[0]

        return point_wise_loss

