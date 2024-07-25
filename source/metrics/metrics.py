import torch
from source.metrics.base_metric import BaseMetric
from source.utils.register import register
from typing import *
from argparse import ArgumentParser
from seqeval.metrics import classification_report as classification_report_seqeval
from seqeval.scheme import IOB2, IOB1
from typing import Type
from enum import Enum
from source.data.data_modules.WNUT17DataModule import WNUT17_label
from source.data.data_modules.bc5cdrDataModule import BC5CDR_label
from source.data.data_modules.CNLLPPDataModule import CNLLPP_label
from sklearn.metrics._classification import classification_report

@register('METRICS')
class Accuracy(BaseMetric):
    """
    Process Accuracy
    """
    _names = ['Accuracy', 'Acc']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, acc_threshold : float, **kwargs):
        super().__init__(**kwargs)

        assert acc_threshold > 0.0
        self.threshold = acc_threshold

        self.add_state("correct_prediction_count", default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state("total_prediction_count", default=torch.zeros(1), dist_reduce_fx='sum')


    def update(self, batch : dict) -> None:
        prd, tgt = self._prep_inputs(batch)

        # converting logits to label
        prd = prd.argmax(dim=-1)

        # creating mask for token selection
        mask = tgt != -100

        # selecting tokens
        prd = prd.masked_select(mask).detach()
        tgt = tgt.masked_select(mask).detach()

        # counting correct predictions
        correct_predictions = torch.sum((prd == tgt).int())
        total_prediction = tgt.shape[0]

        # updating state
        self.correct_prediction_count += correct_predictions
        self.total_prediction_count += total_prediction

    def compute(self) -> Any :
        return (self.correct_prediction_count / self.total_prediction_count).item()

    @staticmethod
    def add_metric_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Accuracy metric')
        group.add_argument("--acc_threshold", type=float, default=0.5)

        return parser

    def _prep_inputs(self, model_outputs):
        prd, tgt = model_outputs['prediction'] , model_outputs['labels']


        # getting number of classes
        n_classes = model_outputs['prediction'].shape[-1]

        # reshaping
        prd, tgt = prd.view(-1, n_classes), tgt.view(-1)


        return prd, tgt

@register('METRICS')
class ClassificationReport(BaseMetric):
    """
    Process Accuracy
    """
    _names = ['ClassificationReport']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state(f"prediction", default=torch.Tensor([]), dist_reduce_fx='cat')
        self.add_state(f"target", default=torch.Tensor([]), dist_reduce_fx='cat')


    def update(self, batch : dict) -> None:

        self.prediction = torch.cat((self.prediction, batch['prediction'].argmax(dim=-1).detach()), dim=0)
        self.target = torch.cat((self.target, batch['labels'].detach()), dim=0)

    def compute(self) -> Any :

        # getting info
        prediction = self.prediction.view(-1).cpu().numpy()
        labels = self.target.view(-1).cpu().numpy()

        # processing classification report
        output = classification_report(
            y_true=labels,
            y_pred=prediction,
            output_dict=True,
            zero_division=1.0
        )

        # formating dict
        output_dict = {}
        for k, v in output.items() :

            if isinstance(v, dict) :
                for sub_k, sub_v in v.items() :
                    output_dict["SELECTOR_" + k + "_" + sub_k] = sub_v
            else :
                output_dict[k] = v

        return [(k, v) for k, v in output_dict.items()]


    @staticmethod
    def add_metric_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Accuracy metric')
        group.add_argument("--acc_threshold", type=float, default=0.5)

        return parser

    def _prep_inputs(self, model_outputs):
        prd, tgt = model_outputs['prediction'] , model_outputs['labels']


        # getting number of classes
        n_classes = model_outputs['prediction'].shape[-1]

        # reshaping
        prd, tgt = prd.view(-1, n_classes), tgt.view(-1)


        return prd, tgt

class BaseSeqEval(BaseMetric):

    _names = ['SeqEval']

    def __init__(self, enum_label : Type[Enum], **kwargs):
        super().__init__(**kwargs)

        self.epsilone = 1e-6

        # adding stat for entity
        self.add_state(f"prediction_without_context", default=torch.Tensor([]), dist_reduce_fx='cat')
        self.add_state(f"prediction_with_context", default=torch.Tensor([]), dist_reduce_fx='cat')
        self.add_state(f"target", default=torch.Tensor([]), dist_reduce_fx='cat')


        self.id2label = {i.value : i.name for i in enum_label if i.value != -100}

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__


    def update(self, batch : dict) -> None :
        """
        update metric
        """

        self.prediction_with_context = torch.cat((self.prediction_with_context, batch['prediction_with_context'].detach()), dim = 0)
        self.prediction_without_context = torch.cat((self.prediction_without_context, batch['prediction_without_context'].detach()), dim = 0)

        self.target = torch.cat((self.target, batch['labels'].detach()), dim = 0)


    def convert_to_labels(self, prediction : torch.tensor, target : torch.tensor) \
            -> Tuple[List[List[str]], List[List[str]]] :
        """
        Convert prediction and target sequence to their list[str] equivalent
        :param prediction:
        :param target:
        :return:
        """

        def convert(sequences, labels) :
            sequence = [
                [self.id2label[p] for (p, l) in zip(sequence, label) if l != -100]
                for sequence, label in zip(sequences, labels)
            ]

            return sequence

        if len(prediction.shape) == 3 : # n_gpu x batch_size x seq_len
            _, _, seq_len = prediction.shape
            prediction = prediction.view(-1, seq_len)
            target = target.view(-1, seq_len)

        prediction = prediction.tolist()
        target = target.tolist()



        prediction = convert(prediction, labels=target)
        target = convert(target, labels=target)

        return prediction, target

    def format_dict(self, results : Dict, prefix = "") -> List[Tuple[str, float]] :
        """
        Reformat the input dictionary to comply with pytorch-lightning ing log function
        :param results:
        :return:
        """

        final_results = {}

        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    name = f"{prefix}{key}_{n}".replace(" ", "_")
                    final_results[name] = v
            else:
                final_results[key] = value

        return list(final_results.items())


    def compute(self) -> List[Tuple[str, float]]:
        """
        Process metric final result
        """


        prediction_with_context = self.prediction_with_context#.argmax(dim=-1)
        prediction_without_context = self.prediction_without_context#.argmax(dim=-1)

        # prep input
        prediction_with_context, target = self.convert_to_labels(prediction_with_context, self.target)
        prediction_without_context, target = self.convert_to_labels(prediction_without_context, self.target)

        res_with_context = classification_report_seqeval(
            y_true=target,
            y_pred=prediction_with_context,
            output_dict=True,
            scheme=IOB2,
            mode="strict"
        )

        res_without_context = classification_report_seqeval(
            y_true=target,
            y_pred=prediction_without_context,
            output_dict=True,
            scheme=IOB2,
            mode="strict"
        )

        res_with_context = self.format_dict(res_with_context, prefix="CONTEXT_")
        res_without_context = self.format_dict(res_without_context, prefix="NO_CONTEXT_")

        return res_with_context + res_without_context

    def _prep_inputs(self, model_outputs):
        prd, tgt = model_outputs['prediction'], model_outputs['labels']


        # getting number of classes
        n_classes = model_outputs['prediction'].shape[-1]

        # reshaping
        prd = prd.view(-1, n_classes)
        tgt = tgt.view(-1)

        return prd, tgt

    @staticmethod
    def add_metric_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        return parser

@register('METRICS')
class SeqEvalWNUT17(BaseSeqEval):

    _names = ['SeqEvalWNUT17']

    def __init__(self, **kwargs):
        super().__init__(WNUT17_label, **kwargs)

@register('METRICS')
class SeqEvalBC5CDR(BaseSeqEval):

    _names = ['SeqEvalbc5cdr']

    def __init__(self, **kwargs):
        super().__init__(BC5CDR_label, **kwargs)

@register('METRICS')
class SeqEvalCNLLPP(BaseSeqEval):
    _names = ['SeqEvalbc5cdr']

    def __init__(self, **kwargs):
        super().__init__(CNLLPP_label, **kwargs)

