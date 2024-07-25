import argparse
import logging
import pathlib
from argparse import ArgumentParser, Namespace
from typing import List, Any, Optional, Dict
from source.utils.register import Registers
import pytorch_lightning as pl
import torch


class BaseTask(pl.LightningModule):
    def __init__(self,
                 train_set_names: Optional[List[str]] = None,
                 val_set_names: Optional[List[str]] = None,
                 test_set_names: Optional[List[str]] = None,
                 lr_scheduler_param : dict = None,
                 warm_up_step : int = 0,
                 warm_up_rate : float = 0.1,
                 lr_decay_rate : float = 0.95,
                 **kwargs):
        super().__init__()
        args = Namespace(**kwargs)


        # optimizer param
        self.warm_up_step = warm_up_step
        self.warm_up_rate = warm_up_rate
        self.lr_decay_rate = lr_decay_rate


        # datasets names
        self.train_set_names : Optional[List[str]] = train_set_names
        self.val_set_names : Optional[List[str]] = val_set_names
        self.test_set_names : Optional[List[str]] = test_set_names

        # building model and loss
        self.model = self.build_model_from_args(args)

        # storing model hparams
        if hasattr(self.model, "export_hparams") :
            self.model.export_hparams(self.hparams)
            self.save_hyperparameters()

        # defining loss
        self.loss, self._loss_name = self.build_loss_from_args(args)

        self.lr_scheduler_param = lr_scheduler_param

        # storing metrics
        self.metrics = {
            'trainset': list(),
            'valset': list(),
            'testset': list()
        }

        self.build_metrics(args)

        self.counter_step = 0

        self.args = kwargs

    @staticmethod
    def add_task_specific_args(parent_parser):


        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('SequenceTaggingTask')

        group.add_argument("--dataset", type=str, default='RotoWireDataModule',
                           choices=Registers['DATASETS'].keys())

        group.add_argument("--warm_up_step", type = int, default = 0)
        group.add_argument("--warm_up_rate", type = float, default = 0.1)
        group.add_argument("--lr_decay_rate", type = float, default = 1.0)



        return parser

    def get_model_sample_creation_method(self, dataset) :
        return self.model.get_sample_creation_method(dataset)

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(**vars(args))

    @staticmethod
    def build_model_from_args(args):
        return Registers['MODELS'][args.model].from_args(args)

    @staticmethod
    def build_loss_from_args(args):
        loss = Registers['LOSSES'][args.loss].from_args(args)
        return loss, loss.name

    @staticmethod
    def build_metrics_from_args(args : Namespace, set : str) -> torch.nn.ModuleList:
        metrics = []
        # for every metric in the parser namespace in the form (name of the metric, class of metric, set name)
        for name, metric_name, set_name in args.metrics :

            # if set name correpond to the current set
            if set_name == "all" or set_name == set :
                args.log_name = name
                metric = Registers["METRICS"][metric_name].from_args(args)
                metrics.append(metric)

        return torch.nn.ModuleList(metrics)

    def model_gradient_norm(self):
        accumulator = 0
        for name, param in self.model.named_parameters() :

            if param.grad is not None :
                accumulator += (param.grad.detach().norm(2) ** 2).item()

        return accumulator ** (1/2)


    def GPU_memory_usage(self):
        return torch.cuda.memory_allocated() / (1e6)


    def build_metrics(self, args : argparse.Namespace) :

        def None2one(set) :
            return len(set) if set is not None else 1

        num_sets = [
            None2one(self.train_set_names),
            None2one(self.val_set_names),
            None2one(self.test_set_names)
        ]

        # storing metrics
        self.metrics = torch.nn.ModuleDict({
            'trainset': torch.nn.ModuleList(),
            'valset': torch.nn.ModuleList(),
            'testset': torch.nn.ModuleList()
        })


        # iterating over the sets
        for dataset, num_set in zip(self.metrics.keys(), num_sets) :

            # for every sub set in a set (n test sets for exemple)
            for i in range(num_set) :

                # build the list of metrics
                metric_list = self.build_metrics_from_args(args, dataset)

                # add to the list of metrics
                self.metrics[dataset].append(metric_list)


    def get_memory_count(self, msg) :
        with pathlib.Path("./memory_logs.csv").open("a+") as file :
            file.write(f"\n{msg},{torch.cuda.memory_allocated() / (1e6)}")

    def step(self, batch : dict, subset : str, name : str = None, metrics : Optional[List] = None) :

        # processing model output
        batch = self.model(batch)  # prediction

        loss = self.loss(batch)  # loss compute

        log = {}

        name = name if name is not None else subset

        # handling multi task loss case
        if isinstance(loss, tuple):
            log.update({f"{key}/{name}": value.detach().cpu().item() for key, value in loss[1].items()})
            loss = loss[0]

        # log loss
        log.update({f'{self._loss_name}/{name}': loss.detach().cpu().item()})


        # log mode gradient and memory usage
        if name == "trainset" :
            log.update({f"model_gradient/{name}" : self.model_gradient_norm()})
            log.update({f"memory_usage" : self.GPU_memory_usage()})

        # update metrics
        if metrics is not None :
            for metric in metrics :
                metric.update(batch)


        # free memory from gpu
        self.transfer_to_cpu(batch)

        torch.cuda.empty_cache()

        return loss, log, None, batch

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        name = self._get_dataset_name(self.train_set_names, dataloader_idx)
        idx = 0 if dataloader_idx is None else dataloader_idx
        metrics = self.metrics['trainset'][idx]

        loss, log, _, _  = self.step(batch, 'trainset', name=name, metrics=metrics)

        self.log_dict(log, prog_bar=True, sync_dist=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        name = self._get_dataset_name(self.val_set_names, dataloader_idx)
        idx = 0 if dataloader_idx is None else dataloader_idx

        metrics = self.metrics['valset'][idx]
        #with torch.no_grad() :
        loss, log, _, _  = self.step(batch, 'valset', name=name, metrics=metrics)

        self.log_dict(log, prog_bar=True, sync_dist=True)

        return { "loss" : loss.detach().cpu().item() }

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        # switching to test mode
        if hasattr(self.model, "test_mode") :
            self.model.test_mode = True

        name = self._get_dataset_name(self.test_set_names, dataloader_idx)

        metrics = self.metrics['testset'][dataloader_idx if dataloader_idx is not None else 0]
        loss, log, outputs, loss_sample = self.step(batch, 'testset', name=name, metrics=metrics)

        self.log_dict(log, prog_bar=True, sync_dist=True)


        return outputs

    def transfer_to_cpu(self, sample : dict) -> None :
        """
        Transfer tensor to CPU.
        :param sample: batch as dictionary
        """
        for key, value in sample.items() :
            if isinstance(value, torch.Tensor) :
                sample[key] = value.detach().cpu()
                del value

    def _get_dataset_name(self, names, dataloader_idx) :

        if names is None or dataloader_idx is None :
            return None

        if dataloader_idx >= len(names) :
            return None

        return names[dataloader_idx]

    def epoch_end(self, subset : str, names_dataLoader : Optional[List[str]]) -> None :
        """
        Log the metrics. Metrics can either return the result or a list of results of the form [(name, result)]
        :param subset: subset to select metrics
        """

        def check_tensor_to_detach(input : Any) -> Any :
            """
            if the input data is a tensor then it is detached fron the computational graph else do nothing.
            Memory saving sub-routine
            :param input:
            :return:
            """
            if isinstance(input, torch.Tensor) :
                input = input.detach().cpu().item()
            return input

        with torch.no_grad() :

            # if subset admit metric
            if self.metrics[subset] :
                log = {}

                # iterate over metrics
                for set_idx, metric_list in enumerate(self.metrics[subset]) :

                    name_dataLoader = n if (n := self._get_dataset_name(names_dataLoader, set_idx)) else subset

                    for metric in metric_list :


                        try :
                            # compute results
                            metric_result = metric.compute()
                        except Exception as e :
                            logging.error(f"ERROR while processing metric {metric.__class__.__name__}\nmsg :\n{str(e)}")
                            raise e

                        # if metric result is a list
                        if isinstance(metric_result, list) :
                            # log individually
                            for name, result in metric_result :
                                log[f'{name}/{name_dataLoader}'] = check_tensor_to_detach(result)
                        else :
                            log[f'{metric.name}/{name_dataLoader}'] = check_tensor_to_detach(metric_result)

                        # reset metric
                        metric.reset()

                self.log_dict(log, prog_bar=True, sync_dist=True)

    def on_train_epoch_end(self) -> None :
        self.epoch_end('trainset', self.train_set_names)

    def on_validation_epoch_end(self) -> None:
        self.epoch_end('valset', self.val_set_names)

    def on_test_epoch_end(self) -> None:
        self.epoch_end('testset', self.test_set_names)

    def get_progress_bar_dict(self):
        return dict()

    def lr_sechdule(self, epoch) :
        if epoch < self.warm_up_step:
            # warm up lr
            lr_scale = self.warm_up_rate ** (self.warm_up_step - epoch)
        else:
            lr_scale = self.lr_decay_rate ** epoch

        return lr_scale

    def configure_optimizers(self):

        optimizers, schedulers = self.model.get_optimizer_config(self.lr, **self.args)
        if schedulers is not None :
            return optimizers, schedulers
        else :
            return optimizers

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(*args)

