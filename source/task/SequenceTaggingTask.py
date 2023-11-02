from source.task.base_task import BaseTask
from source.utils.register import register
from argparse import ArgumentParser
from typing import List

@register("TASKS")
class SequenceTaggingTask(BaseTask) :

    def __init__(self, lr : float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr


    @staticmethod
    def add_task_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('SequenceTaggingTask')

        group.add_argument("--lr", type=float, default=1e-5)

        # adding base class argument
        parser = BaseTask.add_task_specific_args(parser)

        return parser