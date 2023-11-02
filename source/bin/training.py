from argparse import Namespace
from source.utils.register import Registers
from source.utils.misc import store_results
from source.callbacks.factory import CallbacksFactory
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import os

proxy = 'http://192.168.0.100:3128'

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy



def train(args : Namespace) -> int :
    """
    Initialize model, task and dataset in order to train a model
    :param args: Namespace with all the needed arguments
    :return: 0 if successful 1 otherwise
    """

    # overwritting log_every_n_step
    args.log_every_n_steps = 1

    # setting logging to display output
    logging.basicConfig(level=logging.INFO)

    # Init task and dataset
    dataset = Registers["DATASETS"][args.dataset].from_args(args)
    args    = dataset.update_args(args)
    task    = Registers["TASKS"][args.task].from_args(args)

    # building callbacks
    callback_factory = CallbacksFactory(args)
    callbacks = callback_factory.build_callbacks()

    # building logger
    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        version=args.version,
        name = 'lightning_logs'
    )

    # Init trainer
    trainer = Trainer(
        logger=logger,
        default_root_dir=args.default_root_dir,
        gradient_clip_val=args.gradient_clip_val,
        devices=args.gpus,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        strategy='ddp_find_unused_parameters_true',
        callbacks=callbacks
    )

    # launching training
    trainer.fit(model=task, datamodule=dataset)

    # saving model
    task.model.eval()

    # testing
    results = trainer.test(dataloaders = dataset._get_dataloader("test"))

    # storing results
    store_results(results[0], task.logger.log_dir)


    return 0


