from __future__ import annotations

import warnings
from argparse import ArgumentParser, Namespace
from datasets import Dataset
from typing import Callable, Type, Optional, List
import sys
import logging
import pathlib
from source.data.data_modules.base_datamodule import BaseDataModule
from enum import Enum
from torch.utils.data.dataloader import DataLoader
from datasets import Features, Sequence, ClassLabel, Value
from functools import cached_property
from transformers import AutoTokenizer
from source.utils.misc import list_file
import os
from source.utils.HyperParametersManagers import HyperParametersManager
class NERDataModule(BaseDataModule) :
    """
    Base class for NER DataLoading
    """

    def __init__(self,
                 collector_log_dir: pathlib.Path,
                 data_dir: pathlib.Path,
                 file_loader: Callable,
                 label_enum : Type[Enum],
                 file_detector: Callable = list_file,
                 verbose: int = logging.INFO,
                 seed: int = 42,
                 batch_size: int = 32,
                 batch_size_test: int = 10,
                 single_sample_training : bool = False,
                 tokenizer_local_load : Optional[str] = None,
                 tokenizer_key : str = "xlm-roberta-large",
                 gpus : int = None,
                 **kwargs
        ) :
        """

        :param collector_log_dir:
        :param data_dir:
        :param file_loader:
        :param verbose:
        :param seed:
        :param batch_size:
        :param batch_size_test:
        :param single_sample_training:
        :param kwargs:
        """

        super().__init__(**kwargs)

        self.load_from_file = file_loader
        self.label_enum = label_enum
        self.list_file = file_detector
        self.tokenizer_key = tokenizer_key

        # Init logger
        logging.basicConfig(
            filename=collector_log_dir / "logs.log",
            encoding='utf-8',
            level=verbose
        )

        # adding standard output level logging
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info("DataModule init...")


        # checking datadir exist
        self.data_dir = data_dir

        assert 1 <= batch_size, "batch size should be at least 1"
        self.batch_size = batch_size

        assert 1 <= batch_size_test, "batch size test should be at least 1"
        self.batch_size_test = batch_size_test

        self.seed = seed

        self.single_sample = single_sample_training
        if single_sample_training:
            logging.warn("#### SINGLE SAMPLE MODE ENABLED")

        self.data_loaded = False

        self.tokenizer_local_load = tokenizer_local_load

        if gpus is None :
            warnings.warn(f"### WARNING : No GPU is defined, assuming only one is used or CPU only. Setting to 1.")
            gpus = 1
        self.gpus = gpus



    def update_args(self, args : Namespace) :
        return args

    @classmethod
    def from_args(cls, args : Namespace) -> NERDataModule :
        """
        Build Rotowire Task 1 Datamodule from parser Namspace args
        :param args: namespace from main parser
        :return: RotoWireTask1DataModule object
        """
        return cls(**vars(args))

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add Data Module specific args to the main parser
        :param parent_parser: main parser
        :return: updated main parser
        """
        parent_parser = BaseDataModule.add_data_specific_args(parent_parser)

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_dir", type = pathlib.Path, default = None)

        parser.add_argument("--batch_size", type = int, default = 32)
        parser.add_argument("--batch_size_test", type = int, default = 10)

        parser.add_argument("--verbose", action = "store_true", default = False)
        parser.add_argument("--seed", type = int, default = 42)
        parser.add_argument("--single_sample_training", action="store_true", default = False)
        parser.add_argument("--tokenizer_local_load", type = str, default = None)
        parser.add_argument("--tokenizer_key", type = str, default = "xlm-roberta-large")

        return parser


    def prepare_data(self) -> None :
        pass


    def setup(self, stage: Optional[str] = None) -> None :

            # listing text files
            files = self.list_file(self.data_dir)

            # sanity check
            if "train" not in files :
                raise RuntimeError(f"ERROR loading files : Missing \"train\" file. Found {' '.join(files.keys())}"
                                   f". This error might occur when selecting the wrong folder (e.g bc5cdrDataLoader with WNUT17 folder)")
            if "test" not in files :
                raise RuntimeError(f"ERROR loading files : Missing \"test\" file. Found {' '.join(files.keys())}"
                                   f". This error might occur when selecting the wrong folder (e.g bc5cdrDataLoader with WNUT17 folder)")
            if "dev" not in files :
                raise RuntimeError(f"ERROR loading files : Missing \"dev\" file. Found {' '.join(files.keys())}"
                                   f". This error might occur when selecting the wrong folder (e.g bc5cdrDataLoader with WNUT17 folder)")

            # loading from files
            self.sets = {
                "train": self.load_from_file(files['train']),
                "val": self.load_from_file(files['dev']),
                "test": self.load_from_file(files['test'])
            }

            def gen(samples) :
                for sample in samples :
                    yield sample

            # loading from files
            self.sets = {
                "train": self.create_dataLoader(Dataset.from_generator(gen, gen_kwargs={"samples" : self.sets['train']})),
                "val": self.create_dataLoader(Dataset.from_generator(gen, gen_kwargs={"samples" : self.sets['val']})),
                "test": self.create_dataLoader(Dataset.from_generator(gen, gen_kwargs={"samples" : self.sets['test']}))
            }

            # storing info
            HyperParametersManager()['n_training_steps'] = len(self.sets['train']) / self.gpus
            HyperParametersManager()['n_val_steps'] = len(self.sets['val']) / self.gpus
            HyperParametersManager()['n_test_steps'] = len(self.sets['test']) / self.gpus

            self.data_loaded = True

    @cached_property
    def model_tokenizer(self) :

        if self.tokenizer_local_load is None :
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_key,
                use_fast = True,
                add_prefix_space=True,
                model_max_length=512
            )
        else :
            tokenizer = self.load_tokenizer_locally(self.tokenizer_local_load)

        tokenizer.add_tokens(["<EOS>"], special_tokens=True)

        return tokenizer

    def load_tokenizer_locally(self, tokenizer_local_load) :
        """
        Load the tokenizer from local files given the env variable MODEL_FILES is defined.
        :param tokenizer_local_load: subdirectory holding the model files
        :return : the tokenizer
        """

        # sanity check
        if "MODEL_FILES" not in os.environ :
            raise EnvironmentError(f"No $MODEL_FILES env variable have been defined. Make sure to set it "
                                   f"model file location in order to load the tokenizer locally.")

        # creating path
        path_tokenizer = pathlib.Path(os.environ['MODEL_FILES']) / tokenizer_local_load

        if not path_tokenizer.exists() :
            raise FileNotFoundError(f"Directory {path_tokenizer} not found.")

        # loading model
        model = AutoTokenizer.from_pretrained(
            path_tokenizer,
            use_fast = True,
            add_prefix_space = True,
            padding="max_length",
            is_split_into_words=True,
            truncation=True,
            return_tensors="pt",
            model_max_length=512
        )

        return model

    def process_labels(
            self,
            encoding,
            labels_samples,
            ignore_label: int = -100,
    ) -> List:
        """
        Take the tokenized text and associate bounding boxes and label to each sub-token
        :param tokens: list of input sub-tokens
        :param labels: list of label token wise
        :param bboxes: list of bounding boxes token wise
        :param ignore_label: label used in case of special sub-tokens
        :return: tuple formed by the final lanels and bounding box sequence
        """
        res_labels = []

        label2id = {i.name : i.value for i in self.label_enum }

        for index_sample in range(len(encoding['attention_mask'])):

            word_ids = encoding.word_ids(batch_index=index_sample)
            labels = [
                label2id[label_str] for label_str in labels_samples[index_sample]
            ]

            previous_word_idx = None
            current_labels = []

            for word_idx in word_ids:
                if word_idx is None:
                    current_labels.append(ignore_label)

                elif word_idx == previous_word_idx:

                    current_labels.append(ignore_label)

                else:
                    current_labels.append(labels[word_idx])

                previous_word_idx = word_idx

            res_labels.append(current_labels)

        return res_labels

    def process_attention_mask_without_context(self, input_ids, attention_mask, eos_token_id) :

        res_attention_mask = []
        res_token_type_ids = []

        for sequence, attention_mask in zip(input_ids, attention_mask) :

            current_token_type_ids = []
            current_attention_mask = []
            context_found = False

            for id, attention in zip(sequence, attention_mask) :

                # attention
                if attention == 0 or context_found: # if outside attention or in context just ignore
                    current_attention_mask.append(0)
                elif id == eos_token_id : # if EOS found we are in context
                    context_found = True
                    current_attention_mask.append(0)
                else :
                    current_attention_mask.append(1)

                # context type id
                current_token_type_ids.append(int(context_found))


            res_attention_mask.append(current_attention_mask)
            res_token_type_ids.append(current_token_type_ids)

        return res_attention_mask, res_token_type_ids

    def preprocess_data(self, samples) :

        # extracting data
        tokens = samples['text']
        labels = samples['ner_tags']

        # tokenizing data
        encoded_input = self.model_tokenizer(
            text = tokens,
            padding = "max_length",
            is_split_into_words = True,
            truncation = True,
            return_tensors = "pt"
        )

        try :
            # processing labels
            labels = self.process_labels(
                encoded_input,
                labels
            )
        except Exception as e :

            # looking for index
            index = 0
            for i in range(len(labels)) :
                if "Morganization" in labels[i] :
                    index = i

            print(labels[index])
            print(tokens[index])
            exit()

        # processing attention mask without context and token type ids
        eos_token_id = self.model_tokenizer.convert_tokens_to_ids('<EOS>')
        attention_mask_without_context, token_type_ids = self.process_attention_mask_without_context(
            encoded_input['input_ids'],
            encoded_input['attention_mask'],
            self.model_tokenizer.convert_tokens_to_ids('<EOS>')
        )

        # replacing <EOS> by separator tokens
        input_ids = encoded_input['input_ids']
        input_ids[input_ids == eos_token_id] = self.model_tokenizer.convert_tokens_to_ids(
            self.model_tokenizer.sep_token
        )


        # storing info
        encoded_input['labels'] = labels
        encoded_input['token_type_ids'] = token_type_ids
        encoded_input['attention_mask_without_context'] = attention_mask_without_context
        encoded_input['token_type_ids'] = token_type_ids

        return encoded_input

    def create_dataLoader(self, raw_samples) -> DataLoader :
        """
        Create a dataloader for the given samples
        :param data: sample to to wrap in the dataLoader
        :param name: name of the dataset
        :param weights: weight used for loss
        :Return: Dataloader
        """

        # we need to define custom features
        features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'attention_mask_without_context': Sequence(Value(dtype='int64')),
            'labels': Sequence(ClassLabel(names=[self.label_enum(i).name for i in range(len(self.label_enum) - 1)])),
            'token_type_ids': Sequence(feature=Value(dtype='int64')),

        })

        # pre processing data
        datasets = raw_samples.map(
            self.preprocess_data,
            batched=True,
            features=features,
            remove_columns=['text', 'ner_tags']
        )

        datasets.set_format(type="torch")

        dataLoader = DataLoader(
            datasets,
            batch_size=self.batch_size,
            shuffle=True
        )

        return dataLoader

    def _get_dataloader(self, split='train') -> Type[DataLoader]:

        if split not in self.sets :
            raise ValueError(f"Unavailable split {split}. Available {' '.join(self.sets.keys())}")

        return self.sets[split]









