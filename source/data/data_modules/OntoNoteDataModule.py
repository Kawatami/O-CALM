from __future__ import annotations
from argparse import ArgumentParser, Namespace
from datasets import Dataset
from source.utils.register import register
from typing import *
import sys
import logging
import pathlib
from source.data.data_modules.base_datamodule import BaseDataModule
from enum import Enum
from torch.utils.data.dataloader import DataLoader
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, load_dataset
from functools import cached_property
from transformers import AutoTokenizer, AutoModel


@register("DATASETS")
class OntoNoteDataModule(BaseDataModule) :

    def __init__(self,
                 collector_log_dir: pathlib.Path,
                 verbose: int = logging.INFO,
                 seed: int = 42,
                 batch_size: int = 32,
                 batch_size_test: int = 10,
                 single_sample_training : bool = False,
                 **kwargs) :

        super().__init__(**kwargs)

        # Init logger
        logging.basicConfig(
            filename=collector_log_dir / "logs.log",
            encoding='utf-8',
            level=verbose
        )

        # adding standard output level logging
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info("DataModule init...")

        assert 1 <= batch_size, "batch size should be at least 1"
        self.batch_size = batch_size

        assert 1 <= batch_size_test, "batch size test should be at least 1"
        self.batch_size_test = batch_size_test

        self.seed = seed

        self.single_sample = single_sample_training
        if single_sample_training:
            logging.warn("#### SINGLE SAMPLE MODE ENABLED")

        self.sets = None



    def update_args(self, args : Namespace) :

        return args

    @classmethod
    def from_args(cls, args : Namespace) -> OntoNoteDataModule :
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
        group = parser.add_argument_group('OntoNote DataModule')


        parser.add_argument("--batch_size", type = int, default = 32)
        parser.add_argument("--batch_size_test", type = int, default = 10)

        parser.add_argument("--verbose", action = "store_true", default = False)
        parser.add_argument("--seed", type = int, default = 42)
        parser.add_argument("--single_sample_training", action="store_true", default = False)

        return parser




    def prepare_data(self) -> None :

        if self.sets is None :

            self.data = load_dataset("conll2012_ontonotesv5", "english_v12")

            labels = self.data['train'].features['sentences'][0]['named_entities'].feature.names

            self.id2label = {v: k for v, k in enumerate(labels)}
            self.label2id = {k:v for v, k in enumerate(labels)}

            self.sets = {
                "train": self.data['train'],
                "val": self.data['validation'],
                "test": self.data['test']

            }

            self.data_loaded = True

    def setup(self, stage: Optional[str] = None) -> None:

        if self.data_loaded :

            # loading from files
            self.sets = {
                "train": self.create_dataLoader(self.data['train']),
                "val": self.create_dataLoader(self.data['train']),
                "test": self.create_dataLoader(self.data['train'])
            }

    @cached_property
    def model_tokenizer(self) :
        return AutoTokenizer.from_pretrained(
            "xlm-roberta-large",
            use_fast = True,
            add_prefix_space=True,
            #additional_special_tokens = ["<EOS>"]
        )

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


        for index_sample in range(len(encoding['attention_mask'])):

            word_ids = encoding.word_ids(batch_index=index_sample)
            labels = [label for label in labels_samples[index_sample]]

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
        tokens = [sample[0]['words'] for sample in samples['sentences']]
        labels = [sample[0]['named_entities'] for sample in samples['sentences']]

        # tokenizing data
        encoded_input = self.model_tokenizer(
            text = tokens,
            padding = "longest",
            is_split_into_words = True,
            truncation=True
        )

        # processing labels
        labels = self.process_labels(
            encoded_input,
            labels
        )

        # processing attention mask without context and token type ids
        attention_mask_without_context, token_type_ids = self.process_attention_mask_without_context(
            encoded_input['input_ids'],
            encoded_input['attention_mask'],
            self.model_tokenizer.convert_tokens_to_ids('<EOS>')
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
            'labels': Sequence(ClassLabel(names=list(self.label2id.values()))),
            'token_type_ids': Sequence(feature=Value(dtype='int64')),

        })

        # pre processing data
        datasets = raw_samples.map(self.preprocess_data, batched=True,
                                   features=features, remove_columns=['document_id', 'sentences'])

        datasets.set_format(type="torch")

        dataLoader = DataLoader(
            datasets,
            batch_size=self.batch_size,
            shuffle=True
        )

        return dataLoader

    def _get_dataloader(self, split='train') -> Type[DataLoader]:

        return self.sets[split] if split in self.sets else []








