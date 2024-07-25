from __future__ import annotations

import pathlib
from argparse import ArgumentParser, Namespace
from source.utils.register import register
from enum import Enum
from source.utils.misc import load_from_file, list_file
from source.data.data_modules.NERDataModule import NERDataModule
from typing import Optional, List, Dict
from datasets import Dataset
from source.utils.HyperParametersManagers import HyperParametersManager
from tqdm import tqdm

# defining WNUT17 labels
class WNUT17_label(Enum):
    B_corporation = 0
    B_creative_work = 1
    B_group = 2
    B_location = 3
    B_person = 4
    B_product = 5
    I_corporation = 6
    I_creative_work = 7
    I_group = 8
    I_location = 9
    I_person = 10
    I_product = 11
    O = 12
    B_X = -100 # always ignore context token in loss processing



@register("DATASETS")
class WNUT17DataModule(NERDataModule) :

    def __init__(self, **kwargs) :

        super().__init__(
            file_loader = load_from_file,
            label_enum = WNUT17_label,
            **kwargs
        )


    @classmethod
    def from_args(cls, args : Namespace) -> WNUT17DataModule :
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
        parser = NERDataModule.add_data_specific_args(parent_parser)

        return parser


@register("DATASETS")
class WNUT17FullDataModule(WNUT17DataModule) :

    _merging_types : List[str] = [
        "rowWise",
        "sampleWise"
    ]

    def __init__(
            self,
            data_directories : List[str],
            merging_type : str = "rowWise",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.data_directories = data_directories

        self.merging_type = merging_type

        self.files = {
            "train": [],
            "val": [],
            "test": []
        }

        self.sets = {
            "train": [],
            "val": [],
            "test": []
        }

    def load_data(self, files: List[pathlib.Path]) :
        """
        Load data from a list of file and returns a single list of sample
        """
        if self.merging_type not in WNUT17FullDataModule._merging_types :
            raise ValueError(f"Unsupported merging type \"{self.merging_type}\". Avalaible {', '.join(WNUT17FullDataModule._merging_types)}")
        else :
            datas = [self.load_from_file(path) for path in tqdm(files, total=len(files), leave=False, desc="## LOADING FILES")]

        if self.merging_type == "rowWise" :
            return sum(datas, [])
        elif self.merging_type == "sampleWise" :

            def process_sample(samples) :



                # getting input text
                input = samples[0]['text'][:samples[0]['text'].index("<EOS>")]

                # retreive context
                contexts = [
                    sample['text'][sample['text'].index("<EOS>"):] for sample in samples
                ]

                new_text =  input + sum(contexts, [])



                output_sample = {
                    "text" : new_text,
                    "ner_tags" : samples[0]['ner_tags'] + ["B_X"] * (len(new_text) - len(samples[0]['ner_tags']))
                }

                # for k, v in output_sample.items() :
                #     print(f"{k} : {len(v)}")
                #
                # exit()

                return output_sample

            output_datas = [ process_sample(samples) for samples in zip(*datas) ]

            return output_datas

        else:
            raise NotImplementedError(f"merging type {self.merging_type} not Implemented")

    def prepare_data(self) -> None :

        # listing files
        for directory in self.data_directories:

            # creating path
            path = self.data_dir / directory

            # listing file
            current_files = path.rglob("*")

            # filtering file
            current_files = [file for file in current_files if file.is_file() and file.suffix == ".txt"]

            if current_files == []:
                raise ValueError(f"No file has been found at \n\t{path}\n Have you set data_dir correctly ?")

            # storing info
            for file in current_files:
                # checking name
                set = file.stem.split(".", maxsplit=1)[0]

                self.files[set if set != "dev" else "val"].append(file)

    def setup(self, stage: Optional[str] = None) -> None :


        if stage == "fit" :

            self.sets['train'] = self.create_dataLoader(Dataset.from_list(self.load_data(self.files['train'])))
            self.sets['val'] = self.create_dataLoader(Dataset.from_list(self.load_data(self.files['val'])))

            # storing info
            HyperParametersManager()['n_training_steps'] = len(self.sets['train']) / self.gpus
            HyperParametersManager()['n_val_steps'] = len(self.sets['val']) / self.gpus
            HyperParametersManager()['n_test_steps'] = len(self.sets['test']) / self.gpus

        elif stage == "validate" :
            self.sets['val'] = self.create_dataLoader(Dataset.from_list(self.load_data(self.files['val'])))
        elif stage == 'test' :
            self.sets['test'] =  self.create_dataLoader(Dataset.from_list(self.load_from_file(self.files['test'][0])))


        # loading data
        # self.sets = {
        #     "train": self.load_data(self.sets['train']),
        #     "val": self.load_data(self.sets['val']),
        #     "test": self.load_from_file(self.sets['test'][0])
        # }

        # loading from files
        # self.sets = {
        #      k : self.create_dataLoader(Dataset.from_list(self.sets[k])) for k in self.sets.keys()
        # }



    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add Data Module specific args to the main parser
        :param parent_parser: main parser
        :return: updated main parser
        """
        parser = WNUT17DataModule.add_data_specific_args(parent_parser)

        parser.add_argument("--data_directories", type = str, nargs = "+")
        parser.add_argument("--merging_type", type = str, default = "rawWise")
        return parser


@register("DATASETS")
class WNUT17UnstructuredDataModule(WNUT17DataModule) :


    def __init__(
            self,
            data_directories : List[str],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.data_directories = data_directories


        self.sets = {
            "train": [],
            "val": [],
            "test": []
        }

    def extract(self, input, tags) :

        entities_res = []

        for index in range(len(input)) :


            if tags[index] == "B_X" :
                break

            # detecting entity
            if tags[index][0] == 'B' :

                entity_text = ""
                category = tags[index][2:]

                while tags[index][2:] == category :
                    entity_text += f" {input[index]}"
                    index += 1

                entities_res.append((entity_text, category))

        return entities_res

    def format_entities(self, entities) :

        return sum([[entity, ":", category, ","] for entity, category in entities], [])


    def create_context_from_labels(self, samples) :
        """

        """

        for sample in samples :


            # extracting input
            input = sample['text'][:sample['text'].index("<EOS>")]
            labels = sample['ner_tags'][:sample['text'].index("<EOS>")]

            # extracting entities
            entities = self.extract(input, sample['ner_tags'])

            # formating entities
            entites_context = self.format_entities(entities)


            # update data
            sample['text'] = input + ['<EOS>'] + entites_context
            sample['ner_tags'] = labels + ["B_X"] * (len(entites_context) + 1)

        return samples


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

            # changing context
            self.sets = {
                "train": self.create_context_from_labels(self.sets['train']),
                "val": self.create_context_from_labels(self.sets['val']),
                "test": self.create_context_from_labels(self.sets['test'])
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


    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add Data Module specific args to the main parser
        :param parent_parser: main parser
        :return: updated main parser
        """
        parser = WNUT17DataModule.add_data_specific_args(parent_parser)

        parser.add_argument("--data_directories", type = str, nargs = "+")
        parser.add_argument("--merging_type", type = str, default = "rawWise")
        return parser




