# Test ontonote import:
import argparse, pathlib

from source import CNLLPPDataModule

source_path = pathlib.Path(__file__).parents[1]
import sys
sys.path.append(str(source_path))
import numpy as np
from source.utils.misc import load_from_file_cnllpp
from source.data.data_modules.CNLL03DataModule import CNLL03DataModule
from colorama import Fore, Back, Style
from tqdm import tqdm


def pretty_print(tokenizer, labels, input_ids) :

    for token_id, label in zip(input_ids, labels) :

        token = tokenizer.convert_ids_to_tokens(token_id)

        if label == -100 :
            token = f"{Fore.RED}{token}{Style.RESET_ALL}"
        elif label != 9 :
            token = f"{Fore.GREEN}{token}{Style.RESET_ALL}"


        print(f"{token} ", end = "")
    print()

def pretty_print_mask(tokenizer, labels, input_ids) :

    for token_id, label in zip(input_ids, labels) :

        token = tokenizer.convert_ids_to_tokens(token_id)

        if label == 0 :
            token = f"{Fore.RED}{token}{Style.RESET_ALL}"



        print(f"{token} ", end = "")
    print()

def conll03_test():

    path_data = pathlib.Path(
        "./data/WNUT17/CLNER_datasets/conll_03_english")

    collector_log_dir = pathlib.Path("/tmp")


    loader = CNLL03DataModule(
        collector_log_dir=collector_log_dir,
        data_dir=path_data,
        batch_size=1
    )

    # preparing data
    loader.prepare_data()

    # seting up
    loader.setup()

    # get generator
    test_generator = loader._get_dataloader("val")

    withou_context = 0

    for i, sample in tqdm(enumerate(test_generator)):

        if i < 3:
            continue

        print(sample['labels'][0].tolist())

        pretty_print(loader.model_tokenizer, sample['labels'][0].tolist(), sample['input_ids'][0].tolist())
        pretty_print_mask(loader.model_tokenizer, sample['attention_mask'][0].tolist(), sample['input_ids'][0].tolist())
        pretty_print_mask(loader.model_tokenizer, sample['attention_mask_without_context'][0].tolist(),
                          sample['input_ids'][0].tolist())

        break

def conllpp_test():

    path_data = pathlib.Path(
        "./data/WNUT17/CLNER_datasets/conll++")

    collector_log_dir = pathlib.Path("/tmp")


    loader = CNLLPPDataModule(
        collector_log_dir=collector_log_dir,
        data_dir=path_data,
        batch_size=1
    )

    # preparing data
    loader.prepare_data()

    # seting up
    loader.setup()

    # get generator
    test_generator = loader._get_dataloader("val")

    withou_context = 0

    for i, sample in tqdm(enumerate(test_generator)):

        if i > 3:
            break

        print(sample['labels'][0].tolist())

        pretty_print(loader.model_tokenizer, sample['labels'][0].tolist(), sample['input_ids'][0].tolist())
        pretty_print_mask(loader.model_tokenizer, sample['attention_mask'][0].tolist(), sample['input_ids'][0].tolist())
        pretty_print_mask(loader.model_tokenizer, sample['attention_mask_without_context'][0].tolist(),
                          sample['input_ids'][0].tolist())

        continue


def main(args: argparse.Namespace) -> None:

    conllpp_test()
    conll03_test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
