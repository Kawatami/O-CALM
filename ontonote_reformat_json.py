import json

def convert_json_to_new_format(json_file, output_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        # Ouvrir le fichier de sortie
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # Lire le fichier ligne par ligne
            for line in file:
                # Analyser chaque ligne comme un objet JSON
                item = json.loads(line)
                tokens = item['tokens']
                tags = item['tags']

                # Associer chaque token avec son tag et écrire dans le fichier de sortie
                for token, tag in zip(tokens, tags):
                    outfile.write(f"{token}\t{tag}\n")

                # Ajouter un saut de ligne après chaque élément
                outfile.write('\n')
# Exemple d'utilisation
#convert_json_to_new_format('./data/WNUT17/CLNER_datasets/ontonote5/train.json', './data/WNUT17/CLNER_datasets/ontonote5/train.txt')
#convert_json_to_new_format('./data/WNUT17/CLNER_datasets/ontonote5/test.json', './data/WNUT17/CLNER_datasets/ontonote5/test.txt')
#convert_json_to_new_format('./data/WNUT17/CLNER_datasets/ontonote5/dev.json', './data/WNUT17/CLNER_datasets/ontonote5/dev.txt')


# Test ontonote import:
import argparse, pathlib
source_path = pathlib.Path(__file__).parents[1]
import sys
sys.path.append(str(source_path))
import numpy as np
from source.utils.misc import load_from_file_cnllpp
from source.data.data_modules.ontonoteDataModule import OntonoteDataModule
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


def main(args: argparse.Namespace) -> None:


    path_data = pathlib.Path(
        "./data/WNUT17/CLNER_datasets/ontonote5")

    collector_log_dir = pathlib.Path("/tmp")


    loader = OntonoteDataModule(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
