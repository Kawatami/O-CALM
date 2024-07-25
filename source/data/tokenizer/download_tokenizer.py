import pathlib
from tokenizer import path_tokenizer_ressources
import argparse
from transformers import BertTokenizerFast, LayoutLMv2TokenizerFast, RobertaTokenizerFast
import os
#os.environ['http_proxy'] = "http://192.168.0.100:3128"
#os.environ['https_proxy'] = "http://192.168.0.100:3128"

def main(args : argparse.Namespace) -> None :


    tokenizer = RobertaTokenizerFast.from_pretrained(args.key)

    # creating directory
    destination_path = path_tokenizer_ressources / args.name
    destination_path.mkdir(parents=True, exist_ok=True)
    print(f"creating directory : {destination_path}")
    tokenizer.save_pretrained(str(destination_path))

    print("done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("key", type = str)
    parser.add_argument("name", type = str)

    args = parser.parse_args()

    main(args)