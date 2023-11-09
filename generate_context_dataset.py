import argparse, pathlib
import numpy as np
from source.data.generation.ContextGenerator import ContextGenerator
import re
from source.data.generation.PromptGenerator import PromptGenerator
import json
import os
from string import punctuation
from source.utils.misc import  load_from_file

proxy = 'http://192.168.0.100:3128'

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy


def format_sequence(sequences) :

    # spacing ponctuation
    for punctuation_elt in punctuation :
        sequences = [context.replace(punctuation_elt, " " + punctuation_elt + " ") for context in sequences]

    pre_tokenized_sequences = [[elt for elt in context.split(" ") if elt != '' and elt != " " ] for context in sequences]

    output = [["<EOS>"] + context for context in pre_tokenized_sequences]
    labels_context = [["B_X"] * len(context) for context in output]

    return output, labels_context


def main(args: argparse.Namespace) -> None:


    # init generator
    generator = ContextGenerator(
        LLM_key=args.LLM_key,
        use_fast_tokenizer=args.use_fast_tokenizer,
        batch_size=args.batch_size,
        generation_max_length=args.generation_max_length,
        is_split_into_words=args.is_split_into_words,
        use_cuda=args.use_cuda,
        prompt_generator=None
    )

    # listing file in directory
    files = [x for x in args.path.iterdir() if x.is_file()]

    with args.prompts_path.open("r") as file :
        prompts = json.load(file)

    for key, prompt in prompts.items() :

        print(f"## PROCESSING FOR PROMPTS\n{prompt}")

        # init prompt generator
        prompt_generator = PromptGenerator(
            prompt=prompt,
            system_tags=("", "Context :")
        )

        # update promt_generator
        generator.prompt_generator = prompt_generator


        # creating detination folder
        destination = args.destination / key
        if not destination.exists() :
            destination.mkdir(parents=True, exist_ok=True)


        for file in files :

            # checking if destination already exist
            destination_file = destination / f"{file.stem}_with_context.txt"
            destination_file_json = destination / f"{file.stem}_with_context.json"

            if destination_file.exists() and args.skip_already_processed :
                print(f"##{destination_file} already found skipping...")
                continue

            # laoding data
            print(f"Loading : {file}")
            samples = load_from_file(file)

            texts = [ sample['text'] for sample in samples ]
            labels = [ sample['ner_tags'] for sample in samples ]

            # calling generator
            output, output_post_processed = generator.generate(texts)

            # pot processing generated sequence
            output, labels_context = format_sequence(output)

            final_output_text = [original_text + context for original_text, context in zip(texts, output)]
            final_output_label = [original_labels + context for original_labels, context in zip(labels, labels_context)]

            # processing string file
            final_results = [[f"{token}\t{label.replace('_', '-')}" for token, label in zip(tokens_seq, labels_seq)]
                             for tokens_seq, labels_seq in zip(final_output_text, final_output_label)]

            final_results = ["\n".join(x) for x in final_results]

            final_results = "\n\n".join(final_results)



            # storing data for training
            with destination_file.open("w+") as dest_file :
                dest_file.write(final_results)

            # storing data in json for statistics
            json_content = {"prompt" : prompt, "data" : output_post_processed}
            with destination_file_json.open("w+") as dest_file :
                json.dump(json_content, dest_file, indent=4)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = ContextGenerator.add_specific_args(parent_parser=parser)
    parser.add_argument("--path", type=pathlib.Path)
    parser.add_argument("--prompts_path", type=pathlib.Path)
    parser.add_argument("--destination", type=pathlib.Path)
    parser.add_argument("--skip_already_processed", action='store_true', default=False)

    args = parser.parse_args()

    main(args)
