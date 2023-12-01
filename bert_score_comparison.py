import argparse, pathlib
import numpy as np
from evaluate import load
import json
from source.utils.misc import load_from_file
from tqdm import tqdm

import os

proxy = 'http://192.168.0.100:3128'

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

def remove_context(list) :
    if '<EOS>' in list['text'] :
        return list['text'][:list['text'].index('<EOS>')]
    else :
        return list['text']

def remove_inputs(list) :

    if '<EOS>' in list['text'] :
        return list['text'][list['text'].index('<EOS>'):]
    else :
        return ['']



# loading metric
bertscore = load("bertscore")


def process_avg_bert_score(input, context) :

    # processing bertsore
    print("## PROCESSING BERTSCORE")
    results = bertscore.compute(
        references=input,
        predictions=context,
        lang="en"
    )

    return np.mean(results['f1'])


def main(args: argparse.Namespace) -> None:


    generation_directories = [
        pathlib.Path("data/WNUT17/CLNER_datasets/annot_context_variation_CNLLPP"),
        pathlib.Path("data/WNUT17/CLNER_datasets/annot_NER_CNLLPP"),
        pathlib.Path("data/WNUT17/CLNER_datasets/annot_reformulation_CNLLPP"),
    ]

    # Processing bert score for ref
    ref_data = load_from_file(args.path_reference)
    inputs_ref = [remove_context(sample) for sample in ref_data]
    context_ref = [remove_inputs(sample) for sample in ref_data]
    inputs_ref = [" ".join(sample) for sample in inputs_ref]
    context_ref = [" ".join(sample) for sample in context_ref]
    mean_f1_ref = process_avg_bert_score(inputs_ref, context_ref)

    scores = {
        "ref_score" : mean_f1_ref
    }

    for path in tqdm(generation_directories) :

        prompt_name = path.stem

        # listing all prompts
        path_prompts = [ x for x in path.iterdir() if x.is_dir() ]

        current_scores = {}

        for path_prompt in path_prompts :

            variation_name = path_prompt.stem

            # convert to string
            path_data = path_prompt / "conllpp_train_with_context.txt"
            generated_data = load_from_file(path_data)
            context_generated = [remove_inputs(sample) for sample in generated_data]
            context_generated = [" ".join(sample) for sample in context_generated]

            # processing bertscore
            berts_score = process_avg_bert_score(inputs_ref, context_generated)

            # storing score
            with (path_prompt / "bert_score.json").open("w+") as file :
                json.dump({"bertscore" : berts_score}, file)


            current_scores[variation_name] = berts_score

        scores[prompt_name] = current_scores

    path = pathlib.Path("./result_bert_score_context_conllpp.json")

    with path.open("w+") as file:
        json.dump(scores, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_reference", type=pathlib.Path)
    args = parser.parse_args()

    main(args)
