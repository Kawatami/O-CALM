import argparse, pathlib
import numpy as np
from evaluate import load
import json
from source.utils.misc import load_from_file


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

def main(args: argparse.Namespace) -> None:

    # loading reference data
    ref_data = load_from_file(args.path_reference)
    inputs_ref = [ remove_context(sample)  for sample in ref_data ]
    context_ref = [ remove_inputs(sample) for sample in ref_data ]

    # convert to string
    inputs_ref = [ " ".join(sample)  for sample in inputs_ref ]
    context_ref = [ " ".join(sample[1:]) for sample in context_ref ]
    data = [{ "text" : text, "generated_context" : context }  for text, context in zip(inputs_ref, context_ref) ]


    # creating result object
    res = {
        "prompt" : "",
        "data" : data
    }

    destination = args.path_reference.parent / (args.path_reference.stem + ".json")

    with destination.open("w+") as file :
        json.dump(res, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_reference", type=pathlib.Path)
    args = parser.parse_args()

    main(args)
