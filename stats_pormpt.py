import argparse, pathlib

import matplotlib.pyplot as plt
import numpy as np
import re
import json
import matplotlib
matplotlib.use('Webagg')
from collections import Counter
from scipy.stats import gaussian_kde

def is_error(text, limit = 20, comparator = lambda x, y : x < y) :

    if text == "" :
        return False

    if "I apologize" in text :
        return False

    vocabulary = set(text.split(" "))

    return  comparator(len(vocabulary), limit)
def plot_histogram(points, n_bins = 100) :
    counts, bins = np.histogram(points, bins=n_bins)

    plt.figure(figsize=(10, 10))
    plt.hist(points, bins = bins)
    plt.title("bert score distribution")
    plt.xlabel("bert score")
    plt.ylabel("proportion")
    plt.show()

def process_quartile(point, limit) :

    point = [p for p in point if p > 0]

    quantile = np.quantile(point, limit)

    return quantile


def load_from_file(path) :

    # checking path
    if not path.exists() :
        raise FileNotFoundError(f"File {path} not found.")

    with path.open("r", encoding='utf-8') as file:
        data_str = file.read()


    # splitting over samples
    data = data_str.split('\n\n')

    # splitting over tokens
    sample = [
        [re.split(r"\t| ", token) for token in point.split("\n")]
        for point in data
    ]

    samples = [
        {"text" : [elt[0] for elt in sequence], "ner_tags" : [elt[1].replace("-", "_") for elt in sequence]}
        for sequence in sample
    ]

    return samples

def plot_bar_cat(tags, title) :

    # removing B_X
    tags = [t for t in tags if t != "B_X"]

    # counting elements
    elts_counter = Counter(tags)
    elts_counter.pop("O")

    tags = list(elts_counter.keys())
    count_tags = list(elts_counter.values())
    count_tags = [elt / sum(list(elts_counter.values())) for elt in count_tags]


    plt.title(title)
    plt.xlabel("Categories")
    plt.ylabel("proportion")
    plt.ylim(0, 1.0)
    plt.bar(tags, count_tags)

def plot_len_cat(points) :

    max_len = max(sum(list(points.values()), []))


    for key, value in points.items() :
        value = [v for v in value if v != "B_XS"]
        points[key] = gaussian_kde(value)
        points[key].covariance_factor = lambda: .25
        points[key]._compute_covariance()

    xs = np.linspace(0, max_len, 200)

    # Set the figure size
    plt.figure(figsize=(14, 8))

    # Make the chart
    # We're actually building a line chart where x values are set all along the axis and y value are
    # the corresponding values from the density function
    colors = ["r", "b", "g", "y"]
    for (key, value ), c in zip(points.items(), colors) :
        plt.plot(xs, points[key](xs), label  = key, color = c)

    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

    plt.show()


def main(args: argparse.Namespace) -> None :

    # loading data
    with args.path.open("r") as file :
        data = json.load(file)

    samples = data['data']

    # zeros
    zeros_text  = [(sample['generated_context'], i) for i, sample in enumerate(samples) if sample['generated_context'] == ""]

    # error
    errors = [(sample['generated_context'], i) for i, sample in enumerate(samples) if is_error(text=sample['generated_context']) ]

    # remaining
    good = [(sample['generated_context'], i) for i, sample in enumerate(samples) if is_error(text=sample['generated_context'], comparator=lambda x, y : x >= y) ]

    # denied answer
    denied = [(sample['generated_context'], i) for i, sample in enumerate(samples) if "I apologize" in sample['generated_context'] ]

    res = {
        "empty" : zeros_text,
        "errors" : errors,
        "denied" : denied,
        "good" : good ,

    }

    total_samples = len(samples)
    print(f"-> total samples : {total_samples}")

    for key, value in res.items() :
        print(f"-> {key} : {len(value)} ({len(value) * 100 / total_samples:.2f}%)")

    with pathlib.Path("./results.json").open("w+") as file :
        json.dump(res, file, indent=4)


    ner_path = args.path.parent / (str(args.path.stem) + ".txt")
    data_ner_orig = load_from_file(ner_path)
    data_ner = [sample['ner_tags'] for sample in data_ner_orig]


    # # zeros tags
    # zeros_tags = sum([data_ner[i] for _, i in zeros_text], [])
    #
    # # error tags
    # error_tags = sum([data_ner[i] for _, i in errors], [])
    #
    # # remaining tags
    # good_tags = sum([data_ner[i] for _, i in good], [])
    #
    # plt.subplot(2, 2, 1)
    # plot_bar_cat(zeros_tags, "zeros tags")
    # plt.subplot(2, 2, 2)
    # plot_bar_cat(error_tags, "error tags")
    # plt.subplot(2, 2, 3)
    # plot_bar_cat(good_tags, "good tags")
    # plt.subplot(2, 2, 4)
    # plot_bar_cat(zeros_tags + error_tags + good_tags, "all samples")
    #
    # plt.show()

    # zeros tags
    zeros_tags = [len([elt for elt in data_ner[i] if elt != "B_X" ])  for _, i in zeros_text]

    # error tags
    error_tags = [len([elt for elt in data_ner[i] if elt != "B_X" ])  for _, i in errors]

    # error tags
    denied_tags = [len([elt for elt in data_ner[i] if elt != "B_X"]) for _, i in denied]

    # remaining tags
    good_tags = [len([elt for elt in data_ner[i] if elt != "B_X" ])  for _, i in good]



    truc = {
        "zeros" : zeros_tags,
        "error" : error_tags,
        "denied" : denied_tags,
        "good" : good_tags,
    }


    #plot_len_cat(truc)








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = pathlib.Path)
    args = parser.parse_args()

    main(args)
