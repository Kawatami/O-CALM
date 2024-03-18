# O-CALM


This repo contains the code for reproducing experiments conducted in the paper [Augmented NER via LLm prompting]().

### Repo organization

```
.
├── data : Contains the data for all the experiments
├── source : Contains the source code 
├── script : Contains the bash script lauching each experiments
├── README.md : This file
├── bert_score_comparison.py : Script used to compare reference context with a generated one in term of bert score
├── environment.yml : environement file for conda
├── generate_context_dataset.py : Generate the context based on a LLM and input data
├── prompts_NER.json : file holding the prompt for the NER category
├── prompts_context_variation.json : file holding the prompt for the context variation category
├── prompts_reformulation.json : file holding the prompt for the context variation category
├── run.py : Main script lauching the trainings
└── stats_pormpt.py : Script to process the statistics on generated context
```

### Setting Environment

All the necessary module are listed in ```environment.yml```. One can create an environement from this file 
with this command 
```bash
conda env create --name envname --file=environments.yml
```

Note that ```INFERENCE_ROOT``` is env variable used to indicate where to store the tensorboard logs. You can set it up via
```bash
INFERENCE_ROOT=path/to/logs
```

Note that ```MODEL_FILES``` is env variable used to indicate where per-trained transformer files a stored in case you
want to load them locally
```bash
MODEL_FILES=path/to/model/files
```

### Training

The training of a given experiment can be done within the ```script``` folder. ***DO NOT ENTER SUBFOLDER
BEFORE TRAINING***

here an exemple of launching training for the [baseline](https://arxiv.org/abs/2105.03654) model for WNUT17 dataset

```bash
# from root folder
cd script
sh baseline/baaseline_WNUT17.sh
```

### Sources organization
```
.
├── __init__.py
├── bin : Contains code for arguments parsing and high-level training procedure
├── callbacks : Contains code for callbacks object (ModelCheckPoint, LearningMonitor...)
├── data : Contains code for data management such as loading, preprocessing and DataGenerator creation
├── losses : Code for the losses 
├── metrics : Code for the metrics 
├── models : Code for the models and sub modules used such as CRF layer
├── task : Code for the low-level training logic
└── utils : code for helper functions

```
