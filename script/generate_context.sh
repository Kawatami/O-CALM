#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=context_generation

#SBATCH --nodes=1

#SBATCH --time=5-00:00:00

#SBATCH --gpus-per-node=1

#SBATCH --mail-type=ALL

#SBATCH --output=/home/luiggit/project/contextual_ner/script/logs/baseline_large-%j.out

#SBATCH --error=/home/luiggit/project/contextual_ner/script/logs/baseline_large-%j.out

source ~/miniconda3/etc/profile.d/conda.sh

conda activate deep

for N_RUN in 1
do



echo ===== RUN $N_RUN ==============


python    ../generate_context_dataset.py  \
          --path ../data/WNUT17/CLNER_datasets/wnut17/\
          --destination ../test_annot \
          --use_cuda \
          --batch_size  2 \
          --prompts_path ../prompts_reformulation.json \
          --is_split_into_words


done