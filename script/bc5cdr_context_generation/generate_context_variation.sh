#!/bin/bash

#SBATCH --partition=hard

#SBATCH --nodelist=aerosmith

#SBATCH --job-name=context_generation

#SBATCH --nodes=1

#SBATCH --time=5-00:00:00

#SBATCH --gpus-per-node=1

#SBATCH --mail-type=ALL

#SBATCH --output=/home/luiggit/project/contextual_ner/script/logs/generation_b5cdr_variation.out

#SBATCH --error=/home/luiggit/project/contextual_ner/script/logs/generation_b5cdr_variation.out

source ~/miniconda3/etc/profile.d/conda.sh

conda activate deep

for N_RUN in 1
do



echo ===== RUN $N_RUN ==============


python    ../generate_context_dataset.py  \
          --path ../data/WNUT17/CLNER_datasets/bc5cdr/\
          --destination ../annot_variationER_bc5cdr \
          --use_cuda \
          --batch_size  2 \
          --prompts_path ../prompts_context_variation.json \
          --is_split_into_words \
          --skip_already_processed


done