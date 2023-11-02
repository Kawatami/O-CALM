#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=baseline_large

#SBATCH --nodes=1

#SBATCH --time=5-00:00:00

#SBATCH --gpus-per-node=3

#SBATCH --mail-type=ALL

#SBATCH --output=/home/luiggit/project/contextual_ner/script/logs/baseline_large.out

#SBATCH --error=/home/luiggit/project/contextual_ner/script/logs/baseline_large.out

source ~/miniconda3/etc/profile.d/conda.sh

conda activate deep

for N_RUN in 1
do



echo ===== RUN $N_RUN ==============


python    ../run.py \
          --task SequenceTaggingTask \
          --dataset WNUT17DataModule \
          --data_dir ../data/WNUT17/CLNER_datasets/annot_NER/All/ \
          --model BaselineModel \
          --metrics class-SeqEvalWNUT17 \
          --loss MultiTaskLoss  \
          --losses CRFLoss CRFLoss KLDivergence \
          --loss_input_keys  CRF_loss_with_context CRF_loss_without_context CRF_posterior_without_context \
          --loss_target_keys CRF_loss_with_context CRF_loss_without_context CRF_posterior_with_context \
          --loss_weights 0.3 0.3 0.3 \
          --loss_names NLL_EXT NLL KL \
          --default_root_dir $INFERENCE_ROOT/logs/Contextual_NER/Llama_NER \
          --batch_size  1 \
          --gpus 1 \
          --collector_log_dir $INFERENCE_ROOT/collector/Contextual_NER/Llama_NER \
          --ignore_warnings \
          --lr 0.000005 \
          --no_early_stopping \
          --max_epochs 5 \
          --accumulate_grad_batches 4 \
          --gradient_clip_val 1 \
          --modelCheckpoint_monitor MT/valset

done