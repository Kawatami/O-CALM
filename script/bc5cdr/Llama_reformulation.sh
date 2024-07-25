#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=annot_NER

#SBATCH --nodes=1

#SBATCH --time=5-00:00:00

#SBATCH --gpus-per-node=3

#SBATCH --mail-type=ALL

#SBATCH --output=/home/luiggit/project/contextual_ner/script/logs/baseline_large.out

#SBATCH --error=/home/luiggit/project/contextual_ner/script/logs/baseline_large.out

source ~/miniconda3/etc/profile.d/conda.sh

conda activate deep

ANNOT_DIR=annot_reformulation_bc5cdr

for VARIATION in All Answer_format Classic Persona Reflexion_pattern
do
  for N_RUN in 1 2 3
  do

  echo ===== RUN $N_RUN $VARIATION ==============

  python    ../run.py \
            --task SequenceTaggingTask \
            --dataset Bc5cdrDataModule \
            --data_dir ../data/WNUT17/CLNER_datasets/$ANNOT_DIR/$VARIATION/ \
            --model BaselineModel \
            --num_label 5 \
            --metrics class-SeqEvalWNUT17 \
            --loss MultiTaskLoss  \
            --losses CRFLoss CRFLoss KLDivergence \
            --loss_input_keys  CRF_loss_with_context CRF_loss_without_context CRF_posterior_without_context \
            --loss_target_keys CRF_loss_with_context CRF_loss_without_context CRF_posterior_with_context \
            --loss_weights 0.25 0.25 0.5 \
            --loss_names NLL_EXT NLL KL \
            --default_root_dir $INFERENCE_ROOT/logs/Contextual_NER/$ANNOT_DIR/$VARIATION \
            --batch_size  2 \
            --gpus 3 \
            --collector_log_dir $INFERENCE_ROOT/collector/Contextual_NER/$ANNOT_DIR/$VARIATION/ \
            --ignore_warnings \
            --lr 0.000005 \
            --no_early_stopping \
            --max_epochs 10 \
            --accumulate_grad_batches 1 \
            --gradient_clip_val 1.0 \
            --modelCheckpoint_monitor CONTEXT_micro_avg_f1-score/valset \
            --modelCheckpoint_mode max \
            --dropout 0.1 \
            --version $N_RUN \
            --training_key dmis-lab/biobert-large-cased-v1.1 \
            --tokenizer_key dmis-lab/biobert-large-cased-v1.1 \
            --merge_train_dev \
            --use_proxy

  done
done