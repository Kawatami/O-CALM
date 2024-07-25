#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=baseline_bc5cdr

#SBATCH --nodes=1

#SBATCH --time=5-00:00:00

#SBATCH --gpus-per-node=3

#SBATCH --mail-type=ALL

#SBATCH --output=/home/luiggit/project/contextual_ner/script/logs/baseline_large_bc5cdr.out

#SBATCH --error=/home/luiggit/project/contextual_ner/script/logs/baseline_large_bc5cdr.out

source ~/miniconda3/etc/profile.d/conda.sh

conda activate deep

for N_RUN in 1 2 3
do



echo ===== RUN $N_RUN ==============


python    ../run.py \
          --task SequenceTaggingTask \
          --dataset Bc5cdrDataModule \
          --data_dir ../data/WNUT17/CLNER_datasets/bc5cdr_bertscore_eos_doc_full \
          --model BaselineModel \
          --num_label 5 \
          --metrics class-SeqEvalBC5CDR \
          --loss MultiTaskLoss  \
          --losses CRFLoss CRFLoss KLDivergence \
          --loss_input_keys  CRF_loss_with_context CRF_loss_without_context CRF_posterior_without_context \
          --loss_target_keys CRF_loss_with_context CRF_loss_without_context CRF_posterior_with_context \
          --loss_weights 0.25 0.25 0.5 \
          --loss_names NLL_EXT NLL KL \
          --default_root_dir $INFERENCE_ROOT/logs/Contextual_NER/baseline_large_bc5cdr_merge \
          --batch_size  2 \
          --gpus 3 \
          --collector_log_dir $INFERENCE_ROOT/collector/Contextual_NER/baseline_large_bc5cdr_merge \
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