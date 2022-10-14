#!/bin/bash 
export CUDA_VISIBLE_DEVICES=1 &&
python run_summarization.py \
--mode train \
--data_path "../../../data/dsc_msmo/finished_files/chunked_files/chunked_text/train_*" \
--img_feature_path "../../../data/dsc_msmo/finished_files/chunked_files/chunk_features_fc7/features_*" \
--vocab_path "../../../data/dsc_msmo/finished_files/vocab" \
--sim_img_feature_path "../../../data/dsc_msmo/finished_files/chunked_files/chunk_sim_features_fc7/USE_TT/features_*"  \
--dissim_img_feature_path "../../../data/dsc_msmo/finished_files/chunked_files/chunk_dissim_features_fc7/features_*"  \
--log_root "log_fast/" \
--experiment DSC_MSMO_SIMPAD_TIS_fast \
--max_train_iterations 258000 
# --classifier_wt 0.5
# --max_enc_steps 256 \
# --max_dec_steps 64

# export CUDA_VISIBLE_DEVICES=1 &&
# python run_summarization.py \
# --mode train \
# --data_path "../../data/dsc_msmo/finished_files/chunked_files/chunked_text/train_*" \
# --img_feature_path "../../data/dsc_msmo/finished_files/chunked_files/chunk_features_fc7/features_*" \
# --vocab_path "../../data/dsc_msmo/finished_files/vocab" \
# --log_root "log/" \
# --experiment DSC_MSMO_SIMPAD_TIS \
# --max_train_iterations 400000 \