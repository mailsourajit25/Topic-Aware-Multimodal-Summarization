#!/bin/bash 
export CUDA_VISIBLE_DEVICES=0 &&
python run_summarization.py \
--mode train \
--data_path "../../../data/dsc_msmo/finished_files/chunked_files/chunked_text/train_*" \
--img_feature_path "../../../data/dsc_msmo/finished_files/chunked_files/chunk_features_fc7/features_*" \
--sim_img_feature_path "../../../data/dsc_msmo/finished_files/chunked_files/chunk_sim_features_fc7/USE_TT/features_*"  \
--dissim_img_feature_path "../../../data/dsc_msmo/finished_files/chunked_files/chunk_dissim_features_fc7/features_*"  \
--vocab_path "../../../data/dsc_msmo/finished_files/vocab" \
--log_root "log_fast/" \
--experiment DSC_MSMO_SIMPAD_TIS_fast  \
--lr 0.01  \
--coverage  \
--max_train_iterations 300000