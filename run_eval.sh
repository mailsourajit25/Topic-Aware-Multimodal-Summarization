#!/bin/bash 
# export CUDA_VISIBLE_DEVICES=1 &&
# python run_summarization.py \
# --mode eval \
# --data_path "../../data/dsc_msmo/finished_files/chunked_files/chunked_text/val_*" \
# --img_feature_path "../../data/dsc_msmo/finished_files/chunked_files/chunk_features_fc7/valid_features_*" \
# --vocab_path "../../data/dsc_msmo/finished_files/vocab" \
# --log_root "log/" \
# --experiment DSC_MSMO_SIMPAD_TIS  \


export CUDA_VISIBLE_DEVICES=0 &&
python run_summarization.py \
--mode eval \
--data_path "../../../data/dsc_msmo/finished_files/chunked_files/chunked_text/val_*" \
--img_feature_path "../../../data/dsc_msmo/finished_files/chunked_files/chunk_features_fc7/valid_features_*" \
--vocab_path "../../../data/dsc_msmo/finished_files/vocab" \
--log_root "log_fast/" \
--experiment DSC_MSMO_SIMPAD_TIS_fast  \
--coverage
# --max_enc_steps 200 \
# --max_dec_steps 50