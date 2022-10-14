#!/bin/bash 
# export CUDA_VISIBLE_DEVICES=1 &&
# python run_summarization.py \
# --mode test \
# --data_path ../data/msmo/finished_files/chunked/test_* \
# --vocab_path ../data/msmo/finished_files/vocab \
# --log_root log/test/ \
# --experiment DSC_MSMO_TIS  \
# --img_feature_path ../data/msmo/finished_files/chunk_fc7/test_features_* \
# --single_pass \
# --test_csv_path dsc_msmo/datasets/test_details.csv \
# --test_save_dir test_save_dir/ \
# --test_img_path test_data/img \
# --max_dec_steps 120
# --num_to_test 9851

# export CUDA_VISIBLE_DEVICES=0 &&
# python run_summarization.py \
# --mode test \
# --data_path "../../data/dsc_msmo/finished_files/chunked_files/chunked_text/test_*" \
# --vocab_path "../../data/dsc_msmo/finished_files/vocab" \
# --img_feature_path "../../data/dsc_msmo/finished_files/chunked_files/chunk_features_fc7/test_features_*" \
# --log_root "log/" \
# --experiment DSC_MSMO_TIS  \
# --single_pass \
# --test_csv_path "../old_dsc_msmo/datasets/test_details.csv" \
# --test_save_dir test_save_dir/ \
# --test_img_path test_data/img \
# --max_dec_steps 120
# --test_save_steps 1000 \
# --num_to_test 5000


export CUDA_VISIBLE_DEVICES=0 &&
python run_summarization.py \
--mode test \
--data_path "../../../data/dsc_msmo/finished_files/chunked_files/chunked_text/test_*" \
--vocab_path "../../../data/dsc_msmo/finished_files/vocab" \
--img_feature_path "../../../data/dsc_msmo/finished_files/chunked_files/chunk_features_fc7/test_features_*" \
--log_root "log_fast/" \
--experiment "DSC_MSMO_SIMPAD_TIS_fast"  \
--single_pass \
--test_csv_path "../../server_dump/final.csv" \
--test_save_dir "test_save_dir2/" \
--test_img_path "test_data/img" \
--max_dec_steps 120 \
--test_save_steps 1 \
--num_to_test 9851 \
--coverage 
# --test_csv_path "../../old_dsc_msmo/datasets/test_details.csv" \
