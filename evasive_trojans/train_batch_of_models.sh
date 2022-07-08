#!/bin/sh

# ================= Train Normal Trojans ================= #
python train_batch_of_models.py \
    --save_dir "./models/trojan" \
    --trojan_type "trojan" \
    --start_idx "0" \
    --num_train "200" \

# ================= Train Evasive Trojans Baseline ================= #
# first we need to train clean models to initialize from
python train_batch_of_models.py \
    --save_dir "./models/clean_init" \
    --trojan_type "clean" \
    --start_idx "0" \
    --num_train "200" \

python train_batch_of_models.py \
    --save_dir "./models/trojan_evasion" \
    --trojan_type "trojan_evasion" \
    --start_idx "0" \
    --num_train "200" \
