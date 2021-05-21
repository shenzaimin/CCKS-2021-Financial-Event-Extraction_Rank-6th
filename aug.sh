#!/bin/bash
# train for each event
python3 transfer_train_roberta_model_aug.py --dataset data --num_epochs 15 --model_folder pretrained --seed 2020 --device_num '0' --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 2 --learning_rate 5e-5 --gradient_accumulation_steps 4  --train_or_predict 1

python3 transfer_train_roberta_model_ensemble.py --dataset data --num_epochs 15 --model_folder saved_model_roberta_sg_1_1 --seed 2020 --device_num '0' --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 2 --learning_rate 5e-5 --gradient_accumulation_steps 4 --type "收购" --train_or_predict 1 --ensemble 10
python3 transfer_train_roberta_model_ensemble.py --dataset data --num_epochs 15 --model_folder saved_model_roberta_pj_1_1 --seed 2020 --device_num '0' --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 2 --learning_rate 5e-5 --gradient_accumulation_steps 4 --type "判决" --train_or_predict 1 --ensemble 10
python3 transfer_train_roberta_model_ensemble.py --dataset data --num_epochs 15 --model_folder saved_model_roberta_db_1_1 --seed 2020 --device_num '0' --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 2 --learning_rate 5e-5 --gradient_accumulation_steps 4 --type "担保" --train_or_predict 1 --ensemble 10
python3 transfer_train_roberta_model_ensemble.py --dataset data --num_epochs 15 --model_folder saved_model_roberta_zb_1_1 --seed 2020 --device_num '0' --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 2 --learning_rate 5e-5 --gradient_accumulation_steps 4 --type "中标" --train_or_predict 1 --ensemble 10
python3 transfer_train_roberta_model_ensemble.py --dataset data --num_epochs 15 --model_folder saved_model_roberta_qsht_1_1 --seed 2020 --device_num '0' --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 2 --learning_rate 5e-5 --gradient_accumulation_steps 4 --type "签署合同" --train_or_predict 1 --ensemble 10

# predict for test data
python3 transfer_train_roberta_model_ensemble.py --dataset data --num_epochs 100 --model_folder saved_model_roberta --seed 2020 --device_num '0' --bert_model_dir chinese_roberta_wwm_ext_pytorch --batch_size 2 --learning_rate 1e-4 --gradient_accumulation_steps 1 --train_or_predict 2

# fix trigger for events
python3 reader.py 