python3 Cls_data_preprocess.py
#python3 run_bert.py --do_data
python3 run_bert.py --do_train --save_best  --n_gpu '0'
python3 run_bert.py --do_test  --n_gpu '0'
