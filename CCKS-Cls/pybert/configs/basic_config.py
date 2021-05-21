
from pathlib import Path
BASE_DIR = Path('./pybert')
now_path = Path.cwd()
config = {
    'raw_data_path': now_path.absolute() / 'dataset/train_sample.csv',
    'test_path': now_path.absolute() / 'dataset/test.csv',

    'data_dir': BASE_DIR / 'dataset',
    'log_dir': BASE_DIR / 'output/log',
    'writer_dir': BASE_DIR / "output/TSboard",
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'cache_dir': BASE_DIR / 'model/',
    'result': BASE_DIR / "output/result",
    'test_output':BASE_DIR.absolute().parent/'test_output',

    'bert_vocab_path': now_path.absolute()/'pretrained_model/Bert-wwm-ext/vocab.txt',
    'bert_config_file': now_path.absolute()/'pretrained_model/Bert-wwm-ext/config.json',
    'bert_model_dir': now_path.absolute()/'pretrained_model/Bert-wwm-ext/'
}

import os
if not os.path.exists(config['log_dir']):
    os.mkdir(config['log_dir'])
if not os.path.exists(config['checkpoint_dir']):
    os.mkdir(config['checkpoint_dir'])
if not os.path.exists(config['test_output']):
    os.mkdir(config['test_output'])
