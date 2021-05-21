import random
import pandas as pd
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar

class TaskData(object):
    def __init__(self):
        pass
    def train_val_split(self,X, y,valid_size,stratify=False,shuffle=True,save = True,
                        seed = None,data_name = None,data_dir = None):
        pbar = ProgressBar(n_total=len(X),desc='bucket')
        logger.info('split raw data into train and valid')
        if stratify:
            num_classes = len(list(set(y)))
            train, valid = [], []
            bucket = [[] for _ in range(num_classes)]
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
                pbar(step=step)
            del X, y
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * valid_size)
                if shuffle:
                    random.seed(seed)
                    random.shuffle(bt)
                valid.extend(bt[:test_size])
                train.extend(bt[test_size:])
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        else:
            trans_data = []
            base_data = []
            data = []
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y))
                if data_y[5] or data_y[6]:
                    trans_data.append((data_x, data_y))
                else:
                    base_data.append((data_x, data_y))
                pbar(step=step)
            del X, y
            N = len(data)
            test_size = int(N * valid_size)
            if shuffle:
                random.seed(seed)
                random.shuffle(data)
                random.shuffle(trans_data)
                random.shuffle(base_data)
            valid = data[:test_size]
            train = data[test_size:]
            trans_valid = trans_data[:int(0.2 * len(trans_data))]
            trans_train = trans_data[int(0.2 * len(trans_data)):]
            base_valid = base_data[:int(0.2 * len(base_data))]
            base_train = base_data[int(0.2 * len(base_data)):]
            # 混洗train数据集
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
                random.shuffle(base_train)
                random.shuffle(trans_train)
        if save:
            train_path = data_dir / f"{data_name}.train.pkl"
            valid_path = data_dir / f"{data_name}.valid.pkl"
            save_pickle(data=train,file_path=train_path)
            save_pickle(data = valid,file_path=valid_path)
            save_pickle(data = trans_valid, file_path = data_dir / f"{data_name}.trans_valid.pkl")
            save_pickle(data = trans_train, file_path = data_dir / f"{data_name}.trans_train.pkl")
            save_pickle(data = base_valid, file_path = data_dir / f"{data_name}.base_valid.pkl")
            save_pickle(data = base_train, file_path = data_dir / f"{data_name}.base_train.pkl")
        return train, valid, trans_valid, trans_train, base_valid, base_train

    def read_data(self,raw_data_path,preprocessor = None,is_train=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        ids, targets, sentences = [], [], []
        data = pd.read_csv(raw_data_path)
        for row in data.values:
            if is_train:
                target = row[2:]
            else:
                target = [-1,-1,-1,-1,-1,-1,-1]
            sentence = str(row[1])
            number = str(row[0])
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                ids.append(number)
                targets.append(target)
                sentences.append(sentence)
        return ids,targets,sentences
